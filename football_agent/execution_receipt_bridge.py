"""Execution receipts and context rebuilds for autonomous acquisitions.

Round136 deliberately stops at a reservation proposal.  This module is the
next seam: a provider may accept that proposal, later complete it, or fail/
arrive late.  Only a completed receipt with cutoff-visible evidence can build a
new actor context.  Provider payloads are represented by digests and bounded
evidence rows; result labels and evaluator data never enter the actor view.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from .platform_acquisition_context import build_platform_acquisition_context

VERSION = "round137-execution-receipt-bridge-v1"
ACCEPTED = "accepted"
COMPLETED = "completed"
FAILED = "failed"
LATE = "late"
AMBIGUOUS = "ambiguous"
TERMINAL = frozenset({COMPLETED, FAILED, LATE, AMBIGUOUS})

_RESULT_KEYS = frozenset({
    "result", "outcome", "score", "scores", "goals", "home_score", "away_score",
    "home_goals", "away_goals", "final_score", "label", "target", "status",
    "evaluator", "sidecar", "realized_loss", "decision_loss", "actual_result",
})


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _acquisition_id(trace: Mapping[str, Any], candidate: str) -> str:
    """Return a stable action-level identity for one selected acquisition.

    A transport request is allowed to fan out into multiple provider requests;
    the acquisition identity must therefore be derived from the selected
    action trace rather than from any one request id.
    """
    explicit = trace.get("acquisition_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    action_kind = trace.get("selected_action_kind") or "query"
    return "acq-" + _digest({
        "trace_digest": trace.get("trace_digest"),
        "candidate_id": candidate,
        "action_kind": action_kind,
    })[:24]


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _costs(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): _number(amount, f"{name}.{key}") for key, amount in sorted(value.items())}


def _safe(value: Any, path: str = "payload") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}.key must be a string")
            if key.casefold() in _RESULT_KEYS:
                raise ValueError(f"{path} contains an evaluator field")
            result[key] = _safe(child, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_safe(item, f"{path}[]") for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a nonfinite number")
    if value is None or isinstance(value, (str, bool, int, float)):
        return deepcopy(value)
    raise ValueError(f"{path} contains an unsupported value")


def _evidence_rows(value: Any, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        raise ValueError("evidence_rows must be a nonempty sequence")
    rows = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"evidence_rows[{index}] must be a mapping")
        safe = _safe(row, f"evidence_rows[{index}]")
        if not isinstance(safe.get("source_id"), str) or not safe["source_id"].strip():
            raise ValueError(f"evidence_rows[{index}].source_id is required")
        rows.append(safe)
    return rows


def _canonical_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the dispatched request, normalizing only its equivalent UTC clock."""
    if not isinstance(request, Mapping):
        raise ValueError("execution request must be a mapping")
    copied = _safe(request, "execution_request")
    for field in ("request_id", "acquisition_id", "candidate_id", "action_id", "source_id",
                  "state_digest", "context_digest", "trace_digest", "action_kind"):
        _text(copied.get(field), f"execution_request.{field}")
    copied["prediction_cutoff"] = _iso(_time(copied.get("prediction_cutoff"), "prediction_cutoff"))
    sources = copied.get("source_ids")
    if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or not s.strip() for s in sources):
        raise ValueError("execution request source_ids must be nonempty")
    route = copied.get("route")
    if (copied["source_id"] not in sources or not isinstance(route, Mapping)
            or route.get("source_id") != copied["source_id"]
            or route.get("action_id") != copied["action_id"]):
        raise ValueError("execution request route identity mismatch")
    return copied


def _verify_request_binding(receipt: Mapping[str, Any]) -> None:
    status = receipt.get("request_binding_status", "legacy_unbound")
    if status == "legacy_unbound":
        if receipt.get("request_binding") is not None or receipt.get("request_digest") is not None:
            raise ValueError("unbound receipt has a request binding")
        return
    if status != "bound":
        raise ValueError("invalid receipt request binding status")
    bound = _canonical_request(receipt.get("request_binding"))
    if _digest(bound) != receipt.get("request_digest"):
        raise ValueError("receipt request digest mismatch")
    for field in ("request_id", "acquisition_id", "candidate_id", "action_id", "action_kind",
                  "source_id", "source_ids", "state_digest", "context_digest", "trace_digest",
                  "prediction_cutoff"):
        if bound[field] != receipt.get(field):
            raise ValueError(f"receipt request binding {field} mismatch")


def _verify_receipt(receipt: Mapping[str, Any]) -> None:
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != VERSION:
        raise ValueError("invalid execution receipt version")
    recorded = receipt.get("receipt_digest")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if not isinstance(recorded, str) or _digest(body) != recorded:
        raise ValueError("execution receipt digest mismatch")
    status = receipt.get("status")
    if status not in {ACCEPTED, *TERMINAL}:
        raise ValueError("execution receipt has unsupported status")
    if "evidence_rows" in receipt:
        rows = receipt.get("evidence_rows")
        if not isinstance(rows, list):
            raise ValueError("execution receipt evidence_rows must be a list")
        if receipt.get("evidence_digest") != _digest(rows):
            raise ValueError("execution receipt evidence digest mismatch")
    if "capture_status" in receipt and receipt["capture_status"] not in {"known", "unknown"}:
        raise ValueError("execution receipt has unsupported capture status")
    _verify_request_binding(receipt)


def accept_action(trace: Mapping[str, Any], *, provider_id: str,
                  accepted_at: str | datetime, receipt_id: str | None = None,
                  acquisition_id: str | None = None,
                  execution_request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Turn a selected reservation into a provider acceptance receipt."""
    if not isinstance(trace, Mapping) or trace.get("decision") != "acquire":
        raise ValueError("only an acquire trace can be accepted")
    reservation = trace.get("reservation")
    if not isinstance(reservation, Mapping) or reservation.get("actual_spend_observed") is not False:
        raise ValueError("trace reservation is missing or already spent")
    candidate = _text(reservation.get("candidate_id"), "candidate_id")
    if trace.get("selected_candidate_id") != candidate:
        raise ValueError("trace and reservation candidate mismatch")
    accepted = _time(accepted_at, "accepted_at")
    cutoff = _time(trace.get("prediction_cutoff"), "prediction_cutoff")
    if accepted > cutoff:
        raise ValueError("action accepted after prediction cutoff")
    provider = _text(provider_id, "provider_id")
    acquisition = _text(acquisition_id, "acquisition_id") if acquisition_id is not None else _acquisition_id(trace, candidate)
    body = {
        "schema_version": VERSION, "receipt_id": receipt_id or "receipt-" + _digest({
            "trace_digest": trace.get("trace_digest"), "candidate_id": candidate,
            "provider_id": provider, "accepted_at": _iso(accepted),
        }),
        "status": ACCEPTED, "provider_id": provider, "candidate_id": candidate,
        "acquisition_id": acquisition,
        "action_kind": trace.get("selected_action_kind"),
        "context_digest": _text(trace.get("context_digest"), "context_digest"),
        "state_digest": _text(trace.get("state_digest"), "state_digest"),
        "trace_digest": _text(trace.get("trace_digest"), "trace_digest"),
        "prediction_cutoff": _iso(cutoff), "accepted_at": _iso(accepted),
        "declared_utility_cost": deepcopy(reservation.get("utility_cost")),
        "declared_resource_cost": deepcopy(reservation.get("resource_cost")),
        "actual_resource_cost": None, "result_digest": None, "evidence_digest": None,
        "replanning_required": False, "execution_receipt_id": None,
    }
    body["request_binding_status"] = "legacy_unbound"
    if execution_request is not None:
        bound = _canonical_request(execution_request)
        for field in ("candidate_id", "acquisition_id", "action_kind", "context_digest",
                      "state_digest", "trace_digest", "prediction_cutoff"):
            if bound[field] != body[field]:
                raise ValueError(f"execution request {field} differs from selection")
        body.update(request_binding_status="bound", request_binding=bound, request_digest=_digest(bound))
        for field in ("request_id", "action_id", "source_id", "source_ids"):
            body[field] = deepcopy(bound[field])
    body["execution_receipt_id"] = body["receipt_id"]
    body["receipt_digest"] = _digest(body)
    return body


def complete_action(receipt: Mapping[str, Any], *, result_payload: Mapping[str, Any],
                    evidence_rows: Iterable[Mapping[str, Any]],
                    actual_resource_cost: Mapping[str, Any],
                    observed_at: str | datetime, operational_available_at: str | datetime,
                    captured_at: str | datetime | None = None,
                    status: str | None = None, error_code: str | None = None) -> dict[str, Any]:
    """Close an accepted action without silently treating it as a context update."""
    _verify_receipt(receipt)
    if receipt.get("status") != ACCEPTED:
        raise ValueError("only an accepted receipt can be completed")
    if not isinstance(result_payload, Mapping):
        raise ValueError("result_payload must be a mapping")
    # The payload is hashed, while explicit result-like fields are rejected so
    # the receipt cannot become a hidden evaluator side channel.
    safe_result = _safe(result_payload, "result_payload")
    actual_cost = _costs(actual_resource_cost, "actual_resource_cost")
    observed = _time(observed_at, "observed_at")
    available = _time(operational_available_at, "operational_available_at")
    accepted = _time(receipt["accepted_at"], "accepted_at")
    # A provider must report when the response was actually captured.  Never
    # manufacture a capture timestamp from an observation or availability
    # clock: doing so can turn an unresolved response into cutoff-visible
    # evidence. Explicit FAILED/LATE terminal states remain auditable without
    # a capture clock, but an implicit/successful completion becomes AMBIGUOUS.
    captured = _time(captured_at, "captured_at") if captured_at is not None else None
    cutoff = _time(receipt["prediction_cutoff"], "prediction_cutoff")
    if observed > available:
        raise ValueError("execution clocks are not ordered")
    if captured is not None and (observed > captured or available > captured or accepted > captured):
        raise ValueError("execution clocks are not ordered")
    inferred_late = observed > cutoff or available > cutoff or (captured is not None and captured > cutoff)
    final = status or (FAILED if error_code else LATE if inferred_late else AMBIGUOUS if captured is None else COMPLETED)
    if final not in TERMINAL:
        raise ValueError("unsupported terminal execution status")
    if final == COMPLETED and error_code:
        final = FAILED
    late_reason = None
    if observed > cutoff:
        late_reason = "source_observed_after_cutoff"
    elif available > cutoff:
        late_reason = "operational_available_after_cutoff"
    elif captured is not None and captured > cutoff:
        late_reason = "captured_after_cutoff"
    if final == COMPLETED:
        if late_reason is not None:
            final = LATE
        elif captured is None:
            final = AMBIGUOUS
    # A timeout or other unsuccessful terminal result can legitimately have
    # no evidence. Preserve its actual spend without inventing an observation.
    # Successful completion still requires explicit usable evidence rows.
    rows = _evidence_rows(evidence_rows, allow_empty=final != COMPLETED)
    if receipt.get("request_binding_status") == "bound" and any(
            row["source_id"] not in receipt["source_ids"] for row in rows):
        raise ValueError("evidence source differs from accepted request binding")
    body = deepcopy(dict(receipt))
    body.update({
        "status": final, "observed_at": _iso(observed),
        "operational_available_at": _iso(available),
        "captured_at": _iso(captured) if captured is not None else None,
        "capture_status": "known" if captured is not None else "unknown",
        "actual_resource_cost": actual_cost,
        "result_digest": _digest(safe_result), "evidence_digest": _digest(rows),
        "evidence_rows": rows, "error_code": error_code,
        "clock_status": ("late_after_cutoff" if late_reason else
                         "capture_unknown" if captured is None else "cutoff_visible"),
        "late_reason": late_reason,
        "replanning_required": final != COMPLETED,
    })
    body.pop("receipt_digest", None)
    body["receipt_digest"] = _digest(body)
    return body


def rebuild_context_after_receipt(actor_context: Mapping[str, Any], receipt: Mapping[str, Any],
                                  match: Mapping[str, Any],
                                  existing_evidence_rows: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Rebuild a new public context only after cutoff-visible completion."""
    _verify_receipt(receipt)
    if receipt.get("context_digest") != actor_context.get("context_digest"):
        raise ValueError("receipt context digest does not match actor context")
    if receipt.get("status") not in TERMINAL:
        raise ValueError("only a terminal receipt can be applied")
    if receipt.get("status") != COMPLETED:
        return {"actor_context": deepcopy(dict(actor_context)), "receipt": deepcopy(dict(receipt)),
                "context_changed": False, "replanning_required": True}
    # A completed receipt must carry an explicit, cutoff-visible capture
    # event. This is intentionally checked again at the context boundary so a
    # forged/replayed receipt cannot bypass the bridge's classification.
    cutoff = _time(receipt.get("prediction_cutoff"), "prediction_cutoff")
    observed = _time(receipt.get("observed_at"), "observed_at")
    available = _time(receipt.get("operational_available_at"), "operational_available_at")
    captured_value = receipt.get("captured_at")
    if captured_value is None or receipt.get("capture_status") != "known":
        raise ValueError("completed receipt requires an explicit captured_at")
    captured = _time(captured_value, "captured_at")
    if observed > available or available > captured or captured > cutoff:
        raise ValueError("completed receipt is outside prediction cutoff")
    if receipt.get("clock_status") != "cutoff_visible" or receipt.get("late_reason") is not None:
        raise ValueError("completed receipt has inconsistent cutoff status")
    rows = list(existing_evidence_rows or []) + list(receipt.get("evidence_rows") or [])
    rebuilt = build_platform_acquisition_context(
        match, rows, prediction_cutoff=receipt["prediction_cutoff"],
        information_mode=actor_context.get("information_mode"),
    )
    if rebuilt["actor_context"]["context_digest"] == actor_context.get("context_digest"):
        raise ValueError("completed evidence did not change public context")
    return {"actor_context": rebuilt["actor_context"], "audit": rebuilt["audit"],
            "evaluator_sidecar": rebuilt["evaluator_sidecar"],
            "receipt": deepcopy(dict(receipt)), "context_changed": True,
            "replanning_required": False,
            "previous_context_digest": actor_context["context_digest"],
            "new_context_digest": rebuilt["actor_context"]["context_digest"]}


__all__ = [
    "VERSION", "ACCEPTED", "COMPLETED", "FAILED", "LATE", "AMBIGUOUS",
    "accept_action", "complete_action", "rebuild_context_after_receipt",
]
