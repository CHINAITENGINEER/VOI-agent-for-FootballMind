"""Normalize provider results into auditable evidence and completion receipts."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping

from .candidate_action_registry import VERSION as REGISTRY_VERSION
from .execution_receipt_bridge import complete_action, _canonical_request, _digest, _verify_receipt

VERSION = "execution-action-adapter-v1"

_FORBIDDEN = frozenset({
    "result", "outcome", "score", "scores", "goals", "home_score",
    "away_score", "home_goals", "away_goals", "final_score", "label",
    "evaluator", "sidecar", "realized_loss", "decision_loss",
    "actual_result", "future_result", "sequence_losses", "action_returns",
})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _same_clock(left: Any, right: Any) -> bool:
    """Compare equivalent UTC timestamps despite Z/+00:00 serialization."""
    try:
        lval = datetime.fromisoformat(str(left).replace("Z", "+00:00"))
        rval = datetime.fromisoformat(str(right).replace("Z", "+00:00"))
        if lval.tzinfo is None or rval.tzinfo is None:
            return left == right
        return lval.astimezone(timezone.utc) == rval.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return left == right


def _safe(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            name = _text(str(key), f"{path}.key")
            if name.casefold() in _FORBIDDEN:
                raise ValueError(f"{path} contains an evaluator field")
            result[name] = _safe(child, f"{path}.{name}")
        return result
    if isinstance(value, (list, tuple)):
        return [_safe(child, f"{path}[]") for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return deepcopy(value)
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} contains an unsupported value")


def _verify_request(request: Mapping[str, Any]) -> None:
    if not isinstance(request, Mapping) or request.get("schema_version") != REGISTRY_VERSION:
        raise ValueError("invalid execution request version")
    for field in ("request_id", "candidate_id", "action_id", "source_id"):
        _text(request.get(field), f"request.{field}")
    source_ids = request.get("source_ids")
    if not isinstance(source_ids, (list, tuple)) or not source_ids:
        raise ValueError("request.source_ids must be nonempty")
    if request["source_id"] not in source_ids:
        raise ValueError("request source is absent from request sources")
    if "acquisition_id" in request:
        _text(request["acquisition_id"], "request.acquisition_id")


def _request_match_id(request: Mapping[str, Any]) -> str | None:
    parameters = request.get("parameters")
    if parameters is None:
        return None
    parameters = _mapping(parameters, "request.parameters")
    target = parameters.get("match_id")
    return None if target is None else _text(target, "request.parameters.match_id")


def _verify_match_identity(
    value: Mapping[str, Any], target: str | None, path: str, *, envelopes: bool,
) -> None:
    """Check explicit canonical identity only in the current evidence envelope.

    Provider fixture IDs can use a separate namespace. Historical feature rows
    can reference other matches; neither is a declaration of this response's
    canonical target, so neither is traversed or compared here.
    """
    if target is None:
        return
    for field in ("match_id", "canonical_match_id", "target_key"):
        if field in value:
            identity = _text(value[field], f"{path}.{field}")
            if identity != target:
                raise ValueError("response match identity differs from request")
    if envelopes:
        for field in ("values", "payload", "context"):
            child = value.get(field)
            if isinstance(child, Mapping):
                _verify_match_identity(child, target, f"{path}.{field}", envelopes=True)


def _normalize_provider_observation(
    request: Mapping[str, Any],
    *,
    result_payload: Mapping[str, Any],
    evidence_rows: Iterable[Mapping[str, Any]],
    allow_empty: bool,
) -> dict[str, Any]:
    """Validate a provider response without making it an evaluator channel."""
    _verify_request(request)
    target = _request_match_id(request)
    safe_result = _safe(result_payload, "result_payload")
    _verify_match_identity(_mapping(safe_result, "result_payload"), target,
                           "result_payload", envelopes=False)
    rows = []
    for index, raw in enumerate(evidence_rows):
        row = _mapping(raw, f"evidence_rows[{index}]")
        source_id = _text(row.get("source_id"), f"evidence_rows[{index}].source_id")
        if source_id not in request["source_ids"]:
            raise ValueError("evidence source is absent from request sources")
        safe = _safe(row, f"evidence_rows[{index}]")
        _verify_match_identity(safe, target, f"evidence_rows[{index}]", envelopes=True)
        safe["source_id"] = source_id
        rows.append(safe)
    if not rows and not allow_empty:
        raise ValueError("evidence_rows must be nonempty")
    return {
        "schema_version": VERSION, "request_id": request["request_id"],
        "candidate_id": request["candidate_id"], "action_id": request["action_id"],
        **({"acquisition_id": request["acquisition_id"]} if request.get("acquisition_id") else {}),
        "source_id": request["source_id"], "result_payload": safe_result,
        "evidence_rows": rows,
        "contract": {
            "provider_response_normalized": True, "evaluator_fields_used": False,
            "real_source_requests": 0, "real_model_calls": 0,
        },
    }


def normalize_provider_observation(
    request: Mapping[str, Any],
    *,
    result_payload: Mapping[str, Any],
    evidence_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize a successful public observation, requiring evidence."""
    return _normalize_provider_observation(
        request, result_payload=result_payload, evidence_rows=evidence_rows,
        allow_empty=False,
    )


def complete_execution_request(
    receipt: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    result_payload: Mapping[str, Any],
    evidence_rows: Iterable[Mapping[str, Any]],
    actual_resource_cost: Mapping[str, Any],
    observed_at: str | datetime,
    operational_available_at: str | datetime,
    captured_at: str | datetime | None = None,
    status: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    """Complete a receipt only when the provider route matches the request."""
    _verify_request(request)
    if receipt.get("request_binding_status") == "bound":
        _verify_receipt(receipt)
        if _digest(_canonical_request(request)) != receipt.get("request_digest"):
            raise ValueError("execution request differs from accepted request binding")
    if receipt.get("candidate_id") != request["candidate_id"]:
        raise ValueError("receipt candidate differs from request")
    if request.get("acquisition_id") is not None and receipt.get("acquisition_id") != request["acquisition_id"]:
        raise ValueError("receipt acquisition differs from request")
    # Bind every identity the request exposes when the receipt carries the
    # same field. Older accepted receipts did not persist action/source/state
    # fields, so those remain optional for backwards-compatible replay.
    for field in ("action_id", "source_id", "state_digest", "context_digest", "trace_digest", "prediction_cutoff"):
        expected = request.get(field)
        actual = receipt.get(field)
        equivalent = _same_clock(expected, actual) if field == "prediction_cutoff" else expected == actual
        if expected is not None and actual is not None and not equivalent:
            raise ValueError(f"receipt {field} differs from request")
    # Keep the normalizer strict for stand-alone observations. In the receipt
    # path, the bridge must classify and validate a non-success terminal state
    # before an empty evidence response can be returned to the caller.
    normalized = _normalize_provider_observation(
        request, result_payload=result_payload, evidence_rows=evidence_rows,
        allow_empty=True,
    )
    completed = complete_action(
        receipt, result_payload=normalized["result_payload"],
        evidence_rows=normalized["evidence_rows"],
        actual_resource_cost=actual_resource_cost, observed_at=observed_at,
        operational_available_at=operational_available_at, captured_at=captured_at,
        status=status, error_code=error_code,
    )
    return {
        "schema_version": VERSION, "receipt": completed,
        "normalized": normalized,
        "contract": {
            "receipt_completed_through_existing_bridge": True,
            "request_binding_verified": receipt.get("request_binding_status") == "bound",
            "evaluator_fields_used": False, "real_source_requests": 0,
            "real_model_calls": 0,
        },
    }


__all__ = ["VERSION", "normalize_provider_observation", "complete_execution_request"]
