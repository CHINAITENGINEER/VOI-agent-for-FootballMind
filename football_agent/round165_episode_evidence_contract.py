"""Round165 contract for complete action-level VoI episode evidence.

Round164 binds a historical result to an actor fixture.  This module adds the
next boundary: it checks whether a run actually recorded the chosen actions,
cutoff-safe evidence returns, posterior predictions, a terminal stop, and an
independent loss path.  It does not estimate value, call a model, or promote a
declared value estimate to observed utility.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


VERSION = "round165-episode-evidence-contract-v1"
SUPPORTED_LABELS = frozenset({"INDEPENDENT_GOLD", "INDEPENDENT_GOLD_RECONSTRUCTED"})
_FORBIDDEN = frozenset({
    "result", "outcome", "score", "scores", "goals", "home_score", "away_score",
    "home_goals", "away_goals", "final_score", "label", "loss", "decision_loss",
    "realized_gain", "realized_loss", "evaluator", "sidecar", "sequence_losses",
    "action_returns", "post_state", "after_state", "actual_result", "target_label",
})
_EPS = 1e-8
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _finite(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if minimum is not None and value < minimum - _EPS:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _probability_vector(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty probability vector")
    values = tuple(_finite(item, f"{name}[]", minimum=0) for item in value)
    if abs(sum(values) - 1.0) > _EPS:
        raise ValueError(f"{name} must sum to one")
    return values


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _forbidden(value: Any, path: str = "public") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if name.casefold() in _FORBIDDEN:
                found.append(f"{path}.{name}")
            found.extend(_forbidden(child, f"{path}.{name}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_forbidden(child, f"{path}[{index}]"))
    return found


def _public_projection(public: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(public, Mapping):
        raise ValueError("public episode must be a mapping")
    leaks = _forbidden(public)
    if leaks:
        raise ValueError("public episode contains evaluator fields: " + ",".join(leaks))
    return deepcopy(dict(public))


def _candidate_menu(public: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = public.get("candidate_menu")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("public candidate_menu must be nonempty")
    menu: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ValueError(f"candidate_menu[{index}] must be a mapping")
        candidate = _text(row.get("candidate_id"), f"candidate_menu[{index}].candidate_id")
        if candidate in menu:
            raise ValueError("candidate IDs must be unique")
        _finite(row.get("cost"), f"candidate_menu[{index}].cost", minimum=0)
        menu[candidate] = row
    return menu


def _aggregate_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(_text(row.get(key), f"aggregate.{key}") for key in ("episode_id", "target_key", "candidate_id", "request_id"))


def validate_round165_episode_evidence(
    public: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    *,
    evaluator_sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a conservative evidence report; unknown fields remain unknown."""
    reasons: list[str] = []
    try:
        actor = _public_projection(public)
        episode_id = _text(actor.get("episode_id"), "public.episode_id")
        target_key = _text(actor.get("target_key"), "public.target_key")
        cutoff = _time(actor.get("prediction_cutoff"), "public.prediction_cutoff")
        menu = _candidate_menu(actor)
        if actor.get("public_snapshot_digest") is not None and actor["public_snapshot_digest"] != _digest(actor.get("match_snapshot", {})):
            reasons.append("public_snapshot_digest_mismatch")
    except (TypeError, ValueError) as exc:
        return {"schema_version": VERSION, "status": "INVALID", "eligible_for_replay": False,
                "numeric_voi_training_eligible": False, "reasons": [str(exc)]}
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)) or not traces:
        reasons.append("traces_missing")
        traces = ()
    if not isinstance(aggregates, Sequence) or isinstance(aggregates, (str, bytes)):
        reasons.append("aggregates_missing")
        aggregates = ()
    ordered = list(traces)
    aggregate_map: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for aggregate in aggregates:
        try:
            key = _aggregate_key(aggregate)
            if key in aggregate_map:
                reasons.append("duplicate_aggregate")
            aggregate_map[key] = aggregate
        except (TypeError, ValueError) as exc:
            reasons.append(str(exc))
    action_rows = []
    previous_prediction: tuple[float, ...] | None = None
    stop_seen = False
    for position, trace in enumerate(ordered):
        if not isinstance(trace, Mapping):
            reasons.append(f"trace_{position}_not_mapping")
            continue
        try:
            if trace.get("episode_id") != episode_id or trace.get("target_key") != target_key:
                reasons.append(f"trace_{position}_identity_mismatch")
            if trace.get("step") != position:
                reasons.append(f"trace_{position}_step_not_contiguous")
            _time(trace.get("captured_at"), f"trace[{position}].captured_at")
            if _time(trace["captured_at"], f"trace[{position}].captured_at") > cutoff:
                reasons.append(f"trace_{position}_after_cutoff")
            action_id = _text(trace.get("action_id"), f"trace[{position}].action_id")
            candidate_id = _text(trace.get("candidate_id"), f"trace[{position}].candidate_id")
            _require_digest(trace.get("state_before_digest"), f"trace[{position}].state_before_digest")
            if action_id == "STOP":
                if candidate_id != "STOP" or not _text(trace.get("stop_reason"), f"trace[{position}].stop_reason"):
                    reasons.append(f"trace_{position}_invalid_stop")
                if position != len(ordered) - 1:
                    reasons.append("stop_must_be_terminal")
                stop_seen = True
                if previous_prediction is not None:
                    before = _probability_vector(trace.get("prediction_before"), f"trace[{position}].prediction_before")
                    if before != previous_prediction:
                        reasons.append(f"trace_{position}_prediction_continuity")
                if trace.get("state_after_digest") is not None:
                    _require_digest(trace.get("state_after_digest"), f"trace[{position}].state_after_digest")
                continue
            if stop_seen:
                reasons.append("action_after_stop")
            if candidate_id not in menu:
                reasons.append(f"trace_{position}_candidate_not_in_menu")
            request_id = _text(trace.get("request_id"), f"trace[{position}].request_id")
            before = _probability_vector(trace.get("prediction_before"), f"trace[{position}].prediction_before")
            after = _probability_vector(trace.get("prediction_after"), f"trace[{position}].prediction_after")
            if previous_prediction is not None and before != previous_prediction:
                reasons.append(f"trace_{position}_prediction_continuity")
            _require_digest(trace.get("state_after_digest"), f"trace[{position}].state_after_digest")
            previous_prediction = after
            if trace.get("observed_outcome") is None:
                reasons.append(f"trace_{position}_observation_missing")
            key = (episode_id, target_key, candidate_id, request_id)
            aggregate = aggregate_map.get(key)
            if aggregate is None:
                reasons.append(f"trace_{position}_aggregate_missing")
            else:
                if aggregate.get("prediction_cutoff") != actor.get("prediction_cutoff"):
                    reasons.append(f"trace_{position}_aggregate_cutoff_mismatch")
                if aggregate.get("availability") != "available" or aggregate.get("quality") not in {"VALIDATED", "PARTIAL"}:
                    reasons.append(f"trace_{position}_aggregate_unusable")
                _require_digest(aggregate.get("aggregate_id"), f"aggregate[{position}].aggregate_id")
                if not isinstance(aggregate.get("source_ids"), Sequence) or not aggregate.get("source_ids"):
                    reasons.append(f"trace_{position}_aggregate_sources_missing")
                available_at = _time(aggregate.get("operational_available_at"), f"aggregate[{position}].operational_available_at")
                captured_at = _time(aggregate.get("captured_at"), f"aggregate[{position}].captured_at")
                if not captured_at <= available_at <= cutoff:
                    reasons.append(f"trace_{position}_aggregate_after_cutoff")
                if abs(_finite(trace.get("expected_cost"), f"trace[{position}].expected_cost") - float(menu[candidate_id]["cost"])) > _EPS:
                    reasons.append(f"trace_{position}_cost_mismatch")
            action_rows.append({"step": position, "candidate_id": candidate_id, "request_id": request_id,
                                "prediction_before": list(before), "prediction_after": list(after),
                                "aggregate_bound": aggregate is not None})
        except (TypeError, ValueError, KeyError) as exc:
            reasons.append(str(exc))
    if not stop_seen:
        reasons.append("terminal_stop_missing")
    sidecar = evaluator_sidecar if isinstance(evaluator_sidecar, Mapping) else {}
    label_status = sidecar.get("label_status")
    if label_status not in SUPPORTED_LABELS:
        reasons.append("independent_result_label_missing")
    sequence = ">".join(row["candidate_id"] for row in action_rows)
    if not isinstance(sidecar.get("sequence_losses"), Mapping) or sequence not in sidecar.get("sequence_losses", {}):
        reasons.append("executed_sequence_loss_missing")
    if not isinstance(sidecar.get("action_returns"), Mapping):
        reasons.append("action_returns_missing")
    else:
        for row in action_rows:
            returned = sidecar["action_returns"].get(row["candidate_id"])
            if not isinstance(returned, Mapping):
                reasons.append(f"action_return_missing:{row['candidate_id']}")
    ready = not reasons
    return {
        "schema_version": VERSION,
        "status": "COMPLETE_ACTION_EVIDENCE" if ready else "PARTIAL_ACTION_EVIDENCE",
        "eligible_for_replay": ready,
        "numeric_voi_training_eligible": False,
        "episode_id": episode_id,
        "target_key": target_key,
        "prediction_cutoff": actor.get("prediction_cutoff"),
        "public_snapshot_digest": actor.get("public_snapshot_digest"),
        "trace_count": len(ordered),
        "action_count": len(action_rows),
        "selected_sequence": sequence,
        "actions": action_rows,
        "stop_reason": ordered[-1].get("stop_reason") if ordered and isinstance(ordered[-1], Mapping) else None,
        "independent_result_status": label_status,
        "reasons": sorted(set(reasons)),
        "public_projection": {
            "episode_id": episode_id, "target_key": target_key,
            "prediction_cutoff": actor.get("prediction_cutoff"),
            "trace_count": len(ordered), "selected_sequence": sequence,
            "hidden_result_included": False,
        },
    }


__all__ = ["VERSION", "SUPPORTED_LABELS", "validate_round165_episode_evidence"]
