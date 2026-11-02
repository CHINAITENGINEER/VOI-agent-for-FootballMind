"""Offline evidence-state evaluation, isolated from runtime selection/training.

This evaluator consumes an already recorded acquisition sequence. It does not
select actions, generate predictions, call tools, or certify expected VoI.
Public evidence and evaluator-only sequence losses enter through separate
arguments. Output ``public`` is the sole actor-safe projection; ``sidecar`` and
``summary`` are evaluator artifacts and must stay outside runtime prompts.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Mapping


VERSION = "round161-offline-evidence-evaluator-v1"
CONDITIONS = frozenset({"complete", "missing", "stale", "conflict", "late"})
_PRIVATE = frozenset({
    "sidecar", "evaluator", "gold", "label", "labels", "outcome", "result",
    "home_score", "away_score", "final_score", "actual_result", "true_outcome",
    "decision_loss", "decision_loss_before", "decision_loss_after", "loss",
    "realized_loss", "sequence_losses", "observed_gain", "risk_reduction",
})


def _object(value: Any, keys: set[str], name: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name}: exact fields required")
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name}: nonempty unpadded string required")
    return value


def _number(value: Any, name: str, maximum: float | None = None) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid or (maximum is not None and value > maximum):
        raise ValueError(f"{name}: finite number in the declared range required")
    return float(value)


def _time(value: Any, name: str) -> datetime:
    _text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone required")
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: timezone-aware timestamp required") from exc


def _public(value: Any, path: str) -> Any:
    if type(value) is dict:
        output = {}
        for key, child in value.items():
            _text(key, path + ".key")
            if key.casefold() in _PRIVATE or key.casefold().startswith(("private_", "evaluator_")):
                raise ValueError(f"{path}: evaluator field in public data")
            output[key] = _public(child, path + "." + key)
        return output
    if type(value) is list:
        return [_public(child, path + "[]") for child in value]
    if value is None or type(value) in (str, bool, int):
        return deepcopy(value)
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"{path}: finite JSON values required")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _losses(sidecar: dict | None, episode: str, cutoff: datetime,
            step_count: int) -> tuple[dict[int, float], dict]:
    if sidecar is None:
        return {}, {"label_status": "unavailable", "evaluation_id": None,
                    "label_available_at": None, "loss_unit": "decision_loss"}
    _object(sidecar, {"episode_id", "evaluation_id", "label_status", "label_available_at",
                      "loss_unit", "sequence_losses"}, "evaluator_sidecar")
    if sidecar["episode_id"] != episode:
        raise ValueError("evaluator sidecar episode mismatch")
    _text(sidecar["evaluation_id"], "evaluation_id")
    if sidecar["label_status"] not in {"independent_evaluator", "synthetic_fixture", "unavailable"}:
        raise ValueError("unsupported evaluator label_status")
    if sidecar["loss_unit"] != "decision_loss":
        raise ValueError("normalized decision_loss is the only supported loss unit")
    if _time(sidecar["label_available_at"], "label_available_at") < cutoff:
        raise ValueError("evaluation label must not precede prediction cutoff")
    if type(sidecar["sequence_losses"]) is not list:
        raise ValueError("sequence_losses: list required")
    rows = {}
    for item in sidecar["sequence_losses"]:
        _object(item, {"sequence", "loss"}, "sequence loss")
        sequence = item["sequence"]
        if type(sequence) is not int or not 0 <= sequence <= step_count or sequence in rows:
            raise ValueError("sequence loss: unique in-range sequence required")
        rows[sequence] = _number(item["loss"], "sequence loss", maximum=1)
    return rows, {key: deepcopy(sidecar[key]) for key in
                  ("label_status", "evaluation_id", "label_available_at", "loss_unit")}


def evaluate_episode(public_trace: Mapping[str, Any], evaluator_sidecar: dict | None = None) -> dict:
    """Replay evidence admission and score only fully supplied sequence losses.

    A freshness threshold and cutoff are frozen in the public trace. Evidence
    marked complete is downgraded when its timestamps violate that boundary.
    Sequence 0 is the initial loss; each subsequent sequence is one completed
    acquisition attempt, including failed or unusable returns. Missing losses
    make the full sequence ineligible rather than silently scoring its endpoints.
    Supplied losses/provenance remain declarations, not verified gold or training.
    """
    _object(public_trace, {"episode_id", "target_key", "prediction_cutoff", "max_age_seconds",
                           "initial_context", "steps", "terminal_action", "terminal_reason",
                           "policy_origin"}, "public_trace")
    episode = _text(public_trace["episode_id"], "episode_id")
    target = _text(public_trace["target_key"], "target_key")
    cutoff = _time(public_trace["prediction_cutoff"], "prediction_cutoff")
    max_age = _number(public_trace["max_age_seconds"], "max_age_seconds")
    if public_trace["terminal_action"] not in {"stop", "defer", "continue"}:
        raise ValueError("unsupported terminal_action")
    _text(public_trace["terminal_reason"], "terminal_reason")
    if public_trace["policy_origin"] not in {"recorded_runtime", "scripted_fixture"}:
        raise ValueError("unsupported policy_origin")
    if type(public_trace["initial_context"]) is not dict or type(public_trace["steps"]) is not list:
        raise ValueError("initial_context must be an object and steps a list")
    context = _public(public_trace["initial_context"], "initial_context")
    initial = deepcopy(context)
    events, receipt_ids = [], set()
    for sequence, raw in enumerate(public_trace["steps"], 1):
        _object(raw, {"sequence", "action_id", "receipt_id", "condition", "read_succeeded",
                      "observed_at", "available_at", "source_refs", "context_patch",
                      "declared_cost"}, "step")
        if type(raw["sequence"]) is not int or raw["sequence"] != sequence:
            raise ValueError("step sequences must be consecutive starting at one")
        _text(raw["action_id"], "action_id")
        rid = _text(raw["receipt_id"], "receipt_id")
        if rid in receipt_ids:
            raise ValueError("duplicate receipt_id")
        receipt_ids.add(rid)
        condition = raw["condition"]
        if type(condition) is not str or condition not in CONDITIONS:
            raise ValueError("unsupported evidence condition")
        if type(raw["read_succeeded"]) is not bool:
            raise ValueError("read_succeeded must be boolean")
        if type(raw["context_patch"]) is not dict or type(raw["source_refs"]) is not list:
            raise ValueError("context_patch must be an object and source_refs a list")
        patch = _public(raw["context_patch"], "context_patch")
        refs = [_text(ref, "source_ref") for ref in raw["source_refs"]]
        if len(refs) != len(set(refs)):
            raise ValueError("source_refs must be unique")
        if condition == "conflict" and len(refs) < 2:
            raise ValueError("conflict evidence requires at least two source_refs")
        observed = None if raw["observed_at"] is None else _time(raw["observed_at"], "observed_at")
        available = None if raw["available_at"] is None else _time(raw["available_at"], "available_at")
        if observed is not None and available is not None and observed > available:
            raise ValueError("source observation must precede availability")
        if condition == "complete" and (not raw["read_succeeded"] or not refs or observed is None or available is None):
            raise ValueError("complete evidence requires successful read, refs and both clocks")
        effective = condition
        if available is not None and available > cutoff or observed is not None and observed > cutoff:
            effective = "late"
        elif condition == "complete" and (cutoff - observed).total_seconds() > max_age:
            effective = "stale"
        before = _digest(context)
        admitted = effective == "complete"
        if admitted:
            context.update(deepcopy(patch))
        changed = before != _digest(context)
        events.append({
            "sequence": sequence, "action_id": raw["action_id"], "receipt_id": rid,
            "declared_condition": condition, "effective_condition": effective,
            "read_succeeded": raw["read_succeeded"], "evidence_admitted": admitted,
            "context_changed": changed, "context_digest_before": before,
            "context_digest_after": _digest(context), "replanning_required": not admitted,
            "replan_reason": None if admitted else "evidence_" + effective,
            "source_refs": refs, "observed_at": raw["observed_at"],
            "available_at": raw["available_at"],
            "declared_cost": _number(raw["declared_cost"], "declared_cost"),
            "cost_unit": "decision_loss", "cost_basis": "declared_not_calibrated",
        })
    public = {
        "schema_version": VERSION, "episode_id": episode, "target_key": target,
        "prediction_cutoff": public_trace["prediction_cutoff"], "max_age_seconds": max_age,
        "initial_context": initial, "final_context": deepcopy(context), "events": events,
        "terminal_action": public_trace["terminal_action"], "terminal_reason": public_trace["terminal_reason"],
        "policy_origin": public_trace["policy_origin"], "policy_origin_verified": False,
        "numeric_voi_calibrated": False, "training_eligible": False,
    }
    losses, provenance = _losses(evaluator_sidecar, episode, cutoff, len(events))
    missing = sorted(set(range(len(events) + 1)) - set(losses))
    eligible = not missing and provenance["label_status"] != "unavailable"
    reason = ("complete_declared_sequence_losses" if eligible else
              "sequence_loss_missing" if missing else "evaluation_labels_unavailable")
    costs = math.fsum(row["declared_cost"] for row in events)
    gain = losses[0] - losses[len(events)] if eligible else None
    sidecar = {
        "schema_version": VERSION, "episode_id": episode, "evaluator_only": True,
        **provenance, "provenance_verified": False, "public_trace_digest": _digest(public),
        "sequence_loss_eligible": eligible, "eligibility_reason": reason,
        "missing_loss_sequences": missing,
        "sequence_losses": [{"sequence": key, "loss": value} for key, value in sorted(losses.items())],
        "step_measurements": [{"sequence": row["sequence"], "action_id": row["action_id"],
             "read_succeeded": row["read_succeeded"], "evidence_admitted": row["evidence_admitted"],
             "observed_loss_reduction": losses[row["sequence"] - 1] - losses[row["sequence"]] if eligible else None}
             for row in events],
        "observed_loss_reduction": gain, "declared_acquisition_cost": costs,
        "observed_net_loss_reduction": gain - costs if eligible else None,
        "read_success_means_gain": False, "observed_gain_is_expected_voi": False,
        "numeric_voi_calibrated": False, "training_eligible": False,
    }
    summary = {
        "schema_version": VERSION, "episode_id": episode, "evaluator_only": True,
        "action_count": len(events), "read_success_count": sum(row["read_succeeded"] for row in events),
        "evidence_admitted_count": sum(row["evidence_admitted"] for row in events),
        "context_change_count": sum(row["context_changed"] for row in events),
        "replan_count": sum(row["replanning_required"] for row in events),
        "condition_counts": {key: sum(row["effective_condition"] == key for row in events) for key in sorted(CONDITIONS)},
        "sequence_loss_eligible": eligible, "eligibility_reason": reason,
        "observed_loss_reduction": gain, "declared_acquisition_cost": costs,
        "observed_net_loss_reduction": gain - costs if eligible else None,
        "terminal_action": public_trace["terminal_action"], "stop_optimality": "unmeasured",
        "policy_execution_measured": False, "numeric_voi_calibrated": False, "training_eligible": False,
    }
    return {"public": public, "sidecar": sidecar, "summary": summary}


__all__ = ["VERSION", "CONDITIONS", "evaluate_episode"]
