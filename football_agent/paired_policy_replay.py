"""Strict paired policy replay and trajectory alignment.

This is a projection layer on top of :mod:`counterfactual_policy_replay`.
It gives the four arms used by the research protocol stable names:

``autonomous``
    The state-bound interval/VoI selector.  The selector is supplied by the
    caller and never reads evaluator-only fields.
``fixed_order`` and ``full_query``
    Declared controls from the existing replay engine.
``stop``
    An explicit zero-query control.

All arms replay the same public initial state, candidate menu and budget.  The
returned trajectory alignment is useful even when an episode cannot be scored.
Missing prediction-before/after vectors, independent outcome, exact sequence
loss, or action clocks makes a pair ``comparable=False``.  Unknown metrics are
represented by ``None`` and are never replaced by zero.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any

from .counterfactual_policy_replay import (
    INDEPENDENT_GOLD,
    INDEPENDENT_GOLD_RECONSTRUCTED,
    replay_counterfactual_episode,
)


VERSION = "paired-policy-replay-v1"
POLICIES = ("autonomous", "fixed_order", "full_query", "stop")
_AUTONOMOUS_ENGINE_POLICY = "interval_voi"
_SYNTHETIC_OFFLINE = "SYNTHETIC_OFFLINE"
_EPS = 1e-12


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _vector(value: Any, name: str) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return None
    result = [_finite(item, f"{name}[]") for item in value]
    if any(item < -_EPS for item in result):
        return None
    total = math.fsum(result)
    if abs(total - 1.0) > 1e-8:
        return None
    return result


def _valid_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _loss_value(metric: str, vector: Sequence[float], outcome_index: int) -> float | None:
    """Recompute a loss from the explicit prediction and independent label."""
    if outcome_index < 0 or outcome_index >= len(vector):
        return None
    if metric == "log_loss":
        return -math.log(max(float(vector[outcome_index]), _EPS))
    if metric == "brier":
        return math.fsum((float(value) - (1.0 if pos == outcome_index else 0.0)) ** 2
                         for pos, value in enumerate(vector))
    return None


def _public_manifest(episode: Mapping[str, Any]) -> dict[str, Any]:
    public = episode.get("public")
    if not isinstance(public, Mapping):
        raise ValueError("episode.public is required")
    raw_menu = public.get("candidate_menu", public.get("candidates"))
    if not isinstance(raw_menu, Sequence) or isinstance(raw_menu, (str, bytes)) or not raw_menu:
        raise ValueError("public candidate_menu is required")
    menu: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_menu):
        row = {"candidate_id": raw} if isinstance(raw, str) else raw
        if not isinstance(row, Mapping):
            raise ValueError(f"candidate_menu[{index}] must be a mapping")
        candidate_id = _text(row.get("candidate_id"), f"candidate_menu[{index}].candidate_id")
        if candidate_id in seen:
            raise ValueError("candidate IDs must be unique")
        seen.add(candidate_id)
        cost = row.get("cost")
        if cost is not None:
            cost = _finite(cost, f"candidate_menu[{index}].cost")
            if cost < -_EPS:
                raise ValueError("candidate cost must be nonnegative")
        menu.append({"candidate_id": candidate_id, "cost": cost,
                     "source_id": row.get("source_id"), "action_id": row.get("action_id"),
                     "action_kind": row.get("action_kind", "query")})
    cutoff = _time(public.get("prediction_cutoff"), "prediction_cutoff")
    budget = _finite(public.get("budget"), "public.budget")
    if budget < -_EPS:
        raise ValueError("public.budget must be nonnegative")
    return {
        "target_key": _text(public.get("target_key"), "public.target_key"),
        "kickoff": None if public.get("kickoff") is None else _time(public.get("kickoff"), "public.kickoff").isoformat(),
        "prediction_cutoff": cutoff.isoformat(),
        "initial_context": deepcopy(dict(public.get("initial_context", {}))) if isinstance(public.get("initial_context", {}), Mapping) else {},
        "initial_prediction": deepcopy(public.get("initial_prediction")),
        "candidate_generation_complete": public.get("candidate_generation_complete") is True,
        "budget": budget,
        "loss_unit": public.get("loss_unit", "decision_loss"),
        "candidate_menu": menu,
    }


def _stop_result(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Produce a zero-query control without consulting a provider or VoI estimate."""
    manifest = _public_manifest(episode)
    episode_id = _text(episode.get("episode_id"), "episode_id")
    initial_state = {
        "schema_version": VERSION,
        "episode_id": episode_id,
        "target_key": manifest["target_key"],
        "prediction_cutoff": manifest["prediction_cutoff"],
        "step": 0,
        "context": deepcopy(manifest["initial_context"]),
        "remaining_candidate_ids": [row["candidate_id"] for row in manifest["candidate_menu"]],
        "candidate_generation_complete": manifest["candidate_generation_complete"],
    }
    state_digest = _digest(initial_state)
    trace = {
        "selected_actions": [], "state_digests": [state_digest],
        "transition_statuses": [], "stop_reason": "declared_stop",
        "query_count": 0, "spent_cost": 0.0, "loss_unit": manifest["loss_unit"],
        "final_context": deepcopy(manifest["initial_context"]),
        "hidden_result_included": False,
    }
    sidecar = episode.get("evaluation_sidecar", episode.get("evaluator_sidecar"))
    sidecar = sidecar if isinstance(sidecar, Mapping) else {}
    raw_losses = sidecar.get("sequence_losses")
    stop_loss = None
    if isinstance(raw_losses, Mapping) and "" in raw_losses:
        try:
            stop_loss = _finite(raw_losses[""], "sequence_losses[]")
        except ValueError:
            stop_loss = None
    baseline = sidecar.get("baseline_loss")
    try:
        baseline = None if baseline is None else _finite(baseline, "baseline_loss")
    except ValueError:
        baseline = None
    return {
        "schema_version": VERSION, "episode_id": episode_id, "policy": "stop",
        "runtime_eligible": True, "actor_trace": trace,
        "evaluator": {"counterfactual_eligible": False,
                       "counterfactual_ineligible_reason": "stop_control_requires_explicit_prediction_and_loss",
                       "sequence": [], "final_loss": stop_loss, "baseline_loss": baseline,
                       "decision_loss_reduction": None, "net_utility": None,
                       "unknown_metrics_are_not_zero": True},
    }


def _action_return(sidecar: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any] | None:
    returns = sidecar.get("action_returns")
    if not isinstance(returns, Mapping) or not isinstance(returns.get(candidate_id), Mapping):
        return None
    return returns[candidate_id]


def _trajectory_rows(episode: Mapping[str, Any], policy: str, result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Project actor actions plus explicitly supplied evaluator clocks/predictions."""
    sidecar = episode.get("evaluation_sidecar", episode.get("evaluator_sidecar"))
    if not isinstance(sidecar, Mapping):
        sidecar = {}
    selected = result.get("actor_trace", {}).get("selected_actions", [])
    state_digests = result.get("actor_trace", {}).get("state_digests", [])
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for step, candidate_id in enumerate(selected):
        returned = _action_return(sidecar, candidate_id)
        row: dict[str, Any] = {
            "step": step, "candidate_id": candidate_id,
            "action_id": None if returned is None else returned.get("action_id"),
            "state_before_digest": state_digests[step] if step < len(state_digests) else None,
            "state_after_digest": None if returned is None else returned.get("next_state_digest"),
            "prediction_before": None if returned is None else deepcopy(returned.get("prediction_before")),
            "prediction_after": None if returned is None else deepcopy(returned.get("prediction_after")),
            "observed_at": None if returned is None else returned.get("observed_at"),
            "operational_available_at": None if returned is None else returned.get("operational_available_at", returned.get("available_at")),
            "captured_at": None if returned is None else returned.get("captured_at"),
        }
        if returned is None:
            reasons.append(f"step_{step}_action_return_missing")
        before_vector = _vector(row.get("prediction_before"), f"{policy}.step_{step}.prediction_before")
        after_vector = _vector(row.get("prediction_after"), f"{policy}.step_{step}.prediction_after")
        for key, vector in (("prediction_before", before_vector), ("prediction_after", after_vector)):
            if vector is None:
                reasons.append(f"step_{step}_{key}_missing_or_invalid")
        if before_vector is not None and after_vector is not None and len(before_vector) != len(after_vector):
            reasons.append(f"step_{step}_prediction_dimension_mismatch")
        if row["state_before_digest"] is None:
            reasons.append(f"step_{step}_state_before_missing")
        clocks: dict[str, datetime] = {}
        for key in ("observed_at", "operational_available_at", "captured_at"):
            try:
                if row[key] is None:
                    raise ValueError
                clock = _time(row[key], f"{policy}.step_{step}.{key}")
                clocks[key] = clock
                cutoff = _time(_public_manifest(episode)["prediction_cutoff"], "prediction_cutoff")
                if clock > cutoff:
                    reasons.append(f"step_{step}_{key}_after_cutoff")
            except (TypeError, ValueError):
                reasons.append(f"step_{step}_{key}_missing_or_invalid")
        if len(clocks) == 3:
            if not clocks["observed_at"] <= clocks["operational_available_at"] <= clocks["captured_at"]:
                reasons.append(f"step_{step}_clock_order_invalid")
        rows.append(row)
    terminal = {
        "step": len(rows), "candidate_id": "STOP", "action_id": "STOP",
        "state_before_digest": state_digests[-1] if state_digests else None,
        "state_after_digest": None, "stop_reason": result.get("actor_trace", {}).get("stop_reason"),
        "prediction_before": None, "prediction_after": None,
        "observed_at": None,
        "operational_available_at": None, "captured_at": None,
    }
    rows.append(terminal)
    # A zero-query stop has no acquisition clocks.  Its terminal prediction is
    # the explicitly recorded public initial prediction; setting both sides to
    # that vector makes the no-op transition visible without borrowing a
    # prediction from another policy.
    if policy == "stop":
        public = episode.get("public") if isinstance(episode.get("public"), Mapping) else {}
        initial = _vector(public.get("initial_prediction"), "public.initial_prediction")
        if initial is not None:
            rows[-1]["prediction_before"] = list(initial)
            rows[-1]["prediction_after"] = list(initial)
    return rows, sorted(set(reasons))


def _strict_evidence(episode: Mapping[str, Any], policy: str, result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], reasons: Sequence[str]) -> tuple[bool, list[str], dict[str, Any]]:
    sidecar = episode.get("evaluation_sidecar", episode.get("evaluator_sidecar"))
    sidecar = sidecar if isinstance(sidecar, Mapping) else {}
    all_reasons = list(reasons)
    label_status = sidecar.get("label_status")
    if label_status not in {INDEPENDENT_GOLD, INDEPENDENT_GOLD_RECONSTRUCTED}:
        all_reasons.append("independent_gold_label_missing")
    if not isinstance(sidecar.get("independent_result_bound"), bool) or sidecar.get("independent_result_bound") is not True:
        all_reasons.append("independent_outcome_missing_or_unbound")
    if sidecar.get("outcome") is None and not (isinstance(sidecar.get("independent_result"), Mapping) and sidecar["independent_result"].get("outcome") is not None):
        all_reasons.append("independent_outcome_missing")
    evaluator = result.get("evaluator") if isinstance(result.get("evaluator"), Mapping) else {}
    if evaluator.get("final_loss") is None:
        all_reasons.append("executed_sequence_loss_missing")
    if evaluator.get("baseline_loss") is None:
        all_reasons.append("baseline_loss_missing")
    # ``STOP`` has no acquisition rows, so it needs a declared zero-query
    # baseline and an explicit initial prediction to be scored.  Do not infer
    # either from another policy's trace.
    if policy == "stop":
        public = episode.get("public") if isinstance(episode.get("public"), Mapping) else {}
        if _vector(public.get("initial_prediction"), "public.initial_prediction") is None:
            all_reasons.append("stop_initial_prediction_missing")
    return (not all_reasons), sorted(set(all_reasons)), {
        "label_status": label_status,
        "independent_outcome_bound": sidecar.get("independent_result_bound") is True and (
            sidecar.get("outcome") is not None or
            (isinstance(sidecar.get("independent_result"), Mapping) and sidecar["independent_result"].get("outcome") is not None)
        ),
        "final_loss": evaluator.get("final_loss") if not all_reasons else None,
        "baseline_loss": evaluator.get("baseline_loss") if not all_reasons else None,
    }


def _paired_delta(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    base_ok, other_ok = base.get("comparable") is True, other.get("comparable") is True
    reasons = sorted(set(list(base.get("incomparable_reasons", [])) + list(other.get("incomparable_reasons", []))))
    return {
        "episode_id": base.get("episode_id"), "baseline_policy": base.get("policy"),
        "policy": other.get("policy"), "comparable": bool(base_ok and other_ok),
        "incomparable_reasons": [] if base_ok and other_ok else reasons,
        "final_loss_delta": (other["final_loss"] - base["final_loss"]) if base_ok and other_ok else None,
        "query_count_delta": (other.get("query_count") - base.get("query_count")) if base.get("query_count") is not None and other.get("query_count") is not None else None,
        "spent_cost_delta": (other.get("spent_cost") - base.get("spent_cost")) if base.get("spent_cost") is not None and other.get("spent_cost") is not None else None,
    }


def project_independent_registry_sidecar(
    outcome_record: Mapping[str, Any], loss_registry: Mapping[str, Any], *,
    action_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project an independent-outcome registry into the paired sidecar.

    The registry's per-action loss rows do not contain source availability or
    prediction vectors by themselves.  Those fields must therefore be
    supplied explicitly under ``action_evidence`` (or the registry's own
    ``action_evidence`` extension).  Missing fields produce ``PARTIAL`` or
    ``UNKNOWN`` and are never reconstructed from the outcome clock or loss
    values.
    """
    reasons: list[str] = []
    if not isinstance(outcome_record, Mapping) or not isinstance(loss_registry, Mapping):
        return {"schema_version": VERSION, "status": "UNKNOWN", "sidecar": {},
                "reasons": ["outcome_record_or_loss_registry_missing"],
                "numeric_voi_training_eligible": False}
    if outcome_record.get("status") != INDEPENDENT_GOLD or outcome_record.get("independent_from_actor") is not True:
        reasons.append("independent_outcome_record_not_gold_or_not_independent")
    record_digest = outcome_record.get("record_digest")
    if not isinstance(record_digest, str) or len(record_digest) != 64:
        reasons.append("outcome_record_digest_missing")
    else:
        record_copy = dict(outcome_record)
        record_copy.pop("record_digest", None)
        if _digest(record_copy) != record_digest:
            reasons.append("outcome_record_digest_invalid")
    if loss_registry.get("outcome_record_digest") != record_digest:
        reasons.append("loss_registry_outcome_digest_mismatch")
    if loss_registry.get("status") != "COMPUTED":
        reasons.append("loss_registry_not_computed")
    if loss_registry.get("numeric_voi_training_eligible") is not True:
        reasons.append("loss_registry_not_numeric_voi_eligible")
    episode_id = outcome_record.get("episode_id")
    target_key = outcome_record.get("target_key")
    cutoff = outcome_record.get("prediction_cutoff")
    # A digest and a self-declared ``INDEPENDENT_GOLD`` flag are insufficient
    # provenance.  Require the source and both acquisition clocks before a
    # result can be projected into an evaluator sidecar.  The result clock is
    # deliberately after the actor cutoff: it can score a completed episode,
    # but it cannot become actor-visible evidence.
    for field in ("source", "source_id", "source_payload_digest", "actor_binding_digest"):
        value = outcome_record.get(field)
        if field.endswith("digest"):
            if not _valid_digest(value):
                reasons.append(f"independent_{field}_missing_or_invalid")
        elif not isinstance(value, str) or not value.strip():
            reasons.append(f"independent_{field}_missing")
    actor_source_ids = outcome_record.get("actor_source_ids")
    actor_source_digests = outcome_record.get("actor_source_digests")
    actor_sources = outcome_record.get("actor_sources")
    if not isinstance(actor_source_ids, Sequence) or isinstance(actor_source_ids, (str, bytes)):
        reasons.append("independent_actor_source_ids_missing")
        actor_source_ids = []
    if not isinstance(actor_source_digests, Sequence) or isinstance(actor_source_digests, (str, bytes)):
        reasons.append("independent_actor_source_digests_missing")
        actor_source_digests = []
    if not isinstance(actor_sources, Sequence) or isinstance(actor_sources, (str, bytes)):
        reasons.append("independent_actor_sources_missing")
        actor_sources = []
    if outcome_record.get("source") in set(actor_sources):
        reasons.append("independent_source_overlaps_actor")
    source_id = outcome_record.get("source_id")
    source_id_tokens = {str(item) for item in actor_source_ids}
    source_id_tokens.update(str(item).rsplit(":", 1)[-1] for item in actor_source_ids)
    if source_id in source_id_tokens:
        reasons.append("independent_source_id_overlaps_actor")
    if outcome_record.get("source_payload_digest") in {str(item) for item in actor_source_digests}:
        reasons.append("independent_payload_digest_overlaps_actor")
    try:
        cutoff_time = _time(cutoff, "prediction_cutoff")
        kickoff_result = _time(outcome_record.get("kickoff"), "outcome.kickoff")
        observed_result = _time(outcome_record.get("observed_at"), "outcome.observed_at")
        available_result = _time(outcome_record.get("available_at"), "outcome.available_at")
        if kickoff_result > observed_result:
            reasons.append("independent_outcome_observed_before_kickoff")
        if observed_result > available_result:
            reasons.append("independent_outcome_clock_order_invalid")
        if available_result <= cutoff_time:
            reasons.append("independent_outcome_available_before_cutoff")
    except ValueError:
        reasons.append("independent_outcome_clocks_missing_or_invalid")
    label = outcome_record.get("outcome_label")
    mapping = outcome_record.get("label_mapping")
    outcome_index = None
    if not isinstance(label, str) or not isinstance(mapping, Mapping) or label not in mapping:
        reasons.append("independent_outcome_label_mapping_missing")
    elif (any(not isinstance(key, str) for key in mapping)
          or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                 for value in mapping.values())
          or len(set(mapping.values())) != len(mapping)):
        reasons.append("independent_outcome_label_mapping_invalid")
    else:
        outcome_index = mapping[label]
    raw_sequence = loss_registry.get("sequence_losses")
    sequence_losses: dict[str, float] | None = None
    if isinstance(raw_sequence, Mapping):
        sequence_losses = {}
        for key, value in raw_sequence.items():
            if not isinstance(key, str):
                reasons.append("sequence_loss_key_invalid")
                continue
            try:
                sequence_losses[key] = _finite(value, "sequence_loss")
            except ValueError:
                reasons.append(f"sequence_loss_invalid:{key}")
        if "" not in sequence_losses:
            reasons.append("baseline_sequence_loss_missing")
    else:
        reasons.append("sequence_losses_missing")
    evidence = action_evidence if action_evidence is not None else loss_registry.get("action_evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
        reasons.append("action_evidence_missing")
    raw_results = loss_registry.get("results")
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)) or not raw_results:
        reasons.append("loss_registry_results_missing")
        raw_results = []
    seen_candidates: set[str] = set()
    action_returns: dict[str, dict[str, Any]] = {}
    for index, loss_row in enumerate(raw_results):
        if not isinstance(loss_row, Mapping):
            reasons.append(f"loss_result_{index}_invalid")
            continue
        candidate = loss_row.get("candidate_id")
        if not isinstance(candidate, str) or not candidate.strip():
            reasons.append(f"loss_result_{index}_candidate_missing")
            continue
        if candidate in seen_candidates:
            reasons.append(f"loss_result_{candidate}_duplicate")
            continue
        seen_candidates.add(candidate)
        if loss_row.get("status") != "COMPUTED":
            reasons.append(f"loss_result_{candidate}_not_computed")
        if loss_row.get("outcome_record_digest") != record_digest:
            reasons.append(f"loss_result_{candidate}_outcome_digest_mismatch")
        item = evidence.get(candidate)
        if not isinstance(item, Mapping):
            reasons.append(f"action_evidence_missing:{candidate}")
            continue
        required = ("available", "prediction_before", "prediction_after", "observed_at", "available_at", "captured_at", "context_update")
        missing = [key for key in required if key not in item]
        if missing:
            reasons.append(f"action_evidence_incomplete:{candidate}:{','.join(missing)}")
            continue
        if item.get("available") is not True:
            reasons.append(f"action_evidence_unavailable:{candidate}")
            continue
        if loss_row.get("action_id") != item.get("action_id"):
            reasons.append(f"action_evidence_action_mismatch:{candidate}")
            continue
        if loss_row.get("request_id") != item.get("request_id"):
            reasons.append(f"action_evidence_request_mismatch:{candidate}")
            continue
        if not isinstance(item.get("action_id"), str) or not item["action_id"].strip():
            reasons.append(f"action_evidence_action_id_missing:{candidate}")
            continue
        if not isinstance(item.get("request_id"), str) or not item["request_id"].strip():
            reasons.append(f"action_evidence_request_id_missing:{candidate}")
            continue
        if not isinstance(item.get("context_update"), Mapping):
            reasons.append(f"action_evidence_context_update_invalid:{candidate}")
            continue
        before_vector = _vector(item.get("prediction_before"), f"action_evidence.{candidate}.prediction_before")
        after_vector = _vector(item.get("prediction_after"), f"action_evidence.{candidate}.prediction_after")
        if before_vector is None or after_vector is None:
            reasons.append(f"action_evidence_prediction_invalid:{candidate}")
            continue
        if len(before_vector) != len(after_vector):
            reasons.append(f"action_evidence_prediction_dimension_mismatch:{candidate}")
            continue
        metric = loss_row.get("metric", "log_loss")
        expected_before = _loss_value(metric, before_vector, outcome_index) if outcome_index is not None else None
        expected_after = _loss_value(metric, after_vector, outcome_index) if outcome_index is not None else None
        if expected_before is None or expected_after is None:
            reasons.append(f"loss_outcome_dimension_mismatch:{candidate}")
            continue
        try:
            declared_before = _finite(loss_row.get("decision_loss_before"), "decision_loss_before")
            declared_after = _finite(loss_row.get("decision_loss_after"), "decision_loss_after")
        except ValueError:
            reasons.append(f"loss_values_missing:{candidate}")
            continue
        if abs(declared_before - expected_before) > 1e-8 or abs(declared_after - expected_after) > 1e-8:
            reasons.append(f"loss_values_do_not_match_predictions:{candidate}")
            continue
        loss_digest = loss_row.get("loss_digest")
        if not _valid_digest(loss_digest):
            reasons.append(f"loss_digest_missing_or_invalid:{candidate}")
        else:
            loss_copy = dict(loss_row)
            loss_copy.pop("loss_digest", None)
            if _digest(loss_copy) != loss_digest:
                reasons.append(f"loss_digest_invalid:{candidate}")
        try:
            observed = _time(item.get("observed_at"), f"action_evidence.{candidate}.observed_at")
            available = _time(item.get("available_at"), f"action_evidence.{candidate}.available_at")
            captured = _time(item.get("captured_at"), f"action_evidence.{candidate}.captured_at")
            cutoff_time = _time(cutoff, "prediction_cutoff")
            if not observed <= available <= captured <= cutoff_time:
                reasons.append(f"action_evidence_clock_order_invalid:{candidate}")
                continue
        except ValueError:
            reasons.append(f"action_evidence_clock_invalid:{candidate}")
            continue
        action_returns[candidate] = {
            "available": item["available"],
            "context_update": deepcopy(item["context_update"]),
            "action_id": item.get("action_id"),
            "request_id": item.get("request_id"),
            "prediction_before": deepcopy(item["prediction_before"]),
            "prediction_after": deepcopy(item["prediction_after"]),
            "observed_at": item.get("observed_at"),
            "available_at": item.get("available_at"),
            "captured_at": item.get("captured_at"),
        }
    critical_outcome_failure = any(
        reason.startswith("independent_outcome_record") or
        reason.startswith("outcome_record_digest") or
        reason.startswith("loss_registry_outcome_digest") or
        reason.startswith("independent_outcome_label_mapping") or
        reason.startswith("independent_source") or
        reason.startswith("independent_payload") or
        reason.startswith("independent_actor_source") or
        reason.startswith("independent_actor_binding") or
        reason.startswith("independent_outcome_clock") or
        reason.startswith("independent_outcome_available") or
        reason.startswith("independent_outcome_observed")
        for reason in reasons
    )
    sidecar: dict[str, Any] = {
        "label_status": INDEPENDENT_GOLD if not critical_outcome_failure else None,
        "independent_result_bound": not critical_outcome_failure,
        "action_returns": action_returns,
    }
    if outcome_index is not None:
        sidecar["outcome"] = outcome_index
    if sequence_losses is not None and "" in sequence_losses:
        sidecar["sequence_losses"] = sequence_losses
        sidecar["baseline_loss"] = sequence_losses[""]
    if "cost_to_loss_rate" in loss_registry:
        try:
            sidecar["cost_to_loss_rate"] = _finite(loss_registry["cost_to_loss_rate"], "cost_to_loss_rate")
        except ValueError:
            reasons.append("cost_to_loss_rate_invalid")
    # Never put the score, outcome label, source payload or registry rows into
    # the sidecar projection.  The paired replay only needs the explicit
    # integer outcome index and action/sequence evidence above.
    status = "COMPUTED" if not reasons else ("PARTIAL" if action_returns or sequence_losses else "UNKNOWN")
    sidecar["independent_result_bound"] = sidecar["independent_result_bound"] and status != "UNKNOWN"
    return {
        "schema_version": VERSION, "status": status, "sidecar": sidecar,
        "reasons": sorted(set(reasons)),
        "numeric_voi_training_eligible": status == "COMPUTED",
        "public_projection": {"episode_id": episode_id, "target_key": target_key,
                               "prediction_cutoff": cutoff,
                               "outcome_visible_to_actor": False,
                               "loss_visible_to_actor": False},
    }


def compare_aligned_policies(
    episodes: Iterable[Mapping[str, Any]], *,
    interval_provider: Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]] | None = None,
    policies: Sequence[str] = POLICIES,
    accepted_label_statuses: Sequence[str] = (INDEPENDENT_GOLD, INDEPENDENT_GOLD_RECONSTRUCTED, _SYNTHETIC_OFFLINE),
) -> dict[str, Any]:
    """Run four named arms on matched public episodes and align each action.

    ``autonomous`` is the only arm allowed to use an interval/VoI provider.
    Controls never consult the provider and no arm receives the evaluator
    sidecar.  The evaluator projection is deliberately conservative: a pair
    is comparable only if all required evidence fields are present.
    """
    selected_policies = tuple(policies)
    if not selected_policies or len(set(selected_policies)) != len(selected_policies):
        raise ValueError("policies must be nonempty and unique")
    if any(policy not in POLICIES for policy in selected_policies):
        raise ValueError("unknown paired policy")
    raw = list(episodes)
    if not raw:
        raise ValueError("at least one episode is required")
    manifests = [_public_manifest(ep) for ep in raw]
    manifest_digest = _digest(manifests[0])
    if any(_digest(manifest) != manifest_digest for manifest in manifests[1:]):
        raise ValueError("paired episodes must share the same public initial state, menu, cutoff and budget")
    episode_ids = [_text(ep.get("episode_id"), "episode_id") for ep in raw]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate episode_id")
    results: dict[str, list[dict[str, Any]]] = {policy: [] for policy in selected_policies}
    alignments: list[dict[str, Any]] = []
    for ep in raw:
        by_policy: dict[str, dict[str, Any]] = {}
        for policy in selected_policies:
            if policy == "autonomous":
                if interval_provider is None:
                    raise ValueError("autonomous policy requires interval_provider")
                result = replay_counterfactual_episode(
                    ep, _AUTONOMOUS_ENGINE_POLICY, interval_provider=interval_provider,
                    accepted_label_statuses=accepted_label_statuses,
                )
            elif policy == "stop":
                result = _stop_result(ep)
            else:
                result = replay_counterfactual_episode(ep, policy,
                                                        accepted_label_statuses=accepted_label_statuses)
            rows, row_reasons = _trajectory_rows(ep, policy, result)
            comparable, evidence_reasons, evidence = _strict_evidence(ep, policy, result, rows, row_reasons)
            actor = result["actor_trace"]
            projection = {
                "episode_id": result["episode_id"], "policy": policy,
                "selected_actions": list(actor.get("selected_actions", [])),
                "query_count": actor.get("query_count"), "spent_cost": actor.get("spent_cost"),
                "stop_reason": actor.get("stop_reason"), "trajectory": rows,
                "comparable": comparable, "incomparable_reasons": evidence_reasons,
                "final_loss": evidence["final_loss"], "baseline_loss": evidence["baseline_loss"],
                "hidden_result_included": False,
            }
            results[policy].append(projection)
            by_policy[policy] = projection
        max_steps = max(len(row["trajectory"]) for row in by_policy.values())
        steps = [{"step": step, "by_policy": {
            policy: (deepcopy(by_policy[policy]["trajectory"][step]) if step < len(by_policy[policy]["trajectory"]) else None)
            for policy in selected_policies
        }} for step in range(max_steps)]
        alignments.append({"episode_id": ep["episode_id"], "public_manifest_digest": manifest_digest,
                           "steps": steps})
    baseline_policy = "autonomous" if "autonomous" in selected_policies else selected_policies[0]
    paired: dict[str, list[dict[str, Any]]] = {policy: [] for policy in selected_policies if policy != baseline_policy}
    for index in range(len(raw)):
        base = results[baseline_policy][index]
        for policy in paired:
            paired[policy].append(_paired_delta(base, results[policy][index]))
    summary: dict[str, dict[str, Any]] = {}
    for policy, rows in paired.items():
        comparable = [row for row in rows if row["comparable"]]
        summary[policy] = {
            "paired_episode_count": len(rows), "comparable_count": len(comparable),
            "incomparable_count": len(rows) - len(comparable),
            "mean_final_loss_delta": (math.fsum(row["final_loss_delta"] for row in comparable) / len(comparable)) if comparable else None,
            "unknown_metrics_are_not_zero": True,
        }
    return {
        "schema_version": VERSION, "policy_order": list(selected_policies),
        "baseline_policy": baseline_policy, "episode_count": len(raw),
        "public_manifest": {"digest": manifest_digest, "same_initial_state_menu_budget": True,
                            "episodes": [deepcopy(manifest) for manifest in manifests]},
        "results": results, "trajectory_alignments": alignments,
        "paired_deltas": paired, "paired_summary": summary,
        "contract": {
            "same_public_initial_state_menu_budget": True,
            "autonomous_selection_separated_from_controls": True,
            "fixed_order_is_declared_control": True,
            "full_query_is_declared_control": True,
            "stop_is_zero_query_control": True,
            "actor_trace_excludes_evaluator_fields": True,
            "unknown_metrics_are_not_zero": True,
            "missing_prediction_outcome_loss_clock_is_incomparable": True,
            "paired_by_episode_and_step": True,
        },
        "limitations": [
            "The autonomous interval provider is not calibrated by this adapter.",
            "Missing before/after predictions, independent outcome, loss, or action clocks remain incomparable.",
            "This adapter does not claim numeric VoI or football performance from synthetic or partial episodes.",
        ],
    }


# Descriptive aliases make the boundary easy to discover without duplicating
# the implementation or changing the historical Round162 API.
compare_paired_policies = compare_aligned_policies
align_policy_trajectories = compare_aligned_policies


__all__ = ["VERSION", "POLICIES", "compare_aligned_policies",
           "compare_paired_policies", "align_policy_trajectories",
           "project_independent_registry_sidecar"]
