"""Leakage-safe offline replay for the adaptive information policy.

This module is evaluator-only.  It replays a public episode one action at a
time and exposes only the selected action's transition to the next actor state.
The sidecar keeps action returns and sequence losses private.  A missing
sequence loss makes the result ``counterfactual_ineligible`` rather than
inventing a baseline, an additive gain, or an interpolation.

The interval policy accepts a small callback that receives the actor state and
returns state-bound intervals.  In production this callback is supplied by
``StateBoundContextValueEstimator``; the replay module itself never reads the
sidecar while choosing an action.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any


VERSION = "counterfactual-policy-replay-v1"
POLICIES = ("fixed_order", "full_query", "interval_voi")
INDEPENDENT_GOLD = "INDEPENDENT_GOLD"
INDEPENDENT_GOLD_RECONSTRUCTED = "INDEPENDENT_GOLD_RECONSTRUCTED"
_EPS = 1e-12
_FORBIDDEN = frozenset({
    "result", "outcome", "score", "scores", "goals", "home_score", "away_score",
    "final_score", "label", "loss", "decision_loss", "realized_gain", "realized_loss",
    "evaluator", "sidecar", "sequence_losses", "action_returns", "post_state", "after_state",
})


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


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if minimum is not None and value < minimum - _EPS:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _public(value: Any, path: str = "public") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            key_text = _text(str(key), f"{path}.key")
            if key_text.casefold() in _FORBIDDEN:
                raise ValueError(f"{path} contains an evaluator-only field")
            result[key_text] = _public(child, f"{path}.{key_text}")
        return result
    if isinstance(value, (list, tuple)):
        return [_public(child, f"{path}[]") for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} contains an unsupported value")


def _menu(public: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = public.get("candidate_menu", public.get("candidates"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("public candidate_menu must be a nonempty sequence")
    result, seen = [], set()
    for index, item in enumerate(raw):
        row = {"candidate_id": item} if isinstance(item, str) else dict(item)
        candidate_id = _text(row.get("candidate_id"), f"candidate_menu[{index}].candidate_id")
        if candidate_id in seen:
            raise ValueError("candidate IDs must be unique")
        seen.add(candidate_id)
        cost = None if row.get("cost") is None else _finite(row.get("cost"), f"{candidate_id}.cost", minimum=0)
        source_id = None if row.get("source_id") is None else _text(row.get("source_id"), f"{candidate_id}.source_id")
        action_id = None if row.get("action_id") is None else _text(row.get("action_id"), f"{candidate_id}.action_id")
        result.append({"candidate_id": candidate_id, "cost": cost, "action_kind": row.get("action_kind", "query"),
                       "source_id": source_id, "action_id": action_id})
    return tuple(result)


def _sequence_key(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = tuple(_text(item, "sequence candidate") for item in value)
    else:
        raw = str(value)
        parts = () if raw == "" else tuple(_text(item, "sequence candidate") for item in raw.replace("|", ">").split(">"))
    if len(set(parts)) != len(parts):
        raise ValueError("sequence contains duplicate candidates")
    return parts


def _sidecar(episode: Mapping[str, Any], *, accepted_label_statuses: Sequence[str] = (INDEPENDENT_GOLD,)) -> Mapping[str, Any]:
    raw = episode.get("evaluation_sidecar", episode.get("evaluator_sidecar"))
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation_sidecar is required")
    statuses = tuple(_text(status, "accepted_label_status") for status in accepted_label_statuses)
    if raw.get("label_status") not in statuses:
        expected = ", ".join(statuses)
        raise ValueError(f"evaluation_sidecar label_status must be one of: {expected}")
    return raw


def _episode_public(episode: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], tuple[dict[str, Any], ...], float, str]:
    episode_id = _text(episode.get("episode_id"), "episode_id")
    public_raw = episode.get("public")
    if not isinstance(public_raw, Mapping):
        raise ValueError("episode.public is required")
    public = _public(public_raw, "public")
    target_key = _text(public.get("target_key"), "public.target_key")
    cutoff = _time(public.get("prediction_cutoff"), "public.prediction_cutoff")
    kickoff = public.get("kickoff")
    if kickoff is not None and cutoff > _time(kickoff, "public.kickoff"):
        raise ValueError("prediction_cutoff must not follow kickoff")
    menu = _menu(public)
    budget = _finite(public.get("budget"), "public.budget", minimum=0)
    loss_unit = _text(public.get("loss_unit", "decision_loss"), "public.loss_unit")
    if not isinstance(public.get("initial_context", {}), Mapping):
        raise ValueError("public.initial_context must be a mapping")
    return episode_id, {**public, "target_key": target_key, "prediction_cutoff": cutoff.isoformat(),
                       "initial_context": _public(public.get("initial_context", {}), "public.initial_context")}, menu, budget, loss_unit


def _transition(sidecar: Mapping[str, Any], candidate_id: str, cutoff: datetime, *, state_digest: str,
                route: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    returns = sidecar.get("action_returns")
    if not isinstance(returns, Mapping) or not isinstance(returns.get(candidate_id), Mapping):
        return None, "missing_action_specific_transition"
    row = returns[candidate_id]
    if row.get("previous_state_digest") is not None and row.get("previous_state_digest") != state_digest:
        return None, "previous_state_digest_mismatch"
    if route.get("source_id") is not None and row.get("source_id") not in (None, route["source_id"]):
        return None, "source_route_mismatch"
    if route.get("action_id") is not None and row.get("action_id") not in (None, route["action_id"]):
        return None, "action_route_mismatch"
    update = row.get("context_update", {})
    if not isinstance(update, Mapping):
        return None, "context_update_missing"
    safe = _public(update, f"action_returns.{candidate_id}.context_update")
    available = row.get("available", row.get("status") == "available")
    if not isinstance(available, bool):
        return None, "availability_unknown"
    observed = row.get("observed_at")
    operational = row.get("operational_available_at", row.get("available_at"))
    if observed is not None and _time(observed, "observed_at") > cutoff:
        return None, "observed_after_cutoff"
    if operational is not None and _time(operational, "operational_available_at") > cutoff:
        return None, "available_after_cutoff"
    if not available:
        return {"context_update": {}, "available": False,
                "next_state_digest": row.get("next_state_digest")}, "action_unavailable"
    return {"context_update": safe, "available": True,
            "next_state_digest": row.get("next_state_digest")}, "available"


def _actor_state(episode_id: str, target_key: str, prediction_cutoff: str, step: int,
                 context: Mapping[str, Any], remaining: Sequence[str], complete: bool) -> dict[str, Any]:
    state = {"schema_version": VERSION, "episode_id": episode_id, "target_key": target_key,
             "prediction_cutoff": prediction_cutoff, "step": step,
             "context": deepcopy(dict(context)), "remaining_candidate_ids": list(remaining),
             "candidate_generation_complete": complete}
    state["state_digest"] = _digest(state)
    return state


@dataclass(frozen=True)
class ReplayResult:
    episode_id: str
    policy: str
    runtime_eligible: bool
    actor_trace: Mapping[str, Any]
    evaluator: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": VERSION, "episode_id": self.episode_id, "policy": self.policy,
                "runtime_eligible": self.runtime_eligible, "actor_trace": deepcopy(dict(self.actor_trace)),
                "evaluator": deepcopy(dict(self.evaluator))}


def _interval_choice(state: Mapping[str, Any], menu: Mapping[str, Mapping[str, Any]], remaining: Sequence[str],
                     provider: Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]], *, complete: bool) -> tuple[str | None, str]:
    raw = provider(deepcopy(dict(state)))
    if not isinstance(raw, Mapping):
        raise ValueError("interval provider must return a mapping")
    positive, unresolved, all_safe = [], False, True
    for cid in remaining:
        row = raw.get(cid)
        if not isinstance(row, Mapping):
            unresolved = True; all_safe = False; continue
        if row.get("state_digest") not in (None, state["state_digest"]):
            raise ValueError("interval provider returned a stale state digest")
        lower = row.get("lower_bound")
        upper = row.get("upper_bound")
        cost = menu[cid].get("cost")
        if cost is None or lower is None or upper is None:
            unresolved = True; all_safe = False; continue
        lower, upper = _finite(lower, f"{cid}.lower_bound"), _finite(upper, f"{cid}.upper_bound")
        if lower > upper:
            raise ValueError("interval lower bound exceeds upper bound")
        if lower > cost:
            positive.append((lower, cid)); all_safe = False
        elif upper > cost:
            unresolved = True; all_safe = False
    if positive:
        return max(positive, key=lambda item: (item[0], item[1]))[1], "lower_bound_above_cost"
    if complete and not unresolved and all_safe:
        return None, "all_upper_bounds_at_or_below_cost"
    return None, "interval_crosses_cost_or_support_unknown"


def replay_counterfactual_episode(
    episode: Mapping[str, Any], policy: str, *,
    interval_provider: Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]] | None = None,
    accepted_label_statuses: Sequence[str] = (INDEPENDENT_GOLD,),
) -> dict[str, Any]:
    """Replay one episode while keeping evaluator data out of actor_trace.

    ``accepted_label_statuses`` is evaluator-only.  The default remains strict
    ``INDEPENDENT_GOLD``.  Offline fixture adapters may explicitly add a
    synthetic label, but their report must keep structural results separate
    from independent-gold performance claims.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    episode_id, public, menu_rows, budget, loss_unit = _episode_public(episode)
    sidecar = _sidecar(episode, accepted_label_statuses=accepted_label_statuses)
    cutoff = _time(public["prediction_cutoff"], "prediction_cutoff")
    menu = {row["candidate_id"]: row for row in menu_rows}
    remaining = [row["candidate_id"] for row in menu_rows]
    context = dict(public["initial_context"])
    spent = 0.0
    selected: list[str] = []
    state_digests: list[str] = []
    transition_statuses: list[str] = []
    stop_reason = "policy_exhausted"
    complete = public.get("candidate_generation_complete") is True
    while remaining:
        state = _actor_state(episode_id, public["target_key"], public["prediction_cutoff"], len(selected),
                             context, remaining, complete)
        state_digests.append(state["state_digest"])
        if policy == "interval_voi":
            if interval_provider is None:
                raise ValueError("interval_voi requires interval_provider")
            candidate_id, reason = _interval_choice(state, menu, remaining, interval_provider, complete=complete)
            if candidate_id is None:
                stop_reason = reason
                break
        elif policy == "fixed_order":
            candidate_id = remaining[0]
            reason = "declared_menu_order"
        else:
            affordable = [cid for cid in sorted(remaining) if menu[cid].get("cost") is not None and
                           spent + float(menu[cid]["cost"]) <= budget + _EPS]
            if not affordable:
                stop_reason = "budget_or_cost_exhausted"
                break
            candidate_id, reason = affordable[0], "all_affordable_canonical_order"
        cost = menu[candidate_id].get("cost")
        if cost is None or spent + float(cost) > budget + _EPS:
            stop_reason = "budget_or_cost_exhausted"
            break
        spent += float(cost)
        selected.append(candidate_id)
        remaining.remove(candidate_id)
        transition, transition_status = _transition(sidecar, candidate_id, cutoff, state_digest=state["state_digest"],
                                                    route=menu[candidate_id])
        transition_statuses.append(transition_status)
        if transition is None:
            stop_reason = transition_status
            break
        if transition["available"]:
            updates = transition["context_update"]
            context.update(updates)
        expected_next_digest = transition.get("next_state_digest")
        if expected_next_digest is not None:
            next_state = _actor_state(episode_id, public["target_key"], public["prediction_cutoff"], len(selected),
                                      context, remaining, complete)
            if expected_next_digest != next_state["state_digest"]:
                stop_reason = "next_state_digest_mismatch"
                transition_statuses[-1] = stop_reason
                break
        if policy == "fixed_order" and not remaining:
            stop_reason = reason
    sequence = tuple(selected)
    raw_losses = sidecar.get("sequence_losses")
    sequence_losses = {}
    if isinstance(raw_losses, Mapping):
        sequence_losses = {_sequence_key(key): _finite(value, "sequence_loss", minimum=0)
                           for key, value in raw_losses.items()}
    final_loss = sequence_losses.get(sequence)
    baseline = sidecar.get("baseline_loss")
    baseline_value = None if baseline is None else _finite(baseline, "baseline_loss", minimum=0)
    rate = sidecar.get("cost_to_loss_rate")
    rate_value = None if rate is None else _finite(rate, "cost_to_loss_rate", minimum=0)
    evaluator_eligible = final_loss is not None and all(status in {"available", "action_unavailable"} for status in transition_statuses)
    reduction = None if not evaluator_eligible or baseline_value is None else baseline_value - final_loss
    net = None if reduction is None or rate_value is None else reduction - spent * rate_value
    actor_trace = {
        "selected_actions": list(selected), "state_digests": state_digests,
        "transition_statuses": transition_statuses, "stop_reason": stop_reason,
        "query_count": len(selected), "spent_cost": spent, "loss_unit": loss_unit,
        "final_context": deepcopy(context), "hidden_result_included": False,
    }
    evaluator = {
        "counterfactual_eligible": evaluator_eligible,
        "counterfactual_ineligible_reason": None if evaluator_eligible else (
            "missing_sequence_loss_or_action_transition"),
        "sequence": list(sequence), "final_loss": final_loss if evaluator_eligible else None,
        "baseline_loss": baseline_value if evaluator_eligible else None,
        "decision_loss_reduction": reduction, "net_utility": net,
        "unknown_metrics_are_not_zero": True,
    }
    return ReplayResult(episode_id, policy, evaluator_eligible, actor_trace, evaluator).as_dict()


def compare_counterfactual_policies(
    episodes: Iterable[Mapping[str, Any]], *,
    interval_provider: Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]] | None = None,
    policies: Sequence[str] = POLICIES,
    accepted_label_statuses: Sequence[str] = (INDEPENDENT_GOLD,),
) -> dict[str, Any]:
    raw = list(episodes)
    if not raw:
        raise ValueError("at least one episode is required")
    if any(policy not in POLICIES for policy in policies):
        raise ValueError("unknown policy in comparison")
    results = {policy: [replay_counterfactual_episode(
                    ep, policy, interval_provider=interval_provider,
                    accepted_label_statuses=accepted_label_statuses,
                ) for ep in raw]
               for policy in policies}
    paired: dict[str, list[dict[str, Any]]] = {}
    for policy in policies:
        if policy == policies[0]:
            continue
        baseline_by_episode = {row["episode_id"]: row for row in results[policies[0]]}
        paired[policy] = []
        for row in results[policy]:
            base = baseline_by_episode[row["episode_id"]]
            paired[policy].append({
                "episode_id": row["episode_id"],
                "counterfactual_eligible": bool(row["evaluator"]["counterfactual_eligible"] and base["evaluator"]["counterfactual_eligible"]),
                "net_utility_delta": None if not (row["evaluator"]["counterfactual_eligible"] and base["evaluator"]["counterfactual_eligible"]) else row["evaluator"]["net_utility"] - base["evaluator"]["net_utility"],
                "final_loss_delta": None if not (row["evaluator"]["counterfactual_eligible"] and base["evaluator"]["counterfactual_eligible"]) else row["evaluator"]["final_loss"] - base["evaluator"]["final_loss"],
            })
    return {"schema_version": VERSION, "policy_order": list(policies), "episode_count": len(raw),
            "results": results, "paired_deltas": paired,
            "contract": {"actor_trace_excludes_evaluator_fields": True,
                         "selected_transition_only": True, "unknown_metrics_are_not_zero": True,
                         "paired_by_episode": True}}


def temporal_episode_split(
    episodes: Iterable[Mapping[str, Any]], *, calibration_as_of: str,
    evaluation_target_keys: Iterable[str] = (), evaluation_cutoff: str | None = None,
) -> dict[str, Any]:
    """Create a cutoff-safe train/evaluation split without reading labels."""
    as_of = _time(calibration_as_of, "calibration_as_of")
    eval_cutoff = None if evaluation_cutoff is None else _time(evaluation_cutoff, "evaluation_cutoff")
    target_exclusions = {_text(value, "evaluation_target_key") for value in evaluation_target_keys}
    train, evaluation, excluded = [], [], []
    seen: set[str] = set()
    for episode in episodes:
        episode_id, public, _, _, _ = _episode_public(episode)
        if episode_id in seen:
            raise ValueError("duplicate episode_id")
        seen.add(episode_id)
        cutoff = _time(public["prediction_cutoff"], "prediction_cutoff")
        target = public["target_key"]
        if cutoff <= as_of and target not in target_exclusions:
            train.append(episode_id)
        elif cutoff <= as_of and target in target_exclusions:
            excluded.append({"episode_id": episode_id, "reason": "evaluation_target_exclusion"})
        elif eval_cutoff is None or cutoff <= eval_cutoff:
            evaluation.append(episode_id)
        else:
            excluded.append({"episode_id": episode_id, "reason": "outside_evaluation_window"})
    return {
        "schema_version": VERSION, "calibration_as_of": as_of.isoformat(),
        "evaluation_cutoff": None if eval_cutoff is None else eval_cutoff.isoformat(),
        "train_episode_ids": train, "evaluation_episode_ids": evaluation,
        "excluded": excluded, "same_target_excluded": sorted(target_exclusions),
        "labels_read": False,
    }


__all__ = ["VERSION", "POLICIES", "ReplayResult", "replay_counterfactual_episode",
           "compare_counterfactual_policies", "temporal_episode_split"]
