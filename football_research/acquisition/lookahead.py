"""Small finite two-step acquisition lookahead.

The first decision uses only a mask-transition projection.  It never reads the
future returned values or target label.  After a real reveal, the normal policy
re-estimates the remaining action from the new visible state.
"""
from __future__ import annotations

from copy import deepcopy
import math
from .policy import select_or_stop
from .state import GROUP_FIELDS, sanitize_state


def mask_transition_state(state: dict, action: dict) -> dict:
    """Build a value-only continuation state; revealed values stay unknown."""
    result = deepcopy(state)
    history = list(result.get("action_history", []))
    history.append({"action_id": action["action_id"], "status": "projected"})
    result["action_history"] = history
    for group in action["group_ids"]:
        item = result.setdefault("groups", {}).setdefault(group, {"status": "not_acquired", "values": {}})
        item["status"] = "observed"
        item["values"] = {field: None for field in GROUP_FIELDS[group]}
        item.setdefault("available_at", result["prediction_cutoff"])
        item.setdefault("version", "projected-mask-transition")
        item.setdefault("evidence_refs", [])
    clean = sanitize_state(result)
    clean["action_history"] = history
    return clean


def select_two_step(state: dict, candidates: list[dict], *, predictor, estimator,
                    budget: dict, lambda_cost: dict, risk_level: str = "mean",
                    min_support: int = 12) -> dict:
    """Choose a first action using at most one projected continuation."""
    current_prediction = predictor(state)
    first = estimator.estimate_value(state, candidates, prediction=current_prediction, risk_level=risk_level)
    scored = []
    for estimate in first:
        action = estimate["action"]
        first_policy = select_or_stop(state, [estimate], budget, lambda_cost,
                                      conservative=risk_level == "conservative", min_support=min_support)
        if first_policy["decision"] != "action":
            first_net = first_policy.get("net_gain")
            scored.append({"action_id": action["action_id"], "first": estimate,
                           "continuation": None, "net_gain": first_net if first_net is not None else -1e30,
                           "reason": first_policy["reason"]})
            continue
        projected = mask_transition_state(state, action)
        projected_prediction = predictor(projected)
        remaining = [candidate for candidate in candidates if candidate["action_id"] != action["action_id"]]
        spent = {key: float(action["cost"].get(key, 0.0)) for key in ("money", "requests", "latency_ms", "tokens")}
        remaining_budget = {key: max(0.0, float(budget.get(key, 0.0)) - spent[key]) for key in budget}
        continuation_estimates = estimator.estimate_value(projected, remaining,
                                                           prediction=projected_prediction, risk_level=risk_level)
        continuation = select_or_stop(projected, continuation_estimates, remaining_budget, lambda_cost,
                                       conservative=risk_level == "conservative", min_support=min_support)
        continuation_net = continuation.get("net_gain", 0.0) if continuation.get("decision") == "action" else 0.0
        first_net = first_policy.get("net_gain", -float("inf"))
        scored.append({"action_id": action["action_id"], "first": estimate,
                       "continuation": continuation, "net_gain": first_net + continuation_net,
                       "reason": "two_step_projected_mask"})
    scored.sort(key=lambda item: (-item["net_gain"], item["action_id"]))
    if not scored or not math.isfinite(scored[0]["net_gain"]) or scored[0]["net_gain"] <= 0:
        return {"decision": "stop", "action_id": None,
                "reason": "nonpositive_two_step_net_gain", "two_step": scored,
                "policy_version": "two-step-mask-lookahead-v1"}
    best = scored[0]
    return {"decision": "action", "action_id": best["action_id"],
            "action": best["first"]["action"], "reason": "positive_two_step_net_gain",
            "net_gain": best["net_gain"], "two_step": scored,
            "policy_version": "two-step-mask-lookahead-v1"}
