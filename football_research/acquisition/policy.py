"""One-step empirical net-gain selection under explicit unit conversions."""
from __future__ import annotations

from collections import Counter
import math

from .value import COST_UNITS, _action, _mask


def _amount(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def select_or_stop(state: dict, candidates: list[dict], budget: dict,
                   lambda_cost: dict, conservative: bool = False, min_support: int = 12) -> dict:
    """Choose highest supported gain minus lambda dot cost; never inspect labels.

    budget is *remaining* quantities, with omitted dimensions unconstrained.
    lambda is nats per corresponding unit; omitted dimensions have weight zero.
    Runtime costs themselves must explicitly supply all four component values.
    """
    _, visible = _mask(state)
    if set(lambda_cost) - set(COST_UNITS) or set(budget) - set(COST_UNITS) - {"deadline_ms_remaining"}:
        raise ValueError("Unknown budget/lambda component")
    weights = {unit: _amount(lambda_cost.get(unit, 0), f"lambda.{unit}") for unit in COST_UNITS}
    remaining = {unit: _amount(value, f"budget.{unit}") for unit, value in budget.items()}
    rejected, eligible = [], []
    for estimate in candidates:
        action = _action(estimate.get("action", estimate))
        aid, groups = action["action_id"], action["group_ids"]
        reason = None
        statuses = [visible.get("groups", {}).get(g, {}).get("status", "not_acquired") for g in groups]
        if not action["available"]:
            reason = "no_capability"
        elif all(s == "observed" for s in statuses):
            reason = "already_observed"
        elif all(s in {"observed", "source_missing"} for s in statuses) and not action["retry_allowed"]:
            reason = "source_missing_no_retry"
        elif all(s in {"observed", "stale"} for s in statuses) and not action["refresh_allowed"]:
            reason = "stale_no_refresh"
        # sanitizer intentionally omits history; inspect action IDs only here.
        history = state.get("action_history", [])
        attempts = sum(h.get("action_id") == aid for h in history if isinstance(h, dict))
        max_attempts = action.get("max_attempts", 1)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be positive integer")
        if not reason and attempts >= max_attempts:
            reason = "attempt_limit"
        costs = action["cost"]
        if not reason and (action["cost_status"] not in {"known", "estimated", "simulated"} or set(costs) != set(COST_UNITS) or any(v is None for v in costs.values())):
            reason = "unknown_cost"
        if not reason:
            costs = {unit: _amount(costs[unit], f"cost.{unit}") for unit in COST_UNITS}
            if "deadline_ms_remaining" in remaining and costs["latency_ms"] > remaining["deadline_ms_remaining"]:
                reason = "deadline"
            elif any(costs[unit] > remaining[unit] + 1e-12 for unit in COST_UNITS if unit in remaining):
                reason = "budget"
        if not reason and (estimate.get("support_matches", 0) < min_support or estimate.get("reason") in {"unsupported_action", "unsupported_state_action"}):
            reason = "insufficient_support"
        gain = estimate.get("lower_gain" if conservative else "gain")
        if not reason and (gain is None or not math.isfinite(gain)):
            reason = "uncertainty_unavailable" if conservative else "insufficient_support"
        if reason:
            # Round-five estimators expose a more specific, risk-aware gate
            # reason (for example ``stop_uncertainty_unavailable``).  Preserve
            # the stable policy reason for existing callers while carrying the
            # estimator explanation through the audit trail.
            rejected.append({"action_id": aid, "reason": reason,
                             "gate_reason": estimate.get("gate_reason")})
            continue
        penalty = sum(weights[unit] * costs[unit] for unit in COST_UNITS)
        net = gain - penalty
        eligible.append({"action_id": aid, "action": action, "gain": estimate["gain"],
                         "decision_gain": gain, "cost": costs, "cost_penalty": penalty,
                         "net_gain": net, "support_matches": estimate["support_matches"],
                         "estimator_version": estimate.get("version"),
                         "gate_reason": estimate.get("gate_reason")})
    eligible.sort(key=lambda row: (-row["net_gain"], row["action_id"]))
    result = {"decision": "stop", "action_id": None, "reason": "no_capability",
              "conservative": conservative, "candidates": eligible, "rejected": rejected,
              "gate_reason_counts": dict(Counter(item.get("gate_reason") for item in [*eligible, *rejected]
                                                  if item.get("gate_reason"))),
              "lambda_cost": weights, "lambda_units": {unit: f"log_loss_nats_per_{name}" for unit, name in COST_UNITS.items()},
              "policy_version": "one-step-empirical-net-gain-v1",
              "limitation": "One-step stopping can miss complementary information groups."}
    if eligible:
        best = eligible[0]
        result.update({"reason": "positive_net_gain" if best["net_gain"] > 0 else "nonpositive_net_gain",
                       "net_gain": best["net_gain"], "selected_estimate": best})
        if best["net_gain"] > 0:
            result.update({"decision": "action", "action_id": best["action_id"], "action": best["action"]})
    elif rejected:
        counts = Counter(r["reason"] for r in rejected)
        result["reason_counts"] = dict(counts)
        # Closest-to-executable rejection first; expose the complete breakdown.
        for reason in ("insufficient_support", "uncertainty_unavailable", "unknown_cost", "deadline", "budget", "attempt_limit", "source_missing_no_retry", "stale_no_refresh", "already_observed", "no_capability"):
            if reason in counts:
                result["reason"] = reason
                break
    return result
