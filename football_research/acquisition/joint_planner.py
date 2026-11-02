"""Exact one/two-step Bayes-risk planning under a supplied joint return model.

Physical resource budgets are never multiplied by utility weights. Hypothetical
responses travel only through explicitly named planning APIs, not through the
model's real-observation prediction entry point.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from collections.abc import Mapping


VERSION = "finite-joint-bellman-v1"
UNITS = ("money", "requests", "latency_ms", "tokens")
TOLERANCE = 1e-10
VALUE_TOLERANCE = 1e-12


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _probabilities(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name} must contain three probabilities")
    probabilities = tuple(_number(v, name) for v in values)
    if any(p > 1 for p in probabilities) or abs(math.fsum(probabilities) - 1) > TOLERANCE:
        raise ValueError(f"{name} must be a normalized probability distribution")
    return probabilities


def entropy(probabilities):
    return -math.fsum(p * math.log(p) for p in probabilities if p > 0)


def _cost(raw):
    if raw.get("cost_status") not in ("known", "estimated", "simulated"):
        raise ValueError("Exact planning requires declared known, estimated, or simulated costs")
    vector = raw.get("cost")
    if not isinstance(vector, Mapping) or set(vector) - set(UNITS) or not {"money", "requests"} <= set(vector):
        raise ValueError("Action cost must explicitly include physical money and requests")
    return {unit: _number(vector.get(unit, 0), "Action " + unit) for unit in UNITS}


def _budget(value):
    if not isinstance(value, Mapping) or set(value) != {"limits", "spent"}:
        raise ValueError("Budget requires distinct limits and spent mappings")
    limits, spent = value["limits"], value["spent"]
    if not isinstance(limits, Mapping) or not isinstance(spent, Mapping):
        raise ValueError("Budget limits and spent must be mappings")
    if not {"money", "requests"} <= set(limits) or set(limits) - set(UNITS) or set(spent) - set(UNITS):
        raise ValueError("Physical money and request limits are required")
    checked_limits = {u: _number(v, "Budget limit " + u) for u, v in limits.items()}
    checked_spent = {u: _number(spent.get(u, 0), "Budget spent " + u) for u in UNITS}
    if any(checked_spent[u] > limit + VALUE_TOLERANCE for u, limit in checked_limits.items()):
        raise ValueError("Spent resources exceed the physical budget")
    return checked_limits, checked_spent


class JointRiskPlanner:
    """Model-conditional exact planning, without model-parameter confidence claims.

    `model` implements predict_partial(real), predict_hypothetical(branch),
    observed_actions(state, hypothetical=...), and enumerate_action(state,
    action, hypothetical=...). Every recursive branch can stop. horizon is
    explicitly limited to at most two future acquisitions.
    """

    def __init__(self, model, *, horizon=1, lambda_cost=None):
        if type(horizon) is not int or horizon not in (1, 2):
            raise ValueError("Planning horizon must be one or two")
        weights = {} if lambda_cost is None else lambda_cost
        if not isinstance(weights, Mapping) or set(weights) - set(UNITS):
            raise ValueError("Unknown loss-conversion unit")
        self.model, self.horizon = model, horizon
        self.lambda_cost = {u: _number(weights.get(u, 0), "Lambda " + u) for u in UNITS}
        config = {"implementation": VERSION, "model_version": model.model_version, "horizon": horizon, "lambda_cost": self.lambda_cost}
        self.version = VERSION + ":" + sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]

    @property
    def manifest(self):
        return {"trained_as_of": self.model.metadata["training_as_of"], "horizon": self.horizon,
                "risk": "expected_predictive_log_loss", "risk_mode": "mean_only",
                "uncertainty": "joint_parameter_uncertainty_not_estimated"}

    def _prediction(self, state, hypothetical):
        result = self.model.predict_hypothetical(state) if hypothetical else self.model.predict_partial(state)
        return _probabilities(result["probabilities"], "Current prediction")

    def plan(self, state, actions, budget):
        if not isinstance(state, Mapping):
            raise ValueError("Public state must be a mapping")
        if "joint_return_hypothesis" in state:
            raise ValueError("The real planning entry point rejects hypothetical response tokens")
        limits, spent = _budget(budget)
        if not isinstance(actions, (list, tuple)):
            raise ValueError("Actions must be a sequence")
        checked, seen = [], set()
        for raw in actions:
            if not isinstance(raw, Mapping):
                raise ValueError("Action entries must be mappings")
            aid = raw.get("action_id")
            if not isinstance(aid, str) or not aid or aid in seen:
                raise ValueError("Actions must have unique nonempty IDs")
            seen.add(aid)
            row = deepcopy(dict(raw))
            row["cost"] = _cost(row)
            if type(row.get("available", True)) is not bool:
                raise ValueError("Action availability must be Boolean")
            if row.get("evidence_id") is not None and (not isinstance(row["evidence_id"], str) or not row["evidence_id"]):
                raise ValueError("Evidence identity must be an inert nonempty string")
            checked.append(row)
        checked.sort(key=lambda row: row["action_id"])
        observed = set(self.model.observed_actions(state, hypothetical=False))
        for receipt in state.get("action_history", []):
            if isinstance(receipt, Mapping) and isinstance(receipt.get("action_id"), str):
                observed.add(receipt["action_id"])  # One-shot also after failed attempts.
        return self._solve(state, checked, limits, spent, self.horizon, observed, hypothetical=False)

    def _solve(self, state, actions, limits, spent, horizon, used, *, hypothetical):
        probabilities = self._prediction(state, hypothetical)
        risk = entropy(probabilities)
        empty_cost = {u: 0.0 for u in UNITS}
        best = {"action_id": None, "expected_terminal_risk": risk, "expected_plan_cost_vector": empty_cost,
                "expected_plan_penalty": 0.0, "expected_loss_improvement": 0.0,
                "planning_net_gain": 0.0, "selection_score": 0.0, "expected_total_risk": risk}
        observed = set(self.model.observed_actions(state, hypothetical=hypothetical)) | set(used)
        used_evidence = {a["evidence_id"] for a in actions if a["action_id"] in observed and a.get("evidence_id") is not None}
        candidates = []
        if horizon:
            for action in actions:
                aid = action["action_id"]
                reason = ("action_unavailable" if not action.get("available", True) else
                          "already_attempted" if aid in observed else
                          "evidence_already_observed" if action.get("evidence_id") in used_evidence else
                          "physical_budget_exhausted" if any(spent[u] + action["cost"][u] > cap + VALUE_TOLERANCE
                                                             for u, cap in limits.items()) else None)
                if reason:
                    candidates.append({"action_id": aid, "feasible": False, "reason": reason})
                    continue
                enumeration = self.model.enumerate_action(state, action, hypothetical=hypothetical)
                if enumeration.get("action_id") != aid:
                    raise ValueError("Return enumeration changed action identity")
                before = _probabilities(enumeration["current_probabilities"], "Enumeration current prediction")
                if max(abs(a - b) for a, b in zip(before, probabilities)) > TOLERANCE:
                    raise ValueError("Return enumeration uses a different current prediction")
                branches = enumeration.get("branches")
                if not isinstance(branches, list) or not branches:
                    raise ValueError("Exact integration requires a nonempty return distribution")
                mass, posterior_mean = 0.0, [0.0] * 3
                expected_immediate_risk = expected_terminal_risk = 0.0
                expected_cost = dict(action["cost"])
                integrated, observations = [], set()
                for branch in branches:
                    probability = _number(branch["probability"], "Return probability")
                    if probability <= 0 or probability > 1:
                        raise ValueError("Enumerated return branches must have positive probabilities")
                    observation = branch.get("observation")
                    if not isinstance(observation, str) or not observation or observation in observations:
                        raise ValueError("Return categories must be unique nonempty strings")
                    observations.add(observation)
                    status = branch.get("return_status")
                    if status not in ("observed", "summary_observed", "summary_missing"):
                        raise ValueError("Provider failures are not trained summary-missing observations")
                    posterior = _probabilities(branch["posterior_probabilities"], "Return posterior")
                    next_state = branch["next_state"]
                    predicted = self._prediction(next_state, hypothetical=True)
                    if max(abs(a - b) for a, b in zip(predicted, posterior)) > TOLERANCE:
                        raise ValueError("Hypothetical-state prediction disagrees with its return posterior")
                    next_spent = {u: spent[u] + action["cost"][u] for u in UNITS}
                    continuation = self._solve(next_state, actions, limits, next_spent, horizon - 1,
                                               observed | {aid}, hypothetical=True)
                    mass += probability
                    for y in range(3):
                        posterior_mean[y] += probability * posterior[y]
                    expected_immediate_risk += probability * entropy(posterior)
                    expected_terminal_risk += probability * continuation["expected_terminal_risk"]
                    for u in UNITS:
                        expected_cost[u] += probability * continuation["expected_plan_cost_vector"][u]
                    integrated.append({"observation": observation, "probability": probability, "return_status": status,
                                       "posterior_probabilities": list(posterior), "hypothetical": True,
                                       "continuation_action": continuation["action_id"],
                                       "continuation_expected_terminal_risk": continuation["expected_terminal_risk"],
                                       "continuation_expected_cost_vector": continuation["expected_plan_cost_vector"],
                                       "continuation_expected_total_risk": continuation["expected_total_risk"]})
                if abs(mass - 1) > TOLERANCE:
                    raise ValueError("Return probabilities must sum to one")
                if max(abs(a - b) for a, b in zip(posterior_mean, probabilities)) > TOLERANCE:
                    raise ValueError("Posterior martingale check failed for the joint model")
                immediate_evsi = risk - expected_immediate_risk
                if immediate_evsi < -TOLERANCE:
                    raise ValueError("Negative exact EVSI contradicts the joint Bayes model")
                penalty = math.fsum(self.lambda_cost[u] * expected_cost[u] for u in UNITS)
                improvement = risk - expected_terminal_risk
                net = improvement - penalty
                # Mathematically tied acquisition orders can differ at ~1e-16
                # by summation order. The executor and runner share this score;
                # retain every unrounded risk/value for numerical verification.
                selection_score = round(net, 12)
                result = {"action_id": aid, "feasible": True, "reason": "model_conditional_expected_value",
                          "immediate_evsi": max(0.0, immediate_evsi), "expected_risk_after_observation": expected_immediate_risk,
                          "immediate_cost_vector": dict(action["cost"]),
                          "immediate_penalty": math.fsum(self.lambda_cost[u] * action["cost"][u] for u in UNITS),
                          "expected_terminal_risk": expected_terminal_risk, "expected_plan_cost_vector": expected_cost,
                          "expected_plan_penalty": penalty, "expected_loss_improvement": improvement,
                          "planning_net_gain": net, "selection_score": selection_score,
                          "expected_total_risk": expected_terminal_risk + penalty,
                          "posterior_martingale_max_error": max(abs(a - b) for a, b in zip(posterior_mean, probabilities)),
                          "branches": integrated}
                candidates.append(result)
                if selection_score > best["selection_score"]:
                    best = {k: deepcopy(result[k]) for k in best}
        feasible = [c for c in candidates if c["feasible"]]
        reason = ("positive_expected_net_value" if best["action_id"] is not None else
                  "horizon_exhausted" if horizon == 0 else
                  "no_feasible_actions" if not feasible else "no_positive_expected_net_value")
        return {"schema_version": VERSION, "planner_version": self.version, "model_version": self.model.model_version,
                "decision": "acquire" if best["action_id"] is not None else "stop", "reason": reason,
                "horizon": horizon, "prediction": {"probabilities": list(probabilities)}, "current_risk": risk,
                "physical_budget": {"limits": dict(limits), "spent": dict(spent)},
                "lambda_cost": dict(self.lambda_cost), "risk_mode": "mean_only",
                "selection_score_precision_decimal_places": 12,
                "uncertainty_kind": "joint_parameter_uncertainty_not_estimated",
                "optimality_scope": "specified fitted joint model and at most two remaining acquisitions",
                "candidates": candidates, **best}
