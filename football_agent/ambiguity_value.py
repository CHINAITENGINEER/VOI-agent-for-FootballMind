"""One-query decision value for a finite, explicitly declared model set.

All models evaluate the same executable terminal response policy.  Models are
fixed over an episode; they are not selected afresh in each possible branch.
The reported envelopes are sensitivity ranges, not statistical confidence
intervals.  The one-query value requires execution of its frozen terminal
policy: arbitrary conditional replanning does not preserve that value.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
import math
from types import MappingProxyType

from .decision_loss import FiniteDecisionLossModel
from .decision_value import VALUE_TOLERANCE, _names, _number


VERSION = "finite-ambiguity-terminal-value-v1"
EVIDENCE_CLASS = "DECLARED_MODEL_SET_NOT_STATISTICAL_CONFIDENCE"
POLICY_CLASS = "shared_deterministic_one_query_terminal_response_policy"
LIMITS = MappingProxyType({
    "models": 16, "actions": 32, "terminal_decisions": 32,
    "target_labels": 64, "total_latent_states": 4096,
    "returns_per_action": 8, "policies_per_action": 100_000,
    "total_enumeration_work": 2_000_000, "total_precomputation_work": 2_000_000,
})
BOUNDS_SCOPE = MappingProxyType({
    "models": "declared_finite_set_conditioned_on_actual_history",
    "model_semantics": "one_fixed_model_per_episode_not_branchwise_adversaries",
    "policies": POLICY_CLASS, "queries": "registered_feasible_actions_only",
    "horizon": 1, "statistical_coverage": None,
    "global_open_world_stop_proven": False,
})
EXECUTION_CONTRACT = (
    "Freeze response_policy before acquiring one observation and execute its "
    "terminal decision for a return supported by at least one active model. "
    "A return impossible in all active models invalidates the policy certificate; "
    "its zero-probability placeholder is not an authorized conclusion. "
    "Conditional replanning or further "
    "queries are not covered by this one-query expected-risk value."
)


def _spread(values):
    return {"minimum": min(values), "maximum": max(values),
            "interpretation": "finite_declared_model_envelope_not_confidence_interval"}


def _canonical_copy(model):
    # Own the numerical inputs and use the same latent summation order after a
    # sorted JSON round trip.  Preserve the explicitly declared decision order.
    raw = deepcopy(model.manifest())
    raw["hypotheses"] = {key: raw["hypotheses"][key] for key in sorted(raw["hypotheses"])}
    raw["prior"] = {key: raw["prior"][key] for key in raw["hypotheses"]}
    raw["tools"] = {key: raw["tools"][key] for key in sorted(raw["tools"])}
    return FiniteDecisionLossModel.from_manifest(raw)


class FiniteAmbiguityDecisionModel:
    """Exact minimax enumeration within a bounded, deterministic policy class.

    ``remaining_budget`` is a current loss-unit budget supplied by the caller;
    historical acquisition charges are not charged again here.  Every history
    observation is validated before model incompatibility is considered.
    """

    VERSION = VERSION

    def __init__(self, models: Mapping[str, FiniteDecisionLossModel]):
        if not isinstance(models, Mapping):
            raise ValueError("Named finite terminal decision models required")
        self.model_names = _names(tuple(models), "Model IDs")
        if len(models) > LIMITS["models"]:
            raise ValueError("Model count exceeds exact enumeration limit")
        if any(not isinstance(model, FiniteDecisionLossModel) for model in models.values()):
            raise ValueError("Each member must be a FiniteDecisionLossModel")
        self._check_sizes(models)
        copied = {name: _canonical_copy(models[name]) for name in self.model_names}
        self.models = MappingProxyType(copied)
        self.reference_model = copied[self.model_names[0]]
        reference = self.reference_model
        self.outcome_labels = reference.outcome_labels
        self.decision_names = reference.decision_names
        self.decision_losses = reference.decision_losses
        self.cost_unit = reference.cost_unit
        self.tools = reference.tools
        self.action_ids = tuple(sorted(reference.tools))
        for name, model in copied.items():
            if (model.outcome_labels != self.outcome_labels
                    or model.decision_names != self.decision_names
                    or dict(model.decision_losses) != dict(self.decision_losses)
                    or model.cost_unit != self.cost_unit):
                raise ValueError(f"Model {name} has a different target, decision, loss or cost-unit contract")
            if set(model.tools) != set(self.tools):
                raise ValueError(f"Model {name} has a different registered action contract")
            for action, tool in model.tools.items():
                other = self.tools[action]
                if (tool.outcomes != other.outcomes or tool.cost != other.cost
                        or tool.evidence_id != other.evidence_id):
                    raise ValueError(f"Model {name}/{action} has a different return, cost or evidence-ID contract")

    @staticmethod
    def _check_sizes(models):
        total_latent = sum(len(model.hypothesis_names) for model in models.values())
        if total_latent > LIMITS["total_latent_states"]:
            raise ValueError("Latent state count exceeds exact enumeration limit")
        enumeration_work = 0
        precomputation_work = 0
        for model in models.values():
            d, y, k = len(model.decision_names), len(model.outcome_labels), len(model.hypothesis_names)
            if (d > LIMITS["terminal_decisions"] or y > LIMITS["target_labels"]
                    or len(model.tools) > LIMITS["actions"]):
                raise ValueError("Action, label or terminal decision count exceeds exact enumeration limit")
            precomputation_work += k*d*y
            for tool in model.tools.values():
                z = len(tool.outcomes)
                if z > LIMITS["returns_per_action"]:
                    raise ValueError("Return count exceeds exact enumeration limit")
                policies = d**z
                if policies > LIMITS["policies_per_action"]:
                    raise ValueError("Response policy count exceeds exact enumeration limit")
                enumeration_work += policies*z
                precomputation_work += k*d*z
        if enumeration_work > LIMITS["total_enumeration_work"]:
            raise ValueError("Total policy enumeration work exceeds exact enumeration limit")
        if precomputation_work > LIMITS["total_precomputation_work"]:
            raise ValueError("Total precomputation work exceeds exact enumeration limit")

    def manifest(self):
        return {"schema_version": VERSION, "evidence_class": EVIDENCE_CLASS,
                "model_order": list(self.model_names),
                "models": {name: model.manifest() for name, model in self.models.items()},
                "objective": "minimize_worst_expected_terminal_loss_plus_cost",
                "policy_class": POLICY_CLASS, "horizon": 1,
                "bounds_scope": dict(BOUNDS_SCOPE), "limits": dict(LIMITS),
                "execution_contract": EXECUTION_CONTRACT,
                "cost_interpretation": "declared_terminal_loss_penalty_not_measured_money",
                "model_weights": None, "statistical_confidence_level": None,
                "limitations": [
                    "Model membership and probabilities are declared assumptions, not fitted calibration.",
                    "Worst-risk improvement need not improve each model relative to the same stopping decision.",
                    "Independent per-model EVSI envelopes omit model-identification value.",
                    "One-query stopping excludes complementary sequences and unregistered open queries.",
                ]}

    @classmethod
    def from_manifest(cls, manifest):
        if (not isinstance(manifest, Mapping) or manifest.get("schema_version") != VERSION
                or manifest.get("evidence_class") != EVIDENCE_CLASS
                or manifest.get("policy_class") != POLICY_CLASS
                or type(manifest.get("horizon")) is not int or manifest["horizon"] != 1
                or manifest.get("bounds_scope") != dict(BOUNDS_SCOPE)
                or manifest.get("limits") != dict(LIMITS)
                or manifest.get("execution_contract") != EXECUTION_CONTRACT
                or manifest.get("objective") != "minimize_worst_expected_terminal_loss_plus_cost"
                or manifest.get("model_weights") is not None
                or manifest.get("statistical_confidence_level") is not None):
            raise ValueError("Unexpected finite ambiguity decision model contract")
        if not isinstance(manifest.get("model_order"), (list, tuple)):
            raise ValueError("Model order must be a sequence of IDs")
        order = _names(manifest["model_order"], "Model order")
        raw = manifest.get("models")
        if not isinstance(raw, Mapping) or set(order) != set(raw):
            raise ValueError("Model order must name every model exactly once")
        if len(raw) > LIMITS["models"]:
            raise ValueError("Model count exceeds exact enumeration limit")
        return cls({name: FiniteDecisionLossModel.from_manifest(raw[name]) for name in order})

    def _checked_observations(self, observations):
        if not isinstance(observations, (list, tuple)):
            raise ValueError("Observations must be an ordered action/outcome sequence")
        history, used_actions, used_evidence = [], set(), set()
        for pair in observations:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or not all(isinstance(item, str) for item in pair)):
                raise ValueError("Observation entries must be action/outcome string pairs")
            action, outcome = pair
            if action not in self.tools:
                raise ValueError(f"Unknown observation action: {action}")
            if outcome not in self.tools[action].outcomes:
                raise ValueError(f"Unknown observation outcome: {action}/{outcome}")
            if action in used_actions:
                raise ValueError("Observation history repeats a one-shot action")
            evidence = self.tools[action].evidence_id
            if evidence is not None and evidence in used_evidence:
                raise ValueError("Observation history repeats the same evidence ID")
            used_actions.add(action)
            if evidence is not None:
                used_evidence.add(evidence)
            history.append((action, outcome))
        return tuple(history)

    def _replay(self, history):
        states, active = {}, {}
        for name, model in self.models.items():
            logs = [math.log(p) if p else -math.inf for p in model.prior]
            log_evidence = 0.0
            for index, (action, outcome) in enumerate(history):
                tool = model.tools[action]
                column = tool.outcomes.index(outcome)
                weights = [lp + math.log(tool.likelihoods[theta][column])
                           if lp != -math.inf and tool.likelihoods[theta][column] > 0 else -math.inf
                           for theta, lp in zip(model.hypothesis_names, logs)]
                scale = max(weights)
                if scale == -math.inf:
                    states[name] = {"status": "INCOMPATIBLE_WITH_OBSERVED_HISTORY",
                                    "excluded_at_observation": index, "action_id": action, "outcome": outcome,
                                    "reason": "exact_zero_likelihood_under_declared_model",
                                    "posterior": None, "history_log_likelihood": None}
                    break
                normalizer = scale + math.log(math.fsum(math.exp(w-scale) for w in weights))
                logs = [w-normalizer for w in weights]
                log_evidence += normalizer
            else:
                belief = tuple(math.exp(w) for w in logs)
                # Normalize harmless finite precision drift, while log-space
                # replay retains tiny hypotheses that can recover later.
                total = math.fsum(belief)
                belief = tuple(p/total for p in belief)
                active[name] = belief
                prediction = tuple(math.fsum(p*row[y] for p, row in zip(belief, model.outcome_probabilities))
                                   for y in range(len(self.outcome_labels)))
                states[name] = {"status": "COMPATIBLE_WITH_OBSERVED_HISTORY",
                                "posterior": dict(zip(model.hypothesis_names, belief)),
                                "log_posterior": {theta: lp if math.isfinite(lp) else None
                                                  for theta, lp in zip(model.hypothesis_names, logs)},
                                "prediction": list(prediction), "history_log_likelihood": log_evidence}
        return active, states

    def evaluate(self, observations=(), *, remaining_budget, horizon=1):
        if type(horizon) is not int or horizon != 1:
            raise ValueError("Finite ambiguity value supports only the explicit one-query horizon=1")
        budget = _number(remaining_budget, "Remaining budget")
        history = self._checked_observations(observations)
        active, states = self._replay(history)
        model_hash = sha256(json.dumps(self.manifest(), ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        result = {"schema_version": VERSION, "evidence_class": EVIDENCE_CLASS,
                  "model_set_hash": model_hash, "horizon": 1, "remaining_budget": budget,
                  "cost_unit": self.cost_unit, "bounds_scope": dict(BOUNDS_SCOPE),
                  "execution_contract": EXECUTION_CONTRACT,
                  "global_open_world_stop_proven": False, "statistical_confidence_level": None,
                  "observations": [list(pair) for pair in history], "model_states": states,
                  "active_model_ids": list(active),
                  "excluded_model_ids": [name for name in states if name not in active],
                  "candidates": [], "stop_certificate": None, "dominance_stop_certificate": None}
        if not active:
            return {**result, "status": "MODEL_SET_MISSPECIFIED", "action_id": None,
                    "reason": "actual_history_has_zero_likelihood_in_every_declared_model",
                    "terminal_decision_if_stopping": None, "current_risk": None,
                    "decision_risk_spread": None, "expected_total_risk": None,
                    "robust_net_value": None}

        # Expected terminal loss for each latent hypothesis is reusable across
        # both current decision risks and all acquisition response policies.
        latent_losses, risks = {}, {}
        for name, belief in active.items():
            model = self.models[name]
            latent_losses[name] = tuple(
                tuple(math.fsum(p*loss for p, loss in zip(row, self.decision_losses[d]))
                      for d in self.decision_names) for row in model.outcome_probabilities)
            risks[name] = tuple(math.fsum(p*row[j] for p, row in zip(belief, latent_losses[name]))
                                for j in range(len(self.decision_names)))
        worst_by_decision = tuple(max(risk[j] for risk in risks.values()) for j in range(len(self.decision_names)))
        # Tuple order is the frozen tie order; no model-specific terminal
        # oracle is used as the Agent's pre-query decision.
        baseline_index = min(range(len(self.decision_names)), key=lambda j: worst_by_decision[j])
        baseline = self.decision_names[baseline_index]
        r0 = worst_by_decision[baseline_index]
        baseline_risks = {name: row[baseline_index] for name, row in risks.items()}
        stop = {"decision": baseline, "worst_expected_loss": r0,
                "per_model_expected_risk": baseline_risks,
                "per_decision_expected_risk": {d: {name: row[j] for name, row in risks.items()}
                                               for j, d in enumerate(self.decision_names)}}
        result.update(status="SUPPORTED_DECLARED_MODEL_SET", terminal_decision_if_stopping=stop,
                      current_risk=r0, decision_risk_spread=_spread(tuple(baseline_risks.values())))
        chosen, best_risk = "STOP", r0
        used = {action for action, _ in history}
        used_evidence = {self.tools[action].evidence_id for action in used if self.tools[action].evidence_id is not None}
        for action in self.action_ids:
            tool = self.tools[action]
            reason = ("already_observed" if action in used else
                      "evidence_already_observed" if tool.evidence_id is not None and tool.evidence_id in used_evidence else
                      "budget_exhausted" if tool.cost > budget + VALUE_TOLERANCE else None)
            if reason:
                result["candidates"].append({"action_id": action, "feasible": False, "reason": reason})
                continue
            candidate = self._evaluate_action(action, active, latent_losses, baseline_risks, r0)
            result["candidates"].append(candidate)
            if candidate["worst_expected_risk"] < best_risk - VALUE_TOLERANCE:
                chosen, best_risk = action, candidate["worst_expected_risk"]
        feasible = [candidate for candidate in result["candidates"] if candidate["feasible"]]
        reason = "positive_robust_one_step_value"
        if chosen == "STOP":
            reason = "no_positive_robust_one_step_value" if feasible else "no_feasible_actions"
            result["stop_certificate"] = ("NO_ROBUST_ONE_STEP_IMPROVEMENT_IN_DECLARED_MODEL_SET" if feasible
                                          else "NO_FEASIBLE_REGISTERED_ACTION")
            if feasible and max(c["dominance_upper_gain"] for c in feasible) <= VALUE_TOLERANCE:
                result["dominance_stop_certificate"] = "NO_POSITIVE_ONE_STEP_GAIN_RELATIVE_TO_SHARED_BASELINE_IN_ANY_DECLARED_MODEL"
        result.update(action_id=chosen, reason=reason, expected_total_risk=best_risk,
                      robust_net_value=max(0.0, r0-best_risk))
        return result

    def _evaluate_action(self, action, active, latent_losses, baseline_risks, r0):
        tool = self.tools[action]
        columns, probabilities = {}, {}
        for name, belief in active.items():
            model = self.models[name]
            rows, pz = [], []
            for z in range(len(tool.outcomes)):
                weights = tuple(p*model.tools[action].likelihoods[theta][z]
                                for theta, p in zip(model.hypothesis_names, belief))
                pz.append(math.fsum(weights))
                rows.append(tuple(math.fsum(w*row[j] for w, row in zip(weights, latent_losses[name]))
                                  for j in range(len(self.decision_names))))
            columns[name], probabilities[name] = tuple(rows), tuple(pz)
        best_policy, best_worst, best_risks = None, math.inf, None
        for policy in product(range(len(self.decision_names)), repeat=len(tool.outcomes)):
            losses = {name: tool.cost + math.fsum(rows[z][d] for z, d in enumerate(policy))
                      for name, rows in columns.items()}
            worst = max(losses.values())
            if not math.isfinite(worst):
                raise ArithmeticError("Expected risk overflow exceeds numerical support")
            if worst < best_worst:
                best_policy, best_worst, best_risks = policy, worst, losses
        gains = {name: baseline_risks[name]-best_risks[name] for name in active}
        dominance = {name: baseline_risks[name]-tool.cost-math.fsum(min(row) for row in rows)
                     for name, rows in columns.items()}
        response_policy = {outcome: self.decision_names[best_policy[z]] for z, outcome in enumerate(tool.outcomes)}
        branches = [{"observation": outcome, "terminal_decision": response_policy[outcome],
                     "status": ("POSSIBLE_IN_DECLARED_MODEL_SET" if any(probabilities[name][z] > 0 for name in active)
                                else "IMPOSSIBLE_IN_ALL_ACTIVE_MODELS"),
                     "per_model_probability": {name: probabilities[name][z] for name in active},
                     "per_model_conditional_terminal_risk": {
                         name: columns[name][z][best_policy[z]]/probabilities[name][z] if probabilities[name][z] > 0 else None
                         for name in active}}
                    for z, outcome in enumerate(tool.outcomes)]
        return {"action_id": action, "feasible": True, "cost": tool.cost,
                "response_policy": response_policy, "branches": branches,
                "enumerated_policies": len(self.decision_names)**len(tool.outcomes),
                "per_model_expected_risk": best_risks, "worst_expected_risk": best_worst,
                "robust_net_value": r0-best_worst, "per_model_net_gain": gains,
                "decision_risk_spread": _spread(tuple(best_risks.values())),
                "net_gain_spread": _spread(tuple(gains.values())),
                "dominance_upper_gain": max(dominance.values()),
                "per_model_dominance_upper_gain": dominance,
                "risk_includes_acquisition_cost": True,
                "policy_execution_contract": EXECUTION_CONTRACT}
