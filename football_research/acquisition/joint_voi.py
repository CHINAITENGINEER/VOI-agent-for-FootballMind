"""Strict one-step VoI from a fitted model, with an explicitly incomplete menu.

This adapter reuses the joint predictor and physical-resource contracts. It
does not learn returns from a test match, execute tools, or infer that all
open-world information has been enumerated. Unsupported tools retain UNKNOWN
value alongside computed candidates. Exploration remains a separate policy.
"""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import math

from .expected_voi import evaluate_candidate, select_candidate, unknown_candidate
from .joint_planner import UNITS, _budget, _cost, _number
from .joint_returns import JointReturnPredictor
from .state import GROUP_FIELDS, sanitize_state
from .value import _canonical

VERSION = "learned-joint-strict-voi-v1"
MODELED_GROUPS = {"results_view": {"elo", "recent_form"}, "xg_view": {"xg"}}


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


class LearnedJointVOIPolicy:
    """Frozen, out-of-sample, model-conditional predictive VoI.

    Minima are declared support guards, not learned confidence intervals.
    ``lambda_cost`` converts each physical resource into the selected loss's
    units; the original resource vector separately determines affordability.
    """

    def __init__(self, model, *, lambda_cost, loss="log_loss",
                 min_effective_matches=12., min_class_weight=1.):
        if not isinstance(model, JointReturnPredictor) or not model.model_version:
            raise ValueError("A frozen fitted JointReturnPredictor is required")
        if loss not in {"log_loss", "brier"}:
            raise ValueError("Unsupported predictive loss")
        if not isinstance(lambda_cost, Mapping) or set(lambda_cost) - set(UNITS):
            raise ValueError("Declare physical resource to loss conversion weights")
        self.model, self.loss = model, loss
        self.lambda_cost = {u: _number(lambda_cost.get(u, 0.), "Lambda " + u) for u in UNITS}
        self.min_effective_matches = _number(min_effective_matches, "Minimum effective matches")
        self.min_class_weight = _number(min_class_weight, "Minimum class weight")
        if self.min_effective_matches <= 0 or self.min_class_weight <= 0:
            raise ValueError("Positive support minima required")
        self.manifest = {"schema": VERSION, "model_version": model.model_version,
                         "trained_as_of": model.metadata["training_as_of"],
                         "loss": loss, "lambda_cost": self.lambda_cost,
                         "min_effective_matches": self.min_effective_matches,
                         "min_class_weight": self.min_class_weight,
                         "support_scope": "unsmoothed currently observed return context",
                         "value_scope": "one-step expectation under fitted joint model",
                         "calibration_status": "not_established",
                         "parameter_uncertainty": "not_estimated"}
        self.version = VERSION + ":" + _digest(self.manifest)[:20]
        self.cost_version = "declared-loss-conversion:" + _digest({"loss": loss, "weights": self.lambda_cost})[:20]

    def _state(self, state):
        # Validate real-state/failure boundaries before sanitize_state removes
        # non-feature metadata; hypothetical branch tokens cannot be admitted.
        prediction = self.model.predict_partial(state)
        clean = sanitize_state(state)
        trained = set(self.model.metadata["training_match_ids"]) | set(self.model.metadata["base_training_match_ids"])
        if clean["match_id"] in trained:
            raise ValueError("Out-of-sample decision cannot reuse a training match")
        history = state.get("action_history", [])
        if not isinstance(history, list):
            raise ValueError("Action history must be a list")
        used = []
        for item in history:
            if not isinstance(item, Mapping) or not isinstance(item.get("action_id"), str) or not item["action_id"]:
                raise ValueError("Action history requires explicit action identities")
            used.append(item["action_id"])
        clean["action_history"] = sorted(set(used))
        return clean, prediction

    def plan(self, state, actions, budget, *, candidate_generation_complete=False):
        if type(candidate_generation_complete) is not bool:
            raise ValueError("Menu completeness must be Boolean")
        clean, prediction = self._state(state)
        support = self.model.observation_support(state)
        reason = ("unknown_fixture_identity" if prediction.get("unknown_identity_fields") else
                  "insufficient_conditional_return_support" if
                  support["kernel_effective_matches"] < self.min_effective_matches or
                  min(support["class_kernel_weight"].values()) < self.min_class_weight else None)
        limits, spent = _budget(budget)
        if not isinstance(actions, (list, tuple)):
            raise ValueError("Actions must be a sequence")
        checked, seen = [], set()
        for raw in actions:
            if not isinstance(raw, Mapping):
                raise ValueError("Actions must be mappings")
            aid = raw.get("action_id")
            if not isinstance(aid, str) or not aid or aid in seen:
                raise ValueError("Unique nonempty action identities required")
            seen.add(aid)
            if type(raw.get("available", True)) is not bool:
                raise ValueError("Action availability must be Boolean")
            groups = raw.get("group_ids", [])
            if (not isinstance(groups, (list, tuple)) or
                    any(not isinstance(g, str) or not g for g in groups) or len(set(groups)) != len(groups)):
                raise ValueError("Distinct action group identifiers required")
            if aid in MODELED_GROUPS and set(groups) != MODELED_GROUPS[aid]:
                raise ValueError("Modeled action changed its trained observation groups")
            checked.append({"action_id": aid, "group_ids": list(groups),
                            "available": raw.get("available", True),
                            "cost_status": raw.get("cost_status"), "cost": _cost(raw)})
        used = set(self.model.observed_actions(state)) | set(clean["action_history"])
        values, candidates = [], []
        for action in sorted(checked, key=lambda a: a["action_id"]):
            aid = action["action_id"]
            excluded = ("action_unavailable" if not action["available"] else
                        "already_attempted" if aid in used else
                        "physical_budget_exhausted" if any(spent[u] + action["cost"][u] > cap + 1e-12
                                                          for u, cap in limits.items()) else None)
            if excluded:
                candidates.append({**action, "feasible": False, "reason": excluded, "value": None})
                continue
            cost = math.fsum(self.lambda_cost[u] * action["cost"][u] for u in UNITS)
            unavailable = "response_model_unavailable" if aid not in MODELED_GROUPS else reason
            branches = None
            if unavailable:
                value = unknown_candidate(aid, prediction["probabilities"], reason=unavailable,
                                          cost=cost, loss=self.loss, predictor_version=self.model.model_version,
                                          cost_model_version=self.cost_version)
            else:
                enumeration = self.model.enumerate_action(state, action)
                branches = [{"branch_id": b["observation"], "probability": b["probability"],
                             "posterior": b["posterior_probabilities"]} for b in enumeration["branches"]]
                value = evaluate_candidate(aid, prediction["probabilities"], branches, cost=cost,
                                           loss=self.loss, predictor_version=self.model.model_version,
                                           response_model_version=self.model.model_version,
                                           cost_model_version=self.cost_version)
            values.append(value)
            candidates.append({**action, "feasible": True, "reason": value.reason,
                               "value": value.as_dict(), "branches": branches})
        selection = select_candidate(values)
        if selection["decision"] == "stop":
            if any(c["reason"] == "physical_budget_exhausted" for c in candidates) and not values:
                selection.update(decision="halt", reason="physical_budget_exhausted", stop_certificate=None)
            elif not candidate_generation_complete:
                selection.update(decision="defer", reason="candidate_generation_incomplete", stop_certificate=None)
            else:
                selection["stop_certificate"].update(
                    scope="declared_feasible_one_step_menu_only", model_version=self.model.model_version,
                    does_not_bound_multi_step_or_undiscovered_information=True)
        result = {"schema": VERSION, "policy_version": self.version, "model_version": self.model.model_version,
                  "training_cutoff": self.model.metadata["training_as_of"], "match_id": clean["match_id"],
                  "prediction_cutoff": clean["prediction_cutoff"], "state_digest": _digest(clean),
                  "information_mode": clean["information_mode"], "prediction": prediction,
                  "support": support, "candidates": candidates, "selection": selection,
                  "physical_budget": {"limits": limits, "spent": spent},
                  "candidate_generation_complete": candidate_generation_complete,
                  "selection_scope": "best_positive_computed_candidate_not_open_world_optimum",
                  "calibration_status": "not_established", "numeric_voi_training_eligible": False}
        result["decision_digest"] = _digest(result)
        return result

    def verify_update(self, decision, before, after):
        """Bind an admitted actual return to its pre-query model branch.

        The caller owns acquisition/admission. This method never accepts a
        hypothetical branch in place of the real returned evidence state.
        Provider failures, cutoff violations and unrelated group changes fail.
        """
        if decision.get("decision_digest") != _digest({k: v for k, v in decision.items() if k != "decision_digest"}):
            raise ValueError("Decision digest mismatch")
        if decision.get("policy_version") != self.version or decision.get("model_version") != self.model.model_version:
            raise ValueError("Decision model/policy changed")
        old, old_prediction = self._state(before)
        new, prediction = self._state(after)
        if _digest(old) != decision["state_digest"]:
            raise ValueError("Pre-query state does not match decision")
        if decision["selection"]["decision"] != "acquire":
            raise ValueError("Only a selected acquisition can produce an update")
        aid = decision["selection"]["candidate_id"]
        candidate = next(c for c in decision["candidates"] if c["action_id"] == aid)
        for key in ("match_id", "prediction_cutoff", "information_mode", "base"):
            if old[key] != new[key]:
                raise ValueError("Response changed fixture or prediction cutoff")
        for group in GROUP_FIELDS:
            if group in candidate["group_ids"]:
                if old["groups"][group]["status"] != "not_acquired" or new["groups"][group]["status"] not in {"observed", "source_missing"}:
                    raise ValueError("Selected response lacks complete newly acquired groups")
            elif old["groups"][group] != new["groups"][group]:
                raise ValueError("Response revealed or altered unrelated evidence")
        expected_history = set(old["action_history"]) | {aid}
        if set(new["action_history"]) != expected_history:
            raise ValueError("Response history must add exactly the selected action")
        token = self.model.observed_categories(after)[aid]
        branch = next((b for b in candidate["branches"] if b["branch_id"] == token), None)
        if branch is None or max(abs(p-q) for p, q in zip(prediction["probabilities"], branch["posterior"])) > 1e-10:
            raise ValueError("Observed posterior disagrees with pre-query response model")
        return {"schema": VERSION, "match_id": old["match_id"], "action_id": aid,
                "prediction_cutoff": old["prediction_cutoff"], "decision_digest": decision["decision_digest"],
                "before_state_digest": _digest(old), "after_state_digest": _digest(new),
                "prediction_before": old_prediction["probabilities"], "prediction_after": prediction["probabilities"],
                "observed_category": token, "predicted_response_probability": branch["probability"],
                "model_version": self.model.model_version, "realized_loss_reduction": None,
                "scope": "actual admitted state conditioned under frozen model; outcome not accessed"}
