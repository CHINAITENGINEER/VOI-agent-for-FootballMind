"""Connect learned text returns to the existing budgeted acquisition runtime.

The numerical fixture model supplies p0 only. All subsequent forecasts and
query values come from the same ordered text response kernel. Mixed structured
and text updates would need a jointly trained response law and are rejected.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math

from football_research.acquisition.expected_voi import CandidateValue, select_candidate
from football_research.acquisition.joint_planner import UNITS, _budget, _cost
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy, MODELED_GROUPS
from football_research.acquisition.state import sanitize_state
from football_research.acquisition.text_returns import TextReturnModel, VERSION as EXACT_TEXT_VERSION
from football_research.acquisition.value import _canonical

VERSION = "learned-text-strict-voi-v1"
DISCOVERY_VERSION = "learned-text-two-step-voi-v1"
RISK_VERSION = "empirical-text-risk-policy-v1"
NEIGHBORHOOD_RISK_VERSION = "empirical-neighborhood-text-risk-policy-v1"


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


class _Forecast:
    def __init__(self, policy):
        self.policy = policy
        self.model_version = policy.text_model.model_version

    def predict_partial(self, state):
        return self.policy._state(state)[1]


class TextVOIPolicy:
    """History-conditional VoI, optionally including learned discovery value."""

    def __init__(self, base_policy, text_model, *, discovery_model=None, risk_model=None):
        if not isinstance(base_policy, LearnedJointVOIPolicy):
            raise ValueError("A frozen initial fixture policy is required")
        if not isinstance(text_model, TextReturnModel) or not text_model.model_version:
            raise ValueError("A frozen fitted text response model is required")
        neighborhood_risk = False
        if risk_model is not None:
            from football_research.acquisition.neighborhood_text_risk import NeighborhoodTextRiskCalibrator
            neighborhood_risk = isinstance(risk_model, NeighborhoodTextRiskCalibrator)
        if text_model.version != EXACT_TEXT_VERSION and (discovery_model is not None or
                (risk_model is not None and not neighborhood_risk)):
            raise ValueError("A new history kernel requires separately validated risk/discovery models; only one-step mode is supported")
        self.base_policy, self.text_model = base_policy, text_model
        self.discovery_model = discovery_model
        self.risk_model = risk_model
        if risk_model is not None:
            from football_research.acquisition.text_risk import TextRiskCalibrator
            if discovery_model is not None:
                raise ValueError("Empirical one-step risk cannot be combined with uncalibrated two-step discovery")
            if (not isinstance(risk_model, TextRiskCalibrator) or not risk_model.model_version or
                    risk_model.text_model.model_version != text_model.model_version):
                raise ValueError("A frozen empirical risk model bound to this text predictor is required")
            if base_policy.loss != "log_loss":
                raise ValueError("Empirical text risk currently supports log_loss only")
        if discovery_model is not None:
            from football_research.acquisition.discovery_returns import DiscoveryReturnModel
            if (not isinstance(discovery_model, DiscoveryReturnModel) or
                    not discovery_model.model_version or
                    discovery_model.text_model.model_version != text_model.model_version):
                raise ValueError("A frozen discovery model bound to this text response model is required")
        self.loss, self.lambda_cost = base_policy.loss, deepcopy(base_policy.lambda_cost)
        self.manifest = {
            "schema": VERSION, "base_policy": deepcopy(base_policy.manifest),
            "model_version": text_model.model_version,
            "trained_as_of": text_model.metadata["training_as_of"],
            "evidence_origin": text_model.metadata["evidence_origin"],
            "encoder_version": text_model.metadata["encoder_version"],
            "loss": self.loss, "lambda_cost": self.lambda_cost,
            "support_scope": text_model.metadata["support_scope"],
            "value_scope": "one_step_history_conditional_text_response_model",
            "mixed_structured_updates_supported": False,
            "calibration_status": "not_established", "production_ready": False,
        }
        self.schema = DISCOVERY_VERSION if discovery_model is not None else VERSION
        if text_model.version != EXACT_TEXT_VERSION:
            self.manifest.update(
                history_kernel=deepcopy(text_model.metadata["history_kernel"]),
                value_scope="experimental_adjacent_history_conditional_text_response_model",
                generalization_status="conditional_smoothing_assumption_not_established")
        if discovery_model is not None:
            self.manifest.update(
                schema=self.schema, discovery_model_version=discovery_model.model_version,
                discovery_training_cutoff=discovery_model.metadata["training_as_of"],
                value_scope="bounded_two_step_learned_candidate_transition",
                planning_horizon=2,
                candidate_transition_scope="training_supported_hypotheses_not_executable_future_links")
        if risk_model is not None:
            self.schema = NEIGHBORHOOD_RISK_VERSION if neighborhood_risk else RISK_VERSION
            self.manifest.update(
                schema=self.schema, risk_model_version=risk_model.model_version,
                trained_as_of=risk_model.manifest["training_as_of"],
                risk_training_cutoff=risk_model.manifest["training_as_of"],
                risk_evidence_origin=risk_model.manifest["evidence_origin"],
                value_scope="empirical_fixed_text_predictor_risk_reduction_not_bayes_evsi",
                calibration_status="empirical_residual_fit_no_coverage_guarantee",
                stop_scope="empirical_current_menu_rule_not_a_risk_certificate")
            if neighborhood_risk:
                self.manifest.update(
                    empirical_context=deepcopy(risk_model.manifest["context_spec"]),
                    calibration_population=deepcopy(risk_model.manifest["calibration_population"]))
        self.version = self.schema+":"+_digest(self.manifest)[:20]
        self.model = _Forecast(self)

    def _state(self, state):
        clean, initial = self.base_policy._state(state)
        if any(g["status"] != "not_acquired" for g in clean["groups"].values()):
            raise ValueError("Mixed structured and text evidence requires a joint trained response model")
        text_state = state.get("open_text")
        if text_state is None:
            text_state = self.text_model.start(initial["probabilities"], clean["match_id"],
                                                clean["prediction_cutoff"])
        prediction = self.text_model.predict(text_state)
        if self.risk_model is not None:
            self.risk_model.validate_state(text_state)
        if (text_state["match_id"] != clean["match_id"] or
                text_state["prediction_cutoff"] != clean["prediction_cutoff"] or
                text_state["prior_probabilities"] != initial["probabilities"]):
            raise ValueError("Text history is bound to its original fixture and initial forecast")
        clean["open_text"] = deepcopy(text_state)
        return clean, {**prediction, "initial_model_version": self.base_policy.model.model_version,
                       "information_mode": clean["information_mode"],
                       "evidence_origin": self.text_model.metadata["evidence_origin"],
                       "calibration_status": "not_established"}

    def public_state(self, state):
        clean, _ = self._state(state)
        # Preserve actual execution order; the base policy's digest projection
        # uses sorted action identities only for its older structured protocol.
        clean["action_history"] = deepcopy(state.get("action_history", []))
        return clean

    def plan(self, state, actions, budget, *, candidate_generation_complete=False):
        if type(candidate_generation_complete) is not bool:
            raise ValueError("Menu completeness must be Boolean")
        clean, prediction = self._state(state)
        limits, spent = _budget(budget)
        used = set(clean["action_history"])
        candidates, values, seen = [], [], set()
        for raw in sorted(actions, key=lambda row: row["action_id"]):
            aid = raw["action_id"]
            if aid in seen or not isinstance(aid, str) or not aid:
                raise ValueError("Unique action identities required")
            if aid in MODELED_GROUPS or raw.get("group_ids", []):
                raise ValueError("Text acquisition cannot insert a structured-model update")
            if raw.get("tool") not in {"read_url", "search_web"}:
                raise ValueError("Text acquisition requires an explicit web tool")
            if type(raw.get("available", True)) is not bool:
                raise ValueError("Action availability must be Boolean")
            seen.add(aid)
            action = {**deepcopy(raw), "cost": _cost(raw)}
            excluded = ("action_unavailable" if not action.get("available", True) else
                        "already_attempted" if aid in used else
                        "physical_budget_exhausted" if any(spent[u]+action["cost"][u] > cap+1e-12
                                                          for u, cap in limits.items()) else None)
            if excluded:
                candidates.append({**action, "feasible": False, "reason": excluded, "value": None})
                continue
            cost = math.fsum(self.lambda_cost[u]*action["cost"][u] for u in UNITS)
            enumeration = self.text_model.enumerate(clean["open_text"], action,
                                                     cost=cost, loss=self.loss)
            continuation = None
            if self.discovery_model is not None:
                continuation = self.discovery_model.enumerate(
                    clean["open_text"], action, budget={"limits": limits, "spent": spent},
                    lambda_cost=self.lambda_cost, current_actions=actions, loss=self.loss)
            selection_value = continuation["value"] if continuation is not None else enumeration["value"]
            empirical = None
            if self.risk_model is not None:
                empirical = self.risk_model.estimate(clean["open_text"], action, cost=cost, loss=self.loss)
                selection_value = empirical["value"]
            value = CandidateValue(**{k: v for k, v in selection_value.items()
                                       if k != "schema_version"})
            values.append(value)
            candidates.append({**action, "feasible": True, "reason": value.reason,
                               "value": value.as_dict(), "branches": enumeration["branches"],
                               "response_enumeration": enumeration})
            if continuation is not None:
                candidates[-1]["continuation_enumeration"] = continuation
            if empirical is not None:
                candidates[-1]["empirical_risk_estimate"] = empirical["empirical_estimate"]
        selection = select_candidate(values)
        if selection["decision"] == "stop":
            if not values and any(c["reason"] == "physical_budget_exhausted" for c in candidates):
                selection.update(decision="halt", reason="physical_budget_exhausted", stop_certificate=None)
            elif not candidate_generation_complete:
                selection.update(decision="defer", reason="candidate_generation_incomplete", stop_certificate=None)
            else:
                selection["stop_certificate"].update(
                    scope="declared_feasible_one_step_menu_only", model_version=self.text_model.model_version,
                    does_not_bound_multi_step_or_undiscovered_information=True)
                if self.discovery_model is not None:
                    selection["stop_certificate"].update(
                        scope="declared_supported_two_step_menu_only",
                        discovery_model_version=self.discovery_model.model_version,
                        planning_horizon=2,
                        does_not_bound_multi_step_or_undiscovered_information=True)
        if self.risk_model is not None:
            # A descriptive residual estimate is not a Bayes-risk guarantee.
            # Unknown actions still defer through the unchanged selector.
            selection["stop_certificate"] = None
            if selection["decision"] == "stop":
                selection["reason"] = "empirical_current_menu_nonpositive"
            elif selection["decision"] == "acquire":
                selection["reason"] = "best_positive_empirical_net_gain"
        result = {
            "schema": self.schema, "policy_version": self.version,
            "model_version": self.text_model.model_version,
            "training_cutoff": self.text_model.metadata["training_as_of"],
            "match_id": clean["match_id"], "prediction_cutoff": clean["prediction_cutoff"],
            "state_digest": _digest(clean), "information_mode": clean["information_mode"],
            "prediction": prediction, "candidates": candidates, "selection": selection,
            "physical_budget": {"limits": limits, "spent": spent},
            "candidate_generation_complete": candidate_generation_complete,
            "selection_scope": "best_positive_computed_candidate_not_open_world_optimum",
            "calibration_status": "not_established", "numeric_voi_training_eligible": False,
        }
        if self.risk_model is not None:
            result.update(risk_model_version=self.risk_model.model_version,
                          training_cutoff=self.risk_model.manifest["training_as_of"],
                          value_kind="empirical_predictor_risk_reduction",
                          calibration_status="empirical_residual_fit_no_coverage_guarantee")
        result["decision_digest"] = _digest(result)
        return result

    def apply_return(self, decision, before, action, returned, admitted_document):
        """Update only from the actual admitted response and pre-query branches."""
        clean, _ = self._state(before)
        if (decision.get("decision_digest") != _digest({k: v for k, v in decision.items()
                                                        if k != "decision_digest"}) or
                decision.get("policy_version") != self.version or
                decision.get("state_digest") != _digest(clean)):
            raise ValueError("Text return does not belong to this frozen decision")
        candidate = next((c for c in decision["candidates"] if c["action_id"] == action["action_id"]), None)
        if candidate is None or not candidate["feasible"]:
            raise ValueError("Actual text action was not feasible in the frozen decision")
        bound = {k: v for k, v in candidate.items()
                 if k not in {"feasible", "reason", "value", "branches", "response_enumeration",
                              "continuation_enumeration", "empirical_risk_estimate"}}
        if bound != {**action, "cost": _cost(action)}:
            raise ValueError("Actual action differs from its pre-query request")
        if returned["status"] != "OK":
            response = {"status": returned["status"], "kind": "unknown", "text": ""}
        elif admitted_document is None:
            # A late/unusable body is not observed content. Preserve the attempt
            # as unsupported; it must not reuse an empty-history likelihood.
            response = {"status": "OK", "kind": "unknown", "text": ""}
        else:
            from football_research.data_adapters.open_text_episodes import response_for_model
            normalized = response_for_model(admitted_document)
            response = {key: normalized[key] for key in ("status", "kind", "text")}
        update = self.text_model.update(clean["open_text"], action, response,
                                         decision=candidate["response_enumeration"])
        after = self.public_state(before)
        after["action_history"].append({"action_id": action["action_id"]})
        after["open_text"] = update["state"]
        record = {key: deepcopy(value) for key, value in update.items() if key != "state"}
        record.update(prediction_before=decision["prediction"]["probabilities"],
                      action_id=action["action_id"], response_kind=response["kind"],
                      actual_response_digest=_digest(response),
                      document_id=admitted_document.get("document_id") if admitted_document else None,
                      probability_changed=update["prediction_after"] != decision["prediction"]["probabilities"],
                      evidence_origin=self.text_model.metadata["evidence_origin"])
        return after, record


__all__ = ["TextVOIPolicy", "VERSION", "DISCOVERY_VERSION", "RISK_VERSION", "NEIGHBORHOOD_RISK_VERSION"]
