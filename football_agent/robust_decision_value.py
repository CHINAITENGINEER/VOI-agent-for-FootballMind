"""Robust value-of-information envelopes for uncertain observation laws.

``FiniteDecisionModel`` assumes that each tool's observation law is known.
That is useful as an exact reference solver, but an open information
environment also has source-quality uncertainty: a report can be stale,
missing, or misleading.  This module evaluates the same candidate action in
several explicitly declared observation-law models and exposes the lower
envelope of their planning value.

The envelope is a decision guard, not a probability mixture and not a claim
that the listed scenarios exhaust the open world.  It is deliberately
conservative: a tool is selected only when its worst declared net value is
positive.  All models must agree on the latent hypotheses, outcome model,
candidate IDs, costs, and observation alphabets; they may differ in the
likelihoods that describe source quality.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .decision_value import (
    BeliefState,
    FiniteDecisionModel,
    PROBABILITY_TOLERANCE,
    VALUE_TOLERANCE,
)


VERSION = "robust-finite-voi-envelope-v1"
ROBUST_EVIDENCE_CLASS = "SYNTHETIC_ASSUMPTIONS"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _close_sequence(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= PROBABILITY_TOLERANCE for a, b in zip(left, right)
    )


class RobustFiniteDecisionModel:
    """Evaluate a common action set under a declared model uncertainty set."""

    VERSION = VERSION

    def __init__(self, models: Mapping[str, FiniteDecisionModel], *, assumptions: Sequence[str] = ()):
        if not isinstance(models, Mapping) or not models:
            raise ValueError("models must be a nonempty mapping")
        normalized: dict[str, FiniteDecisionModel] = {}
        for raw_id, model in models.items():
            scenario_id = _text(raw_id, "scenario_id")
            if not isinstance(model, FiniteDecisionModel):
                raise ValueError("models must contain FiniteDecisionModel values")
            if scenario_id in normalized:
                raise ValueError("scenario IDs must be unique")
            normalized[scenario_id] = model
        if isinstance(assumptions, (str, bytes)) or any(not isinstance(item, str) for item in assumptions):
            raise ValueError("assumptions must be a sequence of strings")
        self.models = dict(sorted(normalized.items()))
        self.assumptions = tuple(assumptions)
        self._validate_compatibility()

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(self.models)

    @property
    def reference(self) -> FiniteDecisionModel:
        return self.models[self.scenario_ids[0]]

    def _validate_compatibility(self) -> None:
        reference = self.reference
        for scenario_id, model in self.models.items():
            if model.hypothesis_names != reference.hypothesis_names:
                raise ValueError(f"{scenario_id} changes the hypothesis set")
            if model.outcome_labels != reference.outcome_labels:
                raise ValueError(f"{scenario_id} changes the outcome labels")
            if not _close_sequence(model.prior, reference.prior):
                raise ValueError(f"{scenario_id} changes the prior")
            for left, right in zip(model.outcome_probabilities, reference.outcome_probabilities):
                if not _close_sequence(left, right):
                    raise ValueError(f"{scenario_id} changes P(Y|hypothesis)")
            if model.loss != reference.loss or model.cost_unit != reference.cost_unit:
                raise ValueError(f"{scenario_id} changes the loss or cost unit")
            if set(model.tools) != set(reference.tools):
                raise ValueError(f"{scenario_id} changes the candidate action set")
            for action_id, tool in reference.tools.items():
                other = model.tools[action_id]
                if tool.outcomes != other.outcomes:
                    raise ValueError(f"{scenario_id}/{action_id} changes the observation alphabet")
                if abs(tool.cost - other.cost) > VALUE_TOLERANCE:
                    raise ValueError(f"{scenario_id}/{action_id} changes the acquisition cost")
                if tool.evidence_id != other.evidence_id:
                    raise ValueError(f"{scenario_id}/{action_id} changes the evidence identity")

    def _state_for(self, model: FiniteDecisionModel, observations: Sequence[Sequence[str]]) -> BeliefState:
        state = model.initial_state()
        for index, pair in enumerate(observations):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"observations[{index}] must be an action/outcome pair")
            state = model.observe(state, str(pair[0]), str(pair[1]))
        return state

    @staticmethod
    def _candidate_by_action(decision: Mapping[str, Any], action_id: str) -> Mapping[str, Any] | None:
        return next(
            (row for row in decision.get("candidates", ()) if row.get("action_id") == action_id),
            None,
        )

    def evaluate(
        self,
        *,
        remaining_budget: float,
        horizon: int = 1,
        observations: Sequence[Sequence[str]] = (),
    ) -> dict[str, Any]:
        """Return worst-case and mean diagnostics for each feasible action.

        The observed history is replayed separately in every model.  A history
        that one declared model considers impossible is rejected instead of
        silently dropping that scenario from the lower envelope.
        """
        budget = _finite(remaining_budget, "remaining_budget")
        if budget < 0:
            raise ValueError("remaining_budget must be nonnegative")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in (0, 1, 2):
            raise ValueError("horizon must be 0, 1 or 2")
        history = tuple(tuple(pair) for pair in observations)
        decisions: dict[str, dict[str, Any]] = {}
        for scenario_id, model in self.models.items():
            state = self._state_for(model, history)
            decisions[scenario_id] = model.evaluate(
                state, remaining_budget=budget, horizon=horizon
            )

        # The initial risk is shared because the models have the same prior and
        # outcome law.  After an observation, however, source-quality models
        # can produce different posteriors and therefore different current
        # risks.  Each model's planning net value is already measured relative
        # to its own current risk; preserving that difference is the point of
        # the robust envelope.
        current_risks = {scenario_id: float(decision["current_risk"]) for scenario_id, decision in decisions.items()}
        reference_risk = current_risks[self.scenario_ids[0]]

        candidates: list[dict[str, Any]] = []
        for action_id in sorted(self.reference.tools):
            rows = {scenario_id: self._candidate_by_action(decision, action_id)
                    for scenario_id, decision in decisions.items()}
            if any(row is None for row in rows.values()):
                continue
            if any(not bool(row["feasible"]) for row in rows.values()):
                candidates.append({
                    "action_id": action_id,
                    "feasible": False,
                    "reason": "infeasible_in_at_least_one_declared_model",
                    "scenario_rows": {
                        scenario_id: {"feasible": bool(row["feasible"]), "reason": row.get("reason")}
                        for scenario_id, row in rows.items()
                    },
                })
                continue
            net_values = {scenario_id: float(row["planning_net_value"])
                          for scenario_id, row in rows.items()}
            immediate_values = {scenario_id: float(row["immediate_net_value"])
                                for scenario_id, row in rows.items()}
            expected_risks = {scenario_id: float(row["expected_total_risk"])
                              for scenario_id, row in rows.items()}
            worst_case = min(net_values.values())
            mean_value = math.fsum(net_values.values()) / len(net_values)
            candidates.append({
                "action_id": action_id,
                "feasible": True,
                "worst_case_planning_net_value": worst_case,
                "mean_planning_net_value": mean_value,
                "worst_case_immediate_net_value": min(immediate_values.values()),
                "scenario_planning_net_values": net_values,
                "scenario_immediate_net_values": immediate_values,
                "scenario_expected_total_risks": expected_risks,
            })

        feasible = [row for row in candidates if row.get("feasible")]
        selected = "STOP"
        selected_raw_value = 0.0
        reason = "horizon_exhausted" if horizon == 0 else "no_positive_worst_case_net_value"
        if feasible:
            best = max(feasible, key=lambda row: (row["worst_case_planning_net_value"], row["action_id"]))
            selected_raw_value = float(best["worst_case_planning_net_value"])
            if selected_raw_value > VALUE_TOLERANCE:
                selected = best["action_id"]
                reason = "positive_worst_case_net_value"

        return {
            "schema_version": self.VERSION,
            "evidence_class": ROBUST_EVIDENCE_CLASS,
            "action_id": selected,
            "reason": reason,
            "horizon": horizon,
            "observations": [list(pair) for pair in history],
            "current_risk": reference_risk,
            "scenario_current_risks": current_risks,
            "worst_case_planning_net_value": selected_raw_value if selected != "STOP" else (
                max((row["worst_case_planning_net_value"] for row in feasible), default=0.0)
            ),
            "remaining_budget": budget,
            "cost_unit": self.reference.cost_unit,
            "scenario_ids": list(self.scenario_ids),
            "candidates": candidates,
            "certificate": {
                "kind": "ROBUST_WORST_CASE_ENVELOPE",
                "scope": "declared_observation_law_set",
                "global_claim": False,
                "candidate_set_closed": True,
                "scenario_count": len(self.models),
                "stop_rule": "select only when the maximum declared worst_case_planning_net_value is positive",
            },
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.VERSION,
            "evidence_class": ROBUST_EVIDENCE_CLASS,
            "scenario_ids": list(self.scenario_ids),
            "scenario_manifests": {scenario_id: model.manifest() for scenario_id, model in self.models.items()},
            "assumptions": list(self.assumptions),
            "interpretation": "lower envelope across declared observation laws; not a posterior mixture",
            "limitations": [
                "The scenario set is manually declared and may not cover the open world.",
                "The envelope is conditional on a shared latent outcome model and action set.",
                "No sampled match result is used in policy evaluation.",
            ],
        }


__all__ = ["VERSION", "ROBUST_EVIDENCE_CLASS", "RobustFiniteDecisionModel"]
