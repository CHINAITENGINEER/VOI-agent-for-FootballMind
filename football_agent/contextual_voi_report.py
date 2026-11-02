"""Public context-conditioned VoI reports for constructed research scenarios.

The report makes context affect the value of information through an explicit
finite model. It never turns tournament context into a deterministic lineup or
tactical label, and it excludes sampled outcomes and hidden hypotheses.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from .decision_value import EVIDENCE_CLASS, FiniteDecisionModel


VERSION = "contextual-voi-report-v1"


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def build_contextual_report(scenarios: Mapping, *, budget: float | None = None, horizon: int = 1) -> dict:
    """Evaluate declared context scenarios without exposing evaluator state."""
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise ValueError("At least one contextual scenario is required")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in (0, 1, 2):
        raise ValueError("horizon must be 0, 1 or 2")
    rows = []
    for scenario_id, scenario in scenarios.items():
        if not isinstance(scenario_id, str) or not scenario_id.strip() or not isinstance(scenario, Mapping):
            raise ValueError("Contextual scenarios need string IDs and mappings")
        model = scenario.get("model")
        if not isinstance(model, FiniteDecisionModel):
            raise ValueError("Contextual scenario model must be a FiniteDecisionModel")
        context = scenario.get("context")
        if not isinstance(context, Mapping):
            raise ValueError("Context must be a mapping")
        scenario_budget = scenario.get("budget", .08) if budget is None else budget
        scenario_budget = _finite(scenario_budget, "budget")
        decision = model.evaluate(model.initial_state(), remaining_budget=scenario_budget, horizon=horizon)
        candidates = []
        for candidate in decision["candidates"]:
            candidates.append({
                "action_id": candidate["action_id"],
                "feasible": bool(candidate["feasible"]),
                "cost": candidate.get("cost"),
                "immediate_evsi": candidate.get("immediate_evsi"),
                "immediate_net_value": candidate.get("immediate_net_value"),
                "reason": candidate.get("reason"),
            })
        rows.append({
            "scenario_id": scenario_id,
            "context": dict(context),
            "budget": scenario_budget,
            "horizon": horizon,
            "base_prediction": list(decision["prediction"]),
            "current_risk": decision["current_risk"],
            "selected_action": decision["action_id"],
            "selection_reason": decision["reason"],
            "candidates": candidates,
            "assumptions": list(model.assumptions),
            "evidence_class": EVIDENCE_CLASS,
            "hidden_result_included": False,
        })
    return {
        "schema_version": VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "horizon": horizon,
        "scenarios": rows,
        "interpretation": "Context changes declared information value; it does not deterministically imply a lineup or tactical choice.",
        "hidden_result_included": False,
    }


__all__ = ["VERSION", "build_contextual_report"]
