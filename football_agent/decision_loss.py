"""Explicit terminal decision losses on the existing finite observation model."""
from __future__ import annotations

import math
from types import MappingProxyType

from .decision_value import FiniteDecisionModel, ObservationTool, EVIDENCE_CLASS, _number, _names


class FiniteDecisionLossModel(FiniteDecisionModel):
    VERSION = "finite-terminal-decision-loss-v1"

    def __init__(self, hypotheses, prior, tools, *, outcome_labels, decision_losses, cost_unit, assumptions=()):
        if not isinstance(cost_unit, str) or not cost_unit.strip():
            raise ValueError("An explicit terminal-loss cost unit is required")
        super().__init__(hypotheses, prior, tools, outcome_labels=outcome_labels, assumptions=assumptions)
        if not isinstance(decision_losses, dict):
            raise ValueError("Named terminal decisions and loss vectors required")
        self.decision_names = _names(tuple(decision_losses), "Terminal decisions")
        rows = {}
        for name in self.decision_names:
            raw = decision_losses[name]
            if not isinstance(raw, (list, tuple)) or len(raw) != len(self.outcome_labels):
                raise ValueError("Terminal loss vector must cover every outcome")
            rows[name] = tuple(_number(v, "Terminal loss") for v in raw)
        self.decision_losses = MappingProxyType(rows)
        self.loss, self.cost_unit = "terminal_decision_matrix", cost_unit

    def terminal_decision(self, state):
        prediction = self.prediction(state)
        risks = {name: math.fsum(p * loss for p, loss in zip(prediction, self.decision_losses[name]))
                 for name in self.decision_names}
        # Manifest order is the declared deterministic tie order.
        chosen = min(self.decision_names, key=lambda name: risks[name])
        return {"decision": chosen, "expected_loss": risks[chosen], "risks": risks,
                "cost_unit": self.cost_unit, "evidence_class": EVIDENCE_CLASS}

    def bayes_risk(self, state):
        return self.terminal_decision(state)["expected_loss"]

    def evaluate(self, state, *, remaining_budget, horizon=1):
        result = super().evaluate(state, remaining_budget=remaining_budget, horizon=horizon)
        result["terminal_decision_if_stopping"] = self.terminal_decision(state)
        return result

    def manifest(self):
        return {**super().manifest(), "decision_order": list(self.decision_names),
                "decision_losses": {name: list(row) for name, row in self.decision_losses.items()},
                "risk_definition": "min_d sum_y P(y|acquired observations) L(d,y)",
                "cost_interpretation": "declared penalty in terminal-loss units, not measured money"}

    @classmethod
    def from_manifest(cls, manifest):
        if (manifest.get("schema_version") != cls.VERSION or manifest.get("evidence_class") != EVIDENCE_CLASS
                or manifest.get("loss") != "terminal_decision_matrix"):
            raise ValueError("Unexpected terminal decision model contract")
        if set(manifest["decision_order"]) != set(manifest["decision_losses"]) or len(manifest["decision_order"]) != len(manifest["decision_losses"]):
            raise ValueError("Decision order must name each loss row once")
        return cls(manifest["hypotheses"], manifest["prior"],
            {a: ObservationTool(tuple(t["outcomes"]), t["likelihoods"], t["cost"], t.get("description", ""), t.get("evidence_id"))
             for a, t in manifest["tools"].items()}, outcome_labels=manifest["outcome_labels"],
            decision_losses={name: manifest["decision_losses"][name] for name in manifest["decision_order"]},
            cost_unit=manifest["cost_unit"], assumptions=manifest["assumptions"])
