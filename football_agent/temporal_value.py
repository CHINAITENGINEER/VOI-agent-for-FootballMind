"""Finite conditional EVSI masked by time-eligible source actions.

`FiniteDecisionModel` supplies the explicit Bayesian reference model.  This
wrapper adds the operational question: is a registered observation tool in the
action set at the selected prediction cutoff?  Ineligible tools are retained
for provenance but receive an explicit temporal-unavailable status and cannot
be mistaken for a zero-value query.
"""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from typing import Any, Mapping

from .decision_value import FiniteDecisionModel, BeliefState, VALUE_TOLERANCE
from .temporal_inquiry import TemporalInquiryGate


VERSION = "temporal-conditional-evsi-v1"


class TemporalEVSIEvaluator:
    def __init__(self, model: FiniteDecisionModel, bindings: Mapping[str, str], gate: TemporalInquiryGate):
        if not isinstance(model, FiniteDecisionModel) or not isinstance(gate, TemporalInquiryGate):
            raise ValueError("Finite model and temporal gate are required")
        if set(bindings) != set(model.tools) or any(not isinstance(value, str) or not value for value in bindings.values()):
            raise ValueError("Temporal bindings must cover every model action")
        source_ids = {entry["source_id"] for entry in gate.entries}
        if set(bindings.values()) - source_ids:
            raise ValueError("Every model action needs a registered temporal source")
        self.model, self.bindings, self.gate = model, dict(bindings), gate

    def configuration(self) -> dict[str, Any]:
        return {"version": VERSION, "model": self.model.manifest(), "bindings": deepcopy(self.bindings),
                "temporal_gate": self.gate.configuration(),
                "scope": "declared_finite_evsi_masked_by_cutoff_eligibility",
                "open_world_value": None}

    def _status_by_action(self) -> dict[str, dict[str, Any]]:
        entries = {entry["source_id"]: entry for entry in self.gate.entries}
        return {action_id: entries[self.bindings[action_id]] for action_id in self.model.tools}

    def evaluate(self, state: BeliefState, *, remaining_budget: float, horizon: int = 1) -> dict[str, Any]:
        statuses = self._status_by_action()
        # Give temporally unavailable tools a cost strictly above this call's
        # budget.  Their laws remain in the proxy so already acquired evidence
        # continues to validate; their candidate rows are relabeled below.
        masked_tools = {}
        for action_id, tool in self.model.tools.items():
            entry = statuses[action_id]
            if entry["availability_status"] == "ELIGIBLE":
                masked_tools[action_id] = tool
            else:
                masked_tools[action_id] = replace(tool, cost=float(remaining_budget) + max(1.0, tool.cost + 1.0))
        proxy = FiniteDecisionModel(
            dict(zip(self.model.hypothesis_names, self.model.outcome_probabilities)),
            dict(zip(self.model.hypothesis_names, self.model.prior)), masked_tools,
            outcome_labels=self.model.outcome_labels, loss=self.model.loss,
            assumptions=self.model.assumptions + ("Temporal eligibility mask is operational, not a zero-value claim.",),
        )
        result = proxy.evaluate(state, remaining_budget=remaining_budget, horizon=horizon)
        rows = []
        observed = {action for action, _ in state.observations}
        observed_evidence = {self.model.tools[action].evidence_id for action in observed
                             if self.model.tools[action].evidence_id is not None}
        for row in result["candidates"]:
            action_id = row["action_id"]
            entry = statuses[action_id]
            if (entry["availability_status"] != "ELIGIBLE" and action_id not in observed
                    and self.model.tools[action_id].evidence_id not in observed_evidence):
                rows.append({"action_id": action_id, "feasible": False,
                             "reason": "temporally_unavailable", "availability_status": entry["availability_status"],
                             "availability_reason": entry["availability_reason"],
                             "source_id": entry["source_id"], "immediate_evsi": None,
                             "planning_net_value": None})
            else:
                rows.append({**row, "availability_status": entry["availability_status"],
                             "source_id": entry["source_id"]})
        result["candidates"] = rows
        result.update({"schema_version": VERSION, "finite_model_schema_version": self.model.VERSION,
                       "temporal_gate_version": self.gate.configuration()["version"],
                       "temporal_mask_scope": "unavailable_is_not_zero_value",
                       "availability_by_action": {action_id: statuses[action_id]["availability_status"]
                                                   for action_id in sorted(statuses)},
                       "open_world_value": None, "calibrated": False})
        return result

