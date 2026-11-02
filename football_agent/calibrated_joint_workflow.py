"""Explicit single-step calibrated gate, preserving the frozen joint executor."""
from __future__ import annotations

from .calibrated_joint_bridge import GATE_KIND
from .joint_workflow import JointAcquisitionWorkflow


class CalibratedJointAcquisitionWorkflow(JointAcquisitionWorkflow):
    version = "visible-calibrated-joint-acquisition-executor-v1"

    def start(self, match_id, cutoff, strategy="all_candidates", budget=None, seed=0, risk_mode=None):
        gate = self.backend.configuration().get("calibrated_gate", {})
        if gate.get("kind") != GATE_KIND or gate.get("horizon") != 1:
            raise ValueError("A single-step calibrated gate backend is required")
        configured = gate["risk_level"]
        if risk_mode is not None and risk_mode != configured:
            raise ValueError("Workflow risk mode differs from the frozen calibrated gate")
        return super().start(match_id, cutoff, strategy=strategy, budget=budget, seed=seed, risk_mode=configured)

    def _merge_values(self, run, legal, values):
        valid = super()._merge_values(run, legal, values)
        for item in valid:
            item["net_gain"] = round(item["net_gain"], 12)
            if item["conservative_net_gain"] is not None:
                item["conservative_net_gain"] = round(item["conservative_net_gain"], 12)
            for candidate in run["candidates"]:
                if candidate["action_id"] == item["action_id"]:
                    candidate["value"] = item
        return valid
