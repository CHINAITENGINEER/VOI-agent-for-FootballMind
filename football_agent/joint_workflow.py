"""Keep the legacy executor while refreshing modeled missing-return beliefs."""
from __future__ import annotations

from copy import deepcopy

from .acquisition import AcquisitionWorkflow


class JointAcquisitionWorkflow(AcquisitionWorkflow):
    """A missing categorical return can be informative without numeric features.

    The legacy workflow marks numerical group/mask changes as new evidence. A
    learned joint model also conditions on an acquired `summary_missing` token.
    This subclass compares public conditioning identities after an acquisition
    and invalidates the old prediction only when that identity really changes.
    """
    version = "visible-joint-acquisition-executor-v1"

    def step(self, run_id):
        with self.store.lock(run_id):
            before = self.get_run(run_id)
            result = super().step(run_id)
            if len(result["history"]) == len(before["history"]) or result["state"] in {"STOPPED", "DEFERRED", "WAITING"}:
                return result
            status = result["history"][-1]["result"].get("status")
            if status in {"FAILED", "TIMEOUT", "PROVIDER_ERROR", "BAD_DATA", "STALE", "UNAVAILABLE"}:
                self._finish(result, "DEFERRED", "JOINT_OBSERVATION_MODEL_MISMATCH",
                             "Unmodeled provider return: " + status)
                return deepcopy(result)
            try:
                previous = self.backend.prediction_conditioning(before["visible_state"])
                current = self.backend.prediction_conditioning(result["visible_state"])
            except (ValueError, TypeError, KeyError) as exc:
                self._finish(result, "DEFERRED", "JOINT_OBSERVATION_MODEL_MISMATCH", str(exc))
                return deepcopy(result)
            if previous != current and result["state"] != "ACQUIRED":
                result["state"], result["reason"] = "ACQUIRED", "JOINT_CONDITIONING_CHANGED"
                self._event(result, "joint_conditioning_changed", {
                    "before": previous, "after": current,
                    "reason": "modeled_return_changed_without_numeric_group_revelation",
                    "next_step": "recompute_partial_prediction"})
            return deepcopy(result)
