"""Opt-in bridge for attaching training exploration to a collector.

The bridge is deliberately a sidecar.  Existing collectors do not call it by
default, and it has no network or f dependencies.  A caller must explicitly
set ``enabled=true`` and pass the already frozen candidate menu.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .g_exploration_policy import TrainingExplorationPolicy


class TrainingExplorationAdapter:
    """Generate a propensity trace without changing production selection."""

    def __init__(self, *, enabled: bool = False, epsilon: float = 0.2, seed: int | None = None) -> None:
        self.enabled = bool(enabled)
        self.policy = TrainingExplorationPolicy(epsilon=epsilon, seed=seed)

    def select_for_collection(
        self,
        menu: Sequence[Mapping[str, Any]],
        predicted_signed_gain: Mapping[str, float],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "schema": "footballmind-g-training-exploration-adapter-v1",
                "enabled": False,
                "selection_authority": "none",
                "selected_action_id": None,
                "candidate_menu": [],
                "reason": "opt_in_required",
            }
        result = self.policy.select(menu, predicted_signed_gain)
        result["adapter_schema"] = "footballmind-g-training-exploration-adapter-v1"
        result["enabled"] = True
        result["network_requests"] = 0
        result["f_invoked"] = False
        result["label_status"] = "pending_outcome"
        return result


__all__ = ["TrainingExplorationAdapter"]
