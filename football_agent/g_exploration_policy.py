"""Opt-in exploration policy for collecting g training labels.

It is intentionally separate from the production VOI selector.  The policy
never emits STOP: every candidate receives a positive propensity so the
collector can observe counterexamples to the current g estimate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .g_label_contract import (
    EXPLORATION_POLICY_SCHEMA,
    epsilon_greedy_propensities,
    sample_action,
)


class TrainingExplorationPolicy:
    """Freeze a finite candidate menu and sample one action for data collection."""

    def __init__(self, *, epsilon: float = 0.2, seed: int | None = None) -> None:
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("training exploration epsilon must be in (0, 1]")
        self.epsilon = float(epsilon)
        self.seed = seed

    @staticmethod
    def _menu_digest(menu: Sequence[Mapping[str, Any]]) -> str:
        canonical = json.dumps(list(menu), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def select(
        self,
        menu: Sequence[Mapping[str, Any]],
        predicted_signed_gain: Mapping[str, float],
    ) -> dict[str, Any]:
        if not menu:
            raise ValueError("training menu cannot be empty")
        action_ids = [str(item.get("action_id", "")) for item in menu]
        if any(not action_id for action_id in action_ids):
            raise ValueError("every menu item needs an action_id")
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("candidate action_id values must be unique")
        scores = {action_id: float(predicted_signed_gain[action_id]) for action_id in action_ids}
        propensities = epsilon_greedy_propensities(action_ids, scores, epsilon=self.epsilon)
        selected_action, selected_propensity = sample_action(propensities, seed=self.seed)
        return {
            "schema": EXPLORATION_POLICY_SCHEMA,
            "phase": "training_collection",
            "selection_authority": "data_collection_only",
            "empirical_g_enabled": False,
            "stop_allowed": False,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "menu_digest": self._menu_digest(menu),
            "candidate_menu": [
                {
                    "action_id": action_id,
                    "predicted_signed_gain": scores[action_id],
                    "propensity": propensities[action_id],
                    "selected": action_id == selected_action,
                    "realized_signed_gain": None,
                }
                for action_id in action_ids
            ],
            "selected_action_id": selected_action,
            "selected_propensity": selected_propensity,
            "label_status": "pending_outcome",
        }


__all__ = ["TrainingExplorationPolicy"]
