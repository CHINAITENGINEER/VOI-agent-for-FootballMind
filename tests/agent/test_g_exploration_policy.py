import pytest

from football_agent.g_exploration_policy import TrainingExplorationPolicy


MENU = [
    {"action_id": "lineup", "cost": 0.01},
    {"action_id": "injury", "cost": 0.01},
    {"action_id": "weather", "cost": 0.01},
]


def test_training_policy_selects_without_stop_and_logs_all_propensities():
    result = TrainingExplorationPolicy(epsilon=0.4, seed=3).select(
        MENU, {"lineup": -0.5, "injury": -0.7, "weather": -0.9}
    )
    assert result["stop_allowed"] is False
    assert result["empirical_g_enabled"] is False
    assert result["selected_action_id"] in {"lineup", "injury", "weather"}
    assert sum(row["propensity"] for row in result["candidate_menu"]) == pytest.approx(1.0)
    assert all(row["propensity"] > 0 for row in result["candidate_menu"])
    assert all(row["realized_signed_gain"] is None for row in result["candidate_menu"])


def test_training_policy_rejects_duplicate_action_ids():
    with pytest.raises(ValueError, match="unique"):
        TrainingExplorationPolicy().select(
            [{"action_id": "same"}, {"action_id": "same"}], {"same": 0.1}
        )


def test_training_policy_does_not_turn_negative_prediction_into_label():
    result = TrainingExplorationPolicy(epsilon=1.0, seed=1).select(
        MENU, {"lineup": -0.1, "injury": -0.2, "weather": -0.3}
    )
    assert result["label_status"] == "pending_outcome"
    assert all(row["realized_signed_gain"] is None for row in result["candidate_menu"])
