from football_agent.training_exploration_adapter import TrainingExplorationAdapter


MENU = [
    {"action_id": "lineup"},
    {"action_id": "injury"},
]


def test_adapter_is_disabled_by_default_and_does_not_select():
    result = TrainingExplorationAdapter().select_for_collection(
        MENU, {"lineup": 0.1, "injury": -0.1}
    )
    assert result["enabled"] is False
    assert result["selected_action_id"] is None
    assert result["selection_authority"] == "none"


def test_enabled_adapter_emits_propensity_sidecar_only():
    result = TrainingExplorationAdapter(enabled=True, epsilon=1.0, seed=4).select_for_collection(
        MENU, {"lineup": 0.1, "injury": -0.1}
    )
    assert result["enabled"] is True
    assert result["selection_authority"] == "data_collection_only"
    assert result["network_requests"] == 0
    assert result["f_invoked"] is False
    assert result["selected_action_id"] in {"lineup", "injury"}
    assert sum(row["propensity"] for row in result["candidate_menu"]) == 1.0
    assert all(row["realized_signed_gain"] is None for row in result["candidate_menu"])
