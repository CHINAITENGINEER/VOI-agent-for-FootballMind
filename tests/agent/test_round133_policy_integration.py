from copy import deepcopy

import pytest

from football_agent.policy_benchmark import normalize_episode, replay_policy
from football_agent.round133_policy_integration import (
    VERSION,
    build_fixture_benchmark_episodes,
    fixture_to_benchmark_episode,
)
from football_agent.scenario_fixture_pack import get_scenario_fixture


def test_fixture_join_hides_author_hypothesis_and_uses_public_menu_only():
    fixture = get_scenario_fixture("qualified-but-strong-performance")
    episode = fixture_to_benchmark_episode(fixture)
    public = episode["public"]
    assert public["candidate_generation_complete"] is True
    assert {row["candidate_id"] for row in public["candidate_menu"]} == {
        row["candidate_id"] for row in fixture["candidate_menu"]
    }
    assert all("expected_value" not in row or row["expected_value"] is None for row in public["candidate_menu"])
    assert "expected_policy" not in repr(episode)
    assert "3-0" not in repr(episode)
    assert episode["integration"]["runtime_reads_sidecar"] is False


def test_join_is_stable_and_policy_gate_defers_unknown_numeric_voi():
    fixture = get_scenario_fixture("draw-secures-group-first")
    first = fixture_to_benchmark_episode(fixture)
    second = fixture_to_benchmark_episode(fixture)
    assert first == second
    assert normalize_episode(first)["candidate_generation_complete"] is True
    result = replay_policy(first, "current_voi_gate")
    assert result["selected_candidates"] == []
    assert result["selection_status"] == "defer_unknown_estimate"
    assert result["final_loss"] == result["baseline_loss"]


def test_sidecar_loss_is_independent_and_singletons_are_the_only_known_sequences():
    episode = fixture_to_benchmark_episode(get_scenario_fixture("key-forward-injury"))
    sidecar = episode["evaluation_sidecar"]
    assert sidecar["label_status"] == "INDEPENDENT_EVALUATOR_GOLD"
    assert sidecar["mechanism_only"] is True
    assert sidecar["complete_sequence_evaluation"] is False
    assert set(sidecar["sequence_losses"]) == {"", "key_forward_availability", "replacement_quality", "weather_check"}
    result = replay_policy(episode, "fixed_all")
    assert result["final_loss"] is None
    assert result["net_utility"] is None


def test_explicit_cost_conversion_remains_marked_synthetic():
    episode = fixture_to_benchmark_episode(
        get_scenario_fixture("weather-low-value"), cost_to_loss_rate=0.01,
    )
    result = replay_policy(episode, "first_query")
    assert result["net_utility"] is not None
    assert episode["integration"]["evidence_origin"] == "synthetic_fixture"
    assert episode["integration"]["world_truth_verified"] is False


def test_build_bundle_is_deterministic_and_does_not_mutate_fixture():
    fixture = get_scenario_fixture("stale-source")
    before = deepcopy(fixture)
    episodes = build_fixture_benchmark_episodes([fixture])
    assert len(episodes) == 1
    assert fixture == before
    assert episodes[0]["integration"]["version"] == VERSION


def test_invalid_budget_and_unknown_candidate_loss_are_rejected():
    fixture = get_scenario_fixture("missing-source")
    with pytest.raises(ValueError, match="budget"):
        fixture_to_benchmark_episode(fixture, budget=-1)
    with pytest.raises(ValueError, match="SIDECAR_CANDIDATE_OUTSIDE_MENU"):
        fixture_to_benchmark_episode(fixture, candidate_losses={"unknown": 0.1})
