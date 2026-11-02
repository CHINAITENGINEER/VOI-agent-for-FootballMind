import math

import pytest

from football_agent.decision_value import ObservationTool, FiniteDecisionModel, synthetic_scenarios
from football_agent.policy_replay import (
    POLICY_AGENT, POLICY_BASE, POLICY_FIXED, POLICY_FULL, compare_policies, replay_policy,
)


def simple_model():
    return FiniteDecisionModel(
        {"a": (.8, .1, .1), "b": (.1, .1, .8)},
        {"a": .5, "b": .5},
        {
            "cheap": ObservationTool(("a", "b"), {"a": (1, 0), "b": (0, 1)}, .01),
            "expensive": ObservationTool(("a", "b"), {"a": (1, 0), "b": (0, 1)}, .20),
        },
    )


def test_same_seed_matches_hidden_streams_but_trace_has_no_hidden_result():
    model = synthetic_scenarios()["draw_sufficient_tactics_unknown"]["model"]
    rows = [replay_policy(model, policy, seed=31, budget=.08) for policy in (POLICY_AGENT, POLICY_FIXED, POLICY_FULL, POLICY_BASE)]
    assert len({(row["evaluator_only"]["sampled_hypothesis"], row["evaluator_only"]["sampled_result"]) for row in rows}) == 1
    for row in rows:
        assert "sampled_result" not in row["trace"]
        assert all("result" not in event for event in row["trace"])


def test_autonomous_policy_recomputes_value_and_can_stop_after_contextual_update():
    model = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    row = replay_policy(model, POLICY_AGENT, seed=7, budget=.08)
    assert row["trace"][0]["selected_action"] == "lineup"
    assert row["query_count"] in (1, 2)
    assert row["trace"][-1]["selected_action"] is None
    assert row["trace"][-1]["selection_basis"] in {"no_positive_expected_net_value", "horizon_exhausted"}


def test_fixed_and_full_controls_share_order_but_have_distinct_action_caps():
    model = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    fixed = replay_policy(model, POLICY_FIXED, seed=7, budget=.08)
    full = replay_policy(model, POLICY_FULL, seed=7, budget=.08)
    assert fixed["query_count"] == 1
    assert full["query_count"] == 2
    assert fixed["action_order"] == full["action_order"] == ["lineup", "reserve_quality"]


def test_full_budget_skips_unaffordable_early_action_and_uses_later_affordable_tool():
    model = simple_model()
    row = replay_policy(model, POLICY_FULL, seed=4, budget=.02, action_order=("expensive", "cheap"))
    assert row["query_count"] == 1
    assert row["trace"][0]["selected_action"] == "cheap"
    assert row["stop_reason"] == "budget_exhausted"


def test_base_stop_has_zero_acquisition_and_preserves_base_prediction():
    model = simple_model()
    row = replay_policy(model, POLICY_BASE, seed=4, budget=.02)
    assert row["query_count"] == 0
    assert row["spent"] == 0
    assert row["final_prediction"] == pytest.approx(list(model.prediction(model.initial_state())))


def test_compare_policies_emits_matched_contract_and_evaluator_only_summary():
    model = synthetic_scenarios()["complementary_xor_reference"]["model"]
    result = compare_policies(model, seeds=range(5), budget=.04, horizon=2)
    assert result["schema_version"] == "matched-policy-replay-v1"
    assert result["visibility"] == "evaluator_only"
    contract = result["comparison_contract"]
    assert contract["matched_hidden_streams"] is True
    assert contract["eventual_match_result_is_policy_input"] is False
    assert set(result["summaries"]) == {POLICY_AGENT, POLICY_FIXED, POLICY_FULL, POLICY_BASE}
    assert result["summaries"][POLICY_BASE]["mean_spent"] == 0
    assert len(result["episodes"][POLICY_AGENT]) == 5


@pytest.mark.parametrize("bad_policy", ["llm_guess", "", "STOP"])
def test_unknown_policy_is_rejected(bad_policy):
    with pytest.raises(ValueError, match="Unknown policy"):
        replay_policy(simple_model(), bad_policy, seed=1, budget=.1)


def test_invalid_action_order_must_cover_registered_menu():
    with pytest.raises(ValueError, match="cover every registered"):
        replay_policy(simple_model(), POLICY_FULL, seed=1, budget=.1, action_order=("cheap",))
