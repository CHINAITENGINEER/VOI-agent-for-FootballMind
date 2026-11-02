from copy import deepcopy
import pytest

from football_agent.semantic_scenario_model import SemanticScenarioModel
from tools.analyze_semantic_scenario_costs import analyze_run, render_report


def saved_run():
    proposal = {"schema": "semantic-scenario-proposal-v1", "assumptions": ["Constructed test only"],
        "hypotheses": [{"id": "a", "description": "a", "prior": .5, "outcome_probabilities": [.8, .1, .1]},
                       {"id": "b", "description": "b", "prior": .5, "outcome_probabilities": [.2, .3, .5]}],
        "actions": [{"action_id": action, "question": action, "observation_group": action,
                     "likelihoods": [{"hypothesis_id": "a", "probabilities": [.8, .1, .1]},
                                     {"hypothesis_id": "b", "probabilities": [.1, .8, .1]}]}
                    for action in ("read", "search_read")]}
    model = SemanticScenarioModel(proposal, [.5, .2, .3], action_request_costs={"search_read": 2},
                                   observation_group_policy="first_attempt")
    return {"run_id": "constructed-test", "configuration": {"request_budget": 3},
            "cycles": [{"index": 0, "model": model.manifest(), "observations": [],
                        "remaining_requests": 3, "requests_spent": 0,
                        "decision": model.evaluate([], 3)}]}


def test_cost_sensitivity_preserves_frozen_state_and_scales_threshold_by_requests():
    run = saved_run()
    before = deepcopy(run)
    result = analyze_run(run)
    assert run == before
    assert result["http_requests"] == result["semantic_calls"] == 0
    assert result["replay_verification"]["status"] == "passed"
    rows = {row["action_id"]: row for row in result["action_thresholds"]}
    assert rows["read"]["gross_lower"] == pytest.approx(rows["search_read"]["gross_lower"])
    assert rows["read"]["robust_break_even_cost_per_request"] == pytest.approx(
        2 * rows["search_read"]["robust_break_even_cost_per_request"])
    for variant in result["cost_variants"]:
        assert variant["prediction_nominal"] == pytest.approx([.5, .2, .3])
        for candidate in variant["candidates"]:
            assert candidate["lower"] == pytest.approx(candidate["gross_lower"] -
                variant["request_loss_cost"] * candidate["request_cost"])
    assert "不用于选择新的成本" in render_report(result)


def test_crossing_derived_threshold_changes_only_the_decision_not_assumed_gross_value():
    run = saved_run()
    preliminary = analyze_run(run)
    maximum = max(row["robust_break_even_cost_per_request"] for row in preliminary["action_thresholds"])
    result = analyze_run(run, costs=(maximum / 2, maximum * 2))
    assert result["cost_variants"][0]["selected_action_id"] == "read"
    assert result["cost_variants"][1]["selected_action_id"] is None
    assert result["cost_variants"][1]["stop_reason"] == "no_positive_declared_lower_net_value"


def test_source_decision_must_replay_and_cost_grid_is_validated():
    run = saved_run()
    run["cycles"][0]["decision"]["prediction_nominal"] = [.3, .3, .4]
    with pytest.raises(ValueError, match="replay mismatch"):
        analyze_run(run)
    with pytest.raises(ValueError, match="nonnegative finite"):
        analyze_run(saved_run(), costs=(0., float("nan")))
