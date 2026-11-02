import json

import pytest

from football_agent.autonomous_acquisition_eval import build_scenarios
from football_agent.autonomous_policy_comparison import (
    COMPARISON_POLICIES,
    VERSION,
    compare_scenarios,
    public_projection,
    reviewer_projection,
    runtime_projection,
)


@pytest.fixture(scope="module")
def report():
    return compare_scenarios()


def test_comparison_contains_requested_policies_and_current_schema(report):
    assert report["schema_version"] == VERSION
    assert report["policy_order"] == list(COMPARISON_POLICIES)
    assert all(set(row["policies"]) == set(COMPARISON_POLICIES) for row in report["scenarios"])


def test_hindsight_oracle_is_an_evaluator_upper_bound(report):
    rows = {row["scenario"]["scenario_id"]: row for row in report["scenarios"]}
    high = rows["qualification-high-value"]["policies"]
    assert high["oracle_upper_bound"]["selected_candidates"] == ["qualification-refresh"]
    assert high["oracle_upper_bound"]["regret_vs_oracle"] == pytest.approx(0.0)
    assert high["voi_gate"]["regret_vs_oracle"] == pytest.approx(0.0)
    assert high["heuristic"]["regret_vs_oracle"] > 0


def test_oracle_can_choose_zero_queries_when_information_has_no_realized_gain(report):
    rows = {row["scenario"]["scenario_id"]: row for row in report["scenarios"]}
    low = rows["weather-low-value"]["policies"]["oracle_upper_bound"]
    assert low["selected_candidates"] == []
    assert low["query_count"] == 0
    assert low["acquisition_cost"] == 0


def test_late_and_ambiguous_paths_are_scored_without_promoting_hidden_context(report):
    rows = {row["scenario"]["scenario_id"]: row for row in report["scenarios"]}
    for scenario_id in ("qualification-late", "qualification-ambiguous"):
        trace = rows[scenario_id]["policies"]["voi_gate"]
        assert trace["hidden_result_included"] is False
        assert trace["statuses"]
        assert "qualification" not in rows[scenario_id]["scenario"]["initial_context"]


def test_aggregate_contains_risk_calibration_cost_stopping_and_regret_metrics(report):
    aggregate = report["aggregate"]
    expected = {
        "mean_final_brier", "mean_log_loss", "mean_confidence_error", "top_class_accuracy",
        "mean_acquisition_cost", "mean_query_count", "mean_query_fraction", "zero_query_rate",
        "cost_savings_vs_fixed_all", "brier_gain_per_cost", "mean_regret_vs_oracle",
        "oracle_optimal_rate", "calibration_bins",
    }
    assert all(expected <= set(aggregate[policy]) for policy in COMPARISON_POLICIES)
    assert aggregate["fixed_all"]["mean_query_fraction"] == pytest.approx(1.0)
    assert aggregate["voi_gate"]["mean_acquisition_cost"] < aggregate["fixed_all"]["mean_acquisition_cost"]
    assert aggregate["voi_gate"]["mean_regret_vs_oracle"] >= 0


def test_pareto_frontier_is_explicit_and_uses_cost_and_brier(report):
    pareto = report["pareto"]
    assert pareto["objective"] == "minimize_mean_acquisition_cost_and_mean_final_brier"
    assert set(pareto["frontier"]) <= set(COMPARISON_POLICIES)
    assert set(pareto["dominated_by"]) == set(COMPARISON_POLICIES)
    assert set(pareto["points"]) == set(COMPARISON_POLICIES)


def test_public_projection_has_no_private_truth_or_response_tape(report):
    projection = public_projection(report)
    encoded = json.dumps(projection, ensure_ascii=False).lower()
    for forbidden in ('"true_outcome"', '"private_outcome"', '"private_response_labels"', '"outcome"'):
        assert forbidden not in encoded
    assert projection["hidden_result_included"] is False
    assert projection["projection_scope"] == "reviewer_evaluation_only"
    assert projection["runtime_safe"] is False
    assert report["evaluator_only"]["truth_in_policy_view"] is False


def test_reviewer_scorecard_is_separate_from_runtime_public_projection(report):
    reviewer = reviewer_projection(report)
    runtime = runtime_projection(report)
    assert reviewer["runtime_safe"] is False
    assert runtime["runtime_safe"] is True
    assert runtime["projection_scope"] == "runtime_public_context"
    assert "aggregate" not in runtime
    assert "pareto" not in runtime
    assert all("policies" not in row for row in runtime["scenarios"])
    encoded = json.dumps(runtime, ensure_ascii=False).lower()
    for forbidden in ('"final_brier"', '"log_loss"', '"regret_vs_oracle"', '"oracle_upper_bound"'):
        assert forbidden not in encoded


def test_empty_scenario_collection_is_rejected():
    with pytest.raises(ValueError, match="at least one scenario"):
        compare_scenarios(())


def test_policy_run_now_exposes_prediction_for_evaluator_scoring():
    scenario = build_scenarios()[0]
    from football_agent.autonomous_acquisition_eval import run_policy

    record = run_policy(scenario, "voi_gate").as_dict()
    assert len(record["final_prediction"]) == 3
    assert sum(record["final_prediction"]) == pytest.approx(1.0)
