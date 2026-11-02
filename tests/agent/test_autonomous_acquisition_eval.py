import json

import pytest

from football_agent.autonomous_acquisition_eval import (
    POLICIES,
    ReturnSpec,
    build_scenarios,
    evaluate_all,
    run_policy,
)


def scenario_map():
    return {item.scenario_id: item for item in build_scenarios()}


def test_public_scenario_manifest_excludes_private_response_tape_and_outcome():
    scenario = build_scenarios()[0]
    public = scenario.public_manifest()
    assert public["evaluator_only"] is False
    assert "private_response_labels" not in public
    assert "private_outcome" not in public
    assert all("result" not in query for query in public["queries"])


def test_voi_prefers_high_value_qualification_over_first_weather_query():
    scenario = scenario_map()["qualification-high-value"]
    voi = run_policy(scenario, "voi_gate").as_dict()
    first = run_policy(scenario, "first_query").as_dict()
    assert voi["selected_candidates"] == ["qualification-refresh"]
    assert first["selected_candidates"] == ["weather-refresh"]
    assert voi["final_loss"] < first["final_loss"]
    assert voi["net_utility"] > first["net_utility"]


def test_voi_stops_when_all_remaining_information_has_negative_net_value():
    scenario = scenario_map()["weather-low-value"]
    voi = run_policy(scenario, "voi_gate").as_dict()
    fixed = run_policy(scenario, "fixed_all").as_dict()
    assert voi["query_count"] == 0
    assert voi["selected_candidates"] == []
    assert fixed["query_count"] == 2
    assert voi["acquisition_cost"] < fixed["acquisition_cost"]
    assert "nonpositive net value" in voi["stop_reason"]


def test_late_query_is_audit_only_and_triggers_replanning():
    scenario = scenario_map()["qualification-late"]
    run = run_policy(scenario, "voi_gate").as_dict()
    assert run["statuses"] == ["LATE"]
    assert run["replan_reasons"]
    assert "qualification" not in run["final_context"]
    assert run["transition_events"] == ["INITIALIZED", "QUERY_SELECTED", "REPLANNED", "STOPPED"]


def test_ambiguous_query_keeps_context_unchanged_and_records_replan():
    scenario = scenario_map()["qualification-ambiguous"]
    run = run_policy(scenario, "voi_gate").as_dict()
    assert run["statuses"] == ["AMBIGUOUS"]
    assert run["replan_reasons"]
    assert "qualification" not in run["final_context"]
    assert "REPLANNED" in run["transition_events"]


def test_fixed_policy_pays_for_low_value_information_and_reads_all_candidates():
    scenario = scenario_map()["qualification-high-value"]
    fixed = run_policy(scenario, "fixed_all").as_dict()
    assert fixed["selected_candidates"] == ["weather-refresh", "qualification-refresh"]
    assert fixed["query_count"] == 2
    assert fixed["acquisition_cost"] == pytest.approx(.02)
    assert fixed["final_context"]["qualification"] == "already_qualified"


def test_policy_traces_keep_result_truth_in_explicit_evaluator_section():
    scenario = scenario_map()["qualification-high-value"]
    record = run_policy(scenario, "voi_gate").as_dict()
    policy_view = {key: value for key, value in record.items() if key != "evaluator_only"}
    encoded = json.dumps(policy_view, ensure_ascii=False).lower()
    assert "true_outcome" not in encoded
    assert "sampled_result" not in encoded
    assert record["hidden_result_included"] is False
    assert record["evaluator_only"]["true_outcome"] == "home_win"


def test_all_scenarios_have_three_policy_traces_and_json_serialize():
    report = evaluate_all()
    assert len(report["scenarios"]) == 4
    assert report["policy_order"] == list(POLICIES)
    assert set(report["scenarios"][0]["policies"]) == set(POLICIES)
    json.dumps(report, ensure_ascii=False)


def test_return_spec_rejects_evaluator_only_payload_fields():
    with pytest.raises(ValueError, match="evaluator-only"):
        ReturnSpec(
            "bad", {"outcome": "home_win"}, {}, None, ("raw:bad",),
            captured_at="2026-09-29T10:01:00Z",
        )


def test_each_policy_finishes_with_stop_and_contiguous_execution_entries():
    for scenario in build_scenarios():
        for policy in POLICIES:
            run = run_policy(scenario, policy).as_dict()
            assert run["transition_events"][0] == "INITIALIZED"
            assert run["transition_events"][-1] == "STOPPED"
            assert run["execution_entries"] == run["query_count"] * 2
