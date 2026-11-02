import copy
from itertools import permutations

import pytest

from football_agent.historical_policy_comparison import (
    POLICIES,
    VERSION,
    compare_historical_episodes,
    runtime_projection,
)
from football_agent.value_evaluation_contract import INDEPENDENT_GOLD


def episode(episode_id="historic-1"):
    menu = [
        {"candidate_id": "lineup", "cost": 0.02, "expected_value": 0.10, "latency_ms": 120, "reliability": 0.9},
        {"candidate_id": "weather", "cost": 0.01, "expected_value": 0.001, "latency_ms": 50, "reliability": 0.95},
        {"candidate_id": "tactics", "cost": 0.03, "expected_value": 0.04, "latency_ms": 200, "reliability": 0.8},
    ]
    sequence_losses = {"": 0.60, "lineup": 0.20, "weather": 0.60, "tactics": 0.35,
                       "lineup>weather": 0.20, "weather>lineup": 0.20,
                       "lineup>tactics": 0.15, "tactics>lineup": 0.15,
                       "weather>tactics": 0.60, "tactics>weather": 0.35,
                       "lineup>weather>tactics": 0.15, "lineup>tactics>weather": 0.15,
                       "weather>lineup>tactics": 0.15, "weather>tactics>lineup": 0.15,
                       "tactics>lineup>weather": 0.15, "tactics>weather>lineup": 0.35}
    rows = [
        ("lineup", 0.20, True, 0.90),
        ("weather", 0.60, False, 0.95),
        ("tactics", 0.35, True, 0.80),
    ]
    return {
        "episode_id": episode_id,
        "public": {
            "prediction_cutoff": "2026-09-30T14:00:00Z",
            "kickoff": "2026-09-30T15:00:00Z",
            "candidate_generation_complete": True,
            "cost_unit": "resource_units",
            "cost_to_loss_rate": 1.0,
            "candidate_menu": menu,
        },
        "budget": 0.05,
        "evaluation_sidecar": {
            "evaluator_id": "independent-gold-v1",
            "label_status": INDEPENDENT_GOLD,
            "loss_unit": "log_loss",
            "complete_sequence_evaluation": True,
            "baseline_loss": 0.60,
            "candidate_evaluations": [
                {"candidate_id": candidate, "evaluator_id": "independent-gold-v1",
                 "baseline_loss": 0.60, "loss_unit": "log_loss", "decision_loss_after": loss,
                 "useful": useful, "reliability": reliability, "label_status": INDEPENDENT_GOLD}
                for candidate, loss, useful, reliability in rows
            ],
            "sequence_losses": sequence_losses,
        },
    }


def test_compares_named_arms_on_one_matched_asof_episode():
    report = compare_historical_episodes([episode()])
    assert report["schema_version"] == VERSION
    assert report["policy_order"] == list(POLICIES)
    assert all(report["contract"][key] for key in (
        "matched_historical_episode_set", "actor_uses_public_snapshot_only",
        "evaluator_sidecar_required", "unknown_metrics_are_not_zero",
    ))
    assert report["contract"]["runtime_policy_reads_evaluator_labels"] is False
    assert set(report["rows"]) == set(POLICIES)
    assert report["rows"]["autonomous_voi"][0]["policy_id"] == "autonomous_voi"
    assert report["rows"]["fixed_order"][0]["selected_candidates"] == ["lineup", "weather"]
    assert report["aggregate"]["fixed_order"]["mean_resource_cost"] == pytest.approx(0.03)


def test_autonomous_and_fixed_all_are_paired_against_same_episode():
    report = compare_historical_episodes([episode(), episode("historic-2")])
    for policy in POLICIES:
        assert report["aggregate"][policy]["episode_count"] == 2
        assert [row["episode_id"] for row in report["rows"][policy]] == ["historic-1", "historic-2"]
    delta = report["paired_deltas"]["autonomous_voi"]["episodes"]
    assert len(delta) == 2
    assert delta[0]["episode_id"] == "historic-1"


def test_pareto_frontier_contains_only_observed_points():
    report = compare_historical_episodes([episode()])
    frontier = report["resource_risk_pareto"]
    assert frontier["objective"] == "minimize_mean_resource_cost_and_mean_final_loss"
    assert set(frontier["points"]) == set(POLICIES)
    assert all(name in POLICIES for name in frontier["frontier"])


def test_autonomous_voi_can_stop_when_fixed_all_would_keep_spending():
    value = episode("conditional-stop")
    value["public"]["candidate_menu"][2]["expected_value"] = 0.02
    value["evaluation_sidecar"]["sequence_losses"]["lineup>tactics"] = 0.19
    value["evaluation_sidecar"]["sequence_losses"]["tactics>lineup"] = 0.19
    report = compare_historical_episodes([value])
    autonomous = report["rows"]["autonomous_voi"][0]
    fixed = report["rows"]["fixed_all"][0]
    assert autonomous["selected_candidates"] == ["lineup"]
    assert fixed["selected_candidates"] == ["lineup", "tactics"]
    assert report["aggregate"]["autonomous_voi"]["mean_resource_cost"] < report["aggregate"]["fixed_all"]["mean_resource_cost"]


def test_runtime_projection_does_not_copy_evaluator_metrics():
    projection = runtime_projection(compare_historical_episodes([episode()]))
    assert projection["truth_in_policy_view"] is False
    assert projection["hidden_result_included"] is False
    assert "final_loss" not in str(projection)


@pytest.mark.parametrize("field", ["prediction_cutoff", "kickoff"])
def test_asof_clock_is_required(field):
    value = episode()
    value["public"].pop(field)
    with pytest.raises(ValueError, match="historical episode requires"):
        compare_historical_episodes([value])


def test_cutoff_after_kickoff_is_rejected():
    value = episode()
    value["public"]["prediction_cutoff"] = "2026-10-01T00:00:00Z"
    with pytest.raises(ValueError, match="must not follow kickoff"):
        compare_historical_episodes([value])


def test_public_outcome_leak_is_rejected():
    value = episode()
    value["public"]["outcome"] = 1
    with pytest.raises(ValueError, match="evaluator fields"):
        compare_historical_episodes([value])


def test_nested_public_outcome_leak_is_rejected():
    value = episode()
    value["public"]["known"] = {"lineup": {"result": "leak"}}
    with pytest.raises(ValueError, match="evaluator fields"):
        compare_historical_episodes([value])


def test_duplicate_historical_episode_id_is_rejected():
    with pytest.raises(ValueError, match="duplicate historical episode_id"):
        compare_historical_episodes([episode(), copy.deepcopy(episode())])


def test_fixed_order_follows_public_order_when_budget_changes():
    value = episode()
    value["budget"] = 0.02
    report = compare_historical_episodes([value])
    assert report["rows"]["fixed_order"][0]["selected_candidates"] == ["lineup"]


def test_no_network_model_or_numeric_voi_claim():
    report = compare_historical_episodes([episode()])
    assert report["contract"]["real_source_requests"] == 0
    assert report["contract"]["real_model_calls"] == 0
    assert report["contract"]["numeric_voi_training_eligible"] is False
