import pytest
from itertools import permutations

from football_agent.policy_benchmark import (
    INDEPENDENT_GOLD,
    POLICIES,
    VERSION,
    compare_policies,
    normalize_episode,
    replay_policy,
)


def episode():
    return {
        "schema_version": "value-evaluation-contract-v1",
        "episode_id": "synthetic-qualified-match-1",
        "public": {
            "candidate_generation_complete": True,
            "cost_unit": "resource_units",
            "cost_to_loss_rate": 1.0,
            "candidate_menu": [
                {"candidate_id": "lineup", "cost": .02, "expected_value": .10, "latency_ms": 120, "reliability": .90},
                {"candidate_id": "weather", "cost": .01, "expected_value": .001, "latency_ms": 50, "reliability": .95},
                {"candidate_id": "tactics", "cost": .03, "expected_value": .04, "latency_ms": 200, "reliability": .80},
            ],
            "fixed_subset": ["lineup", "tactics"],
        },
        "evaluation_sidecar": {
            "evaluator_id": "independent-gold-v1",
            "label_status": INDEPENDENT_GOLD,
            "loss_unit": "log_loss",
            "complete_sequence_evaluation": True,
            "baseline_loss": .60,
            "candidate_evaluations": [
                {"candidate_id": "lineup", "evaluator_id": "independent-gold-v1", "baseline_loss": .60, "loss_unit": "log_loss", "decision_loss_after": .20, "useful": True,
                 "reliability": .90, "latency_ms": 110, "contradiction": False,
                 "label_status": INDEPENDENT_GOLD},
                {"candidate_id": "weather", "evaluator_id": "independent-gold-v1", "baseline_loss": .60, "loss_unit": "log_loss", "decision_loss_after": .60, "useful": False,
                 "reliability": .95, "latency_ms": 50, "contradiction": False,
                 "label_status": INDEPENDENT_GOLD},
                {"candidate_id": "tactics", "evaluator_id": "independent-gold-v1", "baseline_loss": .60, "loss_unit": "log_loss", "decision_loss_after": .35, "useful": True,
                 "reliability": .60, "latency_ms": 210, "contradiction": True,
                 "label_status": INDEPENDENT_GOLD},
            ],
            "sequence_losses": {"": .60, "lineup": .20, "lineup>tactics": .15,
                                 "lineup>weather": .20, "tactics>lineup": .15},
        },
        "budget": .05,
    }


def _complete_sequence_losses(value):
    rows = value["evaluation_sidecar"]["candidate_evaluations"]
    losses = value["evaluation_sidecar"]["sequence_losses"]
    ids = [row["candidate_id"] for row in rows]
    for length in range(len(ids) + 1):
        for seq in permutations(ids, length):
            key = ">".join(seq)
            if key not in losses:
                if len(seq) == 1:
                    losses[key] = next(row["decision_loss_after"] for row in rows if row["candidate_id"] == seq[0])
                else:
                    losses[key] = .15 if set(seq) == {"lineup", "tactics"} else (.20 if "lineup" in seq else .60)
    return value


def test_normalize_round132_style_episode_and_keep_sidecar_separate():
    case = normalize_episode(episode())
    assert [item["candidate_id"] for item in case["menu"]] == ["lineup", "weather", "tactics"]
    assert case["baseline_loss"] == pytest.approx(.60)
    assert case["rows"]["lineup"]["loss_after"] == pytest.approx(.20)


@pytest.mark.parametrize("policy,expected", [
    ("stop_immediately", []),
    ("first_query", ["lineup"]),
    ("fixed_subset", ["lineup", "tactics"]),
    ("least_cost", ["weather", "lineup"]),
    ("current_voi_gate", ["lineup", "tactics"]),
])
def test_policy_selection_uses_public_menu_only(policy, expected):
    row = replay_policy(episode(), policy, seed=17)
    assert row["selected_candidates"] == expected
    assert row["evaluator_only"] is True
    assert row["runtime_safe"] is False


def test_oracle_uses_sidecar_only_as_evaluator_upper_bound():
    row = replay_policy(_complete_sequence_losses(episode()), "oracle_upper_bound")
    assert row["selected_candidates"] == ["lineup", "tactics"]
    assert row["final_loss"] == pytest.approx(.15)
    assert row["oracle_optimal"] is True
    assert row["decision_loss_reduction_per_cost"] == pytest.approx(.45 / .05)


def test_compare_reports_required_metrics_and_cost_savings():
    report = compare_policies([_complete_sequence_losses(episode())], seed=3, reliability_threshold=.7)
    assert report["schema_version"] == VERSION
    assert report["policy_order"] == list(POLICIES)
    aggregate = report["aggregate"]["stop_immediately"]
    assert aggregate["decision_loss_reduction_per_cost"] is None
    assert aggregate["stop_quality_rate"] is None
    assert report["aggregate"]["current_voi_gate"]["mean_candidate_precision"] == pytest.approx(1)
    assert report["aggregate"]["fixed_all"]["selective_risk_coverage"] is None
    assert report["contract"]["runtime_policy_reads_evaluator_labels"] is False


def test_random_is_reproducible_and_sidecar_required():
    one = replay_policy(episode(), "random", seed=42)
    two = replay_policy(episode(), "random", seed=42)
    assert one == two
    bad = episode()
    del bad["evaluation_sidecar"]
    with pytest.raises(ValueError, match="sidecar"):
        normalize_episode(bad)


def test_missing_joint_sequence_is_explicitly_unidentifiable_and_oracle_is_excluded():
    value = episode()
    value["budget"] = .06
    row = replay_policy(value, "fixed_all")
    assert row["selected_candidates"] == ["lineup", "weather", "tactics"]
    assert row["final_loss"] is None
    assert row["decision_loss_reduction"] is None
    assert row["regret_vs_oracle"] is None
    assert row["reliability_coverage"] is None
    assert replay_policy(episode(), "oracle_upper_bound")["oracle_utility"] is None


def test_mixed_loss_units_are_rejected_before_scoring():
    bad = episode()
    bad["evaluation_sidecar"]["loss_unit"] = "brier"
    bad["evaluation_sidecar"]["candidate_evaluations"][0]["loss_unit"] = "brier"
    bad["evaluation_sidecar"]["candidate_evaluations"][1]["loss_unit"] = "log_loss"
    with pytest.raises(ValueError, match="loss_unit"):
        normalize_episode(bad)


def test_candidate_and_evaluator_tampering_is_rejected():
    bad = episode()
    bad["evaluation_sidecar"]["candidate_evaluations"].append({"candidate_id": "lineup", "loss_after": .1})
    with pytest.raises(ValueError, match="duplicate evaluator candidate"):
        normalize_episode(bad)


def test_unknown_public_value_defers_and_unqualified_oracle_is_null():
    value = episode()
    value["public"]["candidate_menu"][0]["expected_value"] = None
    row = replay_policy(value, "current_voi_gate")
    assert row["stop_reason"] == "defer_unknown_estimate"
    assert row["value_baseline_type"] == "static_declared_value_baseline"
    oracle = replay_policy(value, "oracle_upper_bound")
    assert oracle["selected_candidates"] is None
    assert oracle["final_loss"] is None
    assert oracle["oracle_utility"] is None


def test_explicit_cost_conversion_and_duplicate_episode_guard():
    value = episode()
    value["public"].pop("cost_to_loss_rate")
    value["public"].pop("cost_unit")
    row = replay_policy(value, "fixed_all")
    assert row["spent_cost"] == pytest.approx(.05)
    assert row["net_utility"] is None
    with pytest.raises(ValueError, match="duplicate episode_id"):
        compare_policies([episode(), episode()], policies=["fixed_all"])
