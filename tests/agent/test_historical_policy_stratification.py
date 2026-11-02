import copy

import pytest

from football_agent.historical_policy_stratification import (
    DEFAULT_CONTEXT_FIELDS,
    VERSION,
    compare_by_context,
)
from football_agent.value_evaluation_contract import INDEPENDENT_GOLD


def make_episode(episode_id, context=None, *, stop_after_lineup=False):
    sequence = {
        "": 0.60, "lineup": 0.20, "weather": 0.60, "tactics": 0.35,
        "lineup>weather": 0.20, "weather>lineup": 0.20,
        "lineup>tactics": 0.15 if not stop_after_lineup else 0.19,
        "tactics>lineup": 0.15 if not stop_after_lineup else 0.19,
        "weather>tactics": 0.60, "tactics>weather": 0.35,
        "lineup>weather>tactics": 0.15, "lineup>tactics>weather": 0.15,
        "weather>lineup>tactics": 0.15, "weather>tactics>lineup": 0.15,
        "tactics>lineup>weather": 0.15, "tactics>weather>lineup": 0.35,
    }
    menu = [
        {"candidate_id": "lineup", "cost": 0.02, "expected_value": 0.10, "reliability": 0.9},
        {"candidate_id": "weather", "cost": 0.01, "expected_value": 0.001, "reliability": 0.95},
        {"candidate_id": "tactics", "cost": 0.03, "expected_value": 0.02 if stop_after_lineup else 0.04, "reliability": 0.8},
    ]
    losses = {"lineup": 0.20, "weather": 0.60, "tactics": 0.35}
    return {
        "episode_id": episode_id,
        "public": {
            "prediction_cutoff": "2026-09-30T14:00:00Z",
            "kickoff": "2026-09-30T15:00:00Z",
            "context": context or {},
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
                {"candidate_id": key, "evaluator_id": "independent-gold-v1",
                 "baseline_loss": 0.60, "loss_unit": "log_loss", "decision_loss_after": value,
                 "useful": value < 0.60, "reliability": rel, "label_status": INDEPENDENT_GOLD}
                for (key, value, rel) in (("lineup", losses["lineup"], .9),
                                           ("weather", losses["weather"], .95),
                                           ("tactics", losses["tactics"], .8))
            ],
            "sequence_losses": sequence,
        },
    }


def test_groups_by_public_match_context_and_keeps_global_report():
    episodes = [
        make_episode("qualified", {"qualification_state": "already_qualified", "tactical_layout": "attacking"}),
        make_episode("must-win", {"qualification_state": "must_win", "tactical_layout": "defensive"}),
    ]
    report = compare_by_context(episodes)
    assert report["schema_version"] == VERSION
    assert report["policy_order"] == ["autonomous_voi", "fixed_all", "fixed_order"]
    assert report["episode_ids"] == ["qualified", "must-win"]
    assert "qualification_state=already_qualified" in report["groups"]
    assert "qualification_state=must_win" in report["groups"]
    assert report["groups"]["qualification_state=must_win"]["episode_count"] == 1
    assert report["global"]["contract"]["matched_historical_episode_set"] is True


def test_context_stratum_preserves_autonomous_stop_behavior():
    episodes = [
        make_episode("high-rotation", {"rotation_risk": "high"}, stop_after_lineup=True),
        make_episode("low-rotation", {"rotation_risk": "low"}),
    ]
    report = compare_by_context(episodes, context_fields=("rotation_risk",))
    high = report["groups"]["rotation_risk=high"]["summary"]
    low = report["groups"]["rotation_risk=low"]["summary"]
    assert high["aggregate"]["autonomous_voi"]["mean_query_count"] == pytest.approx(1.0)
    assert high["aggregate"]["fixed_all"]["mean_query_count"] == pytest.approx(2.0)
    assert high["paired_deltas"]["autonomous_voi"]["mean_resource_cost_delta"] < 0
    assert low["aggregate"]["autonomous_voi"]["mean_query_count"] >= 1


def test_missing_context_is_an_explicit_unknown_group():
    report = compare_by_context([make_episode("unknown")], context_fields=("qualification_state",))
    group = report["groups"]["qualification_state=unknown"]
    assert group["episode_ids"] == ["unknown"]
    assert report["contract"]["unknown_context_is_explicit_group"] is True


def test_custom_context_fields_must_be_unique_and_nonempty():
    value = make_episode("one", {"x": "a"})
    with pytest.raises(ValueError, match="nonempty and unique"):
        compare_by_context([value], context_fields=("x", "x"))
    with pytest.raises(ValueError, match="nonempty string"):
        compare_by_context([value], context_fields=("",))


def test_context_values_must_be_scalar():
    value = make_episode("bad", {"qualification_state": ["must_win"]})
    with pytest.raises(ValueError, match="must be scalar"):
        compare_by_context([value], context_fields=("qualification_state",))


def test_context_group_uses_same_policy_set_and_paired_episode_ids():
    report = compare_by_context([
        make_episode("a", {"tactical_layout": "defensive"}),
        make_episode("b", {"tactical_layout": "defensive"}),
    ], context_fields=("tactical_layout",))
    summary = report["groups"]["tactical_layout=defensive"]["summary"]
    assert set(summary["aggregate"]) == {"autonomous_voi", "fixed_all", "fixed_order"}
    assert [item["episode_id"] for item in summary["paired_deltas"]["fixed_order"]["episodes"]] == ["a", "b"]


def test_group_summary_keeps_unknown_metrics_explicit():
    value = make_episode("unknown-metric", {"rotation_risk": "unknown"})
    value["evaluation_sidecar"]["sequence_losses"].pop("lineup>tactics")
    value["evaluation_sidecar"]["sequence_losses"].pop("tactics>lineup")
    report = compare_by_context([value], context_fields=("rotation_risk",))
    summary = report["groups"]["rotation_risk=unknown"]["summary"]
    assert summary["aggregate"]["fixed_all"]["unknown_final_loss_count"] == 1
    assert summary["aggregate"]["fixed_all"]["mean_final_loss"] is None


def test_contract_marks_sidecar_and_runtime_boundaries():
    report = compare_by_context([make_episode("one", {"dominant_uncertainty": "lineup"})])
    contract = report["contract"]
    assert contract["group_assignment_uses_public_context_only"] is True
    assert contract["runtime_policy_reads_evaluator_labels"] is False
    assert contract["evaluator_sidecar_required"] is True
    assert contract["real_source_requests"] == 0
    assert contract["real_model_calls"] == 0
    assert contract["numeric_voi_training_eligible"] is False
