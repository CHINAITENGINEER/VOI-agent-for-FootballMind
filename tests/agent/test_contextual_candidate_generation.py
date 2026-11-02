import copy

import pytest

from football_agent.contextual_candidate_generation import (
    VERSION,
    apply_contextual_candidate_order,
    compare_contextual_candidates,
    generate_contextual_candidates,
)
from football_agent.value_evaluation_contract import INDEPENDENT_GOLD


def episode(context=None, episode_id="candidate-episode"):
    menu = [
        {"candidate_id": "weather", "cost": 0.01, "expected_value": 0.001,
         "context_affinity": {"tactical_layout": ["defensive"]}},
        {"candidate_id": "lineup", "cost": 0.02, "expected_value": 0.10,
         "context_affinity": {"rotation_risk": ["high"], "dominant_uncertainty": ["lineup"]}},
        {"candidate_id": "tactics", "cost": 0.03, "expected_value": 0.04,
         "context_affinity": {"qualification_state": ["must_win", "draw_sufficient"]},
         "declared_priority": 1},
    ]
    losses = {"": .60, "weather": .60, "lineup": .20, "tactics": .35,
              "weather>lineup": .20, "lineup>weather": .20,
              "weather>tactics": .35, "tactics>weather": .35,
              "lineup>tactics": .15, "tactics>lineup": .15,
              "weather>lineup>tactics": .15, "weather>tactics>lineup": .15,
              "lineup>weather>tactics": .15, "lineup>tactics>weather": .15,
              "tactics>weather>lineup": .15, "tactics>lineup>weather": .15}
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
        "budget": .05,
        "evaluation_sidecar": {
            "evaluator_id": "independent-gold-v1",
            "label_status": INDEPENDENT_GOLD,
            "loss_unit": "log_loss",
            "complete_sequence_evaluation": True,
            "baseline_loss": .60,
            "candidate_evaluations": [
                {"candidate_id": key, "evaluator_id": "independent-gold-v1", "baseline_loss": .60,
                 "loss_unit": "log_loss", "decision_loss_after": after, "useful": after < .60,
                 "label_status": INDEPENDENT_GOLD}
                for key, after in (("weather", .60), ("lineup", .20), ("tactics", .35))
            ],
            "sequence_losses": losses,
        },
    }


def test_generation_prioritizes_context_matches_and_explains_them():
    trace = generate_contextual_candidates(episode({
        "qualification_state": "must_win",
        "tactical_layout": "defensive",
        "rotation_risk": "high",
        "dominant_uncertainty": "lineup",
    }))
    assert trace["schema_version"] == VERSION
    assert trace["candidate_order"] == ["lineup", "tactics", "weather"]
    assert trace["candidates"][0]["matched_context"] == ["dominant_uncertainty=lineup", "rotation_risk=high"]
    assert trace["evaluator_fields_used"] is False


def test_generation_changes_with_public_match_state():
    must_win = generate_contextual_candidates(episode({"qualification_state": "must_win"}))
    defensive = generate_contextual_candidates(episode({"tactical_layout": "defensive"}))
    assert must_win["candidate_order"][0] == "tactics"
    assert defensive["candidate_order"][0] == "weather"


def test_unknown_context_is_explicit_and_does_not_invent_match():
    trace = generate_contextual_candidates(episode({"rotation_risk": None}))
    lineup = next(item for item in trace["candidates"] if item["candidate_id"] == "lineup")
    assert lineup["candidate_generation_status"] == "context_unknown"
    assert lineup["unknown_context_fields"] == ["dominant_uncertainty", "rotation_risk"]


def test_apply_reorders_public_menu_and_preserves_sidecar():
    value = episode({"qualification_state": "must_win"})
    before = copy.deepcopy(value["evaluation_sidecar"])
    updated, trace = apply_contextual_candidate_order(value)
    assert [item["candidate_id"] for item in updated["public"]["candidate_menu"]] == ["tactics", "lineup", "weather"]
    assert updated["evaluation_sidecar"] == before
    assert updated["public"]["candidate_generation_trace"]["evaluator_fields_used"] is False
    assert trace["candidate_order"] == ["tactics", "lineup", "weather"]


def test_contextual_comparison_returns_policy_report_and_generation_trace():
    report = compare_contextual_candidates([
        episode({"qualification_state": "must_win"}, "must-win"),
        episode({"rotation_risk": "high", "dominant_uncertainty": "lineup"}, "rotation"),
    ])
    assert report["schema_version"] == VERSION
    assert report["comparison"]["episode_ids"] == ["must-win", "rotation"]
    assert len(report["candidate_generation"]) == 2
    assert report["comparison"]["policy_order"] == ["autonomous_voi", "fixed_all", "fixed_order"]
    assert report["contract"]["candidate_generation_uses_public_context_only"] is True


def test_candidate_ids_are_unique():
    value = episode()
    value["public"]["candidate_menu"].append(copy.deepcopy(value["public"]["candidate_menu"][0]))
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        generate_contextual_candidates(value)


def test_affinity_values_and_context_values_are_scalar():
    value = episode({"qualification_state": ["must_win"]})
    with pytest.raises(ValueError, match="must be scalar"):
        generate_contextual_candidates(value)
    value = episode()
    value["public"]["candidate_menu"][0]["context_affinity"] = {"tactical_layout": "defensive"}
    with pytest.raises(ValueError, match="scalar lists"):
        generate_contextual_candidates(value)


def test_declared_priority_is_finite():
    value = episode()
    value["public"]["candidate_menu"][0]["declared_priority"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        generate_contextual_candidates(value)


def test_generation_never_reads_evaluator_sidecar():
    value = episode({"qualification_state": "must_win"})
    del value["evaluation_sidecar"]
    trace = generate_contextual_candidates(value)
    assert trace["candidate_order"] == ["tactics", "lineup", "weather"]
