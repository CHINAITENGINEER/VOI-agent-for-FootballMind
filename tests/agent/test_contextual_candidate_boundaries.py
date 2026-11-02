"""Black-box adversarial checks for public-context proposal generation.

These tests deliberately vary evaluator labels and proposal order while keeping
the actor's declared value estimates fixed. Proposal relevance is not numeric
VOI, and a comparator must not silently improve its fixed-order control arm.
"""
from copy import deepcopy
import json

import pytest

from football_agent.contextual_candidate_generation import (
    compare_contextual_candidates,
    generate_contextual_candidates,
)
from football_agent.historical_policy_comparison import compare_historical_episodes
from football_agent.value_evaluation_contract import INDEPENDENT_GOLD


def historical_episode():
    return {
        "episode_id": "contextual-boundary",
        "budget": .02,
        "public": {
            "prediction_cutoff": "2026-09-30T14:00:00Z",
            "kickoff": "2026-09-30T15:00:00Z",
            "context": {"dominant_uncertainty": "lineup"},
            "candidate_generation_complete": True,
            "cost_unit": "resource_units",
            "cost_to_loss_rate": 1.0,
            "candidate_menu": [
                {"candidate_id": "weather", "cost": .02, "expected_value": .01,
                 "context_affinity": {"dominant_uncertainty": ["weather"]}},
                {"candidate_id": "lineup", "cost": .02, "expected_value": .10,
                 "context_affinity": {"dominant_uncertainty": ["lineup"]}},
            ],
        },
        "evaluation_sidecar": {
            "evaluator_id": "boundary-evaluator",
            "label_status": INDEPENDENT_GOLD,
            "loss_unit": "log_loss",
            "complete_sequence_evaluation": True,
            "baseline_loss": .60,
            "candidate_evaluations": [
                {"candidate_id": cid, "evaluator_id": "boundary-evaluator",
                 "label_status": INDEPENDENT_GOLD, "loss_unit": "log_loss",
                 "baseline_loss": .60, "decision_loss_after": loss}
                for cid, loss in (("weather", .60), ("lineup", .20))
            ],
            "sequence_losses": {"": .60, "weather": .60, "lineup": .20,
                                "weather>lineup": .20, "lineup>weather": .20},
        },
    }


def test_evaluator_mutation_and_removal_do_not_change_generated_proposals():
    value = historical_episode()
    before = generate_contextual_candidates(value)
    altered = deepcopy(value)
    altered["evaluation_sidecar"] = {"outcome": "away", "decision_loss_after": 1000,
                                     "context": {"dominant_uncertainty": "weather"}}
    assert generate_contextual_candidates(altered) == before
    del altered["evaluation_sidecar"]
    assert generate_contextual_candidates(altered) == before


@pytest.mark.parametrize("leak", ["outcome", "decision_loss_after", "baseline_loss"])
def test_direct_generator_rejects_evaluator_context_keys(leak):
    value = historical_episode()
    value["public"]["context"][leak] = "private-value"
    with pytest.raises(ValueError):
        generate_contextual_candidates(value)


def test_direct_generator_rejects_affinity_to_private_outcome():
    value = historical_episode()
    value["public"]["candidate_menu"][0]["context_affinity"] = {"outcome": ["home"]}
    with pytest.raises(ValueError):
        generate_contextual_candidates(value)


def test_direct_generator_rejects_nested_private_payload():
    value = historical_episode()
    value["public"]["candidate_menu"][0]["metadata"] = {"nested": [{"result": "3:0"}]}
    with pytest.raises(ValueError):
        generate_contextual_candidates(value)


def test_context_rank_does_not_create_a_numeric_voi_estimate():
    value = historical_episode()
    for item in value["public"]["candidate_menu"]:
        item.pop("expected_value")
    trace = generate_contextual_candidates(value)
    assert trace["candidate_order"][0] == "lineup"
    assert all("expected_value" not in item for item in trace["candidates"])
    assert "decision_loss_after" not in json.dumps(trace)


def test_declared_values_fixed_context_rank_does_not_change_voi_action():
    first = historical_episode()
    second = deepcopy(first)
    second["public"]["context"] = {"dominant_uncertainty": "weather"}
    assert generate_contextual_candidates(first)["candidate_order"] != generate_contextual_candidates(second)["candidate_order"]
    a = compare_contextual_candidates([first])["comparison"]["rows"]["autonomous_voi"][0]
    b = compare_contextual_candidates([second])["comparison"]["rows"]["autonomous_voi"][0]
    assert a["selected_candidates"] == b["selected_candidates"] == ["lineup"]


def test_comparison_keeps_original_baseline_menu_order_and_metrics():
    value = historical_episode()
    before = deepcopy(value)
    baseline = compare_historical_episodes([value])
    contextual = compare_contextual_candidates([value])
    assert contextual["candidate_generation"][0]["candidate_order"] == ["lineup", "weather"]
    for policy in ("fixed_all", "fixed_order"):
        assert contextual["comparison"]["rows"][policy] == baseline["rows"][policy]
    assert contextual["comparison"]["rows"]["fixed_order"][0]["selected_candidates"] == ["weather"]
    assert value == before


def test_already_qualified_does_not_fabricate_rotation_or_tactical_context():
    value = historical_episode()
    value["public"]["context"] = {"qualification_state": "already_qualified"}
    value["public"]["candidate_menu"][1]["context_affinity"] = {"rotation_risk": ["high"]}
    trace = generate_contextual_candidates(value)
    lineup = next(item for item in trace["candidates"] if item["candidate_id"] == "lineup")
    assert "rotation_risk" in lineup["unknown_context_fields"]
    assert "rotation_risk=high" not in lineup["matched_context"]
    assert trace["context"].get("rotation_risk", "unknown") == "unknown"

