"""Strict matched four-arm policy replay and trajectory alignment."""
from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.paired_policy_replay import (
    POLICIES,
    compare_aligned_policies,
    project_independent_registry_sidecar,
)
from football_agent.independent_outcome_loss_registry import digest as registry_digest


def _episode(*, complete: bool = True):
    returns = {}
    for candidate in ("weather", "lineup"):
        returns[candidate] = {
            "available": True,
            "context_update": {"qualification_state": "already_qualified"} if candidate == "lineup" else {"weather": "clear"},
            "observed_at": "2026-10-01T09:00:00Z",
            "available_at": "2026-10-01T09:01:00Z",
            "captured_at": "2026-10-01T09:02:00Z",
            "prediction_before": [.5, .3, .2],
            "prediction_after": [.6, .2, .2],
        }
    return {
        "episode_id": "paired-1",
        "public": {
            "target_key": "match-1",
            "kickoff": "2026-10-01T18:00:00Z",
            "prediction_cutoff": "2026-10-01T12:00:00Z",
            "initial_context": {"qualification_state": "unknown"},
            "initial_prediction": [.5, .3, .2],
            "candidate_generation_complete": complete,
            "budget": .2,
            "loss_unit": "decision_loss",
            "candidate_menu": [
                {"candidate_id": "weather", "cost": .1, "source_id": "weather"},
                {"candidate_id": "lineup", "cost": .1, "source_id": "lineup"},
            ],
        },
        "evaluation_sidecar": {
            "label_status": "INDEPENDENT_GOLD",
            "independent_result_bound": True,
            "outcome": 0,
            "baseline_loss": .8,
            "cost_to_loss_rate": 1.0,
            "sequence_losses": {
                "": .8, "weather": .7, "lineup": .6,
                "weather>lineup": .5, "lineup>weather": .4,
            },
            "action_returns": returns,
        },
    }


def _provider(state):
    result = {
        cid: {"state_digest": state["state_digest"], "lower_bound": 0.0, "upper_bound": .01}
        for cid in state["remaining_candidate_ids"]
    }
    if state["context"].get("qualification_state") == "unknown" and "lineup" in result:
        result["lineup"] = {"state_digest": state["state_digest"], "lower_bound": .3, "upper_bound": .8}
    return result


def test_four_arms_share_manifest_and_align_steps_without_mixing_controls():
    report = compare_aligned_policies([_episode()], interval_provider=_provider)
    assert tuple(report["policy_order"]) == POLICIES
    assert report["contract"]["same_public_initial_state_menu_budget"] is True
    assert report["contract"]["autonomous_selection_separated_from_controls"] is True
    selected = {policy: rows[0]["selected_actions"] for policy, rows in report["results"].items()}
    assert selected["autonomous"] == ["lineup"]
    assert selected["fixed_order"] == ["weather", "lineup"]
    assert selected["full_query"] == ["lineup", "weather"]
    assert selected["stop"] == []
    assert len(report["trajectory_alignments"][0]["steps"]) == 3
    assert all(rows[0]["comparable"] for rows in report["results"].values())
    assert report["paired_deltas"]["stop"][0]["final_loss_delta"] == pytest.approx(.2)


def test_missing_prediction_outcome_loss_or_clock_is_incomparable_and_not_zero():
    episode = _episode()
    episode["evaluation_sidecar"].pop("outcome")
    episode["evaluation_sidecar"]["sequence_losses"].pop("weather>lineup")
    episode["evaluation_sidecar"]["action_returns"]["weather"].pop("captured_at")
    report = compare_aligned_policies([episode], interval_provider=_provider)
    for policy in POLICIES:
        row = report["results"][policy][0]
        assert row["comparable"] is False
        assert row["final_loss"] is None
    delta = report["paired_deltas"]["fixed_order"][0]
    assert delta["comparable"] is False
    assert delta["final_loss_delta"] is None
    assert report["paired_summary"]["fixed_order"]["mean_final_loss_delta"] is None
    assert report["contract"]["unknown_metrics_are_not_zero"] is True


def test_controls_do_not_require_or_consult_autonomous_provider():
    report = compare_aligned_policies([_episode()], policies=("fixed_order", "full_query", "stop"))
    assert report["baseline_policy"] == "fixed_order"
    assert report["results"]["fixed_order"][0]["selected_actions"] == ["weather", "lineup"]


def test_public_initial_state_menu_and_budget_must_match():
    other = _episode()
    other["episode_id"] = "paired-2"
    other["public"]["candidate_menu"] = list(reversed(other["public"]["candidate_menu"]))
    with pytest.raises(ValueError, match="same public initial state"):
        compare_aligned_policies([_episode(), other], interval_provider=_provider)


def test_trajectory_projection_does_not_expose_sidecar():
    report = compare_aligned_policies([_episode()], interval_provider=_provider)
    encoded = str(report["results"]).lower()
    assert "sequence_losses" not in encoded
    assert "action_returns" not in encoded
    assert "independent_result" not in encoded
    assert "observed_outcome" not in encoded


def test_explicit_stop_prediction_is_read_only_from_public_initial_state():
    episode = _episode()
    episode["public"].pop("initial_prediction")
    # A prediction present in an acquisition return cannot be borrowed to
    # score the zero-query control.
    report = compare_aligned_policies([episode], interval_provider=_provider)
    stop = report["results"]["stop"][0]
    assert stop["comparable"] is False
    assert "stop_initial_prediction_missing" in stop["incomparable_reasons"]


def _registry_pair():
    outcome = {
        "schema_version": "round167-independent-outcome-loss-v1",
        "status": "INDEPENDENT_GOLD",
        "independent_from_actor": True,
        "evaluator_id": "independent-outcome-v1",
        "source": "independent-results",
        "source_id": "result-1",
        "source_payload_digest": "a" * 64,
        "actor_binding_digest": "b" * 64,
        "actor_source_ids": ["understat:match-1"],
        "actor_source_digests": ["c" * 64],
        "actor_sources": ["understat"],
        "episode_id": "paired-1",
        "target_key": "match-1",
        "prediction_cutoff": "2026-10-01T12:00:00Z",
        "kickoff": "2026-10-01T18:00:00Z",
        "observed_at": "2026-10-01T19:00:00Z",
        "available_at": "2026-10-01T19:01:00Z",
        "outcome_label": "home",
        "label_mapping": {"home": 0, "draw": 1, "away": 2},
        "home_goals": 1,
        "away_goals": 0,
    }
    outcome["record_digest"] = registry_digest(outcome)
    evidence = {
        "lineup": {
            "available": True, "action_id": "read_lineup", "request_id": "req-lineup",
            "prediction_before": [.5, .3, .2], "prediction_after": [.6, .2, .2],
            "observed_at": "2026-10-01T09:00:00Z", "available_at": "2026-10-01T09:01:00Z",
            "captured_at": "2026-10-01T09:02:00Z", "context_update": {"lineup": "known"},
        },
    }
    registry = {
        "status": "COMPUTED", "numeric_voi_training_eligible": True,
        "outcome_record_digest": outcome["record_digest"],
        "sequence_losses": {"": .8, "lineup": .6}, "cost_to_loss_rate": 1.0,
        "results": [{"status": "COMPUTED", "candidate_id": "lineup",
                      "action_id": "read_lineup", "request_id": "req-lineup",
                      "metric": "log_loss",
                      "decision_loss_before": 0.69314718056,
                      "decision_loss_after": 0.510825623765,
                      "outcome_record_digest": outcome["record_digest"]}],
        "action_evidence": evidence,
    }
    registry["results"][0]["loss_digest"] = registry_digest(registry["results"][0])
    return outcome, registry


def test_independent_registry_projects_only_explicit_paired_sidecar_fields():
    outcome, registry = _registry_pair()
    projected = project_independent_registry_sidecar(outcome, registry)
    assert projected["status"] == "COMPUTED"
    sidecar = projected["sidecar"]
    assert sidecar["label_status"] == "INDEPENDENT_GOLD"
    assert sidecar["outcome"] == 0
    assert sidecar["sequence_losses"] == {"": .8, "lineup": .6}
    assert sidecar["action_returns"]["lineup"]["captured_at"].endswith("09:02:00Z")
    encoded = str(projected).lower()
    assert "home_goals" not in encoded
    assert "away_goals" not in encoded
    assert "outcome_label" not in encoded
    assert projected["public_projection"]["outcome_visible_to_actor"] is False


def test_incomplete_registry_is_partial_and_never_invents_loss_or_action_clocks():
    outcome, registry = _registry_pair()
    registry.pop("sequence_losses")
    registry.pop("action_evidence")
    projected = project_independent_registry_sidecar(outcome, registry)
    assert projected["status"] in {"PARTIAL", "UNKNOWN"}
    assert projected["numeric_voi_training_eligible"] is False
    assert "sequence_losses" not in projected["sidecar"]
    assert projected["sidecar"]["action_returns"] == {}
    assert "sequence_losses_missing" in projected["reasons"]
    assert "action_evidence_missing" in projected["reasons"]


def test_noncomputed_registry_cannot_be_upgraded_by_complete_rows():
    outcome, registry = _registry_pair()
    registry["status"] = "PARTIAL"
    registry["numeric_voi_training_eligible"] = False
    projected = project_independent_registry_sidecar(outcome, registry)
    assert projected["status"] == "PARTIAL"
    assert projected["numeric_voi_training_eligible"] is False
    assert "loss_registry_not_computed" in projected["reasons"]
    assert "loss_registry_not_numeric_voi_eligible" in projected["reasons"]


@pytest.mark.parametrize("field,reason", [
    ("source", "independent_source_missing"),
    ("source_id", "independent_source_id_missing"),
    ("source_payload_digest", "independent_source_payload_digest_missing_or_invalid"),
    ("actor_binding_digest", "independent_actor_binding_digest_missing_or_invalid"),
])
def test_projector_requires_independent_provenance_fields(field, reason):
    outcome, registry = _registry_pair()
    outcome.pop(field)
    outcome["record_digest"] = registry_digest({k: v for k, v in outcome.items() if k != "record_digest"})
    projected = project_independent_registry_sidecar(outcome, registry)
    assert projected["status"] in {"PARTIAL", "UNKNOWN"}
    assert projected["numeric_voi_training_eligible"] is False
    assert reason in projected["reasons"]


def test_projector_rejects_result_available_before_prediction_cutoff():
    outcome, registry = _registry_pair()
    outcome["available_at"] = "2026-10-01T11:59:00Z"
    outcome["record_digest"] = registry_digest({k: v for k, v in outcome.items() if k != "record_digest"})
    registry["outcome_record_digest"] = outcome["record_digest"]
    for row in registry["results"]:
        row["outcome_record_digest"] = outcome["record_digest"]
    projected = project_independent_registry_sidecar(outcome, registry)
    assert projected["status"] in {"PARTIAL", "UNKNOWN"}
    assert "independent_outcome_available_before_cutoff" in projected["reasons"]
    assert projected["sidecar"]["label_status"] is None


def test_projector_recomputes_loss_and_requires_loss_digest():
    outcome, registry = _registry_pair()
    registry["results"][0]["decision_loss_after"] = 0.01
    projected = project_independent_registry_sidecar(outcome, registry)
    assert projected["status"] == "PARTIAL"
    assert "loss_values_do_not_match_predictions:lineup" in projected["reasons"]
    outcome, registry = _registry_pair()
    registry["results"][0].pop("loss_digest")
    projected = project_independent_registry_sidecar(outcome, registry)
    assert projected["status"] == "PARTIAL"
    assert "loss_digest_missing_or_invalid:lineup" in projected["reasons"]


def test_projector_rejects_invalid_mapping_and_prediction_dimension():
    outcome, registry = _registry_pair()
    outcome["label_mapping"] = {"home": -1, "draw": -1, "away": 2}
    outcome["record_digest"] = registry_digest({k: v for k, v in outcome.items() if k != "record_digest"})
    registry["outcome_record_digest"] = outcome["record_digest"]
    for row in registry["results"]:
        row["outcome_record_digest"] = outcome["record_digest"]
    projected = project_independent_registry_sidecar(outcome, registry)
    assert projected["status"] in {"PARTIAL", "UNKNOWN"}
    assert "independent_outcome_label_mapping_invalid" in projected["reasons"]

