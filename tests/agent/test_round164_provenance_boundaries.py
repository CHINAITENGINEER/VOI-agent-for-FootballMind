"""Regression cases for real-source clocks and evaluator isolation."""
from copy import deepcopy
import hashlib
import json

import pytest

from football_agent.historical_asof_episode import build_historical_asof_episode
from football_agent.round163_historical_episode_bridge import (
    build_round163_episode, validate_round163_episode,
)
from football_agent.round164_independent_result_registry import (
    digest, join_result_to_public, reconstruct_spftll_fixture_public,
    registry_entry_from_row,
)


KICKOFF = "2025-08-15T19:00:00Z"
CUTOFF = "2025-08-15T18:00:00Z"


def _inputs():
    public = {
        "target_key": "spftll:match-1", "kickoff": KICKOFF,
        "prediction_cutoff": CUTOFF,
        "match_snapshot": {
            "source": "spftll", "match_id": "match-1", "kickoff": KICKOFF,
            "home_team_id": "home-1", "away_team_id": "away-1",
            "home_team_name": "Home", "away_team_name": "Away",
            "competition_id": "league-1", "competition_name": "League",
            "source_observations": [{
                "source_id": "spftll:match-1:fixture", "observation_id": "actor-observation",
                "kind": "fixture", "observed_at": "2025-08-15T17:00:00Z",
                "available_at": "2025-08-15T17:01:00Z", "content_digest": "a" * 64,
                "values": {"fixture_state": "scheduled"},
            }],
        },
    }
    row = {
        "source": "understat", "source_id": "result-1", "match_id": "result-match-1",
        "kickoff": KICKOFF, "home_team_id": "home-2", "away_team_id": "away-2",
        "home_team_name": "Home", "away_team_name": "Away",
        "observed_at": "2025-08-15T21:30:00Z",
        "result_available_at": "2025-08-15T21:31:00Z",
        "source_published_at": "2025-08-15T21:00:00Z",
        "information_mode": "operational", "home_goals": 2, "away_goals": 1,
        "score_period": "regulation", "status": "finished",
        "source_payload_digest": "b" * 64, "observation_id": "result-observation",
    }
    entry = registry_entry_from_row(row, mapping_evidence_digest="e" * 64)
    mapping = {
        "mapping_status": "confirmed", "actor_source": "spftll", "actor_match_id": "match-1",
        "actor_home_team_id": "home-1", "actor_away_team_id": "away-1",
        "result_source": "understat", "result_source_id": "result-1",
        "result_home_team_id": "home-2", "result_away_team_id": "away-2",
        "actor_observation_id": "actor-observation", "result_observation_id": "result-observation",
        "evidence_digest": "e" * 64, "mapping_available_before_cutoff": True,
        "mapping_uses_postmatch_evidence": False,
        "observed_at": "2025-08-15T17:02:00Z", "available_at": "2025-08-15T17:03:00Z",
    }
    _sign_mapping(mapping)
    return public, entry, mapping


def _sign_mapping(mapping):
    mapping["mapping_digest"] = digest({key: value for key, value in mapping.items() if key != "mapping_digest"})


def test_operational_binding_with_complete_provenance_is_accepted_without_claiming_utility():
    joined = join_result_to_public(*_inputs())
    assert joined["accepted"] is True
    assert joined["online_availability_claim"] is True
    assert joined["counterfactual_replay_ready"] is False
    assert joined["numeric_voi_training_eligible"] is False
    assert joined["retrospective_utility_claim"] is False


@pytest.mark.parametrize("changes", [
    {"observed_at": "2025-08-15T22:00:00Z"},
    {"available_at": "2025-08-15T22:00:00Z"},
    {"observed_at": None},
    {"mapping_uses_postmatch_evidence": True},
])
def test_mapping_flags_cannot_override_missing_or_late_evidence(changes):
    public, entry, mapping = _inputs()
    mapping.update(changes)
    _sign_mapping(mapping)
    joined = join_result_to_public(public, entry, mapping)
    assert joined["accepted"] is True
    assert joined["online_availability_claim"] is False
    assert joined["mapping_cutoff_verified"] is False


@pytest.mark.parametrize("field", ["actor_observation_id", "result_observation_id"])
def test_mapping_must_point_to_the_actual_actor_and_result_observations(field):
    public, entry, mapping = _inputs()
    mapping[field] = "different-observation"
    _sign_mapping(mapping)
    joined = join_result_to_public(public, entry, mapping)
    assert joined["accepted"] is False
    assert joined["reason"] == "join_checks_failed"


@pytest.mark.parametrize("field", ["mapping_digest", "actor_observation_id", "result_observation_id"])
def test_incomplete_mapping_lineage_keeps_online_claim_false(field):
    public, entry, mapping = _inputs()
    del mapping[field]
    if field != "mapping_digest":
        _sign_mapping(mapping)
    joined = join_result_to_public(public, entry, mapping)
    assert joined["accepted"] is True
    assert joined["online_availability_claim"] is False


def test_late_actor_observation_is_not_backdated_even_with_operational_result():
    public, entry, mapping = _inputs()
    source = public["match_snapshot"]["source_observations"][0]
    source.update(observed_at="2026-09-20T11:00:00Z", available_at="2026-09-20T11:00:00Z")
    joined = join_result_to_public(public, entry, mapping)
    assert joined["accepted"] is True
    assert joined["online_availability_claim"] is False
    assert joined["actor_cutoff_verified"] is False


@pytest.mark.parametrize("key", ["home_goals", "outcome", "sequence_losses"])
def test_nested_public_evaluator_fields_cannot_pass_a_result_join(key):
    public, entry, mapping = _inputs()
    public["match_snapshot"]["source_observations"][0]["values"][key] = 1
    joined = join_result_to_public(public, entry, mapping)
    assert joined["accepted"] is False
    assert "public_excludes_evaluator_fields" in joined["failed_checks"]


def test_wrapper_kickoff_cannot_hide_a_conflicting_actor_kickoff():
    public, entry, mapping = _inputs()
    public["match_snapshot"]["kickoff"] = "2025-08-16T19:00:00Z"
    joined = join_result_to_public(public, entry, mapping)
    assert joined["accepted"] is False
    assert "public_kickoff_consistent" in joined["failed_checks"]


def test_mapping_evidence_tampering_is_rejected_even_after_mapping_is_rehashed():
    public, entry, mapping = _inputs()
    mapping["evidence_digest"] = "f" * 64
    _sign_mapping(mapping)
    joined = join_result_to_public(public, entry, mapping)
    assert joined["accepted"] is False
    assert "mapping_evidence_bound" in joined["failed_checks"]


def _episode(loss_fields):
    public, entry, mapping = _inputs()
    joined = join_result_to_public(public, entry, mapping)
    historical = build_historical_asof_episode({
        **public["match_snapshot"], "target_key": public["target_key"], "prediction_cutoff": CUTOFF,
        "candidate_generation_complete": True,
        "candidate_menu": [{"candidate_id": "fixture", "kind": "fixture",
                            "source_ids": ["spftll:match-1:fixture"], "declared_cost": {"requests": 1}}],
    })
    return build_round163_episode(
        historical, candidate_menu=[{"candidate_id": "fixture", "cost": .1}], budget=.1,
        independent_result=joined["independent_result"], evaluator_sidecar=loss_fields,
    )


def _loss_fields():
    return {
        "baseline_loss": .8, "cost_to_loss_rate": 1., "sequence_losses": {"": .8, "fixture": .5},
        "action_returns": {"fixture": {"available": True, "context_update": {"fixture_state": "known"},
                                       "observed_at": "2025-08-15T17:00:00Z", "available_at": "2025-08-15T17:01:00Z"}},
    }


def test_operational_registry_result_passes_bridge_without_rewriting_clocks():
    episode = _episode(_loss_fields())
    sidecar = episode["evaluation_sidecar"]
    assert sidecar["result_observed_at"] == "2025-08-15T21:30:00Z"
    assert sidecar["label_available_at"] == "2025-08-15T21:31:00Z"
    assert episode["audit"]["counterfactual_replay_ready"] is True
    assert validate_round163_episode(episode)["valid"] is True


@pytest.mark.parametrize("fields", [{}, {"baseline_loss": .8}, {"sequence_losses": {"": .8}}, _loss_fields() | {"action_returns": {}}])
def test_result_or_partial_loss_sidecar_does_not_establish_replay_readiness(fields):
    episode = _episode(fields)
    assert episode["evaluation_sidecar"]["independent_result_bound"] is True
    assert episode["audit"]["counterfactual_replay_ready"] is False
    assert validate_round163_episode(episode)["counterfactual_replay_eligible"] is False


def test_late_action_returns_do_not_establish_replay_readiness():
    fields = _loss_fields()
    fields["action_returns"]["fixture"]["available_at"] = "2025-08-15T22:00:00Z"
    assert _episode(fields)["audit"]["counterfactual_replay_ready"] is False


def test_raw_fixture_projection_verifies_payload_and_hides_finished_status():
    body = {"code": 0, "data": {"matchId": "match-1", "matchTime": "2025-08-16T03:00:00",
                               "homeId": "home-1", "awayId": "away-1", "leagueId": "league-1",
                               "home": "Home", "away": "Away", "league": "League",
                               "homeScore": 2, "awayScore": 1, "status": "finished"}}
    payload = json.dumps(body).encode()
    observation = {"id": "actor-observation", "source": "spftll", "http_status": 200,
                   "observed_at": "2026-09-20T11:00:00Z", "content_hash": hashlib.sha256(payload).hexdigest()}
    public = reconstruct_spftll_fixture_public(payload, observation, match_id="match-1")
    assert public["kickoff"] == KICKOFF
    assert public["prediction_cutoff"] == CUTOFF
    assert public["match_snapshot"]["source_observations"][0]["observed_at"] == observation["observed_at"]
    assert "finished" not in json.dumps(public)
    assert "homeScore" not in json.dumps(public)
    with pytest.raises(ValueError, match="digest differs"):
        reconstruct_spftll_fixture_public(payload + b" ", observation, match_id="match-1")

