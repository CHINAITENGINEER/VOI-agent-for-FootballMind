"""Round146 tests for cutoff-safe football scenario context binding."""
from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.historical_asof_episode import build_historical_asof_episode
from football_agent.replay_episode_adapter import build_replay_episode_from_historical_envelope
from football_agent.scenario_context_binding import apply_scenario_context, bind_scenario_context

BASE = "2026-10-04T08:00:00Z"


def _obs():
    return [
        {
            "source_id": "competition-rulebook",
            "source_version": "v1",
            "available_at": "2026-10-04T07:00:00Z",
            "observed_at": "2026-10-04T06:50:00Z",
            "quality": "validated",
            "values": {"context": {
                "qualification_state": "already_qualified",
                "match_importance": "low",
            }},
        },
        {
            "source_id": "tactical-report",
            "available_at": "2026-10-04T07:30:00Z",
            "observed_at": "2026-10-04T07:25:00Z",
            "values": {"tactical_layout": "defensive", "rotation_risk": "high"},
        },
    ]


def test_binds_observed_fields_and_preserves_unknowns():
    result = bind_scenario_context(_obs(), target_key="match-146", prediction_cutoff=BASE)
    assert result["context"]["qualification_state"] == "already_qualified"
    assert result["context"]["tactical_layout"] == "defensive"
    assert result["context"]["rotation_risk"] == "high"
    assert result["context"]["weather_state"] == "unknown"
    assert result["numeric_voi_estimated"] is False
    assert result["evaluator_fields_used"] is False


def test_does_not_infer_rotation_from_qualification():
    result = bind_scenario_context([_obs()[0]], target_key="match-146", prediction_cutoff=BASE)
    assert result["context"]["qualification_state"] == "already_qualified"
    assert result["context"]["rotation_risk"] == "unknown"
    assert "rotation_risk" not in result["bound_fields"]


def test_conflicting_sources_become_unknown_with_audit():
    observations = _obs() + [{
        "source_id": "lineup-note",
        "available_at": "2026-10-04T07:40:00Z",
        "values": {"context": {"tactical_layout": "balanced"}},
    }]
    result = bind_scenario_context(observations, target_key="match-146", prediction_cutoff=BASE)
    assert result["context"]["tactical_layout"] == "unknown"
    assert result["conflicting_fields"] == ["tactical_layout"]
    assert result["supporting_source_ids"]["tactical_layout"] == ["lineup-note", "tactical-report"]


def test_late_and_missing_availability_are_ignored():
    observations = _obs() + [
        {"source_id": "late-news", "available_at": "2026-10-04T09:00:00Z", "values": {"context": {"weather_state": "storm"}}},
        {"source_id": "untracked-news", "values": {"context": {"weather_state": "rain"}}},
    ]
    result = bind_scenario_context(observations, target_key="match-146", prediction_cutoff=BASE)
    assert result["context"]["weather_state"] == "unknown"
    assert {item["reason"] for item in result["ignored_observations"]} == {"late_for_cutoff", "missing_availability"}


def test_unknown_tokens_normalize_without_becoming_evaluator_labels():
    result = bind_scenario_context([{
        "source_id": "weather",
        "available_at": "2026-10-04T07:00:00Z",
        "values": {"weather_state": "N/A"},
    }], target_key="match-146", prediction_cutoff=BASE, required_fields=["weather_state"])
    assert result["context"] == {"weather_state": "unknown"}
    assert result["bound_fields"] == ["weather_state"]


def test_rejects_future_result_or_evaluator_fields():
    with pytest.raises(ValueError, match="evaluator fields"):
        bind_scenario_context([{
            "source_id": "bad-source",
            "available_at": "2026-10-04T07:00:00Z",
            "values": {"context": {"result": "3:0"}},
        }], target_key="match-146", prediction_cutoff=BASE)


def test_apply_context_preserves_sidecar_and_does_not_mutate_episode():
    envelope = build_historical_asof_episode({
        "match_id": "match-146",
        "kickoff": "2026-10-04T12:00:00Z",
        "prediction_cutoff": BASE,
        "competition_id": "asian-games",
        "competition_name": "Asian Games",
        "home_team_id": "jp", "away_team_id": "th",
        "home_team_name": "Japan", "away_team_name": "Thailand",
        "known": {"qualification_state": "already_qualified"},
        "candidate_generation_complete": True,
        "source_observations": [{
            "source_id": "fixture", "kind": "fixture",
            "available_at": "2026-10-04T07:00:00Z",
            "observed_at": "2026-10-04T06:50:00Z", "values": {},
        }],
        "candidate_menu": [{"candidate_id": "lineup", "source_ids": ["fixture"], "declared_cost": {"requests": 1}, "availability": "available"}],
        "home_goals": 3, "away_goals": 0,
        "label_status": "INDEPENDENT_EVALUATOR_GOLD",
    })
    sidecar = {"label_status": "INDEPENDENT_GOLD", "baseline_loss": .5, "sequence_losses": {"": .5}}
    episode = build_replay_episode_from_historical_envelope(
        envelope, candidate_routes={"lineup": {"source_id": "api", "action_id": "lineup", "cost": .1}},
        budget=.1, evaluator_sidecar=sidecar)
    before = deepcopy(episode)
    binding = bind_scenario_context(_obs(), target_key="match-146", prediction_cutoff=BASE)
    updated = apply_scenario_context(episode, binding)
    assert episode == before
    assert updated["public"]["context"]["qualification_state"] == "already_qualified"
    assert updated["public"]["context"]["tactical_layout"] == "defensive"
    assert updated["evaluation_sidecar"] == sidecar
    assert updated["public"]["context_binding"]["evaluator_fields_used"] is False


def test_apply_context_rejects_wrong_binding_version():
    with pytest.raises(ValueError, match="schema_version mismatch"):
        apply_scenario_context({"public": {}}, {"schema_version": "old"})
