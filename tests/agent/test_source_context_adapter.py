"""Round147 tests for collector-to-scenario context normalization."""
from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.historical_asof_episode import build_historical_asof_episode
from football_agent.replay_episode_adapter import build_replay_episode_from_historical_envelope
from football_agent.source_context_adapter import (
    apply_source_backed_context,
    build_source_backed_context,
    normalize_source_context_observations,
)

BASE = "2026-10-04T08:00:00Z"


def _rows():
    return [
        {
            "source_id": "rulebook",
            "available_at": "2026-10-04T07:00:00Z",
            "observed_at": "2026-10-04T06:50:00Z",
            "values": {"qualification_status": "draw_sufficient", "stage": "group"},
        },
        {
            "source_id": "tactics-feed",
            "available_at": "2026-10-04T07:20:00Z",
            "values": {"context": {"tactical_style": "defensive", "rotation": "unknown"}},
        },
    ]


def test_normalizer_maps_explicit_aliases_and_keeps_trace():
    normalized, issues = normalize_source_context_observations(_rows())
    assert issues == []
    assert normalized[0]["values"] == {"qualification_state": "draw_sufficient", "competition_stage": "group"}
    assert "qualification_status->qualification_state" in normalized[0]["normalization_trace"]
    assert normalized[1]["values"]["tactical_layout"] == "defensive"


def test_builder_keeps_qualification_and_rotation_independent():
    report = build_source_backed_context(_rows(), target_key="match-147", prediction_cutoff=BASE)
    context = report["binding"]["context"]
    assert context["qualification_state"] == "draw_sufficient"
    assert context["tactical_layout"] == "defensive"
    assert context["rotation_risk"] == "unknown"
    assert report["contract"]["cross_field_inference"] is False


def test_alias_conflict_is_reported_and_field_is_not_bound():
    rows = [{
        "source_id": "mixed-source",
        "available_at": "2026-10-04T07:00:00Z",
        "values": {"qualification": "already_qualified", "qualification_state": "must_win"},
    }]
    report = build_source_backed_context(rows, target_key="match-147", prediction_cutoff=BASE)
    assert report["binding"]["context"]["qualification_state"] == "unknown"
    assert report["normalization_issues"][0]["reason"] == "field_alias_conflict"


def test_missing_source_id_is_audited_without_mutating_input():
    rows = _rows() + [{"values": {"weather": "rain"}, "available_at": "2026-10-04T07:00:00Z"}]
    before = deepcopy(rows)
    normalized, issues = normalize_source_context_observations(rows)
    assert normalized and issues[-1]["reason"] == "missing_source_id"
    assert rows == before


def test_late_rows_are_left_for_binding_cutoff_audit():
    rows = [{
        "source_id": "late-weather",
        "available_at": "2026-10-04T09:00:00Z",
        "values": {"weather": "storm"},
    }]
    report = build_source_backed_context(rows, target_key="match-147", prediction_cutoff=BASE)
    assert report["binding"]["context"]["weather_state"] == "unknown"
    assert report["binding"]["ignored_observations"][0]["reason"] == "late_for_cutoff"


def test_rejects_outcome_fields_before_binding():
    with pytest.raises(ValueError, match="evaluator fields"):
        build_source_backed_context([{
            "source_id": "bad",
            "available_at": "2026-10-04T07:00:00Z",
            "values": {"context": {"outcome": "home_win"}},
        }], target_key="match-147", prediction_cutoff=BASE)


def test_apply_source_backed_context_preserves_sidecar():
    envelope = build_historical_asof_episode({
        "match_id": "match-147",
        "kickoff": "2026-10-04T12:00:00Z",
        "prediction_cutoff": BASE,
        "competition_id": "asian-games",
        "competition_name": "Asian Games",
        "home_team_id": "cn", "away_team_id": "ae",
        "home_team_name": "China", "away_team_name": "UAE",
        "known": {}, "candidate_generation_complete": True,
        "source_observations": [{"source_id": "fixture", "kind": "fixture", "available_at": "2026-10-04T07:00:00Z", "observed_at": "2026-10-04T06:50:00Z", "values": {}}],
        "candidate_menu": [{"candidate_id": "lineup", "source_ids": ["fixture"], "declared_cost": {"requests": 1}, "availability": "available"}],
        "home_goals": 1, "away_goals": 0, "label_status": "INDEPENDENT_EVALUATOR_GOLD",
    })
    sidecar = {"label_status": "INDEPENDENT_GOLD", "baseline_loss": .5, "sequence_losses": {"": .5}}
    episode = build_replay_episode_from_historical_envelope(envelope, candidate_routes={"lineup": {"source_id": "api", "action_id": "lineup", "cost": .1}}, budget=.1, evaluator_sidecar=sidecar)
    before = deepcopy(episode)
    updated = apply_source_backed_context(episode, build_source_backed_context(_rows(), target_key="match-147", prediction_cutoff=BASE))
    assert episode == before
    assert updated["public"]["context"]["qualification_state"] == "draw_sufficient"
    assert updated["evaluation_sidecar"] == sidecar


def test_report_contract_exposes_no_runtime_calls():
    report = build_source_backed_context(_rows(), target_key="match-147", prediction_cutoff=BASE)
    assert report["contract"]["real_source_requests"] == 0
    assert report["contract"]["real_model_calls"] == 0
