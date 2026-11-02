"""Round145 tests for the historical-to-replay episode adapter."""
from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.historical_asof_episode import build_historical_asof_episode
from football_agent.counterfactual_policy_replay import replay_counterfactual_episode
from football_agent.replay_episode_adapter import build_replay_episode_from_historical_envelope


def _envelope():
    return build_historical_asof_episode({
        "match_id": "spftll:match-145",
        "kickoff": "2026-10-04T12:00:00Z",
        "prediction_cutoff": "2026-10-04T08:00:00Z",
        "competition_id": "asian-games",
        "competition_name": "Asian Games",
        "home_team_id": "cn",
        "away_team_id": "ae",
        "home_team_name": "China",
        "away_team_name": "UAE",
        "known": {"qualification_state": "draw_sufficient",
                   "tactical_layout": "defensive"},
        "candidate_generation_complete": True,
        "source_observations": [{
            "source_id": "spftll-fixture", "kind": "fixture",
            "available_at": "2026-10-04T07:00:00Z",
            "observed_at": "2026-10-04T06:50:00Z",
            "values": {"stage": "group"},
        }],
        "candidate_menu": [
            {"candidate_id": "lineup", "source_ids": ["spftll-fixture"],
             "declared_cost": {"requests": 1}, "availability": "available"},
            {"candidate_id": "tactics", "source_ids": ["spftll-fixture"],
             "declared_cost": {"requests": 1}, "availability": "available"},
        ],
        "home_goals": 3, "away_goals": 0,
        "label_status": "INDEPENDENT_EVALUATOR_GOLD",
    })


def _sidecar():
    return {
        "label_status": "INDEPENDENT_GOLD",
        "baseline_loss": .8,
        "sequence_losses": {"": .8, "lineup": .4, "lineup>tactics": .3},
        "action_returns": {"lineup": {"available": True, "context_update": {}},
                           "tactics": {"available": True, "context_update": {}}},
    }


def _routes():
    return {
        "lineup": {"source_id": "api-football", "action_id": "query-lineup", "cost": .1},
        "tactics": {"source_id": "news-feed", "action_id": "query-tactics", "cost": .05},
    }


def test_adapter_projects_public_context_and_keeps_sidecar_sibling():
    result = build_replay_episode_from_historical_envelope(
        _envelope(), candidate_routes=_routes(), budget=.15, evaluator_sidecar=_sidecar())
    assert result["episode_id"].startswith("historical:")
    assert result["public"]["initial_context"]["qualification_state"] == "draw_sufficient"
    assert [row["candidate_id"] for row in result["public"]["candidate_menu"]] == ["lineup", "tactics"]
    assert result["public"]["candidate_menu"][0]["action_id"] == "query-lineup"
    assert result["evaluation_sidecar"]["label_status"] == "INDEPENDENT_GOLD"
    assert "sequence_losses" not in result["public"]
    assert result["adapter"]["runtime_can_read_sidecar"] is False


def test_adapter_rejects_route_set_mismatch_and_missing_route_identity():
    routes = _routes()
    routes.pop("tactics")
    with pytest.raises(ValueError, match="route set mismatch"):
        build_replay_episode_from_historical_envelope(_envelope(), candidate_routes=routes, budget=.15, evaluator_sidecar=_sidecar())
    routes = _routes()
    routes["lineup"].pop("action_id")
    with pytest.raises(ValueError, match="action_id"):
        build_replay_episode_from_historical_envelope(_envelope(), candidate_routes=routes, budget=.15, evaluator_sidecar=_sidecar())


def test_adapter_rejects_non_independent_sidecar():
    sidecar = _sidecar()
    sidecar["label_status"] = "ACTOR_PROVISIONAL"
    with pytest.raises(ValueError, match="INDEPENDENT_GOLD"):
        build_replay_episode_from_historical_envelope(_envelope(), candidate_routes=_routes(), budget=.15, evaluator_sidecar=sidecar)


def test_adapter_rejects_evaluator_fields_in_explicit_initial_context():
    context = {"qualification_state": "unknown", "result": "3:0"}
    with pytest.raises(ValueError, match="evaluator-only"):
        build_replay_episode_from_historical_envelope(_envelope(), candidate_routes=_routes(), budget=.15, evaluator_sidecar=_sidecar(), initial_context=context)


def test_adapter_does_not_mutate_input_sidecar_or_envelope():
    envelope = _envelope()
    sidecar = _sidecar()
    before = deepcopy(sidecar)
    build_replay_episode_from_historical_envelope(envelope, candidate_routes=_routes(), budget=.15, evaluator_sidecar=sidecar)
    assert sidecar == before
    assert envelope["episode"]["pre_action"]["candidate_ids"] == ["lineup", "tactics"]


def test_adapter_output_is_directly_replayable():
    episode = build_replay_episode_from_historical_envelope(
        _envelope(), candidate_routes=_routes(), budget=.15, evaluator_sidecar=_sidecar())
    report = replay_counterfactual_episode(episode, "fixed_order")
    assert report["actor_trace"]["selected_actions"] == ["lineup", "tactics"]
    assert report["evaluator"]["counterfactual_eligible"] is True
    assert report["evaluator"]["final_loss"] == .3
