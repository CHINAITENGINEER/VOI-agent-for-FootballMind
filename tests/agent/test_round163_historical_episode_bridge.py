"""Round163 tests for historical episode/result-source separation."""
from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.counterfactual_policy_replay import INDEPENDENT_GOLD
from football_agent.historical_asof_episode import build_historical_asof_episode
from football_agent.round162_policy_replay_adapter import compare_policies
from football_agent.round163_historical_episode_bridge import (
    SOURCE_ONLY_NO_GOLD,
    build_round163_episode,
    validate_round163_episode,
)


CUTOFF = "2026-09-26T12:00:00Z"
KICKOFF = "2026-09-26T13:00:00Z"


def _historical_envelope():
    return build_historical_asof_episode({
        "match_id": "2981481",
        "target_key": "spftll:2981481",
        "kickoff": KICKOFF,
        "prediction_cutoff": CUTOFF,
        "competition_id": "name:欧国联",
        "competition_name": "欧国联",
        "home_team_id": "632",
        "away_team_id": "641",
        "home_team_name": "斯洛文尼",
        "away_team_name": "苏格兰",
        "source": "spftll",
        "source_observations": [
            {"source_id": "fixture:2981481", "kind": "fixture",
             "observed_at": "2026-09-26T11:44:50Z",
             "available_at": "2026-09-26T11:45:00Z",
             "content_digest": "a" * 64,
             "values": {"competition": "欧国联", "fixture_state": "scheduled"}},
            {"source_id": "odds:2981481", "kind": "odds",
             "observed_at": "2026-09-26T11:46:00Z",
             "available_at": "2026-09-26T11:47:00Z",
             "content_digest": "b" * 64,
             "values": {"market": "1x2", "home_price": 2.45}},
        ],
        "candidate_menu": [
            {"candidate_id": "fixture_context", "kind": "fixture",
             "source_ids": ["fixture:2981481"], "declared_cost": {"requests": 1}},
            {"candidate_id": "odds_snapshot", "kind": "odds",
             "source_ids": ["odds:2981481"], "declared_cost": {"requests": 1}},
        ],
        "candidate_generation_complete": True,
        "known": {"baseline": "pre_match", "future_field": "must be dropped"},
    })


def _menu():
    return [
        {"candidate_id": "fixture_context", "cost": .1,
         "source_id": "fixture:2981481", "action_id": "read_fixture"},
        {"candidate_id": "odds_snapshot", "cost": .1,
         "source_id": "odds:2981481", "action_id": "read_odds"},
    ]


def _result(**changes):
    value = {
        "source_id": "independent:result-provider:2981481",
        "observed_at": "2026-09-26T15:30:00Z",
        "available_at": "2026-09-26T15:35:00Z",
        "content_digest": "c" * 64,
        "home_goals": 1,
        "away_goals": 0,
        "score_period": "regulation",
    }
    value.update(changes)
    return value


def _loss_sidecar():
    return {
        "baseline_loss": .8,
        "cost_to_loss_rate": 1.0,
        "sequence_losses": {"": .8, "fixture_context>odds_snapshot": .5,
                             "odds_snapshot>fixture_context": .4},
        "action_returns": {
            "fixture_context": {"available": True, "context_update": {"fixture_state": "scheduled"},
                                 "observed_at": "2026-09-26T11:50:00Z",
                                 "available_at": "2026-09-26T11:51:00Z"},
            "odds_snapshot": {"available": True, "context_update": {"market_seen": True},
                               "observed_at": "2026-09-26T11:52:00Z",
                               "available_at": "2026-09-26T11:53:00Z"},
        },
    }


def test_real_archive_can_be_bridged_as_source_only_without_gold_label():
    episode = build_round163_episode(
        _historical_envelope(), candidate_menu=_menu(),
        initial_context={"qualification_state": "unknown"}, budget=.2)
    assert episode["evaluation_sidecar"]["label_status"] == SOURCE_ONLY_NO_GOLD
    assert episode["evaluation_sidecar"]["independent_result_bound"] is False
    assert "home_goals" not in episode["evaluation_sidecar"]
    assert episode["audit"]["counterfactual_replay_ready"] is False
    assert validate_round163_episode(episode)["valid"] is True


def test_source_only_ignores_loss_declarations_until_result_is_bound():
    episode = build_round163_episode(
        _historical_envelope(), candidate_menu=_menu(), budget=.2,
        evaluator_sidecar=_loss_sidecar())
    sidecar = episode["evaluation_sidecar"]
    assert sidecar["label_status"] == SOURCE_ONLY_NO_GOLD
    assert "sequence_losses" not in sidecar
    assert sidecar["loss_fields_ignored"] == ["action_returns", "baseline_loss",
                                                 "cost_to_loss_rate", "sequence_losses"]


def test_independent_result_requires_post_kickoff_separate_source_and_digest():
    episode = build_round163_episode(
        _historical_envelope(), candidate_menu=_menu(), budget=.2,
        evaluator_sidecar=_loss_sidecar(), independent_result=_result())
    sidecar = episode["evaluation_sidecar"]
    assert sidecar["label_status"] == INDEPENDENT_GOLD
    assert sidecar["independent_result_bound"] is True
    assert sidecar["outcome"] == 0
    assert sidecar["outcome_digest"]
    assert episode["audit"]["counterfactual_replay_ready"] is True
    report = validate_round163_episode(episode)
    assert report["valid"] is True
    assert report["independent_gold"] is True


def test_independent_gold_bridge_can_enter_round162_replay():
    episode = build_round163_episode(
        _historical_envelope(), candidate_menu=_menu(), budget=.2,
        evaluator_sidecar=_loss_sidecar(), independent_result=_result())

    def interval_provider(state):
        return {
            candidate_id: {
                "state_digest": state["state_digest"],
                "lower_bound": 0.0,
                "upper_bound": .01,
            }
            for candidate_id in state["remaining_candidate_ids"]
        }

    report = compare_policies([episode], interval_provider=interval_provider)
    assert report["evaluation_label_status"] == INDEPENDENT_GOLD
    assert report["independent_gold_labels"] == 1
    assert report["eligible_episode_count_by_policy"]["fixed_order"] == 1
    assert report["eligible_episode_count_by_policy"]["interval_voi"] == 1


def test_result_availability_must_follow_observation_clock():
    with pytest.raises(ValueError, match="follow observation"):
        build_round163_episode(
            _historical_envelope(), candidate_menu=_menu(), budget=.2,
            independent_result=_result(available_at="2026-09-26T11:59:59Z"))


def test_result_before_kickoff_is_rejected():
    with pytest.raises(ValueError, match="follow kickoff"):
        build_round163_episode(
            _historical_envelope(), candidate_menu=_menu(), budget=.2,
            independent_result=_result(observed_at="2026-09-26T12:30:00Z"))


def test_same_actor_source_or_digest_cannot_be_called_independent():
    with pytest.raises(ValueError, match="separate from actor sources"):
        build_round163_episode(
            _historical_envelope(), candidate_menu=_menu(), budget=.2,
            independent_result=_result(source_id="odds:2981481"))
    with pytest.raises(ValueError, match="separate from actor evidence"):
        build_round163_episode(
            _historical_envelope(), candidate_menu=_menu(), budget=.2,
            independent_result=_result(content_digest="a" * 64))


def test_public_projection_has_stable_snapshot_digest_and_no_result_fields():
    first = build_round163_episode(
        _historical_envelope(), candidate_menu=_menu(), budget=.2,
        independent_result=_result())
    second = build_round163_episode(
        _historical_envelope(), candidate_menu=_menu(), budget=.2,
        independent_result=_result())
    assert first["public"] == second["public"]
    assert first["public"]["public_snapshot_digest"]
    encoded = str(first["public"]).lower()
    for forbidden in ("home_goals", "away_goals", "outcome", "sequence_losses", "action_returns"):
        assert forbidden not in encoded


def test_candidate_menu_must_match_historical_candidate_ids():
    with pytest.raises(ValueError, match="does not match historical menu"):
        build_round163_episode(
            _historical_envelope(), candidate_menu=[{"candidate_id": "weather", "cost": .1}], budget=.2)


def test_tampered_public_snapshot_is_detected():
    episode = build_round163_episode(
        _historical_envelope(), candidate_menu=_menu(), budget=.2)
    tampered = deepcopy(episode)
    tampered["public"]["match_snapshot"]["home_team_name"] = "other"
    report = validate_round163_episode(tampered)
    assert report["valid"] is False
    assert "public_snapshot_digest_mismatch" in report["reasons"]


def test_tampered_historical_snapshot_digest_is_rejected_before_bridge():
    envelope = _historical_envelope()
    envelope["episode"]["pre_action"]["snapshot"]["home_team_name"] = "other"
    with pytest.raises(ValueError, match="historical snapshot digest mismatch"):
        build_round163_episode(envelope, candidate_menu=_menu(), budget=.2)
