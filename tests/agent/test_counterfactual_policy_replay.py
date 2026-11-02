"""Round143 tests for evaluator-isolated counterfactual policy replay."""
from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.counterfactual_policy_replay import (
    compare_counterfactual_policies,
    replay_counterfactual_episode,
    temporal_episode_split,
)


CUTOFF = "2026-10-01T12:00:00Z"


def _episode(*, sequence_losses=None, action_returns=None):
    return {
        "episode_id": "replay-episode-1",
        "public": {
            "target_key": "match-1",
            "kickoff": "2026-10-01T18:00:00Z",
            "prediction_cutoff": CUTOFF,
            "initial_context": {"qualification_state": "unknown", "tactical_layout": "unknown"},
            "candidate_generation_complete": True,
            "budget": .2,
            "loss_unit": "decision_loss",
            "candidate_menu": [
                {"candidate_id": "weather", "cost": .1},
                {"candidate_id": "lineup", "cost": .1},
            ],
        },
        "evaluation_sidecar": {
            "label_status": "INDEPENDENT_GOLD",
            "baseline_loss": .8,
            "cost_to_loss_rate": 1.0,
            "sequence_losses": sequence_losses or {
                "": .8, "weather>lineup": .6, "lineup>weather": .2, "weather": .7, "lineup": .4,
            },
            "action_returns": action_returns or {
                "weather": {"available": True, "context_update": {"tactical_layout": "clear"},
                            "observed_at": "2026-10-01T09:00:00Z", "available_at": "2026-10-01T09:05:00Z"},
                "lineup": {"available": True, "context_update": {"qualification_state": "already_qualified"},
                           "observed_at": "2026-10-01T09:00:00Z", "available_at": "2026-10-01T09:05:00Z"},
            },
        },
    }


def test_fixed_order_and_full_query_have_distinct_declared_control_semantics():
    episode = _episode()
    fixed = replay_counterfactual_episode(episode, "fixed_order")
    full = replay_counterfactual_episode(episode, "full_query")
    assert fixed["actor_trace"]["selected_actions"] == ["weather", "lineup"]
    assert full["actor_trace"]["selected_actions"] == ["lineup", "weather"]
    assert fixed["evaluator"]["final_loss"] == .6
    assert full["evaluator"]["final_loss"] == .2
    assert "sequence_losses" not in str(fixed["actor_trace"])
    assert "action_returns" not in str(fixed["actor_trace"])


def test_interval_policy_replans_only_after_selected_transition():
    episode = _episode()
    seen = []

    def provider(state):
        seen.append(deepcopy(state))
        assert "evaluation_sidecar" not in state
        assert "sequence_losses" not in state
        result = {}
        for cid in state["remaining_candidate_ids"]:
            result[cid] = {"state_digest": state["state_digest"], "lower_bound": 0.0, "upper_bound": .02}
        if state["context"]["qualification_state"] == "unknown":
            result["lineup"] = {"state_digest": state["state_digest"], "lower_bound": .3, "upper_bound": .8}
        return result

    report = replay_counterfactual_episode(episode, "interval_voi", interval_provider=provider)
    assert report["actor_trace"]["selected_actions"] == ["lineup"]
    assert report["actor_trace"]["stop_reason"] == "all_upper_bounds_at_or_below_cost"
    assert report["actor_trace"]["final_context"]["qualification_state"] == "already_qualified"
    assert len(seen) == 2
    assert seen[1]["context"]["qualification_state"] == "already_qualified"
    assert report["evaluator"]["final_loss"] == .4


def test_crossing_interval_defers_without_querying():
    episode = _episode()

    def provider(state):
        return {cid: {"state_digest": state["state_digest"], "lower_bound": .0, "upper_bound": .2}
                for cid in state["remaining_candidate_ids"]}

    report = replay_counterfactual_episode(episode, "interval_voi", interval_provider=provider)
    assert report["actor_trace"]["selected_actions"] == []
    assert report["actor_trace"]["stop_reason"] == "interval_crosses_cost_or_support_unknown"
    assert report["evaluator"]["final_loss"] == .8


def test_missing_sequence_loss_is_counterfactual_ineligible_and_stays_unknown():
    episode = _episode(sequence_losses={"": .8, "lineup": .4})
    report = replay_counterfactual_episode(episode, "fixed_order")
    assert report["actor_trace"]["selected_actions"] == ["weather", "lineup"]
    assert report["evaluator"]["counterfactual_eligible"] is False
    assert report["evaluator"]["final_loss"] is None
    assert report["evaluator"]["decision_loss_reduction"] is None


def test_missing_action_specific_transition_marks_trace_ineligible():
    episode = _episode(action_returns={"lineup": {"available": True, "context_update": {}}})
    report = replay_counterfactual_episode(episode, "fixed_order")
    assert report["actor_trace"]["selected_actions"] == ["weather"]
    assert report["actor_trace"]["stop_reason"] == "missing_action_specific_transition"
    assert report["evaluator"]["counterfactual_eligible"] is False
    assert report["evaluator"]["final_loss"] is None


def test_public_snapshot_cannot_contain_sidecar_fields():
    episode = _episode()
    episode["public"]["initial_context"]["result"] = "home"
    with pytest.raises(ValueError, match="evaluator-only"):
        replay_counterfactual_episode(episode, "fixed_order")


def test_paired_comparison_keeps_unknown_deltas_out_of_zero_average():
    complete = _episode()
    partial = _episode(sequence_losses={"": .8, "weather": .7, "lineup": .4})
    report = compare_counterfactual_policies([complete, partial], policies=("fixed_order", "interval_voi"),
                                             interval_provider=lambda state: {
                                                 cid: {"state_digest": state["state_digest"], "lower_bound": .3,
                                                       "upper_bound": .8} if cid == "lineup" else
                                                 {"state_digest": state["state_digest"], "lower_bound": 0.0,
                                                  "upper_bound": .01}
                                                 for cid in state["remaining_candidate_ids"]})
    deltas = report["paired_deltas"]["interval_voi"]
    assert len(deltas) == 2
    assert all("counterfactual_eligible" in row for row in deltas)


def test_temporal_split_excludes_evaluation_target_from_training_history():
    old_target = _episode()
    old_target["episode_id"] = "old-target"
    old_target["public"]["prediction_cutoff"] = "2026-09-29T12:00:00Z"
    old_same_target = deepcopy(old_target)
    old_same_target["episode_id"] = "old-same-target"
    old_other_target = deepcopy(old_target)
    old_other_target["episode_id"] = "old-other-target"
    old_other_target["public"]["target_key"] = "old-match"
    future = deepcopy(old_target)
    future["episode_id"] = "future-eval"
    future["public"]["target_key"] = "new-match"
    future["public"]["prediction_cutoff"] = CUTOFF
    split = temporal_episode_split(
        [old_target, old_same_target, old_other_target, future], calibration_as_of="2026-09-30T00:00:00Z",
        evaluation_target_keys=("match-1",), evaluation_cutoff="2026-10-02T00:00:00Z",
    )
    assert split["train_episode_ids"] == ["old-other-target"]
    assert split["evaluation_episode_ids"] == ["future-eval"]
    assert {row["episode_id"] for row in split["excluded"]} == {"old-target", "old-same-target"}
