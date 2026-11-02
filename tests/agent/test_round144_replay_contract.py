"""Round144 contract checks for Round143 replay state and route guards."""
from __future__ import annotations

from football_agent.counterfactual_policy_replay import replay_counterfactual_episode


def _episode(action_return):
    return {
        "episode_id": "contract-episode",
        "public": {
            "target_key": "match-contract",
            "kickoff": "2026-10-01T18:00:00Z",
            "prediction_cutoff": "2026-10-01T12:00:00Z",
            "initial_context": {"qualification_state": "unknown"},
            "candidate_generation_complete": True,
            "budget": .1,
            "candidate_menu": [{"candidate_id": "lineup", "cost": .1,
                                "source_id": "lineup-v1", "action_id": "query-lineup"}],
        },
        "evaluation_sidecar": {
            "label_status": "INDEPENDENT_GOLD",
            "baseline_loss": .8,
            "sequence_losses": {"lineup": .4},
            "action_returns": {"lineup": action_return},
        },
    }


def test_previous_state_digest_mismatch_stops_selected_transition():
    report = replay_counterfactual_episode(_episode({
        "available": True, "context_update": {}, "previous_state_digest": "stale",
    }), "fixed_order")
    assert report["actor_trace"]["transition_statuses"] == ["previous_state_digest_mismatch"]
    assert report["evaluator"]["counterfactual_eligible"] is False


def test_source_and_action_routes_are_bound_to_public_menu():
    report = replay_counterfactual_episode(_episode({
        "available": True, "context_update": {}, "source_id": "other-source", "action_id": "query-lineup",
    }), "fixed_order")
    assert report["actor_trace"]["transition_statuses"] == ["source_route_mismatch"]
    assert report["evaluator"]["counterfactual_eligible"] is False


def test_next_state_digest_mismatch_stops_after_selected_action():
    report = replay_counterfactual_episode(_episode({
        "available": True, "context_update": {"qualification_state": "known"},
        "next_state_digest": "wrong-next-digest",
    }), "fixed_order")
    assert report["actor_trace"]["transition_statuses"] == ["next_state_digest_mismatch"]
    assert report["actor_trace"]["selected_actions"] == ["lineup"]
    assert report["evaluator"]["counterfactual_eligible"] is False
