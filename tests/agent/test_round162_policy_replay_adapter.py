"""Round162 tests for the matched counterfactual policy adapter."""
from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.round162_policy_replay_adapter import (
    INDEPENDENT_GOLD,
    POLICIES,
    SYNTHETIC_OFFLINE,
    compare_policies,
    runtime_projection,
)


CUTOFF = "2026-10-01T12:00:00Z"


def _episode(*, label_status=SYNTHETIC_OFFLINE, sequence_losses=None, episode_id="round162-episode"):
    return {
        "episode_id": episode_id,
        "public": {
            "target_key": "match-1",
            "kickoff": "2026-10-01T18:00:00Z",
            "prediction_cutoff": CUTOFF,
            "initial_context": {"qualification_state": "unknown", "tactical_layout": "unknown"},
            "candidate_generation_complete": True,
            "budget": .2,
            "loss_unit": "decision_loss",
            "candidate_menu": [
                {"candidate_id": "weather", "cost": .1, "source_id": "weather"},
                {"candidate_id": "lineup", "cost": .1, "source_id": "lineup"},
            ],
        },
        "evaluation_sidecar": {
            "label_status": label_status,
            "baseline_loss": .8,
            "cost_to_loss_rate": 1.0,
            "sequence_losses": sequence_losses or {
                "": .8, "weather>lineup": .6, "lineup>weather": .2,
                "weather": .7, "lineup": .4,
            },
            "action_returns": {
                "weather": {"available": True, "context_update": {"tactical_layout": "clear"},
                            "observed_at": "2026-10-01T09:00:00Z", "available_at": "2026-10-01T09:05:00Z"},
                "lineup": {"available": True, "context_update": {"qualification_state": "already_qualified"},
                            "observed_at": "2026-10-01T09:00:00Z", "available_at": "2026-10-01T09:05:00Z"},
            },
        },
    }


def _interval_provider(state):
    result = {}
    for cid in state["remaining_candidate_ids"]:
        result[cid] = {"state_digest": state["state_digest"], "lower_bound": 0.0, "upper_bound": .01}
    if state["context"]["qualification_state"] == "unknown" and "lineup" in result:
        result["lineup"] = {"state_digest": state["state_digest"], "lower_bound": .3, "upper_bound": .8}
    return result


def test_synthetic_replay_requires_explicit_offline_opt_in():
    with pytest.raises(ValueError, match="allow_offline"):
        compare_policies([_episode()])


def test_three_policies_share_one_public_episode_and_have_distinct_controls():
    report = compare_policies([_episode()], interval_provider=_interval_provider, allow_offline=True)
    assert tuple(report["policy_order"]) == POLICIES
    selected = {row["policy"]: row["selected_actions"] for row in report["actor_contracts"]}
    assert selected["fixed_order"] == ["weather", "lineup"]
    assert selected["full_query"] == ["lineup", "weather"]
    assert selected["interval_voi"] == ["lineup"]
    assert report["contract"]["same_public_episode_menu_cutoff_budget"] is True


def test_synthetic_report_is_structural_only():
    report = compare_policies([_episode()], interval_provider=_interval_provider, allow_offline=True)
    assert report["evaluation_scope"] == "synthetic_structural_mechanism"
    assert report["synthetic_or_offline_only"] is True
    assert report["numeric_voi_training_eligible"] is False
    assert all(row["numeric_performance_reportable"] is False for row in report["paired_summary"].values())
    assert all(row["mean_final_loss_delta"] is None for row in report["paired_summary"].values())


def test_public_manifest_digest_is_shared_by_policy_replay():
    report = compare_policies([_episode()], interval_provider=_interval_provider, allow_offline=True)
    assert len(report["manifests"]) == 1
    assert len(report["manifests"][0]["public_snapshot_digest"]) == 64
    assert report["manifests"][0]["candidate_ids"] == ["weather", "lineup"]


def test_actor_contract_excludes_sidecar_and_hidden_outcome_fields():
    report = compare_policies([_episode()], interval_provider=_interval_provider, allow_offline=True)
    encoded = str(report["actor_contracts"]).lower()
    assert "sequence_losses" not in encoded
    assert "action_returns" not in encoded
    assert "final_loss" not in encoded
    assert all(row["hidden_result_included"] is False for row in report["actor_contracts"])


def test_runtime_projection_contains_actor_trace_only():
    report = compare_policies([_episode()], interval_provider=_interval_provider, allow_offline=True)
    projection = runtime_projection(report)
    encoded = str(projection).lower()
    assert projection["projection_scope"] == "runtime_actor_trace"
    assert "sequence_losses" not in encoded
    assert "evaluator" not in encoded
    assert "final_loss" not in encoded
    assert projection["hidden_result_included"] is False


def test_unknown_sequence_loss_is_excluded_from_paired_summary():
    partial = _episode(sequence_losses={"": .8, "lineup": .4})
    report = compare_policies([partial], interval_provider=_interval_provider, allow_offline=True)
    assert all(value["paired_eligible_count"] == 0 for value in report["paired_summary"].values())
    assert all(value["paired_ineligible_count"] == 1 for value in report["paired_summary"].values())


def test_missing_action_transition_is_not_promoted_to_zero_loss():
    episode = _episode()
    episode["evaluation_sidecar"]["action_returns"].pop("weather")
    report = compare_policies([episode], interval_provider=_interval_provider, allow_offline=True)
    assert report["eligible_episode_count_by_policy"]["fixed_order"] == 0
    assert report["eligible_episode_count_by_policy"]["full_query"] == 0


def test_late_action_transition_is_ineligible():
    episode = _episode()
    episode["evaluation_sidecar"]["action_returns"]["weather"]["available_at"] = "2026-10-01T13:00:01Z"
    report = compare_policies([episode], interval_provider=_interval_provider, allow_offline=True)
    assert report["eligible_episode_count_by_policy"]["fixed_order"] == 0
    assert report["eligible_episode_count_by_policy"]["full_query"] == 0
    assert report["eligible_episode_count_by_policy"]["interval_voi"] == 1


def test_interval_provider_receives_public_state_without_sidecar():
    seen = []

    def provider(state):
        seen.append(deepcopy(state))
        assert "evaluation_sidecar" not in state
        assert "sequence_losses" not in state
        return _interval_provider(state)

    compare_policies([_episode()], interval_provider=provider, allow_offline=True)
    assert seen
    assert all("episode_id" in state and "context" in state for state in seen)


def test_interval_provider_replans_after_transition():
    report = compare_policies([_episode()], interval_provider=_interval_provider, allow_offline=True)
    interval_rows = [row for row in report["actor_contracts"] if row["policy"] == "interval_voi"]
    assert interval_rows[0]["stop_reason"] == "all_upper_bounds_at_or_below_cost"
    assert interval_rows[0]["query_count"] == 1


def test_mixed_label_status_batch_is_rejected():
    gold = _episode(label_status=INDEPENDENT_GOLD, episode_id="gold")
    offline = _episode(episode_id="offline")
    with pytest.raises(ValueError, match="one evaluation label status"):
        compare_policies([gold, offline], allow_offline=True)


def test_independent_gold_requires_no_offline_opt_in():
    report = compare_policies([_episode(label_status=INDEPENDENT_GOLD)], interval_provider=_interval_provider)
    assert report["evaluation_scope"] == "independent_gold_performance"
    assert report["synthetic_or_offline_only"] is False
    assert all(value["numeric_performance_reportable"] is True for value in report["paired_summary"].values())


def test_unsupported_label_status_is_rejected():
    with pytest.raises(ValueError, match="unsupported evaluation label status"):
        compare_policies([_episode(label_status="synthetic_fixture")], allow_offline=True)


def test_public_outcome_field_is_rejected_before_replay():
    episode = _episode()
    episode["public"]["outcome"] = "home"
    with pytest.raises(ValueError, match="evaluator fields"):
        compare_policies([episode], allow_offline=True)


def test_duplicate_episode_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate episode_id"):
        compare_policies([_episode(), deepcopy(_episode())], allow_offline=True)


def test_empty_episode_batch_is_rejected():
    with pytest.raises(ValueError, match="at least one episode"):
        compare_policies([], allow_offline=True)
