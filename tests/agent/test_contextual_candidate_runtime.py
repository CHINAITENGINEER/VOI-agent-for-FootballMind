"""Runtime boundary tests for context-conditioned candidate proposals.

The only values eligible for proposal ranking come from cutoff-visible public
observations.  These tests never provide evaluator labels to the proposal
function and pass its returned acquisition state through the real action gate.
"""
from copy import deepcopy

import pytest

from football_agent.autonomous_acquisition_step import select_acquisition_action
from football_agent.contextual_candidate_generation import prepare_contextual_acquisition
from football_agent.execution_receipt_bridge import accept_action, complete_action, rebuild_context_after_receipt
from football_agent.platform_acquisition_context import build_platform_acquisition_context


CUTOFF = "2026-10-01T12:00:00Z"


def _match():
    return {
        "match_id": "runtime-context-match",
        "kickoff": "2026-10-01T18:00:00Z",
        "prediction_cutoff": CUTOFF,
        "observed_at": "2026-10-01T08:00:00Z",
        "ingested_at": "2026-10-01T08:01:00Z",
        "home_team_id": "home",
        "away_team_id": "away",
    }


def _row(source_id, context, *, observed_at="2026-10-01T10:00:00Z",
         valid_until="2026-10-02T00:00:00Z"):
    return {
        "source_id": source_id,
        "kind": "pre_match_context",
        "observed_at": observed_at,
        "ingested_at": observed_at,
        "freshness": "fresh",
        "availability": "available",
        "valid_until": valid_until,
        "digest": (source_id[0] * 64),
        "values": {"context": dict(context)},
    }


def _actor(rows):
    return build_platform_acquisition_context(_match(), rows)["actor_context"]


def _registry(*, source_ids=("missing-lineup",)):
    return [
        {
            "candidate_id": "lineup",
            "source_ids": list(source_ids),
            "kind": "lineup",
            "action_availability": True,
            "context_affinity": {"rotation_risk": ["high"]},
            "resource_cost": {"http_calls": 1},
            "utility_cost": {
                "amount": 0.1,
                "unit": "decision_loss",
                "source": "declared_conversion",
            },
        },
        {
            "candidate_id": "weather",
            "source_ids": ["missing-weather"],
            "kind": "weather",
            "action_availability": True,
            "context_affinity": {"weather_state": ["severe"]},
            "resource_cost": {"http_calls": 1},
            "utility_cost": {
                "amount": 0.1,
                "unit": "decision_loss",
                "source": "declared_conversion",
            },
        },
    ]


def test_cutoff_visible_values_context_binds_proposal_and_keeps_value_unestimated():
    result = prepare_contextual_acquisition(
        _actor([_row("rotation", {"rotation_risk": "high"})]),
        _registry(),
        registry_complete=False,
    )
    assert result["context_binding"]["context"] == {"rotation_risk": "high"}
    assert result["context_binding"]["supporting_source_ids"] == {"rotation_risk": ["rotation"]}
    assert result["candidate_generation"]["candidate_order"] == ["lineup", "weather"]
    assert result["candidate_generation"]["numeric_voi_estimated"] is False
    assert result["contract"]["state_bound_selection_required"] is True


def test_conflicting_cutoff_visible_context_is_unknown_and_does_not_promote_candidate():
    result = prepare_contextual_acquisition(
        _actor([
            _row("rotation-a", {"rotation_risk": "high"}),
            _row("rotation-b", {"rotation_risk": "low"}),
        ]),
        _registry(),
    )
    binding = result["context_binding"]
    assert binding["context"] == {"rotation_risk": "unknown"}
    assert binding["conflicting_fields"] == ["rotation_risk"]
    lineup = next(row for row in result["candidate_generation"]["candidates"] if row["candidate_id"] == "lineup")
    assert lineup["candidate_generation_status"] == "context_unknown"
    assert lineup["matched_context"] == []


def test_expired_context_is_ignored_and_cannot_change_priority():
    bundle = build_platform_acquisition_context(
        _match(), [_row("expired", {"rotation_risk": "high"}, valid_until="2026-10-01T11:59:00Z")]
    )
    result = prepare_contextual_acquisition(bundle["actor_context"], _registry())
    assert result["context_binding"]["context"] == {}
    assert "expired" in result["context_binding"]["ignored_source_ids"]
    assert result["candidate_generation"]["context"] == {}
    assert all(row["matched_context"] == [] for row in result["candidate_generation"]["candidates"])


def test_future_observation_is_excluded_before_context_binding():
    bundle = build_platform_acquisition_context(
        _match(),
        [_row("future", {"rotation_risk": "high"}, observed_at="2026-10-01T13:00:00Z")],
    )
    result = prepare_contextual_acquisition(bundle["actor_context"], _registry())
    assert result["context_binding"]["context"] == {}
    assert result["candidate_generation"]["context"] == {}
    assert all(row["matched_context"] == [] for row in result["candidate_generation"]["candidates"])


def test_missing_source_does_not_promote_high_affinity_candidate():
    registry = _registry()
    registry[0]["candidate_id"] = "z_lineup"
    registry[0]["source_ids"] = ["missing-rotation"]
    registry[1]["candidate_id"] = "a_weather"
    result = prepare_contextual_acquisition(_actor([]), registry)
    assert result["context_binding"]["context"] == {}
    assert result["candidate_generation"]["candidate_order"] == ["a_weather", "z_lineup"]
    assert next(row for row in result["candidate_generation"]["candidates"]
                if row["candidate_id"] == "z_lineup")["matched_context"] == []


def test_state_can_enter_real_gate_without_context_priority_becoming_expected_value():
    result = prepare_contextual_acquisition(
        _actor([_row("rotation", {"rotation_risk": "high"})]),
        _registry(),
        registry_complete=False,
    )
    state = result["acquisition_state"]
    decision = select_acquisition_action(
        state,
        loss_unit="decision_loss",
        estimates={},
        remaining_budget={"http_calls": 2},
    )
    assert decision["decision"] == "defer"
    assert decision["selected_candidate_id"] is None
    assert result["candidate_generation"]["numeric_voi_estimated"] is False
    assert all(row["assessment"]["expected_value"] is None for row in decision["evaluations"])


def test_context_binding_reads_hashed_snapshot_not_unhashed_actor_projection():
    actor = _actor([_row("rotation", {"rotation_risk": "high"})])
    prepared = prepare_contextual_acquisition(actor, _registry())
    tampered = deepcopy(actor)
    tampered["source_observations"][1]["values"]["context"]["rotation_risk"] = "low"
    replay = prepare_contextual_acquisition(tampered, _registry())
    assert replay["context_binding"] == prepared["context_binding"]
    assert replay["acquisition_state"]["state_digest"] == prepared["acquisition_state"]["state_digest"]


def test_sidecar_like_fields_in_actor_observation_are_removed_before_binding():
    row = _row("rotation", {"rotation_risk": "high"})
    row["values"]["context"]["outcome"] = "home"
    actor = _actor([row])
    result = prepare_contextual_acquisition(actor, _registry())
    # A mixed result-bearing row loses its content digest and therefore is not
    # usable for actor state; dropping the whole observation is conservative.
    assert result["context_binding"]["context"] == {}
    assert "outcome" not in str(result["candidate_generation"])


def test_completed_receipt_rebuilds_state_and_changes_next_action():
    registry = [
        {"candidate_id": "qualification", "source_ids": ["qualification"], "action_kind": "query",
         "action_availability": True, "resource_cost": {"http_calls": 1},
         "utility_cost": {"amount": .1, "unit": "decision_loss", "source": "state-bound-test"},
         "context_affinity": {"qualification_state": ["already_qualified"]}},
        {"candidate_id": "tactical", "source_ids": ["tactical"], "action_kind": "query",
         "action_availability": True, "resource_cost": {"http_calls": 1},
         "utility_cost": {"amount": .1, "unit": "decision_loss", "source": "state-bound-test"},
         "context_affinity": {"tactical_layout": ["defensive"]}},
    ]
    initial = build_platform_acquisition_context(_match(), [])
    first = prepare_contextual_acquisition(initial["actor_context"], registry, registry_complete=True)
    old_state = first["acquisition_state"]

    def estimate(state, candidate_id, value):
        return {"candidate_id": candidate_id, "state_digest": state["state_digest"],
                "expected_value": value, "unit": "decision_loss", "source": "test-estimator",
                "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}

    estimates = {"qualification": estimate(old_state, "qualification", .8),
                 "tactical": estimate(old_state, "tactical", .4)}
    action = select_acquisition_action(old_state, loss_unit="decision_loss", estimates=estimates,
                                       remaining_budget={"http_calls": 2})
    assert action["decision"] == "acquire" and action["selected_candidate_id"] == "qualification"
    accepted = accept_action(action, provider_id="context-provider", accepted_at="2026-10-01T09:00:00Z")
    completed = complete_action(
        accepted, result_payload={"provider_revision": "r1"},
        evidence_rows=[_row("qualification", {"qualification_state": "already_qualified"})],
        actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T09:05:00Z",
        operational_available_at="2026-10-01T09:06:00Z",
        captured_at="2026-10-01T09:06:01Z",
    )
    rebuilt = rebuild_context_after_receipt(initial["actor_context"], completed, _match())
    assert rebuilt["context_changed"] is True
    second = prepare_contextual_acquisition(rebuilt["actor_context"], registry, registry_complete=True)
    assert second["context_binding"]["context"] == {"qualification_state": "already_qualified"}

    stale = select_acquisition_action(
        second["acquisition_state"], loss_unit="decision_loss",
        estimates={"qualification": estimate(old_state, "qualification", 0),
                   "tactical": estimate(old_state, "tactical", .9)},
        remaining_budget={"http_calls": 1},
    )
    assert stale["decision"] == "defer"
    assert all(row["assessment"]["reason"] == "estimate_state_or_candidate_mismatch"
               for row in stale["evaluations"])
    new_state = second["acquisition_state"]
    replanned = select_acquisition_action(
        new_state, loss_unit="decision_loss",
        estimates={"qualification": estimate(new_state, "qualification", 0),
                   "tactical": estimate(new_state, "tactical", .9)},
        remaining_budget={"http_calls": 1},
    )
    assert replanned["decision"] == "acquire" and replanned["selected_candidate_id"] == "tactical"

