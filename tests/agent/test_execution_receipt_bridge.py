"""Offline execution receipt and context-rebuild contract tests."""
from copy import deepcopy
import pytest

from football_agent.autonomous_acquisition_step import (
    prepare_acquisition_state, select_acquisition_action,
)
from football_agent.execution_receipt_bridge import (
    ACCEPTED, AMBIGUOUS, COMPLETED, FAILED, LATE,
    accept_action, complete_action, rebuild_context_after_receipt,
)
from football_agent.platform_acquisition_context import build_platform_acquisition_context

CUTOFF = "2026-10-01T12:00:00Z"


def match():
    return {"match_id": "fixture-137", "kickoff": "2026-10-01T18:00:00Z",
            "prediction_cutoff": CUTOFF, "observed_at": "2026-10-01T08:00:00Z",
            "ingested_at": "2026-10-01T08:01:00Z", "home_team_id": "home",
            "away_team_id": "away", "home_score": 2, "away_score": 1}


def evidence(sid="injury", **extra):
    row = {"source_id": sid, "kind": "injury",
           "observed_at": "2026-10-01T09:00:00Z",
           "ingested_at": "2026-10-01T09:01:00Z",
           "availability": "available", "freshness": "fresh",
           "digest": "a" * 64, "values": {"starter_status": "unknown"}}
    row.update(extra)
    return row


def trace(rows=()):
    actor = build_platform_acquisition_context(match(), rows)["actor_context"]
    state = prepare_acquisition_state(actor, [{
        "candidate_id": "injury", "action_kind": "query", "source_ids": ["injury"],
        "action_availability": True, "resource_cost": {"http_calls": 1},
        "utility_cost": {"amount": .1, "unit": "decision_loss", "source": "test"},
    }], candidate_generation_complete=True)
    estimate = {"candidate_id": "injury", "state_digest": state["state_digest"],
                "expected_value": .4, "unit": "decision_loss", "source": "test-model",
                "training_cutoff": "2026-09-01T00:00:00Z",
                "available_at": "2026-09-30T00:00:00Z"}
    return select_acquisition_action(state, loss_unit="decision_loss",
                                    estimates={"injury": estimate},
                                    remaining_budget={"http_calls": 2}), actor


def test_acceptance_binds_trace_and_does_not_mark_spend():
    selected, _ = trace()
    receipt = accept_action(selected, provider_id="fixture-provider",
                            accepted_at="2026-10-01T10:00:00Z")
    assert receipt["status"] == ACCEPTED
    assert receipt["actual_resource_cost"] is None
    assert receipt["execution_receipt_id"] == receipt["receipt_id"]


def test_completed_receipt_rebuilds_public_context_and_changes_digest():
    selected, actor = trace()
    accepted = accept_action(selected, provider_id="fixture-provider",
                             accepted_at="2026-10-01T10:00:00Z")
    completed = complete_action(
        accepted, result_payload={"rotation_confirmed": True},
        evidence_rows=[evidence(values={"rotation_confirmed": True})],
        actual_resource_cost={"http_calls": 1, "local_reads": 2},
        observed_at="2026-10-01T10:30:00Z",
        operational_available_at="2026-10-01T10:31:00Z",
        captured_at="2026-10-01T10:31:01Z",
    )
    assert completed["status"] == COMPLETED
    assert completed["actual_resource_cost"] == {"http_calls": 1.0, "local_reads": 2.0}
    rebuilt = rebuild_context_after_receipt(actor, completed, match())
    assert rebuilt["context_changed"] is True
    assert rebuilt["replanning_required"] is False
    assert rebuilt["new_context_digest"] != actor["context_digest"]
    assert any(row["source_id"] == "injury" for row in rebuilt["actor_context"]["source_observations"])


@pytest.mark.parametrize("status,expected_replan", [(FAILED, True), (LATE, True), (AMBIGUOUS, True)])
def test_noncompleted_receipt_never_updates_actor_context(status, expected_replan):
    selected, actor = trace()
    accepted = accept_action(selected, provider_id="fixture-provider",
                             accepted_at="2026-10-01T10:00:00Z")
    kwargs = {"status": status, "error_code": "PROVIDER_TIMEOUT" if status == FAILED else None,
              "result_payload": {"rotation_confirmed": True},
              "evidence_rows": [evidence()], "actual_resource_cost": {"http_calls": 1},
              "observed_at": "2026-10-01T10:30:00Z",
              "operational_available_at": "2026-10-01T13:00:00Z" if status == LATE else "2026-10-01T10:31:00Z"}
    completed = complete_action(accepted, **kwargs)
    assert completed["status"] == status
    rebuilt = rebuild_context_after_receipt(actor, completed, match())
    assert rebuilt["context_changed"] is False
    assert rebuilt["replanning_required"] is expected_replan


def test_late_clock_is_classified_even_when_requested_as_completed():
    selected, _ = trace()
    accepted = accept_action(selected, provider_id="fixture-provider",
                             accepted_at="2026-10-01T10:00:00Z")
    completed = complete_action(
        accepted, result_payload={"rotation_confirmed": True}, evidence_rows=[evidence()],
        actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T12:30:00Z",
        operational_available_at="2026-10-01T13:00:00Z", captured_at="2026-10-01T13:00:01Z",
        status=COMPLETED)
    assert completed["status"] == LATE


def test_result_like_fields_and_tampering_are_rejected():
    selected, _ = trace()
    accepted = accept_action(selected, provider_id="fixture-provider",
                             accepted_at="2026-10-01T10:00:00Z")
    with pytest.raises(ValueError, match="evaluator"):
        complete_action(accepted, result_payload={"score": 2}, evidence_rows=[evidence()],
                        actual_resource_cost={"http_calls": 1},
                        observed_at="2026-10-01T10:30:00Z",
                        operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T10:31:01Z")
    changed = dict(accepted)
    changed["provider_id"] = "other"
    with pytest.raises(ValueError, match="digest"):
        complete_action(changed, result_payload={"rotation_confirmed": True},
                        evidence_rows=[evidence()], actual_resource_cost={"http_calls": 1},
                        observed_at="2026-10-01T10:30:00Z",
                        operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T10:31:01Z")


def test_old_context_receipt_cannot_be_applied_to_new_context():
    selected, actor = trace()
    accepted = accept_action(selected, provider_id="fixture-provider",
                             accepted_at="2026-10-01T10:00:00Z")
    completed = complete_action(
        accepted, result_payload={"rotation_confirmed": True}, evidence_rows=[evidence()],
        actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:30:00Z",
        operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T10:31:01Z")
    changed_actor = build_platform_acquisition_context(match(), [evidence(digest="b" * 64)])[
        "actor_context"]
    with pytest.raises(ValueError, match="context digest"):
        rebuild_context_after_receipt(changed_actor, completed, match())


def test_terminal_receipt_cannot_be_completed_twice():
    selected, _ = trace()
    accepted = accept_action(selected, provider_id="fixture-provider",
                             accepted_at="2026-10-01T10:00:00Z")
    completed = complete_action(
        accepted, result_payload={"rotation_confirmed": True}, evidence_rows=[evidence()],
        actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:30:00Z",
        operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T10:31:01Z")
    with pytest.raises(ValueError, match="accepted receipt"):
        complete_action(completed, result_payload={"rotation_confirmed": False},
                        evidence_rows=[evidence()], actual_resource_cost={"http_calls": 1},
                        observed_at="2026-10-01T10:30:00Z",
                        operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T10:31:01Z")


@pytest.mark.parametrize("status", [None, COMPLETED])
def test_missing_capture_does_not_invent_success_or_update_context(status):
    selected, actor = trace()
    accepted = accept_action(selected, provider_id="offline", accepted_at="2026-10-01T10:00:00Z")
    closed = complete_action(
        accepted, result_payload={"ok": True}, evidence_rows=[evidence()],
        actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T09:00:00Z",
        operational_available_at="2026-10-01T09:01:00Z", status=status)
    assert closed["status"] == AMBIGUOUS
    assert closed["captured_at"] is None
    assert closed["clock_status"] == "capture_unknown"
    assert closed["actual_resource_cost"] == {"http_calls": 1}
    assert rebuild_context_after_receipt(actor, closed, match())["actor_context"] == actor


@pytest.mark.parametrize("status", [None, COMPLETED])
def test_known_lateness_is_preserved_even_when_capture_is_unknown(status):
    selected, _ = trace()
    accepted = accept_action(selected, provider_id="offline", accepted_at="2026-10-01T10:00:00Z")
    closed = complete_action(
        accepted, result_payload={}, evidence_rows=[], actual_resource_cost={"http_calls": 1},
        observed_at="2026-10-01T11:00:00Z", operational_available_at="2026-10-01T13:00:00Z",
        status=status)
    assert closed["status"] == LATE
    assert closed["captured_at"] is None
    assert closed["late_reason"] == "operational_available_after_cutoff"


def test_completed_receipt_does_not_share_nested_mutable_reservations():
    selected, _ = trace()
    accepted = accept_action(selected, provider_id="offline", accepted_at="2026-10-01T10:00:00Z")
    closed = complete_action(
        accepted, result_payload={}, evidence_rows=[evidence()], actual_resource_cost={"http_calls": 1},
        observed_at="2026-10-01T10:30:00Z", operational_available_at="2026-10-01T10:31:00Z",
        captured_at="2026-10-01T10:31:01Z")
    saved = deepcopy(closed)
    accepted["declared_resource_cost"]["http_calls"] = 999
    assert closed == saved
