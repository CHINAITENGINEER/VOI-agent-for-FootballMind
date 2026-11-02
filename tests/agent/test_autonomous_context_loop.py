from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from football_agent.autonomous_context_loop import LoopReturn, VERSION, run_autonomous_context_loop
from football_agent.query_execution_envelope import QueryExecutionLedger
from football_agent.query_value_contract import DEFER
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate


CUTOFF = "2026-09-29T12:00:00Z"
TARGET = "loop-match"


def board():
    return StateQueryBoard(
        target_key=TARGET, prediction_cutoff=CUTOFF,
        hypotheses=[
            StateHypothesis("qualification-h", TARGET, "qualification", ("known", "unknown")),
            StateHypothesis("lineup-h", TARGET, "lineup", ("known", "unknown")),
        ],
        candidates=[
            StateQueryCandidate("qualification-refresh", TARGET, CUTOFF, "qualification", "standings", "read", .01),
            StateQueryCandidate("lineup", TARGET, CUTOFF, "lineup", "lineup", "read", .01),
        ],
    )


def routes():
    return (
        {"candidate_id": "qualification-refresh", "route_kind": "context",
         "required_context_keys": (), "provides_context_keys": ("qualification",)},
        {"candidate_id": "lineup", "route_kind": "domain",
         "required_context_keys": ("fixture_id", "competition_id", "qualification")},
    )


def returns():
    stamp = CUTOFF
    return {
        "qualification-refresh": LoopReturn(
            "qualification-refresh", {"qualification": "already_qualified"},
            {"source_observation": "standings"}, ("raw:standings",),
            observed_at=stamp, operational_available_at=stamp, captured_at=stamp,
        ),
        "lineup": LoopReturn(
            "lineup", {"lineup": "known"},
            {"source_observation": "lineup"}, ("raw:lineup",),
            observed_at=stamp, operational_available_at=stamp, captured_at=stamp,
        ),
    }


def test_loop_replans_after_context_return_and_then_unlocks_domain_route():
    result = run_autonomous_context_loop(
        board=board(), initial_context={"fixture_id": "fixture", "competition_id": "competition"},
        routes=routes(), values={"qualification-refresh": .04, "lineup": .02}, returns=returns(),
        episode_id="loop-test",
    )
    assert result.selected_candidates == ("qualification-refresh", "lineup")
    assert result.statuses == ("SUCCEEDED", "SUCCEEDED")
    assert result.final_context["qualification"] == "already_qualified"
    assert result.final_context["lineup"] == "known"


def test_loop_transition_replay_has_selection_update_pairs_and_stop():
    result = run_autonomous_context_loop(
        board=board(), initial_context={"fixture_id": "fixture", "competition_id": "competition"},
        routes=routes(), values={"qualification-refresh": .04, "lineup": .02}, returns=returns(),
        episode_id="loop-test-events",
    )
    assert result.transition_events == (
        "INITIALIZED", "QUERY_SELECTED", "CONTEXT_UPDATED",
        "QUERY_SELECTED", "CONTEXT_UPDATED", "STOPPED",
    )
    assert result.execution_entries == 4


def test_loop_records_real_initial_evidence_refs():
    result = run_autonomous_context_loop(
        board=board(), initial_context={"fixture_id": "fixture", "competition_id": "competition"},
        routes=routes(), values={"qualification-refresh": .04, "lineup": .02}, returns=returns(),
        episode_id="loop-test-evidence", initial_evidence_refs=("raw:fixture",),
    )
    assert result.selected_candidates == ("qualification-refresh", "lineup")


def test_loop_stops_when_gate_has_no_positive_candidate():
    result = run_autonomous_context_loop(
        board=board(), initial_context={"fixture_id": "fixture", "competition_id": "competition"},
        routes=routes(), values={"qualification-refresh": .005, "lineup": .005}, returns=returns(),
        episode_id="loop-test-stop",
    )
    assert result.selected_candidates == ()
    assert result.statuses == ()
    assert result.stop_reason
    assert result.transition_events == ("INITIALIZED", "STOPPED")


def test_loop_output_has_explicit_schema_and_no_hidden_result_flag():
    result = run_autonomous_context_loop(
        board=board(), initial_context={"fixture_id": "fixture", "competition_id": "competition"},
        routes=routes(), values={"qualification-refresh": .04, "lineup": .02}, returns=returns(),
        episode_id="loop-test-json",
    )
    payload = result.as_dict()
    assert payload["schema_version"] == VERSION
    assert payload["hidden_result_included"] is False
    json.dumps(payload, ensure_ascii=False)


def test_missing_return_capture_cannot_update_existing_context_or_unlock_a_new_request(tmp_path):
    initial = {"fixture_id": "fixture", "competition_id": "competition", "qualification": "unknown"}
    responses = returns()
    responses["qualification-refresh"] = replace(responses["qualification-refresh"], captured_at=None)
    result = run_autonomous_context_loop(
        board=board(), initial_context=initial, routes=routes(),
        values={"qualification-refresh": .04, "lineup": .02}, returns=responses,
        episode_id="loop-missing-capture", root=tmp_path,
    )

    assert result.selected_candidates == ("qualification-refresh",)
    assert result.statuses == ("FAILED",)
    assert result.final_action == DEFER
    assert result.final_context == initial
    assert result.remaining_budget > .02
    assert result.transition_events == ("INITIALIZED", "QUERY_SELECTED", "REPLANNED", "STOPPED")
    planned, failed = QueryExecutionLedger(tmp_path / "executions").entries()
    assert planned.status == "PLANNED"
    assert failed.status == "FAILED"
    assert failed.observed_at is None
    assert failed.operational_available_at is None
    assert not failed.context_updates


@pytest.mark.parametrize("mode", ["exception", "hidden_payload", "mismatched_candidate"])
def test_execution_or_validation_failure_records_local_capture_without_source_timestamps(tmp_path, mode):
    initial = {"fixture_id": "fixture", "competition_id": "competition", "qualification": "unknown"}
    calls = []

    def fail(plan, context):
        calls.append(plan.candidate_id)
        if mode == "exception":
            raise RuntimeError("private provider response")
        response = returns()[plan.candidate_id]
        if mode == "hidden_payload":
            return replace(response, result_payload={"outcome": "private provider response"})
        return replace(response, candidate_id="wrong-candidate")

    before = datetime.now(timezone.utc)
    result = run_autonomous_context_loop(
        board=board(), initial_context=initial, routes=routes(),
        values={"qualification-refresh": .04, "lineup": .02}, execute=fail,
        episode_id="loop-exception-capture", root=tmp_path,
    )
    after = datetime.now(timezone.utc)

    assert calls == ["qualification-refresh"]
    assert result.statuses == ("FAILED",)
    assert result.final_context == initial
    assert result.final_action == DEFER
    planned, failed = QueryExecutionLedger(tmp_path / "executions").entries()
    assert failed.query_id == planned.query_id
    assert failed.status == "FAILED"
    assert failed.error_code == "EXECUTION_OR_VALIDATION_FAILED"
    assert before <= datetime.fromisoformat(failed.captured_at) <= after
    assert failed.observed_at is None
    assert failed.operational_available_at is None
    assert not failed.context_updates
    assert "private provider response" not in json.dumps([result.as_dict(), failed.as_dict()])
