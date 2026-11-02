from pathlib import Path
from dataclasses import replace
import hashlib
import json

import pytest

from football_agent.context_transition import ContextTransition, ContextTransitionLedger
from football_agent.query_execution_envelope import (
    AMBIGUOUS,
    FAILED,
    LATE,
    PLANNED,
    SUCCEEDED,
    QueryExecutionEnvelope,
    QueryExecutionLedger,
)
from football_agent.query_value_contract import QueryValueAssessment, QueryValueGate, QUERY
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate


CUTOFF = "2026-09-29T12:00:00Z"
TARGET = "match-envelope"


def board():
    return StateQueryBoard(
        target_key=TARGET,
        prediction_cutoff=CUTOFF,
        hypotheses=[StateHypothesis("qualification-h", TARGET, "qualification", ("already_qualified", "must_win"))],
        candidates=[StateQueryCandidate("qualification-refresh", TARGET, CUTOFF, "qualification", "standings", "read", .01)],
    )


def selected():
    decision = QueryValueGate.decide(
        board(),
        [QueryValueAssessment.modeled("qualification-refresh", expected_value=.06, cost=.01)],
        remaining_budget=.05,
    )
    assert decision.action == QUERY
    return decision


def planned():
    return QueryExecutionEnvelope.from_decision(
        selected(), board(),
        {"target": TARGET, "state_variable": "qualification", "request_nonce": "fixture-1"},
        tool_version="standings-fixture-v1",
        expected_context_keys=("qualification",),
        requested_at="2026-09-29T10:00:00Z",
        evidence_refs=("request:fixture-1",),
    )


def test_planned_envelope_carries_selection_cost_cutoff_and_request_hash():
    envelope = planned()
    assert envelope.status == PLANNED
    assert envelope.query_id.startswith("query-")
    assert envelope.target_key == TARGET
    assert envelope.expected_context_keys == ("qualification",)
    assert envelope.cost == pytest.approx(.01)
    assert len(envelope.request_hash) == 64
    assert envelope.result_hash is None
    assert envelope.as_dict()["hidden_result_included"] is False


def test_successful_fixture_result_hashes_public_result_and_updates_context():
    envelope = planned().complete(
        {"qualification": "already_qualified", "source_observation": "standing-1"},
        context_updates={"qualification": "already_qualified"},
        evidence_refs=("raw:standing-1",),
        observed_at="2026-09-29T10:01:00Z",
        operational_available_at="2026-09-29T10:01:01Z",
        captured_at="2026-09-29T10:01:02Z",
    )
    assert envelope.status == SUCCEEDED
    assert envelope.success and not envelope.late and not envelope.ambiguous
    assert len(envelope.result_hash) == 64
    assert envelope.context_updates == {"qualification": "already_qualified"}
    transition = envelope.to_context_transition(
        episode_id="episode-envelope", sequence=3,
        previous_context={"qualification": None},
    )
    assert transition.event_type == "CONTEXT_UPDATED"
    assert transition.next_context["qualification"] == "already_qualified"
    assert transition.selected_candidate_id == "qualification-refresh"
    assert "query:" + envelope.query_id in transition.evidence_refs


@pytest.mark.parametrize("status", [FAILED, LATE, AMBIGUOUS])
def test_non_successful_results_are_audit_only_and_trigger_replanning(status):
    kwargs = {
        "result_payload": {"source_observation": "fixture-failure", "status_code": 503},
        "evidence_refs": ("raw:failure",),
        "status": status,
        "error_code": "fixture_failure" if status != AMBIGUOUS else "conflicting_sources",
        "observed_at": "2026-09-29T13:01:00Z" if status == LATE else None,
        "operational_available_at": "2026-09-29T13:01:01Z" if status == LATE else None,
        "captured_at": "2026-09-29T13:01:02Z" if status == LATE else "2026-09-29T10:01:02Z",
    }
    envelope = planned().complete(**kwargs)
    assert envelope.status == status
    assert envelope.replanning_required is True
    assert envelope.context_updates == {}
    transition = envelope.to_context_transition(
        episode_id="episode-envelope", sequence=4,
        previous_context={"qualification": None},
    )
    assert transition.event_type == "REPLANNED"
    assert dict(transition.previous_context) == dict(transition.next_context)
    assert transition.trigger == f"QUERY_RESULT_{status}"


def test_late_timestamp_cannot_be_claimed_as_success():
    with pytest.raises(ValueError, match="after prediction cutoff"):
        planned().complete(
            {"qualification": "already_qualified"},
            context_updates={"qualification": "already_qualified"},
            evidence_refs=("raw:late",),
            observed_at="2026-09-29T13:00:00Z",
            operational_available_at="2026-09-29T13:00:01Z",
            captured_at="2026-09-29T13:00:02Z",
            status=SUCCEEDED,
        )


def test_success_with_error_code_is_rejected_instead_of_updating_context():
    with pytest.raises(ValueError, match="error code"):
        planned().complete(
            {"qualification": "already_qualified"},
            context_updates={"qualification": "already_qualified"},
            evidence_refs=("raw:warning",), observed_at="2026-09-29T10:01:00Z",
            captured_at="2026-09-29T10:01:02Z",
            error_code="provider_warning", status=SUCCEEDED,
        )


def test_capture_after_cutoff_is_late_even_when_operationally_available_before_cutoff():
    envelope = planned().complete(
        {"qualification": "already_qualified"},
        context_updates={"qualification": "already_qualified"},
        evidence_refs=("raw:captured-late",),
        observed_at="2026-09-29T10:01:00Z",
        operational_available_at="2026-09-29T10:01:01Z",
        captured_at="2026-09-29T13:00:00Z",
    )
    assert envelope.status == LATE
    assert envelope.replanning_required is True
    assert envelope.context_updates == {}
    transition = envelope.to_context_transition(
        episode_id="episode-envelope", sequence=4,
        previous_context={"qualification": None},
    )
    assert transition.event_type == "REPLANNED"
    assert dict(transition.previous_context) == dict(transition.next_context)


def test_explicit_late_status_is_valid_when_only_capture_crosses_cutoff():
    envelope = planned().complete(
        {"source_observation": "fixture"},
        evidence_refs=("raw:captured-late-explicit",),
        status=LATE,
        observed_at="2026-09-29T10:01:00Z",
        operational_available_at="2026-09-29T10:01:01Z",
        captured_at="2026-09-29T13:00:00Z",
    )
    assert envelope.status == LATE


def test_completed_envelope_context_updates_are_immutable_after_construction():
    envelope = planned().complete(
        {"qualification": "already_qualified"},
        context_updates={"qualification": "already_qualified"},
        evidence_refs=("raw:immutable",), observed_at="2026-09-29T10:01:00Z",
        captured_at="2026-09-29T10:01:02Z",
    )
    with pytest.raises(TypeError):
        envelope.context_updates["qualification"] = "must_win"


def test_ledger_rejects_record_with_recomputed_checksum_but_wrong_content_identity(tmp_path: Path):
    envelope = planned().complete(
        {"qualification": "already_qualified"},
        context_updates={"qualification": "already_qualified"},
        evidence_refs=("raw:identity",), observed_at="2026-09-29T10:01:00Z",
        captured_at="2026-09-29T10:01:02Z",
    )
    record = envelope.as_dict()
    record["context_updates"] = {"qualification": "must_win"}
    wrapped = {
        "schema_version": "query-execution-envelope-v1",
        "record": record,
        "record_sha256": hashlib.sha256(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    root = tmp_path / "executions"
    root.mkdir()
    (root / "query-executions.jsonl").write_text(
        json.dumps(wrapped, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        QueryExecutionLedger(root).entries()


def test_request_result_and_context_hidden_fields_are_rejected():
    with pytest.raises(ValueError, match="hidden result"):
        QueryExecutionEnvelope.from_decision(
            selected(), board(), {"score": "3:0"}, tool_version="fixture-v1",
        )
    with pytest.raises(ValueError, match="hidden result"):
        planned().complete(
            {"score": "3:0"}, context_updates={"qualification": "already_qualified"},
            evidence_refs=("raw:poison",), observed_at="2026-09-29T10:01:00Z",
            captured_at="2026-09-29T10:01:02Z",
        )
    with pytest.raises(ValueError, match="hidden result"):
        planned().complete(
            {"qualification": "already_qualified"}, context_updates={"result": "home"},
            evidence_refs=("raw:poison",), observed_at="2026-09-29T10:01:00Z",
            captured_at="2026-09-29T10:01:02Z",
        )


def test_context_update_must_match_declared_expected_keys():
    with pytest.raises(ValueError, match="outside expected_context_keys"):
        planned().complete(
            {"qualification": "already_qualified"}, context_updates={"tactical_intent": "defensive"},
            evidence_refs=("raw:wrong-key",), observed_at="2026-09-29T10:01:00Z",
            captured_at="2026-09-29T10:01:02Z",
        )


def test_hash_and_cutoff_contracts_are_checked():
    envelope = planned()
    body = envelope.as_dict()
    body["request_hash"] = "0" * 63 + "x"
    with pytest.raises(ValueError, match="request_hash"):
        QueryExecutionEnvelope(**{key: value for key, value in body.items()
                                  if key not in {"schema_version", "hidden_result_included"}})
    with pytest.raises(ValueError, match="after prediction cutoff"):
        planned().complete(
            {"qualification": "already_qualified"}, context_updates={"qualification": "already_qualified"},
            evidence_refs=("raw:cutoff",), observed_at="2026-09-29T13:00:00Z",
            operational_available_at="2026-09-29T13:00:01Z",
            captured_at="2026-09-29T13:00:02Z", status=SUCCEEDED,
        )


def test_planned_query_cannot_be_requested_after_prediction_cutoff():
    with pytest.raises(ValueError, match="prediction cutoff"):
        QueryExecutionEnvelope.from_decision(
            selected(), board(), {"target": TARGET}, tool_version="fixture-v1",
            requested_at="2026-09-29T13:00:00Z",
        )


def test_request_and_result_payloads_are_structured_mappings():
    with pytest.raises(ValueError, match="request_payload"):
        QueryExecutionEnvelope.from_decision(
            selected(), board(), ["qualification"], tool_version="fixture-v1",
        )
    with pytest.raises(ValueError, match="result_payload"):
        planned().complete(
            ["already_qualified"], context_updates={"qualification": "already_qualified"},
            evidence_refs=("raw:bad-shape",), observed_at="2026-09-29T10:01:00Z",
            captured_at="2026-09-29T10:01:02Z",
        )


def test_envelope_ledger_and_context_transition_ledger_replay_together(tmp_path: Path):
    envelope = planned()
    successful = envelope.complete(
        {"qualification": "already_qualified"}, context_updates={"qualification": "already_qualified"},
        evidence_refs=("raw:standing-1",), observed_at="2026-09-29T10:01:00Z",
        captured_at="2026-09-29T10:01:02Z",
    )
    execution_ledger = QueryExecutionLedger(tmp_path / "executions")
    execution_ledger.append((envelope, successful))
    assert [item.status for item in execution_ledger.for_query(envelope.query_id)] == [PLANNED, SUCCEEDED]

    transition_ledger = ContextTransitionLedger(tmp_path / "transitions")
    initial = ContextTransition.from_transition(
        episode_id="episode-envelope", target_key=TARGET, sequence=1,
        event_type="INITIALIZED", previous_context={}, next_context={"qualification": None},
        trigger="fixture", reason="initial public context", prediction_cutoff=CUTOFF,
        captured_at="2026-09-29T09:59:00Z",
    )
    selected_event = ContextTransition.from_transition(
        episode_id="episode-envelope", target_key=TARGET, sequence=2,
        event_type="QUERY_SELECTED", previous_context=initial.next_context,
        next_context=initial.next_context, trigger="voi_gate", reason="positive modeled net value",
        prediction_cutoff=CUTOFF, captured_at="2026-09-29T10:00:00Z",
        selected_candidate_id=envelope.candidate_id, action=QUERY,
        source_id=envelope.tool_version, evidence_refs=envelope.evidence_refs,
    )
    updated = successful.to_context_transition(
        episode_id="episode-envelope", sequence=3, previous_context=selected_event.next_context,
    )
    transition_ledger.append((initial, selected_event, updated))
    replay = transition_ledger.replay_episode("episode-envelope", target_key=TARGET)
    assert replay["final_context"]["qualification"] == "already_qualified"
    assert [item["event_type"] for item in replay["events"]] == [
        "INITIALIZED", "QUERY_SELECTED", "CONTEXT_UPDATED",
    ]


@pytest.mark.parametrize("status", [None, SUCCEEDED, FAILED, AMBIGUOUS])
@pytest.mark.parametrize("observation", [None, "2026-09-29T10:01:00Z"])
def test_completion_never_substitutes_an_observation_or_request_for_capture(status, observation):
    with pytest.raises(ValueError, match="captured_at is required"):
        planned().complete(
            {"qualification": "already_qualified"},
            context_updates={"qualification": "already_qualified"},
            evidence_refs=("raw:no-capture",), observed_at=observation,
            operational_available_at=observation, status=status,
        )


def successful(envelope=None):
    return (envelope or planned()).complete(
        {"qualification": "already_qualified"},
        context_updates={"qualification": "already_qualified"},
        evidence_refs=("raw:standing-1",),
        observed_at="2026-09-29T10:01:00Z",
        operational_available_at="2026-09-29T10:01:01Z",
        captured_at="2026-09-29T10:01:02Z",
    )


@pytest.mark.parametrize("capture,reason", [
    ("2026-09-29T10:01:01Z", "precede execution capture"),
    ("2026-09-29T12:00:01Z", "after prediction cutoff"),
])
def test_successful_transition_cannot_backdate_capture_or_apply_after_cutoff(capture, reason):
    with pytest.raises(ValueError, match=reason):
        successful().to_context_transition(
            episode_id="capture-relabel", sequence=3,
            previous_context={"qualification": None}, captured_at=capture,
        )


def test_successful_transition_can_be_applied_later_within_its_cutoff():
    transition = successful().to_context_transition(
        episode_id="capture-apply", sequence=3,
        previous_context={"qualification": None}, captured_at=CUTOFF,
    )
    assert transition.captured_at == "2026-09-29T12:00:00+00:00"
    assert transition.next_context["qualification"] == "already_qualified"


def write_records(root, entries):
    """Build independent on-disk fixtures with a valid checksum per record."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        record = entry.as_dict()
        raw = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        rows.append(json.dumps({
            "schema_version": "query-execution-envelope-v1", "record": record,
            "record_sha256": hashlib.sha256(raw).hexdigest(),
        }))
    path = root / "query-executions.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def replacement_plan(original):
    return QueryExecutionEnvelope.from_decision(
        selected(), board(), {"request_nonce": "different-request"},
        tool_version=original.tool_version, query_id=original.query_id,
        expected_context_keys=original.expected_context_keys,
        requested_at=original.requested_at, evidence_refs=original.evidence_refs,
    )


@pytest.mark.parametrize("scenario,reason", [
    ("no-plan", "preceding planned intent"),
    ("duplicate-plan", "already has a planned intent"),
    ("duplicate-terminal", "already has a terminal receipt"),
    ("different-request", "differs from its planned intent"),
])
@pytest.mark.parametrize("boundary", ["append", "read"])
def test_ledger_checks_one_intent_and_one_bound_terminal_per_query(tmp_path, scenario, reason, boundary):
    intent = planned()
    complete = successful(intent)
    altered_plan = replacement_plan(intent)
    if scenario == "no-plan":
        prefix, final = (), complete
    elif scenario == "duplicate-plan":
        prefix, final = (intent,), altered_plan
    elif scenario == "duplicate-terminal":
        prefix = (intent, complete)
        final = intent.complete(
            {"provider_state": "unavailable"}, status=FAILED, error_code="source_failed",
            captured_at="2026-09-29T10:02:00Z",
        )
    else:
        prefix, final = (intent,), successful(altered_plan)
    ledger = QueryExecutionLedger(tmp_path / "executions")
    if boundary == "append":
        if prefix:
            ledger.append(prefix)
        original_bytes = ledger.path.read_bytes() if ledger.path.exists() else None
        with pytest.raises(ValueError, match=reason):
            ledger.append((final,))
        assert (ledger.path.read_bytes() if ledger.path.exists() else None) == original_bytes
    else:
        write_records(ledger.root, (*prefix, final))
        original_bytes = ledger.path.read_bytes()
        with pytest.raises(ValueError, match=reason):
            ledger.entries()
        assert ledger.path.read_bytes() == original_bytes


@pytest.mark.parametrize("field,value", [
    ("target_key", "another-match"),
    ("candidate_id", "another-query"),
    ("prediction_cutoff", "2026-09-29T11:00:00Z"),
    ("expected_context_keys", ("qualification", "lineup")),
    ("cost", .02),
    ("selected_by", "another-selection-basis"),
    ("tool_version", "another-adapter-version"),
    ("requested_at", "2026-09-29T10:00:01Z"),
])
@pytest.mark.parametrize("boundary", ["append", "read"])
def test_ledger_rejects_terminal_identity_cost_and_clock_drift(tmp_path, field, value, boundary):
    intent = planned()
    complete = replace(successful(intent), **{field: value})
    ledger = QueryExecutionLedger(tmp_path / "executions")
    if boundary == "append":
        ledger.append((intent,))
        original_bytes = ledger.path.read_bytes()
        with pytest.raises(ValueError, match="differs from its planned intent"):
            ledger.append((complete,))
    else:
        write_records(ledger.root, (intent, complete))
        original_bytes = ledger.path.read_bytes()
        with pytest.raises(ValueError, match="differs from its planned intent"):
            ledger.entries()
    assert ledger.path.read_bytes() == original_bytes


@pytest.mark.parametrize("boundary", ["append", "read"])
@pytest.mark.parametrize("field,value,reason", [
    ("cost", -.01, "finite and nonnegative"),
    ("captured_at", None, "captured_at is required"),
    ("captured_at", "2026-09-29T10:01:00Z", "cannot follow captured_at"),
    ("captured_at", "2026-09-29T12:01:00Z", "captured after prediction cutoff"),
    ("requested_at", "2026-09-29T10:02:00Z", "requested_at cannot follow captured_at"),
    ("error_code", "provider_warning", "cannot contain an error code"),
])
def test_ledger_revalidates_invalid_terminal_data_on_append_and_replay(tmp_path, boundary, field, value, reason):
    intent = planned()
    complete = successful(intent)
    # Simulate an object bypassing frozen assignment or a rewritten file whose
    # record checksum was recomputed; neither path may bypass data validation.
    object.__setattr__(complete, field, value)
    ledger = QueryExecutionLedger(tmp_path / "executions")
    if boundary == "append":
        ledger.append((intent,))
        original_bytes = ledger.path.read_bytes()
        with pytest.raises(ValueError, match=reason):
            ledger.append((complete,))
    else:
        write_records(ledger.root, (intent, complete))
        original_bytes = ledger.path.read_bytes()
        with pytest.raises(ValueError, match=reason):
            ledger.entries()
    assert ledger.path.read_bytes() == original_bytes
