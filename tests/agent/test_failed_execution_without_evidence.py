"""Failure receipts retain spend without inventing observations or clocks."""
from copy import deepcopy

import pytest

from football_agent.candidate_action_registry import VERSION as REQUEST_VERSION
from football_agent.execution_action_adapter import (
    complete_execution_request, normalize_provider_observation,
)
from football_agent.execution_receipt_bridge import (
    accept_action, complete_action, rebuild_context_after_receipt,
)


CUTOFF = "2026-10-01T12:00:00Z"
ACCEPTED_AT = "2026-10-01T10:00:00Z"


def bundle():
    trace = {
        "decision": "acquire", "selected_candidate_id": "lineup",
        "selected_action_kind": "query", "context_digest": "context-152",
        "state_digest": "state-152", "trace_digest": "trace-152",
        "prediction_cutoff": CUTOFF,
        "reservation": {
            "candidate_id": "lineup", "actual_spend_observed": False,
            "utility_cost": {"amount": .1, "unit": "decision_loss"},
            "resource_cost": {"http_calls": 1},
        },
    }
    request = {
        "schema_version": REQUEST_VERSION, "request_id": "request-152",
        "candidate_id": "lineup", "action_id": "query-lineup",
        "source_id": "injury", "source_ids": ["injury"],
    }
    receipt = accept_action(trace, provider_id="fixture-provider",
                            accepted_at=ACCEPTED_AT)
    return request, receipt


def completion(**extra):
    values = {
        "result_payload": {}, "evidence_rows": [],
        "actual_resource_cost": {"http_calls": 1, "latency_ms": 2000},
        "observed_at": "2026-10-01T10:30:00Z",
        "operational_available_at": "2026-10-01T10:31:00Z",
    }
    values.update(extra)
    return values


@pytest.mark.parametrize("status,error_code,available,expected", [
    ("failed", None, "2026-10-01T10:31:00Z", "failed"),
    (None, "PROVIDER_TIMEOUT", "2026-10-01T10:31:00Z", "failed"),
    ("completed", "PROVIDER_TIMEOUT", "2026-10-01T10:31:00Z", "failed"),
    ("completed", "PROVIDER_TIMEOUT", "2026-10-01T13:00:00Z", "failed"),
    ("late", None, "2026-10-01T13:00:00Z", "late"),
    (None, None, "2026-10-01T13:00:00Z", "late"),
    ("ambiguous", None, "2026-10-01T10:31:00Z", "ambiguous"),
])
@pytest.mark.parametrize("through_adapter", [False, True])
def test_unsuccessful_empty_response_closes_receipt_with_actual_cost(
    status, error_code, available, expected, through_adapter,
):
    request, accepted = bundle()
    before_request, before_receipt = deepcopy(request), deepcopy(accepted)
    kwargs = completion(status=status, error_code=error_code,
                        operational_available_at=available)
    if through_adapter:
        result = complete_execution_request(accepted, request, **kwargs)
        closed = result["receipt"]
        assert result["normalized"]["evidence_rows"] == []
    else:
        closed = complete_action(accepted, **kwargs)
    assert closed["status"] == expected
    assert closed["actual_resource_cost"] == {"http_calls": 1.0, "latency_ms": 2000.0}
    assert closed["evidence_rows"] == []
    assert closed["evidence_digest"]
    assert closed["error_code"] == error_code
    assert closed["replanning_required"] is True
    assert request == before_request and accepted == before_receipt
    actor = {"context_digest": "context-152"}
    rebuilt = rebuild_context_after_receipt(actor, closed, {})
    assert rebuilt["actor_context"] == actor
    assert rebuilt["context_changed"] is False


@pytest.mark.parametrize("through_adapter", [False, True])
def test_success_still_requires_evidence(through_adapter):
    request, receipt = bundle()
    with pytest.raises(ValueError, match="nonempty"):
        if through_adapter:
            complete_execution_request(receipt, request, **completion(
                status="completed", captured_at="2026-10-01T10:31:01Z"))
        else:
            complete_action(receipt, **completion(
                status="completed", captured_at="2026-10-01T10:31:01Z"))


def test_standalone_normalizer_still_requires_evidence():
    request, _ = bundle()
    with pytest.raises(ValueError, match="nonempty"):
        normalize_provider_observation(request, result_payload={}, evidence_rows=[])


@pytest.mark.parametrize("through_adapter", [False, True])
def test_failed_empty_response_rejects_evaluator_payload(through_adapter):
    request, receipt = bundle()
    kwargs = completion(status="failed", result_payload={"nested": {"home_score": 2}})
    with pytest.raises(ValueError, match="evaluator"):
        if through_adapter:
            complete_execution_request(receipt, request, **kwargs)
        else:
            complete_action(receipt, **kwargs)


@pytest.mark.parametrize("row,expected", [
    ({"source_id": "weather"}, "absent from request sources"),
    ({"source_id": "injury", "values": {"label": 1}}, "evaluator"),
])
def test_failed_response_retains_source_and_evidence_checks(row, expected):
    request, receipt = bundle()
    with pytest.raises(ValueError, match=expected):
        complete_execution_request(receipt, request,
                                   **completion(status="failed", evidence_rows=[row]))


def test_empty_failure_rejects_invalid_actual_cost():
    request, receipt = bundle()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        complete_execution_request(receipt, request, **completion(
            status="failed", actual_resource_cost={"http_calls": -1}))


@pytest.mark.parametrize("status,rows", [
    ("completed", [{"source_id": "injury", "values": {"rotation_risk": "low"}}]),
    ("failed", []),
])
def test_missing_capture_stays_unknown_with_cached_clocks(status, rows):
    request, receipt = bundle()
    closed = complete_execution_request(receipt, request, **completion(
        status=status, evidence_rows=rows,
        observed_at="2026-10-01T08:00:00Z",
        operational_available_at="2026-10-01T08:01:00Z"))["receipt"]
    assert closed["captured_at"] is None
    assert closed["capture_status"] == "unknown"
    assert closed["observed_at"] == "2026-10-01T08:00:00Z"
    assert closed["status"] == ("ambiguous" if status == "completed" else status)


def test_explicit_capture_before_acceptance_is_rejected():
    request, receipt = bundle()
    with pytest.raises(ValueError, match="clocks are not ordered"):
        complete_execution_request(receipt, request, **completion(
            status="failed", observed_at="2026-10-01T08:00:00Z",
            operational_available_at="2026-10-01T08:01:00Z",
            captured_at="2026-10-01T09:00:00Z"))


def test_empty_failure_receipt_cannot_be_completed_twice():
    request, receipt = bundle()
    closed = complete_execution_request(receipt, request,
                                        **completion(status="failed"))["receipt"]
    with pytest.raises(ValueError, match="accepted receipt"):
        complete_execution_request(closed, request, **completion(status="failed"))


@pytest.mark.parametrize("field", ["match_id", "canonical_match_id", "target_key"])
@pytest.mark.parametrize("envelope", [None, "values", "payload", "context"])
def test_explicit_foreign_evidence_target_is_rejected(field, envelope):
    request, receipt = bundle()
    request["parameters"] = {"match_id": "current-match"}
    row = {"source_id": "injury"}
    if envelope is None:
        row[field] = "other-match"
    else:
        row[envelope] = {field: "other-match"}
    with pytest.raises(ValueError, match="match identity differs"):
        complete_execution_request(receipt, request,
                                   **completion(status="completed", evidence_rows=[row]))


@pytest.mark.parametrize("field", ["match_id", "canonical_match_id", "target_key"])
def test_explicit_foreign_result_target_is_rejected_even_for_empty_failure(field):
    request, receipt = bundle()
    request["parameters"] = {"match_id": "current-match"}
    with pytest.raises(ValueError, match="match identity differs"):
        complete_execution_request(receipt, request, **completion(
            status="failed", result_payload={field: "other-match"}))


def test_nested_current_evidence_envelopes_are_identity_checked():
    request, receipt = bundle()
    request["parameters"] = {"match_id": "current-match"}
    row = {"source_id": "injury", "values": {"payload": {"context": {
        "canonical_match_id": "other-match", "rotation_risk": "low"}}}}
    with pytest.raises(ValueError, match="match identity differs"):
        complete_execution_request(receipt, request, **completion(evidence_rows=[row]))


def test_native_fixture_ids_and_historical_features_keep_their_own_identity():
    request, receipt = bundle()
    request["parameters"] = {"match_id": "current-match"}
    row = {
        "source_id": "injury", "match_id": "current-match", "fixture_id": 18217,
        "values": {
            "context": {"target_key": "current-match", "rotation_risk": "low"},
            "recent_matches": [{"match_id": "past-match", "rest_days": 4}],
            "historical_summary": {"canonical_match_id": "past-match"},
        },
    }
    before = deepcopy(row)
    closed = complete_execution_request(receipt, request, **completion(
        result_payload={"canonical_match_id": "current-match", "fixture_id": 18217},
        evidence_rows=[row], captured_at="2026-10-01T10:31:01Z"))["receipt"]
    assert closed["status"] == "completed"
    assert closed["evidence_rows"] == [before]
    assert row == before


def test_missing_request_target_remains_compatible_with_existing_responses():
    request, receipt = bundle()
    row = {"source_id": "injury", "match_id": "provider-known-match"}
    closed = complete_execution_request(receipt, request,
                                        **completion(evidence_rows=[row], captured_at="2026-10-01T10:31:01Z"))["receipt"]
    assert closed["status"] == "completed"
