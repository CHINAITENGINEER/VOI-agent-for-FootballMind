"""Round150 tests for provider-result normalization and completion routing."""
from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.candidate_action_registry import build_execution_request, build_state_from_menu
from football_agent.candidate_menu_compiler import compile_candidate_menu
from football_agent.autonomous_acquisition_step import select_acquisition_action
from football_agent.execution_action_adapter import VERSION, complete_execution_request, normalize_provider_observation
from football_agent.execution_receipt_bridge import accept_action
from football_agent.platform_acquisition_context import build_platform_acquisition_context

CUTOFF = "2026-10-01T12:00:00Z"


def match():
    return {"match_id": "fixture-150", "kickoff": "2026-10-01T18:00:00Z",
            "prediction_cutoff": CUTOFF, "observed_at": "2026-10-01T08:00:00Z",
            "ingested_at": "2026-10-01T08:01:00Z", "home_team_id": "home",
            "away_team_id": "away"}


def actor():
    return build_platform_acquisition_context(match())["actor_context"]


def catalog():
    return [{"candidate_id": "lineup", "kind": "query", "source_ids": ["injury"],
             "route": {"source_id": "injury", "action_id": "query-lineup", "cost": .1, "unit": "decision_loss"},
             "declared_cost": {"http_calls": 1}, "utility_cost": {"amount": .1, "unit": "decision_loss"}}]


def request_bundle():
    report = compile_candidate_menu(catalog(), context_report={"target_key": "fixture-150", "prediction_cutoff": CUTOFF, "context": {}, "evaluator_fields_used": False}, prediction_cutoff=CUTOFF, source_rows=[{"source_id": "injury", "available_at": "2026-10-01T09:00:00Z"}])
    bundle = build_state_from_menu(actor(), report)
    state = bundle["state"]
    estimate = {"candidate_id": "lineup", "state_digest": state["state_digest"], "expected_value": .4, "unit": "decision_loss", "source": "test", "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}
    trace = select_acquisition_action(state, loss_unit="decision_loss", estimates={"lineup": estimate}, remaining_budget={"http_calls": 2})
    request = build_execution_request(trace, bundle["registry"])
    receipt = accept_action(trace, provider_id="fixture-provider", accepted_at="2026-10-01T10:00:00Z")
    return request, receipt


def evidence():
    return {"source_id": "injury", "kind": "injury", "observed_at": "2026-10-01T10:30:00Z", "ingested_at": "2026-10-01T10:31:00Z", "values": {"context": {"rotation_risk": "low"}}}


def test_normalizer_keeps_route_identity_and_public_payload():
    request, _ = request_bundle()
    normalized = normalize_provider_observation(request, result_payload={"rotation_confirmed": True}, evidence_rows=[evidence()])
    assert normalized["schema_version"] == VERSION
    assert normalized["action_id"] == "query-lineup"
    assert normalized["evidence_rows"][0]["source_id"] == "injury"
    assert normalized["contract"]["evaluator_fields_used"] is False


def test_complete_execution_request_delegates_to_existing_receipt_bridge():
    request, receipt = request_bundle()
    result = complete_execution_request(receipt, request, result_payload={"rotation_confirmed": True}, evidence_rows=[evidence()], actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:30:00Z", operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T10:31:01Z")
    assert result["receipt"]["status"] == "completed"
    assert result["receipt"]["candidate_id"] == "lineup"
    assert request["acquisition_id"] == result["receipt"]["acquisition_id"]
    assert result["contract"]["receipt_completed_through_existing_bridge"] is True


def test_mismatched_receipt_candidate_is_rejected():
    request, receipt = request_bundle()
    bad = dict(receipt)
    bad["candidate_id"] = "weather"
    with pytest.raises(ValueError, match="receipt candidate differs"):
        complete_execution_request(bad, request, result_payload={"ok": True}, evidence_rows=[evidence()], actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:30:00Z", operational_available_at="2026-10-01T10:31:00Z")


def test_forbidden_provider_payload_is_rejected():
    request, _ = request_bundle()
    with pytest.raises(ValueError, match="evaluator"):
        normalize_provider_observation(request, result_payload={"score": 3}, evidence_rows=[evidence()])


def test_foreign_evidence_source_is_rejected():
    request, _ = request_bundle()
    row = evidence(); row["source_id"] = "weather"
    with pytest.raises(ValueError, match="absent from request sources"):
        normalize_provider_observation(request, result_payload={"ok": True}, evidence_rows=[row])


def test_late_completion_stays_late_through_adapter():
    request, receipt = request_bundle()
    result = complete_execution_request(receipt, request, result_payload={"rotation_confirmed": True}, evidence_rows=[evidence()], actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T12:30:00Z", operational_available_at="2026-10-01T13:00:00Z", captured_at="2026-10-01T13:00:01Z")
    assert result["receipt"]["status"] == "late"


def test_capture_after_cutoff_stays_late_through_adapter():
    request, receipt = request_bundle()
    result = complete_execution_request(
        receipt, request, result_payload={"rotation_confirmed": True}, evidence_rows=[evidence()],
        actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:30:00Z",
        operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T12:01:00Z",
    )
    assert result["receipt"]["status"] == "late"


def test_adapter_does_not_mutate_request_or_rows():
    request, receipt = request_bundle()
    rows = [evidence()]
    before_request, before_rows = deepcopy(request), deepcopy(rows)
    complete_execution_request(receipt, request, result_payload={"ok": True}, evidence_rows=rows, actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:30:00Z", operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T10:31:01Z")
    assert request == before_request
    assert rows == before_rows


def test_adapter_rejects_empty_evidence():
    request, _ = request_bundle()
    with pytest.raises(ValueError, match="nonempty"):
        normalize_provider_observation(request, result_payload={"ok": True}, evidence_rows=[])


def test_adapter_contract_reports_no_real_calls():
    request, _ = request_bundle()
    result = normalize_provider_observation(request, result_payload={"ok": True}, evidence_rows=[evidence()])
    assert result["contract"]["real_source_requests"] == 0
    assert result["contract"]["real_model_calls"] == 0
