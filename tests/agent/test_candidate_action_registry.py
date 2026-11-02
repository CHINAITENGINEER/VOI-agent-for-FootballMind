"""Round149 tests for menu-to-action routing and receipt replanning."""
from __future__ import annotations

from copy import deepcopy

import pytest

from football_agent.candidate_action_registry import (
    VERSION, build_action_registry, build_execution_request, build_state_from_menu,
    replan_after_execution_receipt,
)
from football_agent.candidate_menu_compiler import compile_candidate_menu
from football_agent.autonomous_acquisition_step import select_acquisition_action
from football_agent.execution_receipt_bridge import accept_action, complete_action
from football_agent.platform_acquisition_context import build_platform_acquisition_context

CUTOFF = "2026-10-01T12:00:00Z"


def match():
    return {"match_id": "fixture-149", "kickoff": "2026-10-01T18:00:00Z",
            "prediction_cutoff": CUTOFF, "observed_at": "2026-10-01T08:00:00Z",
            "ingested_at": "2026-10-01T08:01:00Z", "home_team_id": "home",
            "away_team_id": "away"}


def catalog():
    return [
        {
            "candidate_id": "lineup", "kind": "query", "source_ids": ["injury"],
            "route": {"source_id": "injury", "action_id": "query-lineup", "cost": .1, "unit": "decision_loss"},
            "declared_cost": {"http_calls": 1}, "utility_cost": {"amount": .1, "unit": "decision_loss"},
            "context_affinity": {"rotation_risk": ["high"]},
        },
        {
            "candidate_id": "weather", "kind": "query", "source_ids": ["weather"],
            "route": {"source_id": "weather", "action_id": "query-weather", "cost": .05, "unit": "decision_loss"},
            "declared_cost": {"http_calls": 1}, "utility_cost": {"amount": .05, "unit": "decision_loss"},
        },
    ]


def actor(rows=()):
    return build_platform_acquisition_context(match(), rows)["actor_context"]


def menu(rows=()):
    if not rows:
        rows = [
            {"source_id": "injury", "available_at": "2026-10-01T09:00:00Z"},
            {"source_id": "weather", "available_at": "2026-10-01T09:00:00Z"},
        ]
    report = {"target_key": "fixture-149", "context": {"rotation_risk": "high"},
              "prediction_cutoff": CUTOFF, "evaluator_fields_used": False}
    return compile_candidate_menu(catalog(), context_report=report, prediction_cutoff=CUTOFF, source_rows=rows)


def evidence(source="injury", **values):
    return {"source_id": source, "kind": source, "observed_at": "2026-10-01T09:00:00Z",
            "ingested_at": "2026-10-01T09:01:00Z", "availability": "available",
            "freshness": "fresh", "digest": "a" * 64, "values": values or {"context": {"rotation_risk": "high"}}}


def test_registry_preserves_routes_and_exposes_state_candidates():
    report = menu([{"source_id": "injury", "available_at": "2026-10-01T09:00:00Z"}])
    registry = build_action_registry(report, actor_context=actor())
    assert registry["schema_version"] == VERSION
    assert registry["by_candidate"]["lineup"]["action_id"] == "query-lineup"
    assert registry["acquisition_candidates"][0]["resource_cost"] == {"http_calls": 1.0}
    assert registry["contract"]["selection_delegated_to_state_bound_voi"] is True


def test_state_from_menu_keeps_selection_inside_existing_voi():
    report = menu()
    bundle = build_state_from_menu(actor(), report)
    state = bundle["state"]
    estimate = {"candidate_id": "lineup", "state_digest": state["state_digest"],
                "expected_value": .4, "unit": "decision_loss", "source": "test",
                "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}
    trace = select_acquisition_action(state, loss_unit="decision_loss", estimates={"lineup": estimate}, remaining_budget={"http_calls": 2})
    assert trace["decision"] == "acquire"
    assert trace["selected_candidate_id"] == "lineup"


def test_execution_request_routes_candidate_to_action():
    report = menu()
    bundle = build_state_from_menu(actor(), report)
    state = bundle["state"]
    estimate = {"candidate_id": "lineup", "state_digest": state["state_digest"],
                "expected_value": .4, "unit": "decision_loss", "source": "test",
                "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}
    trace = select_acquisition_action(state, loss_unit="decision_loss", estimates={"lineup": estimate}, remaining_budget={"http_calls": 2})
    request = build_execution_request(trace, bundle["registry"], parameters={"match_id": "fixture-149"})
    assert request["action_id"] == "query-lineup"
    assert request["source_id"] == "injury"
    assert request["contract"]["real_source_requests"] == 0


def test_duplicate_action_id_is_rejected():
    report = menu()
    report["routes"]["weather"]["action_id"] = "query-lineup"
    with pytest.raises(ValueError, match="duplicate action_id"):
        build_action_registry(report, actor_context=actor())


def test_execution_request_rejects_evaluator_parameters():
    report = menu()
    bundle = build_state_from_menu(actor(), report)
    state = bundle["state"]
    estimate = {"candidate_id": "lineup", "state_digest": state["state_digest"],
                "expected_value": .4, "unit": "decision_loss", "source": "test",
                "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}
    trace = select_acquisition_action(state, loss_unit="decision_loss", estimates={"lineup": estimate}, remaining_budget={"http_calls": 2})
    with pytest.raises(ValueError, match="evaluator"):
        build_execution_request(trace, bundle["registry"], parameters={"result": "3:0"})


def test_completed_receipt_rebuilds_context_and_next_state():
    initial_rows = [evidence("fixture", context={})]
    actor_context = actor(initial_rows)
    report = menu()
    bundle = build_state_from_menu(actor_context, report)
    state = bundle["state"]
    estimate = {"candidate_id": "lineup", "state_digest": state["state_digest"],
                "expected_value": .4, "unit": "decision_loss", "source": "test",
                "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}
    trace = select_acquisition_action(state, loss_unit="decision_loss", estimates={"lineup": estimate}, remaining_budget={"http_calls": 2})
    accepted = accept_action(trace, provider_id="fixture-provider", accepted_at="2026-10-01T10:00:00Z")
    completed = complete_action(accepted, result_payload={"rotation_confirmed": True},
        evidence_rows=[evidence("injury", context={"rotation_risk": "low"})],
        actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:30:00Z",
        operational_available_at="2026-10-01T10:31:00Z", captured_at="2026-10-01T10:31:01Z")
    replanned = replan_after_execution_receipt(actor_context, completed, match(), menu_report=report, catalog=catalog(), existing_evidence_rows=initial_rows)
    assert replanned["context_changed"] is True
    assert replanned["state"]["state_digest"] != state["state_digest"]
    assert replanned["replan_trace"]["next_context_digest"] != actor_context["context_digest"]
    assert replanned["contract"]["selection_delegated_to_state_bound_voi"] is True


def test_failed_receipt_keeps_context_and_requires_replan():
    actor_context = actor()
    report = menu()
    bundle = build_state_from_menu(actor_context, report)
    state = bundle["state"]
    estimate = {"candidate_id": "lineup", "state_digest": state["state_digest"],
                "expected_value": .4, "unit": "decision_loss", "source": "test",
                "training_cutoff": "2026-09-01T00:00:00Z", "available_at": "2026-09-30T00:00:00Z"}
    trace = select_acquisition_action(state, loss_unit="decision_loss", estimates={"lineup": estimate}, remaining_budget={"http_calls": 2})
    accepted = accept_action(trace, provider_id="fixture-provider", accepted_at="2026-10-01T10:00:00Z")
    failed = complete_action(accepted, result_payload={"provider_error": True}, evidence_rows=[evidence()],
        actual_resource_cost={"http_calls": 1}, observed_at="2026-10-01T10:30:00Z",
        operational_available_at="2026-10-01T10:31:00Z", status="failed", error_code="TIMEOUT")
    replanned = replan_after_execution_receipt(actor_context, failed, match(), menu_report=report, catalog=catalog())
    assert replanned["context_changed"] is False
    assert replanned["replanning_required"] is True
    assert replanned.get("state") is None


def test_registry_and_menu_inputs_are_not_mutated():
    report = menu()
    before = deepcopy(report)
    build_action_registry(report, actor_context=actor())
    assert report == before

