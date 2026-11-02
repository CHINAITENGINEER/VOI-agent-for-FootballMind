"""Round151 tests for the autonomous runtime loop."""
from __future__ import annotations

import pytest

from football_agent.autonomous_runtime_loop import VERSION, run_information_loop, run_runtime_cycle
from football_agent.candidate_action_registry import build_state_from_menu
from football_agent.candidate_menu_compiler import compile_candidate_menu
from football_agent.platform_acquisition_context import build_platform_acquisition_context
from football_agent.state_bound_context_value import StateBoundContextValueEstimator
from football_agent.voi_evidence_bridge import VoIEvidenceLink

CUTOFF = "2026-10-01T12:00:00Z"


def match():
    return {"match_id": "fixture-151", "kickoff": "2026-10-01T18:00:00Z", "prediction_cutoff": CUTOFF, "observed_at": "2026-10-01T08:00:00Z", "ingested_at": "2026-10-01T08:01:00Z", "home_team_id": "home", "away_team_id": "away"}


def catalog():
    return [{"candidate_id": "lineup", "kind": "query", "source_ids": ["injury"], "route": {"source_id": "injury", "action_id": "query-lineup", "cost": .1, "unit": "decision_loss"}, "declared_cost": {"http_calls": 1}, "utility_cost": {"amount": .1, "unit": "decision_loss"}}]


def actor():
    return build_platform_acquisition_context(match())["actor_context"]


def bundle():
    report = compile_candidate_menu(catalog(), context_report={"target_key": "fixture-151", "prediction_cutoff": CUTOFF, "context": {"qualification_state": "must_win"}, "evaluator_fields_used": False}, prediction_cutoff=CUTOFF, source_rows=[{"source_id": "injury", "available_at": "2026-10-01T09:00:00Z"}])
    return actor(), report, build_state_from_menu(actor(), report)


def estimator(value=.8):
    links, contexts = [], {}
    for index in range(30):
        link = VoIEvidenceLink(link_id=f"r151-{index}", trace_id=f"trace-{index}", episode_id=f"episode-{index}", target_key=f"historical-{index}", request_id=f"request-{index}", action_id="query-lineup", candidate_id="lineup", evidence_entry_id=f"entry-{index}", evidence_availability="available", evidence_quality="VALIDATED", requested_at="2026-09-01T08:00:00Z", source_version="v1", source_id="injury", dependency_key=f"dep-{index}", prediction_cutoff="2026-09-01T12:00:00Z", expected_information_value=value, observed_risk_reduction=value, calibration_error=None, status="linked_available")
        links.append(link); contexts[link.link_id] = {"qualification_state": "must_win"}
    from football_agent.state_bound_context_value import StateBoundContextValueEstimator
    return StateBoundContextValueEstimator(links, contexts, {"lineup": {"source_id": "injury", "source_version": "v1", "action_id": "query-lineup", "cost": .1, "unit": "decision_loss"}}, context_keys=("qualification_state",), min_support=5, confidence=.9, calibration_as_of="2026-09-30T00:00:00Z", family_size=1)


def provider(request):
    return {"result_payload": {"rotation_confirmed": True}, "evidence_rows": [{"source_id": "injury", "kind": "injury", "observed_at": "2026-10-01T10:30:00Z", "ingested_at": "2026-10-01T10:31:00Z", "values": {"context": {"rotation_risk": "low"}}}], "actual_resource_cost": {"http_calls": 1}, "observed_at": "2026-10-01T10:30:00Z", "operational_available_at": "2026-10-01T10:31:00Z", "captured_at": "2026-10-01T10:31:01Z"}


def test_runtime_cycle_acquires_without_a_user_question_and_replans():
    actor_context, report, prepared = bundle()
    result = run_runtime_cycle(actor_context=actor_context, state_bundle=prepared, menu_report=report, catalog=catalog(), match=match(), estimator=estimator(), context={"qualification_state": "must_win"}, loss_unit="decision_loss", remaining_budget={"http_calls": 1}, provider_id="fixture-provider", accepted_at="2026-10-01T10:00:00Z", execute_fn=provider)
    assert result["status"] == "replanned"
    assert result["request"]["action_id"] == "query-lineup"
    assert result["contract"]["agent_selected_without_user_question"] is True


def test_runtime_loop_stops_after_evidence_becomes_usable():
    actor_context, report, prepared = bundle()
    result = run_information_loop(actor_context=actor_context, state_bundle=prepared, menu_report=report, catalog=catalog(), match=match(), estimator=estimator(), context={"qualification_state": "must_win"}, loss_unit="decision_loss", remaining_budget={"http_calls": 1}, provider_id="fixture-provider", accepted_at="2026-10-01T10:00:00Z", execute_fn=provider, max_steps=3)
    assert result["steps"] >= 1
    assert result["terminal_status"] in {"terminal", "max_steps"}
    assert result["contract"]["selection_delegated_to_state_bound_voi"] is True


def test_runtime_cycle_stops_on_non_acquire_decision():
    actor_context, report, prepared = bundle()
    result = run_runtime_cycle(actor_context=actor_context, state_bundle=prepared, menu_report=report, catalog=catalog(), match=match(), estimator=estimator(0.0), context={"qualification_state": "must_win"}, loss_unit="decision_loss", remaining_budget={"http_calls": 1}, provider_id="fixture-provider", accepted_at="2026-10-01T10:00:00Z", execute_fn=provider)
    assert result["status"] == "terminal"
    assert result["request"] is None


class AskUserEstimator:
    def select(self, state, context, *, loss_unit, remaining_budget, target_key):
        return {"selection": {"decision": "ask_user", "reason": "fixture_requires_human_input"}}


def test_runtime_cycle_marks_ask_user_as_human_input_required():
    actor_context, report, prepared = bundle()
    result = run_runtime_cycle(
        actor_context=actor_context, state_bundle=prepared, menu_report=report,
        catalog=catalog(), match=match(), estimator=AskUserEstimator(),
        context={"qualification_state": "must_win"}, loss_unit="decision_loss",
        remaining_budget={"http_calls": 1}, provider_id="fixture-provider",
        accepted_at="2026-10-01T10:00:00Z", execute_fn=provider,
    )
    assert result["status"] == "human_input_required"
    assert result["contract"]["human_input_required"] is True
    assert result["contract"]["autonomous_completion"] is False
    assert result["contract"]["replay_eligible"] is False
    assert result["contract"]["agent_selected_without_user_question"] is False


def test_runtime_loop_preserves_explicit_step_cap():
    actor_context, report, prepared = bundle()
    result = run_information_loop(actor_context=actor_context, state_bundle=prepared, menu_report=report, catalog=catalog(), match=match(), estimator=estimator(), context={"qualification_state": "must_win"}, loss_unit="decision_loss", remaining_budget={"http_calls": 10}, provider_id="fixture-provider", accepted_at="2026-10-01T10:00:00Z", execute_fn=provider, max_steps=1)
    assert result["steps"] == 1
    assert result["terminal_status"] in {"replanned", "terminal", "max_steps"}


def test_runtime_loop_rejects_invalid_step_cap():
    actor_context, report, prepared = bundle()
    try:
        run_information_loop(actor_context=actor_context, state_bundle=prepared, menu_report=report, catalog=catalog(), match=match(), estimator=estimator(), context={"qualification_state": "must_win"}, loss_unit="decision_loss", remaining_budget={"http_calls": 1}, provider_id="fixture-provider", accepted_at="2026-10-01T10:00:00Z", execute_fn=provider, max_steps=0)
    except ValueError as exc:
        assert "max_steps" in str(exc)
    else:
        raise AssertionError("expected max_steps validation")


@pytest.mark.parametrize("decision", [None, "complete", "STOP", ""])
def test_unknown_selection_cannot_complete_or_call_provider(decision):
    class UnknownEstimator:
        def select(self, *args, **kwargs):
            return {"selection": {"decision": decision}}
    actor_context, report, prepared = bundle()
    def forbidden(request):
        pytest.fail("unsupported selection dispatched a provider")
    with pytest.raises(ValueError, match="unsupported selection decision"):
        run_runtime_cycle(
            actor_context=actor_context, state_bundle=prepared, menu_report=report,
            catalog=catalog(), match=match(), estimator=UnknownEstimator(),
            context={}, loss_unit="decision_loss", remaining_budget={"http_calls": 1},
            provider_id="offline", accepted_at="2026-10-01T10:00:00Z", execute_fn=forbidden)
