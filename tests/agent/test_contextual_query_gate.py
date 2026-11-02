import pytest

from football_agent.adaptive_voi_calibration import SUPPORTED
from football_agent.contextual_adaptive_voi import CONTEXT_INSUFFICIENT
from football_agent.contextual_query_gate import VERSION, decide_contextual_query
from football_agent.query_value_contract import DEFER, QUERY, VALUE_UNMEASURED
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate
from football_agent.voi_evidence_bridge import VoIEvidenceLink


AS_OF = "2026-09-29T10:00:00Z"
TARGET = "contextual-match"
HIGH = {"qualification": "already_qualified", "reserve_strength": "unknown"}
NOVEL = {"qualification": "draw_sufficient", "reserve_strength": "unknown"}


def link(index, *, value=.72, requested="2026-09-28T08:00:00Z", version="v2"):
    return VoIEvidenceLink(
        link_id=f"gate-link-{index}", trace_id=f"gate-trace-{index}", episode_id=f"gate-episode-{index}",
        target_key=TARGET, request_id=f"gate-request-{index}", action_id="lineup-read",
        candidate_id="lineup-refresh", evidence_entry_id=f"gate-entry-{index}",
        evidence_availability="available", evidence_quality="VALIDATED", requested_at=requested,
        source_version=version, source_id="api-football", dependency_key=f"gate-dep-{index}",
        prediction_cutoff="2026-09-28T12:00:00Z", expected_information_value=value,
        observed_risk_reduction=value, calibration_error=0.0, status="linked_available",
    )


def board():
    return StateQueryBoard(
        target_key=TARGET, prediction_cutoff="2026-09-29T12:00:00Z",
        hypotheses=[
            StateHypothesis("lineup-h", TARGET, "lineup", ("known", "unknown")),
            StateHypothesis("weather-h", TARGET, "weather", ("low", "high")),
        ],
        candidates=[
            StateQueryCandidate("lineup-refresh", TARGET, "2026-09-29T12:00:00Z", "lineup", "api", "read", .01),
            StateQueryCandidate("weather-refresh", TARGET, "2026-09-29T12:00:00Z", "weather", "api", "read", .01),
        ],
    )


def history():
    rows, contexts = [], {}
    for index in range(30):
        item = link(index, value=.72 + (index % 3) * .02)
        rows.append(item)
        contexts[item.link_id] = HIGH
    return rows, contexts


def test_exact_context_estimate_enters_gate_and_global_diagnostic_is_retained():
    rows, contexts = history()
    result = decide_contextual_query(
        board(), {"lineup-refresh": {"source_id": "api-football", "source_version": "v2", "cost": .01}},
        context=HIGH, links=rows, context_by_link_id=contexts, as_of=AS_OF, remaining_budget=.05,
    )
    assert result.estimates[0].support_status == SUPPORTED
    assert result.decision.action == QUERY
    assert result.decision.candidate_id == "lineup-refresh"
    assert result.as_dict()["hidden_result_included"] is False


def test_novel_context_withholds_global_value_and_defers():
    rows, contexts = history()
    result = decide_contextual_query(
        board(), {"lineup-refresh": {"source_id": "api-football", "source_version": "v2", "cost": .01}},
        context=NOVEL, links=rows, context_by_link_id=contexts, as_of=AS_OF, remaining_budget=.05,
    )
    estimate = result.estimates[0]
    assert estimate.support_status == CONTEXT_INSUFFICIENT
    assert estimate.global_reference.support_status == SUPPORTED
    assert result.assessments[0].value_status == VALUE_UNMEASURED
    assert result.decision.action == DEFER


def test_future_rows_and_wrong_source_version_do_not_authorize_query():
    rows = [link(i, requested="2026-09-30T08:00:00Z") for i in range(30)]
    contexts = {item.link_id: HIGH for item in rows}
    result = decide_contextual_query(
        board(), {"lineup-refresh": {"source_id": "api-football", "source_version": "v2", "cost": .01}},
        context=HIGH, links=rows, context_by_link_id=contexts, as_of=AS_OF, remaining_budget=.05,
    )
    assert result.estimates[0].support_status != SUPPORTED
    assert result.decision.action == DEFER

    rows = [link(i, version="v1") for i in range(30)]
    contexts = {item.link_id: HIGH for item in rows}
    result = decide_contextual_query(
        board(), {"lineup-refresh": {"source_id": "api-football", "source_version": "v2", "cost": .01}},
        context=HIGH, links=rows, context_by_link_id=contexts, as_of=AS_OF, remaining_budget=.05,
    )
    assert result.decision.action == DEFER


def test_unregistered_route_is_rejected_before_calibration():
    with pytest.raises(ValueError, match="actionable"):
        decide_contextual_query(
            board(), {"missing-route": {"source_id": "api", "source_version": "v1", "cost": .01}},
            context=HIGH, links=[], context_by_link_id={}, as_of=AS_OF, remaining_budget=.05,
        )


def test_contextual_result_serializes_exact_assessment_and_decision():
    rows, contexts = history()
    result = decide_contextual_query(
        board(), {"lineup-refresh": {"source_id": "api-football", "source_version": "v2", "cost": .01}},
        context=HIGH, links=rows, context_by_link_id=contexts, as_of=AS_OF, remaining_budget=.05,
    )
    payload = result.as_dict()
    assert payload["schema_version"] == VERSION
    assert payload["estimates"][0]["transfer_policy"] == "exact_context_only"
    assert payload["decision"]["candidate_id"] == "lineup-refresh"
