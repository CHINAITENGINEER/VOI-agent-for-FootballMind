import pytest

from football_agent.adaptive_voi_calibration import SUPPORTED
from football_agent.contextual_adaptive_voi import (
    CONTEXT_INSUFFICIENT,
    calibrate_contextual_candidate_value,
    context_fingerprint,
    normalize_context,
)
from football_agent.query_value_contract import QueryValueGate, QUERY, STOP, VALUE_UNMEASURED
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate
from football_agent.voi_evidence_bridge import VoIEvidenceLink


AS_OF = "2026-09-29T10:00:00Z"
HIGH = {"qualification": "already_qualified", "reserve_strength": "unknown", "tactical_intent": "unknown"}
LOW = {"qualification": "must_win", "reserve_strength": "known_strong", "tactical_intent": "unknown"}


def link(index, *, value, requested="2026-09-28T08:00:00Z"):
    return VoIEvidenceLink(
        link_id=f"context-link-{index}", trace_id=f"context-trace-{index}", episode_id=f"context-episode-{index}",
        target_key="context-match", request_id=f"context-request-{index}", action_id="lineup",
        candidate_id="historical-lineup", evidence_entry_id=f"context-entry-{index}",
        evidence_availability="available", evidence_quality="VALIDATED",
        requested_at=requested, source_version="v2", source_id="api-football",
        dependency_key=f"context-dependency-{index}", prediction_cutoff="2026-09-28T12:00:00Z",
        expected_information_value=value, observed_risk_reduction=value,
        calibration_error=0.0, status="linked_available",
    )


def history():
    rows, contexts = [], {}
    for i in range(30):
        item = link(i, value=.72 + (i % 3) * .02)
        rows.append(item); contexts[item.link_id] = HIGH
    for i in range(30, 60):
        item = link(i, value=.05 + (i % 3) * .01)
        rows.append(item); contexts[item.link_id] = LOW
    return rows, contexts


def test_context_fingerprint_is_canonical_and_excludes_evaluator_fields():
    assert context_fingerprint({"b": "x", "a": "y"}) == context_fingerprint({"a": "y", "b": "x"})
    assert normalize_context({"qualification": "already_qualified"}) == {"qualification": "already_qualified"}
    with pytest.raises(ValueError, match="evaluator-only"):
        context_fingerprint({"result": "home_win"})


def test_exact_context_changes_value_without_transferring_global_average():
    rows, contexts = history()
    high = calibrate_contextual_candidate_value(
        "lineup", source_id="api-football", source_version="v2", cost=.01,
        context=HIGH, links=rows, context_by_link_id=contexts, as_of=AS_OF,
        min_independent_dependencies=5,
    )
    low = calibrate_contextual_candidate_value(
        "lineup", source_id="api-football", source_version="v2", cost=.01,
        context=LOW, links=rows, context_by_link_id=contexts, as_of=AS_OF,
        min_independent_dependencies=5,
    )
    assert high.support_status == SUPPORTED
    assert low.support_status == SUPPORTED
    assert high.exact.expected_information_value > low.exact.expected_information_value
    assert high.global_reference.expected_information_value > low.exact.expected_information_value
    assert high.transfer_policy == "exact_context_only"


def test_context_with_sparse_history_stays_unknown_even_when_global_source_is_supported():
    rows, contexts = history()
    novel = {"qualification": "draw_sufficient", "reserve_strength": "unknown", "tactical_intent": "unknown"}
    estimate = calibrate_contextual_candidate_value(
        "lineup", source_id="api-football", source_version="v2", cost=.01,
        context=novel, links=rows, context_by_link_id=contexts, as_of=AS_OF,
        min_independent_dependencies=5,
    )
    assert estimate.support_status == CONTEXT_INSUFFICIENT
    assert estimate.global_reference.support_status == SUPPORTED
    assert estimate.as_query_assessment().value_status == VALUE_UNMEASURED


def test_query_gate_uses_exact_context_and_stops_when_contextual_value_is_zero():
    rows, contexts = history()
    target = "context-match"
    board = StateQueryBoard(
        target_key=target, prediction_cutoff="2026-09-29T12:00:00Z",
        hypotheses=[StateHypothesis("lineup-h", target, "lineup", ("known", "unknown"))],
        candidates=[StateQueryCandidate("lineup", target, "2026-09-29T12:00:00Z", "lineup", "source", "read", .01)],
    )
    high = calibrate_contextual_candidate_value(
        "lineup", source_id="api-football", source_version="v2", cost=.01,
        context=HIGH, links=rows, context_by_link_id=contexts, as_of=AS_OF,
        min_independent_dependencies=5,
    ).as_query_assessment()
    low = calibrate_contextual_candidate_value(
        "lineup", source_id="api-football", source_version="v2", cost=.01,
        context=LOW, links=rows, context_by_link_id=contexts, as_of=AS_OF,
        min_independent_dependencies=5,
    ).as_query_assessment()
    assert QueryValueGate.decide(board, [high], remaining_budget=.05).action == QUERY
    assert QueryValueGate.decide(board, [low], remaining_budget=.05).action == STOP


def test_context_mapping_requires_scalar_public_values():
    with pytest.raises(ValueError, match="scalar JSON"):
        normalize_context({"rotation": ["unknown"]})

