import pytest

from football_agent.adaptive_voi_calibration import (
    SUPPORTED,
    calibrate_candidate_value,
    calibrate_candidate_values,
)
from football_agent.query_value_contract import QueryValueGate, QUERY, DEFER, VALUE_UNMEASURED
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate
from football_agent.voi_evidence_bridge import VoIEvidenceLink


AS_OF = "2026-09-29T10:00:00Z"


def link(index, *, source_id="api-football", version="v2", status="linked_available",
         value=.60, dependency=None, requested="2026-09-28T08:00:00Z"):
    available = status == "linked_available"
    return VoIEvidenceLink(
        link_id=f"link-{index}", trace_id=f"trace-{index}", episode_id=f"episode-{index}",
        target_key="match", request_id=f"request-{index}", action_id="lineup",
        candidate_id="historical-lineup", evidence_entry_id=f"entry-{index}" if available else None,
        evidence_availability="available" if available else "failed",
        evidence_quality="VALIDATED" if available else "PROVIDER_ERROR",
        requested_at=requested, source_version=version, source_id=source_id,
        dependency_key=dependency or f"dep-{index}", prediction_cutoff="2026-09-28T12:00:00Z",
        expected_information_value=value if available else None,
        observed_risk_reduction=value if available else None,
        calibration_error=0.0 if available else None,
        status=status,
    )


def test_future_safe_source_value_uses_independent_clusters_and_lower_bounds():
    rows = [link(i, value=.60 + (i % 3) * .05) for i in range(8)]
    rows += [link(8, status="linked_unusable"), link(9, status="linked_unusable")]
    estimate = calibrate_candidate_value(
        "lineup-refresh", source_id="api-football", source_version="v2", cost=.05,
        links=rows, as_of=AS_OF, min_independent_dependencies=5, confidence=.10,
    )
    assert estimate.support_status == SUPPORTED
    assert estimate.independent_dependency_count == 10
    assert estimate.available_attempts == 8
    assert estimate.expected_information_value is not None
    assert 0 < estimate.expected_information_value < estimate.mean_observed_risk_reduction
    assert estimate.as_query_assessment().value_status != VALUE_UNMEASURED


def test_repeated_dependency_does_not_inflate_value_support():
    rows = [link(i, dependency="same-dependency") for i in range(8)]
    estimate = calibrate_candidate_value(
        "lineup-refresh", source_id="api-football", source_version="v2", cost=.01,
        links=rows, as_of=AS_OF, min_independent_dependencies=2,
    )
    assert estimate.independent_dependency_count == 1
    assert estimate.value_sample_count == 1
    assert estimate.support_status == "INSUFFICIENT_INDEPENDENT_SUPPORT"
    assert estimate.as_query_assessment().value_status == VALUE_UNMEASURED


def test_future_rows_are_excluded_and_out_of_range_values_remain_unmodeled():
    rows = [link(i, requested="2026-09-30T08:00:00Z") for i in range(8)]
    estimate = calibrate_candidate_value(
        "lineup-refresh", source_id="api-football", source_version="v2", cost=.01,
        links=rows, as_of=AS_OF, min_independent_dependencies=2,
    )
    assert estimate.support_status == "NO_SOURCE_MATCH"
    bad = [link(i, value=1.2) for i in range(8)]
    estimate = calibrate_candidate_value(
        "lineup-refresh", source_id="api-football", source_version="v2", cost=.01,
        links=bad, as_of=AS_OF, min_independent_dependencies=2,
    )
    assert estimate.support_status == "OBSERVED_VALUE_OUT_OF_RANGE"
    assert estimate.expected_information_value is None


def test_source_version_is_part_of_calibration_identity():
    rows = [link(i, version="v1") for i in range(6)]
    estimate = calibrate_candidate_value(
        "lineup-refresh", source_id="api-football", source_version="v2", cost=.01,
        links=rows, as_of=AS_OF, min_independent_dependencies=2,
    )
    assert estimate.support_status == "NO_SOURCE_MATCH"


def test_gate_queries_supported_route_and_defers_when_value_is_unknown():
    target = "match"
    board = StateQueryBoard(
        target_key=target, prediction_cutoff="2026-09-29T12:00:00Z",
        hypotheses=[StateHypothesis("lineup-h", target, "lineup", ("known", "unknown"))],
        candidates=[
            StateQueryCandidate("lineup-refresh", target, "2026-09-29T12:00:00Z", "lineup", "source", "read", .05),
            StateQueryCandidate("lineup-unknown", target, "2026-09-29T12:00:00Z", "lineup", "source", "read", .05),
        ],
    )
    supported = calibrate_candidate_value(
        "lineup-refresh", source_id="api-football", source_version="v2", cost=.05,
        links=[link(i) for i in range(8)], as_of=AS_OF, min_independent_dependencies=5,
    ).as_query_assessment()
    unknown = calibrate_candidate_value(
        "lineup-unknown", source_id="new-source", source_version="v1", cost=.05,
        links=[link(i) for i in range(8)], as_of=AS_OF, min_independent_dependencies=5,
    ).as_query_assessment()
    decision = QueryValueGate.decide(board, [supported, unknown], remaining_budget=.10)
    assert decision.action == QUERY
    assert decision.candidate_id == "lineup-refresh"
    assert unknown.value_status == VALUE_UNMEASURED


def test_batch_calibration_keeps_candidate_order_and_costs():
    estimates = calibrate_candidate_values(
        {
            "a": {"source_id": "api-football", "source_version": "v2", "cost": .01},
            "b": {"source_id": "missing", "source_version": "v1", "cost": .02},
        }, [link(i) for i in range(6)], as_of=AS_OF, min_independent_dependencies=5,
    )
    assert [item.candidate_id for item in estimates] == ["a", "b"]
    assert [item.cost for item in estimates] == [.01, .02]

