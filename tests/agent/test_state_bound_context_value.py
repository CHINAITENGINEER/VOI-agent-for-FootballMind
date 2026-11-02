"""Round142 tests for interval-valued, state-bound contextual VoI."""
from __future__ import annotations

import pytest

from football_agent.platform_acquisition_context import build_platform_acquisition_context
from football_agent.contextual_candidate_generation import prepare_contextual_acquisition
from football_agent.state_bound_context_value import (
    CONTEXT_UNKNOWN,
    INSUFFICIENT_SUPPORT,
    NO_MATCH,
    SUPPORTED_INTERVAL,
    StateBoundContextValueEstimator,
    select_prepared_contextual_action,
)
from football_agent.voi_evidence_bridge import VoIEvidenceLink


CUTOFF = "2026-10-01T12:00:00Z"


def _link(index: int, value: float | None, context_value: str, *, status: str = "linked_available",
          requested: str = "2026-09-28T08:00:00Z", target: str | None = None) -> tuple[VoIEvidenceLink, dict]:
    link = VoIEvidenceLink(
        link_id=f"round142-link-{index}", trace_id=f"round142-trace-{index}",
        episode_id=f"episode-{index}", target_key=target or f"historical-match-{index}",
        request_id=f"request-{index}", action_id="lineup-query", candidate_id="lineup",
        evidence_entry_id=None if status != "linked_available" else f"entry-{index}",
        evidence_availability="available" if status == "linked_available" else "unavailable",
        evidence_quality="VALIDATED" if status == "linked_available" else None,
        requested_at=requested, source_version="api-v2", source_id="api-football",
        dependency_key=f"dependency-{index}", prediction_cutoff="2026-09-28T12:00:00Z",
        expected_information_value=value, observed_risk_reduction=value,
        calibration_error=None, status=status,
    )
    return link, {"qualification_state": context_value}


def _estimator(values: list[float | None], context_value: str = "high") -> StateBoundContextValueEstimator:
    links, contexts = [], {}
    for index, value in enumerate(values):
        link, context = _link(index, value, context_value)
        links.append(link)
        contexts[link.link_id] = context
    return StateBoundContextValueEstimator(
        links, contexts,
        {"lineup": {"source_id": "api-football", "source_version": "api-v2",
                     "action_id": "lineup-query", "cost": .1, "unit": "decision_loss"}},
        context_keys=("qualification_state",), min_support=5, confidence=.9,
        calibration_as_of="2026-09-30T00:00:00Z", family_size=1,
    )


def _state(*, complete: bool = True):
    match = {
        "match_id": "round142-current", "kickoff": "2026-10-01T18:00:00Z",
        "prediction_cutoff": CUTOFF, "observed_at": "2026-10-01T08:00:00Z",
        "ingested_at": "2026-10-01T08:01:00Z", "home_team_id": "home", "away_team_id": "away",
    }
    actor = build_platform_acquisition_context(match, [])['actor_context']
    registry = [{
        "candidate_id": "lineup", "source_ids": ["lineup-source"], "action_kind": "query",
        "action_availability": True, "resource_cost": {"http_calls": 1},
        "utility_cost": {"amount": .1, "unit": "decision_loss", "source": "round142"},
    }]
    return prepare_contextual_acquisition(actor, registry, registry_complete=complete)["acquisition_state"]


def test_same_candidate_has_different_intervals_for_different_contexts():
    estimator = _estimator([.8] * 30)
    high = estimator.estimate_candidate("lineup", {"qualification_state": "high"}, cutoff=CUTOFF,
                                       target_key="round142-current", cost=.1, cost_unit="decision_loss")
    low_estimator = _estimator([.0] * 1000, "low")
    low = low_estimator.estimate_candidate("lineup", {"qualification_state": "low"}, cutoff=CUTOFF,
                                           target_key="round142-current", cost=.1, cost_unit="decision_loss")
    assert high.support_status == SUPPORTED_INTERVAL
    assert high.lower_bound is not None and high.lower_bound > .1
    assert low.support_status == SUPPORTED_INTERVAL
    assert low.upper_bound is not None and low.upper_bound <= .1


def test_crossing_interval_is_omitted_and_selector_defers():
    estimator = _estimator([.2] * 30)
    state = _state()
    report = estimator.select(state, {"qualification_state": "high"}, loss_unit="decision_loss",
                              remaining_budget={"http_calls": 1}, target_key="round142-current")
    interval = report["intervals"]["lineup"]
    assert interval["support_status"] == SUPPORTED_INTERVAL
    assert interval["lower_bound"] <= .1 < interval["upper_bound"]
    assert report["gate_estimates"] == {}
    assert report["selection"]["decision"] == "defer"


def test_lower_bound_above_cost_acquires_through_real_state_gate():
    estimator = _estimator([.8] * 30)
    report = estimator.select(_state(), {"qualification_state": "high"}, loss_unit="decision_loss",
                              remaining_budget={"http_calls": 1}, target_key="round142-current")
    assert report["gate_estimates"]["lineup"]["interval_decision"] == "lower_bound_above_cost"
    assert report["selection"]["decision"] == "acquire"
    assert report["selection"]["selected_candidate_id"] == "lineup"


def test_upper_bound_at_cost_allows_bounded_registry_stop():
    estimator = _estimator([.0] * 1000)
    report = estimator.select(_state(complete=True), {"qualification_state": "high"}, loss_unit="decision_loss",
                              remaining_budget={"http_calls": 1}, target_key="round142-current")
    assert report["gate_estimates"]["lineup"]["interval_decision"] == "upper_bound_at_or_below_cost"
    assert report["selection"]["decision"] == "stop"


def test_incomplete_context_and_sparse_context_never_use_global_fallback():
    estimator = _estimator([.8] * 30)
    unknown = estimator.estimate_candidate("lineup", {"qualification_state": "unknown"}, cutoff=CUTOFF,
                                          target_key="round142-current", cost=.1, cost_unit="decision_loss")
    assert unknown.support_status == CONTEXT_UNKNOWN
    sparse = StateBoundContextValueEstimator(
        estimator.links[:2], estimator.context_by_link_id, estimator.routes,
        context_keys=("qualification_state",), min_support=5, confidence=.9,
        calibration_as_of="2026-09-30T00:00:00Z", family_size=1,
    ).estimate_candidate("lineup", {"qualification_state": "high"}, cutoff=CUTOFF,
                         target_key="round142-current", cost=.1, cost_unit="decision_loss")
    assert sparse.support_status == INSUFFICIENT_SUPPORT


def test_future_history_and_current_target_are_excluded():
    links, contexts = [], {}
    for index in range(10):
        link, context = _link(index, .8, "high")
        links.append(link); contexts[link.link_id] = context
    future, context = _link(99, .8, "high", requested="2026-10-02T08:00:00Z")
    links.append(future); contexts[future.link_id] = context
    current, context = _link(100, .8, "high", target="round142-current")
    links.append(current); contexts[current.link_id] = context
    estimator = StateBoundContextValueEstimator(
        links, contexts,
        {"lineup": {"source_id": "api-football", "source_version": "api-v2", "action_id": "lineup-query",
                     "cost": .1, "unit": "decision_loss"}},
        context_keys=("qualification_state",), min_support=5, confidence=.9,
        calibration_as_of="2026-09-30T00:00:00Z", family_size=1,
    )
    estimate = estimator.estimate_candidate("lineup", {"qualification_state": "high"}, cutoff=CUTOFF,
                                             target_key="round142-current", cost=.1, cost_unit="decision_loss")
    assert estimate.support_status == SUPPORTED_INTERVAL
    assert estimate.attempt_count == 10


def test_prepared_bundle_routes_actor_context_into_estimator_without_a_manual_question():
    estimator = _estimator([.8] * 30)
    state = _state()
    report = select_prepared_contextual_action(
        {"acquisition_state": state, "context_binding": {"context": {"qualification_state": "high"}}},
        estimator, loss_unit="decision_loss", remaining_budget={"http_calls": 1},
        target_key="round142-current",
    )
    assert report["selection"]["decision"] == "acquire"
    assert report["selection"]["selected_candidate_id"] == "lineup"


def test_history_context_with_evaluator_payload_is_rejected_at_the_boundary():
    link, context = _link(1, .8, "high")
    context["result"] = "home"
    with pytest.raises(ValueError, match="evaluator-only"):
        StateBoundContextValueEstimator(
            [link], {link.link_id: context},
            {"lineup": {"source_id": "api-football", "source_version": "api-v2",
                         "action_id": "lineup-query", "cost": .1, "unit": "decision_loss"}},
            context_keys=("qualification_state",), min_support=1,
            calibration_as_of="2026-09-30T00:00:00Z", family_size=1,
        )

