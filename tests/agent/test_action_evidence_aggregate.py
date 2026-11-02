import pytest
from dataclasses import replace

from football_agent.action_evidence_aggregate import aggregate_entries, link_trace_to_aggregate
from football_agent.action_readiness import (
    filter_candidate_routes, filter_candidate_routes_by_bindings, prepare_query_board_from_bindings,
    readiness_for, readiness_from_bindings,
)
from football_agent.evidence_binding import EvidenceBinding
from football_agent.decision_value import simulate_episode, synthetic_scenarios
from football_agent.evidence_ledger import EvidenceLedgerEntry
from football_agent.voi_trace import VoITrace
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate


CUTOFF = "2026-09-28T10:00:00Z"


def entry(source, request, *, availability="available", quality="VALIDATED",
          observed="2026-09-28T08:00:00Z", operational="2026-09-28T08:10:00Z"):
    return EvidenceLedgerEntry.from_row(
        {"candidate_id": "lineup", "source_id": source, "dependency_key": f"{source}:{request}",
         "quality": quality, "availability": availability, "observed_at": observed,
         "operational_available_at": operational, "evidence_refs": [f"raw:{source}"],
         "content_hash": f"sha:{source}"},
        episode_id="episode-1", target_key="match-1", action_id="lineup",
        request_id="action-request-1", sequence=1, prediction_cutoff=CUTOFF,
        captured_at="2026-09-28T08:20:00Z", requested_at="2026-09-28T07:59:00Z",
        source_version="v1",
    )


def trace():
    model = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    simulated = simulate_episode(model, seed=500, budget=.08, horizon=2, max_actions=2)
    return VoITrace.from_episode_event(
        simulated["trace"][0], episode_id="episode-1", target_key="match-1",
        request_id="action-request-1", model_version=model.VERSION,
        captured_at="2026-09-28T08:30:00Z",
    )


def test_all_required_preserves_independent_dependencies_and_requires_every_source():
    aggregate = aggregate_entries([entry("injury", "r1"), entry("lineup", "r2")], mode="all_required", cutoff=CUTOFF)
    assert aggregate.availability == "available"
    assert aggregate.quality == "VALIDATED"
    assert len(aggregate.entry_ids) == 2
    assert aggregate.dependency_keys == ("injury:r1", "lineup:r2")

    late = entry("lineup", "r2", availability="late", observed="2026-09-28T11:00:00Z",
                 operational="2026-09-28T11:10:00Z")
    assert aggregate_entries([entry("injury", "r1"), late], mode="all_required", cutoff=CUTOFF).availability == "late"


def test_any_usable_accepts_one_valid_source_and_retains_partial_quality():
    failed = entry("lineup", "r2", availability="failed", quality="PROVIDER_ERROR",
                   observed=None, operational=None)
    aggregate = aggregate_entries([entry("injury", "r1"), failed], mode="any_usable", cutoff=CUTOFF)
    assert aggregate.availability == "available"
    assert aggregate.quality == "VALIDATED"


def test_aggregate_rejects_mixed_action_identity():
    other = EvidenceLedgerEntry.from_row(
        {"candidate_id": "other", "source_id": "lineup", "dependency_key": "lineup:r2",
         "quality": "VALIDATED", "availability": "available", "observed_at": "2026-09-28T08:00:00Z",
         "operational_available_at": "2026-09-28T08:10:00Z", "evidence_refs": ["raw:other"], "content_hash": "sha:other"},
        episode_id="episode-1", target_key="match-1", action_id="lineup", request_id="action-request-1",
        sequence=2, prediction_cutoff=CUTOFF, captured_at="2026-09-28T08:20:00Z",
        requested_at="2026-09-28T07:59:00Z", source_version="v1",
    )
    with pytest.raises(ValueError, match="share episode/action/request/candidate identity"):
        aggregate_entries([entry("injury", "r1"), other], mode="any_usable", cutoff=CUTOFF)


def test_aggregate_link_reuses_voi_link_contract_without_hidden_results():
    aggregate = aggregate_entries([entry("injury", "r1"), entry("lineup", "r2")], mode="all_required", cutoff=CUTOFF)
    linked = link_trace_to_aggregate(trace(), aggregate, cutoff=CUTOFF)
    assert linked.status == "linked_available"
    assert linked.evidence_entry_id == aggregate.aggregate_id
    assert linked.dependency_key == f"aggregate:{aggregate.aggregate_id}"
    assert linked.observed_risk_reduction is not None
    assert linked.hidden_result_included is False


def test_any_usable_keeps_timely_evidence_when_other_source_is_captured_later():
    timely = entry("injury", "r1")
    late = replace(entry("lineup", "r2", availability="late", observed="2026-09-28T11:00:00Z",
                         operational="2026-09-28T11:10:00Z"), captured_at="2026-09-28T11:20:00Z")
    aggregate = aggregate_entries([timely, late], mode="any_usable", cutoff=CUTOFF)
    assert link_trace_to_aggregate(trace(), aggregate, cutoff=CUTOFF).status == "linked_available"


def test_shared_dependency_retries_are_retained_without_duplicate_support():
    success = entry("injury", "r1")
    failure = entry("injury", "r1", availability="failed", quality="PROVIDER_ERROR",
                    observed=None, operational=None)
    aggregate = aggregate_entries([failure, success], mode="all_required", cutoff=CUTOFF)
    assert aggregate.availability == "available"
    assert len(aggregate.entry_ids) == 2
    assert aggregate.dependency_keys == ("injury:r1",)


def test_link_rejects_a_different_cutoff_instead_of_reusing_a_stale_aggregate():
    aggregate = aggregate_entries([entry("injury", "r1")], mode="all_required", cutoff=CUTOFF)
    with pytest.raises(ValueError, match="cutoff"):
        link_trace_to_aggregate(trace(), aggregate, cutoff="2026-09-28T12:00:00Z")


def test_all_required_marks_an_undeclared_source_as_missing():
    aggregate = aggregate_entries(
        [entry("injury", "r1")], mode="all_required", cutoff=CUTOFF,
        required_dependency_keys=("injury:r1", "lineup:r2"),
    )
    assert aggregate.availability == "failed"
    assert aggregate.quality == "SOURCE_MISSING"
    assert aggregate.missing_dependency_keys == ("lineup:r2",)


def test_readiness_exposes_the_next_dependency_step_without_claiming_value():
    missing = aggregate_entries(
        [entry("injury", "r1")], mode="all_required", cutoff=CUTOFF,
        required_dependency_keys=("injury:r1", "lineup:r2"),
    )
    failed = entry("lineup", "r2", availability="failed", quality="PROVIDER_ERROR",
                   observed=None, operational=None)
    alternative = aggregate_entries([failed], mode="any_usable", cutoff=CUTOFF)
    assert readiness_for(missing).next_step == "QUERY_MISSING_DEPENDENCIES"
    assert readiness_for(missing).actionable_dependency_keys == ("lineup:r2",)
    assert readiness_for(alternative).next_step == "QUERY_ALTERNATIVE_SOURCE"
    assert readiness_for(alternative).failed_dependency_keys == ("lineup:r2",)


def test_readiness_filters_primary_retry_and_alternative_routes_before_voi():
    failed = entry("lineup", "r2", availability="failed", quality="PROVIDER_ERROR",
                   observed=None, operational=None)
    aggregate = aggregate_entries([failed], mode="any_usable", cutoff=CUTOFF)
    readiness = readiness_for(aggregate)
    rows = filter_candidate_routes(readiness, [
        {"candidate_id": "same-provider-retry", "dependency_key": "lineup:r2", "route_role": "retry"},
        {"candidate_id": "other-provider", "dependency_key": "lineup:r2", "route_role": "alternative"},
        {"candidate_id": "weather", "dependency_key": "weather:r3", "route_role": "alternative"},
    ])
    eligible = {row["candidate_id"] for row in rows if row["eligible"]}
    assert eligible == {"other-provider"}
    assert next(row for row in rows if row["candidate_id"] == "same-provider-retry")["reason"] == "route_not_alternative_for_failed_dependency"


def binding(dependency, *, availability="available", quality="VALIDATED", candidate="lineup"):
    return EvidenceBinding(
        candidate_id=candidate, source_id="source-1", dependency_key=dependency,
        quality=quality, observed_at="2026-09-28T08:00:00Z",
        operational_available_at="2026-09-28T08:10:00Z", expires_at=None,
        evidence_refs=("raw:1",), content_hash="sha:1", availability=availability,
        reason="test",
    )


def test_real_source_bindings_feed_readiness_without_creating_information_value():
    readiness = readiness_from_bindings([
        binding("injury:r1"),
        binding("lineup:r2", availability="late", candidate="lineup-late"),
    ], mode="all_required", required_dependency_keys=("injury:r1", "lineup:r2"))
    assert readiness.availability == "late"
    assert readiness.next_step == "WAIT_FOR_LATE_EVIDENCE"
    assert readiness.as_dict().get("expected_value") is None


def test_missing_required_binding_is_kept_distinct_from_provider_failure():
    readiness = readiness_from_bindings(
        [binding("injury:r1")], mode="all_required",
        required_dependency_keys=("injury:r1", "lineup:r2"),
    )
    assert readiness.availability == "failed"
    assert readiness.quality == "SOURCE_MISSING"
    assert readiness.next_step == "QUERY_MISSING_DEPENDENCIES"


def test_duplicate_binding_requests_an_independent_source():
    readiness = readiness_from_bindings(
        [binding("lineup:r2", availability="duplicate", quality="DUPLICATE")],
        mode="any_usable",
    )
    assert readiness.availability == "failed"
    assert readiness.quality == "DUPLICATE"
    assert readiness.next_step == "QUERY_ALTERNATIVE_SOURCE"
    assert readiness.duplicate_dependency_keys == ("lineup:r2",)
    routes = filter_candidate_routes(readiness, [
        {"candidate_id": "independent", "dependency_key": "lineup:r2", "route_role": "alternative"},
    ])
    assert routes[0]["eligible"] is True


def test_binding_quality_conflict_and_unverified_survive_readiness_conversion():
    conflict = readiness_from_bindings(
        [binding("list:r1", availability="failed", quality="CONFLICT")], mode="any_usable",
    )
    unverified = readiness_from_bindings(
        [binding("odds:r2", availability="available", quality="UNVERIFIED")], mode="any_usable",
    )
    assert conflict.quality == "CONFLICT"
    assert unverified.quality == "UNVERIFIED"


def test_stale_binding_routes_to_refresh_or_independent_source():
    stale = readiness_from_bindings(
        [binding("lineup:r4", availability="failed", quality="STALE")], mode="any_usable",
    )
    assert stale.quality == "STALE"
    assert stale.next_step == "QUERY_ALTERNATIVE_SOURCE"
    assert stale.failed_dependency_keys == ("lineup:r4",)


def test_binding_level_query_plan_exposes_alternative_without_losing_action_level_any_usable():
    available = binding("recent:r1", candidate="recent")
    conflict = binding("list:r2", availability="failed", quality="CONFLICT", candidate="list")
    routes = [
        {"candidate_id": "recent", "dependency_key": "recent:r1", "route_role": "primary"},
        {"candidate_id": "list", "dependency_key": "list:r2", "route_role": "primary"},
        {"candidate_id": "list-alt", "dependency_key": "list:r2", "route_role": "alternative"},
        {"candidate_id": "odds", "dependency_key": "odds:r3", "route_role": "primary"},
    ]
    route_rows, by_dependency = filter_candidate_routes_by_bindings([available, conflict], routes)
    assert by_dependency["recent:r1"].next_step == "STOP_ACTION_SATISFIED"
    assert by_dependency["list:r2"].next_step == "QUERY_ALTERNATIVE_SOURCE"
    assert {row["candidate_id"] for row in route_rows if row["eligible"]} == {"list-alt", "odds"}

    target = "match-1"
    board = StateQueryBoard(
        target_key=target, prediction_cutoff=CUTOFF,
        hypotheses=[StateHypothesis("recent-h", target, "recent", ("known", "unknown")),
                    StateHypothesis("list-h", target, "list", ("known", "unknown")),
                    StateHypothesis("odds-h", target, "odds", ("known", "unknown"))],
        candidates=[StateQueryCandidate(candidate, target, CUTOFF, variable, "source", "read", .01)
                    for candidate, variable in (("recent", "recent"), ("list", "list"), ("list-alt", "list"), ("odds", "odds"))],
    )
    plan = prepare_query_board_from_bindings(board, [available, conflict], routes)
    assert plan.eligible_candidate_ids == ("list-alt", "odds")
    assert [candidate.candidate_id for candidate in plan.board.candidates] == ["list-alt", "odds"]
    assert plan.as_dict()["readiness_by_dependency"]["list:r2"]["quality"] == "CONFLICT"
