import pytest

from football_agent.decision_value import simulate_episode, synthetic_scenarios
from football_agent.evidence_ledger import EvidenceLedgerEntry
from football_agent.voi_evidence_bridge import link_trace
from football_agent.voi_trace import VoITrace


CUTOFF = "2026-09-27T10:00:00Z"


def action_trace():
    model = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    episode = simulate_episode(model, seed=500, budget=.08, horizon=2, max_actions=2)
    event = episode["trace"][0]
    return VoITrace.from_episode_event(
        event, episode_id="episode-1", target_key="match-1", request_id="request-1",
        model_version=model.VERSION, captured_at="2026-09-27T08:00:00Z",
    )


def evidence(trace, *, availability="available", observed="2026-09-27T08:00:00Z",
             available="2026-09-27T08:10:00Z", captured="2026-09-27T08:20:00Z",
             requested="2026-09-27T07:55:00Z", source_version="fixture-v1"):
    return EvidenceLedgerEntry.from_row(
        {"candidate_id": trace.candidate_id, "source_id": "fixture-source",
         "dependency_key": "dependency-lineup", "quality": "VALIDATED" if availability == "available" else "PROVIDER_ERROR",
         "availability": availability, "observed_at": observed, "operational_available_at": available,
         "evidence_refs": ["archive:fixture"], "content_hash": "sha:fixture"},
        episode_id=trace.episode_id, target_key=trace.target_key, action_id=trace.action_id,
        request_id=trace.request_id, sequence=1, prediction_cutoff=CUTOFF,
        captured_at=captured, requested_at=requested, source_version=source_version,
    )


def test_link_joins_request_and_reports_observed_vs_expected_risk():
    trace = action_trace()
    linked = link_trace(trace, [evidence(trace)], cutoff=CUTOFF)
    assert linked.status == "linked_available"
    assert linked.evidence_entry_id
    assert linked.observed_risk_reduction is not None
    assert linked.calibration_error == pytest.approx(
        linked.observed_risk_reduction - trace.expected_information_value
    )
    assert linked.requested_at == "2026-09-27T07:55:00+00:00"
    assert linked.source_version == "fixture-v1"
    assert linked.as_dict()["hidden_result_included"] is False


def test_late_or_failed_evidence_remains_linked_but_unusable():
    trace = action_trace()
    late = evidence(trace, availability="late", observed="2026-09-27T10:15:00Z",
                    available="2026-09-27T10:20:00Z")
    linked = link_trace(trace, [late], cutoff=CUTOFF)
    assert linked.status == "linked_unusable"
    assert linked.evidence_availability == "late"
    assert linked.observed_risk_reduction is None
    assert linked.calibration_error is None


def test_post_cutoff_capture_is_unusable_even_when_provider_says_available():
    trace = action_trace()
    future = evidence(trace, captured="2026-09-27T10:25:00Z")
    linked = link_trace(trace, [future], cutoff=CUTOFF)
    assert linked.status == "linked_unusable"


def test_missing_and_ambiguous_evidence_are_explicit():
    trace = action_trace()
    assert link_trace(trace, [], cutoff=CUTOFF).status == "missing_evidence"
    duplicate = evidence(trace)
    assert link_trace(trace, [duplicate, duplicate], cutoff=CUTOFF).status == "ambiguous_evidence"


def test_stop_trace_is_linked_without_claiming_evidence():
    model = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    episode = simulate_episode(model, seed=500, budget=.08, horizon=2, max_actions=2)
    event = episode["trace"][-1]
    trace = VoITrace.from_episode_event(
        event, episode_id="episode-stop", target_key="match-1", request_id="request-stop",
        model_version=model.VERSION,
    )
    linked = link_trace(trace, [], cutoff=CUTOFF)
    assert linked.status == "stop_recorded"
    assert linked.evidence_entry_id is None
