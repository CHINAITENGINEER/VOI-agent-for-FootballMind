from football_agent.collector_ledger_adapter import observation_event
from football_agent.evidence_ledger import EvidenceLedger
from football_agent.manifest_ledger_ingestor import (
    VERSION, manifest_events_to_ledger_entries, rebind_event_cutoff,
)
from football_agent.voi_evidence_bridge import link_trace
from football_agent.voi_trace import VoITrace
from football_agent.decision_value import simulate_episode, synthetic_scenarios


def obs(observation_id, request_id, observed_at, *, status=200, error=None):
    return {
        "id": observation_id, "source": "spftll", "request_id": request_id,
        "endpoint": "https://xx.spftll.cn/api/match/GetInfo",
        "observed_at": observed_at, "http_status": status, "error_class": error,
        "payload_path": f"raw/spftll/{observation_id}.body.gz",
        "content_hash": f"sha:{observation_id}",
        "context": {"requested_at": observed_at.replace("09:00", "08:59").replace("11:00", "10:59")},
    }


def event(observation_id="obs-1", request_id="req-1", observed_at="2026-09-28T09:00:00Z",
          captured_at="2026-09-28T13:00:00Z"):
    return observation_event(
        obs(observation_id, request_id, observed_at), result_status="OK",
        source_version="spftll-parser-v1", prediction_cutoff="2026-09-28T12:00:00Z",
        candidate_id=f"candidate:{observation_id}", entity_mapping_state="mapped",
        captured_at=captured_at,
    )


def manifest(events):
    return {
        "source": "spftll",
        "collector_events_cutoff_kind": "collector_run_close_not_prediction_cutoff",
        "collector_events": events,
    }


def test_rebind_recomputes_available_and_late_at_decision_cutoff():
    available = rebind_event_cutoff(event(), prediction_cutoff="2026-09-28T10:00:00Z")
    late = rebind_event_cutoff(event("obs-2", "req-2", "2026-09-28T11:00:00Z"),
                               prediction_cutoff="2026-09-28T10:00:00Z")
    assert available["availability"] == "available"
    assert late["availability"] == "late"
    assert late["operational_available_at"] == "2026-09-28T11:00:00+00:00"
    assert available["ingestor_version"] == VERSION


def test_manifest_ingestion_preserves_capture_time_and_excludes_post_cutoff_snapshot(tmp_path):
    entries = manifest_events_to_ledger_entries(
        manifest([event()]), episode_id="episode-1", target_key="match:1",
        action_id="collect_fixture", prediction_cutoff="2026-09-28T10:00:00Z",
    )
    assert len(entries) == 1
    assert entries[0].availability == "available"
    assert entries[0].captured_at == "2026-09-28T13:00:00+00:00"
    ledger = EvidenceLedger(tmp_path)
    ledger.append(entries)
    assert ledger.snapshot_rows(target_key="match:1", prediction_cutoff="2026-09-28T10:00:00Z") == []
    assert len(ledger.snapshot_rows(target_key="match:1", prediction_cutoff="2026-09-28T14:00:00Z")) == 1


def test_manifest_ingestion_rejects_ambiguous_cutoff_semantics():
    raw = {"collector_events": [event()]}
    try:
        manifest_events_to_ledger_entries(
            raw, episode_id="episode-1", target_key="match:1", action_id="collect_fixture",
            prediction_cutoff="2026-09-28T10:00:00Z",
        )
    except ValueError as exc:
        assert "cutoff semantics" in str(exc)
    else:
        raise AssertionError("manifest without explicit collector cutoff semantics was accepted")


def test_manifest_ingestion_binds_source_request_to_agent_action_for_voi_link():
    model = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    episode = simulate_episode(model, seed=500, budget=.08, horizon=2, max_actions=2)
    episode_event = episode["trace"][0]
    trace = VoITrace.from_episode_event(
        episode_event, episode_id="episode-voi", target_key="match-1", request_id="action-request-1",
        model_version=model.VERSION, captured_at="2026-09-28T08:00:00Z",
    )
    source_event = event("obs-1", "source-request-1", "2026-09-28T08:30:00Z", "2026-09-28T08:45:00Z")
    manifest_entries = manifest_events_to_ledger_entries(
        manifest([source_event]), episode_id=trace.episode_id, target_key=trace.target_key,
        action_id=trace.action_id, action_request_id=trace.request_id,
        candidate_id=trace.candidate_id, prediction_cutoff="2026-09-28T09:00:00Z",
    )
    assert manifest_entries[0].request_id == trace.request_id
    assert "source-request:source-request-1" in manifest_entries[0].evidence_refs
    linked = link_trace(trace, manifest_entries, cutoff="2026-09-28T09:00:00Z")
    assert linked.status == "linked_available"


def test_action_binding_rejects_multiple_events_until_an_aggregation_policy_is_explicit():
    try:
        manifest_events_to_ledger_entries(
            manifest([event("obs-1"), event("obs-2", "req-2")]),
            episode_id="episode-1", target_key="match:1", action_id="lineup",
            action_request_id="action-request-1", candidate_id="lineup",
            prediction_cutoff="2026-09-28T10:00:00Z",
        )
    except ValueError as exc:
        assert "exactly one collector event" in str(exc)
    else:
        raise AssertionError("multiple events were bound to one VoI action")


def test_cutoff_rebinding_preserves_processing_delay_and_failed_or_duplicate_status():
    raw = event(captured_at="2026-09-28T09:30:00Z")
    raw["operational_available_at"] = "2026-09-28T11:00:00Z"
    rebound = rebind_event_cutoff(raw, prediction_cutoff="2026-09-28T10:00:00Z")
    assert rebound["availability"] == "late"
    assert rebound["operational_available_at"] == "2026-09-28T11:00:00+00:00"
    for status in ("failed", "duplicate"):
        raw["availability"] = status
        assert rebind_event_cutoff(raw, prediction_cutoff="2026-09-28T12:00:00Z")["availability"] == status
