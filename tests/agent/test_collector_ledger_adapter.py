import json
from pathlib import Path
from types import SimpleNamespace

from football_agent.collector_ledger_adapter import (
    VERSION, collection_events, events_to_ledger_entries, normalized_batch_mapping_state,
    observation_event,
)
from football_agent.action_readiness import readiness_from_bindings
from football_agent.evidence_binding import bind_candidate_evidence
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate
from football_spider.normalization.sources import normalize_spftll


CUTOFF = "2026-09-28T10:00:00Z"
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "collectors" / "spftll"


def obs(**overrides):
    value = {
        "id": "obs-1", "source": "spftll", "request_id": "req-1",
        "endpoint": "https://xx.spftll.cn/api/match/GetInfo",
        "observed_at": "2026-09-28T08:01:00Z",
        "http_status": 200, "payload_path": "raw/spftll/obs.body.gz",
        "content_hash": "sha256:body", "error_class": None,
        "context": {"requested_at": "2026-09-28T08:00:00Z"},
    }
    value.update(overrides)
    return value


def test_real_observation_preserves_request_timing_dependency_and_mapping_state():
    event = observation_event(obs(), result_status="OK", source_version="spftll-parser-v1",
                              prediction_cutoff=CUTOFF, candidate_id="lineup")
    assert event["schema_version"] == VERSION
    assert event["quality"] == "UNVERIFIED"
    assert event["availability"] == "available"
    assert event["requested_at"] == "2026-09-28T08:00:00+00:00"
    assert event["dependency_key"] == "spftll:request:req-1"
    assert event["entity_mapping_state"] == "not_evaluated"
    assert event["operational_available_at"] == event["observed_at"]


def test_late_and_failed_responses_do_not_claim_operational_availability():
    late = observation_event(obs(observed_at="2026-09-28T11:00:00Z"), result_status="OK",
                             source_version="spftll-parser-v1", prediction_cutoff=CUTOFF)
    failed = observation_event(obs(http_status=503, error_class="HTTP_503"), result_status="PARTIAL",
                               source_version="spftll-parser-v1", prediction_cutoff=CUTOFF,
                               entity_mapping_state="ambiguous")
    assert late["availability"] == "late"
    assert late["operational_available_at"] == late["observed_at"]
    assert failed["quality"] == "PROVIDER_ERROR"
    assert failed["availability"] == "failed"
    assert failed["operational_available_at"] is None


def test_collection_result_uses_only_archived_observation_ids_and_records_entity_mapping():
    result = {"source": "spftll", "status": "PARTIAL", "observations": ["obs-1", "obs-2"]}
    events = collection_events(
        result, [obs(), obs(id="obs-2", request_id="req-2", observed_at="2026-09-28T08:02:00Z")],
        source_version="spftll-parser-v1", prediction_cutoff=CUTOFF,
        candidate_by_observation={"obs-1": "fixture-1", "obs-2": "fixture-2"},
        entity_mapping_by_observation={"obs-1": "mapped", "obs-2": "unmapped"},
    )
    assert [event["candidate_id"] for event in events] == ["fixture-1", "fixture-2"]
    assert events[0]["quality"] == "PARTIAL"
    assert events[1]["quality"] == "SOURCE_MISSING"
    assert events[1]["availability"] == "failed"


def test_events_materialize_ledger_entries_with_real_source_metadata():
    event = observation_event(obs(), result_status="OK", source_version="spftll-parser-v1",
                              prediction_cutoff=CUTOFF, candidate_id="fixture-1",
                              entity_mapping_state="mapped")
    entries = events_to_ledger_entries([event], episode_id="episode-1", target_key="match:1",
                                       action_id="collect_fixture", prediction_cutoff=CUTOFF)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.source_id == "spftll"
    assert entry.source_version == "spftll-parser-v1"
    assert entry.dependency_key == "spftll:request:req-1"
    assert entry.requested_at == "2026-09-28T08:00:00+00:00"
    assert entry.quality == "VALIDATED"
    assert entry.availability == "available"


def test_normalized_batch_derives_mapping_state_without_manual_labels():
    stable = {"matches": [{"competition": {"id_quality": "stable"},
                            "home": {"id_quality": "stable"},
                            "away": {"id_quality": "stable"}}], "issues": []}
    source_name = {"matches": [{"competition": {"id_quality": "stable"},
                                 "home": {"id_quality": "source_name"},
                                 "away": {"id_quality": "stable"}}], "issues": []}
    scoped = {"matches": [{"competition": {"id_quality": "stable"},
                            "home": {"id_quality": "match_scoped"},
                            "away": {"id_quality": "stable"}}], "issues": []}
    assert normalized_batch_mapping_state(stable) == "mapped"
    assert normalized_batch_mapping_state(source_name) == "ambiguous"
    assert normalized_batch_mapping_state(scoped) == "unmapped"


def test_collection_events_can_derive_mapping_from_normalized_batch():
    result = {"source": "spftll", "status": "OK", "observations": ["obs-1"]}
    events = collection_events(
        result, [obs()], source_version="spftll-parser-v1", prediction_cutoff=CUTOFF,
        normalized_by_observation={"obs-1": {"matches": [{
            "competition": {"id_quality": "stable"},
            "home": {"id_quality": "source_name"},
            "away": {"id_quality": "stable"},
        }], "issues": []}},
    )
    assert events[0]["entity_mapping_state"] == "ambiguous"
    assert events[0]["quality"] == "CONFLICT"


def test_redacted_real_capture_fixtures_reach_binding_and_readiness_with_quality_boundaries():
    cutoff = "2026-09-20T12:00:00Z"
    rows = []
    expected = {}
    for name, candidate in (("recent", "recent"), ("list", "list"), ("odds", "odds")):
        metadata = json.loads((FIXTURES / f"{name}.metadata.json").read_text(encoding="utf-8"))["observation"]
        payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
        normalized = normalize_spftll(SimpleNamespace(**metadata), payload)
        mapping_state = normalized_batch_mapping_state(normalized)
        event = observation_event(
            metadata, result_status="OK", source_version="spftll-fixture-v1",
            prediction_cutoff=cutoff, candidate_id=candidate,
            entity_mapping_state=mapping_state,
        )
        rows.append(event)
        expected[candidate] = (mapping_state, event["quality"], event["availability"])

    board = StateQueryBoard(
        target_key="fixture-target", prediction_cutoff=cutoff,
        hypotheses=[StateHypothesis(f"h-{candidate}", "fixture-target", candidate, ("known", "unknown"))
                    for candidate in ("recent", "list", "odds")],
        candidates=[StateQueryCandidate(candidate, "fixture-target", cutoff, candidate, "spftll", "read", .01)
                    for candidate in ("recent", "list", "odds")],
    )
    bindings = bind_candidate_evidence(board, rows, cutoff=cutoff).bindings
    by_candidate = {binding.candidate_id: binding for binding in bindings}
    readiness = readiness_from_bindings(bindings, mode="any_usable")
    assert expected["recent"] == ("mapped", "VALIDATED", "available")
    assert expected["list"] == ("ambiguous", "CONFLICT", "failed")
    assert expected["odds"] == ("not_evaluated", "UNVERIFIED", "available")
    assert by_candidate["recent"].availability == "available"
    assert by_candidate["list"].availability == "failed"
    assert readiness.availability == "available"
    assert readiness.next_step == "STOP_ACTION_SATISFIED"
