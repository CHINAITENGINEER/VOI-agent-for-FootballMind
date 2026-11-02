import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_agent.collector_context_adapter import (
    VERSION,
    build_context_snapshot,
    merge_context_snapshots,
)
from football_agent.collector_ledger_adapter import (
    normalized_batch_mapping_state,
    observation_event,
)
from football_spider.normalization.sources import normalize_spftll


CUTOFF = "2026-09-20T12:00:00Z"
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "collectors" / "spftll"


def fixture_event(name: str, *, cutoff: str = CUTOFF, observed_at: str | None = None,
                  status: str = "OK", quality_override: str | None = None):
    metadata = json.loads((FIXTURES / f"{name}.metadata.json").read_text(encoding="utf-8"))["observation"]
    if observed_at is not None:
        metadata["observed_at"] = observed_at
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    normalized = normalize_spftll(SimpleNamespace(**metadata), payload)
    mapping_state = normalized_batch_mapping_state(normalized)
    event = observation_event(
        metadata, result_status=status, source_version="spftll-fixture-v1",
        prediction_cutoff=cutoff, candidate_id=name,
        entity_mapping_state=mapping_state,
    )
    if quality_override is not None:
        event["quality"] = quality_override
    return event, normalized


def test_real_spftll_scheduled_fixture_becomes_public_context_with_provenance():
    event, normalized = fixture_event("detail")
    snapshot = build_context_snapshot(event, normalized, target_source_match_id="2996160", target_key="match:2996160")

    assert snapshot.context.values["fixture_id"] == "2996160"
    assert snapshot.context.values["pre_match_status"] == "scheduled"
    assert snapshot.context.values["kickoff_at"] == "2026-09-20T12:30:00+00:00"
    assert snapshot.context.values["lead_time_seconds"] == 1800.0
    assert snapshot.context.values["odds_available"] is False
    assert snapshot.evidence[0].quality == "VALIDATED"
    assert snapshot.evidence[0].observed_at == "2026-09-20T11:44:27.725604+00:00"
    assert snapshot.evidence[0].content_hash
    assert snapshot.as_dict()["schema_version"] == VERSION


def test_finished_fixture_projects_structure_but_excludes_result_and_finished_status():
    event, normalized = fixture_event("recent")
    snapshot = build_context_snapshot(event, normalized, target_source_match_id="2912895", target_key="match:2912895")
    dumped = json.dumps(snapshot.as_dict(), ensure_ascii=False)

    assert snapshot.context.values["fixture_id"] == "2912895"
    assert "pre_match_status" not in snapshot.context.values
    assert "score" not in dumped
    assert "home_score" not in dumped
    assert "away_score" not in dumped
    public_keys = set(snapshot.context.values)
    assert not {"result", "outcome", "score", "home_score", "away_score"} & public_keys
    assert all(row.context_key not in {"result", "outcome", "score", "home_score", "away_score"}
               for row in snapshot.evidence)
    assert snapshot.hidden_result_included is False


def test_ambiguous_mapping_keeps_audit_rows_but_does_not_promote_context():
    event, normalized = fixture_event("list")
    snapshot = build_context_snapshot(event, normalized, target_source_match_id="3013707", target_key="match:3013707")

    assert snapshot.context.values == {}
    assert snapshot.evidence
    assert all(row.accepted is False for row in snapshot.evidence)
    assert all(row.quality == "CONFLICT" for row in snapshot.evidence)
    assert all("source_available_at_cutoff" not in row.reason for row in snapshot.evidence)


def test_late_or_failed_collection_is_retained_as_unusable_evidence():
    event, normalized = fixture_event("detail", observed_at="2026-09-20T13:00:00Z")
    snapshot = build_context_snapshot(event, normalized, target_source_match_id="2996160")
    assert snapshot.context.values == {}
    assert all(row.accepted is False for row in snapshot.evidence)
    assert all(row.reason.startswith("event_not_usable") for row in snapshot.evidence)


def test_quote_only_snapshot_can_add_odds_context_without_inventing_fixture_identity():
    event, normalized = fixture_event("odds")
    snapshot = build_context_snapshot(event, normalized, target_source_match_id="2996160", target_key="match:2996160")
    assert snapshot.context.values == {"odds_available": True, "quote_count": 14}
    assert all(row.context_key in {"odds_available", "quote_count"} for row in snapshot.evidence)


def test_merging_detail_and_quote_snapshots_preserves_context_and_detects_conflict():
    detail_event, detail_batch = fixture_event("detail")
    odds_event, odds_batch = fixture_event("odds")
    detail = build_context_snapshot(detail_event, detail_batch, target_source_match_id="2996160", target_key="match:2996160")
    odds = build_context_snapshot(odds_event, odds_batch, target_source_match_id="2996160", target_key="match:2996160")
    merged = merge_context_snapshots((detail, odds))
    assert merged.context.values["fixture_id"] == "2996160"
    assert merged.context.values["odds_available"] is True
    assert merged.context.values["quote_count"] == 14
    assert not merged.conflict_keys
    assert len(merged.evidence) == len(detail.evidence) + len(odds.evidence)


def test_hidden_result_marker_is_rejected_before_projection():
    event, normalized = fixture_event("detail")
    event["hidden_result_included"] = True
    with pytest.raises(ValueError, match="hidden result"):
        build_context_snapshot(event, normalized, target_source_match_id="2996160")
