import json

import pytest

from football_agent.evidence_ledger import EvidenceLedger, EvidenceLedgerEntry
from football_platform.information import AcquisitionResult, CostVector, EvidenceGroup, EvidenceTimes


CUTOFF = "2026-09-27T10:00:00Z"


def row(candidate_id="lineup", *, availability="available", observed="2026-09-27T08:00:00Z",
        available="2026-09-27T08:10:00Z", dependency="dep-lineup", quality="VALIDATED"):
    return {
        "candidate_id": candidate_id,
        "source_id": "source:lineup",
        "dependency_key": dependency,
        "quality": quality,
        "availability": availability,
        "observed_at": observed,
        "operational_available_at": available,
        "evidence_refs": [f"archive:{candidate_id}"],
        "content_hash": f"hash:{candidate_id}",
    }


def entry(row_data=None, *, candidate="lineup", captured="2026-09-27T08:20:00Z", sequence=1,
          target="match-1", resolution="UNRESOLVED"):
    return EvidenceLedgerEntry.from_row(
        row_data or row(candidate),
        episode_id="episode-1",
        target_key=target,
        action_id=candidate,
        request_id=f"request-{candidate}-{sequence}",
        sequence=sequence,
        prediction_cutoff=CUTOFF,
        captured_at=captured,
        resolution_status=resolution,
    )


def actual_group():
    return EvidenceGroup(
        group_id="lineup",
        values={"status": "redacted"},
        source="spftll",
        times=EvidenceTimes(
            observed_at="2026-09-27T08:00:00Z",
            operational_available_at="2026-09-27T08:10:00Z",
            feature_available_at="2026-09-27T08:10:00Z",
            basis="actual_observation_and_ingestion",
        ),
        quality="VALIDATED",
        evidence_refs=["archive:lineup"],
        version="v1",
        source_id="spftll:fixture",
        observed_at="2026-09-27T08:00:00Z",
        available_at="2026-09-27T08:10:00Z",
        content_hash="sha:lineup",
        dependency_key="spftll:lineup",
    )


def actual_result():
    return AcquisitionResult(
        episode_id="episode-platform",
        action_id="lineup",
        request_id="request-platform-1",
        status="ACQUIRED",
        revealed_groups={"lineup": actual_group()},
        charged_cost=CostVector(),
        sequence=1,
        resulting_cutoff=CUTOFF,
    )


def test_append_only_ledger_roundtrips_record_and_checksum(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    saved = ledger.append([entry()])
    assert saved[0].entry_id == ledger.entries()[0].entry_id
    assert ledger.entries()[0].resolution_status == "UNRESOLVED"
    line = (tmp_path / "evidence-ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
    envelope = json.loads(line)
    assert envelope["schema_version"] == "evidence-ledger-v1"
    assert envelope["record_sha256"]


def test_duplicate_entry_identity_preserves_append_only_history(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    item = entry()
    ledger.append([item])
    with pytest.raises(ValueError, match="append-only"):
        ledger.append([item])
    assert len(ledger.entries()) == 1


def test_tampered_record_is_detected_before_snapshot(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    ledger.append([entry()])
    path = tmp_path / "evidence-ledger.jsonl"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["record"]["quality"] = "PARTIAL"
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ledger.entries()


def test_snapshot_selects_latest_entry_per_candidate_and_keeps_resolution_status(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    ledger.append([entry(captured="2026-09-27T08:20:00Z", sequence=1)])
    ledger.append([entry(
        row("lineup", availability="late", observed="2026-09-27T10:15:00Z",
            available="2026-09-27T10:20:00Z", quality="VALIDATED"),
        captured="2026-09-27T09:25:00Z", sequence=2, resolution="CONTESTED",
    )])
    rows = ledger.snapshot_rows(target_key="match-1", prediction_cutoff=CUTOFF)
    assert len(rows) == 1
    assert rows[0]["availability"] == "late"
    assert rows[0]["resolution_status"] == "CONTESTED"


def test_snapshot_excludes_late_attempt_captured_after_cutoff(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    ledger.append([entry(captured="2026-09-27T08:20:00Z", sequence=1)])
    ledger.append([entry(
        row("lineup", availability="late", observed="2026-09-27T10:15:00Z",
            available="2026-09-27T10:20:00Z", quality="VALIDATED"),
        captured="2026-09-27T10:25:00Z", sequence=2, resolution="CONTESTED",
    )])
    rows = ledger.snapshot_rows(target_key="match-1", prediction_cutoff=CUTOFF)
    assert rows[0]["availability"] == "available"
    later = ledger.snapshot_rows(target_key="match-1", prediction_cutoff="2026-09-27T11:00:00Z")
    assert later[0]["availability"] == "late"


def test_snapshot_target_and_candidate_filters_are_explicit(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    ledger.append([entry(target="match-1"), entry(candidate="weather", target="match-2")])
    assert [item["candidate_id"] for item in ledger.snapshot_rows(
        target_key="match-1", prediction_cutoff=CUTOFF, candidate_ids=["lineup"]
    )] == ["lineup"]
    assert ledger.snapshot_rows(target_key="match-2", prediction_cutoff=CUTOFF)[0]["candidate_id"] == "weather"


def test_platform_acquisition_result_can_be_appended_without_reconstructing_rows(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    saved = ledger.append_acquisition_result(
        actual_result(), action_to_candidate={"lineup": "lineup"},
        target_key="match-platform", prediction_cutoff=CUTOFF,
        captured_at="2026-09-27T08:15:00Z",
    )
    assert saved[0].episode_id == "episode-platform"
    assert saved[0].source_id == "spftll:fixture"
    assert ledger.snapshot_rows(target_key="match-platform", prediction_cutoff=CUTOFF)[0]["candidate_id"] == "lineup"


def test_request_time_and_source_version_survive_ledger_snapshot(tmp_path):
    item = EvidenceLedgerEntry.from_row(
        row("lineup"), episode_id="episode-1", target_key="match-1", action_id="lineup",
        request_id="request-lineup-2", sequence=2, prediction_cutoff=CUTOFF,
        requested_at="2026-09-27T07:55:00Z", captured_at="2026-09-27T08:20:00Z",
        source_version="spftll-schema-v3",
    )
    ledger = EvidenceLedger(tmp_path)
    ledger.append([item])
    saved = ledger.entries()[0]
    assert saved.requested_at == "2026-09-27T07:55:00+00:00"
    assert saved.source_version == "spftll-schema-v3"
    snapshot = ledger.snapshot_rows(target_key="match-1", prediction_cutoff=CUTOFF)[0]
    assert snapshot["requested_at"] == saved.requested_at
    assert snapshot["source_version"] == "spftll-schema-v3"


def test_acquisition_result_adapter_accepts_request_time_and_source_version(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    saved = ledger.append_acquisition_result(
        actual_result(), action_to_candidate={"lineup": "lineup"},
        target_key="match-platform", prediction_cutoff=CUTOFF,
        requested_at="2026-09-27T07:55:00Z", captured_at="2026-09-27T08:15:00Z",
        source_version="spftll-contract-v3",
    )
    assert saved[0].requested_at == "2026-09-27T07:55:00+00:00"
    assert saved[0].source_version == "spftll-contract-v3"


def test_request_time_after_capture_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="requested_at"):
        EvidenceLedgerEntry.from_row(
            row("lineup"), episode_id="episode-1", target_key="match-1", action_id="lineup",
            request_id="request-lineup-3", sequence=3, prediction_cutoff=CUTOFF,
            requested_at="2026-09-27T08:30:00Z", captured_at="2026-09-27T08:20:00Z",
        )


@pytest.mark.parametrize("availability", ["late", "failed", "duplicate"])
def test_non_available_statuses_are_recorded_with_explicit_identity(tmp_path, availability):
    data = row("lineup", availability=availability)
    if availability == "late":
        data.update(observed_at="2026-09-27T10:15:00Z", operational_available_at="2026-09-27T10:20:00Z")
    elif availability == "failed":
        data.update(observed_at=None, operational_available_at=None, quality="PROVIDER_ERROR")
    item = entry(data)
    ledger = EvidenceLedger(tmp_path)
    ledger.append([item])
    assert ledger.entries()[0].availability == availability
