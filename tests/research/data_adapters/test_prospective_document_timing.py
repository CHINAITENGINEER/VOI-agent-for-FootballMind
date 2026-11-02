import pytest

from football_research.data_adapters.prospective_stream import (
    _document_timing_fields,
    _link_document_versions,
    _result_visibility_fields,
)


def test_document_response_and_projection_clocks_do_not_become_capture_or_handoff():
    row = _document_timing_fields(
        {"observed_at": "2026-10-09T08:00:00Z", "available_at": "2026-10-09T08:00:01Z"},
        {"observed_at": "2026-10-09T08:00:00Z", "context": {}}, {})
    assert row["observed_at"] == "2026-10-09T08:00:00+00:00"
    assert row["captured_at"] is None
    assert row["capture_status"] == row["capture_timing_basis"] == "unknown"
    assert row["available_at"] is None
    assert row["source_projection_available_at"] == "2026-10-09T08:00:01+00:00"


def archive_and_handoff():
    handoff = {"raw_context_timing_proven": True, "availability_basis": "runtime_state_handoff",
               "available_at": "2026-10-09T08:00:04Z", "snapshot_id": "after-1",
               "raw_document_membership_proven": True, "known_document_ids": ["doc1"],
               "raw_path": "raw/context.gz", "raw_sha256": "digest"}
    observation = {"id": "obs1", "content_hash": "body-digest",
        "observed_at": "2026-10-09T08:00:00Z", "context": {
        "archive_save_started_at": "2026-10-09T08:00:02Z",
        "archive_save_timing_basis": "measured_local_utc_archive_save_start"}}
    receipt = {"archive_save_timing": {"observation_id": "obs1", "raw_sha256": "body-digest",
        "archive_save_started_at": "2026-10-09T08:00:02Z", "archive_save_returned_at": "2026-10-09T08:00:03Z",
        "archive_return_timing_basis": "measured_local_utc_archive_save_return"}}
    return observation, handoff, receipt


def test_document_requires_archive_return_and_verified_document_membership():
    observation, handoff, receipt = archive_and_handoff()
    row = _document_timing_fields({"document_id": "doc1", "available_at": "2026-10-09T08:00:03Z"}, observation, handoff, receipt)
    assert row["archive_save_started_at"] == "2026-10-09T08:00:02+00:00"
    assert row["captured_at"] == "2026-10-09T08:00:03+00:00"
    assert row["capture_status"] == "known"
    assert row["persistence_boundary"] == "return_observed_fsync_not_instrumented"
    assert row["document_admitted"] is True
    assert row["available_at"] == "2026-10-09T08:00:04+00:00"
    assert row["document_available_at"] == row["available_at"]
    assert row["availability_raw_path"] == "raw/context.gz"
    handoff["raw_context_timing_proven"] = False
    assert _document_timing_fields({}, observation, handoff)["available_at"] is None


def test_save_start_alone_does_not_prove_completed_capture():
    observation, handoff, _ = archive_and_handoff()
    row = _document_timing_fields({"document_id": "doc1"}, observation, handoff)
    assert row["archive_save_started_at"] is not None
    assert row["captured_at"] is None
    assert row["capture_status"] == "unknown"
    assert row["archive_save_returned_at"] is None


@pytest.mark.parametrize("field,value", [
    ("observation_id", "other"), ("raw_sha256", "other"),
    ("archive_save_started_at", "2026-10-09T08:00:01Z"),
    ("archive_save_returned_at", "2026-10-09T08:00:01Z"),
    ("archive_return_timing_basis", "copied_response_time"),
])
def test_mismatched_return_receipt_does_not_prove_capture(field, value):
    observation, handoff, receipt = archive_and_handoff()
    receipt["archive_save_timing"][field] = value
    assert _document_timing_fields({"document_id": "doc1"}, observation, handoff, receipt)["captured_at"] is None


def test_no_data_receipt_is_observable_but_not_admitted_as_a_document():
    observation, handoff, receipt = archive_and_handoff()
    handoff["known_document_ids"] = []
    row = _document_timing_fields({"document_id": "empty-doc", "text": "", "status": "NO_DATA"}, observation, handoff, receipt)
    visible = _result_visibility_fields({"returned_at": "2026-10-09T08:00:03.500Z"})
    assert visible["result_observed"] is True
    assert visible["receipt_available_at"] is not None
    assert row["captured_at"] is not None
    assert row["document_admitted"] is False
    assert row["available_at"] is row["document_available_at"] is None
    assert row["availability_raw_path"] is None


def test_unproven_raw_membership_cannot_establish_admission_from_a_later_snapshot():
    observation, handoff, receipt = archive_and_handoff()
    handoff.pop("raw_document_membership_proven")
    row = _document_timing_fields({"document_id": "doc1"}, observation, handoff, receipt)
    assert row["document_admitted"] is None
    assert row["available_at"] is None


def test_document_version_links_use_same_url_and_strict_observation_order():
    first = {"document_id": "one", "version_id": "v1", "canonical_url": "https://example/a",
             "observed_at": "2026-10-09T08:00:00Z", "previous_version_id": None, "body": "old"}
    later = {"document_id": "two", "version_id": "v2", "canonical_url": "https://example/a",
             "observed_at": "2026-10-09T08:01:00Z", "previous_version_id": None, "body": "new"}
    other = {"document_id": "three", "canonical_url": "https://example/b",
             "observed_at": "2026-10-09T08:02:00Z", "previous_version_id": None}
    unknown = {"document_id": "four", "canonical_url": "https://example/a",
               "observed_at": None, "previous_version_id": None}
    rows = [later, unknown, other, first]
    _link_document_versions(rows)
    assert later["previous_version_id"] == "v1"
    assert all(row["previous_version_id"] is None for row in [first, other, unknown])
    assert [row["document_id"] for row in rows] == ["two", "four", "three", "one"]
    assert first["body"] == "old" and later["body"] == "new"


def test_equal_observation_clocks_do_not_establish_version_precedence():
    rows = [{"document_id": str(i), "canonical_url": "https://example/a",
             "observed_at": "2026-10-09T08:00:00Z", "previous_version_id": None} for i in range(2)]
    _link_document_versions(rows)
    assert all(row["previous_version_id"] is None for row in rows)
