import json

from football_research.data_adapters.information_stream_contract import validate_stream


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_valid_stream_preserves_modes_and_cross_references(tmp_path):
    _write(tmp_path / "fixtures.jsonl", [{
        "physical_match_id": "m1", "home_team": "A", "away_team": "B",
        "kickoff": "2026-10-01T19:00:00+00:00",
    }])
    _write(tmp_path / "documents.jsonl", [{
        "document_id": "d1", "physical_match_id": "m1", "url": "https://example.test/a",
        "captured_at": "2026-10-01T12:00:00+00:00",
        "source_published_at": "2026-10-01T10:00:00+00:00",
        "available_at": "2026-10-01T10:00:00+00:00",
        "information_mode": "prospective_capture", "raw_sha256": "a" * 64,
    }])
    _write(tmp_path / "discovery.jsonl", [{
        "discovery_id": "q1", "physical_match_id": "m1",
        "observed_at": "2026-10-01T11:00:00+00:00",
        "visible_candidates": [{"candidate_id": "c1", "url": "https://example.test/a"}],
    }])
    _write(tmp_path / "attempts.jsonl", [{
        "attempt_id": "a1", "physical_match_id": "m1", "discovery_id": "q1",
        "candidate_id": "c1", "requested_at": "2026-10-01T11:01:00+00:00",
        "response_at": "2026-10-01T11:01:02+00:00", "returned_document_id": "d1",
    }])
    _write(tmp_path / "context_snapshots.jsonl", [{
        "snapshot_id": "s1", "physical_match_id": "m1",
        "observed_at": "2026-10-01T11:00:00+00:00",
    }])
    _write(tmp_path / "labels" / "results.jsonl", [{
        "physical_match_id": "m1", "result_status": "final",
        "result_observed_at": "2026-10-01T22:00:00+00:00",
    }])
    report = validate_stream(tmp_path)
    assert report["ready"] is True
    assert report["errors"] == []
    assert report["counts"]["document_modes"] == {"prospective_capture": 1}


def test_validator_rejects_future_capture_and_dangling_reference(tmp_path):
    _write(tmp_path / "fixtures.jsonl", [{
        "physical_match_id": "m1", "home_team": "A", "away_team": "B",
        "kickoff": "2026-10-01T19:00:00+00:00",
    }])
    _write(tmp_path / "documents.jsonl", [{
        "document_id": "d1", "physical_match_id": "unknown", "url": "https://example.test/a",
        "captured_at": "2026-10-01T10:00:00+00:00",
        "source_published_at": "2026-10-01T11:00:00+00:00",
        "information_mode": "prospective_capture",
    }])
    _write(tmp_path / "discovery.jsonl", [{
        "discovery_id": "q1", "physical_match_id": "m1",
        "observed_at": "2026-10-01T11:00:00+00:00",
        "visible_candidates": [{"candidate_id": "c1", "url": "https://example.test/a"}],
    }])
    report = validate_stream(tmp_path)
    assert report["ready"] is False
    assert any("publication follows capture" in error for error in report["errors"])
    assert any("unknown physical_match_id" in error for error in report["errors"])
