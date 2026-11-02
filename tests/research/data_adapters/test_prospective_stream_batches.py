from __future__ import annotations

from hashlib import sha256
import json

import pytest

from football_research.data_adapters.prospective_stream import _request_provenance, export_batch


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source_run(folder, match_id):
    config = {
        "created_at": "2026-10-09T10:00:00+00:00",
        "information_mode": "historical_reconstruction",
        "initial_state": {"match_id": match_id, "base": {}},
        "evidence_projection": {"match": {
            "match_id": match_id, "home_team": "Home", "away_team": "Away",
            "competition": "Test", "season": "unknown", "kickoff_at": "2026-10-10T10:00:00+00:00",
            "fixture_available_at": "2026-10-09T09:00:00+00:00",
            "prediction_cutoff": "2026-10-10T09:59:00+00:00",
        }},
    }
    action = {"action_id": "read", "tool": "read_url", "url": "https://source.example/news"}
    _save(folder / "configuration.json", config)
    _save(folder / "run.json", {"cycles": [{
        "decision_at": "2026-10-09T10:01:00+00:00", "generated_actions": [action],
        "selected_action": action, "public_before": {}, "public_after": {},
        "execution_status": "returned", "actual_cost": {"requests": 1},
    }], "terminal_selection": {"reason": "request_limit_reached"}})
    journal = {"run_id": "journal-" + match_id, "request_id": "runtime-step-0",
        "root": "acquisition-journal", "path_scope": "batch_relative",
        "receipt_status": "returned", "receipt_at": "2026-10-09T10:01:02+00:00"}
    _save(folder / "receipt-000.json", {"status": "RETURNED", "receipt": {
        "acquisition_journal": journal,
        "observation": {"request_id": "transport-1", "observed_at": "2026-10-09T10:01:02+00:00",
            "context": {"requested_at": "2026-10-09T10:01:01+00:00",
                "inquiry_request_id": "inquiry-1", "journal_request_id": "runtime-step-0",
                "journal_run_id": "journal-" + match_id}},
    }})
    return journal


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_export_combines_flat_and_direct_child_batches_without_changing_sources(tmp_path):
    batch = tmp_path / "root"
    _source_run(batch / "match-flat", "flat")
    journal = _source_run(batch / "batch-001" / "match-one", "one")
    _source_run(batch / "batch-002" / "match-two", "two")
    # The bounded layout intentionally ignores unrelated/deeper directories.
    _source_run(batch / "other" / "match-ignored", "ignored")
    _source_run(batch / "batch-003" / "batch-deep" / "match-deep", "deep")
    before = {path.relative_to(batch).as_posix(): sha256(path.read_bytes()).hexdigest()
              for path in batch.rglob("*") if path.is_file()}
    output = tmp_path / "export"
    report = export_batch(batch, output)
    assert report["fixtures_registered"] == 3
    fixtures = {row["physical_match_id"]: row for row in _rows(output / "fixtures.jsonl")}
    assert set(fixtures) == {"flat", "one", "two"}
    assert fixtures["one"]["source_run_path"] == "batch-001/match-one"
    assert fixtures["flat"]["source_run_path"] == "match-flat"
    runs = {row["physical_match_id"]: row for row in report["runs"]}
    assert runs["one"]["source_configuration_sha256"] == before["batch-001/match-one/configuration.json"]
    assert runs["one"]["source_run_sha256"] == before["batch-001/match-one/run.json"]
    attempt = next(row for row in _rows(output / "attempts.jsonl") if row["physical_match_id"] == "one")
    assert attempt["acquisition_journal"] == journal
    assert attempt["source_journal_root_path"] == "batch-001/acquisition-journal"
    assert attempt["source_receipt_path"] == "batch-001/match-one/receipt-000.json"
    assert attempt["source_receipt_sha256"] == before[attempt["source_receipt_path"]]
    assert attempt["request_id"] == "inquiry-1"
    assert attempt["journal_request_id"] == "runtime-step-0"
    assert attempt["transport_request_id"] == "transport-1"
    assert before == {path.relative_to(batch).as_posix(): sha256(path.read_bytes()).hexdigest()
                      for path in batch.rglob("*") if path.is_file()}


def test_duplicate_physical_match_is_rejected_before_output_creation(tmp_path):
    batch = tmp_path / "root"
    _source_run(batch / "batch-001" / "match-one", "same")
    _source_run(batch / "batch-002" / "match-two", "same")
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="Duplicate physical_match_id 'same'"):
        export_batch(batch, output)
    assert not output.exists()


@pytest.mark.parametrize("context,expected,basis", [
    ({"inquiry_request_id": "inquiry", "journal_request_id": "journal"}, "inquiry", "observation.context.inquiry_request_id"),
    ({"journal_request_id": "journal"}, "journal", "acquisition_journal_request_id"),
    ({}, "transport", "observation.request_id"),
])
def test_request_id_prefers_runtime_identity_and_retains_transport_id(context, expected, basis):
    result = _request_provenance({}, {"context": context, "request_id": "transport"})
    assert result["request_id"] == expected
    assert result["request_id_basis"] == basis
    assert result["transport_request_id"] == "transport"


def test_receipt_journal_is_preserved_without_raw_context():
    journal = {"run_id": "run", "request_id": "journal", "root": "acquisition-journal"}
    result = _request_provenance({"receipt": {"acquisition_journal": journal}}, {"request_id": None})
    assert result["request_id"] == "journal"
    assert result["acquisition_journal"] == journal
