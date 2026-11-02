import hashlib
import json

from tools.audit_footballmind_information_stream import audit


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _package(tmp_path, *, fixture_id="m1", available=True, tamper=False):
    raw = tmp_path / "raw" / "article.html"
    raw.parent.mkdir(parents=True)
    raw.write_text("article", encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if tamper:
        declared = "0" * 64
    else:
        declared = digest
    kickoff = "2026-10-01T19:00:00+00:00"
    _jsonl(tmp_path / "fixtures.jsonl", [{
        "physical_match_id": fixture_id, "home_team": "A", "away_team": "B", "kickoff": kickoff,
    }])
    doc = {
        "document_id": "d1", "physical_match_id": fixture_id, "url": "https://example.test/a",
        "body": "article", "raw_path": "raw/article.html", "raw_sha256": declared,
        "source_published_at": "2026-10-01T10:00:00+00:00", "captured_at": "2026-10-01T11:00:00+00:00",
        "information_mode": "prospective_capture",
    }
    if available:
        doc["available_at"] = "2026-10-01T11:00:00+00:00"
    _jsonl(tmp_path / "documents.jsonl", [doc])
    _jsonl(tmp_path / "discovery.jsonl", [{
        "physical_match_id": fixture_id, "observed_at": "2026-10-01T11:30:00+00:00",
        "original_agent_menu_proven": True, "menu_status": "prospective",
    }])
    _jsonl(tmp_path / "attempts.jsonl", [{
        "physical_match_id": fixture_id, "requested_at": "2026-10-01T11:31:00+00:00", "status": "returned",
    }])
    _jsonl(tmp_path / "context_snapshots.jsonl", [{
        "physical_match_id": fixture_id, "observed_at": "2026-10-01T11:32:00+00:00",
    }])
    _jsonl(tmp_path / "labels" / "results.jsonl", [{
        "physical_match_id": fixture_id, "result_observed_at": "2026-10-01T22:00:00+00:00",
    }])
    (tmp_path / "manifest.json").write_text(json.dumps({"file_hashes": {"raw/article.html": digest}}), encoding="utf-8")
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "cases.json").write_text(json.dumps([{"fixture": {"physical_match_id": fixture_id}}]), encoding="utf-8")
    fixed = tmp_path / "fixed.json"
    selected = tmp_path / "selected.json"
    fixed.write_text("{}", encoding="utf-8")
    selected.write_text("{}", encoding="utf-8")
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps([{"fixture": {"physical_match_id": "old"}, "documents": []}]), encoding="utf-8")
    return tmp_path, derived, fixed, selected, prior


def test_audit_measures_new_fixture_and_document(tmp_path):
    package, derived, fixed, selected, prior = _package(tmp_path)
    report = audit(package, derived, fixed, selected, prior)
    assert report["corpus_overlap"]["independent_fixture_increase"] == 1
    assert report["corpus_overlap"]["new_document_urls_vs_prior_corpus"] == 1
    assert report["integrity"]["raw_digest_verification"]["status"] == "verified"


def test_audit_rejects_tampered_raw_digest(tmp_path):
    package, derived, fixed, selected, prior = _package(tmp_path, tamper=True)
    report = audit(package, derived, fixed, selected, prior)
    digest = report["integrity"]["raw_digest_verification"]
    assert digest["status"] == "failed"
    assert "raw/article.html" in digest["mismatched"]
    assert report["eligibility"]["online_ope"] == "ineligible"


def test_audit_does_not_infer_missing_availability(tmp_path):
    package, derived, fixed, selected, prior = _package(tmp_path, available=False)
    report = audit(package, derived, fixed, selected, prior)
    assert report["provenance_limits"]["actual_document_visibility_timing"]["unknown"] == 1
    assert report["eligibility"]["online_ope"] == "unknown"
    assert "document_visibility_before_kickoff" in report["eligibility"]["unknown_requirements"]
