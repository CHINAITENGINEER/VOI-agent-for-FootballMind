import hashlib
import json

from football_research.data_adapters.information_stream_panel import build_historical_cases


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_panel_bridge_requires_explicit_result_provenance(tmp_path):
    raw = tmp_path / "raw" / "article.html"
    raw.parent.mkdir()
    raw.write_text("<p>Team news</p>", encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    _write_jsonl(tmp_path / "fixtures.jsonl", [{
        "physical_match_id": "m1", "home_team": "A", "away_team": "B",
        "home_team_id": "a", "away_team_id": "b", "competition_id": "epl",
        "kickoff": "2026-10-01T19:00:00+00:00",
    }])
    _write_jsonl(tmp_path / "documents.jsonl", [{
        "document_id": "d1", "physical_match_id": "m1", "url": "https://example.test/a",
        "title": "Team news", "text": "Team news", "raw_path": "raw/article.html",
        "raw_sha256": digest, "captured_at": "2026-10-01T12:00:00+00:00",
        "source_published_at": "2026-10-01T10:00:00+00:00",
        "information_mode": "archived_snapshot",
    }])
    _write_jsonl(tmp_path / "labels" / "results.jsonl", [{
        "physical_match_id": "m1", "regulation_home_goals": 1,
        "regulation_away_goals": 0, "result_status": "final",
        "result_observed_at": "2026-10-01T22:00:00+00:00",
    }])
    cases, report = build_historical_cases(tmp_path)
    assert cases == []
    assert report["status"] == "not_ready"
    assert any("result source" in item["reason"] for item in report["excluded"])
