import json
from pathlib import Path

from football_research.data_adapters.football_data_co_uk import build_projection, iter_rows


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "crawl"
    raw = root / "raw" / "england"
    raw.mkdir(parents=True)
    payload = b"Date,Time,Home,Away,HG,AG,Res,AvgH,AvgD,AvgA,B365H,B365D,B365A\n01/01/2025,15:00,A,B,2,1,H,2.0,3.4,4.1,1.9,3.5,4.2\n"
    source = raw / "2425-E0.csv"
    source.write_bytes(payload)
    import hashlib
    digest = hashlib.sha256(payload).hexdigest()
    (root / "source-manifest.json").write_text(json.dumps({"files": [{
        "competition_code": "E0", "season_code": "2425", "path": "var/external/football-data-co-uk-20261024-v2/raw/england/2425-E0.csv",
        "url": "https://football-data.co.uk/mmz4281/2425/E0.csv", "sha256": digest}]}), encoding="utf-8")
    (root / "normalized-matches.jsonl").write_text(json.dumps({
        "competition_code": "E0", "season_code": "2425", "date_local": "2025-01-01", "time_local": "15:00",
        "home_team": "A", "away_team": "B", "home_goals": "2", "away_goals": "1", "result": "H",
        "avg_1x2": {"home": "2.0", "draw": "3.4", "away": "4.1"},
        "b365_1x2": {"home": "1.9", "draw": "3.5", "away": "4.2"},
        "kickoff_time_basis": "source_declared_local_clock_timezone_not_normalized"}, ensure_ascii=False) + "\n", encoding="utf-8")
    return root


def test_iter_rows_preserves_unknown_clock_and_provenance(tmp_path):
    rows = list(iter_rows(_fixture(tmp_path)))
    assert len(rows) == 1
    assert rows[0]["outcome_class"] == 0
    assert rows[0]["kickoff_at_utc"] is None
    assert rows[0]["source_available_at"] is None
    assert rows[0]["strict_prequery_eligible"] is False
    assert rows[0]["source_sha256"]


def test_build_projection_is_read_only_and_explicitly_not_g_training(tmp_path):
    root = _fixture(tmp_path)
    report = build_projection(root)
    assert report["network_requests"] == 0
    assert report["g_training_rows"] == 0
    assert report["strict_prequery_eligible"] == 0
    assert json.loads((root / "projected" / "projection-report.json").read_text(encoding="utf-8"))["labels"] == 1


def test_blank_upstream_row_is_skipped_without_an_identity(tmp_path):
    root = _fixture(tmp_path)
    with (root / "normalized-matches.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"competition_code": "E0", "season_code": "2425", "date_local": None,
                                 "home_team": None, "away_team": None}) + "\n")
    assert len(list(iter_rows(root))) == 1
    assert build_projection(root)["skipped_blank_rows"] == 1
