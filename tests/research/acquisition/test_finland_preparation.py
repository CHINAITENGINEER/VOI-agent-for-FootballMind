"""New source preparation through archived official format; no network."""
from datetime import datetime, timezone
import json
from pathlib import Path

from football_research.data_adapters.live_fixture_preparation import prepare_live_fixture
from football_research.data_adapters.verified_fixtures import verify_identity_binding

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "var/parallel/agent/round194/discovery"
NOW = datetime(2026, 10, 7, 6, tzinfo=timezone.utc)


def test_new_match_preparation_binds_observed_calendar_and_reuses_all_requests(tmp_path):
    calls = []
    def transport(spec, *, archive, clock):
        calls.append(spec["stage"])
        if spec["stage"] == "fixture":
            body = json.dumps({"code": 0, "data": {"matchId": 3085210,
                "matchTime": "2026-10-08T00:00:00", "homeId": 2429, "home": "赫尔火花",
                "awayId": 386, "away": "国际图尔", "leagueId": 13, "league": "芬超",
                "footMatchState": 0, "homeScore": None, "awayScore": None}}).encode()
        elif spec["stage"] == "quote":
            body = json.dumps({"code": 0, "data": {"footballMoreSpInfo": {
                "matchId": 3085210, "spf_win": 3.5, "spf_draw": 3.4, "spf_lost": 1.8}}}).encode()
        else:
            file = {"finland_schedule": "gnistan_fixtures.response.bin",
                    "finland_calendar": "gnistan_calendar.response.bin"}[spec["stage"]]
            body = (RAW / file).read_bytes()
        return archive.save(spec["source"], spec["url"], spec["params"], body, status=200,
            observed_at=clock(), context={"requested_at": spec["requested_at"],
                "preparation_request_digest": spec["request_digest"]})
    output = tmp_path / "prepared"
    result = prepare_live_fixture("3085210", output=output, clock=lambda: NOW, transport=transport)
    assert result["status"] == "ready"
    assert calls == ["fixture", "quote", "finland_schedule", "finland_calendar"]
    binding = json.loads((output / "identity-bundle.json").read_text(encoding="utf-8"))["bindings"]["spftll:match:3085210"]
    assert binding["canonical"]["source_event_id"].endswith(":4353542")
    assert binding["canonical"]["home_name"] == "IF Gnistan"
    assert binding["regulation_rule"] is None and binding["settlement_rows"] == []
    assert "https://gnistan.fi/" in result["collection"]["seed_urls"]
    assert result["collection"]["candidate_discovery_version"] == 8
    assert verify_identity_binding(binding, archive_root=output / "identity-archive", as_of=NOW) == binding
    repeated = prepare_live_fixture("3085210", output=output, resume=True, clock=lambda: NOW,
        transport=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("No repeated HTTP")))
    assert repeated == result
