"""End-to-end source identity -> acquisition -> raw replay, with no network."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from html import escape
import json

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.learned_voi_replay import replay_saved_run
from football_research.data_adapters.prospective_text import (
    _save, collect_prepared_batch, select_initializations, join_settlements, verify_run_identity)
from football_research.data_adapters.spftll import load_spftll_archive
from football_research.data_adapters.verified_fixtures import verify_fixture_identity
from football_spider.storage.raw_archive import RawArchive


@pytest.fixture
def verified_inputs(tmp_path):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    kickoff = now + timedelta(hours=3)
    prior = RawArchive(tmp_path / "market")
    source = {"matchId": 101, "leagueId": 1864, "leagueChs": "欧国联",
        "homeId": 11, "awayId": 12, "homeChs": "克罗地亚", "awayChs": "西班牙",
        "footMatchState": 0,
        "matchTime": kickoff.astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None).isoformat(),
        "spfWinFoot": 2., "spfEqualFoot": 3., "spfLoseFoot": 4.}
    prior.save("spftll", "https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll", {},
        json.dumps({"code": 0, "data": [{"list": [source]}]}).encode(),
        status=200, observed_at=now-timedelta(seconds=20))
    plan = select_initializations(load_spftll_archive(prior.root / "raw", as_of=now),
                                  as_of=now-timedelta(seconds=10), limit=1)
    url = "https://www.uefa.com/uefanationsleague/match/2046001--croatia-vs-spain/"
    event = {"@type": "SportsEvent", "name": "Croatia vs Spain", "url": url,
        "description": "UEFA Nations League 2027 League phase: Croatia vs Spain",
        "startDate": kickoff.isoformat(), "eventStatus": "EventScheduled"}
    embedded = {"id": "2046001", "seasonYear": 2027, "status": "Upcoming",
        "competition": {"age": "Adult", "sex": "Male", "sportsType": "Football",
            "teamCategory": "National", "displayName": "UEFA Nations League", "id": "2014"},
        "round": {"displayName": "League phase", "mode": "Group"},
        "kickOffTime": {"dateTime": kickoff.isoformat()}}
    for side, ident, name in (("home", "101", "Croatia"), ("away", "102", "Spain")):
        event[side+"Team"] = {"@type": "SportsTeam", "name": name,
            "sameAs": f"https://www.uefa.com/uefanationsleague/teams/{ident}--{name.lower()}/"}
        embedded[side+"Team"] = {"id": ident, "internationalName": name}
    article = {"@type": "LiveBlogPosting", "mainEntity": event,
        "liveBlogUpdate": [{"@type": "BlogPosting", "datePublished": (now-timedelta(minutes=3)).isoformat(),
            "articleBody": "Croatia may rotate their starting lineup against Spain. "
                "Spain will assess player fitness before the match."}]}
    html = ('<html><script type="application/ld+json">'+json.dumps(article)+
            '</script><div data-options="'+escape(json.dumps(embedded), quote=True)+'"></div></html>').encode()
    archive = RawArchive(tmp_path/"identity")
    obs = archive.save("open_page", url, {}, html, status=200, observed_at=now-timedelta(seconds=5))
    binding = verify_fixture_identity(plan["selected"][0], archive_root=archive.root,
        observation_ids=[obs.id], aliases={"home": ["克罗地亚", "Croatia"], "away": ["西班牙", "Spain"]}, as_of=now)
    assert binding["status"] == "CONFIRMED"
    _save(tmp_path/"plan.json", plan)
    _save(tmp_path/"bindings.json", {"bindings": {binding["match_id"]: binding}})
    return {"root": tmp_path, "plan": plan, "binding": binding, "body": html, "now": now}


def _collect(inputs, monkeypatch, **options):
    calls = []
    def archived_return(self, prepared, *, run_id, request_id):
        calls.append(deepcopy(prepared))
        now = datetime.now(timezone.utc)
        body = b"<rss><channel></channel></rss>" if prepared["tool"] == "search_web" else inputs["body"]
        obs = self.archive.save(prepared["source"], prepared["endpoint"], prepared["params"], body,
            status=200, observed_at=now, context={"requested_at": now.isoformat()})
        return {"document": self.project(obs, prepared), "observation": obs.model_dump(mode="json"),
            "http_attempts": 1, "money": None, "money_status": "not_measured", "latency_seconds": .01}
    monkeypatch.setattr(NewsDiscoveryTools, "execute", archived_return)
    root = inputs["root"]
    report = collect_prepared_batch(root/"plan.json", output=root/"batch", archive_root=root/"market",
        allowed_hosts=["www.uefa.com"], identity_bundle=root/"bindings.json", identity_archive=root/"identity",
        news_provider="bing_rss", requests_per_match=2, seed=180, **options)
    return root/"batch/match-101", calls, report


@pytest.mark.parametrize('profile_version', [1, 2])
def test_verified_collection_uses_source_names_and_replays_actual_body(verified_inputs, monkeypatch, profile_version):
    folder, calls, report = _collect(verified_inputs, monkeypatch, source_profile_version=profile_version)
    config = json.loads((folder/"configuration.json").read_text(encoding="utf-8"))
    run = json.loads((folder/"run.json").read_text(encoding="utf-8"))
    assert config["supervisor_metadata"]["generator"]["match_terms"] == ["Croatia", "Spain"]
    assert config['supervisor_metadata']['generator']['version'] == f'evidence-document-candidates-v{profile_version + 1}'
    batch_config = json.loads((folder.parent/'batch-configuration.json').read_text(encoding='utf-8'))
    assert ('source_profile_version' in batch_config) == (profile_version == 2)
    assert config["evidence_projection"]["match"]["home_aliases"] == ["克罗地亚", "Croatia"]
    assert any("Croatia Spain" in action.get("query", "") for action in run["cycles"][0]["generated_actions"])
    assert any(doc.get("structured_news", {}).get("role") == "liveblog" for doc in run["documents"])
    assert len(calls) == report["http_requests"] == 2
    assert run["initial_probabilities"] == run["final_probabilities"]
    replay = replay_saved_run(folder, output=verified_inputs["root"]/"replay", project_root=verified_inputs["root"])
    assert replay["identical_replay"] and replay["replay_network_requests"] == 0
    proofs = json.loads((folder/"prior-bindings.json").read_text(encoding="utf-8"))
    audit = join_settlements([folder], settlements=[], prior_bindings=proofs,
                            as_of=datetime.now(timezone.utc))
    assert audit["training_records"] == []
    assert audit["independent_result_verification"][0]["status"] == "PENDING"
    with pytest.raises(ValueError, match="original result bytes"):
        join_settlements([folder], settlements=[{"match_id": verified_inputs["binding"]["match_id"],
            "home_goals": 99}], prior_bindings=proofs, as_of=datetime.now(timezone.utc))


def test_disputed_aliases_block_collection_before_any_tool_call(verified_inputs, monkeypatch):
    root = verified_inputs["root"]
    bundle = json.loads((root/"bindings.json").read_text(encoding="utf-8"))
    binding = next(iter(bundle["bindings"].values()))
    binding["aliases"]["home"] = ["克罗地亚", "England"]
    _save(root/"bindings.json", bundle)
    monkeypatch.setattr(NewsDiscoveryTools, "execute", lambda *a, **k: pytest.fail("Identity was not verified"))
    with pytest.raises(ValueError, match="changed"):
        _collect(verified_inputs, monkeypatch)
    assert not (root/"batch").exists()


def test_independent_identity_raw_tamper_breaks_replay(verified_inputs, monkeypatch):
    import gzip
    folder, _, _ = _collect(verified_inputs, monkeypatch)
    payload = next((folder/"identity-archive/raw").rglob("*.body.gz"))
    payload.write_bytes(gzip.compress(b"<html>changed official identity</html>"))
    with pytest.raises(ValueError):
        replay_saved_run(folder, output=verified_inputs["root"]/"tampered-replay", project_root=verified_inputs["root"])


def test_candidate_aliases_cannot_escape_verified_identity(verified_inputs, monkeypatch):
    folder, _, _ = _collect(verified_inputs, monkeypatch)
    config = json.loads((folder/"configuration.json").read_text(encoding="utf-8"))
    config["evidence_projection"]["match"]["home_aliases"].append("England")
    with pytest.raises(ValueError, match="aliases"):
        verify_run_identity(folder, config)
