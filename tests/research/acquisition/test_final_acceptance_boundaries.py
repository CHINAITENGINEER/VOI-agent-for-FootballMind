"""Regression for distinct acquisition attempts of the same frozen fixture.

These tests use archived source fixtures and a local failure response; they make
zero network requests and never assert that constructed returns prove benefit.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket

import pytest

from football_agent.inquiry_transport import RoutedOpenWebTools
from football_agent.learned_voi_replay import replay_saved_run
from football_research.data_adapters import prospective_text as prospective
from football_research.data_adapters.spftll import load_spftll_archive
from football_spider.storage.raw_archive import RawArchive


@pytest.mark.parametrize("same_collection_clock", [True, False])
def test_repeated_match_collections_keep_distinct_runs_but_one_match(tmp_path, monkeypatch, same_collection_clock):
    actual_now = datetime.now(timezone.utc)
    frozen = actual_now - timedelta(minutes=2)
    kickoff = actual_now + timedelta(hours=3)
    source = RawArchive(tmp_path / "prior-source")
    row = {"matchId": 101, "leagueId": 1864, "leagueChs": "fixture league", "homeId": 11, "awayId": 12,
           "homeChs": "Home", "awayChs": "Away", "footMatchState": 0,
           "matchTime": kickoff.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S"),
           "spfWinFoot": 2., "spfEqualFoot": 3., "spfLoseFoot": 4.}
    source.save("spftll", "https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll", {},
        json.dumps({"code": 0, "data": [{"list": [row]}]}).encode(), status=200,
        observed_at=frozen-timedelta(minutes=1))
    plan = prospective.select_initializations(load_spftll_archive(source.root / "raw", as_of=frozen), as_of=frozen, limit=1)
    plan_path = tmp_path / "plan.json"
    prospective._save(plan_path, plan)

    class CollectionClock:
        tick = actual_now-timedelta(minutes=1)

        @classmethod
        def now(cls, tz):
            return cls.tick

    def no_network(*args, **kwargs):
        raise AssertionError("Acceptance regression attempted network I/O")

    def recorded_failure(self, prepared, *, run_id, request_id):
        # Exercise the real archive, executor, projection, actor and replay.
        # Only the transport's remote response is replaced with a local fixture.
        requested = datetime.now(timezone.utc)
        observed = self.archive.save(prepared["source"], prepared["endpoint"], prepared["params"], b"",
            status=None, error_class="SyntheticTestUnavailable", request_id=request_id,
            observed_at=datetime.now(timezone.utc),
            context={"inquiry_run_id": run_id, "inquiry_request_id": request_id,
                     "requested_at": requested.isoformat(), "tool": prepared["tool"]})
        return {"document": self.project(observed, prepared), "observation": observed.model_dump(mode="json"),
                "request_id": request_id, "http_attempts": 1, "latency_seconds": 0., "money": None,
                "money_status": "not_measured"}

    monkeypatch.setattr(prospective, "datetime", CollectionClock)
    monkeypatch.setattr(RoutedOpenWebTools, "execute", recorded_failure)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    reports, folders, configs = [], [], []
    for index in range(2):
        if index and not same_collection_clock:
            CollectionClock.tick += timedelta(seconds=15)
        output = tmp_path / f"collection-{index}"
        report = prospective.collect_prepared_batch(plan_path, output=output, archive_root=source.root,
            allowed_hosts=["club.example"], seed_urls=["https://club.example/news"], requests_per_match=1, seed=182)
        reports.append(report)
        folders.append(Path(report["runs"][0]["folder"]))
        configs.append(json.loads((output / "batch-configuration.json").read_text(encoding="utf-8")))

    # This assertion exposed the prior bug: model + seed described a policy,
    # not the identity of a distinct collection that happened on another call.
    assert reports[0]["runs"][0]["run_id"] != reports[1]["runs"][0]["run_id"]
    assert configs[0]["collection_id"] != configs[1]["collection_id"]
    if same_collection_clock:
        assert configs[0]["started_at"] == configs[1]["started_at"]
    else:
        assert configs[0]["started_at"] != configs[1]["started_at"]
    assert reports[0]["runs"][0]["initial_probabilities"] == reports[1]["runs"][0]["initial_probabilities"]
    for index, folder in enumerate(folders):
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
        assert config["supervisor_metadata"]["collection_id"] == configs[index]["collection_id"]
        assert run["outcome"] is None
        assert run["initial_probabilities"] == run["final_probabilities"]
        acceptance = replay_saved_run(folder, output=tmp_path / f"replay-{index}",
                                      project_root=Path(__file__).resolve().parents[3])
        assert acceptance["identical_replay"] and acceptance["replay_network_requests"] == 0
        assert (folder / "run.json").read_bytes() == (tmp_path / f"replay-{index}" / "run.json").read_bytes()

    policy = prospective.MarketPriorPolicy.from_manifest(plan["selected"][0])
    joined = prospective.join_settlements(folders, settlements=[],
        prior_bindings={policy.model.model_version: policy.prior_binding()}, as_of=datetime.now(timezone.utc))
    assert joined["dataset"]["audit"]["unique_runs"] == 2
    assert joined["dataset"]["audit"]["unique_matches"] == 1
    assert joined["dataset"]["audit"]["rejected"] == []
    assert joined["training_records"] == []
