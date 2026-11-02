"""Learned discovery must unlock actual links, not execute model hypotheses."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json

import pytest
import requests

from football_agent import learned_voi_replay
from football_agent.inquiry_tools import OpenWebTools
from football_agent.learned_voi_runtime import LearnedVOIRuntime, _save
from football_agent.learned_voi_web import MatchDocumentCandidates, _web_action
from football_agent.match_analysis_report import write_analysis_report
from football_agent.text_voi_policy import TextVOIPolicy
from football_research.acquisition.discovery_returns import DiscoveryReturnModel
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.text_returns import TextReturnModel
from football_spider.storage.raw_archive import RawArchive
from test_joint_returns import fitted, state


HOST = "club.synthetic.example"
INDEX = f"https://{HOST}/news"
LIVE_ARTICLE = f"https://{HOST}/news/match-report-heldout"
COST = {"money": .02, "requests": 1.}
TEXTS = ["Albatross crimson forward thriving fit ready selection.",
         "Birch silver balance midfield stable level formation.",
         "Cedar violet backwards depleted absent unavailable roster."]


@pytest.fixture
def discovery_models(fitted):
    base = LearnedJointVOIPolicy(fitted[0], lambda_cost={"money": 1.})
    initial = state()
    initial["prediction_cutoff"] = "2025-07-01T01:00:00+00:00"
    initial["base"]["kickoff"] = "2025-07-02T00:00:00+00:00"
    prior = base._state(initial)[1]["probabilities"]
    records = []
    for i in range(60):
        index = _web_action("read_url", INDEX, cost=COST)
        article = _web_action("read_url", f"https://{HOST}/news/match-report-{i}", cost=COST)
        common = {"match_id": f"synthetic-discovery-train-{i}", "outcome": i % 3,
                  "prior_probabilities": prior, "prediction_cutoff": "2025-04-20T00:00:00Z",
                  "result_available_at": "2025-04-22T00:00:00Z"}
        response = {"status": "OK", "kind": "index", "text": "Arsenal news"}
        records.append({**common, "action": index, "response": response, "history": [],
                        "discovered_actions": [article], "menu_complete": True})
        records.append({**common, "action": article,
                        "response": {"status": "OK", "kind": "article", "text": TEXTS[i % 3]},
                        "history": [{"action": index, "response": response}],
                        "discovered_actions": [], "menu_complete": True})
    text = TextReturnModel().fit(records, as_of="2025-06-01T00:00:00Z",
        dataset_version="runtime-discovery-synthetic-only", evidence_origin="constructed_mechanism",
        text_clusters=3, action_clusters=1)
    discovery = DiscoveryReturnModel(text).fit(records, as_of="2025-06-01T00:00:00Z",
        dataset_version="runtime-discovery-synthetic-only", evidence_origin="constructed_mechanism")
    return base, text, discovery, initial


def _run(tmp_path, models, monkeypatch, *, discovery=True, links=True, requests_budget=2.):
    base, text, transition, initial = models
    monkeypatch.setattr(requests.sessions.Session, "request",
                        lambda *a, **kw: pytest.fail("Synthetic runtime must not perform network I/O"))
    monkeypatch.setattr(learned_voi_replay, "load_frozen_models", lambda *a, **kw: (None, base.model))
    root = tmp_path/"project"
    freeze = root/"var/parallel/agent/round175/frozen-next-period.json"
    _save(freeze, {"scope": "synthetic runtime integration only"})
    policy = TextVOIPolicy(base, text, discovery_model=transition if discovery else None)
    generator = MatchDocumentCandidates(match_terms=["Arsenal", "Leeds"], seed_urls=[INDEX],
        allowed_hosts=[HOST], numeric_actions=[], web_cost=COST)
    folder = tmp_path/"original"
    archive = RawArchive(folder/"source-archive")
    clock = lambda: datetime(2025, 7, 1, tzinfo=timezone.utc)
    tools = OpenWebTools(archive, [HOST], extraction_version=3, clock=clock)
    identity = {"match_id": initial["match_id"], "home_team": "Arsenal", "away_team": "Leeds",
                "kickoff_at": initial["base"]["kickoff"], "prediction_cutoff": initial["prediction_cutoff"]}
    calls, proposals = [], []

    def propose(public):
        actions = generator(public)
        proposals.append(deepcopy(actions))
        return actions

    def execute(action, public):
        config = json.loads((folder/"configuration.json").read_text(encoding="utf-8"))
        for key in ("text_response_model", "discovery_response_model") if discovery else ("text_response_model",):
            binding = config[key]
            assert sha256((folder/binding["file"]).read_bytes()).hexdigest() == binding["sha256"]
        prepared = tools.prepare(action, {"task": {"seed_urls": generator.seed_urls}, "documents": public["documents"]})
        url = prepared["endpoint"]
        assert url in {INDEX, LIVE_ARTICLE}, "A training-only URL must never be executed"
        calls.append(url)
        if url == INDEX:
            link = f'<a href="{LIVE_ARTICLE}">Arsenal match report</a>' if links else ""
            body = f"<html><body><p>Arsenal news</p>{link}</body></html>".encode()
        else:
            assert calls[:-1] == [INDEX]
            body = f"<html><body><p>{TEXTS[0]}</p></body></html>".encode()
        obs = archive.save(prepared["source"], url, prepared["params"], body, status=200, observed_at=clock())
        return {"status": "OK", "document": tools.project(obs, prepared),
                "actual_cost": {"money": 0., "requests": 1., "latency_ms": 0., "tokens": 0.},
                "receipt": {"observation": obs.model_dump(mode="json"), "prepared": prepared}}

    metadata = {"generator": generator.configuration(), "transport": tools.configuration(),
                "input_bindings": {"frozen_model_config_sha256": sha256(freeze.read_bytes()).hexdigest()}}
    runtime = LearnedVOIRuntime(policy, seed=179, exploration_limits={"money": 0., "requests": 0.},
                                max_exploration_actions=0)
    run = runtime.run(initial, run_id="synthetic-discovery-runtime", limits={"money": .04, "requests": requests_budget},
        propose=propose, execute=execute, output=folder, clock=clock, evidence_match=identity,
        provider_version="synthetic-archive-discovery", generator_version=generator.version, run_metadata=metadata)
    write_analysis_report(run, match=identity, output=folder)
    return folder, root, run, calls, proposals


def test_two_step_value_selects_neutral_index_then_only_the_actually_discovered_url(discovery_models, tmp_path, monkeypatch):
    folder, root, run, calls, menus = _run(tmp_path, discovery_models, monkeypatch)
    assert calls == [INDEX, LIVE_ARTICLE]
    assert LIVE_ARTICLE not in {a.get("url") for a in menus[0]}
    assert LIVE_ARTICLE in {a.get("url") for a in menus[1]}
    first, second = run["cycles"][:2]
    chosen = next(c for c in first["strict_decision"]["candidates"] if c["action_id"] == first["selected_action"]["action_id"])
    assert chosen["response_enumeration"]["value"]["gross_evsi"] == pytest.approx(0.)
    assert chosen["response_enumeration"]["value"]["net_voi"] < 0
    assert chosen["value"]["net_voi"] > 0
    assert first["prediction_after"] == pytest.approx(first["prediction_before"])
    assert second["prediction_after"] != pytest.approx(second["prediction_before"])
    assert run["reserved_cost"]["money"] == pytest.approx(.04)
    assert run["exploration_actions"] == 0
    assert run["discovery_response_learning"]["continuation_enabled_selections"] == 1
    replayed = learned_voi_replay.replay_saved_run(folder, output=tmp_path/"replay", project_root=root)
    assert replayed["identical_replay"] and replayed["raw_responses_reprojected"] == 2
    for filename in ("run.json", "text-response-model.json", "discovery-response-model.json", "analysis.json", "report.md"):
        assert (folder/filename).read_bytes() == (tmp_path/"replay"/filename).read_bytes()
    assert "两步信息价值模型" in (folder/"report.md").read_text(encoding="utf-8")


def test_one_step_and_one_request_budget_cannot_buy_the_discovery_chain(discovery_models, tmp_path, monkeypatch):
    _, _, one, calls, _ = _run(tmp_path/"one", discovery_models, monkeypatch, discovery=False)
    assert calls == [] and one["exploration_actions"] == 0
    _, _, short, calls, _ = _run(tmp_path/"short", discovery_models, monkeypatch, requests_budget=1.)
    assert calls == [] and short["exploration_actions"] == 0


def test_model_expected_link_is_not_executed_if_the_actual_page_has_none(discovery_models, tmp_path, monkeypatch):
    _, _, run, calls, menus = _run(tmp_path, discovery_models, monkeypatch, links=False)
    assert calls == [INDEX]
    assert all(LIVE_ARTICLE not in {a.get("url") for a in menu} for menu in menus)
    assert run["final_probabilities"] == pytest.approx(run["initial_probabilities"])
    assert run["reserved_cost"]["money"] == pytest.approx(.02)
    assert run["terminal_selection"]["stop_certificate"] is None


def test_changed_discovery_model_is_rejected_before_raw_archive_replay(discovery_models, tmp_path, monkeypatch):
    folder, root, _, _, _ = _run(tmp_path, discovery_models, monkeypatch)
    path = folder/"discovery-response-model.json"
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="discovery model differs from the pre-query binding"):
        learned_voi_replay.replay_saved_run(folder, output=tmp_path/"reject", project_root=root)

