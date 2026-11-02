"""Empirical query selection with actual archived synthetic returns, no network."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json

import pytest
import requests

from football_agent import learned_voi_replay
from football_agent.inquiry_tools import OpenWebTools
from football_agent.learned_voi_runtime import LearnedVOIRuntime, _save
from football_agent.learned_voi_web import MatchDocumentCandidates
from football_agent.match_analysis_report import write_analysis_report
from football_agent.text_voi_policy import TextVOIPolicy
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.text_returns import TextReturnModel
from football_research.acquisition.text_risk import TextRiskCalibrator
from football_spider.storage.raw_archive import RawArchive
from test_joint_returns import fitted, state
from test_text_returns import FIRST, SECOND, TEXTS, rows, response


def action(request, aid):
    return {**request, "action_id": aid, "information_key": aid, "group_ids": [],
            "available": True, "cost_status": "known", "cost": {"money": .01, "requests": 1.}}


def setup_policy(base, *, harmful=False, neighborhood=False):
    if neighborhood:
        from football_research.acquisition.text_history import HistoryNeighborhoodTextReturnModel
        from football_research.acquisition.neighborhood_text_risk import NeighborhoodTextRiskCalibrator
    text_class = HistoryNeighborhoodTextReturnModel if neighborhood else TextReturnModel
    risk_class = NeighborhoodTextRiskCalibrator if neighborhood else TextRiskCalibrator
    text = text_class().fit(rows(), as_of="2026-02-01T00:00:00Z",
        dataset_version="synthetic-runtime-f-only", evidence_origin="constructed_mechanism",
        text_clusters=3, action_clusters=2)
    initial = state()
    initial["prediction_cutoff"] = "2026-04-01T01:00:00+00:00"
    initial["base"]["kickoff"] = "2026-04-02T00:00:00+00:00"
    base_policy = LearnedJointVOIPolicy(base, lambda_cost={"money": 1.})
    prior = base_policy._state(initial)[1]["probabilities"]
    data = []
    for i in range(60):
        y = i % 3
        token = (y+1) % 3 if harmful else y
        common = {"match_id": f"risk-{i}", "prior_probabilities": prior, "outcome": y,
                  "prediction_cutoff": "2026-03-01T10:00:00Z",
                  "result_available_at": "2026-03-02T13:00:00Z",
                  "response": response(token)}
        data += [{**common, "action": FIRST, "history": []},
                 {**common, "action": SECOND,
                  "history": [{"action": FIRST, "response": response(token)}]}]
    risk = risk_class().fit(data, text_model=text, as_of="2026-03-03T00:00:00Z",
        dataset_version="deliberately-aligned-or-harmful-synthetic-unit-fixture",
        evidence_origin="constructed_mechanism")
    return TextVOIPolicy(base_policy, text, risk_model=risk), initial


@pytest.mark.parametrize("neighborhood", [False, True])
def test_positive_model_evsi_can_be_rejected_by_negative_empirical_gain(fitted, neighborhood):
    policy, initial = setup_policy(fitted[0], harmful=True, neighborhood=neighborhood)
    actions = [action(FIRST, "first")]
    budget = {"limits": {"money": 1., "requests": 2.}, "spent": {}}
    raw = TextVOIPolicy(policy.base_policy, policy.text_model).plan(initial, actions, budget,
        candidate_generation_complete=True)
    empirical = policy.plan(initial, actions, budget, candidate_generation_complete=True)
    assert raw["selection"]["decision"] == "acquire"
    candidate = empirical["candidates"][0]
    assert candidate["response_enumeration"]["value"]["gross_evsi"] > 0
    assert candidate["empirical_risk_estimate"]["expected_gain"] < 0
    assert candidate["value"]["gross_evsi"] is None
    assert empirical["selection"]["decision"] == "stop"
    assert empirical["selection"]["stop_certificate"] is None
    assert policy.plan(initial, actions, budget)["selection"]["decision"] == "defer"
    assert initial.get("open_text") is None


@pytest.mark.parametrize("neighborhood", [False, True])
def test_empty_menu_cannot_bypass_risk_training_clock_or_training_identity(fitted, neighborhood):
    policy, initial = setup_policy(fitted[0], neighborhood=neighborhood)
    budget = {"limits": {"requests": 1.}, "spent": {}}
    future = deepcopy(initial)
    future["prediction_cutoff"] = "2026-02-28T00:00:00Z"
    with pytest.raises(ValueError):
        policy.plan(future, [], budget)
    same = deepcopy(initial)
    same["match_id"] = "risk-0"
    with pytest.raises(ValueError):
        policy.plan(same, [], budget)
    with pytest.raises(ValueError, match="two-step|separately validated"):
        TextVOIPolicy(policy.base_policy, policy.text_model, discovery_model=object(),
                      risk_model=policy.risk_model)


@pytest.fixture(params=[False, True], ids=["exact", "neighborhood"])
def empirical_archive(tmp_path, fitted, monkeypatch, request):
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **kw: pytest.fail("No network in fixture"))
    base = fitted[0]
    policy, initial = setup_policy(base, neighborhood=request.param)
    monkeypatch.setattr(learned_voi_replay, "load_frozen_models", lambda *a, **kw: (None, base))
    root = tmp_path/"project"
    freeze = root/"var/parallel/agent/round175/frozen-next-period.json"
    _save(freeze, {"scope": "constructed empirical-gain integration fixture"})
    identity = {"match_id": initial["match_id"], "home_team": "Arsenal", "away_team": "Leeds",
                "kickoff_at": initial["base"]["kickoff"], "prediction_cutoff": initial["prediction_cutoff"]}
    generator = MatchDocumentCandidates(match_terms=["Arsenal", "Leeds"], seed_urls=[FIRST["url"]],
        allowed_hosts=["club.example"], numeric_actions=[], web_cost={"money": .01, "requests": 1.})
    folder = tmp_path/"original"
    archive = RawArchive(folder/"source-archive")
    clock = lambda: datetime(2026, 4, 1, tzinfo=timezone.utc)
    tools = OpenWebTools(archive, ["club.example"], clock=clock, extraction_version=3)
    called = []

    def execute(request, public):
        config = json.loads((folder/"configuration.json").read_text())
        assert config["text_risk_model"]["sha256"] == sha256((folder/"text-risk-model.json").read_bytes()).hexdigest()
        called.append(request["url"])
        prepared = tools.prepare(request, {"task": {"seed_urls": generator.seed_urls}, "documents": public["documents"]})
        body = (f'<html><body><article>{TEXTS[0]}</article>'
                f'<a href="{SECOND["url"]}">Arsenal update</a></body></html>').encode()
        observation = archive.save(prepared["source"], prepared["endpoint"], prepared["params"], body,
                                   status=200, observed_at=clock())
        return {"status": "OK", "document": tools.project(observation, prepared),
                "actual_cost": {"money": .01, "requests": 1., "latency_ms": 0., "tokens": 0.},
                "receipt": {"observation": observation.model_dump(mode="json"), "prepared": prepared}}

    metadata = {"generator": generator.configuration(), "transport": tools.configuration(),
                "input_bindings": {"frozen_model_config_sha256": sha256(freeze.read_bytes()).hexdigest()}}
    runtime = LearnedVOIRuntime(policy, seed=183, exploration_limits={"money": 0., "requests": 0.},
        max_exploration_actions=0)
    run = runtime.run(initial, run_id="constructed-empirical-text", limits={"money": .04, "requests": 4.},
        propose=generator, execute=execute, output=folder, clock=clock, evidence_match=identity,
        provider_version="synthetic-archived-html", generator_version=generator.version, run_metadata=metadata)
    write_analysis_report(run, match=identity, output=folder)
    return folder, root, run, called


def test_actual_return_replans_and_replays_empirical_decisions(empirical_archive, tmp_path):
    folder, root, run, called = empirical_archive
    assert called == [FIRST["url"]]
    assert run["cycles"][0]["selection_basis"] == "computed_positive_empirical_gain"
    assert run["text_response_learning"]["probability_updates"] == 1
    assert run["text_response_learning"]["query_and_update_share_response_model"] is False
    assert run["terminal_selection"]["stop_certificate"] is None
    assert any(c.get("url") == SECOND["url"] for c in run["cycles"][1]["generated_actions"])
    assert all(c.get("url") != SECOND["url"] for c in run["cycles"][0]["generated_actions"])
    acceptance = learned_voi_replay.replay_saved_run(folder, output=tmp_path/"replay", project_root=root)
    assert acceptance["identical_replay"] is True
    assert acceptance["replay_network_requests"] == 0
    for name in ("run.json", "text-risk-model.json", "analysis.json", "report.md"):
        assert (folder/name).read_bytes() == (tmp_path/"replay"/name).read_bytes()
    report = (folder/"report.md").read_text(encoding="utf-8")
    assert "经验风险改善" in report and "合成机制实验" in report
    assert "查询前的信息价值与取得正文后的概率更新使用同一条件响应模型" not in report


def test_changed_empirical_artifact_is_rejected_before_replay(empirical_archive, tmp_path):
    folder, root, _, _ = empirical_archive
    artifact = folder/"text-risk-model.json"
    artifact.write_bytes(artifact.read_bytes()+b" ")
    with pytest.raises(ValueError, match="empirical risk model differs"):
        learned_voi_replay.replay_saved_run(folder, output=tmp_path/"reject", project_root=root)


@pytest.mark.parametrize("neighborhood", [False, True])
def test_future_empirical_fit_rejected_before_directory_or_execution(fitted, tmp_path, neighborhood):
    policy, initial = setup_policy(fitted[0], neighborhood=neighborhood)
    runtime = LearnedVOIRuntime(policy, seed=183, exploration_limits={"money": 0., "requests": 0.}, max_exploration_actions=0)
    def forbidden(*args):
        pytest.fail("Future calibration cannot reach tools")
    output = tmp_path/"future"
    with pytest.raises(ValueError, match="empirical text risk model"):
        runtime.run(initial, run_id="future-calibration", limits={"money": .01, "requests": 1.},
            propose=forbidden, execute=forbidden, output=output,
            provider_version="synthetic", generator_version="synthetic",
            clock=lambda: datetime(2026, 3, 2, tzinfo=timezone.utc))
    assert not output.exists()
