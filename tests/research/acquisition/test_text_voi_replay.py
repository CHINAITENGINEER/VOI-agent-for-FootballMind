"""Archive-to-replay integration of the text model, using synthetic HTTP bodies."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest
import requests

from football_agent.inquiry_tools import OpenWebTools
from football_agent.learned_voi_runtime import LearnedVOIRuntime, _save
from football_agent.learned_voi_web import MatchDocumentCandidates
from football_agent.text_voi_policy import TextVOIPolicy
from football_agent import learned_voi_replay
from football_agent.match_analysis_report import write_analysis_report
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.text_returns import TextReturnModel
from football_spider.storage.raw_archive import RawArchive
from test_joint_returns import fitted, state


ARTIFACT = Path(__file__).resolve().parents[3]/"var/parallel/agent/round178/evaluation/frozen-text-model.json"
URLS = ["https://context.synthetic.example/news/match-preview",
        "https://medical.synthetic.example/news/injury-bulletin",
        "https://syndication.synthetic.example/news/injury-roundup",
        "https://tactics.synthetic.example/news/formation-confirmation"]
BODIES = [
    "Synthetic context dispatch. Qualification remains undecided and the competitive group is open.",
    "Synthetic injury bulletin. The striker is absent and ruled out of participation.",
    "Synthetic injury bulletin. The striker is absent and ruled out of participation.",
    "Synthetic tactical confirmation. The team will defend with a low block and deep positions.",
]


@pytest.fixture
def archived_text(tmp_path, fitted, monkeypatch):
    def network_forbidden(*args, **kwargs):
        pytest.fail("A synthetic archive integration must not make HTTP requests")
    monkeypatch.setattr(requests.sessions.Session, "request", network_forbidden)
    base = fitted[0]
    monkeypatch.setattr(learned_voi_replay, "load_frozen_models", lambda *args, **kwargs: (None, base))
    root = tmp_path/"project"
    freeze = root/"var/parallel/agent/round175/frozen-next-period.json"
    _save(freeze, {"scope": "synthetic initial-model fixture"})
    text = TextReturnModel.load(ARTIFACT)
    policy = TextVOIPolicy(LearnedJointVOIPolicy(base, lambda_cost={"money": 1.}), text)
    initial = state()
    initial["prediction_cutoff"] = "2025-07-01T01:00:00+00:00"
    initial["base"]["kickoff"] = "2025-07-02T00:00:00+00:00"
    identity = {"match_id": initial["match_id"], "home_team": "Arsenal", "away_team": "Leeds",
                "kickoff_at": initial["base"]["kickoff"], "prediction_cutoff": initial["prediction_cutoff"]}
    hosts = [url.split("/")[2] for url in URLS]
    generator = MatchDocumentCandidates(match_terms=["Arsenal", "Leeds"], seed_urls=[URLS[0]],
        allowed_hosts=hosts, numeric_actions=[], web_cost={"money": 0., "requests": 1.})
    folder = tmp_path/"original"
    archive = RawArchive(folder/"source-archive")
    clock = lambda: datetime(2025, 7, 1, tzinfo=timezone.utc)
    tools = OpenWebTools(archive, hosts, clock=clock, extraction_version=3)

    def execute(action, public):
        # Full model and config must already be frozen before a body is read.
        config = json.loads((folder/"configuration.json").read_text())
        assert config["text_response_model"]["sha256"] == sha256((folder/"text-response-model.json").read_bytes()).hexdigest()
        prepared = tools.prepare(action, {"task": {"seed_urls": generator.seed_urls}, "documents": public["documents"]})
        assert prepared["tool"] == "read_url"
        index = URLS.index(prepared["endpoint"])
        links = "".join(f'<a href="{url}">Arsenal update</a>' for url in URLS[1:]) if index == 0 else ""
        body = f"<html><body><p>{BODIES[index]}</p>{links}</body></html>".encode()
        observation = archive.save(prepared["source"], prepared["endpoint"], prepared["params"], body,
                                   status=200, observed_at=clock())
        return {"status": "OK", "document": tools.project(observation, prepared),
                "actual_cost": {"money": 0., "requests": 1., "latency_ms": 0., "tokens": 0.},
                "receipt": {"observation": observation.model_dump(mode="json"), "prepared": prepared}}

    metadata = {"generator": generator.configuration(), "transport": tools.configuration(),
                "input_bindings": {"frozen_model_config_sha256": sha256(freeze.read_bytes()).hexdigest()}}
    runtime = LearnedVOIRuntime(policy, seed=178, exploration_limits={"money": 0., "requests": 0.},
                                max_exploration_actions=0)
    run = runtime.run(initial, run_id="synthetic-archived-text", limits={"money": 0., "requests": 4.},
        propose=generator, execute=execute, output=folder, clock=clock, evidence_match=identity,
        provider_version="synthetic-archived-html", generator_version=generator.version, run_metadata=metadata)
    write_analysis_report(run, match=identity, output=folder)
    return folder, root, run


def test_actual_archive_replay_reconstructs_text_model_decisions_and_report(archived_text, tmp_path):
    folder, root, run = archived_text
    assert len(run["attempts"]) >= 2
    assert run["text_response_learning"]["probability_updates"] >= 2
    report = learned_voi_replay.replay_saved_run(folder, output=tmp_path/"replay", project_root=root)
    assert report["identical_replay"] is True
    assert report["raw_responses_reprojected"] == len(run["attempts"])
    assert report["replay_network_requests"] == 0
    for filename in ("run.json", "text-response-model.json", "analysis.json", "report.md"):
        assert (folder/filename).read_bytes() == (tmp_path/"replay"/filename).read_bytes()
    text = (folder/"report.md").read_text(encoding="utf-8")
    assert "合成机制实验" in text
    assert "同一条件响应模型" in text


def test_changed_frozen_text_model_is_rejected_before_replay(archived_text, tmp_path):
    folder, root, _ = archived_text
    model = folder/"text-response-model.json"
    model.write_bytes(model.read_bytes()+b" ")
    with pytest.raises(ValueError, match="pre-query binding"):
        learned_voi_replay.replay_saved_run(folder, output=tmp_path/"reject", project_root=root)


def test_initial_model_cannot_use_training_after_actual_acquisition_clock(fitted, tmp_path, monkeypatch):
    base = fitted[0]
    monkeypatch.setitem(base.metadata, "training_as_of", "2025-07-01T12:00:00+00:00")
    policy = TextVOIPolicy(LearnedJointVOIPolicy(base, lambda_cost={"money": 1.}),
                          TextReturnModel.load(ARTIFACT))
    initial = state()
    initial["prediction_cutoff"] = "2025-07-02T00:00:00+00:00"
    initial["base"]["kickoff"] = "2025-07-03T00:00:00+00:00"
    def forbidden(*args):
        pytest.fail("A future-trained initial forecast must not reach generation or execution")
    runtime = LearnedVOIRuntime(policy, seed=178, exploration_limits={"money": 0., "requests": 0.},
                                max_exploration_actions=0)
    with pytest.raises(ValueError, match="initial fixture model"):
        runtime.run(initial, run_id="future-p0", limits={"money": 0., "requests": 1.},
            propose=forbidden, execute=forbidden, output=tmp_path/"future", provider_version="synthetic",
            generator_version="synthetic", clock=lambda: datetime(2025, 7, 1, tzinfo=timezone.utc))
    assert not (tmp_path/"future").exists()
