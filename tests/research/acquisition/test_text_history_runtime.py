"""Versioned neighborhood acquisition executes and replays archived bytes.

Synthetic bodies exercise the actual runtime and loader, not football efficacy.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json

import pytest
import requests

from football_agent import learned_voi_replay
from football_agent.inquiry_tools import OpenWebTools
from football_agent.learned_voi_runtime import LearnedVOIRuntime, _save
from football_agent.learned_voi_web import MatchDocumentCandidates
from football_agent.text_voi_policy import TextVOIPolicy
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.text_history import HistoryNeighborhoodTextReturnModel, load_text_model
from football_spider.storage.raw_archive import RawArchive
from test_joint_returns import fitted, state
from test_text_returns import FIRST, TEXTS, rows


@pytest.fixture(scope="module")
def neighborhood():
    return HistoryNeighborhoodTextReturnModel().fit(
        rows(), as_of="2026-02-01T00:00:00Z", dataset_version="constructed-runtime-only",
        evidence_origin="constructed_mechanism", text_clusters=3, action_clusters=2,
        min_matches=3, min_class_matches=.5)


@pytest.mark.parametrize("mode", ["risk_model", "discovery_model"])
def test_new_kernel_cannot_silently_use_old_auxiliary_model(fitted, neighborhood, mode):
    base = LearnedJointVOIPolicy(fitted[0], lambda_cost={"money": 1.})
    with pytest.raises(ValueError, match="separately validated"):
        TextVOIPolicy(base, neighborhood, **{mode: object()})


def test_neighborhood_runtime_freezes_model_and_replays_without_network(
        fitted, neighborhood, tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Synthetic archive replay attempted network I/O")
    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    monkeypatch.setattr(learned_voi_replay, "load_frozen_models", lambda *a, **kw: (None, fitted[0]))
    root = tmp_path / "project"
    freeze = root / "var/parallel/agent/round175/frozen-next-period.json"
    _save(freeze, {"scope": "constructed fixture only"})
    policy = TextVOIPolicy(LearnedJointVOIPolicy(fitted[0], lambda_cost={"money": 1.}), neighborhood)
    assert policy.manifest["history_kernel"] == neighborhood.metadata["history_kernel"]
    assert policy.manifest["production_ready"] is False
    initial = state()
    initial["prediction_cutoff"] = "2026-03-01T10:00:00Z"
    initial["base"]["kickoff"] = "2026-03-02T00:00:00Z"
    identity = {"match_id": initial["match_id"], "home_team": "Arsenal", "away_team": "Leeds",
                "kickoff_at": initial["base"]["kickoff"], "prediction_cutoff": initial["prediction_cutoff"]}
    generator = MatchDocumentCandidates(match_terms=["Arsenal", "Leeds"], seed_urls=[FIRST["url"]],
        allowed_hosts=["club.example"], numeric_actions=[], web_cost={"money": 0., "requests": 1.})
    folder = tmp_path / "original"
    archive = RawArchive(folder / "source-archive")
    clock = lambda: datetime(2026, 3, 1, tzinfo=timezone.utc)
    tools = OpenWebTools(archive, ["club.example"], clock=clock, extraction_version=3)

    def execute(action, public):
        config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
        frozen_path = folder / "text-response-model.json"
        assert config["text_response_model"]["sha256"] == sha256(frozen_path.read_bytes()).hexdigest()
        assert type(load_text_model(frozen_path)) is HistoryNeighborhoodTextReturnModel
        prepared = tools.prepare(action, {"task": {"seed_urls": generator.seed_urls},
                                          "documents": public["documents"]})
        body = ("<html><body><p>" + TEXTS[0] + "</p></body></html>").encode()
        observation = archive.save(prepared["source"], prepared["endpoint"], prepared["params"],
                                   body, status=200, observed_at=clock())
        return {"status": "OK", "document": tools.project(observation, prepared),
                "actual_cost": {"money": 0., "requests": 1., "latency_ms": 0., "tokens": 0.},
                "receipt": {"observation": observation.model_dump(mode="json"), "prepared": prepared}}

    metadata = {"generator": generator.configuration(), "transport": tools.configuration(),
                "input_bindings": {"frozen_model_config_sha256": sha256(freeze.read_bytes()).hexdigest()}}
    runtime = LearnedVOIRuntime(policy, seed=186, exploration_limits={"money": 0., "requests": 0.},
                                max_exploration_actions=0)
    run = runtime.run(initial, run_id="neighborhood-archive", limits={"money": 0., "requests": 1.},
        propose=generator, execute=execute, output=folder, clock=clock, evidence_match=identity,
        provider_version="synthetic-archived-html", generator_version=generator.version, run_metadata=metadata)
    assert len(run["attempts"]) == 1
    assert run["text_response_learning"]["probability_updates"] == 1
    assert run["initial_probabilities"] != run["final_probabilities"]
    report = learned_voi_replay.replay_saved_run(folder, output=tmp_path / "replay", project_root=root)
    assert report["identical_replay"] is True
    assert report["replay_network_requests"] == 0
    for name in ("run.json", "configuration.json", "text-response-model.json"):
        assert (folder / name).read_bytes() == (tmp_path / "replay" / name).read_bytes()
