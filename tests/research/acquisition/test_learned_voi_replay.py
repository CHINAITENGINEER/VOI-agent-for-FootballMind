from datetime import datetime, timezone
from hashlib import sha256
import json

import pytest

from football_agent.inquiry_tools import OpenWebTools
from football_agent.learned_voi_runtime import LearnedVOIRuntime, _save
from football_agent.learned_voi_web import MatchDocumentCandidates
from football_agent import learned_voi_replay
from football_research.acquisition.conservative_joint import ConservativeJointReturnPredictor
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_spider.storage.raw_archive import RawArchive
from test_joint_returns import ACTIONS, fitted, state


@pytest.fixture
def saved(tmp_path, fitted, monkeypatch):
    # Synthetic mechanism evidence; neither fixtures nor returns are real matches.
    model = ConservativeJointReturnPredictor(fitted[0], strength=0.)
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 1.})
    frozen = tmp_path/"project/var/parallel/agent/round175/frozen-next-period.json"
    _save(frozen, {"synthetic": True})
    monkeypatch.setattr(learned_voi_replay, "load_frozen_models", lambda *_args, **_kwargs: (None, model))
    generator = MatchDocumentCandidates(match_terms=["Arsenal", "Leeds"],
        seed_urls=["https://understat.com/league/EPL/2026"], allowed_hosts=["understat.com"],
        numeric_actions=ACTIONS, web_cost={"money": .01, "requests": 1})
    folder = tmp_path/"original"
    archive = RawArchive(folder/"source-archive")
    clock = lambda: datetime(2025, 4, 30, tzinfo=timezone.utc)
    tools = OpenWebTools(archive, ["understat.com"], clock=clock, extraction_version=3)

    def execute(action, public):
        prepared = tools.prepare(action, {"task": {"seed_urls": generator.seed_urls}, "documents": public["documents"]})
        body = (b'<rss><channel><item><link>https://understat.com/team/Arsenal/2026</link><title>Arsenal</title></item></channel></rss>'
                if prepared["tool"] == "search_web" else b'<html><base href="https://understat.com/"><a href="team/Arsenal/2026">Arsenal</a></html>')
        obs = archive.save(prepared["source"], prepared["endpoint"], prepared["params"], body,
                           status=200, observed_at=clock())
        return {"status": "OK", "document": tools.project(obs, prepared),
                "actual_cost": {"money": None, "requests": 1., "latency_ms": 10., "tokens": 0.},
                "receipt": {"observation": obs.model_dump(mode="json"), "prepared": prepared}}

    metadata = {"generator": generator.configuration(), "transport": tools.configuration(),
                "input_bindings": {"frozen_model_config_sha256": sha256(frozen.read_bytes()).hexdigest()}}
    runtime = LearnedVOIRuntime(policy, seed=17, exploration_limits={"money": .01, "requests": 1},
                                 max_exploration_actions=1)
    original = runtime.run(state(), run_id="synthetic-source-replay", limits={"money": .1, "requests": 2},
        propose=generator, execute=execute, output=folder, provider_version="synthetic", generator_version="synthetic",
        clock=clock, run_metadata=metadata)
    return folder, tmp_path/"project", original


def test_full_replay_reprojects_bytes_and_exports_unknown_not_zero_gain(saved, tmp_path):
    folder, root, _ = saved
    report = learned_voi_replay.replay_saved_run(folder, output=tmp_path/"replay", project_root=root)
    assert report["identical_replay"] is True
    assert report["replay_network_requests"] == 0
    assert report["raw_responses_reprojected"] == 1
    records = json.loads((tmp_path/"replay/learning-records.json").read_text(encoding="utf-8"))
    assert records[0]["estimated_net_voi"] is None
    assert records[0]["realized_predictive_gain"] is None
    assert records[0]["numeric_voi_training_eligible"] is False
    assert records[0]["conditional_selection_probability"] == .5


@pytest.mark.parametrize("fault", ["text", "actual_requests", "cycle", "raw", "configuration"])
def test_replay_rejects_tampered_evidence_or_accounting(saved, tmp_path, fault):
    folder, root, original = saved
    receipt_path = folder/"receipt-000.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if fault == "text":
        receipt["document"]["text"] = "Invented lineup"
        _save(receipt_path, receipt)
    elif fault == "actual_requests":
        receipt["actual_cost"]["requests"] = 0
        _save(receipt_path, receipt)
    elif fault == "cycle":
        path = folder/"cycle-000.json"
        cycle = json.loads(path.read_text(encoding="utf-8"))
        cycle["selection_basis"] = "invented_positive_voi"
        _save(path, cycle)
    elif fault == "configuration":
        path = folder/"configuration.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["inquiry_exploration"]["seed"] = 999
        _save(path, config)
    else:
        import gzip
        path = folder/"source-archive"/receipt["receipt"]["observation"]["payload_path"]
        path.write_bytes(gzip.compress(b'changed body'))
    with pytest.raises(ValueError):
        learned_voi_replay.replay_saved_run(folder, output=tmp_path/"rejected", project_root=root)
