"""Real collector/archive/replay with constructed bytes and zero HTTP.

This checks the acquisition contract, not learned information value or a new
real match sample. Existing model and policy inputs remain unchanged.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket

import pytest

from football_agent.learned_voi_replay import replay_saved_run
from football_agent.learned_voi_runtime import digest
from football_agent.learned_voi_web import _web_action
from football_agent.news_discovery import NewsDiscoveryTools
from football_research.data_adapters.prospective_text import collect_prepared_batch
from test_verified_collection import verified_inputs


ARTICLE = "https://www.uefa.com/uefanationsleague/news/croatia-visible-squad-update/"
EVIDENCE = "match-evidence-extraction-v5"
ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Candidate context integration must not connect to the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def test_news4_candidate7_collects_observed_labels_before_child_and_replays(verified_inputs, monkeypatch):
    inputs, calls = verified_inputs, []
    fixture_url = inputs["binding"]["canonical"]["source_url"]
    parent_body = ("<html><main><div>Official Croatia football updates and the international match schedule. "
        "This directory contains source links, not a report about player availability.</div>"
        f'<a href="{ARTICLE}">6 October 2026</a>'
        f'<a href="{ARTICLE}">Observed Croatia squad headline</a>'
        f'<a href="{ARTICLE}">Read story</a>'
        f'<a href="{ARTICLE}" hidden>HIDDEN_LINK_LABEL</a>'
        "</main></html>").encode()
    article_body = ("<html><script type=\"application/ld+json\">" + json.dumps({
        "@type": "NewsArticle", "headline": "Croatia match news",
        "datePublished": (inputs["now"]-timedelta(minutes=2)).isoformat(),
        "articleBody": "UNREAD_BODY_SENTINEL Croatia may rotate their starting lineup against Spain.",
    }) + "</script></html>").encode()

    def execute(self, prepared, *, run_id, request_id):
        calls.append(deepcopy(prepared))
        batch = read(inputs["root"]/"batch/batch-configuration.json")
        assert (batch["news_projection_version"], batch["candidate_discovery_version"]) == (4, 7)
        assert batch["evidence_version"] == EVIDENCE
        if prepared["tool"] == "search_web":
            body = b"<rss><channel></channel></rss>"
        else:
            assert prepared["endpoint"] in {fixture_url, ARTICLE}
            body = parent_body if prepared["endpoint"] == fixture_url else article_body
        now = datetime.now(timezone.utc)
        observation = self.archive.save(prepared["source"], prepared["endpoint"], prepared["params"],
            body, status=200, observed_at=now,
            context={"requested_at": now.isoformat(), "inquiry_run_id": run_id})
        return {"document": self.project(observation, prepared),
                "observation": observation.model_dump(mode="json"), "http_attempts": 1,
                "money": None, "money_status": "not_measured", "latency_seconds": .01}

    monkeypatch.setattr(NewsDiscoveryTools, "execute", execute)
    base = inputs["root"]
    report = collect_prepared_batch(base/"plan.json", output=base/"batch", archive_root=base/"market",
        allowed_hosts=["www.uefa.com"], identity_bundle=base/"bindings.json", identity_archive=base/"identity",
        news_provider="bing_rss", source_profile_version=1, news_projection_version=4,
        evidence_version=EVIDENCE, candidate_discovery_version=7, requests_per_match=3, seed=192)
    folder = base/"batch/match-101"
    run, config = read(folder/"run.json"), read(folder/"configuration.json")
    assert len(calls) == report["http_requests"] == 3
    assert config["supervisor_metadata"]["generator"]["version"] == "evidence-document-candidates-v7"
    assert config["supervisor_metadata"]["transport"]["news_discovery"]["projection_version"] == 4
    assert ARTICLE not in json.dumps(run["cycles"][0]["public_before"])
    assert ARTICLE not in json.dumps(run["cycles"][0]["generated_actions"])
    selected = next(c for c in run["cycles"] if c["selected_action"] and c["selected_action"].get("url") == ARTICLE)
    action, public = selected["selected_action"], selected["public_before"]
    assert action == _web_action("read_url", ARTICLE, cost=action["cost"])
    assert "UNREAD_BODY_SENTINEL" not in json.dumps(public)
    context = public["candidate_contexts"]
    assert context["schema_version"] == "observed-candidate-context-v1"
    observations = context["by_action_id"][action["action_id"]]["observations"]
    assert len(observations) == 1
    assert [a["text"] for a in observations[0]["observed_anchors"]] == [
        "6 October 2026", "Observed Croatia squad headline", "Read story"]
    assert "HIDDEN_LINK_LABEL" not in json.dumps(context)
    parent = observations[0]["parent"]
    document = next(doc for doc in public["documents"] if doc["document_id"] == parent["document_id"])
    assert parent["raw_sha256"] == document["raw_sha256"]
    assert parent["available_at"] == document["available_at"]
    assert context["semantics"]["numerical_value_assigned"] is False
    assert selected["selection_basis"] == "bounded_unknown_information_exploration"
    assert selected["exploration_record"]["expected_information_value"] is None
    assert selected["public_digest"] == digest(public)
    assert "UNREAD_BODY_SENTINEL" in json.dumps(run["documents"])
    assert run["initial_probabilities"] == run["final_probabilities"]
    assert run["outcome"] is None and not run["open_world_policy_benefit_proven"]

    # A projection/context mutation cannot be accepted merely by changing its
    # saved cycle. Full replay regenerates the sidecar from original raw bytes.
    replay = replay_saved_run(folder, output=base/"replay", project_root=base)
    assert replay["identical_replay"] and replay["replay_network_requests"] == 0
    assert replay["raw_responses_reprojected"] == 3
    for name in ("configuration.json", "run.json", "cycle-000.json", "cycle-001.json",
                 "cycle-002.json", "receipt-000.json", "receipt-001.json", "receipt-002.json"):
        assert (folder/name).read_bytes() == (base/"replay"/name).read_bytes()


def test_frozen_real_v5_candidate_run_stays_exact_and_has_no_new_sidecar(tmp_path):
    folder = ROOT/"var/parallel/agent/round191/live-canada/match-3097883"
    original = (folder/"run.json").read_bytes()
    config = read(folder/"configuration.json")
    assert config["supervisor_metadata"]["generator"]["version"] == "evidence-document-candidates-v5"
    assert all("candidate_contexts" not in c["public_before"] for c in read(folder/"run.json")["cycles"])
    replay = replay_saved_run(folder, output=tmp_path/"legacy-replay", project_root=ROOT)
    assert replay["identical_replay"] and replay["replay_network_requests"] == 0
    assert (tmp_path/"legacy-replay/run.json").read_bytes() == original
    assert (folder/"run.json").read_bytes() == original
