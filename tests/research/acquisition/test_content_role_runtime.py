"""Offline content-role -> candidate -> raw archive -> exact replay checks.

Constructed responses below exercise acquisition contracts, not real policy
benefit. Frozen real returns are reinterpreted only in separate test output;
their original actor state and acquisition clocks are never rewritten.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.learned_voi_replay import replay_saved_run
from football_agent.match_evidence import build_match_evidence
from football_agent.news_discovery import NewsDiscoveryTools
from football_research.data_adapters.prospective_text import collect_prepared_batch
from test_verified_collection import verified_inputs


ROOT = Path(__file__).resolve().parents[3]
CONTENT_VERSION = "match-evidence-extraction-v4"
CANADA = ROOT/"var/parallel/agent/round190/live-canada/match-3097883"
CROATIA = ROOT/"var/parallel/agent/round180/live-batch/match-2981455"
ESTONIA = ROOT/"var/parallel/agent/round184/prospective-panels/actor-2981554/match-2981554"
ARTICLE = "https://www.uefa.com/uefanationsleague/news/croatia-visible-squad-update/"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Content-role integration tests must never use the network")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hashes(folder):
    return {p.relative_to(folder).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


def generator_v5(config):
    original = config["supervisor_metadata"]["generator"]
    fields = ("match_terms", "seed_urls", "allowed_hosts", "numeric_actions", "web_cost",
              "match", "profile", "max_discovered_links", "max_followup_queries", "max_candidates")
    return EvidenceDocumentCandidates(**{key: deepcopy(original[key]) for key in fields}, discovery_version=5)


@pytest.mark.parametrize("case", ["unknown", "article", "failed"])
def test_new_versions_acquire_save_and_replay_without_fake_value(verified_inputs, monkeypatch, case):
    inputs, calls = verified_inputs, []
    original_body = inputs["body"]
    plain = ("<html><body><div>Croatia may rotate their starting lineup against Spain. "
             "NAVIGATION_SENTINEL Coach Education Directory.</div>"
             f'<a href="{ARTICLE}">Croatia visible squad update</a></body></html>').encode()

    def archived_return(self, prepared, *, run_id, request_id):
        # The production collector must persist the opt-in before I/O.
        batch = read(inputs["root"]/"batch/batch-configuration.json")
        assert batch["evidence_version"] == CONTENT_VERSION
        assert batch["candidate_discovery_version"] == 5
        calls.append(deepcopy(prepared))
        observed = datetime.now(timezone.utc)
        search = prepared["tool"] == "search_web"
        body = b"<rss><channel></channel></rss>" if search else original_body if case == "article" else plain
        status = 503 if case == "failed" else 200
        observation = self.archive.save(prepared["source"], prepared["endpoint"], prepared["params"], body,
            status=status, observed_at=observed, context={"requested_at": observed.isoformat()})
        return {"document": self.project(observation, prepared), "observation": observation.model_dump(mode="json"),
                "http_attempts": 1, "money": None, "money_status": "not_measured", "latency_seconds": .01}

    monkeypatch.setattr(NewsDiscoveryTools, "execute", archived_return)
    base = inputs["root"]
    report = collect_prepared_batch(base/"plan.json", output=base/"batch", archive_root=base/"market",
        allowed_hosts=["www.uefa.com"], identity_bundle=base/"bindings.json", identity_archive=base/"identity",
        news_provider="bing_rss", news_projection_version=2, source_profile_version=1,
        evidence_version=CONTENT_VERSION, candidate_discovery_version=5, requests_per_match=2, seed=180)
    folder = base/"batch/match-101"
    config, run = read(folder/"configuration.json"), read(folder/"run.json")
    assert len(calls) == report["http_requests"] == run["measured_resources"]["requests"] == 2
    assert any(call["tool"] == "read_url" for call in calls)
    assert config["evidence_projection"]["version"] == CONTENT_VERSION
    assert config["supervisor_metadata"]["generator"]["version"] == "evidence-document-candidates-v5"
    restored = EvidenceDocumentCandidates.from_configuration(config["supervisor_metadata"]["generator"])
    assert restored.configuration() == config["supervisor_metadata"]["generator"]
    final = run["evidence"]
    assert final["schema_version"] == CONTENT_VERSION
    assert run["initial_probabilities"] == run["final_probabilities"]
    assert run["outcome"] is None and not run["open_world_policy_benefit_proven"]
    assert not run["numeric_voi_training_eligible"]
    assert ARTICLE not in json.dumps(run["cycles"][0]["public_before"])
    assert ARTICLE not in json.dumps(run["cycles"][0]["generated_actions"])
    assert "NAVIGATION_SENTINEL" not in json.dumps(run["cycles"][0]["public_before"])
    for cycle in run["cycles"]:
        for candidate in cycle["strict_decision"]["candidates"]:
            # Exhausted/invalid candidates have no value object at all.
            assert candidate["value"] is None or candidate["value"]["net_voi"] is None
        assert not any("expected_gain" in action for action in cycle["generated_actions"])
    if case == "article":
        assert final["evidence"] and final["state"]["followup_candidates"]
        documents = {doc["document_id"]: doc for doc in run["documents"]}
        for item in final["evidence"]:
            anchor = item["anchor"]
            assert documents[anchor["document_id"]]["text"][anchor["start"]:anchor["end"]] == item["statement"]
        assert any(len([a for a in cycle["generated_actions"] if a["tool"] == "search_web"]) > 1
                   for cycle in run["cycles"])
        assert all(p["expected_gain"] is None for p in final["state"]["followup_candidates"])
    else:
        assert final["evidence"] == final["unscoped_source_quotes"] == []
        assert final["state"]["followup_candidates"] == []
        assert all(sum(a["tool"] == "search_web" for a in c["generated_actions"]) == 1 for c in run["cycles"])
        if case == "failed":
            assert run["documents"] == []
            assert all(c["generated_actions"] == run["cycles"][0]["generated_actions"] for c in run["cycles"])
        else:
            assert any(d.get("structured_news", {}).get("role") == "unknown" for d in run["documents"])
            assert any(a.get("url") == ARTICLE for c in run["cycles"] for a in c["generated_actions"])
    replayed = replay_saved_run(folder, output=base/"replay", project_root=base)
    assert replayed["identical_replay"] and replayed["raw_responses_reprojected"] == 2
    assert replayed["replay_network_requests"] == 0
    for name in ("configuration.json", "run.json", "receipt-000.json", "receipt-001.json"):
        assert (folder/name).read_bytes() == (base/"replay"/name).read_bytes()


@pytest.mark.parametrize("folder,version", [(ESTONIA, 3), (CANADA, 4)])
def test_frozen_old_candidate_versions_replay_without_rewriting(folder, version, tmp_path):
    before = hashes(folder)
    config = read(folder/"configuration.json")
    assert config["supervisor_metadata"]["generator"]["version"] == f"evidence-document-candidates-v{version}"
    assert config["evidence_projection"]["version"] != CONTENT_VERSION
    result = replay_saved_run(folder, output=tmp_path/"replay", project_root=ROOT)
    assert result["identical_replay"] and result["replay_network_requests"] == 0
    assert (folder/"run.json").read_bytes() == (tmp_path/"replay/run.json").read_bytes()
    assert hashes(folder) == before


def test_real_unknown_body_loses_navigation_query_but_keeps_observed_links():
    before = hashes(CANADA)
    config, run = read(CANADA/"configuration.json"), read(CANADA/"run.json")
    public = deepcopy(run["cycles"][-1]["public_before"])
    assert any("Coach Education" in row["query"] for row in public["evidence"]["state"]["followup_candidates"])
    documents = deepcopy(public["documents"])
    public["evidence"] = build_match_evidence(config["evidence_projection"]["match"], documents,
        as_of=public["decision_at"], version=CONTENT_VERSION)
    assert public["evidence"]["evidence"] == public["evidence"]["unscoped_source_quotes"] == []
    assert public["evidence"]["state"]["followup_candidates"] == []
    assert public["documents"] == documents
    actions = generator_v5(config)(public)
    assert sum(a["tool"] == "search_web" for a in actions) == 1
    observed_links = {link["url"] for d in documents for link in d["links"]}
    seeds = set(config["supervisor_metadata"]["generator"]["seed_urls"])
    newly_discovered = [a for a in actions if a["tool"] == "read_url" and a["url"] not in seeds]
    assert "https://news.canadasoccer.com/" in observed_links
    # The actual archived match page has only the news root, no article URL.
    # Reinterpreting it must not import the separately captured homepage menu.
    assert newly_discovered == []
    assert hashes(CANADA) == before


def test_actual_archived_article_still_has_source_anchored_followup():
    before = hashes(CROATIA)
    config, run = read(CROATIA/"configuration.json"), read(CROATIA/"run.json")
    doc = read(CROATIA/"receipt-001.json")["document"]
    assert doc["structured_news"]["role"] == "liveblog"
    evidence = build_match_evidence(config["evidence_projection"]["match"], [doc],
        as_of=run["finished_at"], version=CONTENT_VERSION)
    assert evidence["evidence"]
    assert any("Budimir" in proposal["query"] for proposal in evidence["state"]["followup_candidates"])
    for item in evidence["evidence"]:
        anchor = item["anchor"]
        assert doc["text"][anchor["start"]:anchor["end"]] == item["statement"]
    assert evidence["semantics"]["numeric_voi"] is None
    assert hashes(CROATIA) == before
