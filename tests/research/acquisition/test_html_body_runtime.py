"""New typed HTML paragraphs and actual runtime, without network or new data.

Constructed fixtures exercise live acquisition contracts. The frozen real
return is reinterpreted using a declared test clock only, never reclassified
as a historical successful actor observation or an independent sample.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import socket

import pytest

from football_agent.learned_voi_replay import replay_saved_run
from football_agent.match_evidence import build_match_evidence
from football_agent.news_discovery import NewsDiscoveryTools
from football_research.acquisition.state import parse_time
from football_research.data_adapters.prospective_text import collect_prepared_batch, join_settlements
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from test_verified_collection import verified_inputs
from test_content_role_runtime import ROOT, read, hashes


FOLDER = ROOT/"var/parallel/agent/round191/live-canada/match-3097883"
EVIDENCE = "match-evidence-extraction-v5"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("HTML body integration checks must remain offline")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


@pytest.mark.parametrize("squad", ["target", "other", "unknown"])
def test_metadata_only_html_article_to_versioned_runtime_and_replay(verified_inputs, monkeypatch, squad):
    inputs, calls = verified_inputs, []
    fixture_url = inputs["binding"]["canonical"]["source_url"]
    headline = "Croatia national team selection update"
    lead = ("Croatia midfielder Marko Example may miss Spain with a muscle injury before the next match."
            if squad == "target" else
            "Croatia Women's National Team will host Denmark on 12 October 2026 in the next international window.")
    later = "Croatia are assessing player fitness and may rotate their starting lineup before the next match."
    metadata = {"@context": "https://schema.org", "@type": "NewsArticle", "url": fixture_url,
        "headline": headline, "datePublished": (inputs["now"]-timedelta(minutes=3)).isoformat(),
        "dateModified": (inputs["now"]-timedelta(minutes=2)).isoformat()}
    body = (f'<html lang="en"><script type="application/ld+json">{json.dumps(metadata)}</script>'
        '<nav><p>NAVIGATION_SENTINEL Croatia may rotate.</p></nav><main><article>'
        f'<header><h1>{headline}</h1><p>SUMMARY_SENTINEL Croatia may be uncertain.</p></header>'
        f'<section><p>{lead}</p><p>{later}</p></section>'
        '<footer><p>FOOTER_SENTINEL injury store information.</p></footer></article></main></html>').encode()
    if squad == "unknown":
        body = b"<html><div>UNKNOWN_PAGE_SENTINEL Croatia may rotate against Spain. Site directory and navigation.</div></html>"

    def execute(self, prepared, *, run_id, request_id):
        batch = read(inputs["root"]/"batch/batch-configuration.json")
        assert batch["news_projection_version"] == 3
        assert batch["evidence_version"] == EVIDENCE and batch["candidate_discovery_version"] == 6
        calls.append(deepcopy(prepared))
        now = datetime.now(timezone.utc)
        returned = b"<rss><channel></channel></rss>" if prepared["tool"] == "search_web" else body
        observation = self.archive.save(prepared["source"], prepared["endpoint"], prepared["params"], returned,
            status=200, observed_at=now, context={"requested_at": now.isoformat()})
        return {"document": self.project(observation, prepared), "observation": observation.model_dump(mode="json"),
            "http_attempts": 1, "money": None, "money_status": "not_measured", "latency_seconds": .01}

    monkeypatch.setattr(NewsDiscoveryTools, "execute", execute)
    base = inputs["root"]
    report = collect_prepared_batch(base/"plan.json", output=base/"batch", archive_root=base/"market",
        allowed_hosts=["www.uefa.com"], identity_bundle=base/"bindings.json", identity_archive=base/"identity",
        news_provider="bing_rss", source_profile_version=1, news_projection_version=3,
        evidence_version=EVIDENCE, candidate_discovery_version=6, requests_per_match=2, seed=180)
    folder = base/"batch/match-101"
    run, config = read(folder/"run.json"), read(folder/"configuration.json")
    assert len(calls) == report["http_requests"] == 2
    doc = next(doc for doc in run["documents"] if doc.get("structured_news", {}).get("role") ==
               ("unknown" if squad == "unknown" else "article"))
    if squad == "unknown":
        assert doc["status"] == "OK" and "UNKNOWN_PAGE_SENTINEL" in doc["text"]
        assert doc["structured_news"]["segments"] == []
    else:
        assert lead in doc["text"] and later in doc["text"]
    assert all(marker not in doc["text"] for marker in ("NAVIGATION_SENTINEL", "SUMMARY_SENTINEL", "FOOTER_SENTINEL"))
    assert doc["structured_news"]["version"] == "structured-news-discovery-v3"
    for segment in doc["structured_news"]["segments"]:
        assert segment["content_source"] == "semantic_html_article_paragraph"
        assert segment["source_url"] == fixture_url and segment["content_role"] == "paragraph"
        assert doc["text"][segment["start"]:segment["end"]] == segment["text"]
    evidence = run["evidence"]
    if squad == "target":
        assert evidence["evidence"] and evidence["state"]["followup_candidates"]
        assert all(row["expected_gain"] is None for row in evidence["state"]["followup_candidates"])
    elif squad == "other":
        # The first actual body paragraph identifies another squad. Later bare
        # country-name references must not silently migrate to the target men.
        assert evidence["evidence"] == evidence["state"]["followup_candidates"] == []
        assert evidence["unscoped_source_quotes"]
    else:
        assert evidence["evidence"] == evidence["state"]["followup_candidates"] == []
        assert evidence["unscoped_source_quotes"] == []
        joined = join_settlements([folder], settlements=[], prior_bindings=read(folder/"prior-bindings.json"),
                                  as_of=datetime.now(timezone.utc))
        unknown = next(row for row in joined["dataset"]["episodes"]
                       if row["response"]["document_id"] == doc["document_id"])
        assert unknown["response"]["status"] == "OK"
        assert unknown["response"]["kind"] == "unknown"
        assert unknown["response"]["document_role"] == "unknown_page_role"
        assert "response_kind_unknown" in unknown["training_exclusion_reasons"]
    assert run["initial_probabilities"] == run["final_probabilities"]
    assert run["outcome"] is None and not run["numeric_voi_training_eligible"]
    assert config["supervisor_metadata"]["generator"]["version"] == "evidence-document-candidates-v6"
    assert config["evidence_projection"]["version"] == EVIDENCE
    replay = replay_saved_run(folder, output=base/"replay", project_root=base)
    assert replay["identical_replay"] and replay["raw_responses_reprojected"] == 2
    assert replay["replay_network_requests"] == 0
    for name in ("configuration.json", "run.json", "receipt-000.json", "receipt-001.json"):
        assert (folder/name).read_bytes() == (base/"replay"/name).read_bytes()


def test_real_raw_recovery_keeps_old_nodata_and_wrong_squad_scope():
    before = hashes(FOLDER)
    config, returned = read(FOLDER/"configuration.json"), read(FOLDER/"receipt-003.json")
    archive = RawArchive(FOLDER/"source-archive")
    observation = Observation.model_validate(returned["receipt"]["observation"])
    assert observation.http_status == 200 and len(archive.read_body(observation)) == 124276
    old = NewsDiscoveryTools.from_configuration(archive, config["supervisor_metadata"]["transport"],
        clock=lambda: parse_time(returned["document"]["available_at"]))
    assert old.project(observation, returned["receipt"]["prepared"]) == returned["document"]
    assert returned["status"] == "NO_DATA" and returned["document"]["text"] == ""
    # Deliberately fixed test clock after the original capture. This is a
    # retrospective mechanism check, not evidence that the old actor read it.
    reinterpret_at = parse_time(returned["document"]["available_at"])+timedelta(minutes=1)
    tools = NewsDiscoveryTools(archive, config["supervisor_metadata"]["generator"]["allowed_hosts"],
        provider="bing_rss", default_route="direct", extraction_version=3, projection_version=3,
        index_urls=["https://canadasoccer.com/"], clock=lambda: reinterpret_at)
    doc = tools.project(observation, returned["receipt"]["prepared"])
    assert doc["status"] == "OK" and doc["structured_news"]["role"] == "article"
    assert len(doc["structured_news"]["segments"]) == 7
    assert [p["original_dom_paragraph_index"] for p in doc["structured_news"]["segments"]] == list(range(2, 9))
    assert doc["raw_sha256"] == returned["document"]["raw_sha256"]
    assert doc["observed_at"] == returned["document"]["observed_at"]
    assert doc["available_at"] == reinterpret_at.isoformat()
    assert "Women’s National Team" in doc["text"] and "Denmark" in doc["text"]
    evidence = build_match_evidence(config["evidence_projection"]["match"], [doc],
        as_of=reinterpret_at, version=EVIDENCE)
    assert evidence["evidence"] == evidence["state"]["followup_candidates"] == []
    assert evidence["unscoped_source_quotes"]
    assert evidence["semantics"]["numeric_voi"] is None
    assert hashes(FOLDER) == before
