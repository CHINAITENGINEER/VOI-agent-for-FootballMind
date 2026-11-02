"""Shared structural input checks; no f/critic fit or network dependency."""
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from football_agent.candidate_context import build_candidate_contexts
from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.learned_voi_web import _web_action
from football_research.acquisition.state import parse_time
from football_research.acquisition.visible_state import encode_visible_state, VERSION


ROOT = Path(__file__).resolve().parents[3]
NOW = "2026-10-06T21:00:00+00:00"
ARTICLE = "https://www.uefa.com/uefanationsleague/news/visible-team-update/"
HOSTS = ["www.uefa.com"]
COST = {"money": .01, "requests": 1.}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Visible state encoding must not perform any network access")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def action(url=ARTICLE):
    return _web_action("read_url", url, cost=COST)


def document(text="Read source paragraph.", *, kind="article", docid="doc-one", url=ARTICLE):
    doc = {"document_id": docid, "raw_sha256": "a"*64, "status": "OK", "kind": kind,
           "url": url, "text": text, "links": [], "observed_at": "2026-10-06T20:00:00+00:00",
           "available_at": "2026-10-06T20:00:01+00:00", "source_published_at": None}
    if kind == "article":
        doc["structured_news"] = {"version": "structured-news-discovery-v3", "role": "article",
            "segments": [{"text": text, "start": 0, "end": len(text),
                "content_source": "json_ld_article_body", "json_pointer": "script[0].articleBody",
                "source_published_at": "2026-10-06T19:00:00+00:00",
                "source_modified_at": "2026-10-06T19:30:00+00:00"}]}
    return doc


def parent():
    doc = document("NAVIGATION_FULLTEXT_NOT_ARTICLE", kind="index", docid="doc-parent",
                   url="https://www.uefa.com/")
    doc["links"] = [{"url": ARTICLE, "title": "6 October 2026", "observed_anchors": [
        {"anchor_index": 3, "text": "6 October 2026", "text_truncated": False},
        {"anchor_index": 4, "text": "Observed squad headline before buying body", "text_truncated": False}],
        "observed_anchors_truncated": False}]
    return doc


def public(*documents):
    return {"state": {"match_id": "fixture-visible", "prediction_cutoff": "2026-10-06T23:59:00+00:00",
              "base": {"kickoff": "2026-10-07T00:00:00+00:00", "home_team_id": "home", "away_team_id": "away"},
              "action_history": []}, "decision_at": NOW, "documents": list(documents), "attempts": [],
            "prediction": {"probabilities": [.4, .3, .3], "probability_basis": "observed_market_snapshot",
                           "model_version": "observed-market-test"},
            "budget": {"limits": {"money": .03, "requests": 3}, "spent": {"money": .01, "requests": 1}}}


def attach_context(payload, actions):
    payload["candidate_contexts"] = build_candidate_contexts(payload, actions, allowed_hosts=HOSTS)
    return payload


def test_parent_headline_is_before_information_even_without_candidate_menu():
    payload = attach_context(public(parent()), [action()])
    encoded = encode_visible_state(payload, [action()])
    assert encoded["schema_version"] == VERSION
    assert encoded["before"] == encode_visible_state(payload)["before"]
    links = encoded["before"]["observed_links"]
    assert links[0]["observed_anchors"][1]["text"] == "Observed squad headline before buying body"
    assert encoded["candidates"][action()["action_id"]]["observed_link_refs"] == [links[0]["link_ref"]]
    assert encoded["before"]["contents"] == []
    assert "NAVIGATION_FULLTEXT" not in json.dumps(encoded)
    assert "Observed squad headline" not in json.dumps(encoded["candidates"])
    assert not any(key in json.dumps(encoded) for key in ("expected_gain", "net_voi", "gross_evsi"))


def test_unread_payload_and_independent_label_changes_do_not_affect_encoding():
    payload = attach_context(public(parent()), [action()])
    expected = encode_visible_state(payload, [action()])
    payload.update(outcome=2, future_returns={ARTICLE: {"text": "UNREAD_SECRET", "outcome": 2}},
                   hidden={"label": 0, "gain": 200})
    payload["state"]["label"] = 1
    payload["documents"][0]["links"][0]["target_body"] = "UNREAD_SECRET"
    payload["prediction"]["future_probabilities"] = [.99, .005, .005]
    assert encode_visible_state(payload, [action()]) == expected
    bad = {**action(), "response": {"text": "UNREAD_SECRET"}}
    with pytest.raises(ValueError, match="metadata"):
        encode_visible_state(payload, [bad])


def test_exact_content_deduplicates_while_sources_spans_and_order_survive():
    first = document("A source states that the player is available.")
    second = deepcopy(first)
    second.update(document_id="doc-two", raw_sha256="b"*64, url="https://www.uefa.com/news/copy/")
    third = document("A later source states that the player is unavailable.", docid="doc-three")
    payload = public(first, second, third)
    payload["state"]["action_history"] = [{"action_id": "repeat"}, {"action_id": "repeat"}, {"action_id": "revision"}]
    payload["attempts"] = [{"action_id": aid, "information_key": aid, "status": "OK", "admission": "admitted"}
                           for aid in ("repeat", "repeat", "revision")]
    before = encode_visible_state(payload)["before"]
    assert len(before["contents"]) == 2 and len(before["documents"]) == 3
    references = [doc["content_refs"][0]["content_id"] for doc in before["documents"]]
    assert references[0] == references[1] != references[2]
    assert [doc["raw_sha256"] for doc in before["documents"]] == ["a"*64, "b"*64, "a"*64]
    assert before["history"]["action_history"] == payload["state"]["action_history"]
    assert before["history"]["order_matches_attempts"]
    swapped = deepcopy(payload)
    swapped["state"]["action_history"] = list(reversed(swapped["state"]["action_history"]))
    assert encode_visible_state(swapped)["before"]["history"] != before["history"]


def test_failure_no_data_and_unknown_masks_do_not_become_zero_or_body_evidence():
    failure, missing, opaque = [document("UNTRUSTED_BODY", kind="page", docid=str(i)) for i in range(3)]
    failure["status"], missing["status"] = "FAILED", "NO_DATA"
    payload = public(failure, missing, opaque)
    payload["state"]["action_history"] = [{"action_id": str(i)} for i in range(3)]
    payload["attempts"] = [{"action_id": str(i), "status": status, "admission": "not_admitted"}
                           for i, status in enumerate(("FAILED", "NO_DATA", "OK"))]
    encoded = encode_visible_state(payload)
    masks = [doc["mask"] for doc in encoded["before"]["documents"]]
    assert masks[0]["failure"] and not masks[0]["no_data"]
    assert masks[1]["no_data"] and not masks[1]["failure"]
    assert masks[2]["unknown_role"] and masks[2]["body_unavailable"]
    assert "UNTRUSTED_BODY" not in json.dumps(encoded)
    assert len(encoded["before"]["history"]["attempts"]) == 3
    assert "not_imputed" in encoded["before"]["history"]["clock_scope"]


@pytest.mark.parametrize("changed", [
    {"available_at": "2026-10-06T21:00:01+00:00"},
    {"source_published_at": "2026-10-06T22:00:00+00:00"},
    {"source_modified_at": "2026-10-06T22:00:00+00:00"},
    {"available_at": "2026-10-06T20:00:01"},
    {"raw_sha256": "invalid"},
])
def test_invalid_or_future_document_is_excluded_without_audit_quote(changed):
    doc = document("FUTURE_SECRET")
    doc.update(changed)
    encoded = encode_visible_state(public(doc))
    assert encoded["before"]["documents"] == encoded["before"]["contents"] == []
    assert len(encoded["audit"]["excluded_documents"]) == 1
    assert "FUTURE_SECRET" not in json.dumps(encoded)
    doc["text"] = "DIFFERENT_FUTURE_SECRET"
    assert encode_visible_state(public(doc)) == encoded


def test_cutoff_bounds_visibility_even_at_a_later_decision():
    payload = public(document())
    payload["decision_at"] = "2026-10-08T00:00:00+00:00"
    payload["state"]["prediction_cutoff"] = "2026-10-06T19:59:00+00:00"
    assert encode_visible_state(payload)["before"]["contents"] == []


def test_segment_modified_clock_and_offset_control_exact_body_admission():
    text = "Visible body. FUTURE_SEGMENT_SECRET"
    doc = document(text)
    template = deepcopy(doc["structured_news"]["segments"][0])
    doc["structured_news"]["segments"] = [
        {**template, "start": 0, "end": 13, "text": text[:13]},
        {**template, "start": 14, "end": len(text), "text": text[14:],
         "source_modified_at": "2026-10-06T22:00:00+00:00"}]
    encoded = encode_visible_state(public(doc))
    assert [row["text"] for row in encoded["before"]["contents"]] == ["Visible body."]
    assert "FUTURE_SEGMENT_SECRET" not in json.dumps(encoded)
    assert encoded["audit"]["excluded_segments"][0]["reason"] == "segment_source_time_after_actual_capture"
    doc["structured_news"]["segments"][0]["text"] = "WRONG_OFFSET"
    assert encode_visible_state(public(doc))["before"]["contents"] == []


def test_unknown_local_publication_time_preserved_without_inventing_utc():
    doc = document()
    segment = doc["structured_news"]["segments"][0]
    segment.update(source_published_at=None, source_modified_at=None,
        publication_raw="6. oktoober 2026 19:00", source_publication_date="2026-10-06",
        source_publication_local_time="19:00", publication_timezone="unknown")
    ref = encode_visible_state(public(doc))["before"]["documents"][0]["content_refs"][0]
    assert ref["source_published_at"] is None and ref["publication_timezone"] == "unknown"
    assert ref["publication_raw"] == segment["publication_raw"]


def test_context_tamper_cannot_supply_new_title_or_fake_parent_hash():
    payload = attach_context(public(parent()), [action()])
    observation = payload["candidate_contexts"]["by_action_id"][action()["action_id"]]["observations"][0]
    observation["observed_anchors"][0]["text"] = "INVENTED_TITLE"
    with pytest.raises(ValueError, match="parent source binding"):
        encode_visible_state(payload, [action()])
    payload = attach_context(public(parent()), [action()])
    payload["documents"][0]["raw_sha256"] = "b"*64
    with pytest.raises(ValueError, match="parent source binding"):
        encode_visible_state(payload, [action()])


def test_current_forecast_is_not_silently_promoted_to_original_market_prior():
    payload = public()
    payload["prediction"].pop("probability_basis")
    encoded = encode_visible_state(payload)
    assert encoded["before"]["current_probabilities"] == [.4, .3, .3]
    assert encoded["before"]["prior_probabilities"] is None
    payload["state"]["open_text"] = {"prior_probabilities": [.5, .25, .25], "hidden_outcome": 0}
    assert encode_visible_state(payload)["before"]["prior_probabilities"] == [.5, .25, .25]


def test_encoding_is_deterministic_and_does_not_mutate_or_alias_inputs():
    payload = attach_context(public(parent(), document()), [action()])
    original = deepcopy(payload)
    result = encode_visible_state(payload, [action()])
    assert payload == original and result == encode_visible_state(payload, [action()])
    result["before"]["contents"][0]["text"] = "changed"
    result["before"]["observed_links"][0]["observed_anchors"][0]["text"] = "changed"
    assert payload == original


def test_real_r192_derived_homepage_labels_are_known_before_unread_body():
    source = ROOT/"var/parallel/agent/round192/evaluation/link-projection/document-v4.json"
    original_bytes = source.read_bytes()
    doc = json.loads(original_bytes)
    config = json.loads((ROOT/"var/parallel/agent/round191/live-canada/match-3097883/configuration.json").read_text(encoding="utf-8"))
    run = json.loads((ROOT/"var/parallel/agent/round191/live-canada/match-3097883/run.json").read_text(encoding="utf-8"))
    payload = deepcopy(run["cycles"][3]["public_before"])
    # A declared later derived snapshot, never a rewrite of the old actor.
    payload["decision_at"] = (parse_time(doc["available_at"])+timedelta(seconds=1)).isoformat()
    payload["documents"] = [doc if d["document_id"] == doc["document_id"] else d for d in payload["documents"]]
    cfg = config["supervisor_metadata"]["generator"]
    fields = ("match_terms", "seed_urls", "allowed_hosts", "numeric_actions", "web_cost", "match", "profile",
              "max_discovered_links", "max_followup_queries", "max_candidates")
    gen = EvidenceDocumentCandidates(**{key: deepcopy(cfg[key]) for key in fields}, discovery_version=7)
    menu = gen(payload)
    payload["candidate_contexts"] = gen.candidate_contexts(payload, menu)
    encoded = encode_visible_state(payload, menu)
    assert len(menu) == 13
    assert len(encoded["before"]["observed_links"]) >= 71
    body = json.dumps(encoded["before"], ensure_ascii=False)
    assert "5 Things to Watch" in body and "Melissa Tancredi" in body
    selected = next(a for a in menu if "melissa-tancredi" in a.get("url", ""))
    candidate = encoded["candidates"][selected["action_id"]]
    assert candidate["observed_link_refs"]
    source_doc = next(d for d in encoded["before"]["documents"] if d["document_id"] == doc["document_id"])
    assert source_doc["raw_sha256"] == doc["raw_sha256"]
    assert source_doc["available_at"] == "2026-10-06T20:30:43.104683+00:00"
    assert encoded["before"]["contents"] == []  # index and unknown page, no article fetched here
    assert source.read_bytes() == original_bytes
    assert sha256(source.read_bytes()).hexdigest() == sha256(original_bytes).hexdigest()
