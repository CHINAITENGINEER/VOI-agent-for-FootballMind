"""Observed menu context is not an unopened article or a utility estimate."""
from copy import deepcopy
import json

import pytest

from football_agent.candidate_context import build_candidate_contexts, VERSION
from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.learned_voi_runtime import _actions
from football_agent.learned_voi_web import _web_action
from football_agent.match_source_profiles import resolve_profile


MATCH = {"match_id": "context-match", "home_team": "Arsenal", "away_team": "Leeds",
         "kickoff_at": "2026-10-10T12:00:00Z", "prediction_cutoff": "2026-10-10T11:00:00Z"}
ARTICLE = "https://www.arsenal.com/news/observed-player-update"
COST = {"money": .01, "requests": 1.}
DECISION = "2026-10-09T10:00:00Z"
HOSTS = ["www.arsenal.com"]


def document(*, ident="parent-one", anchors=True):
    link = {"url": ARTICLE, "title": "9 October 2026",
            "UNREAD_BODY": "Never copied from arbitrary linked-page fields"}
    if anchors:
        link.update(observed_anchors=[
            {"anchor_index": 12, "text": "9 October 2026", "text_truncated": False},
            {"anchor_index": 13, "text": "Observed player availability update", "text_truncated": False},
            {"anchor_index": 14, "text": "Read story", "text_truncated": False}],
            observed_anchors_truncated=False)
    return {"document_id": ident, "status": "OK", "url": "https://www.arsenal.com/news",
            "kind": "index", "raw_sha256": "a"*64, "text": "PARENT_FULLTEXT_NOT_A_LABEL",
            "observed_at": "2026-10-09T09:58:00Z", "available_at": "2026-10-09T09:59:00Z",
            "source_published_at": None, "links": [link]}


def public(*documents):
    return {"state": {"prediction_cutoff": MATCH["prediction_cutoff"],
                       "base": {"kickoff": MATCH["kickoff_at"]}},
            "decision_at": DECISION, "documents": list(documents)}


def generate_context(payload, actions=None):
    return build_candidate_contexts(payload, actions or [_web_action("read_url", ARTICLE, cost=COST)],
                                    allowed_hosts=HOSTS)


def generator(version):
    profile = resolve_profile("Arsenal", "Leeds")
    return EvidenceDocumentCandidates(match_terms=["Arsenal", "Leeds"],
        seed_urls=profile["seed_urls"], allowed_hosts=profile["allowed_hosts"], numeric_actions=[],
        web_cost=COST, match=MATCH, profile=profile, discovery_version=version)


def row(result):
    return next(iter(result["by_action_id"].values()))


def test_multiple_observed_labels_and_parent_versions_keep_actual_clocks_only():
    first, second = document(), document(ident="parent-two")
    second.update(raw_sha256="b"*64, observed_at="2026-10-09T09:59:10Z",
                  available_at="2026-10-09T09:59:20Z")
    second["links"][0]["observed_anchors"][1]["text"] = "Updated page link label"
    result = generate_context(public(first, second))
    assert result["schema_version"] == VERSION
    observations = row(result)["observations"]
    assert len(observations) == 2
    assert [item["text"] for item in observations[0]["observed_anchors"]] == [
        "9 October 2026", "Observed player availability update", "Read story"]
    assert observations[0]["parent"]["available_at"] == first["available_at"]
    assert observations[1]["parent"]["raw_sha256"] == "b"*64
    assert observations[0]["parent"]["source_published_at"] is None
    serialized = json.dumps(result)
    assert "PARENT_FULLTEXT" not in serialized and "UNREAD_BODY" not in serialized
    assert "expected_gain" not in serialized and "target_published_at" not in serialized


def test_explicit_empty_filtered_anchors_do_not_revive_old_title():
    doc = document()
    doc["links"][0]["title"] = "HIDDEN_DESCENDANT_FROM_OLD_PROJECTION"
    doc["links"][0]["observed_anchors"] = []
    result = generate_context(public(doc))
    observation = row(result)["observations"][0]
    assert observation["observed_anchors"] == []
    assert observation["label_source"] == "observed_anchors"
    assert "HIDDEN_DESCENDANT" not in json.dumps(result)


def test_legacy_rss_title_is_preserved_without_invented_dom_position_or_date():
    doc = document(anchors=False)
    doc["url"] = "https://www.bing.com/search?q=fixture&format=rss"
    result = generate_context(public(doc))
    observation = row(result)["observations"][0]
    assert observation["label_source"] == "legacy_link_title"
    assert observation["observed_anchors"] == [
        {"anchor_index": None, "text": "9 October 2026", "text_truncated": False}]


@pytest.mark.parametrize("change", [
    {"status": "FAILED"}, {"status": "NO_DATA"}, {"raw_sha256": "absent"},
    {"document_id": None}, {"available_at": "2026-10-09T10:00:01Z"},
    {"observed_at": "2026-10-09T10:00:00Z"}, {"available_at": "2026-10-09T09:59:00"},
])
def test_nonadmitted_or_future_parent_contributes_no_labels(change):
    doc = document()
    doc.update(change)
    assert row(generate_context(public(doc)))["observations"] == []


def test_cutoff_is_enforced_even_if_decision_clock_is_later():
    doc = document()
    payload = public(doc)
    payload["state"]["prediction_cutoff"] = "2026-10-09T09:58:30Z"
    assert row(generate_context(payload))["observations"] == []


def test_context_cannot_expand_menu_or_alias_actions_and_returns():
    doc = document()
    doc["links"].append({"url": "https://www.arsenal.com/news/not-in-menu", "title": "Absent"})
    action = _web_action("read_url", ARTICLE, cost=COST)
    before = deepcopy((doc, action))
    result = generate_context(public(doc), [action])
    assert list(result["by_action_id"]) == [action["action_id"]]
    result["by_action_id"][action["action_id"]]["observations"][0]["observed_anchors"][0]["text"] = "changed"
    assert (doc, action) == before
    assert _actions([action]) == [action]


def test_v7_only_adds_explicit_sidecar_preserving_v6_requests_and_order():
    payload = public(document())
    old, current = generator(6), generator(7)
    assert old(payload) == current(payload)
    assert old.candidate_contexts(payload, old(payload)) is None
    configuration = current.configuration()
    assert configuration["candidate_context_projection"]["version"] == VERSION
    restored = EvidenceDocumentCandidates.from_configuration(configuration)
    assert restored.configuration() == configuration
    assert restored.candidate_contexts(payload, restored(payload)) == current.candidate_contexts(payload, current(payload))
    configuration["candidate_context_projection"]["selection"] = "manual_title_score"
    with pytest.raises(ValueError, match="reconstruct exactly"):
        EvidenceDocumentCandidates.from_configuration(configuration)


@pytest.mark.parametrize("version", range(1, 7))
def test_legacy_generators_do_not_add_context_or_change_configuration(version):
    old = generator(version)
    config = old.configuration()
    assert "candidate_context_projection" not in config
    assert old.candidate_contexts({}, []) is None
    assert EvidenceDocumentCandidates.from_configuration(config).configuration() == config


def test_hidden_target_contents_cannot_change_prequery_context():
    first = public(document())
    second = deepcopy(first)
    first["future_returns"] = {ARTICLE: {"text": "Y=home win", "outcome": 0}}
    second["future_returns"] = {ARTICLE: {"text": "Y=away win", "outcome": 2}}
    assert generate_context(first) == generate_context(second)


@pytest.mark.parametrize("anchors", [None, [{"anchor_index": True, "text": "Bad", "text_truncated": False}],
    [{"anchor_index": 0, "text": "x"*513, "text_truncated": True}]])
def test_invalid_explicit_anchor_contract_cannot_silently_fall_back(anchors):
    doc = document()
    doc["links"][0]["observed_anchors"] = anchors
    with pytest.raises(ValueError, match="anchor|label"):
        generate_context(public(doc))
