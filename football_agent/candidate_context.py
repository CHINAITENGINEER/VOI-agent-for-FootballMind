"""Public, pre-query link labels with their already-read parent provenance.

This sidecar describes observed capabilities. It never reads a target page,
changes request identity, or supplies a numerical value/model feature. A label
may be a date, navigation wording or an inaccurate source assertion.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re

from football_research.acquisition.state import parse_time
from .learned_voi_web import _url

VERSION = "observed-candidate-context-v1"
SPEC = {
    "version": VERSION,
    "source": "admitted_parent_document_links_only",
    "target_content": "not_fetched_or_inferred",
    "labels": "observed_anchors_else_legacy_title_only_when_field_absent",
    "identity": "unchanged_action_and_canonical_request_url",
    "model_features": "unchanged_not_consumed_by_f_or_critic",
    "selection": "unchanged_no_relevance_or_value_score",
}


def _labels(link):
    """An explicitly filtered empty list must not revive an older HTML title."""
    if (link.get("directory_context_basis") == "same_response_cms_article_card_v1"
            and link.get("target_article_fetched_for_context") is False
            and link.get("metadata_visibility") == "serialized_same_response_not_rendered_DOM_anchor"):
        source = link.get("observed_metadata_labels")
        if not isinstance(source, list) or len(source) != 1:
            raise ValueError("A CMS card needs one response-local metadata label")
        item = source[0]
        if (not isinstance(item, Mapping) or not isinstance(item.get("text"), str)
                or not item["text"].strip() or len(item["text"]) > 512
                or type(item.get("text_truncated")) is not bool
                or not isinstance(item.get("pointer"), str)
                or item["pointer"] != link.get("metadata_pointer")
                or not (re.match(r"^(?:data-props\[\d+\]|next_f\[\d+\]\.row\[\d+\])\$", item["pointer"])
                    or (link.get("directory_metadata_adapter") == "serialized-club-directory-metadata-v1"
                        and re.fullmatch(r"(?:jsonld\[\d+\]\$\.itemListElement\[\d+\]|nuxt\[0\]\$\[\d+\]\.attributes\[\d+\])", item["pointer"])))):
            raise ValueError("CMS card label needs its literal response pointer")
        return [{"anchor_index": None, "text": item["text"],
                 "text_truncated": item["text_truncated"]}], "observed_CMS_card_metadata", item["text_truncated"]
    if "observed_anchors" not in link:
        title = link.get("title")
        labels = ([{"anchor_index": None, "text": title[:512],
                    "text_truncated": len(title) > 512}]
                  if isinstance(title, str) and title.strip() else [])
        return labels, "legacy_link_title", bool(labels and labels[0]["text_truncated"])
    source = link["observed_anchors"]
    if not isinstance(source, list) or len(source) > 8:
        raise ValueError("Observed link labels must be the bounded projection list")
    labels = []
    for item in source:
        if (not isinstance(item, Mapping) or type(item.get("anchor_index")) is not int
                or item["anchor_index"] < 0 or not isinstance(item.get("text"), str)
                or len(item["text"]) > 512 or type(item.get("text_truncated")) is not bool):
            raise ValueError("Observed anchor label is not a valid source projection")
        # Select only the declared fields, never arbitrary linked-page metadata.
        labels.append({key: deepcopy(item[key]) for key in ("anchor_index", "text", "text_truncated")})
    truncated = link.get("observed_anchors_truncated", False)
    if type(truncated) is not bool:
        raise ValueError("Observed anchor truncation must be explicit boolean")
    return labels, "observed_anchors", truncated


def build_candidate_contexts(public, actions, *, allowed_hosts):
    """Align already-visible link observations to an existing bounded menu.

    The runtime owns admission. This projection independently requires source
    IDs and observation/availability clocks and does not use future documents
    even when called on an unfiltered public snapshot. Source publication dates
    remain parent-page assertions; anchor date strings are not parsed as dates
    of the unopened target page.
    """
    deadline = min(parse_time(public["decision_at"]),
                   parse_time(public["state"]["prediction_cutoff"]))
    by_url, rows = {}, {}
    for action in actions:
        if action.get("tool") != "read_url":
            continue
        url = _url(action["url"], allowed_hosts)
        aid = action["action_id"]
        rows[aid] = {"action_id": aid, "information_key": action["information_key"],
                     "url": url, "observations": []}
        by_url.setdefault(url, []).append(aid)
    for document in public.get("documents", []):
        if not isinstance(document, Mapping) or document.get("status") != "OK":
            continue
        try:
            observed, available = parse_time(document["observed_at"]), parse_time(document["available_at"])
        except (KeyError, TypeError, ValueError):
            continue
        if (observed > available or available > deadline
                or not isinstance(document.get("document_id"), str) or not document["document_id"]
                or not isinstance(document.get("url"), str) or not document["url"]
                or not re.fullmatch(r"[0-9a-f]{64}", str(document.get("raw_sha256", "")))):
            continue
        parent = {key: deepcopy(document[key]) for key in
                  ("document_id", "url", "raw_sha256", "observed_at", "available_at")}
        parent["source_published_at"] = deepcopy(document.get("source_published_at"))
        for link_index, link in enumerate(document.get("links", [])):
            if not isinstance(link, Mapping):
                continue
            try:
                url = _url(link["url"], allowed_hosts)
            except (KeyError, TypeError, ValueError):
                continue
            if url not in by_url:
                continue
            labels, source, truncated = _labels(link)
            observation = {"parent": deepcopy(parent), "link_index": link_index,
                           "label_source": source, "observed_anchors": labels,
                           "observed_anchors_truncated": truncated}
            for aid in by_url[url]:
                rows[aid]["observations"].append(deepcopy(observation))
    return {"schema_version": VERSION, "decision_at": public["decision_at"],
            "by_action_id": rows,
            "semantics": {"labels_are_source_assertions": True,
                          "contains_target_body": False, "numerical_value_assigned": False,
                          "model_features_changed": False}}


__all__ = ["VERSION", "SPEC", "build_candidate_contexts"]
