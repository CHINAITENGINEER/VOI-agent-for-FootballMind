"""Lossless candidate-label handoff from the saved pre-decision state only.

This module does not read receipts or a target's subsequently revealed body.
Internal parent projection timestamps are retained under their original meaning;
the menu is the proof that these labels were visible by this decision.
"""
from copy import deepcopy
from collections.abc import Mapping

VERSION = "observed-pre-read-metadata-v1"


def candidate_pre_read_metadata(cycle, action):
    public = cycle.get("public_before")
    public = public if isinstance(public, Mapping) else {}
    contexts = public.get("candidate_contexts")
    contexts = contexts if isinstance(contexts, Mapping) else {}
    by_action = contexts.get("by_action_id")
    by_action = by_action if isinstance(by_action, Mapping) else {}
    recorded = by_action.get(action.get("action_id"))
    documents = public.get("documents")
    observations, labels = [], []
    for document_index, document in enumerate(documents if isinstance(documents, list) else []):
        if not isinstance(document, Mapping) or not isinstance(document.get("links"), list):
            continue
        for link_index, link in enumerate(document["links"]):
            if (not isinstance(link, Mapping) or not action.get("url")
                    or link.get("url") != action["url"]):
                continue
            parent = {key: deepcopy(document.get(key)) for key in
                      ("document_id", "url", "raw_sha256", "observed_at", "raw_path")}
            parent["projector_available_at_as_recorded"] = document.get("available_at")
            observation = {"document_index": document_index, "link_index": link_index,
                "parent": parent, "link_metadata": deepcopy(dict(link)),
                "basis": "public_before.documents.links", "target_body_used": False}
            observations.append(observation)
            for field in ("title", "snippet"):
                if isinstance(link.get(field), str) and link[field].strip():
                    labels.append({"kind": field, "text": link[field],
                        "parent_document_id": document.get("document_id"),
                        "link_index": link_index, "basis": "public_before.documents.links." + field})
            for anchor in link.get("observed_anchors") or []:
                if isinstance(anchor, Mapping) and isinstance(anchor.get("text"), str) and anchor["text"].strip():
                    labels.append({"kind": "anchor", "text": anchor["text"],
                        "parent_document_id": document.get("document_id"), "link_index": link_index,
                        "anchor_index": anchor.get("anchor_index"), "basis": "public_before.documents.links.observed_anchors"})
    # Some runtime versions retained contexts but omitted the full link array.
    # Preserve that evidence without pretending it is a separate exposure.
    if not observations and isinstance(recorded, Mapping):
        for item in recorded.get("observations") or []:
            if not isinstance(item, Mapping):
                continue
            for anchor in item.get("observed_anchors") or []:
                if isinstance(anchor, Mapping) and isinstance(anchor.get("text"), str) and anchor["text"].strip():
                    labels.append({"kind": "anchor", "text": anchor["text"],
                        "parent_document_id": (item.get("parent") or {}).get("document_id"),
                        "link_index": item.get("link_index"), "anchor_index": anchor.get("anchor_index"),
                        "basis": "public_before.candidate_contexts.by_action_id.observations"})
    present = bool(observations or isinstance(recorded, Mapping))
    return {"observed_labels": labels, "pre_read_metadata": {
        "schema_version": VERSION, "status": "recorded" if present else "not_recorded",
        "candidate_context_as_recorded": deepcopy(recorded) if isinstance(recorded, Mapping) else None,
        "observations": observations, "visible_by_decision_at": cycle.get("decision_at") or public.get("decision_at"),
        "visibility_basis": "saved_predecision_public_state" if present else None,
        "menu_raw_path": cycle.get("menu_raw_path") or cycle.get("raw_menu_path"),
        "menu_raw_sha256": cycle.get("menu_raw_sha256") or cycle.get("raw_menu_sha256"),
        "target_body_used": False, "f_consumed": None, "f_consumption_basis": "not_recorded_by_metadata_projection",
        "interpretation": "Labels and snippets belong to already-visible input; reading the body must not count them again as new information."}}
