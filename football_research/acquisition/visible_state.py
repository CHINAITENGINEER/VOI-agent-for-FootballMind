"""Deterministic structural input from an already-admitted public snapshot.

No model, HTTP, archive lookup or semantic evaluator lives here. The caller
must supply the real runtime public snapshot and generated menu. We verify
internal source/time/offset bindings, not the truth of an arbitrary caller's
claimed raw digest or capture clock. Extra hidden payloads are never traversed.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import re
from urllib.parse import urlsplit

from football_agent.candidate_context import _labels, build_candidate_contexts, VERSION as CONTEXT_VERSION
from football_agent.learned_voi_runtime import _actions
from football_agent.learned_voi_web import _url
from football_agent.match_evidence_articles import _segment_reason, _body_segment_v5, TIME_FIELDS
from .joint_planner import _budget, _cost
from .state import parse_time
from .text_returns import _p

VERSION = "visible-information-state-v1"
_NEWS = {"structured-news-discovery-v1", "structured-news-discovery-v2", "structured-news-discovery-v3"}
_CLOCKS = ("observed_at", "available_at", "source_published_at", "source_modified_at")


def _url_identity(value):
    return _url(value, [urlsplit(value).hostname])


def _document_reason(document, deadline):
    if (not isinstance(document.get("document_id"), str) or not document["document_id"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(document.get("raw_sha256", "")))):
        return "missing_source_binding"
    try:
        _url_identity(document["url"])
        observed, available = parse_time(document["observed_at"]), parse_time(document["available_at"])
        if observed > available:
            return "inconsistent_operational_clocks"
        if available > deadline:
            return "not_available_before_decision_cutoff"
        published, modified = [parse_time(document[key]) if document.get(key) is not None else None
                               for key in ("source_published_at", "source_modified_at")]
        if any(stamp is not None and stamp > observed for stamp in (published, modified)):
            return "source_time_after_capture"
        if published is not None and modified is not None and modified < published:
            return "source_modification_precedes_publication"
    except (KeyError, TypeError, ValueError, AttributeError):
        return "invalid_source_url_or_clock"
    return None


def _kind(document):
    if document.get("status") == "FAILED":
        return "failure"
    if document.get("status") == "NO_DATA":
        return "no_data"
    if document.get("status") != "OK":
        return "unknown"
    news = document.get("structured_news", {})
    if (isinstance(news, Mapping) and news.get("version") in _NEWS
            and news.get("role") in {"article", "liveblog"}):
        return news["role"]
    if document.get("kind") in {"index", "discovery", "search"}:
        return "discovery" if document["kind"] == "search" else document["kind"]
    return "unknown"


def _history(state, public):
    # Runtime action_history contains IDs, not original response clocks/body.
    # Keep order and multiplicity; neither sort nor collapse repeated attempts.
    actions, attempts = [], []
    for item in state.get("action_history", []):
        aid = item.get("action_id") if isinstance(item, Mapping) else item
        if not isinstance(aid, str) or not aid:
            raise ValueError("An ordered runtime action identity is required")
        actions.append({"action_id": aid})
    for item in public.get("attempts", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("action_id"), str):
            raise ValueError("An ordered runtime attempt is required")
        status = item.get("status")
        attempts.append({key: deepcopy(item.get(key)) for key in
                         ("action_id", "information_key", "status", "admission")})
        attempts[-1]["mask"] = {"failure": status == "FAILED", "no_data": status == "NO_DATA",
                                "unknown_status": status not in {"OK", "FAILED", "NO_DATA"}}
    return {"action_history": actions, "attempts": attempts,
            "order_matches_attempts": [a["action_id"] for a in actions] == [a["action_id"] for a in attempts],
            "clock_scope": "attempt_receipt_clocks_not_present_in_public_snapshot_not_imputed"}


def encode_visible_state(public, actions=()):
    """Return ``{schema_version, before, candidates, audit}``, without fitting.

    ``before`` is independent of the supplied candidate subset: all observed
    parent link labels already belong to known information. Candidate inputs
    contain only request/cost/mask and references to those labels, never a body
    or candidate_contexts supplied by an unverifiable parent. Exact read text
    is stored once; every original document occurrence/span remains referenced.
    Extra top-level outcomes/future_returns and unknown nested fields are not
    encoded. Forged documents masquerading as admitted public inputs require
    the existing raw gateway; this pure function cannot authenticate them.
    """
    state = public["state"]
    now, cutoff = parse_time(public["decision_at"]), parse_time(state["prediction_cutoff"])
    deadline = min(now, cutoff)
    menu = _actions(list(actions))
    prediction = public.get("prediction", {})
    current = _p(prediction["probabilities"]) if prediction.get("probabilities") is not None else None
    prior, prior_basis = None, "not_supplied_no_prior_inferred_from_current_prediction"
    text_state = state.get("open_text", {})
    if text_state.get("prior_probabilities") is not None:
        prior, prior_basis = _p(text_state["prior_probabilities"]), "explicit_open_text_initial_prior"
    elif prediction.get("probability_basis") == "observed_market_snapshot":
        prior, prior_basis = deepcopy(current), "observed_market_snapshot"
    history = _history(state, public)
    before = {"match_id": state["match_id"], "decision_at": now.isoformat(),
              "prediction_cutoff": cutoff.isoformat(),
              "fixture": {key: deepcopy(state.get("base", {}).get(key)) for key in
                          ("home_team_id", "away_team_id", "competition_id", "kickoff")},
              "current_probabilities": current, "prior_probabilities": prior, "prior_basis": prior_basis,
              "prediction_model_version": prediction.get("model_version"), "history": history,
              "documents": [], "contents": [], "observed_links": []}
    if "budget" in public:
        limits, spent = _budget(public["budget"])
        before["budget"] = {"limits": limits, "spent": spent}
    audit = {"excluded_documents": [], "excluded_segments": [],
             "source_validation": "internal_bindings_only_caller_must_verify_original_archive",
             "semantic_scope": "source_text_and_roles_not_match_truth_or_utility",
             "history_scope": "complete_public_order_preserved_not_receipt_history_reconstruction"}
    content_ids, visible_docs, links_by_url = set(), [], {}

    def add_span(document, record, segment, role, segment_index):
        text = segment["text"]
        ident = "text-" + sha256(text.encode("utf-8")).hexdigest()
        if ident not in content_ids:
            before["contents"].append({"content_id": ident, "text": text})
            content_ids.add(ident)
        record["content_refs"].append({"content_id": ident, "role": role,
            "segment_index": segment_index, "start": segment["start"], "end": segment["end"],
            **{key: deepcopy(segment.get(key)) for key in TIME_FIELDS},
            **{key: deepcopy(segment.get(key)) for key in
               ("content_source", "json_pointer", "css_selector", "original_dom_paragraph_index")}})

    for index, document in enumerate(public.get("documents", [])):
        if not isinstance(document, Mapping):
            audit["excluded_documents"].append({"document_index": index, "reason": "invalid_document"})
            continue
        reason = _document_reason(document, deadline)
        if reason:
            # Future/invalid content never leaks through diagnostic quotations.
            audit["excluded_documents"].append({"document_index": index, "reason": reason})
            continue
        role = _kind(document)
        record = {"document_index": index, **{key: deepcopy(document.get(key)) for key in
            ("document_id", "raw_sha256", "url", "status", "kind", "news_projection_version") + _CLOCKS},
            "source_role": role, "content_refs": [], "observed_link_refs": [],
            "mask": {"failure": role == "failure", "no_data": role == "no_data", "unknown_role": role == "unknown"}}
        before["documents"].append(record)
        news = document.get("structured_news", {})
        if role in {"article", "liveblog"}:
            end = 0
            for si, segment in enumerate(news.get("segments", [])):
                reason = _segment_reason(segment, document, end)
                if isinstance(segment, Mapping) and type(segment.get("end")) is int:
                    end = max(end, segment["end"])
                body, headline = None, None
                if not reason:
                    body, reason, headline = _body_segment_v5(segment, document, news)
                if reason:
                    audit["excluded_segments"].append({"document_index": index, "segment_index": si, "reason": reason})
                    continue
                if headline is not None and headline["end"] > headline["start"]:
                    add_span(document, record, {**segment, "start": headline["start"], "end": headline["end"],
                        "text": document["text"][headline["start"]:headline["end"]]}, "headline", si)
                add_span(document, record, body, "body", si)
        record["mask"]["body_unavailable"] = not any(ref["role"] == "body" for ref in record["content_refs"])
        if document.get("status") != "OK":
            continue
        visible_docs.append(document)
        for li, link in enumerate(document.get("links", [])):
            try:
                target = _url_identity(link["url"])
                labels, source, truncated = _labels(link)
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            ref = f"document-{index}:link-{li}"
            before["observed_links"].append({"link_ref": ref, "document_index": index,
                "target_url": target, "label_source": source, "observed_anchors": labels,
                "observed_anchors_truncated": truncated})
            record["observed_link_refs"].append(ref)
            links_by_url.setdefault(target, []).append(ref)

    # The R192 sidecar is an acceleration/audit binding, never a source of new
    # article text or labels. Recompute its current-menu rows from known docs.
    saved = public.get("candidate_contexts")
    if saved is not None:
        if (saved.get("schema_version") != CONTEXT_VERSION or
                parse_time(saved["decision_at"]) != now):
            raise ValueError("Candidate context schema/decision clock differs from public state")
        hosts = {urlsplit(a["url"]).hostname for a in menu if a.get("tool") == "read_url"}
        recomputed = build_candidate_contexts({**public, "documents": visible_docs}, menu, allowed_hosts=hosts)
        for aid, row in recomputed["by_action_id"].items():
            if saved.get("by_action_id", {}).get(aid) != row:
                raise ValueError("Candidate labels or parent source binding differ from known documents")
    candidates = {}
    attempted = {entry["action_id"] for entry in history["action_history"]}
    for action in menu:
        tool = action.get("tool")
        if tool not in {"read_url", "search_web"}:
            raise ValueError("Visible text input supports the existing web menu only")
        request = ({"tool": tool, "url": _url_identity(action["url"])} if tool == "read_url" else
                   {"tool": tool, "query": action["query"], "source": action.get("source", "open_web")})
        candidates[action["action_id"]] = {"action_id": action["action_id"],
            "information_key": action["information_key"], "request": request,
            "cost_status": action["cost_status"], "declared_cost": _cost(action),
            "available": action.get("available", True), "already_attempted": action["action_id"] in attempted,
            "observed_link_refs": deepcopy(links_by_url.get(request.get("url"), []))}
    return {"schema_version": VERSION, "before": before, "candidates": candidates, "audit": audit}


__all__ = ["VERSION", "encode_visible_state"]
