"""Opt-in source-segment evidence projection; never an extra probability model.

The caller supplies admitted, archive-verified documents. This projection
checks visible offsets and per-segment chronology. It does not certify prose
entailment, world truth, source independence or the utility of another query.
"""
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import re
from urllib.parse import urlsplit

from .audit import digest
from .inquiry_contract import utc
from .match_evidence import (
    ARTICLES_VERSION, CONTENT_VERSION, HTML_BODY_VERSION, VERSION as PREVIOUS_VERSION, _identity, _anchor, _sentences,
    _fixture_spans, _record, _mentions, _TOPIC_RE, _REPORT, _other_squads_v2, _squad_categories_v2,
    _query_terms, _opposition, _build_match_evidence, _landing_page_role_v2,
)

NEWS_VERSIONS = {"structured-news-discovery-v1", "structured-news-discovery-v2"}
HTML_NEWS_VERSION = "structured-news-discovery-v3"
TIME_FIELDS = ("source_published_at", "source_modified_at", "publication_raw", "modified_raw",
               "source_publication_date", "source_publication_local_time", "publication_timezone")


def _document_reason(document, as_of, identity):
    if document.get("status") != "OK" or document.get("kind") not in {"article", "page"}:
        return "discovery_or_unsuccessful_response_not_source_evidence"
    if not isinstance(document.get("text"), str) or not document["text"].strip():
        return "empty_text"
    if len(document["text"]) > 57600:
        raise ValueError("Document exceeds existing 24-window extraction bound")
    if not re.fullmatch(r"[0-9a-f]{64}", str(document.get("raw_sha256", ""))):
        return "missing_archive_hash"
    try:
        observed, available = utc(document["observed_at"]), utc(document["available_at"])
    except (ValueError, KeyError, TypeError, AttributeError):
        return "missing_or_invalid_operational_time"
    if observed > available:
        return "inconsistent_source_clock"
    if available > as_of:
        return "unavailable_at_decision_time"
    if available >= utc(identity["kickoff_at"]):
        return "not_acquired_before_kickoff"
    if _landing_page_role_v2(document):
        return "index_page_for_link_discovery_only"
    sections = urlsplit(document["url"]).path.split("/")[:-1]
    if _other_squads_v2(" ".join(sections), identity):
        return "non_target_squad_source_section"
    return None


def _segment_reason(segment, document, previous_end):
    if not isinstance(segment, Mapping):
        return "invalid_segment"
    start, end, text = segment.get("start"), segment.get("end"), segment.get("text")
    if (type(start) is not int or type(end) is not int or not isinstance(text, str) or not text.strip()
            or not 0 <= start < end <= len(document["text"]) or start < previous_end
            or document["text"][start:end] != text):
        return "segment_offsets_do_not_reproduce_ordered_text"
    observed = utc(document["observed_at"])
    try:
        published, modified = [utc(segment[key]) if segment.get(key) is not None else None
                               for key in ("source_published_at", "source_modified_at")]
        if any(value is not None and value > observed for value in (published, modified)):
            return "segment_source_time_after_actual_capture"
        if published is not None and modified is not None and modified < published:
            return "segment_modification_precedes_publication"
        local = segment.get("source_publication_date")
        if local is not None:
            earliest = datetime.fromisoformat(local+"T"+(segment.get("source_publication_local_time") or "00:00"))
            if earliest.tzinfo is not None:
                return "invalid_segment_source_clock"
            if earliest.replace(tzinfo=timezone.utc)-timedelta(hours=14) > observed:
                return "segment_local_publication_certainly_future"
    except (ValueError, TypeError, AttributeError):
        return "invalid_segment_source_clock"
    if segment.get("content_role") == "headline":
        return "headline_not_body_assertion"
    return None


def _body_segment_v4(segment):
    """Restrict known frozen projections to auditable body offsets.

    News v1/v2 normalizes both JSON-LD headline and body to single lines,
    joining an optional independent headline with exactly one newline. Its
    prefix can therefore be removed without guessing prose semantics. The
    semantic-HTML projection flattens headings/navigation with paragraphs;
    that format supplies no body boundary and is deliberately unsupported.
    """
    source, role = segment.get("content_source"), segment.get("content_role")
    if role is not None and (not isinstance(role, str) or role not in {"paragraph", "body"}):
        return None, "non_body_segment_role", None
    if source == "json_ld_article_body":
        pointer = segment.get("json_pointer")
        if not isinstance(pointer, str) or not pointer.startswith("script[") or not pointer.endswith(".articleBody"):
            return None, "jsonld_body_pointer_required", None
        parts = segment["text"].split("\n")
        if len(parts) > 2:
            return None, "unknown_jsonld_segment_boundary_format", None
        if len(parts) == 1:
            if segment.get("truncated"):
                return None, "truncated_jsonld_body_boundary_unknown", None
            return segment, None, None
        start = segment["start"]+len(parts[0])+1
        headline = {"start": segment["start"], "end": start-1,
                    "reason": "projector_headline_prefix_not_body_assertion"}
        if not parts[1].strip():
            return None, "jsonld_body_empty_after_headline", headline
        return {**segment, "start": start, "text": parts[1]}, None, headline
    if source == "jalgpall_news_detail_html":
        if role != "paragraph" or segment.get("css_selector") != "div.news-detail > div.body.def p":
            return None, "source_html_body_paragraph_marker_required", None
        return segment, None, None
    if source == "semantic_html_article":
        return None, "flattened_html_body_boundaries_unavailable", None
    return None, "unregistered_body_segment_source", None


def _body_segment_v5(segment, document, news):
    if segment.get("content_source") == "cms_literal_article_paragraph":
        binding = segment.get("body_binding", {})
        pointer = segment.get("json_pointer")
        index = segment.get("literal_paragraph_index")
        if (news.get("version") != HTML_NEWS_VERSION
                or document.get("news_projection_version") not in {"structured-news-discovery-v12", "structured-news-discovery-v13"}
                or segment.get("content_role") != "paragraph"
                or segment.get("source_url") != document["url"]
                or urlsplit(document["url"]).hostname != "www.fulhamfc.com"
                or not isinstance(pointer, str)
                or not re.fullmatch(r"data\[0\]\.data\.content\[\d+\]\.rowData\.widgetData\.content", pointer)
                or type(index) is not int or not 0 <= index < 1024
                or not isinstance(binding, Mapping)
                or binding.get("version") != "official-cms-response-projection-v1"
                or binding.get("javascript_executed") is not False):
            return None, "typed_CMS_literal_body_provenance_required", None
        return segment, None, None
    if segment.get("content_source") != "semantic_html_article_paragraph":
        return _body_segment_v4(segment)
    if news.get("version") != HTML_NEWS_VERSION:
        return None, "typed_html_paragraph_requires_news_v3", None
    selector, index = segment.get("css_selector"), segment.get("original_dom_paragraph_index")
    if (segment.get("content_role") != "paragraph" or not isinstance(selector, str)
            or not selector.strip() or len(selector) > 512 or type(index) is not int or not 0 <= index < 1024
            or segment.get("source_url") != document["url"]):
        return None, "typed_html_body_paragraph_provenance_required", None
    return segment, None, None


def _html_lead_scope(segment, document, identity):
    """A source's explicit lead subject bounds generic later team references.

    Mixed squads or an explicit current fixture in the lead remain unresolved;
    names, URL keywords and presumed information utility are not classifiers.
    """
    text = segment["text"]
    other = _other_squads_v2(text, identity)
    if not other:
        return None
    target = set().union(*(_squad_categories_v2(name)
                          for names in identity["aliases"].values() for name in names))
    if any(re.fullmatch(r"u\d{1,2}", item) for item in target):
        target.add("youth")
    mixed = bool(target & _squad_categories_v2(text))
    if not target and re.search(r"\bmen(?:['’]s)?\b|男足|男子", text, re.I):
        mixed = True
    bindings = _fixture_spans({**document, "text": text}, identity)
    if mixed or any(row["status"] == "explicit_pair_and_date" for row in bindings):
        return {"status": "mixed_or_explicit_target_reference_requires_per_passage_review",
                "article_wide_exclusion": False}
    return {"status": "non_target_squad_source_lead", "source_squad_categories": other,
        "anchor": _anchor(document, segment["start"], segment["end"]),
        "article_wide_exclusion": True, "semantic_scope_verified": False,
        "reason": "source_article_subject_not_bound_to_target_squad"}


def _quote_records(document, identity, news, review, as_of, *, version=ARTICLES_VERSION):
    previous_end = 0
    html_lead_seen, html_scope = False, None
    for index, segment in enumerate(news["segments"]):
        reason = _segment_reason(segment, document, previous_end)
        if isinstance(segment, Mapping) and type(segment.get("end")) is int:
            previous_end = max(previous_end, segment["end"])
        if not reason and version in {CONTENT_VERSION, HTML_BODY_VERSION}:
            original = segment
            segment, reason, headline = (_body_segment_v5(segment, document, news)
                if version == HTML_BODY_VERSION else _body_segment_v4(segment))
            if headline is not None:
                review["excluded_segments"].append({"segment_index": index, **headline})
            if reason:
                segment = original
        if reason:
            # Rejected source text is never copied into the actor-facing
            # audit, query hooks or cross-document relationships.
            review["excluded_segments"].append({"segment_index": index, "reason": reason,
                "start": segment.get("start") if isinstance(segment, Mapping) else None,
                "end": segment.get("end") if isinstance(segment, Mapping) else None})
            continue
        typed_html = version == HTML_BODY_VERSION and segment.get("content_source") == "semantic_html_article_paragraph"
        if typed_html and not html_lead_seen:
            html_lead_seen = True
            html_scope = _html_lead_scope(segment, document, identity)
            if html_scope is not None:
                review["article_subject_scope"] = html_scope
        segment_document = {**document, "text": segment["text"]}
        bindings = []
        for binding in _fixture_spans(segment_document, identity):
            anchor = binding["anchor"]
            bindings.append({**binding, "anchor": _anchor(document,
                segment["start"]+anchor["start"], segment["start"]+anchor["end"])})
        review["fixture_bindings"].extend(bindings)
        fragments = list(_sentences(segment["text"]))
        if not fragments:
            fragments = [(0, len(segment["text"]), True)]
        for left, right, partial in fragments:
            start, end = segment["start"]+left, segment["start"]+right
            quote = document["text"][start:end]
            topics = [key for key, pattern in _TOPIC_RE.items() if pattern.search(quote)]
            local_review = {"fixture_bindings": bindings}
            record = _record(document, identity, local_review, start, end, topics or ["unclassified_source_text"], partial=partial)
            # A same-segment fixture line does not grant fixture scope to a
            # separate sentence; _record requires the binding inside its quote.
            if any(b["status"] == "explicit_pair_and_date" and start <= b["anchor"]["start"]
                   and b["anchor"]["end"] <= end for b in bindings):
                record["topics"] = sorted(set(record["topics"]+["fixture_confirmation"]))
            if not record["mentioned_team_sides"]:
                record["applicability"] = "unscoped_source_text"
            if _other_squads_v2(quote, identity):
                record["applicability"] = "other_squad_source_text"
            if any(b["status"] == "different_fixture_date" and start <= b["anchor"]["start"]
                   and b["anchor"]["end"] <= end for b in bindings):
                record["applicability"] = "different_fixture_source_text"
            if _REPORT.search(quote) and len(_mentions(quote, identity)) == 2:
                record["applicability"] = "retrospective_source_text"
            if typed_html and html_scope is not None and html_scope["article_wide_exclusion"]:
                record["applicability"] = "source_article_subject_requires_match_binding"
            record.update({key: deepcopy(segment.get(key)) for key in TIME_FIELDS})
            record.update(extracted_at=as_of.isoformat(), source_language=news.get("source_language"),
                publication_status="segment_timezone_unknown" if segment.get("publication_timezone") == "unknown"
                    else "source_segment_metadata" if segment.get("source_published_at") else "unknown",
                source_segment={"index": index, "start": segment["start"], "end": segment["end"],
                    "json_pointer": segment.get("json_pointer"), "css_selector": segment.get("css_selector"),
                    "content_source": segment.get("content_source"), "source_url_assertion": segment.get("source_url"),
                    "truncated": bool(segment.get("truncated"))},
                source_version={"raw_sha256": document["raw_sha256"], "text_sha256": record["anchor"]["text_sha256"],
                    "news_projection_version": news["version"]},
                assertion_kind="quoted_source_passage", normalized_predicate=None, effective_at=None,
                semantic_verification="exact_segment_quote_only_meaning_truth_and_applicability_unverified")
            if typed_html:
                record["source_segment"]["original_dom_paragraph_index"] = segment["original_dom_paragraph_index"]
            record["evidence_id"] = "evidence-"+digest({"projection": version,
                **{key: value for key, value in record.items() if key not in {"evidence_id", "extracted_at"}}})[:24]
            yield record


def _relations(records):
    copies, versions, seen = [], [], set()
    for index, left in enumerate(records):
        for right in records[index+1:]:
            la, ra = left["anchor"], right["anchor"]
            if la["document_id"] == ra["document_id"]:
                continue
            if left["statement"] == right["statement"]:
                copies.append({"evidence_ids": sorted([left["evidence_id"], right["evidence_id"]]),
                    "kind": "exact_quoted_text_match", "independent_corroboration": None,
                    "semantic_equivalence_verified": False, "expected_gain": None})
            key = tuple(sorted([la["document_id"], ra["document_id"]]))
            if la["url"] == ra["url"] and la["raw_sha256"] != ra["raw_sha256"] and key not in seen:
                seen.add(key)
                versions.append({"document_ids": list(key), "url": la["url"],
                    "kind": "same_request_url_different_archived_bytes",
                    "captures": sorted([{"document_id": row["anchor"]["document_id"],
                        "observed_at": row["observed_at"], "available_at": row["available_at"],
                        "raw_sha256": row["anchor"]["raw_sha256"]} for row in (left, right)],
                        key=lambda row: (utc(row["observed_at"]), row["document_id"])),
                    "semantic_revision_or_supersession": "unknown", "expected_gain": None})
    return copies, versions


def _followups(evidence, conflicts, identity):
    unresolved, proposals = [], {}
    def propose(records, reasons):
        hook = _query_terms(records[0]["statement"], identity)
        if not hook:
            return
        query = f'{identity["home_team"]} {identity["away_team"]} {identity["kickoff_at"][:10]} {hook}'
        key = "evidence-query-"+digest(query)[:20]
        item = proposals.setdefault(key, {"candidate_id": key, "tool": "search_web", "query": query,
            "based_on_evidence_ids": [], "reason": "; ".join(reasons), "expected_gain": None,
            "proposal_status": "grounded_query_not_value_ranked"})
        item["based_on_evidence_ids"] = sorted(set(item["based_on_evidence_ids"]+[r["evidence_id"] for r in records]))
    for row in evidence:
        reasons = []
        if row["modality"] == "uncertain_source_statement":
            reasons.append("source_explicitly_leaves_status_unresolved")
        if row["applicability"] == "team_context_requires_match_binding":
            reasons.append("applicability_to_this_fixture_not_established")
        if reasons:
            unresolved.append({"evidence_id": row["evidence_id"], "reasons": reasons, "value_of_resolution": None})
            propose([row], reasons)
    by_id = {r["evidence_id"]: r for r in evidence}
    for conflict in conflicts:
        reasons = ["opposed_source_statements_need_scope_time_or_source_review"]
        unresolved.append({"conflict_id": conflict["conflict_id"], "evidence_ids": conflict["evidence_ids"],
                           "reasons": reasons, "value_of_resolution": None})
        propose([by_id[key] for key in conflict["evidence_ids"]], reasons)
    if not evidence:
        unresolved.append({"reasons": ["no_substantive_match_evidence_observed"], "value_of_resolution": None})
    return unresolved, list(proposals.values())


def build_article_evidence(match, documents, *, as_of=None, max_evidence=64, version=ARTICLES_VERSION):
    if version not in {ARTICLES_VERSION, CONTENT_VERSION, HTML_BODY_VERSION}:
        raise ValueError("Unsupported structured article evidence version")
    news_versions = NEWS_VERSIONS | ({HTML_NEWS_VERSION} if version == HTML_BODY_VERSION else set())
    identity = _identity(match)
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)) or len(documents) > 128:
        raise ValueError("documents must be at most 128 already acquired documents")
    if type(max_evidence) is not int or not 1 <= max_evidence <= 128:
        raise ValueError("max_evidence must be 1..128")
    selected_time = utc(as_of if as_of is not None else identity["prediction_cutoff"])
    if selected_time > utc(identity["prediction_cutoff"]):
        raise ValueError("Evidence as_of must not follow prediction cutoff")
    evidence, unscoped, reviews, fingerprints = [], [], [], {}
    truncated = False
    for document in documents:
        if not isinstance(document, Mapping) or not isinstance(document.get("document_id"), str) or not document["document_id"]:
            raise ValueError("A document with an explicit ID is required")
        did, fingerprint = document["document_id"], digest(document)
        if did in fingerprints:
            if fingerprints[did] != fingerprint:
                raise ValueError("One document ID refers to different acquired content")
            continue
        fingerprints[did] = fingerprint
        news = document.get("structured_news", {})
        if version in {CONTENT_VERSION, HTML_BODY_VERSION}:
            reason = None
            if (not isinstance(news, Mapping) or not isinstance(news.get("version"), str)
                    or news["version"] not in news_versions):
                reason = "registered_structured_body_projection_required"
            elif not isinstance(news.get("role"), str) or news["role"] not in {"article", "liveblog"}:
                reason = "structured_role_not_source_body"
            if reason:
                reviews.append({"document_id": did, "status": "excluded", "reason": reason,
                    "fixture_bindings": [], "excluded_segments": [], "extracted_evidence_ids": []})
                continue
        if "structured_news" in document and (not isinstance(news, Mapping) or news.get("version") not in news_versions):
            reviews.append({"document_id": did, "status": "excluded", "reason": "unsupported_structured_news_version",
                            "fixture_bindings": [], "extracted_evidence_ids": []})
            continue
        if not isinstance(news, Mapping) or news.get("version") not in news_versions or news.get("role") not in {"article", "liveblog"}:
            old = _build_match_evidence(match, [document], as_of=selected_time,
                max_evidence=max_evidence, version=PREVIOUS_VERSION)
            reviews.extend(old["document_reviews"])
            candidates = old["evidence"]
        else:
            reason = _document_reason(document, selected_time, identity)
            if not isinstance(news.get("segments"), list) or not news["segments"] or len(news["segments"]) > 64:
                reason = reason or "bounded_structured_article_segments_required"
            review = {"document_id": did, "status": "excluded" if reason else "eligible_for_extraction",
                "reason": reason or "verified_offsets_source_semantics_unknown", "fixture_bindings": [],
                "excluded_segments": [], "extracted_evidence_ids": []}
            reviews.append(review)
            candidates = [] if reason else _quote_records(document, identity, news, review, selected_time, version=version)
        for record in candidates:
            if len(evidence)+len(unscoped) >= max_evidence:
                truncated = True
                break
            target = record["applicability"] in {"match_scoped_source_assertion", "team_context_requires_match_binding"}
            (evidence if target else unscoped).append(record)
            if "excluded_segments" in reviews[-1]:
                reviews[-1]["extracted_evidence_ids"].append(record["evidence_id"])
    conflicts = _opposition(evidence)
    for conflict in conflicts:
        conflict["effective_time_overlap"] = "unknown_not_resolved_by_publication_order"
    unresolved, followups = _followups(evidence, conflicts, identity)
    copies, versions = _relations(evidence+unscoped)
    result = {"schema_version": version, "match_id": identity["match_id"], "identity": identity,
        "as_of": selected_time.isoformat(), "evidence": evidence, "unscoped_source_quotes": unscoped,
        "document_reviews": reviews,
        "state": {"present_topics": sorted({t for r in evidence if r["applicability"] == "match_scoped_source_assertion" for t in r["topics"]}),
            "context_topics": sorted({t for r in evidence if r["applicability"] != "match_scoped_source_assertion" for t in r["topics"]}),
            "unresolved": unresolved, "conflict_signals": conflicts, "followup_candidates": followups,
            "exact_copy_signals": copies, "source_version_relations": versions,
            "coverage_complete": False, "extraction_truncated": truncated},
        "semantics": {"method": "exact_visible_source_segment_quotes_and_bounded_lexical_signals", "llm_calls": 0,
            "archive_bytes_verified_here": False, "requires_already_admitted_documents": True,
            "publication_is_effective_time": False, "semantic_entailment_verified": False,
            "source_independence_established": False, "numeric_voi": None, "probability_update": None,
            "fixed_information_checklist": False, "topics_are_value_weights": False,
            "missing_evidence_is_negative_answer": False, "followups_are_mandatory": False,
            "historical_actor_extraction_established_by_reprocessing": False,
            "projection_scope": "quoted_evidence_and_candidate_generation_not_frozen_forecast_state"}}
    if version in {CONTENT_VERSION, HTML_BODY_VERSION}:
        result["semantics"].update(full_page_fallback_enabled=False,
            content_role_policy="registered_exact_body_segments_only_headlines_and_navigation_excluded",
            body_sources=["json_ld_article_body", "jalgpall_news_detail_html_paragraph"],
            flattened_html_article_body_supported=False,
            document_links_changed=False)
        if version == HTML_BODY_VERSION:
            result["semantics"]["body_sources"].append("semantic_html_article_paragraph")
            result["semantics"]["typed_html_subject_scope"] = "explicit_admitted_lead_squad_guard_mixed_or_fixture_bound_lead_unresolved"
    return result
