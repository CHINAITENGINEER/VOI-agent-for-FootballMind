"""Ground match evidence in acquired text without inventing a return-value model.

This is a deterministic extraction adapter, not an LLM or semantic verifier.
It quotes what a source says, keeps operational time separate from publication
and effective time, and proposes follow-ups only for observed gaps/signals.
Topic labels organize quoted passages; they are not importance weights.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date
from hashlib import sha256
import re
from urllib.parse import urlsplit

from .audit import digest
from .inquiry_contract import utc

LEGACY_VERSION = "match-evidence-extraction-v1"
VERSION = "match-evidence-extraction-v2"
ARTICLES_VERSION = "match-evidence-extraction-v3"
CONTENT_VERSION = "match-evidence-extraction-v4"
HTML_BODY_VERSION = "match-evidence-extraction-v5"
SUPPORTED_VERSIONS = (LEGACY_VERSION, VERSION, ARTICLES_VERSION, CONTENT_VERSION, HTML_BODY_VERSION)
_TOPICS = {
    "availability": r"\b(?:injur\w*|fitness|withdraw\w*|hamstring|quad|muscle|sidelined|unavailable|ruled out)\b|伤病|伤停|缺阵|伤情",
    "lineup": r"\b(?:starting (?:lineup|eleven|XI)|line[- ]?up|will (?:not )?start|first[- ]choice)\b|首发|出场名单|替补名单",
    "rotation": r"\b(?:rotat\w*|rest(?:ing|ed)? (?:players|starters)|make changes)\b|轮换|轮休",
    "qualification": r"\b(?:qualif\w*|already through|advance[ds]?|top (?:of the )?group)\b|出线|晋级|小组第一",
    "tactics": r"\b(?:tactic\w*|formation|defensive|attacking|low block|high press)\b|战术|防守|进攻|阵型",
}
_TOPIC_RE = {key: re.compile(value, re.I) for key, value in _TOPICS.items()}
_UNCERTAIN = re.compile(r"\b(?:may|might|could|uncertain|doubt\w*|unconfirmed|reportedly|reports? (?:suggest|claim)|no official update|yet to be confirmed|subject to change)\b|可能|尚未|待定|存疑|据报道", re.I)
_NEGATION = re.compile(r"\b(?:not|never|no)\b|不会|没有|并非", re.I)
_REPORT = re.compile(r"\b(?:full[- ]time|match report|post[- ]match|final score)\b|赛后|全场比分|完场", re.I)
_OTHER_CATEGORY = re.compile(r"^\s+(?:Women\b|Women's\b|Ladies\b|U[- ]?\d{2}\b|Under[- ]?\d{2}\b|Reserves\b|Youth\b|II\b)|^女足|^青年队", re.I)
_VERB = re.compile(r"\b(?:is|are|was|were|has|have|had|will|would|may|might|could|can|confirmed|announced|returned|withdrew|face|faces)\b|将|确认|宣布|受伤|缺阵|出线|晋级|可能|采取|变阵", re.I)
_MONTHS = "January February March April May June July August September October November December".split()
_MONTH_NUMBERS = {name.casefold(): i for i, name in enumerate(_MONTHS, 1)}
_MONTH_NUMBERS.update({name[:3].casefold(): i for i, name in enumerate(_MONTHS, 1)})
_MONTH = "(?:" + "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True)) + ")"
_DATE_PATTERNS = [
    re.compile(r"(?P<year>20\d{2})[-/年](?P<month>\d{1,2})[-/月](?P<day>\d{1,2})(?:日)?"),
    re.compile(r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>" + _MONTH + r")\s*,?\s*(?P<year>20\d{2})", re.I),
    re.compile(r"(?P<month>" + _MONTH + r")\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(?P<year>20\d{2})", re.I),
]


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be nonempty text")
    return value.strip()


def _identity(match):
    if not isinstance(match, Mapping):
        raise ValueError("match must be a mapping")
    result = {"match_id": _text(match.get("match_id", match.get("id")), "match_id")}
    aliases = {}
    for side in ("home", "away"):
        value = match.get(side + "_team", match.get(side + "_name"))
        extra = match.get(side + "_aliases", [])
        if isinstance(value, Mapping):
            extra = list(extra) + list(value.get("aliases", []))
            value = value.get("name", value.get("canonical_name"))
        name = _text(value, side + "_team")
        result[side + "_team"] = name
        aliases[side] = sorted({name, *(_text(a, side + "_alias") for a in extra)}, key=lambda a: (-len(a), a))
    if set(a.casefold() for a in aliases["home"]) & set(a.casefold() for a in aliases["away"]):
        raise ValueError("Fixture team aliases must be distinct")
    result["kickoff_at"] = utc(match.get("kickoff_at", match.get("kickoff"))).isoformat()
    result["prediction_cutoff"] = utc(match["prediction_cutoff"]).isoformat()
    if utc(result["prediction_cutoff"]) > utc(result["kickoff_at"]):
        raise ValueError("Pre-match prediction cutoff must not follow kickoff")
    result["aliases"] = aliases
    return result


def _name_pattern(names):
    parts = []
    for name in names:
        # CJK clauses have no word separators: 阿森纳可能轮换 contains a
        # legitimate full team alias followed immediately by a predicate.
        boundary = r"[A-Za-z0-9_]" if re.search(r"[\u3400-\u9fff]", name) else r"\w"
        parts.append(r"(?<!" + boundary + ")" + re.escape(name) + r"(?!" + boundary + ")")
    return "(?:" + "|".join(parts) + ")"


def _mentions(text, identity):
    found = []
    for side, names in identity["aliases"].items():
        for m in re.finditer(_name_pattern(names), text, re.I):
            # Arsenal Women/U21 is not silently bound to Arsenal's men's team.
            if not _OTHER_CATEGORY.search(text[m.end():m.end() + 24]):
                found.append(side)
                break
    return found


def _dates(text):
    found = []
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group("month")
            month = int(raw) if raw.isdigit() else _MONTH_NUMBERS[raw.casefold()]
            try:
                parsed = date(int(m.group("year")), month, int(m.group("day"))).isoformat()
            except ValueError:
                continue
            found.append((m.start(), m.end(), parsed))
    return sorted(set(found))


def _anchor(document, start, end):
    quote = document["text"][start:end]
    return {"document_id": document["document_id"], "source_id": document.get("source_id") or urlsplit(document["url"]).hostname,
            "url": document["url"], "raw_sha256": document["raw_sha256"],
            "text_sha256": sha256(document["text"].encode("utf-8")).hexdigest(),
            "start": start, "end": end, "quote": quote,
            "offset_unit": "zero_based_unicode_code_points_end_exclusive",
            "verification": "exact_quote_location_only"}


def _fixture_spans(document, identity):
    text = document["text"]
    left, right = (_name_pattern(identity["aliases"][side]) for side in ("home", "away"))
    connector = r"\s+(?:v(?:s\.?)?|versus|against|host(?:s)?|welcome(?:s)?|face(?:s)?|play(?:s)?|对阵|迎战)\s+"
    patterns = [left + connector + right, right + connector + left]
    target_date = identity["kickoff_at"][:10]
    spans = []
    for pattern in patterns:
        for pair in re.finditer(pattern, text, re.I):
            if set(_mentions(pair.group(), identity)) != {"home", "away"}:
                continue
            if _OTHER_CATEGORY.search(text[pair.end():pair.end() + 24]):
                continue
            # Bind only to a nearby date after this literal fixture, not a date
            # elsewhere on a long season schedule or navigation page.
            nearby = text[pair.start():pair.end() + 160]
            dates = _dates(nearby)
            if dates:
                between = nearby[pair.end() - pair.start():dates[0][0]]
                if re.search(r"\b(?:v(?:s\.?)?|versus|against)\b|对阵|迎战", between, re.I):
                    dates = []
            if dates:
                start, end, value = dates[0]
                end = pair.start() + end
                spans.append({"status": "explicit_pair_and_date" if value == target_date else "different_fixture_date",
                              "source_date": value, "anchor": _anchor(document, pair.start(), end)})
            else:
                spans.append({"status": "pair_present_date_unresolved", "source_date": None,
                              "anchor": _anchor(document, pair.start(), pair.end())})
    return sorted(spans, key=lambda item: item["anchor"]["start"])


def _sentences(text):
    """Source-order, exact-offset fragments; long passages stay visibly partial."""
    boundaries = [0] + [m.end() for m in re.finditer(r"[.!?。！？](?:\s+|(?=[^\x00-\x7f]))|[\r\n]+", text)] + [len(text)]
    for left, right in zip(boundaries, boundaries[1:]):
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right - left < 20:
            continue
        while right - left > 400:
            end = text.rfind(" ", left + 200, left + 400)
            end = end if end > left else left + 400
            yield left, end, True
            left = end + (1 if text[end:end + 1] == " " else 0)
        if right - left >= 20:
            yield left, right, right - left < 30


def _review(document, identity, as_of):
    review = {"document_id": document.get("document_id"), "status": "excluded", "reason": None,
              "fixture_bindings": [], "extracted_evidence_ids": []}
    if document.get("status") != "OK" or document.get("kind") != "page":
        review["reason"] = "discovery_or_unsuccessful_response_not_source_evidence"
        return review
    text = document.get("text")
    if not isinstance(text, str) or not text.strip():
        review["reason"] = "empty_text"
        return review
    if len(text) > 57600:
        raise ValueError("Document exceeds existing 24-window extraction bound")
    if not re.fullmatch(r"[0-9a-f]{64}", str(document.get("raw_sha256", ""))):
        review["reason"] = "missing_archive_hash"
        return review
    try:
        observed, available = utc(document["observed_at"]), utc(document["available_at"])
        published = utc(document["source_published_at"]) if document.get("source_published_at") else None
    except (ValueError, KeyError, TypeError, AttributeError):
        review["reason"] = "missing_or_invalid_operational_time"
        return review
    if observed > available or published and published > observed:
        review["reason"] = "inconsistent_source_clock"
        return review
    if max(observed, available) > as_of:
        review["reason"] = "unavailable_at_decision_time"
        return review
    if max(observed, available, published or observed) >= utc(identity["kickoff_at"]):
        review["reason"] = "not_acquired_before_kickoff"
        return review
    if not _mentions(text, identity):
        review["reason"] = "no_target_team_in_body"
        return review
    bindings = _fixture_spans(document, identity)
    review["fixture_bindings"] = bindings
    if bindings and all(b["status"] == "different_fixture_date" for b in bindings):
        review["reason"] = "different_fixture_date"
        return review
    lead = text[:700]
    if _REPORT.search(lead) and len(_mentions(lead, identity)) == 2:
        review["reason"] = "post_match_report_text"
        return review
    left, right = (_name_pattern(identity["aliases"][side]) for side in ("home", "away"))
    if re.search(left + r"\s+\d{1,2}\s*[-–:]\s*\d{1,2}\s+" + right, lead, re.I):
        review["reason"] = "target_score_in_report_lead"
        return review
    review.update(status="eligible_for_extraction", reason="source_text_with_target_entity")
    return review


def _record(document, identity, review, start, end, topics, *, partial=False, fixture=False):
    anchor = _anchor(document, start, end)
    quote = anchor["quote"]
    matched = [b for b in review["fixture_bindings"] if b["status"] == "explicit_pair_and_date"]
    # A fixture line establishes that line's match. It does not confer match
    # scope on every other club/player paragraph in a multi-fixture document.
    local_match = any(b["anchor"]["start"] >= start and b["anchor"]["end"] <= end for b in matched)
    applicability = "match_scoped_source_assertion" if fixture or local_match else "team_context_requires_match_binding"
    modality = "uncertain_source_statement" if _UNCERTAIN.search(quote) else "source_statement"
    result = {"evidence_id": "evidence-" + digest({"anchor": anchor, "topics": topics})[:24],
              "status": "quoted_source_text", "statement": quote, "topics": topics,
              "mentioned_team_sides": _mentions(quote, identity),
              "applicability": applicability, "match_id": identity["match_id"],
              "modality": modality, "partial_sentence": partial, "anchor": anchor,
              "observed_at": document["observed_at"], "available_at": document["available_at"],
              "source_published_at": document.get("source_published_at"),
              "publication_status": document.get("publication_status", "unknown"),
              "effective_at": None, "calibrated_confidence": None,
              "semantic_verification": "lexical_extraction_meaning_unverified",
              "source_reliability": None, "expected_gain": None}
    return result


def _landing_page_role_v2(document):
    """Recognize discovery routes, independent of keywords that imply value.

    A query string on a locale homepage does not make its mixed card list an
    article. Article leaf routes (including /news/<slug>) remain eligible. The
    acquired document and its links are retained unchanged for the discoverer.
    """
    parts = [p.casefold() for p in urlsplit(document["url"]).path.split("/") if p]
    if parts and re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", parts[0]):
        parts = parts[1:]
    if not parts or parts in (["home"], ["index"], ["index.html"], ["index.htm"]):
        return "home_page"
    index_names = {"news", "latest-news", "newsroom", "stories", "articles", "updates"}
    sections = {"all", "mens", "womens", "men", "women", "first-team", "academy"}
    if parts[0] in index_names and (
            len(parts) == 1 or
            len(parts) == 2 and parts[1] in sections or
            len(parts) == 3 and parts[1] == "page" and parts[2].isdigit()):
        return "news_index"
    return None


def _squad_categories_v2(text):
    categories = set()
    if re.search(r"\b(?:women(?:['’]s)?|womens|ladies)\b|女足", text, re.I):
        categories.add("women")
    for found in re.finditer(r"\b(?:u|under)[- ]?(\d{1,2})\b", text, re.I):
        categories.add("u" + found.group(1))
    if re.search(r"\b(?:academy|youth)\b|青年队|青训队", text, re.I):
        categories.add("youth")
    if re.search(r"\b(?:reserves|reserve team)\b|预备队", text, re.I):
        categories.add("reserves")
    return categories


def _other_squads_v2(text, identity):
    # Bare club aliases never authorize binding another squad's statements.
    # Explicitly named women's or youth fixtures are supported without making
    # gender/age a measure of information value or changing the source text.
    allowed = set().union(*(_squad_categories_v2(name)
                           for names in identity["aliases"].values() for name in names))
    if any(re.fullmatch(r"u\d{1,2}", item) for item in allowed):
        allowed.add("youth")
    return sorted(_squad_categories_v2(text) - allowed)


def _exclude_passage_v2(review, document, start, end, categories):
    skipped = review.setdefault("excluded_passages", [])
    if not any(p["anchor"]["start"] == start and p["anchor"]["end"] == end for p in skipped):
        skipped.append({"reason": "non_target_squad_or_mixed_subject_fragment",
                        "source_squad_categories": categories,
                        "applicability": "not_established_for_target_squad",
                        "anchor": _anchor(document, start, end)})


def _review_v2(document, identity, as_of):
    review = _review(document, identity, as_of)
    # Preserve all source, archive and time checks before classifying a page.
    if review["reason"] in {
            "discovery_or_unsuccessful_response_not_source_evidence", "empty_text",
            "missing_archive_hash", "missing_or_invalid_operational_time",
            "inconsistent_source_clock", "unavailable_at_decision_time",
            "not_acquired_before_kickoff"}:
        return review
    role = _landing_page_role_v2(document)
    if role:
        review.update(status="discovery_only", reason="index_page_for_link_discovery_only",
                      page_role=role, fixture_bindings=[])
        return review
    # Explicit squad routes identify a different source section even when its
    # body only calls that side "Leeds". A normal /news/<headline> is not a
    # section declaration, so its individual passages are checked below.
    sections = urlsplit(document["url"]).path.split("/")[:-1]
    route_squads = _other_squads_v2(" ".join(sections), identity)
    if route_squads:
        review.update(status="excluded", reason="non_target_squad_source_section",
                      source_squad_categories=route_squads, fixture_bindings=[])
        return review
    if review["status"] != "eligible_for_extraction":
        return review
    retained = []
    fragments = list(_sentences(document["text"]))
    for binding in review["fixture_bindings"]:
        anchor = binding["anchor"]
        # Include the surrounding exact-offset fragment so a preceding
        # "Women's fixture" label cannot disappear from a fixture anchor.
        matching = [(start, end) for start, end, _ in fragments
                    if start < anchor["end"] and anchor["start"] < end]
        start = min((left for left, _ in matching), default=anchor["start"])
        end = max((right for _, right in matching), default=anchor["end"])
        categories = _other_squads_v2(document["text"][start:end], identity)
        if categories:
            _exclude_passage_v2(review, document, start, end, categories)
        else:
            retained.append(binding)
    review["fixture_bindings"] = retained
    return review


def _opposition(evidence):
    grouped = {}
    for record in evidence:
        if record["applicability"] != "match_scoped_source_assertion" or record["modality"] != "source_statement":
            continue
        text = re.sub(r"[^\w\s]", " ", record["statement"].casefold())
        negated = bool(_NEGATION.search(text))
        key = re.sub(r"\s+", " ", _NEGATION.sub("", text)).strip()
        grouped.setdefault(key, []).append((negated, record))
    conflicts = []
    for key, rows in sorted(grouped.items()):
        if len({n for n, _ in rows}) < 2 or len({r["anchor"]["document_id"] for _, r in rows}) < 2:
            continue
        ids = sorted(r["evidence_id"] for _, r in rows)
        conflicts.append({"conflict_id": "opposed-" + digest(ids)[:20], "evidence_ids": ids,
                          "status": "opposed_lexical_source_statements", "verified_contradiction": False,
                          "comparison": "same_wording_except_explicit_negation", "expected_gain": None})
    return conflicts


def _query_terms(quote, identity):
    # Only observed words are used as an investigatory hook. No new player,
    # absence, causal impact, or numeric value is invented by this projection.
    words = re.findall(r"[A-Za-z][A-Za-z’'-]*|[\u3400-\u9fff]{2,}", quote)
    stop = {"the", "a", "an", "and", "or", "to", "for", "of", "his", "their", "this", "that", "he", "she", "it", "in", "on", "at", "with", "as", "has", "have", "had", "is", "are", "was", "were", "will", "would", "who", "which", "but", "be", "been", "from", "against", "premier", "league", "champions", "winger", "midfielder", "forward", "defender", "striker", "centre-half", "given", "also", "after", "before", "while", "so", "far", "him", "man", "impressive", "displays", "timely", "club", "team", "season", "first", "round", "fixtures"}
    stop.update(word.casefold() for names in identity["aliases"].values() for name in names for word in name.split())
    # Prioritize literal names and words describing the observed unknown.
    # This makes "Kai Havertz ... muscle injury" a useful query instead of
    # spending the query length on the passage's opening rhetorical clause.
    signal = re.compile(r"injur|fitness|withdraw|hamstring|quad|muscle|sidelin|unavail|uncertain|doubt|official|update|rotat|lineup|qualif|tactic|伤|轮换|首发|晋级|待定", re.I)
    ordered = [w for w in words if signal.search(w) or w[:1].isupper()] + words
    selected = []
    for word in ordered:
        if word.casefold() not in stop and word.casefold() not in {v.casefold() for v in selected}:
            selected.append(word)
    return " ".join(selected[:12])


def _build_match_evidence(match, documents, *, as_of=None, max_evidence=64, version=LEGACY_VERSION):
    """Project quoted evidence and grounded follow-up proposals for one match.

    ``match`` needs match_id, home_team, away_team, kickoff_at and
    prediction_cutoff; kickoff/id and team mappings are accepted aliases.
    ``documents`` are existing OpenWebTools projections with archived hashes.
    Byte-level archive verification remains the executor/replay's job. The
    returned object contains no outcome labels and never changes probabilities.
    """
    identity = _identity(match)
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)) or len(documents) > 128:
        raise ValueError("documents must be a sequence of at most 128 acquired documents")
    if type(max_evidence) is not int or not 1 <= max_evidence <= 128:
        raise ValueError("max_evidence must be 1..128")
    selected_time = utc(as_of if as_of is not None else identity["prediction_cutoff"])
    if selected_time > utc(identity["prediction_cutoff"]):
        raise ValueError("Evidence as_of must not follow prediction cutoff")
    evidence, reviews, seen, fingerprints = [], [], set(), {}
    truncated = False
    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("Each document must be a mapping")
        doc_id = _text(document.get("document_id"), "document_id")
        fingerprint = digest(document)
        if doc_id in fingerprints:
            if fingerprints[doc_id] != fingerprint:
                raise ValueError("One document ID refers to different acquired content")
            continue
        fingerprints[doc_id] = fingerprint
        review = (_review_v2 if version == VERSION else _review)(document, identity, selected_time)
        reviews.append(review)
        if review["status"] != "eligible_for_extraction":
            continue
        proposed = []
        for binding in review["fixture_bindings"]:
            if binding["status"] == "explicit_pair_and_date":
                anchor = binding["anchor"]
                proposed.append(_record(document, identity, review, anchor["start"], anchor["end"], ["fixture_confirmation"], fixture=True))
        for start, end, partial in _sentences(document["text"]):
            quote = document["text"][start:end]
            topics = [key for key, pattern in _TOPIC_RE.items() if pattern.search(quote)]
            if not topics and any(b["status"] == "explicit_pair_and_date" and
                                  start <= b["anchor"]["start"] < b["anchor"]["end"] <= end
                                  for b in review["fixture_bindings"]):
                # The supported labels are not an exhaustive feature list.
                # Other match-bound assertions remain visible as quotations.
                topics = ["other_match_context"]
            if not topics or not _mentions(quote, identity) or not _VERB.search(quote):
                continue
            if version == VERSION:
                categories = _other_squads_v2(quote, identity)
                if categories:
                    _exclude_passage_v2(review, document, start, end, categories)
                    continue
            if _REPORT.search(quote):
                continue
            proposed.append(_record(document, identity, review, start, end, topics, partial=partial))
        for record in proposed:
            # Exact copied content is not a fresh independent source. Retain
            # every document review while counting its quoted bytes once.
            key = (record["anchor"]["raw_sha256"], record["anchor"]["start"], record["anchor"]["end"])
            if key in seen:
                continue
            if len(evidence) >= max_evidence:
                truncated = True
                review["reason"] = "extraction_budget_exhausted"
                break
            seen.add(key)
            evidence.append(record)
            review["extracted_evidence_ids"].append(record["evidence_id"])
        if not review["extracted_evidence_ids"] and review["reason"] != "extraction_budget_exhausted":
            review["reason"] = "no_substantive_match_passage_extracted"
    conflicts = _opposition(evidence)
    unresolved, followups = [], []
    for record in evidence:
        reasons = []
        if record["modality"] == "uncertain_source_statement":
            reasons.append("source_explicitly_leaves_status_unresolved")
        if record["applicability"] == "team_context_requires_match_binding":
            reasons.append("applicability_to_this_fixture_not_established")
        if reasons:
            unresolved.append({"evidence_id": record["evidence_id"], "reasons": reasons,
                               "source_quote": record["statement"], "value_of_resolution": None})
            hook = _query_terms(record["statement"], identity)
            if hook:
                query = f'{identity["home_team"]} {identity["away_team"]} {identity["kickoff_at"][:10]} {hook}'
                followups.append({"candidate_id": "evidence-query-" + digest(query)[:20], "tool": "search_web",
                                  "query": query, "based_on_evidence_ids": [record["evidence_id"]],
                                  "reason": "; ".join(reasons), "expected_gain": None,
                                  "proposal_status": "grounded_query_not_value_ranked"})
    for conflict in conflicts:
        unresolved.append({"conflict_id": conflict["conflict_id"], "evidence_ids": conflict["evidence_ids"],
                           "reasons": ["opposed_source_statements_need_scope_time_or_source_review"], "value_of_resolution": None})
        quoted = next(e for e in evidence if e["evidence_id"] == conflict["evidence_ids"][0])
        query = f'{identity["home_team"]} {identity["away_team"]} {identity["kickoff_at"][:10]} {_query_terms(quoted["statement"], identity)} official update'
        followups.append({"candidate_id": "evidence-query-" + digest(query)[:20], "tool": "search_web",
                          "query": query, "based_on_evidence_ids": conflict["evidence_ids"],
                          "reason": "opposed_source_statements_need_scope_time_or_source_review",
                          "expected_gain": None, "proposal_status": "grounded_query_not_value_ranked"})
    unique_queries = {}
    for proposal in followups:
        existing = unique_queries.get(proposal["candidate_id"])
        if existing is None:
            unique_queries[proposal["candidate_id"]] = proposal
        else:
            existing["based_on_evidence_ids"] = sorted(set(existing["based_on_evidence_ids"] + proposal["based_on_evidence_ids"]))
    followups = list(unique_queries.values())
    if not evidence:
        unresolved.append({"reasons": ["no_substantive_match_evidence_observed"], "value_of_resolution": None})
    return {"schema_version": version, "match_id": identity["match_id"], "identity": deepcopy(identity),
            "as_of": selected_time.isoformat(), "evidence": evidence, "document_reviews": reviews,
            "state": {"present_topics": sorted({t for e in evidence if e["applicability"] == "match_scoped_source_assertion" for t in e["topics"]}),
                      "context_topics": sorted({t for e in evidence if e["applicability"] != "match_scoped_source_assertion" for t in e["topics"]}),
                      "unresolved": unresolved, "conflict_signals": conflicts, "followup_candidates": followups,
                      "coverage_complete": False, "extraction_truncated": truncated},
            "semantics": {"method": "deterministic_exact_quote_and_lexical_topic_projection", "llm_calls": 0,
                          "topics_are_value_weights": False, "fixed_information_checklist": False,
                          "confidence": "quote_location_checked_semantics_truth_and_reliability_uncalibrated",
                          "archive_bytes_verified_here": False, "source_time_inferred_from_retrieval": False,
                          "probability_update": None, "numeric_voi": None,
                          "missing_evidence_is_negative_answer": False, "followups_are_mandatory": False}}


def build_match_evidence(match, documents, *, as_of=None, max_evidence=64, version=None):
    """Project evidence using a declared, replayable extraction version.

    The default v2 separates link-discovery landing pages and non-target squad
    fragments from match evidence. Explicit v1 retains the original projection
    byte-for-byte, including its known mixed-homepage limitation, so archived
    decisions can be replayed without rewriting their information state.
    """
    selected_version = VERSION if version is None else version
    if selected_version not in SUPPORTED_VERSIONS:
        raise ValueError("Unsupported match evidence version: " + str(selected_version))
    if selected_version in {ARTICLES_VERSION, CONTENT_VERSION, HTML_BODY_VERSION}:
        from .match_evidence_articles import build_article_evidence
        return build_article_evidence(match, documents, as_of=as_of, max_evidence=max_evidence,
                                      version=selected_version)
    return _build_match_evidence(match, documents, as_of=as_of, max_evidence=max_evidence,
                                 version=selected_version)


__all__ = ["VERSION", "LEGACY_VERSION", "ARTICLES_VERSION", "CONTENT_VERSION", "HTML_BODY_VERSION", "SUPPORTED_VERSIONS", "build_match_evidence"]
