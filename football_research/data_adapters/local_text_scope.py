"""Source-scoped Estonian assertions, never global aliases or information value.

Inputs must be raw-verified projections. UEFA verifies fixture identity; the
local lexical correspondences remain explicit, unverified source assertions.
"""
from hashlib import sha256
import json
import re
from datetime import date
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from football_research.acquisition.state import parse_time
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from .verified_fixtures import verify_identity_binding

VERSION = "jalgpall-et-local-scope-v1"
INDEX = "https://jalgpall.ee/koondis/1/uudised"
PAIR = re.compile(r"\bEesti\s*[–—-]\s*Island\b", re.I)
FOOTBALL = re.compile(r"koondis\w*|jalgpall\w*|Rahvuste liiga|mäng\w*|kohtumis\w*", re.I)
OTHER = re.compile(r"\b(?:U[- ]?(?:1[5-9]|2[013])|(?:naiste|noorte|tüdrukute)\s+koondis\w*)\b", re.I)
DAY = re.compile(r"\b(?P<day>[0-3]?\d)\.\s*oktoob(?:ril|er|ri)\b(?:\s+(?P<year>20\d{2})\b)?", re.I)
ALIASES = {"Estonia": ["Eesti"], "Iceland": ["Island", "Islandi", "Islandit"]}


def _digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _anchor(text, start, end, source):
    return {"source": source, "start": start, "end": end, "quote": text[start:end]}


def _local_url(url):
    p = urlsplit(url)
    return p.scheme == "https" and p.netloc == "jalgpall.ee" and not p.query and not p.fragment and (
        p.path == "/koondis/1/uudised" or p.path.startswith("/koondis/1/uudised/"))


def build_local_scope_proof(identity_binding, *, identity_archive, index_observation, index_archive):
    """Bind local source context to archived facts, without proving translation."""
    verified = verify_identity_binding(identity_binding, archive_root=identity_archive, as_of=identity_binding["verified_at"])
    canonical = verified["canonical"]
    if verified["status"] != "CONFIRMED" or canonical["category"] != "senior_men_national" or {
            canonical["home_name"], canonical["away_name"]} != set(ALIASES):
        raise ValueError("Local rule requires the verified Estonia/Iceland senior-men fixture")
    obs = Observation.model_validate(index_observation)
    archive = RawArchive(index_archive)
    if obs not in archive.observations() or obs.endpoint != INDEX or obs.http_status != 200 or obs.error_class:
        raise ValueError("Expected the actual successful archived jalgpall men's index")
    raw = archive.read_body(obs).decode("utf-8", errors="strict")
    soup = BeautifulSoup(raw, "html.parser")
    headers = soup.select('#headernav li.active a[href="/koondis/1/uudised"]')
    language = soup.html.get("lang") if soup.html else None
    supported = language == "et" and len(headers) == 1 and headers[0].get_text(strip=True) == "Mehed"
    entries = []
    for a in soup.select(".news-widget a[href]"):
        title, stamp = a.select_one(".title"), a.select_one(".date")
        url = urljoin(INDEX, a["href"])
        if title is None or stamp is None or not _local_url(url):
            continue
        text, published = title.get_text(" ", strip=True), stamp.get_text(" ", strip=True)
        start = raw.find(text)
        if start < 0:
            continue
        entries.append({"url": url, "title": text, "publication_raw": published,
            "title_evidence": _anchor(raw, start, start + len(text), "index_raw_utf8"),
            "selector": ".news-widget a[href] .title/.date"})
    proof = {"schema": VERSION, "source_url": INDEX, "language": language,
        "source_scope_supported": supported, "alias_status": "unverified_source_assertion",
        "alias_basis": "source_scoped_lexical_rule_not_bilingual_or_native_team_id_proof",
        "aliases": {side: {"canonical_name": canonical[side + "_name"], "canonical_id": canonical[side + "_id"],
                           "local_forms": ALIASES[canonical[side + "_name"]]} for side in ("home", "away")},
        "canonical": {k: canonical[k] for k in ("home_id", "away_id", "home_name", "away_name", "kickoff", "category", "source_event_id", "source_url")},
        "identity_binding_sha256": verified["binding_sha256"], "identity_observations": verified["observations"],
        "index_observation_id": obs.id, "index_raw_sha256": obs.content_hash,
        "index_observed_at": obs.observed_at.isoformat(), "index_entries": entries,
        "source_category_evidence": {"selector": "#headernav li.active a", "text": headers[0].get_text(strip=True) if headers else None},
        "index_fixture_kickoff_proof": None,
        "limitation": "Index gives men's category, links and publication dates; no event date/kickoff or bilingual identity proof."}
    proof["proof_sha256"] = _digest(proof)
    return proof


def associate_local_text(document, *, proof):
    """Describe a raw-verified v2 document; no outcome, ranking or eligibility filter."""
    if proof.get("schema") != VERSION or proof.get("proof_sha256") != _digest({k: v for k, v in proof.items() if k != "proof_sha256"}):
        raise ValueError("Changed local source proof")
    text, news = document.get("text", ""), document.get("structured_news", {})
    result = {"schema": VERSION, "scope": "unscoped", "alias_status": "unverified_source_assertion",
        "identity_binding_sha256": proof["identity_binding_sha256"], "proof_sha256": proof["proof_sha256"],
        "document_raw_sha256": document.get("raw_sha256"), "text_sha256": sha256(text.encode()).hexdigest(),
        "mentioned_sides": [], "mention_evidence": [], "pair_evidence": [], "date_evidence": [],
        "reason": "outside_supported_source_language_or_archived_menu", "independent_identity_established": False,
        "semantics": "source_assertion_only_not_truth_usefulness_eligibility_or_information_value"}
    if (not proof["source_scope_supported"] or not _local_url(document.get("url", ""))
            or document.get("url") not in {row["url"] for row in proof["index_entries"]}
            or news.get("source_language") != "et" or news.get("language_source") != "html@lang"
            or news.get("role") != "article" or document.get("status") != "OK"):
        return result
    segments = news.get("segments", [])
    if not segments or any(text[s["start"]:s["end"]] != s["text"] for s in segments):
        raise ValueError("Article segment offsets do not reproduce its text")
    if OTHER.search(news.get("headline", "")):
        result["reason"] = "other_age_or_sex_category_in_headline"
        return result
    pairs = [m for m in PAIR.finditer(text) if FOOTBALL.search(text[max(0, m.start()-60):m.end()+90])
             and not OTHER.search(text[max(0, m.start()-45):m.end()+45])]
    for side, info in proof["aliases"].items():
        pattern = r"\b(?:" + "|".join(re.escape(a) for a in info["local_forms"]) + r")\b"
        for m in re.finditer(pattern, text, re.I):
            context = text[max(0, m.start()-35):m.end()+65]
            if not OTHER.search(context) and (any(p.start() <= m.start() < p.end() for p in pairs)
                    or re.match(r"\s+(?:meeste\s+)?(?:koondis\w*|jalgpall\w*|koosseis\w*)", text[m.end():], re.I)):
                result["mention_evidence"].append({**_anchor(text, m.start(), m.end(), "document_text"), "side": side,
                    "canonical_id": info["canonical_id"], "mapping_status": "unverified_source_assertion"})
    result["mentioned_sides"] = sorted({e["side"] for e in result["mention_evidence"]})
    result["pair_evidence"] = [_anchor(text, p.start(), p.end(), "document_text") for p in pairs]
    if result["mentioned_sides"]:
        result.update(scope="team_context", reason="source_scoped_team_mentions_without_verified_fixture_binding")
    target = parse_time(proof["canonical"]["kickoff"]).date().isoformat()
    for s in segments:
        if s.get("content_role") == "headline" or OTHER.search(s["text"]):
            continue
        for m in DAY.finditer(s["text"]):
            start, end = s["start"] + m.start(), s["start"] + m.end()
            if not FOOTBALL.search(s["text"][:m.end()]) or not any(abs(start - p.end()) < 400 for p in pairs):
                continue
            year, basis = m.group("year"), "explicit_event_year_in_body"
            lead = max(0, m.start()-15)
            today = re.search(r"\btäna\s*,?\s*$", s["text"][lead:m.start()], re.I)
            publication = s.get("source_publication_date")
            if year is None and today and publication and s.get("publication_raw"):
                published = date.fromisoformat(publication)
                if published.month == 10 and published.day == int(m.group("day")):
                    year, basis = str(published.year), "explicit_body_today_link_to_same_page_source_publication_date"
            try:
                event_date = date(int(year), 10, int(m.group("day"))).isoformat() if year else None
            except ValueError:
                event_date = None
            result["date_evidence"].append({**_anchor(text, start, end, "document_text"),
                "event_date": event_date, "basis": basis if year else "year_unresolved_not_filled_from_fixture",
                "publication_raw": s.get("publication_raw") if today else None,
                "publication_date": publication if today else None, "segment_start": s["start"],
                "today_evidence": _anchor(text, s["start"] + lead + today.start(), start, "document_text") if today else None,
                "date_semantics": "source_asserted_calendar_date_no_local_kickoff_timezone_verification"})
    dates = {e["event_date"] for e in result["date_evidence"] if e["event_date"]}
    if dates == {target} and len(result["mentioned_sides"]) == 2:
        result.update(scope="match_specific", reason="asserted_pair_and_explicitly_grounded_event_date")
    elif dates and dates != {target}:
        result["reason"] = "different_or_conflicting_event_date"
    return result
