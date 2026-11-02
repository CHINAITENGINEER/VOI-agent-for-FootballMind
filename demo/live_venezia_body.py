"""Read the observed Venezia article template without fetching another page."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import re
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from football_agent.inquiry_contract import normalized
from football_agent.link_context import _hidden


VERSION = "footballmind-live-venezia-same-page-body-v1"


def _objects(value):
    if isinstance(value, list):
        for item in value:
            yield from _objects(item)
    elif isinstance(value, dict):
        yield value
        if "@graph" in value:
            yield from _objects(value["@graph"])


def _kind(node, expected):
    values = node.get("@type", [])
    return expected in ([values] if isinstance(values, str) else values if isinstance(values, list) else [])


def _visible(node):
    return not any(_hidden(parent) for parent in [node, *node.parents])


def _timestamp(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Timestamp needs an explicit source timezone")
    return stamp


def project_venezia_body(document, raw, observation):
    """Bind URL-less NewsArticle through same-page WebPage, DOM and OG fields.

    The source supplies one NewsArticle without its own URL, but a separate
    WebPage identifies the requested URL. All explicit identifiers and titles
    must agree. Only article markdown paragraphs enter the returned body;
    header, navigation, recommendations and image descriptions never enter it.
    """
    url = document.get("url", "")
    parts = urlsplit(url)
    if (parts.hostname != "www.veneziafc.it" or not re.fullmatch(r"/news/[^/]+/?", parts.path)
            or parts.query or observation.endpoint != url or observation.params
            or observation.http_status != 200 or observation.error_class
            or (document.get("structured_news") or {}).get("role") == "article"):
        return document

    def reject(reason):
        result = deepcopy(document)
        result["live_venezia_body_projection"] = {"version": VERSION, "status": "rejected", "reason": reason,
                                                  "network_requests": 0}
        return result

    soup = BeautifulSoup(raw, "html.parser")
    canonicals = soup.select('head link[rel="canonical"]')
    og_urls = soup.select('head meta[property="og:url"]')
    og_titles = soup.select('head meta[property="og:title"]')
    og_types = soup.select('head meta[property="og:type"]')
    if not all(len(nodes) == 1 for nodes in (canonicals, og_urls, og_titles, og_types)):
        return reject("unique_canonical_and_og_metadata_required")
    if (canonicals[0].get("href") != url or og_urls[0].get("content") != url
            or og_types[0].get("content") != "article"):
        return reject("canonical_or_og_binding_mismatch")
    scripts = soup.select('script[type="application/ld+json"]')
    if len(scripts) > 32:
        return reject("jsonld_script_bound_exceeded")
    try:
        records = [node for script in scripts for node in _objects(json.loads(script.get_text()))]
    except (ValueError, TypeError, RecursionError):
        return reject("malformed_jsonld")
    if len(records) > 2048:
        return reject("jsonld_record_bound_exceeded")
    article_records = [x for x in records if any(_kind(x, kind) for kind in ("Article", "NewsArticle", "BlogPosting", "LiveBlogPosting"))]
    webpages = [x for x in records if _kind(x, "WebPage")]
    if len(article_records) != 1 or len(webpages) != 1:
        return reject("unique_article_and_webpage_required")
    record, webpage = article_records[0], webpages[0]
    if not _kind(record, "NewsArticle") or webpage.get("url") != url or webpage.get("@id") != url:
        return reject("jsonld_webpage_binding_mismatch")
    for key in ("url", "@id", "mainEntityOfPage"):
        value = record.get(key)
        if value is not None:
            value = value.get("@id", value.get("url")) if isinstance(value, dict) else value
            if not isinstance(value, str) or value.split("#", 1)[0] != url:
                return reject("conflicting_article_identifier")
    articles = [node for node in soup.select("article") if _visible(node)
                and not node.find_parent(["nav", "footer", "aside"])]
    if len(articles) != 1:
        return reject("unique_article_dom_required")
    article = articles[0]
    headings = [x for x in article.select("h1") if _visible(x)]
    if len(headings) != 1:
        return reject("unique_visible_article_heading_required")
    title = normalized(headings[0].get_text(" "))
    if not title or any(normalized(str(value or "")) != title for value in
                        (og_titles[0].get("content"), record.get("headline"), webpage.get("headline"))):
        return reject("article_heading_metadata_mismatch")
    times = [x for x in article.select("time[datetime]") if _visible(x)]
    if len(times) != 1:
        return reject("unique_article_publication_required")
    try:
        published = _timestamp(record.get("datePublished"))
        modified = _timestamp(record["dateModified"]) if record.get("dateModified") else None
        if (_timestamp(times[0]["datetime"]) != published
                or datetime.strptime(normalized(times[0].get_text(" ")), "%d/%m/%Y").date() != published.date()
                or published > observation.observed_at
                or (modified is not None and (modified < published or modified > observation.observed_at))):
            return reject("publication_time_conflict_or_future")
    except (ValueError, TypeError, OverflowError):
        return reject("invalid_explicit_publication_time")

    roots = [x for x in article.select(".markdown-story-part > .html-content") if _visible(x)]
    if not 1 <= len(roots) <= 64:
        return reject("registered_markdown_body_required")
    if any(root.find_parent(["header", "nav", "footer", "aside"]) for root in roots):
        return reject("body_inside_navigation")
    paragraphs = [node for root in roots for node in root.select("p")
                  if _visible(node) and not node.find_parent(["header", "nav", "footer", "aside"])]
    if not 2 <= len(paragraphs) <= 1024:
        return reject("insufficient_or_excessive_body_paragraphs")
    values = []
    for node in paragraphs:
        clean = BeautifulSoup(str(node), "html.parser")
        for element in list(clean.find_all(True)):
            if element.name and (_hidden(element) or element.name in {"nav", "footer", "aside", "button", "svg"}):
                element.decompose()
        value = normalized(clean.get_text(" "))
        if value:
            values.append(value)
    text = "\n".join(values)
    if len(values) < 2 or len(text) < 160:
        return reject("insufficient_real_body_text")
    # Do not silently supply a truncated body to the new full-text encoder.
    if len(text) > 200000:
        return reject("body_character_limit_exceeded_no_truncation")
    authors = record.get("author")
    authors = authors if isinstance(authors, list) else [authors] if authors else []
    names = [a.get("name") if isinstance(a, dict) else a for a in authors]
    names = [normalized(a) for a in names if isinstance(a, str) and a.strip()]
    segments, offset = [], 0
    for index, value in enumerate(values):
        start = offset + bool(segments)
        segments.append({"text": value, "start": start, "end": start + len(value), "truncated": False,
                         "paragraph_index": index, "content_role": "paragraph",
                         "content_source": "same_page_registered_markdown_body",
                         "css_selector": "article .markdown-story-part > .html-content p",
                         "source_url": url, "source_published_at": published.isoformat()})
        offset = start + len(value)
    result = deepcopy(document)
    result.update(kind="article", status="OK", title=title, text=text, text_truncated=False,
                  source_published_at=published.isoformat(), source_modified_at=modified.isoformat() if modified else None,
                  publication_status="declared", source_authors=names,
                  body_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    result["structured_news"] = {
        "version": VERSION, "role": "article", "headline": title, "segments": segments,
        "html_body_adapter": VERSION, "excluded": [], "source_authors": names,
        "source_language": soup.html.get("lang") if soup.html else None,
        "language_source": "html@lang", "previous_projection": document.get("structured_news"),
        "body_binding": {"canonical_url": url, "og_url": url, "webpage_url": webpage["url"],
                         "heading_text": title, "article_identifier_absent": not any(record.get(k) for k in ("url", "@id", "mainEntityOfPage")),
                         "binding_scope": "same_response_unique_metadata_and_article_dom_not_match_truth_or_utility"},
    }
    result["live_venezia_body_projection"] = {"version": VERSION, "status": "completed",
        "body_characters": len(text), "paragraphs": len(values), "network_requests": 0,
        "rendered_visibility_verified": False, "body_scope": "article_markdown_paragraphs_only",
        "publication_scope": "source_assertion_not_first_public_availability"}
    return result
