"""Explicit same-page metadata/DOM binding for news projection version 3.

This adapter extracts structural paragraphs, not football topics or information
value. It does not resolve match applicability, source truth or effective time.
"""
from copy import deepcopy
import json
import re
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag


VERSION = "same-page-single-article-paragraphs-v1"
STRUCTURED_VERSION = "same-page-unique-structured-body-paragraphs-v2"
BODY_SELECTORS = ('[itemprop~="articleBody"]', '.elementor-widget-theme-post-content', '.entry-content')
BLOCKED_TAGS = {"nav", "header", "footer", "aside", "script", "style", "noscript", "form", "button", "dialog"}
BLOCKED_ROLES = {"navigation", "banner", "contentinfo", "complementary", "dialog"}
STRUCTURAL_CLASS = re.compile(
    r"(?:^|[-_])(?:header(?:renderer)?|footer|nav(?:igation)?|aside|share|sharing|social(?:share)?|"
    r"related|recommended|recommendations?|link[-_]?card|card|byline|metadata|cookie|subscribe)(?:[-_]|$)", re.I)


def _blocked(node):
    if not isinstance(node, Tag):
        return False
    if node.name in BLOCKED_TAGS or node.get("role") in BLOCKED_ROLES:
        return True
    if node.has_attr("hidden") or str(node.get("aria-hidden", "")).casefold() == "true":
        return True
    style = re.sub(r"\s+", "", str(node.get("style", ""))).casefold()
    if "display:none" in style or "visibility:hidden" in style:
        return True
    return any(STRUCTURAL_CLASS.search(str(value)) for value in node.get("class", []))


def _within_blocked(node, stop=None):
    for parent in [node, *node.parents]:
        if parent is stop:
            break
        if _blocked(parent):
            return True
    return False


def _selector(node):
    # Use actual structural positions, avoiding invented IDs or unstable text
    # selectors. Every component points into the unchanged captured DOM.
    parts = []
    for current in [node, *node.parents]:
        if not isinstance(current, Tag) or current.name == "[document]":
            break
        index = 1+len(current.find_previous_siblings(current.name))
        parts.append(f"{current.name}:nth-of-type({index})")
    return " > ".join(reversed(parts))


def _page_url(value, source):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("explicit_page_url_required")
    parts = urlsplit(urljoin(source, value.strip()))
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("invalid_article_page_url")
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip("/") or "/", parts.query, ""))


def _structured_body(soup, headline):
    """An explicit CMS/schema body marker, never an arbitrary large text div."""
    from .news_discovery import _plain
    containers = [n for n in soup.select(','.join(BODY_SELECTORS))
                  if not _within_blocked(n) and not _within_template(n)]
    if not containers:
        return None, None, "registered_structural_body_container_missing"
    if len(containers) != 1:
        return None, None, "single_structural_body_container_required"
    container = containers[0]
    headings = [n for n in soup.find_all('h1') if not _within_blocked(n) and not _within_template(n)]
    if len(headings) != 1 or _plain(str(headings[0])).casefold() != headline.casefold():
        return None, None, "article_heading_does_not_uniquely_match_metadata"
    # Related cards without body paragraphs are harmless. Competing full-body
    # article/main containers cannot be silently discarded to obtain a result.
    for node in soup.find_all(['article', 'main']):
        if (node is container or node in container.parents or container in node.parents
                or _within_blocked(node) or _within_template(node)):
            continue
        paragraphs = [p for p in node.find_all('p') if not _within_blocked(p)]
        if len(paragraphs) >= 2:
            return None, None, "competing_semantic_body_container"
    return container, headings, None


def _within_template(node, stop=None):
    for parent in [node, *node.parents]:
        if parent is stop:
            break
        if isinstance(parent, Tag) and parent.name == "template":
            return True
    return False


def project_bound_html_body(document, html, observation, metadata, *, recognized_articles,
                           structural_fallback=False, allow_liverpool_locale_alias=False):
    """Recover paragraphs only when one original metadata node binds the page."""
    from .news_discovery import _plain, _timestamp, _types, MAX_NODES, MAX_SEGMENTS, MAX_TEXT

    adapter_version = STRUCTURED_VERSION if structural_fallback else VERSION

    def reject(reason, *, clear=True):
        metadata["excluded"].append({"reason": reason})
        metadata.update(role="unknown", segments=[], excluded_count=len(metadata["excluded"]),
                        html_body_adapter=adapter_version)
        if clear:
            document.update(kind="page", status="NO_DATA", text="", text_truncated=False)
        return document

    if metadata["excluded"]:
        return reject("html_body_cannot_bypass_excluded_or_incomplete_metadata")
    if metadata["jsonld_parse_errors"] or metadata["nodes_truncated"]:
        return reject("html_body_cannot_bypass_excluded_or_incomplete_metadata", clear=recognized_articles > 0)
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", type="application/ld+json")
    if len(scripts) > 32:
        return reject("article_metadata_script_bound_exceeded", clear=recognized_articles > 0)
    pending, candidates, count = [], [], 0
    for index, script in enumerate(scripts):
        try:
            value = json.loads(script.get_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, RecursionError):
            return reject("invalid_article_metadata_json", clear=recognized_articles > 0)
        pending.append((value, f"script[{index}]"))
    while pending and count < MAX_NODES:
        node, pointer = pending.pop(0)
        count += 1
        if isinstance(node, list):
            pending.extend((value, pointer+f"[{index}]") for index, value in enumerate(node))
        elif isinstance(node, dict):
            types = _types(node)
            if "LiveBlogPosting" in types:
                return reject("liveblog_dom_body_has_no_per_update_clock")
            if types & {"Article", "NewsArticle", "BlogPosting"}:
                candidates.append((node, pointer))
            for key in ("@graph", "mainEntity", "liveBlogUpdate"):
                if key in node:
                    pending.append((node[key], pointer+"."+key))
    if pending or len(candidates) != 1:
        return reject("single_unambiguous_article_metadata_required", clear=bool(candidates) or recognized_articles > 0)
    node, pointer = candidates[0]
    if _plain(node.get("articleBody")):
        return reject("existing_article_body_not_eligible_for_html_replacement")
    headline = _plain(node.get("headline"))
    if not headline:
        return reject("article_metadata_headline_required")
    try:
        published, modified = _timestamp(node.get("datePublished")), _timestamp(node.get("dateModified"))
    except (ValueError, TypeError, OverflowError):
        return reject("invalid_article_metadata_publication_time")
    if any(stamp is not None and stamp > observation.observed_at for stamp in (published, modified)):
        return reject("article_metadata_time_after_actual_capture")
    if published and modified and modified < published:
        return reject("article_metadata_modification_precedes_publication")
    try:
        expected = _page_url(document["url"], document["url"])
        urls = []
        for key in ("url", "mainEntityOfPage"):
            value = node.get(key)
            if isinstance(value, dict):
                value = value.get("@id", value.get("url"))
            if value is not None:
                urls.append(_page_url(value, document["url"]))
        canonical_nodes = soup.find_all("link", rel="canonical")
        canonicals = [_page_url(n.get("href"), document["url"]) for n in canonical_nodes]
        locale_alias = False
        if allow_liverpool_locale_alias and urlsplit(expected).hostname == "www.liverpoolfc.com":
            alternate = soup.select('link[rel="alternate"][hreflang="en"]')
            parts = urlsplit(expected)
            alias = urlunsplit((parts.scheme, parts.netloc, "/en"+parts.path, parts.query, ""))
            locale_alias = (parts.path.startswith("/news/") and not parts.query
                and observation.endpoint == document["url"] and not observation.params
                and canonicals == [expected] and len(alternate) == 1
                and _page_url(alternate[0].get("href"), expected) == expected
                and bool(urls) and all(url == alias for url in urls)
                and str(soup.html.get("lang", "")).casefold() == "en")
        if not urls or any(url != expected for url in urls) and not locale_alias:
            return reject("article_metadata_does_not_bind_requested_page")
        if locale_alias:
            metadata["page_identity_alias"] = {"requested_url": expected, "source_declared_urls": urls,
                "basis": "same_response_exact_html_canonical_english_alternate_jsonld_en_prefix_and_visible_headline",
                "not_an_http_redirect": True}
        if len(canonicals) > 1 or any(url != expected for url in canonicals):
            return reject("page_canonical_conflicts_with_requested_article")
    except (ValueError, TypeError):
        return reject("invalid_or_ambiguous_article_page_binding")
    containers = [n for n in soup.find_all("article") if not _within_blocked(n)]
    if not containers:
        containers = [n for n in soup.find_all("main") if not _within_blocked(n)]
    structural = structural_fallback and bool(soup.select(','.join(BODY_SELECTORS)))
    if structural:
        container, headings, error = _structured_body(soup, headline)
        if error:
            return reject(error)
    elif len(containers) != 1:
        return reject("single_article_or_main_container_required")
    else:
        container = containers[0]
    # A title may legitimately be inside the article's header. It binds the
    # page but never becomes a paragraph, so use its own visibility boundary.
    headings = headings if structural else [n for n in container.find_all("h1") if not any(
        p.name in {"nav", "footer", "aside", "script", "style", "noscript"}
        or p.has_attr("hidden") or str(p.get("aria-hidden", "")).casefold() == "true"
        or "display:none" in re.sub(r"\s+", "", str(p.get("style", ""))).casefold()
        or "visibility:hidden" in re.sub(r"\s+", "", str(p.get("style", ""))).casefold()
        for p in [n, *n.parents] if isinstance(p, Tag) and p is not container)]
    if len(headings) != 1 or _plain(str(headings[0])).casefold() != headline.casefold():
        return reject("article_heading_does_not_uniquely_match_metadata")
    selector = _selector(container)
    paragraphs = container.find_all("p")
    if len(paragraphs) > 1024 or len(selector) > 512:
        return reject("article_dom_paragraph_or_selector_bound_exceeded")
    selected, excluded = [], []
    for index, paragraph in enumerate(paragraphs):
        reason = None
        path = _selector(paragraph)
        if (len(path) > 512 or _within_blocked(paragraph, container)
                or structural and _within_template(paragraph, container)):
            reason = "non_body_structural_paragraph"
        else:
            clean = BeautifulSoup(str(paragraph), "html.parser")
            for child in list(clean.find_all(True)):
                if child.name and (_blocked(child) or child.name in {"h1", "h2", "h3", "h4", "h5", "h6", "time"}
                                   or structural and child.name == "template"):
                    child.decompose()
            text = _plain(str(clean))
            link_text = " ".join(_plain(str(n)) for n in clean.find_all("a"))
            if not text or re.fullmatch(r"[\d\s/:.,-]+", text):
                reason = "empty_or_date_only_paragraph"
            elif paragraph.find_parent("a") is not None or text == link_text:
                reason = "standalone_link_card_paragraph"
            else:
                selected.append((index, path, text))
        if reason:
            excluded.append({"original_dom_paragraph_index": index, "reason": reason})
    if len(selected) < 2:
        return reject("at_least_two_structurally_bound_body_paragraphs_required")
    binding = {"version": adapter_version, "metadata_json_pointer": pointer, "source_url": document["url"],
        "declared_page_urls": urls, "canonical_url": canonicals[0] if canonicals else None,
        "metadata_headline": headline, "heading_text": _plain(str(headings[0])),
        "heading_css_selector": _selector(headings[0]), "container_css_selector": selector,
        "binding_scope": "page_identity_and_source_markup_not_match_scope_or_semantic_truth"}
    if structural:
        binding.update(container_selection="unique_registered_structural_body",
            registered_body_selectors=list(BODY_SELECTORS),
            title_scope="unique_visible_page_h1_bound_to_same_page_article_metadata")
    publication = {"source_published_at": published.isoformat() if published else None,
        "source_modified_at": modified.isoformat() if modified else None,
        "publication_raw": node.get("datePublished"), "modified_raw": node.get("dateModified")}
    parts, offset, segments = [], 0, []
    for index, path, text in selected[:MAX_SEGMENTS]:
        available = MAX_TEXT-offset-(1 if parts else 0)
        if available <= 0:
            break
        value, start = text[:available], offset+(1 if parts else 0)
        segments.append({**publication, "text": value, "start": start, "end": start+len(value),
            "content_source": "semantic_html_article_paragraph", "content_role": "paragraph",
            "css_selector": path, "original_dom_paragraph_index": index,
            "source_url": document["url"], "json_pointer": pointer,
            "body_binding": deepcopy(binding), "truncated": len(value)<len(text)})
        parts.append(value)
        offset = start+len(value)
    if len(segments) < 2:
        return reject("body_paragraphs_outside_projection_bound")
    metadata.update(role="article", segments=segments, html_body_adapter=adapter_version,
        body_binding=binding, excluded_dom_paragraphs=excluded, excluded_count=len(metadata["excluded"]))
    document.update(kind="article", status="OK", text="\n".join(parts),
        source_published_at=publication["source_published_at"],
        publication_status="source_article_metadata" if published else "unknown",
        text_truncated=len(segments)<len(selected) or any(s["truncated"] for s in segments))
    return document
