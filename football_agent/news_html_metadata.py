"""Finite HTML metadata/DOM article binding for news projection version 6.

Registered templates bind same-page metadata to a visible title/date/body
region: a dated permalink, or a semantic Contentful article with JSON Flight
publication data. No host, club, topic, or presumed usefulness selects text.
Date-only source assertions remain date-only.
"""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
import re
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Tag

from .news_html_body import _blocked, _page_url, _selector


VERSION = "same-page-og-dated-news-body-v1"
GALLERY_VERSION = "same-page-contentful-gallery-introduction-v1"
ROOT_SELECTOR = ".text-content"
BODY_SELECTORS = (":scope > .ingress", ":scope > .news-text")
_NON_BODY = re.compile(r"(?:^|[-_])(?:ads?|advert(?:isement|ising)?|sponsor(?:ed)?|promo|banner|image|caption)(?:[-_]|$)", re.I)
_DATED_PATH = re.compile(r"/[^/]+/(\d{4})/(\d{2})/(\d{2})/([^/]+)/?")
CONTENTFUL_ROOT_SELECTOR = '[data-component="Article"]'


def _excluded(node, stop=None):
    for parent in [node, *node.parents]:
        if parent is stop:
            break
        if not isinstance(parent, Tag):
            continue
        if (_blocked(parent) or parent.name in {"template", "iframe", "figure", "figcaption"}
                or parent.has_attr("data-ad") or parent.has_attr("data-ad-slot")
                or any(_NON_BODY.search(str(value)) for value in parent.get("class", []))
                or _NON_BODY.search(str(parent.get("id", "")))):
            return True
    return False


def _contentful_publication(soup, *, source_format="News", title=None, summary=None):
    """Read JSON data in Flight records; never evaluate the enclosing JavaScript."""
    candidates, count = [], 0
    for script_index, script in enumerate(soup.find_all("script")):
        raw = script.get_text()
        if "publishDate" not in raw:
            continue
        match = re.fullmatch(r"\s*self\.__next_f\.push\((\[.*\])\)\s*;?\s*", raw, re.S)
        if not match or len(raw) > 2_000_000:
            raise ValueError("unrecognized or oversized publication data")
        envelope = json.loads(match[1])
        if not isinstance(envelope, list) or len(envelope) != 2 or envelope[0] != 1 or not isinstance(envelope[1], str):
            raise ValueError("invalid Flight data envelope")
        for line in envelope[1].splitlines():
            record = re.fullmatch(r"([\da-f]+):([\[{].*)", line, re.S)
            if not record:
                continue
            pending = [(json.loads(record[2]), f"script[{script_index}].flight[{record[1]}]")]
            while pending:
                node, pointer = pending.pop()
                count += 1
                if count > 20000:
                    raise ValueError("publication data traversal bound exceeded")
                if isinstance(node, dict):
                    if (node.get("format") == source_format and {"title", "summary", "publishDate"} <= node.keys()
                            and (title is None or node["title"] == title)
                            and (summary is None or node["summary"] == summary)):
                        candidates.append((node, pointer))
                    pending.extend((value, pointer+"."+key) for key, value in node.items())
                elif isinstance(node, list):
                    pending.extend((value, pointer+f"[{i}]") for i, value in enumerate(node))
    if len(candidates) != 1:
        raise ValueError("one bound publication record required")
    return candidates[0]


def _project_contentful_body(document, soup, observation, metadata, reject, *, allow_gallery=False):
    """Bind the observed Contentful article, visible header, and UTC publication."""
    from .news_discovery import _plain, _timestamp, MAX_SEGMENTS, MAX_TEXT

    roots = soup.select(CONTENTFUL_ROOT_SELECTOR)
    if len(roots) != 1 or _excluded(roots[0]):
        return reject("unique_visible_contentful_article_required")
    root = roots[0]
    headings = [n for n in root.select('[data-testid="heading"]') if not _excluded(n)]
    dates = [n for n in root.select('[data-testid="publish-date"]') if not _excluded(n)]
    bodies = [n for n in root.select('[data-component="ContentfulRichTextRender"] > article') if not _excluded(n)]
    if len(headings) != 1 or len(dates) != 1 or len(bodies) != 1:
        return reject("contentful_heading_date_and_rich_text_required")
    heading, stamp, body = headings[0], dates[0], bodies[0]
    gallery = allow_gallery and bool(root.select('[data-component="GalleryImages"]'))
    if gallery and (len(root.select('[data-component="GalleryImages"]')) != 1
                    or _excluded(root.select('[data-component="GalleryImages"]')[0])
                    or _plain(str(body))):
        return reject("contentful_gallery_requires_unique_gallery_and_empty_rich_text")
    canonicals = soup.select('link[rel="canonical"]')
    titles = soup.select('meta[property="og:title"]')
    descriptions = soup.select('meta[property="og:description"]')
    if (len(canonicals) != 1 or len(titles) != 1 or len(descriptions) != 1
            or any(n.parent is not soup.head for n in (*canonicals, *titles, *descriptions))):
        return reject("contentful_unique_head_identity_metadata_required")
    try:
        expected = _page_url(document["url"], document["url"])
        if (observation.source != "open_page" or observation.params
                or _page_url(observation.endpoint, observation.endpoint) != expected
                or _page_url(canonicals[0].get("href"), document["url"]) != expected):
            return reject("contentful_canonical_does_not_bind_requested_page")
        record, pointer = (_contentful_publication(soup, source_format="Gallery",
            title=_plain(str(heading)), summary=_plain(descriptions[0].get("content")))
            if gallery else _contentful_publication(soup))
        title, description = _plain(record["title"]), _plain(record["summary"])
        og_title = _plain(titles[0].get("content"))
        if (not title or title != _plain(str(heading))
                or not (og_title == title or og_title.startswith(title+" - "))
                or not description or description != _plain(descriptions[0].get("content"))):
            return reject("contentful_visible_title_or_summary_conflicts_with_metadata")
        raw_published = record["publishDate"]["date"]
        published = _timestamp(raw_published)
        raw_day = _plain(str(stamp))
        day = datetime.strptime(raw_day, "%d %b %y").date()
        if published is None or published.date() != day:
            return reject("contentful_publication_conflicts_with_visible_date")
        if published > observation.observed_at:
            return reject("contentful_publication_after_actual_capture")
    except (ValueError, TypeError, KeyError, OverflowError, AttributeError, RecursionError):
        return reject("invalid_or_ambiguous_contentful_publication")
    # Related cards are outside the registered Article component. Only this
    # semantic rich-text article can supply prediction-visible body paragraphs.
    if gallery:
        # The actual gallery has no rich-text body. Its visible header summary
        # is an authored introduction; image captions remain excluded.
        header = heading.parent
        paragraphs = header.find_all("p", recursive=False)
        if (stamp not in header.descendants or len(paragraphs) != 1
                or _excluded(paragraphs[0]) or _plain(str(paragraphs[0])) != description):
            return reject("contentful_gallery_unique_visible_header_introduction_required")
        body = header
    else:
        paragraphs = body.find_all("p")
    if len(paragraphs) > 1024:
        return reject("contentful_dom_paragraph_bound_exceeded")
    selected = []
    for index, paragraph in enumerate(paragraphs):
        if _excluded(paragraph, body):
            continue
        clean = BeautifulSoup(str(paragraph), "html.parser")
        for node in list(clean.find_all(True)):
            if node.name and _excluded(node):
                node.decompose()
        text = _plain(str(clean))
        if text:
            selected.append((index, _selector(paragraph), text))
    minimum = 1 if gallery else 2
    if len(selected) < minimum:
        return reject("contentful_two_body_paragraphs_required")
    adapter_version = GALLERY_VERSION if gallery else VERSION
    binding = {"version": adapter_version, "metadata_json_pointer": pointer, "source_url": document["url"],
        "declared_page_urls": [expected], "canonical_url": expected,
        "metadata_headline": title, "heading_text": title, "heading_css_selector": _selector(heading),
        "container_css_selector": _selector(body), "date_css_selector": _selector(stamp),
        "container_selection": "same_page_contentful_article_header_rich_text_and_flight_publication",
        "binding_scope": "page_identity_and_source_markup_not_match_scope_or_semantic_truth"}
    if gallery:
        binding.update(container_selection="same_page_contentful_gallery_header_introduction",
            source_format="Gallery", source_text_scope="gallery_introduction_only_captions_excluded")
    publication = {"source_published_at": published.isoformat(), "source_modified_at": None,
        "publication_raw": raw_published, "modified_raw": None, "source_publication_date": day.isoformat(),
        "source_publication_local_time": None, "publication_timezone": "source_timestamp"}
    parts, segments, offset = [], [], 0
    for index, path, text in selected[:MAX_SEGMENTS]:
        remaining = MAX_TEXT-offset-(1 if parts else 0)
        if remaining <= 0:
            break
        value, start = text[:remaining], offset+(1 if parts else 0)
        segments.append({**publication, "text": value, "start": start, "end": start+len(value),
            "content_source": "semantic_html_article_paragraph", "content_role": "paragraph",
            "css_selector": path, "original_dom_paragraph_index": index, "source_url": document["url"],
            "json_pointer": None, "body_binding": deepcopy(binding), "truncated": len(value)<len(text)})
        if gallery:
            segments[-1]["paragraph_kind"] = "gallery_introduction"
        parts.append(value)
        offset = start+len(value)
    if len(segments) < minimum:
        return reject("contentful_body_outside_projection_bound")
    metadata.update(role="article", segments=segments, headline=title, body_binding=binding,
        html_body_adapter=adapter_version, excluded_count=len(metadata["excluded"]))
    if gallery:
        metadata.update(source_format="Gallery", source_text_scope="gallery_introduction_only_captions_excluded")
    document.update(kind="article", status="OK", title=title, text="\n".join(parts),
        source_published_at=published.isoformat(), publication_status="source_article_metadata",
        text_truncated=len(segments)<len(selected) or any(s["truncated"] for s in segments))
    return document


def project_html_metadata_body(document, html, observation, metadata, *, allow_gallery=False):
    """Admit exact source paragraphs only after all independent bindings agree."""
    from .news_discovery import _plain, _timestamp, MAX_SEGMENTS, MAX_TEXT

    def visible_text(node):
        clean = BeautifulSoup(str(node), "html.parser")
        for child in list(clean.find_all(True)):
            if child.name and _excluded(child):
                child.decompose()
        return _plain(str(clean))

    def reject(reason):
        metadata["excluded"].append({"reason": reason})
        metadata.update(role="unknown", segments=[], excluded_count=len(metadata["excluded"]),
                        html_body_adapter=VERSION)
        # An unrecognized source remains unknown. Its flattened page text is
        # never advertised as typed body and cannot bypass visible-state gates.
        return document

    soup = BeautifulSoup(html, "html.parser")
    if (metadata["excluded"] or metadata["jsonld_parse_errors"] or metadata["nodes_truncated"]
            or soup.find_all("script", type="application/ld+json")):
        return reject("html_metadata_cannot_bypass_jsonld_or_excluded_metadata")
    if len(soup.find_all(True)) > 20000 or len(soup.find_all("head")) != 1:
        return reject("single_bounded_metadata_head_required")
    if soup.select(CONTENTFUL_ROOT_SELECTOR):
        return _project_contentful_body(document, soup, observation, metadata, reject, allow_gallery=allow_gallery)
    head = soup.head

    def field(name, *, required=True):
        matches = [n for n in soup.find_all("meta") if n.get("property") == name or n.get("name") == name]
        if not matches and not required:
            return None, None
        if len(matches) != 1 or matches[0].parent is not head:
            raise ValueError("unique_head_metadata_field_required")
        value = matches[0].get("content")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("nonempty_metadata_field_required")
        return value, matches[0]

    try:
        raw_url, url_node = field("og:url")
        raw_title, title_node = field("og:title")
        raw_description, description_node = field("og:description")
        kind, _ = field("og:type")
        if kind not in {"website", "article"}:
            return reject("unsupported_open_graph_page_type")
        expected = _page_url(document["url"], document["url"])
        if (observation.source != "open_page" or observation.params
                or _page_url(observation.endpoint, observation.endpoint) != expected
                or _page_url(raw_url, document["url"]) != expected):
            return reject("open_graph_url_does_not_bind_requested_page")
        canonical_nodes = soup.find_all("link", rel="canonical")
        canonicals = [_page_url(n.get("href"), document["url"]) for n in canonical_nodes]
        if len(canonicals) > 1 or any(n.parent is not head for n in canonical_nodes) or any(u != expected for u in canonicals):
            return reject("page_canonical_conflicts_with_requested_article")
        parts = urlsplit(document["url"])
        match = _DATED_PATH.fullmatch(parts.path)
        if not match or parts.query or parts.fragment or not any(c.isalpha() for c in match[4]):
            return reject("explicit_dated_article_permalink_required")
        day = date(*map(int, match.group(1, 2, 3)))
        raw_published, published_node = field("article:published_time", required=False)
        raw_modified, modified_node = field("article:modified_time", required=False)
        published, modified = _timestamp(raw_published), _timestamp(raw_modified)
        if any(t is not None and t > observation.observed_at for t in (published, modified)):
            return reject("article_metadata_time_after_actual_capture")
        if published and modified and modified < published:
            return reject("article_metadata_modification_precedes_publication")
        if published and datetime.fromisoformat(raw_published.strip().replace("Z", "+00:00")).date() != day:
            return reject("publication_timestamp_conflicts_with_source_date")
    except (ValueError, TypeError, OverflowError, AttributeError):
        return reject("invalid_or_ambiguous_html_article_metadata")
    earliest = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)-timedelta(hours=14)
    if earliest > observation.observed_at:
        return reject("local_publication_certainly_after_actual_capture")
    if modified is not None and modified < earliest:
        return reject("article_modification_precedes_earliest_possible_publication")
    # Never pick a large arbitrary div: the lead/body markers must be direct
    # siblings of the sole visible H1 and publication details in one region.
    roots = [n for n in soup.select(ROOT_SELECTOR) if n.select(":scope > .news-text") and not _excluded(n)]
    headings = [n for n in soup.find_all("h1") if not _excluded(n)]
    if len(roots) != 1 or len(headings) != 1:
        return reject("unique_visible_article_region_and_heading_required")
    root, heading = roots[0], headings[0]
    mains = [n for n in soup.find_all("main") if not _excluded(n)]
    if len(mains) != 1 or mains[0] not in root.parents:
        return reject("single_visible_main_must_contain_article_region")
    lead, bodies, details = [root.select(selector) for selector in (*BODY_SELECTORS, ":scope > .details")]
    if (heading.parent is not root or len(lead) != 1 or len(bodies) != 1 or len(details) != 1
            or any(_excluded(n) for n in [lead[0], bodies[0], details[0]])):
        return reject("direct_article_heading_date_lead_and_body_required")
    order = [n for n in root.children if isinstance(n, Tag)]
    positions = [next(i for i, child in enumerate(order) if child is node)
                 for node in (heading, details[0], lead[0], bodies[0])]
    if positions != sorted(positions):
        return reject("article_title_date_lead_body_order_required")
    if (_plain(raw_title) != visible_text(heading) or not _plain(raw_title)
            or _plain(raw_description) != visible_text(lead[0]) or not _plain(raw_description)):
        return reject("visible_article_title_or_lead_conflicts_with_metadata")
    dates = details[0].select(":scope > .date")
    if len(dates) != 1 or _excluded(dates[0]):
        return reject("unique_visible_article_local_date_required")
    raw_date = visible_text(dates[0])
    try:
        if not re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", raw_date):
            raise ValueError("invalid local date")
        displayed_day = date(*reversed(list(map(int, raw_date.split(".")))))
        if displayed_day != day:
            return reject("visible_article_date_conflicts_with_permalink")
    except ValueError:
        return reject("invalid_visible_article_local_date")
    for node in soup.find_all(["article", "main"]):
        if node in root.parents or node is root or _excluded(node):
            continue
        if len([p for p in node.find_all("p") if not _excluded(p)]) >= 2:
            return reject("competing_semantic_article_body")
    paragraphs = root.find_all("p")
    if len(paragraphs) > 1024 or len(_selector(root)) > 512:
        return reject("article_dom_paragraph_or_selector_bound_exceeded")
    selected, excluded = [], []
    for index, paragraph in enumerate(paragraphs):
        path, reason = _selector(paragraph), None
        if not any(owner in paragraph.parents for owner in (lead[0], bodies[0])) or _excluded(paragraph, root):
            reason = "outside_registered_article_body_or_non_body_structure"
        elif len(path) > 512:
            reason = "paragraph_selector_bound_exceeded"
        else:
            clean = BeautifulSoup(str(paragraph), "html.parser")
            for child in list(clean.find_all(True)):
                if child.name and (_excluded(child) or child.name in {"h1", "h2", "h3", "h4", "h5", "h6", "time"}):
                    child.decompose()
            text = _plain(str(clean))
            links = " ".join(_plain(str(n)) for n in clean.find_all("a"))
            if not text or re.fullmatch(r"[\d\s/:.,-]+", text):
                reason = "empty_or_date_only_paragraph"
            elif paragraph.find_parent("a") is not None or text == links:
                reason = "standalone_link_card_paragraph"
            else:
                selected.append((index, path, text))
        if reason:
            excluded.append({"original_dom_paragraph_index": index, "reason": reason})
    if len(selected) < 2 or any(not any(owner in paragraphs[index].parents for index, _, _ in selected)
                              for owner in (lead[0], bodies[0])):
        return reject("at_least_two_bound_paragraphs_including_news_body_required")
    binding = {"version": VERSION, "metadata_json_pointer": None, "source_url": document["url"],
        "declared_page_urls": [expected], "canonical_url": canonicals[0] if canonicals else None,
        "metadata_headline": _plain(raw_title), "heading_text": visible_text(heading),
        "heading_css_selector": _selector(heading), "container_css_selector": _selector(root),
        "metadata_css_selectors": {"url": _selector(url_node), "title": _selector(title_node),
            "description": _selector(description_node)}, "date_css_selector": _selector(dates[0]),
        "body_css_selectors": [_selector(n) for n in (lead[0], bodies[0])],
        "container_selection": "unique_same_page_og_title_date_lead_news_body",
        "binding_scope": "page_identity_and_source_markup_not_match_scope_or_semantic_truth"}
    publication = {"source_published_at": published.isoformat() if published else None,
        "source_modified_at": modified.isoformat() if modified else None,
        "publication_raw": raw_published or raw_date, "modified_raw": raw_modified,
        "source_publication_date": day.isoformat(), "source_publication_local_time": None,
        "publication_timezone": "source_timestamp" if published else "unknown"}
    parts, segments, offset = [], [], 0
    for index, path, text in selected[:MAX_SEGMENTS]:
        remaining = MAX_TEXT-offset-(1 if parts else 0)
        if remaining <= 0:
            break
        value, start = text[:remaining], offset+(1 if parts else 0)
        segments.append({**publication, "text": value, "start": start, "end": start+len(value),
            "content_source": "semantic_html_article_paragraph", "content_role": "paragraph",
            "css_selector": path, "original_dom_paragraph_index": index, "source_url": document["url"],
            "json_pointer": None, "body_binding": deepcopy(binding), "truncated": len(value)<len(text)})
        parts.append(value)
        offset = start+len(value)
    if len(segments) < 2:
        return reject("body_paragraphs_outside_projection_bound")
    metadata.update(role="article", segments=segments, html_body_adapter=VERSION,
        body_binding=binding, excluded_dom_paragraphs=excluded, excluded_count=len(metadata["excluded"]))
    document.update(kind="article", status="OK", text="\n".join(parts),
        source_published_at=publication["source_published_at"],
        publication_status="source_article_metadata" if published else "source_local_date_timezone_unknown",
        text_truncated=len(segments)<len(selected) or any(s["truncated"] for s in segments))
    return document
