"""Opt-in v12 adapters for already archived official CMS responses.

Only JSON/literal data and explicit article body regions are read. No JavaScript,
extra request, relevance judgement or numerical value estimate is involved.
"""
from copy import deepcopy
import json
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from .news_html_body import _page_url, _selector, _within_blocked

VERSION = "official-cms-response-projection-v1"
CARD_BASIS = "same_response_cms_article_card_v1"


def _json(raw):
    if len(raw) > 2_000_000:
        raise ValueError("CMS payload bound exceeded")
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _walk(value, pointer="$", depth=0, counter=None):
    counter = [0] if counter is None else counter
    counter[0] += 1
    if depth > 64 or counter[0] > 50000:
        raise ValueError("CMS traversal bound exceeded")
    if isinstance(value, dict):
        yield pointer, value
        for key, item in value.items():
            yield from _walk(item, pointer + "." + key, depth + 1, counter)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _walk(item, pointer + f"[{i}]", depth + 1, counter)


def _payloads(soup, host):
    if host == "www.chelseafc.com":
        for i, node in enumerate(soup.select("[data-props]")[:256]):
            try:
                yield f"data-props[{i}]", _json(node["data-props"])
            except (ValueError, RecursionError):
                continue
    elif host == "www.brentfordfc.com":
        for i, node in enumerate(soup.find_all("script")[:512]):
            raw = node.get_text().strip()
            if len(raw) > 2_000_000:
                continue
            match = re.fullmatch(r"self\.__next_f\.push\((\[.*\])\);?", raw, re.S)
            if not match:
                continue
            try:
                envelope = _json(match[1])
            except (ValueError, RecursionError):
                continue
            if (not isinstance(envelope, list) or len(envelope) != 2
                    or envelope[0] != 1 or not isinstance(envelope[1], str)):
                continue
            for j, row in enumerate(envelope[1].splitlines()):
                key, sep, payload = row.partition(":")
                if not sep or not re.fullmatch(r"[0-9a-f]+", key):
                    continue
                try:
                    yield f"next_f[{i}].row[{j}]", _json(payload)
                except (ValueError, RecursionError):
                    continue


def project_index_cards(document, html, allowed_hosts, *, max_links=320):
    """Expose response-local article metadata before navigation links.

    The unopened target contributes nothing. A missing snippet remains missing.
    CMS card titles are explicitly distinct from visible DOM anchor labels.
    """
    from .inquiry_tools import public_url
    host = urlsplit(document["url"]).hostname
    if host not in {"www.brentfordfc.com", "www.chelseafc.com"}:
        return document
    soup = BeautifulSoup(html, "html.parser")
    cards, seen = [], set()
    try:
        for root, payload in _payloads(soup, host):
            for pointer, obj in _walk(payload):
                if host == "www.chelseafc.com":
                    title, path = obj.get("title"), obj.get("url")
                    if not isinstance(path, str) or not path.startswith("/en/news/article/"):
                        continue
                else:
                    title, slug = obj.get("articleTitle"), obj.get("slug")
                    if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9-]+", slug):
                        continue
                    path = "/en/news/" + slug
                if not isinstance(title, str) or not title.strip():
                    continue
                try:
                    url = public_url(urljoin(document["url"], path), allowed_hosts)
                except (TypeError, ValueError):
                    continue
                if urlsplit(url).hostname != host or url in seen:
                    continue
                seen.add(url)
                cards.append({"url": url, "title": title[:512], "snippet": None,
                    "snippet_status": "not_projected", "directory_context_basis": CARD_BASIS,
                    "metadata_pointer": root + pointer, "target_article_fetched_for_context": False,
                    "metadata_visibility": "serialized_same_response_not_rendered_DOM_anchor",
                    "observed_metadata_labels": [{"text": title[:512],
                        "text_truncated": len(title) > 512, "pointer": root + pointer}]})
    except (ValueError, RecursionError):
        # Partial traversal cannot silently become a complete discovery result.
        document["cms_card_projection"] = {"version": VERSION, "status": "bound_exceeded"}
        return document
    old = document.get("links", [])
    merged = cards + [row for row in old if row.get("url") not in seen]
    document.update(links=merged[:max_links], eligible_link_count=len(merged),
                    links_truncated=len(merged) > max_links,
                    cms_card_projection={"version": VERSION, "status": "completed",
                        "cards": len(cards), "order": "CMS_response_order_then_DOM_navigation",
                        "target_requests": 0, "utility_estimated": False})
    return document


def _same_page(soup, url):
    expected = _page_url(url, url)
    canonicals = soup.select('link[rel="canonical"]')
    if not canonicals and urlsplit(url).hostname == "www.chelseafc.com":
        explicit = soup.select('meta[property="og:url"]')
        if len(explicit) == 1 and _page_url(explicit[0].get("content"), url) == expected:
            return expected
    if len(canonicals) != 1 or _page_url(canonicals[0].get("href"), url) != expected:
        raise ValueError("CMS page canonical binding failed")
    return expected


def _publication(published, modified, observed):
    from .news_discovery import _timestamp
    p, m = _timestamp(published), _timestamp(modified)
    if p is None or p > observed or m is not None and (m > observed or m < p):
        raise ValueError("CMS article publication invalid or after response observation")
    return {"source_published_at": p.isoformat(), "source_modified_at": m.isoformat() if m else None,
            "publication_raw": published, "modified_raw": modified}


def project_cms_article(document, html, observation, metadata):
    """Return None for an unregistered source, otherwise a bound body or NO_DATA."""
    from .news_discovery import _plain, MAX_SEGMENTS, MAX_TEXT
    host = urlsplit(document["url"]).hostname
    if host not in {"www.fulhamfc.com", "www.brentfordfc.com", "www.chelseafc.com"}:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        expected = _same_page(soup, document["url"])
        paragraphs = []
        if host == "www.fulhamfc.com":
            from football_research.data_adapters.fulham_nuxt import extract_archived_news
            article = extract_archived_news(html, document["url"])
            title = article["title"]
            publication = _publication(article["source_published_at"], None, observation.observed_at)
            cursor = 0
            for fragment in article["body_pointers"]:
                for i in range(fragment["paragraph_count"]):
                    paragraphs.append({"text": article["paragraphs"][cursor],
                        "content_source": "cms_literal_article_paragraph",
                        "json_pointer": fragment["pointer"], "literal_paragraph_index": i})
                    cursor += 1
            identity_basis = article["title_binding_basis"]
        else:
            if host == "www.brentfordfc.com":
                nodes = []
                for script in soup.find_all("script", type="application/ld+json")[:32]:
                    for _, value in _walk(_json(script.get_text())):
                        if value.get("@type") == "NewsArticle":
                            nodes.append(value)
                heads = soup.find_all("h1")
                roots = soup.select('[class*="articleBody_articleBody__"]')
                if len(nodes) != 1 or len(heads) != 1 or len(roots) != 1:
                    raise ValueError("unique Brentford article binding required")
                node = nodes[0]
                title = _plain(node.get("headline"))
                page = node.get("mainEntityOfPage", {})
                page = page.get("@id") if isinstance(page, dict) else page
                if not title or title != heads[0].get_text(" ", strip=True) or _page_url(page, expected) != expected:
                    raise ValueError("Brentford article identity mismatch")
                publication = _publication(node.get("datePublished"), node.get("dateModified"), observation.observed_at)
                root = roots[0]
                selected = root.find_all("p")
                identity_basis = "unique_JSONLD_url_headline_h1_and_registered_articleBody_container"
            else:
                og_urls = soup.select('meta[property="og:url"]')
                og_titles = soup.select('meta[property="og:title"]')
                headers = soup.select('[data-component="ArticleHeader"][data-props]')
                roots = soup.select('article.news-article-page')
                if any(len(x) != 1 for x in (og_urls, og_titles, headers, roots)):
                    raise ValueError("unique Chelsea article binding required")
                details = _json(headers[0]["data-props"])["articleHeaderDetails"]
                title = details["title"]
                if (not isinstance(title, str) or not title.strip() or title != og_titles[0].get("content")
                        or _page_url(og_urls[0].get("content"), expected) != expected):
                    raise ValueError("Chelsea article identity mismatch")
                publication = _publication(details.get("date"), None, observation.observed_at)
                root = roots[0]
                selected = root.select('section.news-article-page__body contentful-rich-text p')
                identity_basis = "same_page_OG_url_title_ArticleHeader_and_article_body_canonical_if_present"
            # Ancestor theme classes (e.g. gam__share-block-theme) do not make
            # the whole article a share widget. Check explicit visibility on
            # ancestors, and structural non-body regions inside the body.
            hidden_root = any(n.has_attr("hidden") or str(n.get("aria-hidden", "")).casefold() == "true"
                or n.name in {"nav", "footer", "aside", "template", "script", "style"}
                or any(s in re.sub(r"\s+", "", str(n.get("style", ""))).casefold()
                       for s in ("display:none", "visibility:hidden"))
                for n in [root, *root.parents])
            if len(selected) > 1024 or hidden_root:
                raise ValueError("CMS DOM body bound or visibility failed")
            for index, node in enumerate(selected):
                if _within_blocked(node, root):
                    continue
                text = _plain(str(node))
                link_text = " ".join(_plain(str(n)) for n in node.find_all("a"))
                if not text or text == link_text or node.find_parent("a") is not None:
                    continue
                selector = _selector(node)
                if len(selector) > 512:
                    raise ValueError("CMS paragraph selector bound exceeded")
                paragraphs.append({"text": text, "content_source": "semantic_html_article_paragraph",
                    "css_selector": selector, "original_dom_paragraph_index": index, "json_pointer": None})
        if len(paragraphs) < 2:
            raise ValueError("at least two bound CMS body paragraphs required")
        binding = {"version": VERSION, "source_url": expected, "metadata_headline": title,
            "identity_basis": identity_basis, "javascript_executed": False,
            "scope": "source_body_presence_not_match_truth_or_calibrated_value"}
        parts, segments, offset = [], [], 0
        for paragraph in paragraphs[:MAX_SEGMENTS]:
            start = offset + bool(parts)
            remaining = MAX_TEXT - start
            if remaining <= 0:
                break
            text = paragraph["text"][:remaining]
            segments.append({**paragraph, **publication, "text": text,
                "start": start, "end": start + len(text), "content_role": "paragraph",
                "source_url": document["url"], "body_binding": deepcopy(binding),
                "truncated": len(text) < len(paragraph["text"])})
            parts.append(text)
            offset = start + len(text)
        metadata.update(role="article", segments=segments, body_binding=binding,
                        html_body_adapter=VERSION, excluded_count=0)
        document.update(kind="article", status="OK", text="\n".join(parts),
            source_published_at=publication["source_published_at"],
            source_modified_at=publication["source_modified_at"], publication_status="source_article_metadata",
            text_truncated=len(segments) < len(paragraphs) or any(x["truncated"] for x in segments))
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        metadata.update(role="unknown", segments=[], excluded=[{"reason": "cms_body_binding_rejected",
                        "detail": str(exc)[:160]}], excluded_count=1, html_body_adapter=VERSION)
        document.update(kind="page", status="NO_DATA", text="", text_truncated=False)
    return document
