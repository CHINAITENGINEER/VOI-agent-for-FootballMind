"""Live-only, versioned DOM adapters for observed public club templates.

No change to historical NewsDiscoveryTools projection versions. Unknown markup
continues to be unknown; source timestamps without offsets remain local text.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from football_agent.news_html_metadata import _excluded

VERSION = "footballmind-live-webflow-news-body-v1"
DIRECTORY_VERSION = "footballmind-live-literal-directory-cards-v1"


def project_live_directory(document, raw, observation, allowed_hosts):
    """Expose literal same-response card metadata without executing Vue or JS.

    Some official club pages render their entire news archive from a JSON array
    in a component attribute rather than href anchors. Candidate article reads
    remain separate actions; missing publication offsets remain unknown.
    """
    if observation.http_status != 200 or observation.error_class:
        return document
    # URL deduplication may retain an image-only first anchor. The v4+ link
    # provenance already contains later visible labels for that exact URL;
    # retain the first nonempty observed label, without deriving a title from
    # an image alt, URL slug, another card or unobserved target article.
    document = deepcopy(document)
    restored = []
    for link in document.get("links", []):
        if str(link.get("title") or "").strip():
            continue
        labels = [row for row in link.get("observed_anchors", [])
                  if isinstance(row.get("text"), str) and row["text"].strip()]
        if not labels:
            continue
        label = labels[0]  # retain_link_labels preserves original anchor order.
        link.update(title=label["text"].strip()[:512], title_source="same_url_observed_visible_anchor",
                    title_source_anchor_index=label.get("anchor_index"),
                    title_source_truncated=label.get("text_truncated", False))
        restored.append({"url":link["url"],"anchor_index":label.get("anchor_index")})
    if restored:
        document["live_directory_label_projection"] = {
            "version":"same-url-visible-anchor-title-v1", "restored":restored,
            "order":"first_nonempty_observed_anchor_for_same_url", "target_requests":0}
    from .live_spanish_body import project_spanish_directory
    document = project_spanish_directory(document, raw, observation)
    from football_agent.inquiry_tools import public_url
    from football_agent.news_discovery import MAX_INDEX_LINKS
    soup = BeautifulSoup(raw, "html.parser")
    host = urlsplit(document["url"]).hostname
    cards, seen, rejected, records = [], set(), [], 0
    sources = []
    for index, component in enumerate(soup.find_all("load-filter-component")):
        value = component.get(":items")
        if value is not None:
            sources.append((f"load-filter-component[{index}].:items", value))
    # Standard JSON-LD ItemList is portable across official league/club indexes.
    for index, script in enumerate(soup.find_all("script", type="application/ld+json")[:32]):
        value = script.get_text()
        if len(value) > 2_000_000:
            continue
        try:
            node = json.loads(value)
        except (ValueError, RecursionError):
            continue
        lists = node if isinstance(node, list) else node.get("@graph", [node]) if isinstance(node, dict) else []
        for position, record in enumerate(lists):
            if isinstance(record, dict) and record.get("@type") == "ItemList":
                sources.append((f"jsonld[{index}][{position}].itemListElement", record.get("itemListElement")))
    if not sources:
        return document
    try:
        for pointer, payload in sources:
            if isinstance(payload, str):
                if len(payload) > 4_000_000:
                    raise ValueError("literal directory payload bound exceeded")
                payload = json.loads(payload)
            if not isinstance(payload, list) or len(payload) > 2048:
                raise ValueError("literal directory record bound exceeded")
            for i, record in enumerate(payload):
                records += 1
                if records > 2048:
                    raise ValueError("combined directory record bound exceeded")
                if not isinstance(record, dict):
                    continue
                item = record.get("item", record)
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or item.get("headline") or item.get("name")
                target = item.get("url") or item.get("@id")
                if not isinstance(title, str) or not title.strip() or not isinstance(target, str):
                    continue
                try:
                    url = public_url(urljoin(document["url"], target), allowed_hosts)
                    if urlsplit(url).hostname != host:
                        raise ValueError("cross-host directory card")
                except (ValueError, TypeError):
                    rejected.append({"pointer":f"{pointer}[{i}]","reason":"unsafe_or_cross_host_url"})
                    continue
                if url in seen:
                    continue
                seen.add(url)
                stamp = item.get("datePublished")
                published = None
                if isinstance(stamp, str):
                    try:
                        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                        if parsed.tzinfo is not None and parsed <= observation.observed_at:
                            published = parsed.isoformat()
                    except ValueError:
                        pass
                cards.append({"url":url,"title":title.strip()[:512],"source_published_at":published,
                    "publication_raw":stamp or item.get("postDate"),"source_available_at":None,
                    "directory_metadata_adapter":DIRECTORY_VERSION,"metadata_pointer":f"{pointer}[{i}]",
                    "metadata_visibility":"literal_same_response_not_rendered_anchor",
                    "source_category":item.get("newsCategory"),"snippet":None,
                    "target_article_fetched_for_context":False})
    except (ValueError, TypeError, RecursionError) as exc:
        result = deepcopy(document)
        result["live_directory_projection"] = {"version":DIRECTORY_VERSION,"status":"rejected","reason":str(exc)[:160]}
        return result
    merged = cards + [link for link in document.get("links", []) if link.get("url") not in seen]
    result = deepcopy(document)
    result.update(links=merged[:MAX_INDEX_LINKS],eligible_link_count=len(merged),links_truncated=len(merged)>MAX_INDEX_LINKS,
        live_directory_projection={"version":DIRECTORY_VERSION,"status":"completed","records_seen":records,
            "cards":len(cards),"cards_retained":min(len(cards),MAX_INDEX_LINKS),"rejected":rejected,
            "network_requests":0,"target_article_requests":0,"rendered_visibility_verified":False})
    return result


def _project_visible_article(document, raw, observation):
    """Live-only same-page HTML body when no JSON-LD Article is supplied.

    Generic pages need explicit og:type=article, a unique canonical/heading and
    one semantic article. The Fortuna template has independently observed exact
    selectors and keeps its date-only statement separate from availability.
    """
    from football_agent.news_discovery import MAX_SEGMENTS, MAX_TEXT
    from football_agent.news_html_body import _selector
    from football_agent.link_context import _hidden
    soup = BeautifulSoup(raw, "html.parser")
    parts = urlsplit(document.get("url", ""))
    expected = document.get("url")
    if observation.http_status != 200 or observation.error_class or observation.params or observation.endpoint != expected:
        return document
    # A new fallback must not override contradictory/malformed Article records.
    for script in soup.find_all("script", type="application/ld+json")[:32]:
        try:
            node = json.loads(script.get_text())
        except (ValueError, RecursionError):
            return document
        if re.search(r'"(?:Article|NewsArticle|BlogPosting|LiveBlogPosting)"', json.dumps(node)):
            return document
    canonical = soup.select('head link[rel="canonical"]')
    og_title = soup.select('head meta[property="og:title"]')
    if len(canonical) != 1 or canonical[0].get("href", "").rstrip("/") != str(expected).rstrip("/") or len(og_title) != 1:
        return document
    fortuna = parts.hostname == "fortunasittard.nl" and parts.path.startswith("/nieuws/")
    if fortuna:
        heads = soup.select(".news-item__header h1.news-item__title")
        containers = soup.select(".intro-module__primary-text, .intro-module__secondary-text")
        date_nodes = soup.select(".news-item__header .news-item__date")
        if len(heads) != 1 or len(containers) != 2 or len(date_nodes) != 1:
            return document
        local_date = date_nodes[0].get_text(" ", strip=True)
        adapter = "fortuna-visible-news-body-v1"
    else:
        types = soup.select('head meta[property="og:type"]')
        articles = [n for n in soup.find_all("article") if not _excluded(n)]
        if len(types) != 1 or types[0].get("content", "").casefold() != "article" or len(articles) != 1:
            return document
        containers = articles
        heads = [n for n in articles[0].find_all("h1") if not any(_hidden(p) for p in [n,*n.parents])]
        local_date = None
        adapter = "same-page-og-semantic-article-body-v1"
    if (len(heads) != 1 or any(_excluded(n) for n in containers)
            or any(_hidden(p) for n in heads for p in [n,*n.parents])):
        return document
    title = heads[0].get_text(" ", strip=True)
    head_title = og_title[0].get("content", "").strip()
    if not title or not (head_title == title or head_title.startswith(title + " | ") or head_title.startswith(title + " - ")):
        return document
    paragraphs = [n for root in containers for n in root.find_all("p") if not _excluded(n, root)]
    if not 2 <= len(paragraphs) <= 1024:
        return document
    selected=[]
    for paragraph in paragraphs:
        clean=BeautifulSoup(str(paragraph),"html.parser")
        for node in list(clean.find_all(True)):
            if node.name and _excluded(node):
                node.decompose()
        text=clean.get_text(" ",strip=True)
        if text:selected.append((paragraph,text))
    if len(selected)<2 or sum(len(text) for _,text in selected)<160:
        return document
    published = None
    publication_raw = local_date
    stamps = soup.select('head meta[property="article:published_time"], head meta[name="article:published_time"]')
    if stamps:
        if len(stamps) != 1:
            return document
        publication_raw = stamps[0].get("content")
        try:
            parsed = datetime.fromisoformat(str(publication_raw).replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed > observation.observed_at:
                return document
            published = parsed.isoformat()
        except ValueError:
            return document
    language = soup.html.get("lang") if soup.html else None
    segments=[];texts=[];offset=0
    for index,(node,text) in enumerate(selected[:MAX_SEGMENTS]):
        if offset>=MAX_TEXT:break
        value=text[:MAX_TEXT-offset];start=offset+(1 if texts else 0)
        value=value[:max(0,MAX_TEXT-start)]
        if not value:break
        segments.append({"text":value,"start":start,"end":start+len(value),
            "content_source":"same_page_visible_html_paragraph","content_role":"paragraph",
            "paragraph_index":index,"css_selector":_selector(node),"source_url":expected,
            "source_published_at":published,"publication_raw":publication_raw,"truncated":value!=text})
        texts.append(value);offset=start+len(value)
    result=deepcopy(document)
    result.update(kind="article",status="OK",title=title,text="\n".join(texts),source_published_at=published,
        publication_status="declared" if published else "date_only_time_zone_unknown" if local_date else "unknown",
        source_publication_local_time=local_date,text_truncated=len(texts)<len(selected) or any(s["truncated"] for s in segments))
    result["structured_news"]={"version":adapter,"role":"article","segments":segments,
        "headline":title,"html_body_adapter":adapter,"excluded":[],
        "source_language":language,"language_source":"html@lang" if language else None,
        "previous_projection":document.get("structured_news"),
        "body_binding":{"canonical_url":expected,"heading_text":title,
            "binding_scope":"same_page_metadata_and_visible_body_not_match_truth_or_utility"}}
    return result


def project_live_article(document, raw, observation):
    if (document.get("structured_news") or {}).get("role") != "article":
        from .live_spanish_body import project_spanish_body
        projected = project_spanish_body(document, raw, observation)
        if (projected.get("structured_news") or {}).get("role") == "article":
            return projected
        document = projected
    if (document.get("structured_news") or {}).get("role") != "article":
        from .live_dutch_body import project_dutch_body
        projected = project_dutch_body(document, raw, observation)
        if (projected.get("structured_news") or {}).get("role") == "article":
            return projected
        document = projected
    if (document.get("structured_news") or {}).get("role") != "article":
        from .live_venezia_body import project_venezia_body
        projected = project_venezia_body(document, raw, observation)
        if (projected.get("structured_news") or {}).get("role") == "article":
            return projected
        document = projected
    if (document.get("structured_news") or {}).get("role") != "article":
        projected = _project_visible_article(document, raw, observation)
        if (projected.get("structured_news") or {}).get("role") == "article":
            return projected
    url = document.get("url", "")
    parts = urlsplit(url)
    if (parts.hostname != "www.gais.se" or not re.fullmatch(r"/nyheter/[^/]+/?", parts.path)
            or parts.query or observation.http_status != 200 or observation.error_class
            or (document.get("structured_news") or {}).get("role") == "article"):
        return document
    soup = BeautifulSoup(raw, "html.parser")
    headers = soup.select(".news-p-header")
    bodies = soup.select(".news-p-content > .rich-text-article")
    canonicals = soup.select('head link[rel="canonical"]')
    titles = soup.select('head meta[property="og:title"]')
    if (len(headers) != 1 or len(bodies) != 1 or len(canonicals) != 1 or len(titles) != 1
            or canonicals[0].get("href") != url or observation.endpoint != url
            or observation.params or _excluded(headers[0].parent) or _excluded(bodies[0])):
        return document
    header = headers[0]
    style = re.sub(r"\s+", "", header.get("style", "")).lower()
    if header.has_attr("hidden") or header.get("aria-hidden") == "true" or "display:none" in style or "visibility:hidden" in style:
        return document
    headings = headers[0].select("h1")
    if len(headings) != 1 or _excluded(headings[0], headers[0]):
        return document
    plain = lambda node: re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
    title = plain(headings[0])
    if titles[0].get("content", "").strip() != "GAIS | "+title:
        return document
    # Accept only the observed site-level JSON-LD. An Article/LiveBlog or
    # malformed JSON must still go through the existing typed article parser.
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            obj = json.loads(script.get_text())
            kinds = obj.get("@type", [])
            kinds = [kinds] if isinstance(kinds, str) else kinds
            if not kinds or not set(kinds) <= {"Organization", "SportsTeam"}:
                return document
        except (ValueError, TypeError, AttributeError):
            return document
    stamps = [plain(n) for n in headers[0].select(".news-p-header-bottom-text")
              if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", plain(n))]
    if len(stamps) != 1:
        return document
    try:
        local_time = datetime.strptime(stamps[0], "%Y-%m-%d %H:%M")
    except ValueError:
        return document
    # Reject dates certainly in the future without inventing the site's zone.
    if local_time.replace(tzinfo=timezone.utc)-timedelta(hours=14) > observation.observed_at:
        return document
    paragraphs = bodies[0].find_all("p")
    if len(paragraphs) > 1024:
        return document
    selected = [(i, plain(p)) for i, p in enumerate(paragraphs)
                if not _excluded(p) and any(c.isalnum() for c in plain(p))]
    if len(selected) < 2:
        return document
    parts, segments, offset = [], [], 0
    for i, text in selected[:64]:
        start = offset + bool(parts)
        available = 18000-start
        if available <= 0:
            break
        value = text[:available]
        segments.append({"text": value, "start": start, "end": start+len(value),
            "css_selector": ".news-p-content > .rich-text-article p", "original_dom_paragraph_index": i,
            "source_url": url, "content_role": "paragraph", "content_source": "registered_live_html_template",
            "source_published_at": None, "publication_raw": stamps[0], "truncated": value != text})
        parts.append(value)
        offset = start+len(value)
    result = deepcopy(document)
    result.update(kind="article", status="OK", title=title, text="\n".join(parts),
        source_published_at=None, source_publication_local_time=stamps[0], publication_status="local_time_zone_unknown",
        text_truncated=len(parts)<len(selected) or any(s["truncated"] for s in segments))
    result["structured_news"] = {"version": VERSION, "role": "article", "segments": segments,
        "headline": title, "html_body_adapter": VERSION, "publication_timezone": None,
        "previous_projection": document.get("structured_news"), "excluded": [],
        "body_binding": {"canonical_url": url, "heading_text": title,
            "container_css_selector": ".news-p-content > .rich-text-article",
            "binding_scope": "observed_same_page_template_not_match_truth_or_usefulness"}}
    return result
