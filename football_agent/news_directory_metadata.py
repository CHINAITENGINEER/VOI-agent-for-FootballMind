"""v13 response-local JSON-LD/Nuxt directory cards, without target requests.

The registered pages expose serialized metadata rather than rendered anchors.
Only explicit URLs/slugs and fields in those responses are projected; no JS,
payload follow-up, match relevance judgement or information value is evaluated.
"""
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from .inquiry_tools import public_url
from .news_cms_projection import CARD_BASIS, _json

VERSION = "serialized-club-directory-metadata-v1"
DIRECTORIES = {
    "www.cpfc.co.uk": "https://www.cpfc.co.uk/news/",
    "www.nottinghamforest.co.uk": "https://www.nottinghamforest.co.uk/news/",
}
MAX_RECORDS = 2048
MAX_NUXT_VALUES = 50000


def _text(value):
    return value if isinstance(value, str) else None


def _palace(soup):
    scripts = soup.find_all("script", type="application/ld+json")
    if len(scripts) > 32:
        raise ValueError("JSON-LD script bound exceeded")
    for index, script in enumerate(scripts):
        data = _json(script.get_text())
        if not isinstance(data, dict) or data.get("@type") != "ItemList":
            continue
        items = data.get("itemListElement")
        if not isinstance(items, list):
            raise ValueError("ItemList items must be an explicit array")
        for position, item in enumerate(items):
            if not isinstance(item, dict) or item.get("@type") != "ListItem":
                continue
            yield {"title": item.get("name"), "path": item.get("url"),
                   "pointer": f"jsonld[{index}]$.itemListElement[{position}]",
                   "source_position": item.get("position"),
                   "publication_raw": _text(item.get("datePublished")),
                   "source_category": None, "source_saved_timestamp_as_observed": None}


def _nuxt(soup):
    scripts = soup.select("script#__NUXT_DATA__")
    if len(scripts) != 1:
        raise ValueError("One literal Nuxt data array is required")
    data = _json(scripts[0].get_text())
    if not isinstance(data, list) or len(data) > MAX_NUXT_VALUES:
        raise ValueError("Nuxt flat array bound or type invalid")

    def ref(index):
        # Decode only direct references, never arbitrary object graphs or JS.
        if type(index) is not int or not 0 <= index < len(data):
            raise ValueError("Invalid Nuxt field reference")
        return data[index]

    def optional(attrs, name):
        return _text(ref(attrs[name])) if name in attrs else None

    for index, record in enumerate(data):
        if not isinstance(record, dict) or "type" not in record:
            continue
        if ref(record["type"]) != "news-article":
            continue
        attrs_index = record.get("attributes")
        attrs = ref(attrs_index)
        if not isinstance(attrs, dict):
            raise ValueError("Nuxt article attributes must be a referenced object")
        yield {"title": optional(attrs, "postTitle"), "path": optional(attrs, "postSlug"),
               "pointer": f"nuxt[0]$[{index}].attributes[{attrs_index}]",
               "source_position": index,
               "publication_raw": optional(attrs, "publishedDateTime"),
               "source_category": optional(attrs, "postCategoryName"),
               "source_saved_timestamp_as_observed": optional(attrs, "savedTimestamp")}


def project_directory_metadata(document, html, allowed_hosts, *, max_links=320):
    """Keep response order, source pointers and explicit missingness.

    A malformed/oversized serialization does not publish a partial card set.
    Existing DOM links remain visible; truncation and rejection are reported.
    """
    host = urlsplit(document["url"]).hostname
    if document["url"] != DIRECTORIES.get(host):
        return document
    if type(max_links) is not int or not 1 <= max_links <= 320:
        raise ValueError("Directory link cap must be 1..320")
    scope = ("literal_JSONLD_ItemList_only" if host == "www.cpfc.co.uk"
             else "embedded_Nuxt_article_cache_not_complete_main_listing")
    report = {"version": VERSION, "scope": scope, "site_coverage_complete": False,
              "rendered_visibility_verified": False, "target_requests": 0,
              "utility_estimated": False, "order": "serialized_response_order_then_existing_DOM_links",
              "records_seen": 0, "rejected_records": [], "duplicate_urls": 0}
    cards, seen = [], set()
    try:
        soup = BeautifulSoup(html, "html.parser")
        rows = _palace(soup) if host == "www.cpfc.co.uk" else _nuxt(soup)
        for record in rows:
            report["records_seen"] += 1
            if report["records_seen"] > MAX_RECORDS:
                raise ValueError("Serialized directory record bound exceeded")
            title, path, pointer = record["title"], record["path"], record["pointer"]
            try:
                if not isinstance(title, str) or not title.strip() or not isinstance(path, str):
                    raise ValueError("Explicit title and URL required")
                url = public_url(urljoin(document["url"], path), allowed_hosts)
                parsed = urlsplit(url)
                if parsed.hostname != host or not parsed.path.startswith("/news/") or parsed.path == "/news/":
                    raise ValueError("Same-host news target required")
            except (TypeError, ValueError) as exc:
                report["rejected_records"].append({"pointer": pointer, "reason": str(exc)[:160]})
                continue
            if url in seen:
                report["duplicate_urls"] += 1
                continue
            seen.add(url)
            # Publication is a directory assertion, not availability or an
            # independently verified property of the unopened article.
            published = None
            status = "missing"
            if record["publication_raw"] is not None:
                from .news_discovery import _timestamp
                try:
                    stamp = _timestamp(record["publication_raw"])
                    published = stamp.isoformat() if stamp is not None else None
                    status = "directory_source_assertion" if published else "missing"
                except (ValueError, TypeError, OverflowError):
                    status = "unparsed_directory_source_assertion"
            cards.append({"url": url, "title": title[:512], "url_as_observed": path,
                "snippet": None, "snippet_status": "not_projected",
                "directory_context_basis": CARD_BASIS, "directory_metadata_adapter": VERSION,
                "metadata_pointer": pointer,
                "metadata_visibility": "serialized_same_response_not_rendered_DOM_anchor",
                "target_article_fetched_for_context": False,
                "observed_metadata_labels": [{"text": title[:512],
                    "text_truncated": len(title) > 512, "pointer": pointer}],
                "source_position": record["source_position"],
                "source_category": record["source_category"],
                "source_published_at": published, "publication_raw": record["publication_raw"],
                "publication_status": status, "source_modified_at": None,
                "source_saved_timestamp_as_observed": record["source_saved_timestamp_as_observed"],
                "source_available_at": None})
    except (ValueError, TypeError, RecursionError) as exc:
        document["serialized_directory_projection"] = {**report, "status": "rejected",
            "cards": 0, "reason": str(exc)[:160]}
        return document
    merged = cards + [row for row in document.get("links", []) if row.get("url") not in seen]
    document.update(links=merged[:max_links], eligible_link_count=len(merged),
        links_truncated=len(merged) > max_links,
        serialized_directory_projection={**report, "status": "completed", "cards": len(cards),
            "cards_retained": min(len(cards), max_links), "cards_omitted_by_cap": max(0, len(cards)-max_links)})
    return document
