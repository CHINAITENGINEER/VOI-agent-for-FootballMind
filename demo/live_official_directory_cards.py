"""Response-local Italian club card labels; no article fetch or inferred facts."""
from copy import deepcopy
from datetime import datetime
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from football_agent.inquiry_contract import normalized
from football_agent.inquiry_tools import public_url
from football_agent.link_context import _hidden
from football_agent.news_discovery import MAX_INDEX_LINKS


VERSION = "footballmind-live-italian-observed-directory-cards-v1"
DIRECTORIES = {
    "genoacfc.it": "/news",
    "www.veneziafc.it": "/news",
}


def _visible(node):
    return not any(_hidden(parent) for parent in [node, *node.parents])


def _card(url, title, pointer, basis, *, raw_date, date, published=None):
    return {
        "url": url, "title": title, "title_source": basis,
        "directory_metadata_adapter": VERSION, "metadata_pointer": pointer,
        "directory_context_basis": "same_response_explicit_article_card",
        "publication_raw": raw_date, "source_publication_date": date,
        "source_published_at": published, "source_modified_at": None,
        "publication_status": "directory_source_assertion",
        "source_available_at": None,
        "target_article_fetched_for_context": False,
    }


def project_official_directory_cards(document, raw, observation, allowed_hosts):
    """Bind headings/accessibility names to the exact observed card link.

    Genoa's heading is a sibling of the Continua button. Venezia explicitly
    labels its anchor for accessibility and hides its duplicate inner content
    from assistive technology. Neither URL slugs nor image alt text become
    titles. Generic v13 projection/replay is not changed by this live adapter.
    """
    parsed = urlsplit(document.get("url", ""))
    if (parsed.hostname not in DIRECTORIES or parsed.path.rstrip("/") != DIRECTORIES[parsed.hostname]
            or document.get("status") != "OK"
            or not (observation.http_status is not None and 200 <= observation.http_status < 300)
            or observation.error_class):
        return document
    soup = BeautifulSoup(raw, "html.parser")
    cards, rejected = [], []

    if parsed.hostname == "genoacfc.it":
        for index, article in enumerate(soup.select("article")):
            if not _visible(article):
                continue
            roots = article.select("div.article-content")
            if len(roots) != 1:
                continue
            root = roots[0]
            headings = [x for x in root.select("h3") if _visible(x)]
            links = []
            for anchor in root.select("a[href]"):
                if not _visible(anchor):
                    continue
                try:
                    url = public_url(urljoin(document["url"], anchor["href"]), allowed_hosts)
                except (TypeError, ValueError):
                    continue
                target = urlsplit(url)
                if target.hostname == parsed.hostname and re.fullmatch(r"/20\d{2}/\d{2}/\d{2}/[^/]+/?", target.path):
                    links.append(url)
            targets = list(dict.fromkeys(links))
            dates = [normalized(x.get_text(" ")) for x in root.select("p.label-small-regular") if _visible(x)]
            dates = [x for x in dates if re.fullmatch(r"\d{2}\.\d{2}\.\d{2}", x)]
            if len(headings) != 1 or len(targets) != 1 or len(dates) != 1:
                rejected.append({"pointer": f"article[{index}]", "reason": "ambiguous_card_heading_link_or_date"})
                continue
            try:
                date = datetime.strptime(dates[0], "%d.%m.%y").date().isoformat()
            except ValueError:
                rejected.append({"pointer": f"article[{index}]", "reason": "invalid_card_date"})
                continue
            # Cross-check two source assertions; never manufacture a time zone.
            path_date = "/".join(urlsplit(targets[0]).path.strip("/").split("/")[:3])
            title = normalized(headings[0].get_text(" "))
            if path_date != date.replace("-", "/") or not title:
                rejected.append({"pointer": f"article[{index}]", "reason": "card_date_conflict_or_empty_heading"})
                continue
            cards.append(_card(targets[0], title, f"article[{index}].article-content h3",
                               "same_card_observed_heading", raw_date=dates[0], date=date))
    else:
        for index, anchor in enumerate(soup.select("a[href][aria-label]")):
            if not _visible(anchor):
                continue
            try:
                url = public_url(urljoin(document["url"], anchor["href"]), allowed_hosts)
            except (TypeError, ValueError):
                continue
            target = urlsplit(url)
            if target.hostname != parsed.hostname or not re.fullmatch(r"/news/[^/]+/?", target.path):
                continue
            headings, dates = anchor.select("h3"), anchor.select("time")
            pointer = f"a[href][aria-label][{index}]"
            if len(headings) != 1 or len(dates) != 1 or not dates[0].get("datetime"):
                rejected.append({"pointer": pointer, "reason": "ambiguous_accessible_card"})
                continue
            title = normalized(headings[0].get_text(" "))
            date_label = normalized(dates[0].get_text(" "))
            accessible = normalized(anchor["aria-label"])
            if not title or accessible != title + " - " + date_label:
                rejected.append({"pointer": pointer, "reason": "accessible_name_heading_date_mismatch"})
                continue
            try:
                stamp = datetime.fromisoformat(dates[0]["datetime"].replace("Z", "+00:00"))
                day = datetime.strptime(date_label, "%d/%m/%Y").date()
                if stamp.tzinfo is None or stamp.utcoffset() is None or stamp.date() != day:
                    raise ValueError("card date mismatch or missing zone")
            except (TypeError, ValueError):
                rejected.append({"pointer": pointer, "reason": "invalid_or_conflicting_card_date"})
                continue
            card = _card(url, title, pointer, "same_anchor_observed_accessible_name",
                         raw_date=dates[0]["datetime"], date=day.isoformat(), published=stamp.isoformat())
            card["observed_accessible_name"] = accessible
            card["metadata_visibility"] = "accessible_name_with_same_anchor_heading_date_corroboration"
            cards.append(card)

    grouped = {}
    for card in cards:
        grouped.setdefault(card["url"], []).append(card)
    unique = {}
    for url, versions in grouped.items():
        if len({(x["title"], x["source_published_at"], x["source_publication_date"]) for x in versions}) == 1:
            unique[url] = versions[0]
        else:
            rejected.append({"url": url, "reason": "conflicting_same_url_card_metadata"})
    result = deepcopy(document)
    links, seen = [], set()
    for link in result.get("links", []):
        url = link.get("url")
        if url in unique:
            link = {**link, **unique[url], "original_observed_link_title": link.get("title")}
        links.append(link)
        seen.add(url)
    links.extend(card for url, card in unique.items() if url not in seen)
    result.update(links=links[:MAX_INDEX_LINKS], eligible_link_count=len(links),
                  links_truncated=len(links) > MAX_INDEX_LINKS)
    result["official_directory_card_projection"] = {
        "version": VERSION, "cards": len(unique), "rejected": rejected,
        "target_article_requests": 0, "rendered_visibility_verified": False,
        "source_time_semantics": "directory_assertion_not_first_public_availability",
    }
    return result
