"""Opt-in news discovery and time-aware structured article projection.

The adapter never executes JavaScript, follows redirects, guesses article URLs,
or assigns information value. Existing OpenWebTools projections are unchanged.
All article text comes from the archived response, and every admitted segment
keeps its publication metadata and projection offsets.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import json
import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from football_spider.models.contracts import Observation
from .inquiry_contract import normalized
from .inquiry_tools import public_url
from .source_recovery import SourceRoutedOpenWebTools


VERSION = "structured-news-discovery-v1"
PROJECTION_VERSIONS = {1: VERSION, 2: "structured-news-discovery-v2", 3: "structured-news-discovery-v3",
                       4: "structured-news-discovery-v4", 5: "structured-news-discovery-v5",
                       6: "structured-news-discovery-v6", 7: "structured-news-discovery-v7",
                       8: "structured-news-discovery-v8", 9: "structured-news-discovery-v9",
                       10: "structured-news-discovery-v10", 11: "structured-news-discovery-v11",
                       12: "structured-news-discovery-v12", 13: "structured-news-discovery-v13"}
PROVIDERS = {"bing_rss": "https://www.bing.com/search",
             "bing_news_rss": "https://www.bing.com/news/search"}
ARTICLE_TYPES = {"Article", "NewsArticle", "BlogPosting", "LiveBlogPosting"}
MAX_SEGMENTS, MAX_NODES, MAX_TEXT = 64, 512, 18000
# OpenWebTools keeps the first 80 links for its generic projection.  A news
# index is a directory input, so that bound can hide observed article links
# before the runtime is allowed to let the Agent choose among them.  v8 keeps
# a larger, still bounded, structural index projection; it does not fetch or
# rank any target article.
MAX_INDEX_LINKS = 320
_LOCALIZED_MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                     7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def _timestamp(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("Publication timestamp must be text")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            # Bing may localize RSS dates while retaining an explicit GMT
            # suffix, e.g. "周六, 24 10月 2026 05:03:00 GMT".  Translate
            # only this observed calendar form; do not infer a timezone.
            match = re.fullmatch(
                r"\s*周[一二三四五六日天]\s*,?\s*(\d{1,2})\s*(\d{1,2})月\s*(\d{4})\s+(.+?)\s*",
                value.strip())
            if not match:
                match = re.fullmatch(
                    r"\s*(\d{4})年(\d{1,2})月(\d{1,2})日\s+(.+?)\s*",
                    value.strip())
                if match:
                    year, month, day, suffix = match.groups()
                    month_number = int(month)
                    if month_number not in _LOCALIZED_MONTHS:
                        raise ValueError("Localized RSS month is out of range")
                    value = f"{int(day)} {_LOCALIZED_MONTHS[month_number]} {year} {suffix}"
                else:
                    raise
            else:
                day, month, year, suffix = match.groups()
                month_number = int(month)
                if month_number not in _LOCALIZED_MONTHS:
                    raise ValueError("Localized RSS month is out of range")
                value = f"{int(day)} {_LOCALIZED_MONTHS[month_number]} {year} {suffix}"
            parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        raise ValueError("Publication timezone is unknown")
    return parsed.astimezone(timezone.utc)


def _plain(value):
    if not isinstance(value, str):
        return ""
    soup = BeautifulSoup(unescape(value), "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    return normalized(soup.get_text(" ", strip=True))


def _types(node):
    values = node.get("@type", [])
    values = [values] if isinstance(values, str) else values
    return {value.rsplit("/", 1)[-1] for value in values if isinstance(value, str)} if isinstance(values, list) else set()


class NewsDiscoveryTools(SourceRoutedOpenWebTools):
    """Optional projection version, preserving actual acquisition provenance."""

    def __init__(self, archive, allowed_hosts, *, provider="bing_rss", default_route="environment",
                 route_overrides=None, clock=None, extraction_version=3, index_urls=(), projection_version=1):
        if provider not in PROVIDERS or extraction_version != 3:
            raise ValueError("A registered news provider and base extraction version 3 are required")
        if type(projection_version) is not int or projection_version not in PROJECTION_VERSIONS:
            raise ValueError("Unsupported news projection version")
        self.provider = provider
        self.projection_version = projection_version
        super().__init__(archive, allowed_hosts, default_route=default_route,
                         route_overrides=route_overrides, clock=clock, extraction_version=extraction_version)
        self.index_urls = sorted({public_url(url, self.allowed_hosts) for url in index_urls})

    @property
    def body_schema_version(self):
        # v4 adds observed labels; v5 adds bound CMS body containers. Both keep
        # the typed body/offset contract understood by evidence version 5.
        return PROJECTION_VERSIONS[min(self.projection_version, 3)]

    def configuration(self):
        config = {**super().configuration(), "search_provider": self.provider,
                "news_discovery": {"version": VERSION, "search_endpoint": PROVIDERS[self.provider],
                    "index_urls": self.index_urls,
                    "article_types": sorted(ARTICLE_TYPES), "max_segments": MAX_SEGMENTS,
                    "max_jsonld_nodes": MAX_NODES, "max_text_characters": MAX_TEXT,
                    "timestamp_policy": "retain_source_values_exclude_future_or_invalid_modified_segments",
                    "date_only_policy": "retain_date_without_inventing_timezone_or_time",
                    "read_policy": "seed_or_actually_acquired_link_only",
                    "rss_redirect_policy": "decode_explicit_bing_url_parameter_never_network_follow",
                    "article_semantics": "unverified_source_assertions_not_results_or_calibrated_information_value"}}
        if self.projection_version >= 2:
            config["news_discovery"].update(version=PROJECTION_VERSIONS[self.projection_version], projection_version=self.projection_version,
                html_source_adapter="jalgpall-news-detail-v1",
                html_source_policy="exact_official_host_news_permalink_medium_news_unique_detail_head_date_body",
                local_timestamp_policy="retain_source_local_date_and_clock_without_assigning_timezone",
                projection_clock_policy="available_at_is_actual_parse_time_captured_at_remains_original")
        if self.projection_version >= 3:
            config["news_discovery"].update(
                html_body_adapter="same-page-single-article-paragraphs-v1",
                html_body_policy="unique_article_metadata_page_and_heading_binding_then_structural_body_paragraphs",
                html_body_minimum_paragraphs=2, html_body_max_dom_paragraphs=1024,
                html_body_unknown_policy="no_unbound_html_fallback_no_invalid_or_future_metadata_bypass",
                unbound_html_status_policy="preserve_base_unknown_response_never_impute_NO_DATA")
        if self.projection_version >= 4:
            from .link_context import VERSION as LINK_VERSION, MAX_LABELS, MAX_CHARACTERS
            config["news_discovery"]["link_context"] = {
                "version": LINK_VERSION, "max_distinct_labels_per_url": MAX_LABELS,
                "max_label_characters": MAX_CHARACTERS,
                "scope": "observed_parent_html_only_no_target_fetch_or_value_assignment",
                "anchor_index": "original_zero_based_a_href_dom_order",
                "visibility_policy": "exclude_explicit_hidden_dom_no_external_css_rendering",
                "menu_policy": "annotate_existing_links_preserve_url_title_order"}
            config["news_discovery"]["structured_body_version"] = self.body_schema_version
        if self.projection_version >= 5:
            from .news_html_body import STRUCTURED_VERSION, BODY_SELECTORS
            config["news_discovery"].update(html_body_adapter=STRUCTURED_VERSION,
                html_body_policy="unique_same_page_metadata_and_visible_h1_then_single_registered_structural_body_or_semantic_article",
                registered_structural_body_selectors=list(BODY_SELECTORS),
                structural_body_policy="no_host_or_topic_rules_no_competing_full_body_containers")
        if self.projection_version >= 6:
            from .news_html_metadata import VERSION as METADATA_VERSION, ROOT_SELECTOR, BODY_SELECTORS as META_BODY_SELECTORS
            config["news_discovery"].update(html_metadata_adapter=METADATA_VERSION,
                html_metadata_policy="no_jsonld_unique_same_page_og_url_title_description_visible_heading_and_dated_dom_binding",
                html_metadata_root_selector=ROOT_SELECTOR, html_metadata_body_selectors=list(META_BODY_SELECTORS),
                html_metadata_date_policy="url_calendar_date_equals_visible_local_date_no_timezone_or_publication_time_invented",
                html_metadata_semantics="body_presence_only_no_match_relevance_topic_or_information_value_filter")
        if self.projection_version >= 7:
            from .news_html_metadata import GALLERY_VERSION
            config["news_discovery"].update(gallery_introduction_adapter=GALLERY_VERSION,
                gallery_introduction_policy="unique_same_page_gallery_record_header_title_summary_and_publication",
                gallery_text_scope="visible_header_introduction_only_no_image_captions_or_empty_body_imputation")
        if self.projection_version >= 8:
            config["news_discovery"].update(
                index_link_projection="observed_html_links_v2",
                index_max_links=MAX_INDEX_LINKS,
                index_link_visibility="exclude_explicit_hidden_dom_no_external_css_rendering",
                index_link_scope="same_page_public_https_links_only_no_target_fetch_or_value_assignment")
        if self.projection_version >= 9:
            config["news_discovery"].update(
                rss_diagnostics="provider_and_filter_layers_v1",
                rss_relevance_policy="not_assessed_no_semantic_relevance_or_value_inference",
                index_card_context="leeds_visible_contentful_card_v1",
                index_card_context_policy="exact_official_host_visible_same_parent_card_date_and_summary_only_no_article_read")
        if self.projection_version >= 10:
            config["news_discovery"]["additional_index_card_context"] = "liverpool_visible_article_card_v1"
        if self.projection_version >= 11:
            config["news_discovery"]["article_page_alias_policy"] = "liverpool_en_news_jsonld_with_exact_html_canonical_and_english_alternate_v1"
        if self.projection_version >= 12:
            from .news_cms_projection import VERSION as CMS_VERSION
            config["news_discovery"]["cms_response_adapter"] = CMS_VERSION
            config["news_discovery"]["cms_response_policy"] = "bounded_literal_cards_and_bound_body_no_JS_no_target_prefetch"
        if self.projection_version >= 13:
            from .news_directory_metadata import VERSION as DIRECTORY_VERSION, DIRECTORIES
            config["news_discovery"]["serialized_directory_adapter"] = DIRECTORY_VERSION
            config["news_discovery"]["serialized_directory_urls"] = list(DIRECTORIES.values())
            config["news_discovery"]["serialized_directory_policy"] = "literal_ItemList_or_Nuxt_cache_no_main_listing_completeness_or_target_read"
        return config

    @classmethod
    def from_configuration(cls, archive, configuration, *, clock=None):
        if (not isinstance(configuration, Mapping)
                or configuration.get("news_discovery", {}).get("version") not in PROJECTION_VERSIONS.values()):
            raise ValueError("Unsupported news discovery configuration")
        try:
            policy = configuration["source_route_policy"]
            tools = cls(archive, configuration["allowed_hosts"], provider=configuration["search_provider"],
                        default_route=policy["default_route"], route_overrides=policy["route_overrides"],
                        clock=clock, extraction_version=int(configuration["version"][-1]),
                        index_urls=configuration["news_discovery"]["index_urls"],
                        projection_version=configuration["news_discovery"].get("projection_version", 1))
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("Incomplete news discovery configuration") from error
        if tools.configuration() != configuration:
            raise ValueError("News discovery configuration does not reconstruct exactly")
        return tools

    def prepare(self, candidate, public):
        prepared = super().prepare(candidate, public)
        if prepared["tool"] == "search_web":
            prepared["endpoint"] = PROVIDERS[self.provider]
        return prepared

    def execute(self, prepared, *, run_id, request_id):
        if not isinstance(prepared, dict) or set(prepared) != {"tool", "endpoint", "params", "source"}:
            raise ValueError("Prepared action cannot override news transport configuration")
        if prepared["tool"] == "search_web":
            if (prepared["endpoint"] != PROVIDERS[self.provider] or set(prepared["params"]) != {"q", "format"}
                    or prepared["params"]["format"] != "rss"):
                raise ValueError("Search endpoint must match the explicitly configured provider")
            host = "www.bing.com"
        elif prepared["tool"] == "read_url":
            host = urlsplit(public_url(prepared["endpoint"], self.allowed_hosts)).hostname
        else:
            raise ValueError("Unsupported news acquisition tool")
        returned = self._delegate(self.route_overrides.get(host, self.default_route)).execute(
            prepared, run_id=run_id, request_id=request_id)
        returned["document"] = self.project(Observation.model_validate(returned["observation"]), prepared)
        return returned

    def project(self, observation, prepared):
        document = super().project(observation, prepared)
        document["news_projection_version"] = PROJECTION_VERSIONS[self.projection_version]
        if self.projection_version >= 2:
            if _timestamp(document["available_at"]) < observation.observed_at:
                raise ValueError("Projection time cannot precede the archived observation")
            document["projection_provenance"] = {"captured_at": observation.observed_at.isoformat(),
                "projected_at": document["available_at"],
                "scope": "actual_time_versioned_projection_of_archived_response_bytes",
                "network_requests_during_projection": 0}
            if self.projection_version >= 9:
                document["projection_provenance"].pop("captured_at")
                document["projection_provenance"].update(
                    response_observed_at=observation.observed_at.isoformat(),
                    projection_clock_basis="projection_start_not_agent_handoff_or_archive_save")
        if not (observation.http_status is not None and 200 <= observation.http_status < 300 and not observation.error_class):
            return document
        body = self.archive.read_body(observation).decode("utf-8", errors="replace")
        if prepared["tool"] == "search_web":
            return self._project_rss(document, body, observation)
        is_index = document["url"] in self.index_urls
        if is_index and self.projection_version >= 8:
            document = self._project_index_links_v8(document, body)
        if is_index and self.projection_version >= 9:
            document = self._project_index_card_context_v9(document, body)
        if is_index and self.projection_version >= 10:
            document = self._project_liverpool_card_context(document, body)
        if self.projection_version >= 4:
            from .link_context import retain_link_labels
            document = retain_link_labels(document, body, self.allowed_hosts)
        if is_index and self.projection_version >= 12:
            from .news_cms_projection import project_index_cards
            document = project_index_cards(document, body, self.allowed_hosts, max_links=MAX_INDEX_LINKS)
        if is_index and self.projection_version >= 13:
            from .news_directory_metadata import project_directory_metadata
            document = project_directory_metadata(document, body, self.allowed_hosts, max_links=MAX_INDEX_LINKS)
        if is_index:
            document["kind"] = "index"
            document["structured_news"] = {"version": self.body_schema_version, "role": "index", "segments": [],
                "excluded": [], "identity_metadata": [], "directory_basis": "explicit_source_catalog_index_url"}
            return document
        if self.projection_version >= 2 and urlsplit(document["url"]).hostname in {"jalgpall.ee", "www.jalgpall.ee"}:
            return self._project_jalgpall_article(document, body, observation)
        host = urlsplit(document["url"]).hostname
        if self.projection_version >= 13 and host in {"www.fc-union-berlin.de", "www.fcaugsburg.de"}:
            return self._project_german_club_article(document, body, observation, host)
        return self._project_article(document, body, observation)

    def _project_german_club_article(self, document, body, observation, host):
        """Project the observed article containers used by two verified German club sites."""
        soup = BeautifulSoup(body, "html.parser")
        if host == "www.fc-union-berlin.de":
            path = urlsplit(document["url"]).path
            if not path.startswith("/de/meldungen/"):
                return self._project_article(document, body, observation)
            article = next((node for node in soup.select("main article")
                            if node.find(["h1", "h2"]) and node.find("time", datetime=True)), None)
            adapter = "fc-union-berlin-meldung-v1"
        else:
            path = urlsplit(document["url"]).path
            if not path.startswith("/article/"):
                return self._project_article(document, body, observation)
            article = soup.select_one("article.b-article")
            adapter = "fc-augsburg-article-v1"
        metadata = {"version": self.body_schema_version, "role": "unknown", "segments": [],
                    "excluded": [], "identity_metadata": [], "jsonld_parse_errors": 0,
                    "source_adapter": adapter}
        document["structured_news"] = metadata
        if article is None:
            document.update(status="NO_DATA", kind="unknown", text="", source_published_at=None,
                            publication_status="article_container_missing")
            return document
        title_node = article.find(["h1", "h2"])
        title = normalized(title_node.get_text(" ", strip=True)) if title_node else None
        published = None
        time_node = article.find("time", datetime=True)
        if time_node is not None:
            try:
                published = _timestamp(time_node.get("datetime"))
            except (TypeError, ValueError, OverflowError):
                published = None
        if published is None and host == "www.fcaugsburg.de":
            for script in soup.find_all("script", type="application/ld+json")[:32]:
                try:
                    value = json.loads(script.string or script.get_text())
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata["jsonld_parse_errors"] += 1
                    continue
                nodes = value if isinstance(value, list) else [value]
                for node in nodes:
                    if isinstance(node, dict) and node.get("datePublished"):
                        try:
                            published = _timestamp(node["datePublished"])
                        except (TypeError, ValueError, OverflowError):
                            published = None
                        if published is not None:
                            break
                if published is not None:
                    break
        if published and published > observation.observed_at:
            metadata["excluded"].append({"reason": "publication_after_actual_observation"})
            document.update(status="NO_DATA", kind="unknown", text="", source_published_at=None,
                            publication_status="future_source_timestamp")
            return document
        paragraphs = []
        for node in article.find_all("p"):
            text = normalized(node.get_text(" ", strip=True))
            if text and text not in paragraphs:
                paragraphs.append(text)
        if not paragraphs:
            metadata["excluded"].append({"reason": "article_body_paragraphs_missing"})
            document.update(status="NO_DATA", kind="unknown", text="", source_published_at=None,
                            publication_status="body_outside_registered_container")
            return document
        parts, offset = [], 0
        for index, text in enumerate(paragraphs[:MAX_SEGMENTS]):
            start = offset + (1 if parts else 0)
            remaining = MAX_TEXT - start
            if remaining <= 0:
                break
            selected = text[:remaining]
            metadata["segments"].append({"text": selected, "start": start, "end": start+len(selected),
                "content_role": "paragraph", "paragraph_index": index,
                "content_source": adapter, "source_url": document["url"],
                "source_published_at": published.isoformat() if published else None,
                "truncated": len(selected) < len(text)})
            parts.append(selected)
            offset = start + len(selected)
        metadata.update(role="article", title=title, excluded_count=0)
        document.update(kind="article", status="OK", text="\n".join(parts),
                        source_published_at=published.isoformat() if published else None,
                        publication_status="source_datetime_attribute" if published else "unknown",
                        text_truncated=len(parts) < len(paragraphs) or any(s["truncated"] for s in metadata["segments"]))
        return document

    def _project_index_links_v8(self, document, body):
        """Expose up to MAX_INDEX_LINKS visible, observed index URLs.

        The generic OpenWebTools projection is intentionally capped at 80
        links.  That is appropriate for a normal page, but silently drops
        article entries on a large club homepage.  This adapter only parses
        the already archived HTML and keeps the same host/URL policy; the
        Agent still decides which exposed link, if any, merits a later read.
        """
        from .link_context import _hidden

        soup = BeautifulSoup(body, "html.parser")
        base = soup.find("base", href=True)
        base_url = document["url"]
        if base is not None:
            base_url = urljoin(base_url, base["href"])
        anchors = soup.find_all("a", href=True)
        anchors.sort(key=lambda a: (0 if a.find_parent(["main", "article"]) else
                                    2 if a.find_parent(["header", "nav", "footer"]) else 1))
        links, seen = [], set()
        for anchor in anchors:
            if any(_hidden(node) for node in [anchor, *anchor.parents]):
                continue
            try:
                found = public_url(urljoin(base_url, anchor["href"]), self.allowed_hosts)
            except (TypeError, ValueError):
                continue
            if found in seen:
                continue
            seen.add(found)
            visible = deepcopy(anchor)
            for node in list(visible.find_all(True)):
                if _hidden(node):
                    node.decompose()
            links.append({"url": found, "title": normalized(visible.get_text(" "))[:200]})
        document["links"] = links[:MAX_INDEX_LINKS]
        document["eligible_link_count"] = len(links)
        document["links_truncated"] = len(links) > MAX_INDEX_LINKS
        return document

    def _project_index_card_context_v9(self, document, body):
        """Keep only actual Leeds directory-card summaries and date labels.

        A card without a visible summary records null. It never borrows text
        from the target article, serialized CMS data, another card or a title.
        """
        if urlsplit(document["url"]).hostname != "www.leedsunited.com":
            return document
        from .link_context import _hidden

        soup = BeautifulSoup(body, "html.parser")
        cards = {}
        for anchor in soup.find_all("a", href=True):
            card = anchor.find_parent("article")
            if (card is None or not any(str(c).startswith("articleCard_articleCard__") for c in card.get("class", []))
                    or any(_hidden(n) for n in [anchor, *anchor.parents])):
                continue
            try:
                url = public_url(urljoin(document["url"], anchor["href"]), self.allowed_hosts)
            except (TypeError, ValueError):
                continue
            if "/en/news/" not in urlsplit(url).path or "/category/" in urlsplit(url).path:
                continue
            cards.setdefault(url, []).append(card)
        for link in document["links"]:
            matched = cards.get(link["url"], [])
            unique = {id(card): card for card in matched}
            if len(unique) != 1:
                continue
            card = next(iter(unique.values()))
            visible = deepcopy(card)
            for node in list(visible.find_all(True)):
                if node.parent is not None and _hidden(node):
                    node.decompose()
            summaries = [normalized(p.get_text(" ", strip=True)) for p in visible.find_all("p")]
            summaries = [value for value in summaries if value]
            dates = visible.select('[data-testid="publish-date"]')
            raw_date = normalized(dates[0].get_text(" ", strip=True)) if len(dates) == 1 else None
            try:
                source_date = datetime.strptime(raw_date, "%d %b %y").date().isoformat() if raw_date else None
            except ValueError:
                source_date = None
            snippet = " ".join(summaries) if summaries else None
            link.update(snippet=snippet[:600] if snippet else None,
                        snippet_status="observed_directory_card" if snippet else "not_present_in_visible_card",
                        snippet_truncated=len(snippet) > 600 if snippet else False,
                        publication_raw=raw_date, source_publication_date=source_date,
                        publication_timezone="unknown", source_published_at=None,
                        directory_context_basis="leeds_visible_contentful_card_v1",
                        target_article_fetched_for_context=False)
        return document

    def _project_liverpool_card_context(self, document, body):
        """Annotate an exposed link from its visible card, without fetching it."""
        if urlsplit(document["url"]).hostname != "www.liverpoolfc.com":
            return document
        from .link_context import _hidden
        soup = BeautifulSoup(body, "html.parser")
        cards = {}
        for card in soup.select('article[data-testid="article-card"]'):
            if any(_hidden(n) for n in [card, *card.parents]):
                continue
            anchors = card.select('a[data-testid="article-card__link"][href]')
            heads = card.find_all("h3")
            if len(anchors) != 1 or len(heads) != 1:
                continue
            anchor, heading = anchors[0], heads[0]
            if any(_hidden(n) for n in [anchor, heading, *heading.parents]):
                continue
            title = normalized(heading.get_text(" ", strip=True))
            if not title or normalized(anchor.get("aria-label", "")) != title:
                continue
            try:
                url = public_url(urljoin(document["url"], anchor["href"]), self.allowed_hosts)
            except (ValueError, TypeError):
                continue
            dates = {n.get("datetime") for n in card.find_all("time", datetime=True)
                     if not any(_hidden(p) for p in [n, *n.parents])}
            source_date = next(iter(dates)) if len(dates) == 1 else None
            if source_date:
                try:
                    datetime.strptime(source_date, "%Y-%m-%d")
                except ValueError:
                    source_date = None
            cards.setdefault(url, []).append({"title": title, "source_publication_date": source_date,
                "publication_raw": [n.get_text(" ", strip=True) for n in card.find_all("time")],
                "source_published_at": None, "publication_timezone": "unknown", "snippet": None,
                "snippet_status": "not_present_in_visible_card", "target_article_fetched_for_context": False,
                "directory_context_basis": "liverpool_visible_article_card_v1"})
        for link in document["links"]:
            matches = cards.get(link["url"], [])
            if len(matches) == 1:
                link.update(matches[0])
        return document

    def _project_jalgpall_article(self, document, body, observation):
        """Source-bound HTML paragraphs, with no translation or fixture inference."""
        soup = BeautifulSoup(body, "html.parser")
        metadata = {"version": self.body_schema_version, "role": "unknown", "segments": [], "excluded": [],
            "identity_metadata": [], "html_source_adapter": "jalgpall-news-detail-v1",
            "source_language": soup.html.get("lang") if soup.html else None, "language_source": "html@lang",
            "offset_unit": "zero_based_unicode_code_points_end_exclusive_in_projected_text"}
        document["structured_news"] = metadata

        def reject(reason, *, clear=False):
            metadata["excluded"].append({"reason": reason})
            metadata["excluded_count"] = len(metadata["excluded"])
            if clear:
                document.update(text="", status="NO_DATA", text_truncated=False)
            return document

        path = urlsplit(document["url"]).path
        if not re.fullmatch(r"/koondis/(?:\d+/)?uudised/[^/]+-n[1-9]\d*/?", path):
            return reject("not_registered_jalgpall_news_permalink")
        mediums = soup.select('meta[name="medium"]')
        roots = soup.select("div.news-detail")
        if len(mediums) != 1 or mediums[0].get("content") != "news" or len(roots) != 1:
            return reject("unique_source_news_marker_and_detail_required")
        root = roots[0]
        heads, bodies = root.select(":scope > div.head"), root.select(":scope > div.body.def")
        if root.find_parent(["nav", "header", "footer", "aside"]) or len(heads) != 1 or len(bodies) != 1:
            return reject("unique_article_head_and_body_required")
        titles, dates = heads[0].select(":scope > h1"), heads[0].select(":scope > div.date")
        if len(titles) != 1 or not _plain(str(titles[0])) or len(dates) != 1:
            return reject("article_title_and_publication_block_required")
        days, stamps = dates[0].select(":scope > p.day"), dates[0].select(":scope > p.m-y")
        if len(days) != 1 or len(stamps) not in {1, 2}:
            return reject("publication_block_ambiguous")
        raw = " ".join(_plain(str(n)) for n in [days[0], *stamps])
        try:
            local = datetime.strptime(raw, "%d %m.%Y %H:%M" if len(stamps) == 2 else "%d %m.%Y")
        except ValueError:
            return reject("invalid_local_publication_date", clear=True)
        publication = {"publication_raw": raw, "source_publication_date": local.date().isoformat(),
            "source_publication_local_time": local.strftime("%H:%M") if len(stamps) == 2 else None,
            "publication_timezone": "unknown", "source_published_at": None, "source_modified_at": None}
        metadata.update(publication=publication, headline=_plain(str(titles[0])))
        # This is an earliest-possible bound over worldwide UTC offsets, not
        # an assignment of Estonia's timezone to an undated source clock.
        if local.replace(tzinfo=timezone.utc) - timedelta(hours=14) > observation.observed_at:
            return reject("local_publication_certainly_after_actual_capture", clear=True)
        container = bodies[0]
        for node in container.select("script,style,noscript,nav,header,footer,aside,form,button,.modal,.social-share,.main-img"):
            node.decompose()
        paragraphs = [(i, _plain(str(node))) for i, node in enumerate(container.find_all("p"))]
        paragraphs = [(i, text) for i, text in paragraphs if text]
        if not paragraphs:
            return reject("article_body_has_no_nonempty_paragraphs", clear=True)
        candidates = [("headline", None, metadata["headline"], "div.news-detail > div.head > h1")]
        candidates += [("paragraph", i, text, "div.news-detail > div.body.def p") for i, text in paragraphs]
        parts, offset = [], 0
        for role, index, text, selector in candidates[:MAX_SEGMENTS]:
            remaining = MAX_TEXT-offset-(1 if parts else 0)
            if remaining <= 0:
                break
            selected, start = text[:remaining], offset+(1 if parts else 0)
            metadata["segments"].append({**publication, "text": selected, "start": start, "end": start+len(selected),
                "content_role": role, "paragraph_index": index, "css_selector": selector,
                "content_source": "jalgpall_news_detail_html", "source_url": document["url"], "truncated": len(selected)<len(text)})
            parts.append(selected)
            offset = start+len(selected)
        if not any(segment["content_role"] == "paragraph" for segment in metadata["segments"]):
            metadata["segments"] = []
            return reject("article_body_outside_projection_bound", clear=True)
        metadata.update(role="article", excluded_count=0)
        document.update(kind="article", status="OK", text="\n".join(parts), source_published_at=None,
            publication_status="source_local_date_time_timezone_unknown",
            text_truncated=len(metadata["segments"])<len(candidates) or any(s["truncated"] for s in metadata["segments"]))
        return document

    def _project_rss(self, document, body, observation):
        document.update(links=[], text="", text_truncated=False)
        metadata = {"version": self.body_schema_version, "role": "discovery", "segments": [], "excluded": [],
                    "identity_metadata": [], "pending_host_links": []}
        document["structured_news"] = metadata
        try:
            root = ET.fromstring(body)
            if root.tag.lower() != "rss":
                raise ET.ParseError("RSS root required")
        except ET.ParseError:
            document["status"] = "INVALID_CONTENT"
            if self.projection_version >= 9:
                document.update(filter_status="parse_error", relevance_status="not_assessed")
                metadata["retrieval_diagnostics"] = {"provider_status": "parse_error",
                    "provider_result_count": None, "filter_status": "not_run",
                    "eligible_result_count": None, "relevance_status": "not_assessed"}
            return document
        fragments, seen = [], set()
        provider_items = root.findall("./channel/item")
        duplicate_count = 0
        for index, item in enumerate(provider_items[:80]):
            raw_time = item.findtext("pubDate")
            try:
                published = _timestamp(raw_time)
            except (ValueError, TypeError, OverflowError):
                metadata["excluded"].append({"item": index, "reason": "invalid_publication_time", "publication_raw": raw_time})
                continue
            if published and published > observation.observed_at:
                metadata["excluded"].append({"item": index, "reason": "publication_after_actual_observation", "publication_raw": raw_time})
                continue
            link = item.findtext("link") or ""
            parsed = urlsplit(link)
            if parsed.hostname in {"www.bing.com", "cn.bing.com"} and parsed.path == "/news/apiclick.aspx":
                pairs = parse_qsl(parsed.query)
                targets = [v for k, v in pairs if k == "url"]
                link = targets[0] if len(targets) == 1 else ""
            raw_link = (link or "").strip()
            try:
                link = public_url(raw_link, self.allowed_hosts)
            except (ValueError, TypeError):
                # Keep a safe, non-candidate representation of a public search
                # result whose host is not in the static source catalog. The
                # live layer may perform a bounded exact-host review; this
                # projection never grants access or fetches the URL.
                parsed_link = urlsplit(raw_link)
                if (parsed_link.scheme == "https" and parsed_link.hostname
                        and not parsed_link.username and not parsed_link.password
                        and parsed_link.hostname.lower() not in {"www.bing.com", "cn.bing.com"}):
                    pending_url = urlunsplit((parsed_link.scheme, parsed_link.netloc.lower(),
                                              parsed_link.path or "/", parsed_link.query, ""))
                    pending_title = _plain(item.findtext("title")) or pending_url
                    metadata["pending_host_links"].append({"url": pending_url, "title": pending_title[:200],
                        "source_published_at": published.isoformat() if published else None, "publication_raw": raw_time,
                        "review_status": "pending_host_review", "review_reason": "host_not_allowlisted"})
                metadata["excluded"].append({"item": index, "reason": "outside_source_catalog_or_invalid_explicit_target"})
                continue
            if link in seen:
                duplicate_count += 1
                continue
            seen.add(link)
            title = _plain(item.findtext("title")) or link
            document["links"].append({"url": link, "title": title[:200],
                "source_published_at": published.isoformat() if published else None, "publication_raw": raw_time})
            fragments.append(title+": "+_plain(item.findtext("description")))
        text = "\n".join(fragments)
        document.update(text=text[:MAX_TEXT], text_truncated=len(text)>MAX_TEXT,
                        status="OK" if fragments else "NO_DATA")
        if self.projection_version >= 9:
            filtered = len(metadata["excluded"])
            status = "returned" if fragments else "provider_empty" if not provider_items else "filtered_empty"
            query = observation.params.get("q")
            echo_link = root.findtext("./channel/link")
            echo_queries = [value for key, value in parse_qsl(urlsplit(echo_link or "").query) if key == "q"]
            document.update(filter_status=status, relevance_status="not_assessed")
            metadata["retrieval_diagnostics"] = {
                "provider_status": "returned" if provider_items else "provider_empty",
                "provider_result_count": len(provider_items),
                "considered_result_count": min(len(provider_items), 80),
                "provider_items_truncated": len(provider_items) > 80,
                "filtered_result_count": filtered, "duplicate_result_count": duplicate_count,
                "eligible_result_count": len(document["links"]), "filter_status": status,
                "pending_host_review_count": len(metadata["pending_host_links"]),
                "relevance_status": "not_assessed",
                "relevance_boundary": "irrelevant_results_requires_separate_explicit_assessment_not_inferred_from_empty_or_host_filter",
                "provider_query_raw": query, "provider_query_echo_title": root.findtext("./channel/title"),
                "provider_query_echo_link": echo_link,
                "provider_query_echo_matches": echo_queries == [query] if echo_queries else None,
            }
        return document

    def _project_article(self, document, body, observation):
        soup = BeautifulSoup(body, "html.parser")
        metadata = {"version": self.body_schema_version, "role": "unknown", "segments": [], "excluded": [],
                    "identity_metadata": [], "jsonld_parse_errors": 0, "nodes_truncated": False}
        if self.projection_version >= 2:
            metadata.update(source_language=soup.html.get("lang") if soup.html else None, language_source="html@lang")
        document["structured_news"] = metadata
        if self.projection_version >= 12:
            from .news_cms_projection import project_cms_article
            projected = project_cms_article(document, body, observation, metadata)
            if projected is not None:
                return projected
        pending, recognized = [], 0
        for index, script in enumerate(soup.find_all("script", type="application/ld+json")[:32]):
            try:
                value = json.loads(script.get_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except (ValueError, RecursionError):
                metadata["jsonld_parse_errors"] += 1
                continue
            pending.append((value, f"script[{index}]", False))
        segments, count, is_live = [], 0, False
        while pending and count < MAX_NODES:
            node, pointer, inherited_live = pending.pop(0)
            count += 1
            if isinstance(node, list):
                pending.extend((entry, pointer+f"[{index}]", inherited_live) for index, entry in enumerate(node))
                continue
            if not isinstance(node, dict):
                continue
            types = _types(node)
            live = inherited_live or "LiveBlogPosting" in types
            is_live |= live
            if "SportsEvent" in types:
                metadata["identity_metadata"].append({"json_pointer": pointer, "type": "SportsEvent",
                    **{key: node.get(key) for key in ("name", "url", "startDate", "endDate", "eventStatus", "homeTeam", "awayTeam")}})
            if types & ARTICLE_TYPES:
                recognized += 1
                text = _plain(node.get("articleBody"))
                if text:
                    raw_published, raw_modified = node.get("datePublished"), node.get("dateModified")
                    reason = None
                    try:
                        published, modified = _timestamp(raw_published), _timestamp(raw_modified)
                        if live and published is None:
                            reason = "live_update_publication_time_unknown"
                        elif published and published > observation.observed_at:
                            reason = "publication_after_actual_observation"
                        elif modified and modified > observation.observed_at:
                            reason = "modification_after_actual_observation"
                        elif published and modified and modified < published:
                            reason = "source_modification_precedes_publication"
                    except (ValueError, TypeError, OverflowError):
                        published, modified, reason = None, None, "invalid_publication_time"
                    if reason:
                        metadata["excluded"].append({"json_pointer": pointer, "reason": reason,
                            "publication_raw": raw_published, "modified_raw": raw_modified})
                    else:
                        headline = _plain(node.get("headline"))
                        segments.append({"text": (headline+"\n" if headline and not text.startswith(headline) else "")+text,
                            "json_pointer": pointer+".articleBody", "content_source": "json_ld_article_body",
                            "source_url": node.get("url") if isinstance(node.get("url"), str) else document["url"],
                            "source_published_at": published.isoformat() if published else None,
                            "source_modified_at": modified.isoformat() if modified else None,
                            "publication_raw": raw_published, "modified_raw": raw_modified})
            for key in ("@graph", "mainEntity", "liveBlogUpdate"):
                if key in node:
                    pending.append((node[key], pointer+"."+key, live))
        metadata["nodes_truncated"] = bool(pending)
        if self.projection_version >= 6 and not segments and not recognized:
            from .news_html_metadata import project_html_metadata_body
            return project_html_metadata_body(document, body, observation, metadata,
                                               allow_gallery=self.projection_version >= 7)
        if self.projection_version >= 3 and not segments:
            from .news_html_body import project_bound_html_body
            return project_bound_html_body(document, body, observation, metadata, recognized_articles=recognized,
                                           structural_fallback=self.projection_version >= 5,
                                           allow_liverpool_locale_alias=self.projection_version >= 11)
        if not segments and not recognized:
            # Semantic article containers may be surrounded by smaller related
            # cards. Require an unambiguous substantial paragraph container.
            articles = [node for node in soup.find_all("article")
                        if len(node.find_all("p")) >= 2 and len(_plain(str(node))) >= 300
                        and not node.find_parent(["aside", "nav", "footer", "header"])]
            if len(articles) == 1:
                segments.append({"text": _plain(str(articles[0])), "json_pointer": None,
                    "content_source": "semantic_html_article", "source_url": document["url"],
                    "source_published_at": None, "source_modified_at": None,
                    "publication_raw": None, "modified_raw": None})
                stamps = [node for node in soup.select('[data-testid="publish-date"]') if not node.find_parent("article")]
                if len(stamps) == 1:
                    raw = _plain(str(stamps[0]))
                    try:
                        day = datetime.strptime(raw, "%d %b %y").date()
                    except ValueError:
                        day = None
                    if day is not None:
                        segments[0].update(publication_raw=raw, source_publication_date=day.isoformat())
                        # With no declared timezone, only dates strictly after
                        # the next UTC day are certainly future worldwide.
                        if (day-observation.observed_at.date()).days > 1:
                            metadata["excluded"].append({"reason": "publication_date_certainly_future", "publication_raw": raw})
                            segments.clear()
        if segments:
            parts, offset = [], 0
            for segment in segments[:MAX_SEGMENTS]:
                available = MAX_TEXT-offset-(1 if parts else 0)
                if available <= 0:
                    break
                text = segment["text"][:available]
                start = offset+(1 if parts else 0)
                metadata["segments"].append({**segment, "text": text, "start": start, "end": start+len(text),
                                             "truncated": len(text)<len(segment["text"])})
                parts.append(text)
                offset = start+len(text)
            document.update(text="\n".join(parts), status="OK",
                text_truncated=len(metadata["segments"])<len(segments) or any(x["truncated"] for x in metadata["segments"]))
            metadata["role"] = "liveblog" if is_live else "article"
            if self.projection_version >= 2:
                document["kind"] = "article"
            published = [x["source_published_at"] for x in metadata["segments"] if x["source_published_at"]]
            document["source_published_at"] = max(published) if published else None
            document["publication_status"] = "source_segment_metadata" if published else "unknown"
        elif recognized or metadata["excluded"]:
            # Visible markup may mirror excluded future script content; do not
            # fall back to that same unqualified text after excluding it.
            document.update(text="", status="NO_DATA", text_truncated=False)
        metadata["excluded_count"] = len(metadata["excluded"])
        return document


__all__ = ["NewsDiscoveryTools", "VERSION"]
