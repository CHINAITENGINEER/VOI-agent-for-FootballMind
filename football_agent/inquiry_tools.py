"""Bounded generic discovery/read tools using the existing transport and archive."""
from __future__ import annotations

from datetime import datetime, timezone
import ipaddress
import json
import time
import xml.etree.ElementTree as ET
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
import requests

from football_platform.collectors.http_transport import RequestTransport, TransportError
from .inquiry_contract import normalized


class ResponseLimit(requests.RequestException):
    pass


class BoundedSession(requests.Session):
    def get(self, url, **kwargs):
        response = super().get(url, stream=True, **kwargs)
        started, chunks, size = time.monotonic(), [], 0
        try:
            for chunk in response.iter_content(16384):
                size += len(chunk)
                if size > 2_000_000 or time.monotonic() - started > 25:
                    raise ResponseLimit("Response size/time bound reached")
                chunks.append(chunk)
            response._content = b"".join(chunks)
            response._content_consumed = True
            return response
        except Exception:
            response.close()
            raise


def public_url(url, allowed_hosts):
    p = urlsplit(url)
    if p.scheme != "https" or not p.hostname or p.username or p.password or p.port not in {None, 443}:
        raise ValueError("Public HTTPS URL required")
    try:
        ipaddress.ip_address(p.hostname)
    except ValueError:
        pass
    else:
        raise ValueError("Literal IP is outside the source catalog")
    if p.hostname.lower() not in allowed_hosts:
        raise ValueError("Source outside the configured domain catalog")
    if any(k.lower() in {"token", "api_key", "key", "password", "authorization"} for k, _ in parse_qsl(p.query)):
        raise ValueError("Credential query parameter is not a public source")
    return urlunsplit((p.scheme, p.netloc.lower(), p.path or "/", p.query, ""))


class OpenWebTools:
    version = "open-web-read-v2"

    def __init__(self, archive, allowed_hosts, *, session_factory=BoundedSession, clock=None, extraction_version=2):
        self.archive = archive
        self.allowed_hosts = sorted(set(h.lower() for h in allowed_hosts))
        self.session_factory = session_factory
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if extraction_version not in {1, 2, 3}:
            raise ValueError("Unknown extraction version")
        self.extraction_version = extraction_version

    def configuration(self):
        return {"version": "open-web-read-v" + str(self.extraction_version), "allowed_hosts": self.allowed_hosts, "search_provider": "bing_rss",
                "max_response_bytes": 2_000_000, "max_text_characters": 18000,
                "request_attempts_per_action": 1, "redirect_policy": "no_automatic_follow"}

    def prepare(self, candidate, public):
        if candidate["tool"] == "search_web":
            query = candidate["query"].strip()
            return {"tool": "search_web", "endpoint": "https://www.bing.com/search",
                    "params": {"q": query, "format": "rss"}, "source": "open_search"}
        url = public_url(candidate["url"], self.allowed_hosts)
        known = set(public["task"]["seed_urls"])
        for doc in public["documents"]:
            known.update(link["url"] for link in doc.get("links", []))
        if url not in known:
            raise ValueError("URL must come from task seeds or acquired links")
        p = urlsplit(url)
        pairs = parse_qsl(p.query, keep_blank_values=True)
        if len(dict(pairs)) != len(pairs):
            raise ValueError("Repeated query keys are unsupported")
        return {"tool": "read_url", "endpoint": urlunsplit((p.scheme, p.netloc, p.path, "", "")),
                "params": dict(pairs), "source": "open_page"}

    def execute(self, prepared, *, run_id, request_id):
        observations = []

        def archive_attempt(**facts):
            observations.append(self.archive.save(
                facts["source"], facts["endpoint"], facts["params"], facts["body"],
                status=facts["http_status"], headers=facts["response_headers"], request_id=facts["request_id"],
                observed_at=facts["observed_at"], observation_id=facts["observation_id"], error_class=facts["error_class"],
                context={"inquiry_run_id": run_id, "inquiry_request_id": request_id,
                         "requested_at": facts["requested_at"].isoformat(), "tool": prepared["tool"]}))

        started = time.monotonic()
        with RequestTransport(prepared["source"], on_attempt=archive_attempt, session=self.session_factory(),
                              max_attempts=1, request_budget=1, interval=0, timeout=(6, 16), clock=self.clock) as transport:
            try:
                transport.get(prepared["endpoint"], prepared["params"], headers={"User-Agent": "FootballMind/0.1 research"})
            except TransportError:
                pass
        if len(observations) != 1:
            raise RuntimeError("ACQUISITION_ARCHIVE_REQUIRES_RECONCILIATION")
        observation = observations[0]
        document = self.project(observation, prepared)
        return {"document": document, "observation": observation.model_dump(mode="json"),
                "request_id": request_id, "http_attempts": 1, "latency_seconds": time.monotonic() - started,
                "money": None, "money_status": "not_measured"}

    def project(self, observation, prepared):
        body = self.archive.read_body(observation)
        params = sorted(prepared["params"].items()) if self.extraction_version >= 3 else prepared["params"]
        url = prepared["endpoint"] + ("?" + urlencode(params) if params else "")
        okay = observation.http_status is not None and 200 <= observation.http_status < 300 and not observation.error_class
        doc = {"document_id": "doc-" + observation.id, "url": url,
               "kind": "discovery" if prepared["tool"] == "search_web" else "page",
               "status": "OK" if okay else "FAILED", "http_status": observation.http_status,
               "error_class": observation.error_class, "observed_at": observation.observed_at.isoformat(),
               "available_at": self.clock().isoformat(), "source_published_at": None,
               "publication_status": "unknown", "raw_sha256": observation.content_hash,
               "text": "", "links": [], "text_truncated": False,
               "claim_verification": "source_content_not_independent_semantic_verification"}
        if not okay:
            return doc
        # Deterministic extraction: this is not an LLM summary of the source.
        text = body.decode("utf-8", errors="replace")
        if doc["kind"] == "discovery":
            items = []
            try:
                feed = ET.fromstring(text)
            except ET.ParseError:
                doc["status"] = "INVALID_CONTENT"
                return doc
            if feed.tag.lower() != "rss":
                doc["status"] = "INVALID_CONTENT"
                return doc
            for item in feed.findall("./channel/item"):
                link = item.findtext("link")
                title = item.findtext("title")
                description = item.findtext("description")
                if not link:
                    continue
                try:
                    found = public_url(link.strip(), self.allowed_hosts)
                except ValueError:
                    continue
                heading = normalized(title) if title else found
                doc["links"].append({"url": found, "title": heading[:200]})
                items.append(heading + ": " + (BeautifulSoup(description, "html.parser").get_text(" ", strip=True) if description else ""))
            text = "\n".join(items)
        else:
            try:
                data = json.loads(text)
            except ValueError:
                soup = BeautifulSoup(text, "html.parser")
                links = []
                base_url = url
                if self.extraction_version >= 3:
                    # Honor the first HTML base href, as browsers do. Every
                    # resulting URL still passes the configured host policy.
                    base = soup.find("base", href=True)
                    if base is not None:
                        base_url = urljoin(url, base["href"])
                anchors = soup.find_all("a", href=True)
                if self.extraction_version >= 2:
                    # Content links must survive a page with hundreds of global
                    # navigation links. Keep useful navigation after main links.
                    anchors.sort(key=lambda a: (0 if a.find_parent(["main", "article"]) else
                                                2 if a.find_parent(["header", "nav", "footer"]) else 1))
                for anchor in anchors:
                    try:
                        found = public_url(urljoin(base_url, anchor["href"]), self.allowed_hosts)
                    except ValueError:
                        continue
                    if found not in {link["url"] for link in links}:
                        links.append({"url": found, "title": normalized(anchor.get_text(" "))[:200]})
                doc["links"] = links[:80]
                if self.extraction_version >= 2:
                    doc["links_truncated"] = len(links) > 80
                    doc["eligible_link_count"] = len(links)
                for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
                    tag.decompose()
                text = normalized(soup.get_text(" "))
            else:
                text = json.dumps(data, ensure_ascii=False, sort_keys=True)
                if data is None or data == [] or isinstance(data, dict) and data.get("data", "present") is None:
                    doc["status"] = "NO_DATA"
        doc["text_truncated"] = len(text) > 18000
        doc["text"] = text[:18000]
        if not text.strip():
            doc["status"] = "NO_DATA"
        return doc
