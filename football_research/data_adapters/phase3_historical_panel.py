"""Bounded historical panel adapter; all HTTP/archive mechanics are reused.

This is intentionally separate from the prospective runtime. Menus are observed
now and explicitly reconstructed, and publication is never Agent availability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit, parse_qsl, urljoin
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from football_agent.transport_probe import DiagnosticSession
from football_platform.collectors.http_transport import RequestTransport, TransportError
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes, digest

VERSION = "phase3-historical-panel-adapter-v2"
FILES = ("fixtures.jsonl", "documents.jsonl", "discovery.jsonl", "attempts.jsonl",
         "context_snapshots.jsonl", "labels/results.jsonl")


def now():
    return datetime.now(timezone.utc).isoformat()


def load_rows(root, name):
    path = root / name
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def write_rows(root, name, rows):
    atomic_write(root / name, b"".join(json_bytes(r) + b"\n" for r in rows))


def append_row(root, name, row):
    rows = load_rows(root, name)
    rows.append(row)
    write_rows(root, name, rows)


def init(root):
    plan = json.loads((root / "frozen-plan.json").read_bytes())
    if (root / "fixtures.jsonl").exists():
        return
    archive = RawArchive(root)
    evidence_path = Path(plan["identity_and_result_archive_manifest"])
    evidence = json.loads(evidence_path.read_bytes())
    if evidence["http_status"] != 200 or evidence["url"] != "https://understat.com/getLeagueData/EPL/2025":
        raise ValueError("Historical result source is not the frozen verified endpoint")
    obs = archive.save("understat", evidence["url"], {}, Path(evidence["payload"]).read_bytes(),
        status=evidence["http_status"], headers={"Content-Type": evidence["content_type"]},
        observation_id=evidence["evidence_id"], observed_at=datetime.fromisoformat(evidence["observed_at"]),
        context={"original_evidence_manifest": str(evidence_path),
                 "original_requested_at": evidence["request_started_at"],
                 "archive_imported_at": now(),
                 "timezone_policy": "UTC interpreted by existing verified Understat frontend conversion; original date strings retained"})
    if obs.content_hash != evidence["sha256"]:
        raise ValueError("Original raw evidence digest mismatch")
    data = json.loads(archive.read_body(obs))
    matches = {str(x["id"]): x for x in data["dates"]}
    fixtures, labels, contexts = [], [], []
    for chosen in plan["selected_fixtures"]:
        source = matches[chosen["source_id"]]
        if not source["isResult"]:
            raise ValueError("Frozen fixture is not independently finished in raw results")
        fixture = {**chosen, "fixture_id": chosen["physical_match_id"], "competition": "Premier League",
                   "season": "2025/26", "kickoff_at": chosen["kickoff"], "kickoff_timezone": "UTC",
                   "identity_source_url": obs.endpoint, "fixture_source_url": obs.endpoint,
                   "information_mode": "historical_reconstruction", "registered_at": now(),
                   "registration_source": "understat", "pre_kickoff_frozen": False,
                   "raw_path": obs.payload_path, "raw_sha256": obs.content_hash,
                   "source_original_time_text": source["datetime"], "timing_basis": obs.context["timezone_policy"],
                   "source_observed_at": obs.observed_at.isoformat(), "available_at": None,
                   "availability_status": "unknown", "policy_arm": "not_configured"}
        fixtures.append(fixture)
        labels.append({"physical_match_id": chosen["physical_match_id"],
                       "regulation_home_goals": int(source["goals"]["h"]),
                       "regulation_away_goals": int(source["goals"]["a"]),
                       "result_status": "finished", "score_period": "regulation",
                       "result_source_url": obs.endpoint, "source": "understat", "source_id": chosen["source_id"],
                       "result_published_at": None, "result_observed_at": obs.observed_at.isoformat(),
                       "raw_path": obs.payload_path, "raw_sha256": obs.content_hash,
                       "timing_basis": "real original Understat HTTP response observation; source result publication unknown; no synthetic kickoff-plus-duration clock",
                       "information_mode": "post_match_outcome", "independent_outcome_verified": True,
                       "independence_basis": "Understat result publisher distinct from club/news text publishers",
                       "archive_reused": True, "imported_at": now()})
        fields = {name: {"value": None, "observed_at": None, "status": "not_collected"}
                  for name in ("ranking", "points", "qualification_status", "schedule", "injuries", "lineup_status", "odds", "weather")}
        fields["odds"].update(quote_time=None, captured_at=None)
        contexts.append({"snapshot_id": chosen["physical_match_id"] + ":unknown-historical-context",
                         "physical_match_id": chosen["physical_match_id"], "information_mode": "historical_reconstruction",
                         "snapshot_kind": "historical_unknown", "observed_at": None, "captured_at": now(),
                         "available_at": None, "prediction_cutoff": None, "prediction_before": None,
                         "prediction_after": None, "model_status": "NOT_CONFIGURED",
                         "historical_agent_state_proven": False, **fields})
    for name, rows in zip(FILES, (fixtures, [], [], [], contexts, labels)):
        write_rows(root, name, rows)


def publication(soup):
    values = []
    for tag in soup.find_all("meta"):
        key = (tag.get("property") or tag.get("name") or "").lower()
        if key in {"article:published_time", "article:modified_time", "datepublished", "datemodified", "date", "publishdate", "pubdate"}:
            values.append({"field": key, "text": tag.get("content"), "basis": "html_meta"})
    def scan(value):
        if isinstance(value, dict):
            kind = value.get("@type", "")
            if any(k in str(kind) for k in ("Article", "NewsArticle", "BlogPosting")):
                for key in ("datePublished", "dateModified"):
                    if value.get(key):
                        values.append({"field": key.lower(), "text": value[key], "basis": "article_jsonld"})
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    scan(nested)
        elif isinstance(value, list):
            for nested in value:
                scan(nested)
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            scan(json.loads(tag.string or tag.get_text()))
        except (ValueError, TypeError):
            pass
    for tag in soup.find_all("time"):
        values.append({"field": "visible_time", "text": tag.get("datetime") or tag.get_text(" ", strip=True), "basis": "html_time"})
    def exact(kind):
        for item in values:
            if ("modif" in item["field"]) != (kind == "modified"):
                continue
            try:
                dt = datetime.fromisoformat(str(item["text"]).replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    return dt.isoformat(), item
            except ValueError:
                pass
        return None, None
    published, proof = exact("published")
    modified, _ = exact("modified")
    return published, modified, values, proof


class Adapter:
    def __init__(self, root):
        self.root = root
        self.plan = json.loads((root / "frozen-plan.json").read_bytes())
        self.archive = RawArchive(root)
        already = len(self.archive.observations("phase3_public_http"))
        self.current = None
        self.receipt = None
        self.transport = RequestTransport("phase3_public_http", on_attempt=self.persist,
            session=DiagnosticSession("environment"), max_attempts=1,
            request_budget=max(0, self.plan["request_budget"] - already), interval=1.0, timeout=(6,16))

    def persist(self, **facts):
        saved = self.archive.save(facts["source"], facts["endpoint"], facts["params"], facts["body"],
            status=facts["http_status"], headers=facts["response_headers"], request_id=facts["request_id"],
            observation_id=facts["observation_id"], observed_at=facts["observed_at"], error_class=facts["error_class"],
            context={**self.current, "requested_at": facts["requested_at"].isoformat(),
                     "response_observed_at": facts["observed_at"].isoformat(),
                     "response_headers": facts["response_headers"], "attempt": facts["attempt"],
                     "information_mode": "historical_reconstruction", "parser_version": VERSION})
        self.receipt = {"observation": saved, "captured_at": now()}

    def fetch(self, fixture, url, *, query=None, discovery_id=None, candidate_id=None, tool="read_url"):
        decision = now()
        self.current = {"physical_match_id": fixture["physical_match_id"], "tool": tool,
                        "query": query, "discovery_id": discovery_id, "candidate_id": candidate_id,
                        "selection_basis": "manual_predeclared", "decision_at": decision}
        self.receipt = None
        parts = urlsplit(url)
        endpoint = f"{parts.scheme}://{parts.netloc}{parts.path}"
        params = dict(parse_qsl(parts.query))
        if query is not None:
            params = {"q": query, "format": "rss"}
        start = time.monotonic()
        err = None
        try:
            self.transport.get(endpoint, params, headers={"User-Agent": "FootballMind/0.1 research"})
        except TransportError as exc:
            err = exc.category
        if self.receipt is None:
            raise RuntimeError("No HTTP observation: " + str(err))
        obs = self.receipt["observation"]
        status = "returned" if err is None else "timeout" if "TIMEOUT" in str(err).upper() else "http_error" if obs.http_status else "failed"
        attempt = {"attempt_id": "attempt:" + obs.id, "physical_match_id": fixture["physical_match_id"],
                   "discovery_id": discovery_id, "candidate_id": candidate_id, "action_id": tool,
                   "tool": tool, "requested_at": obs.context["requested_at"], "started_at": obs.context["requested_at"],
                   "response_at": obs.observed_at.isoformat(), "completed_at": obs.observed_at.isoformat(),
                   "response_observed_at": obs.observed_at.isoformat(), "captured_at": self.receipt["captured_at"],
                   "status": status, "http_status": obs.http_status, "status_code": obs.http_status,
                   "error_type": err or obs.error_class, "error_code": err or obs.error_class,
                   "returned_document_id": None, "returned_version_ids": [],
                   "latency_ms": round((time.monotonic()-start)*1000, 3), "request_count": 1,
                   "token_cost": None, "monetary_cost": None,
                   "raw_path": obs.payload_path, "raw_sha256": obs.content_hash,
                   "information_mode": "historical_reconstruction", "historical_capture": True,
                   "selected_by_policy": True, "selection_basis": "manual_predeclared",
                   "policy_version": VERSION, "policy_arm": "not_configured", "selection_probability": None,
                   "selection_probability_basis": "not_recorded", "decision_at": decision,
                   "actual_historical_agent_action": False}
        return obs, attempt, self.archive.read_body(obs)

    def search(self, fixture, query):
        obs, attempt, raw = self.fetch(fixture, "https://www.bing.com/search", query=query, tool="search_web")
        candidates = []
        if attempt["status"] == "returned":
            try:
                feed = ET.fromstring(raw)
                for i,item in enumerate(feed.findall("./channel/item"),1):
                    candidates.append({"candidate_id": f"candidate:{obs.id}:{i}", "url": item.findtext("link"),
                        "title": item.findtext("title"), "snippet": item.findtext("description"), "rank": i,
                        "action_type": "read_url", "selected": False, "selection_status": "not_requested"})
                if not candidates:
                    attempt["status"] = "empty"
            except ET.ParseError:
                attempt["status"] = "parse_error"
                attempt["error_type"] = attempt["error_code"] = "invalid_rss"
        discovery = {"discovery_id": "discovery:" + obs.id, "physical_match_id": fixture["physical_match_id"],
                     "observed_at": obs.observed_at.isoformat(), "query_or_parent": query, "query": query,
                     "parent_url": None, "provider": "bing_rss", "raw_path": obs.payload_path,
                     "raw_sha256": obs.content_hash, "visible_candidates": candidates,
                     "status": attempt["status"], "truncation_or_cap": "provider_results_as_returned",
                     "original_agent_menu_proven": False, "information_mode": "historical_reconstruction",
                     "menu_type": "reconstructed_menu", "historical_menu_observed_at": None,
                     "policy_version": VERSION, "available_at": None, "captured_at": attempt["captured_at"]}
        append_row(self.root,"discovery.jsonl",discovery)
        append_row(self.root,"attempts.jsonl",attempt)
        print(json.dumps({"fixture": fixture["source_id"], "status": attempt["status"], "candidates": candidates},ensure_ascii=True),flush=True)

    def discover(self, fixture, url):
        """Retain the complete current official robots/sitemap/page link menu."""
        obs, attempt, raw = self.fetch(fixture,url,tool="official_archive_discovery")
        text=raw.decode("utf-8",errors="replace")
        links=[]
        if attempt["status"]=="returned":
            if url.endswith("robots.txt"):
                links=[{"url":line.split(":",1)[1].strip(),"title":"Sitemap","snippet":line}
                       for line in text.splitlines() if line.lower().startswith("sitemap:")]
            else:
                try:
                    tree=ET.fromstring(text)
                    for node in tree.iter():
                        if node.tag.split("}")[-1] in {"url","sitemap"}:
                            values={x.tag.split("}")[-1]:(x.text or "") for x in node}
                            if values.get("loc"):
                                links.append({"url":values["loc"],"title":"","snippet":json.dumps(values,ensure_ascii=False)})
                except ET.ParseError:
                    soup=BeautifulSoup(text,"html.parser")
                    links=[{"url":urljoin(url,a["href"]),"title":a.get_text(" ",strip=True),"snippet":None}
                           for a in soup.find_all("a",href=True)]
            if not links:
                attempt["status"]="empty"
        candidates=[{"candidate_id":f"candidate:{obs.id}:{i}","rank":i,"action_type":"read_url",
                     "selected":False,"selection_status":"not_requested",**item} for i,item in enumerate(links,1)]
        discovery={"discovery_id":"discovery:"+obs.id,"physical_match_id":fixture["physical_match_id"],
                   "observed_at":obs.observed_at.isoformat(),"query_or_parent":url,"query":None,"parent_url":url,
                   "provider":urlsplit(url).hostname,"raw_path":obs.payload_path,"raw_sha256":obs.content_hash,
                   "visible_candidates":candidates,"status":attempt["status"],"truncation_or_cap":None,
                   "original_agent_menu_proven":False,"information_mode":"historical_reconstruction",
                   "menu_type":"reconstructed_menu","historical_menu_observed_at":None,
                   "policy_version":VERSION,"available_at":None,"captured_at":attempt["captured_at"]}
        append_row(self.root,"discovery.jsonl",discovery)
        append_row(self.root,"attempts.jsonl",attempt)
        print(json.dumps({"fixture":fixture["source_id"],"url":url,"status":attempt["status"],
                          "candidate_count":len(candidates),"first_candidates":candidates[:8]},ensure_ascii=True),flush=True)

    def read(self, fixture, candidate, discovery):
        prior = [row for row in load_rows(self.root, "attempts.jsonl")
                 if row.get("physical_match_id") == fixture["physical_match_id"] and row.get("tool") == "read_url"]
        if len(prior) >= self.plan["read_budget_per_match"]:
            raise ValueError("Predeclared per-fixture article request cap reached")
        obs, attempt, raw = self.fetch(fixture, candidate["url"], discovery_id=discovery["discovery_id"], candidate_id=candidate["candidate_id"])
        if attempt["status"] == "returned":
            soup = BeautifulSoup(raw, "html.parser")
            heading = soup.find("h1") or soup.find("title")
            title = heading.get_text(" ",strip=True) if heading else (candidate.get("title") or "")
            published, modified, source_times, proof = publication(soup)
            article = soup.find("article") or soup.find("main")
            extraction = "article_element" if soup.find("article") else "main_element" if article else "whole_page_no_verified_article_container"
            content = article or soup
            for tag in content(["script","style","noscript","nav","header","footer","form"]):
                tag.decompose()
            body = content.get_text("\n",strip=True)
            structured = {}
            extraction_error = None
            if urlsplit(candidate["url"]).hostname == "www.fulhamfc.com":
                from .fulham_nuxt import extract_archived_news
                try:
                    structured = extract_archived_news(raw, candidate["url"])
                    body = structured["body"]
                    published, modified = structured["source_published_at"], structured["source_modified_at"]
                    extraction = structured["parser_version"]
                    source_times = [{"field": "publishedDateTime", "text": structured["original_time_text"],
                                     "basis": "same_page_literal_nuxt_bound_to_visible_title_and_canonical"}]
                except (ValueError, TypeError, KeyError, IndexError):
                    extraction_error = "unsupported_or_conflicting_nuxt_article"
                    attempt["status"] = "parse_error"
                    attempt["error_type"] = attempt["error_code"] = extraction_error
            kickoff = datetime.fromisoformat(fixture["kickoff"])
            pre = published is not None and datetime.fromisoformat(published) < kickoff
            late_revision = modified is not None and datetime.fromisoformat(modified) >= kickoff
            document = {"document_id": "document:" + obs.id, "version_id": "document:" + obs.id + ":v1",
                "physical_match_id": fixture["physical_match_id"], "url": candidate["url"],
                "publisher": urlsplit(candidate["url"]).hostname, "title": title, "body": body, "language":"en",
                "source_published_at":published,"source_modified_at":modified,"captured_at":attempt["captured_at"],
                "observed_at":obs.observed_at.isoformat(),"available_at":None,"availability_status":"unknown",
                "original_time_text":json.dumps(source_times,ensure_ascii=False),
                "timing_basis":"explicit current HTML article metadata only; original historical body/menu/availability unknown",
                "information_mode":"historical_reconstruction","raw_path":obs.payload_path,"raw_sha256":obs.content_hash,
                "previous_version_id":None,"document_kind":"historical_article_candidate",
                "fixture_binding_basis":"frozen fixture query and present-day candidate URL; semantic binding requires review",
                "published_precision":"timestamp_with_timezone" if published else "unknown_or_date_only",
                "historical_snapshot_proven":False,"later_body_revision_possible":True,
                "source_time_evidence":source_times,"extraction_method":extraction,"body_truncated":False,
                "source_published_before_kickoff":pre,"source_modified_after_kickoff":late_revision,
                "historical_replay_status":"quarantined_pending_semantic_binding_and_version_review",
                "post_match_contamination_risk":True,"empirical_g_eligible":False}
            document.update(extraction_error=extraction_error,
                            source_cms_saved_at=structured.get("source_cms_saved_at"),
                            fixture_tags=structured.get("fixture_tags", []))
            if not body:
                attempt["status"]="empty"
            append_row(self.root,"documents.jsonl",document)
            attempt["returned_document_id"]=document["document_id"]
            attempt["returned_version_ids"]=[document["version_id"]]
        append_row(self.root,"attempts.jsonl",attempt)
        menus=load_rows(self.root,"discovery.jsonl")
        for menu in menus:
            if menu["discovery_id"]==discovery["discovery_id"]:
                for item in menu["visible_candidates"]:
                    if item["candidate_id"]==candidate["candidate_id"]:
                        item.update(selected=True,selection_status=attempt["status"],attempt_id=attempt["attempt_id"])
        write_rows(self.root,"discovery.jsonl",menus)
        print(json.dumps({"url":candidate["url"],"status":attempt["status"],"document":attempt["returned_document_id"]}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--search",action="store_true")
    parser.add_argument("--query-suffix",default="August 2025 preview team news")
    parser.add_argument("--read-candidate-id",action="append",default=[])
    args=parser.parse_args()
    init(args.root)
    adapter=Adapter(args.root)
    fixtures=load_rows(args.root,"fixtures.jsonl")
    try:
        if args.search:
            for fixture in fixtures:
                query=f'{fixture["home_team"]} {fixture["away_team"]} {args.query_suffix}'
                if any(x.get("query")==query for x in load_rows(args.root,"discovery.jsonl")):
                    continue
                adapter.search(fixture,query)
        for cid in args.read_candidate_id:
            menu=next(m for m in load_rows(args.root,"discovery.jsonl") if any(c["candidate_id"]==cid for c in m["visible_candidates"]))
            candidate=next(c for c in menu["visible_candidates"] if c["candidate_id"]==cid)
            if candidate.get("selected"):
                continue
            fixture=next(f for f in fixtures if f["physical_match_id"]==menu["physical_match_id"])
            adapter.read(fixture,candidate,menu)
    finally:
        adapter.transport.close()
        atomic_write(args.root / ("transport-statistics-"+now().replace(":","-")+".json"),json_bytes(adapter.transport.stats))


if __name__=="__main__":
    main()
