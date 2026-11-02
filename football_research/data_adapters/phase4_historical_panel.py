"""Finite historical panels using the existing transport, archive and journal.

The journal measures today's collection job, with no retrospective fixture
binding. Its legacy prospective schema name is not an exported data mode.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl
from uuid import uuid4

from bs4 import BeautifulSoup
from football_agent.inquiry_tools import public_url
from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.transport_probe import DiagnosticSession
from football_platform.collectors.http_transport import RequestTransport
from football_platform.information.acquisition_journal import AcquisitionDecisionJournal
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes, digest
from .phase3_historical_panel import Adapter, append_row, load_rows, write_rows, now, publication

VERSION = "phase4-finite-historical-panel-v1"
SOURCE = "phase4_public_http"


def budget_owners(journal, run_id):
    """Reservations, including unresolved ones, count toward every owner cap."""
    run = journal.read(run_id)
    return Counter(journal.store.get_blob(run_id, intent["input_ref"])["physical_match_id"]
                   for intent in run["intents"].values())


def enforce_owner_budget(plan, owner, owners, gate=None):
    allowed = {x["physical_match_id"] for x in plan["selected_fixtures"]}
    if owner not in allowed:
        raise ValueError("Fixture is outside the frozen cohort")
    if owners[owner] >= plan["budget"]["per_fixture_actual_http_max"]:
        raise ValueError("Frozen per-fixture HTTP cap reached")
    if owner in plan["pilot_order"]:
        if sum(owners[x] for x in plan["pilot_order"]) >= plan["budget"]["pilot_http_max"]:
            raise ValueError("Frozen pilot HTTP cap reached")
    elif not gate or not gate.get("passed"):
        raise ValueError("Remaining cohort requires the completed four-fixture pilot gate")


def paragraphs(body):
    rows, offset = [], 0
    for text in body.splitlines():
        start = offset
        offset += len(text) + 1
        if text.strip():
            rows.append({"paragraph_id": len(rows), "start": start, "end": start + len(text), "text": text})
    return rows


class PanelAdapter(Adapter):
    def __init__(self, root, *, session=None):
        self.root = Path(root)
        self.plan = json.loads((self.root / "frozen-plan.json").read_bytes())
        self.archive = RawArchive(self.root)
        self.journal = AcquisitionDecisionJournal(self.root / "journal", self.archive)
        ref = self.root / "journal-reference.json"
        if not ref.exists():
            run = self.journal.start(request_limit=self.plan["request_budget"],
                policy_version=VERSION, random_seed=self.plan["random_seed"], propensity_logging=False)
            atomic_write(ref, json_bytes({"run_id": run["run_id"], "created_at": run["created_at"],
                "scope": "current historical collection job; no historical Agent trajectory or prospective fixture binding",
                "export_information_mode": "historical_reconstruction"}))
        self.run_id = json.loads(ref.read_bytes())["run_id"]
        run = self.journal.read(self.run_id)
        if run["pending_request_id"]:
            result = self.journal.recover_pending(self.run_id)
            if result["status"] != "RECOVERED_FROM_RAW_ARCHIVE":
                raise ValueError("Unresolved request: reconcile before any further HTTP")
            run = self.journal.read(self.run_id)
        self.current = self.receipt = None
        self.active_request_id = None
        self.candidate_menu = []
        self.transport = RequestTransport(SOURCE, on_attempt=self.persist,
            session=session or DiagnosticSession("environment"), max_attempts=1,
            request_budget=max(0, self.plan["request_budget"] - run["budget"]["charged_requests"]),
            interval=1.0, timeout=(6,16))

    def persist(self, **facts):
        save_start = now()
        self.current.update(journal_run_id=self.run_id, journal_request_id=self.active_request_id,
                            archive_save_started_at=save_start)
        super().persist(**facts)
        self.receipt.update(archive_save_started_at=save_start, archive_save_returned_at=now())

    def fetch(self, fixture, url, **kwargs):
        url = public_url(url, self.plan["allowed_hosts"])
        if kwargs.get("query") is not None:
            raise ValueError("Use explicit original query URL; do not silently route to legacy RSS")
        owner = fixture["physical_match_id"]
        with self.journal.store.lock(self.run_id):
            gate_path = self.root / "reports/pilot-gate.json"
            gate = json.loads(gate_path.read_bytes()) if gate_path.exists() else None
            enforce_owner_budget(self.plan, owner, budget_owners(self.journal, self.run_id), gate)
            # Prevent accidental replay after a caller loses its output. Failed
            # requests also require an explicitly new versioned plan to repeat.
            parts = urlsplit(url)
            endpoint, params = f"{parts.scheme}://{parts.netloc}{parts.path}", dict(parse_qsl(parts.query))
            for old in self.archive.observations(SOURCE):
                if old.endpoint == endpoint and old.params == params:
                    raise ValueError("URL already archived in this batch; reuse raw instead of requesting again")
            rid = str(uuid4())
            action = kwargs.get("candidate_id") or "probe:" + digest(url)[:24]
            metadata = deepcopy(self.candidate_menu) or [{"candidate_id": action, "url": url,
                "title": None, "snippet": None, "rank": 1, "origin": "predeclared_source_probe"}]
            ids = [x["candidate_id"] for x in metadata]
            if action not in ids:
                raise ValueError("Read action was not exposed in the frozen candidate menu")
            begun = self.journal.begin(self.run_id, request_id=rid, action_id=action,
                input_parameters={"endpoint": endpoint, "params": params, "source": SOURCE,
                    "original_url": url, "physical_match_id": owner, "mode": "historical_reconstruction",
                    "tool": kwargs.get("tool", "read_url"), "budget_owner_assigned_before_request": True},
                candidates=ids, candidate_metadata=metadata, selection_basis="manual_predeclared",
                policy_version=VERSION, selection_probability=None)
            if not begun["execute"]:
                raise ValueError("Existing intent cannot execute a second time")
            self.active_request_id = rid
            obs, attempt, raw = super().fetch(fixture, url, **kwargs)
            receipt = self.journal.complete(self.run_id, request_id=rid, observation=obs)
            attempt.update(journal_run_id=self.run_id, journal_request_id=rid,
                policy_version=VERSION, original_policy_trace=False,
                selection_probability=None, selection_probability_basis="historical_policy_unobserved",
                selected_by_policy=False, selected_by_collector=True,
                archive_save_started_at=self.receipt["archive_save_started_at"],
                archive_save_returned_at=self.receipt["archive_save_returned_at"],
                captured_at=self.receipt["archive_save_returned_at"],
                durability_confirmed_at=None, durability_basis="save_return_measured; directory_fsync_not_measured",
                result_observed=True, receipt_available_at=receipt["receipt_at"],
                collector_handoff_at=now(), agent_handoff_at=None, available_at=None,
                document_admitted=False, document_available_at=None,
                response_headers=obs.context.get("response_headers", {}))
            return obs, attempt, raw

    def discover(self, fixture, url):
        self.candidate_menu = []
        super().discover(fixture, url)
        attempt = load_rows(self.root, "attempts.jsonl")[-1]
        menu = load_rows(self.root, "discovery.jsonl")[-1]
        self.journal.record_processing(self.run_id, request_id=attempt["journal_request_id"],
            status="DATA_AVAILABLE" if menu["visible_candidates"] else "NO_DATA",
            extraction_sha256=digest(menu["visible_candidates"]))
        return menu

    def freeze_menu(self, fixture, parent, candidates, rule):
        """Copy only actually exposed pre-read metadata; no body inspection."""
        selected = sorted({c["url"]: deepcopy(c) for c in candidates}.values(), key=lambda c: c["url"])
        menu_id = "finite:" + digest({"fixture": fixture["physical_match_id"],
            "parent": parent["discovery_id"], "candidates": selected, "rule": rule})[:32]
        menu = {**deepcopy(parent), "discovery_id": menu_id, "physical_match_id": fixture["physical_match_id"],
            "parent_discovery_id": parent["discovery_id"], "visible_candidates": selected,
            "finite_menu_frozen_at": now(), "candidate_order": "lexicographic_url", "filter_rule": rule,
            "menu_type": "finite_reconstructed_menu", "original_agent_menu_proven": False,
            "original_policy_trace": False, "selection_probability": None, "policy_version": VERSION,
            "available_at": None, "historical_menu_observed_at": None}
        if not any(x["discovery_id"] == menu_id for x in load_rows(self.root, "discovery.jsonl")):
            append_row(self.root, "discovery.jsonl", menu)
        return menu

    def read(self, fixture, candidate, discovery):
        self.candidate_menu = discovery["visible_candidates"]
        obs, attempt, raw = self.fetch(fixture, candidate["url"], discovery_id=discovery["discovery_id"],
                                      candidate_id=candidate["candidate_id"])
        doc = None
        if attempt["status"] == "returned":
            parsed_at = now()
            soup = BeautifulSoup(raw, "html.parser")
            heading = soup.find("h1") or soup.find("title")
            title = heading.get_text(" ", strip=True) if heading else None
            pub, mod, source_times, proof = publication(soup)
            projector = NewsDiscoveryTools(self.archive, self.plan["allowed_hosts"], projection_version=11)
            projection = projector.project(obs, {"tool": "read_url", "endpoint": obs.endpoint,
                "params": obs.params, "source": "open_web"})
            body, method, pointers, structured = projection.get("text", ""), projection.get("news_projection_version"), [], {}
            try:
                host = urlsplit(candidate["url"]).hostname
                if host == "www.fulhamfc.com":
                    from .fulham_nuxt import extract_archived_news
                    structured = extract_archived_news(raw, candidate["url"])
                    body, method = structured["body"], structured["parser_version"]
                    title = structured.get("title", title)
                    pub, mod = structured["source_published_at"], structured["source_modified_at"]
                    pointers = structured.get("body_pointers", [])
                    source_times = [{"field": "publishedDateTime", "text": structured["original_time_text"],
                                     "basis": "canonical_same_page_literal_nuxt"}]
                elif host == "www.tottenhamhotspur.com":
                    from .historical_visible_body import extract_spurs_visible_body
                    structured = extract_spurs_visible_body(raw, candidate["url"])
                    body, method, title = structured["body"], structured["parser_version"], structured["title"]
                    pointers = structured["body_pointers"]
                elif host == "www.liverpoolfc.com" and projection.get("status") == "OK":
                    leads = soup.select('article > header p[class*="articleHeaderNews__summary__"]')
                    if len(leads) == 1:
                        lead = leads[0].get_text(" ", strip=True)
                        body = lead + "\n\n" + body
                        pointers = [{"selector": 'article > header p[class*="articleHeaderNews__summary__"]', "text": lead}]
                elif projection.get("status") != "OK":
                    body = ""
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                body = ""
                attempt.update(status="parse_error", error_type=type(exc).__name__ + ": " + str(exc)[:160])
            if not body and attempt["status"] == "returned":
                attempt["status"] = "empty"
            kickoff = datetime.fromisoformat(fixture["kickoff"].replace("Z", "+00:00"))
            pre = bool(pub and datetime.fromisoformat(pub) < kickoff)
            modified_late = bool(mod and datetime.fromisoformat(mod) >= kickoff)
            doc = {"document_id": "document:" + obs.id, "version_id": "document:" + obs.id + ":panel-v1",
                "physical_match_id": fixture["physical_match_id"], "url": candidate["url"], "publisher": host,
                "title": title, "body": body, "language": "en", "paragraphs": paragraphs(body),
                "source_published_at": pub, "source_modified_at": mod, "source_time_evidence": source_times,
                "published_precision": "timestamp_with_timezone" if pub else "unknown_or_date_only",
                "original_time_text": json.dumps(source_times, ensure_ascii=False),
                "captured_at": attempt["captured_at"], "observed_at": obs.observed_at.isoformat(),
                "archive_save_started_at": attempt["archive_save_started_at"],
                "archive_save_returned_at": attempt["archive_save_returned_at"], "parsed_at": parsed_at,
                "available_at": None, "availability_status": "unknown", "archive_snapshot_at": None,
                "historical_version_unknown": True, "historical_snapshot_proven": False,
                "timing_basis": "current archived HTTP response; explicit source publication only; no historical body availability proof",
                "information_mode": "historical_reconstruction", "previous_version_id": None,
                "raw_path": obs.payload_path, "raw_sha256": obs.content_hash, "body_content_digest": digest(body),
                "body_digest_basis": "sha256_existing_json_bytes_of_body_string", "parser_version": method,
                "body_pointers": pointers, "links": projection.get("links", []),
                "document_kind": "historical_article_candidate", "fixture_tags": structured.get("fixture_tags", []),
                "source_cms_saved_at": structured.get("source_cms_saved_at"),
                "source_published_before_kickoff": pre, "source_modified_after_kickoff": modified_late,
                "fixture_binding_basis": "pending article-level association review",
                "historical_replay_status": "quarantined_pending_article_association",
                "source_assertion": True, "semantic_facts_independently_verified": False,
                "text_f_consumed": False, "text_f_status": "NOT_CONFIGURED", "model_status": "NOT_CONFIGURED",
                "prediction_before": None, "prediction_after": None, "semantic_delta": None,
                "strict_prospective_g_eligible": False, "empirical_g_eligible": False,
                "original_policy_trace": False, "propensity": None,
                "pre_read_metadata": {"candidate_id": candidate["candidate_id"], "url": candidate["url"],
                    "title": candidate.get("title"), "snippet": candidate.get("snippet"), "rank": candidate["rank"],
                    "parent_discovery_id": discovery["discovery_id"], "parent_raw_path": discovery["raw_path"],
                    "parent_raw_sha256": discovery["raw_sha256"], "observed_at": discovery["observed_at"],
                    "title_from_revealed_body": False, "model_consumed": False}}
            append_row(self.root, "documents.jsonl", doc)
            attempt.update(returned_document_id=doc["document_id"], returned_version_ids=[doc["version_id"]],
                           document_admitted=False, document_available_at=None)
        append_row(self.root, "attempts.jsonl", attempt)
        self.journal.record_processing(self.run_id, request_id=attempt["journal_request_id"],
            status="REQUIRES_REVIEW" if doc and doc["body"] else "NO_DATA",
            extraction_sha256=digest(doc if doc else {"status": attempt["status"]}))
        return attempt, doc

    def close(self):
        self.transport.session.close()

