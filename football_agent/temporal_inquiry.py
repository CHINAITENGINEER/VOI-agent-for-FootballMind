"""Admission gate that connects temporal source eligibility to InquiryWorkflow.

The gate is deliberately optional so existing operational inquiry traces keep
their original semantics.  When enabled, it publishes source timing metadata to
the Agent and rejects a proposed read before any HTTP reservation or request.
The gate checks known source routes; discovery/search remains a separate policy
choice and is disabled by default for historical cutoff replays.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .inquiry_tools import public_url
from .temporal_snapshot import _status, _time


VERSION = "temporal-inquiry-gate-v1"


def _canonical(url: str, hosts: list[str]) -> str:
    normalized = public_url(url, hosts)
    p = urlsplit(normalized)
    return urlunsplit((p.scheme, p.netloc.lower(), p.path or "/", p.query, ""))


class TemporalInquiryGate:
    """Check candidate actions against an explicit as-of source catalog."""

    def __init__(self, *, prediction_cutoff: str | datetime, kickoff: str | datetime,
                 entries: list[dict[str, Any]], allowed_hosts: list[str], allow_discovery: bool = False):
        self.prediction_cutoff = _time(prediction_cutoff)
        self.kickoff = _time(kickoff)
        if self.prediction_cutoff >= self.kickoff:
            raise ValueError("Temporal gate cutoff must precede kickoff")
        self.allowed_hosts = sorted(set(allowed_hosts))
        if not self.allowed_hosts:
            raise ValueError("Temporal gate needs an explicit host catalog")
        self.allow_discovery = bool(allow_discovery)
        self.entries = []
        seen = set()
        for raw in entries:
            required = {"source_id", "url", "publisher", "information_role", "menu_note",
                        "document_scope", "observed_at", "available_at", "http_status", "error_class"}
            if set(raw) != required or raw["source_id"] in seen:
                raise ValueError("Temporal gate source entries must be complete and unique")
            item = deepcopy(raw)
            item["url"] = _canonical(item["url"], self.allowed_hosts)
            status, reason = _status(item, self.prediction_cutoff)
            item.update({"availability_status": status, "availability_reason": reason,
                         "body_hidden": True})
            seen.add(item["source_id"])
            self.entries.append(item)
        if len({item["url"] for item in self.entries}) != len(self.entries):
            raise ValueError("Temporal gate source URLs must be unique")
        self._by_url = {item["url"]: item for item in self.entries}

    def configuration(self) -> dict[str, Any]:
        return {"version": VERSION, "prediction_cutoff": self.prediction_cutoff.isoformat(),
                "kickoff": self.kickoff.isoformat(), "allowed_hosts": self.allowed_hosts,
                "allow_discovery": self.allow_discovery,
                "entries": [{k: item[k] for k in ("source_id", "url", "publisher", "information_role",
                                                    "menu_note", "document_scope", "observed_at", "available_at",
                                                    "http_status", "error_class", "availability_status",
                                                    "availability_reason")} for item in self.entries],
                "selection_semantics": "only_ELIGIBLE_known_reads_are_admitted_before_http"}

    def public(self, *, now: str | datetime) -> dict[str, Any]:
        stamp = _time(now)
        return {"version": VERSION, "now": stamp.isoformat(),
                "prediction_cutoff": self.prediction_cutoff.isoformat(),
                "kickoff": self.kickoff.isoformat(), "allow_discovery": self.allow_discovery,
                "entries": [{k: item[k] for k in ("source_id", "url", "publisher", "information_role",
                                                    "menu_note", "availability_status", "availability_reason",
                                                    "body_hidden")} for item in self.entries],
                "action_scope": "read_url_routes_must_match_an_ELIGIBLE_entry; search_discovery_policy_explicit"}

    def _admit_candidate(self, candidate: dict[str, Any], public: dict[str, Any]):
        if candidate["tool"] == "search_web":
            if not self.allow_discovery:
                raise ValueError("Temporal gate does not admit unbound discovery in historical mode")
            return
        if candidate["tool"] != "read_url":
            raise ValueError("Temporal gate received an unsupported tool")
        url = _canonical(candidate["url"], self.allowed_hosts)
        entry = self._by_url.get(url)
        if entry is None:
            raise ValueError("Read URL is absent from the temporal source catalog")
        if entry["availability_status"] != "ELIGIBLE":
            raise ValueError("Read URL is not temporally eligible")
        known = set(public["task"].get("seed_urls", []))
        for doc in public.get("documents", []):
            known.update(link["url"] for link in doc.get("links", []))
        if url not in {_canonical(item, self.allowed_hosts) for item in known}:
            raise ValueError("Temporal source is not yet an exposed task route")

    def validate_response(self, response: dict[str, Any], public: dict[str, Any]) -> dict[str, Any]:
        proposed = {candidate["candidate_id"]: candidate for candidate in response["candidates"]}
        for candidate in proposed.values():
            self._admit_candidate(candidate, public)
        decision = response["decision"]
        if decision["kind"] == "query":
            if decision["candidate_id"] not in proposed:
                raise ValueError("Temporal query must select a proposed admitted route")
            self._admit_candidate(proposed[decision["candidate_id"]], public)
        return response

