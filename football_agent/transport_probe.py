"""Bounded, archived diagnostics for already-discovered public source URLs.

Every attempt has an on-disk intent before acquisition. The tool records
exception classes and numeric error codes, never exception messages or proxy
values. It does not change process or system networking and does not retry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

from filelock import FileLock
import requests

from football_platform.collectors.http_transport import RequestTransport, TransportError
from football_spider.storage.raw_archive import RawArchive
from .audit import atomic_write, digest
from .inquiry_tools import BoundedSession, OpenWebTools


VERSION = "archived-public-transport-diagnostic-v1"
MAX_ATTEMPTS = 4
USER_AGENTS = {
    "research": "FootballMind/0.1 research",
    "browser": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
}
TARGETS = {
    "england_fixtures": {
        "url": "https://www.englandfootball.com/england/mens-senior-team/fixtures-results",
        "params": {}, "headers": {},
    },
    "england_news": {
        "url": "https://www.englandfootball.com/england/mens-senior-team/news",
        "params": {}, "headers": {},
    },
    "spftll_list": {
        "url": "https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll",
        "params": {"dateFormat": "", "notSingle": "false", "mKind": 0, "reqtype": 0},
        "headers": {"Accept": "application/json, text/plain, */*", "Referer": "https://xx.spftll.cn/",
                    "X-Requested-With": "XMLHttpRequest", "XMIME-TYPE": "3", "AppType": "0", "DeviceKind": "1"},
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment_flags():
    """Only presence booleans; do not export endpoints, file paths or secrets."""
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy",
            "all_proxy", "no_proxy", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR")
    return {key: bool(os.environ.get(key)) for key in keys}


def exception_facts(error):
    """Inspect nested exceptions without serializing their potentially secret text."""
    pending, seen, result = [error], set(), []
    while pending and len(result) < 12:
        current = pending.pop(0)
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        row = {"type": type(current).__name__}
        for key in ("errno", "winerror"):
            value = getattr(current, key, None)
            if type(value) is int:
                row[key] = value
        result.append(row)
        pending.extend([current.__cause__, current.__context__, getattr(current, "reason", None)])
        pending.extend(x for x in current.args if isinstance(x, BaseException))
    names = {r["type"] for r in result}
    codes = {r[k] for r in result for k in ("errno", "winerror") if k in r}
    categories = []
    if "ProxyError" in names:
        categories.append("proxy_error")
    if names & {"NameResolutionError", "gaierror"}:
        categories.append("name_resolution_error")
    if "ConnectionRefusedError" in names or codes & {111, 10061}:
        categories.append("connection_refused")
    if names & {"SSLError", "SSLCertVerificationError", "ClientConnectorCertificateError"}:
        categories.append("tls_error")
    if names & {"ConnectTimeout", "ReadTimeout", "Timeout", "TimeoutError", "ConnectTimeoutError", "ReadTimeoutError"}:
        categories.append("timeout")
    if names & {"ConnectionResetError", "RemoteDisconnected", "ProtocolError"}:
        categories.append("connection_reset_or_closed")
    return {"chain": result, "categories": categories or ["unclassified_connection_failure"],
            "exception_messages_exported": False}


class _NoAuthentication(requests.auth.AuthBase):
    def __call__(self, request):
        return request


class DiagnosticSession(BoundedSession):
    """Same bounded reader; only the explicit session routing policy changes.

    A truthy no-op auth object suppresses requests' automatic netrc lookup.
    Direct mode still honors an explicitly configured CA bundle and never
    disables TLS verification. No environment or global adapter is modified.
    """
    def __init__(self, route):
        if route not in {"environment", "direct"}:
            raise ValueError("Unknown diagnostic route")
        super().__init__()
        self.auth = _NoAuthentication()
        self.trust_env = route == "environment"
        if route == "direct":
            self.verify = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE") or True
        self.failure = None

    def rebuild_auth(self, prepared_request, response):
        # Requests prepares response.next even when redirects are not followed;
        # its default hook may consult netrc there. This diagnostic has no auth.
        prepared_request.headers.pop("Authorization", None)

    def get(self, *args, **kwargs):
        try:
            return super().get(*args, **kwargs)
        except requests.RequestException as exc:
            self.failure = exception_facts(exc)
            raise


def run_probe(root, target, route, *, user_agent_profile="research", session_factory=DiagnosticSession):
    if target not in TARGETS or route not in {"environment", "direct"} or user_agent_profile not in USER_AGENTS:
        raise ValueError("Only registered targets and session policies are accepted")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    probe_id = str(uuid4())
    declared = TARGETS[target]
    with FileLock(str(root / "probe.lock")):
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {
            "version": VERSION, "created_at": utc_now(), "max_http_attempts": MAX_ATTEMPTS,
            "environment_configured": environment_flags(), "probes": [],
            "global_network_configuration_changed": False, "authentication_files_read": False,
        }
        if len(status["probes"]) >= MAX_ATTEMPTS:
            raise ValueError("The diagnostic HTTP attempt budget is exhausted")
        intent = {"version": VERSION, "probe_id": probe_id, "registered_at": utc_now(),
                  "target": target, "method": "GET", "url": declared["url"], "params": declared["params"],
                  "request_headers": {"User-Agent": USER_AGENTS[user_agent_profile], **declared["headers"]},
                  "user_agent_profile": user_agent_profile,
                  "session_route": route, "tls_verification": True, "max_attempts": 1,
                  "connect_timeout_seconds": 4, "read_timeout_seconds": 8,
                  "max_response_bytes": 2_000_000, "automatic_redirects": False,
                  "no_automatic_retry": True, "evidence_scope": "transport_diagnostic_not_agent_policy_trial"}
        folder = root / probe_id
        atomic_write(folder / "intent.json", intent)
        status["probes"].append({"probe_id": probe_id, "target": target, "route": route,
                                  "intent_sha256": digest(intent), "state": "RESERVED_BEFORE_NETWORK"})
        status["updated_at"] = utc_now()
        atomic_write(status_path, status)

    archive, observations = RawArchive(root / "evidence"), []

    def save_attempt(**facts):
        observations.append(archive.save(
            facts["source"], facts["endpoint"], facts["params"], facts["body"],
            status=facts["http_status"], headers=facts["response_headers"], request_id=facts["request_id"],
            observed_at=facts["observed_at"], observation_id=facts["observation_id"], error_class=facts["error_class"],
            context={"diagnostic_probe_id": probe_id, "intent_sha256": digest(intent),
                     "session_route": route, "requested_at": facts["requested_at"].isoformat(),
                     "evidence_scope": "transport_diagnostic_not_agent_policy_trial"}))

    session = session_factory(route)
    category = None
    with RequestTransport("source_recovery", on_attempt=save_attempt, session=session,
                          max_attempts=1, request_budget=1, interval=0, timeout=(4, 8)) as transport:
        try:
            transport.get(declared["url"], declared["params"], headers=intent["request_headers"])
        except TransportError as exc:
            category = exc.category
        stats = dict(transport.stats)
    if len(observations) != 1:
        raise RuntimeError("A reserved diagnostic attempt needs exactly one archive observation")
    obs = observations[0]
    body = archive.read_body(obs)
    from urllib.parse import urlsplit
    prepared = {"tool": "read_url", "endpoint": declared["url"], "params": declared["params"], "source": "source_recovery"}
    projected = OpenWebTools(archive, [urlsplit(declared["url"]).hostname]).project(obs, prepared)
    atomic_write(folder / "document.json", projected)
    result = {"version": VERSION, "probe_id": probe_id, "target": target, "route": route,
              "completed_at": utc_now(), "intent_sha256": digest(intent),
              "observation": obs.model_dump(mode="json"), "body_bytes": len(body),
              "body_sha256": sha256(body).hexdigest(), "error_category": category,
              "failure_detail": getattr(session, "failure", None), "transport_stats": stats,
              "document_status": projected["status"], "extracted_characters": len(projected["text"]),
              "discovered_links": len(projected["links"]),
              "semantic_answer_verified": False, "failed_acquisition_implies_no_lineup": False}
    atomic_write(folder / "result.json", result)
    with FileLock(str(root / "probe.lock")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        record = next(p for p in status["probes"] if p["probe_id"] == probe_id)
        record.update(state="COMPLETED", http_status=obs.http_status, error_class=obs.error_class,
                      body_bytes=len(body), document_status=projected["status"],
                      failure_detail=result["failure_detail"], observation_id=obs.id,
                      result_sha256=digest(result), result_path=str((folder / "result.json").relative_to(root)))
        status["updated_at"] = utc_now()
        status["actual_archived_attempts"] = sum(p["state"] == "COMPLETED" for p in status["probes"])
        atomic_write(status_path, status)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--route", choices=("environment", "direct"), required=True)
    parser.add_argument("--user-agent-profile", choices=sorted(USER_AGENTS), default="research")
    args = parser.parse_args()
    result = run_probe(args.root, args.target, args.route, user_agent_profile=args.user_agent_profile)
    print(json.dumps({k: result[k] for k in ("probe_id", "target", "route", "body_bytes", "error_category", "failure_detail", "document_status", "extracted_characters")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
