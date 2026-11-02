"""Execute a predeclared search-channel diagnostic once, without redirects."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

from filelock import FileLock

from football_platform.collectors.http_transport import RequestTransport, TransportError
from football_spider.storage.raw_archive import RawArchive
from .audit import atomic_write, digest, redact
from .inquiry_tools import OpenWebTools
from .transport_probe import DiagnosticSession


VERSION = "registered-search-channel-diagnostic-v1"
ENDPOINTS = {"https://www.bing.com/search", "https://cn.bing.com/search"}


class _HeaderSession(DiagnosticSession):
    def __init__(self, route):
        super().__init__(route)
        self.location = None

    def get(self, *args, **kwargs):
        response = super().get(*args, **kwargs)
        location = response.headers.get("Location")
        self.location = redact(location) if isinstance(location, str) else None
        return response


def run_registered_probe(root, action_id, *, session_factory=_HeaderSession):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != VERSION or manifest.get("max_physical_http_requests") != 3:
        raise ValueError("The three-request diagnostic must be preregistered")
    actions = {a["action_id"]: a for a in manifest["actions"]}
    if len(actions) != len(manifest["actions"]) or action_id not in actions:
        raise ValueError("A unique preregistered action is required")
    action = actions[action_id]
    if (action["endpoint"] not in ENDPOINTS or set(action["params"]) != {"q", "format"}
            or action["params"]["format"] != "rss" or not isinstance(action["params"]["q"], str)
            or not action["params"]["q"].strip()):
        raise ValueError("Only declared public Bing RSS requests are accepted")
    ledger_path = root / "attempts.json"
    with FileLock(str(root / "diagnostic.lock")):
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"manifest_sha256": digest(manifest), "attempts": []}
        if ledger["manifest_sha256"] != digest(manifest):
            raise ValueError("The request plan changed after registration")
        if len(ledger["attempts"]) >= 3 or any(a["action_id"] == action_id for a in ledger["attempts"]):
            raise ValueError("A diagnostic action cannot be repeated or exceed the physical budget")
        intent = {"registered_at": datetime.now(timezone.utc).isoformat(), "action_id": action_id,
                  "method": "GET", "endpoint": action["endpoint"], "params": action["params"],
                  "route": "direct", "tls_verification": True, "allow_redirects": False,
                  "request_attempts": 1, "connect_timeout_seconds": 4, "read_timeout_seconds": 8,
                  "headers": {"User-Agent": "FootballMind/0.1 research"}, "manifest_sha256": digest(manifest)}
        folder = root / action_id
        atomic_write(folder / "intent.json", intent)
        ledger["attempts"].append({"action_id": action_id, "state": "RESERVED_BEFORE_NETWORK", "intent_sha256": digest(intent)})
        atomic_write(ledger_path, ledger)

    archive, observations = RawArchive(root / "evidence"), []
    session = session_factory("direct")

    def save_attempt(**facts):
        observations.append(archive.save(
            facts["source"], facts["endpoint"], facts["params"], facts["body"],
            status=facts["http_status"], headers=facts["response_headers"], request_id=facts["request_id"],
            observed_at=facts["observed_at"], observation_id=facts["observation_id"], error_class=facts["error_class"],
            context={"diagnostic_action_id": action_id, "intent_sha256": digest(intent),
                     "requested_at": facts["requested_at"].isoformat(), "redirect_location": session.location,
                     "failure_detail": session.failure, "route": "direct", "automatic_redirects": False,
                     "evidence_scope": "search_channel_diagnostic_not_agent_policy_trial"}))

    error = None
    with RequestTransport("search_channel_diagnostic", on_attempt=save_attempt, session=session,
                          max_attempts=1, request_budget=1, interval=0, timeout=(4, 8)) as transport:
        try:
            transport.get(action["endpoint"], action["params"], headers=intent["headers"])
        except TransportError as exc:
            error = exc.category
        stats = dict(transport.stats)
    if len(observations) != 1:
        raise RuntimeError("An uncertain attempt requires reconciliation, not automatic retry")
    obs = observations[0]
    body = archive.read_body(obs)
    tools = OpenWebTools(archive, manifest["allowed_result_hosts"])
    prepared = {"tool": "search_web", "endpoint": action["endpoint"], "params": action["params"], "source": obs.source}
    document = tools.project(obs, prepared)
    try:
        feed = ET.fromstring(body)
        root_tag = feed.tag
        all_links = [item.findtext("link") for item in feed.findall("./channel/item")]
        all_links = [link for link in all_links if isinstance(link, str)]
    except ET.ParseError:
        root_tag, all_links = None, []
    result = {"action_id": action_id, "observation": obs.model_dump(mode="json"), "error_category": error,
              "redirect_location": session.location, "failure_detail": session.failure,
              "transport_stats": stats, "body_bytes": len(body), "rss_root_tag": root_tag,
              "rss_item_count": len(all_links), "rss_result_hosts": sorted({urlsplit(u).hostname or "" for u in all_links}),
              "document_status": document["status"], "allowed_links": document["links"],
              "allowed_link_count": len(document["links"]), "semantic_claims_extracted": 0,
              "source_pages_fetched": 0, "agent_policy_trial": False}
    atomic_write(folder / "document.json", document)
    atomic_write(folder / "result.json", result)
    with FileLock(str(root / "diagnostic.lock")):
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        entry = next(a for a in ledger["attempts"] if a["action_id"] == action_id)
        entry.update(state="COMPLETED", observation_id=obs.id, http_status=obs.http_status,
                     document_status=document["status"], allowed_link_count=len(document["links"]),
                     result_sha256=digest(result))
        atomic_write(ledger_path, ledger)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--action", required=True)
    args = parser.parse_args()
    result = run_registered_probe(args.root, args.action)
    print(json.dumps({k: result[k] for k in ("action_id", "error_category", "redirect_location", "document_status", "rss_root_tag", "rss_item_count", "allowed_link_count", "allowed_links")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
