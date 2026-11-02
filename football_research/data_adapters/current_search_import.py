"""Import a captured external search-tool envelope into reconstructed menus.

No networking, no provider-rank inference, and no historical Agent claims.
RawArchive must already hold the unchanged tool return. A batched provider
response is one shared composite menu, not four invented per-query rankings.
"""
from copy import deepcopy
import json
import re

from .phase3_historical_panel import append_row, load_rows


def import_search_observation(root, observation, archive, fixture_ids):
    payload = json.loads(archive.read_body(observation))
    text = "\n".join(x["text"] for x in payload.get("content", []) if x.get("type") == "text")
    pattern = re.compile(r"(?m)^([^\n]*) \((https?://[^\s]+)\)\n【([^】]+)】([^\n]*)")
    matches = list(pattern.finditer(text))
    candidates = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        excerpt = text[match.end():end].split("\n-----", 1)[0].strip()
        candidates.append({"candidate_id": f"external:{observation.id}:{index + 1}",
            "url": match[2], "title": match[1], "snippet": excerpt or None,
            "rank": index + 1, "rank_basis": "composite_tool_return_order_not_per_query_provider_rank",
            "provider_reference": match[3], "provider_time_and_crawl_text": match[4],
            "action_type": "read_url", "selected": False, "selection_status": "not_requested"})
    existing = {row["discovery_id"] for row in load_rows(root, "discovery.jsonl")}
    for mid in fixture_ids:
        did = f"discovery:{observation.id}:{mid}"
        if did in existing:
            continue
        per_fixture = deepcopy(candidates)
        for candidate in per_fixture:
            candidate["shared_candidate_id"] = candidate["candidate_id"]
            candidate["candidate_id"] += ":" + mid
        append_row(root, "discovery.jsonl", {"discovery_id": did, "physical_match_id": mid,
            "observed_at": observation.observed_at.isoformat(), "query_or_parent": observation.params.get("query_batch"),
            "query": None, "query_batch": observation.params.get("query_batch"), "parent_url": None,
            "provider": "search_service.web_run", "visible_candidates": per_fixture,
            "raw_path": observation.payload_path, "raw_sha256": observation.content_hash,
            "status": "returned" if candidates else "empty", "original_agent_menu_proven": False,
            "information_mode": "historical_reconstruction", "menu_type": "reconstructed_menu",
            "historical_menu_observed_at": None, "available_at": None,
            "truncation_or_cap": "provider bounded composite response; per-query attribution unavailable",
            "shared_discovery_observation_id": observation.id,
            "policy_version": "historical-continuation-r2-v1",
            "observed_at_basis": observation.context.get("observed_at_basis")})
    aid = "external-tool:" + observation.id
    if not any(x["attempt_id"] == aid for x in load_rows(root, "attempts.jsonl")):
        append_row(root, "attempts.jsonl", {"attempt_id": aid, "physical_match_id": fixture_ids[0],
            "related_physical_match_ids": fixture_ids, "discovery_id": f"discovery:{observation.id}:{fixture_ids[0]}",
            "candidate_id": None, "action_id": "current_web_search", "tool": "current_web_search",
            "requested_at": observation.context.get("requested_at"), "started_at": observation.context.get("requested_at"),
            "request_clock_status": observation.context.get("request_clock_status"),
            "completed_at": observation.observed_at.isoformat(), "response_at": observation.observed_at.isoformat(),
            "response_observed_at": observation.observed_at.isoformat(), "status_code": None,
            "status": "returned" if candidates else "empty", "error_type": None, "error_code": None, "http_status": None,
            "returned_document_id": None, "latency_ms": None, "request_count": None,
            "tool_call_count": 1, "provider_internal_http_count": None, "token_cost": None,
            "monetary_cost": None, "monetary_cost_if_known": None, "token_usage_if_available": None,
            "returned_version_ids": [], "raw_path": observation.payload_path, "raw_sha256": observation.content_hash,
            "information_mode": "historical_reconstruction", "actual_historical_agent_action": False,
            "selected_by_policy": True, "policy_arm": "not_configured", "selection_basis": "manual_diagnostic_discovery",
            "selection_probability": None, "selection_probability_basis": "not_recorded",
            "cost_status": "provider_cost_unknown", "shared_observation_counted_once": True})
    return len(candidates)
