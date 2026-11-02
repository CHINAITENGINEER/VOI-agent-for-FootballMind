"""Build a read-only canonical identity audit for the second review set.

This script does not write back to Argilla, modify model data, call a provider,
or create f/g labels.  It joins the already materialized second-review records
with local Football-Data schedule rows and explicitly recorded external
identity corroboration.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(r"D:/a1-足彩")
INPUT = ROOT / "var/repairs/preannotation-v3-second-review-20261024-v1/second-review.jsonl"
OUT = ROOT / "var/repairs/preannotation-v3-canonical-candidate-audit-20261024-v1"
SCHEDULE_ROOT = ROOT / "var/external/football-data-co-uk-20261024-v2/raw/england"

TARGETS = {
    "ur1-009e564cb8bdc7d8d14778df": {
        "identity_kind": "local_schedule_exact",
        "canonical_match_id": "fd:E0:2425:2025-02-19:Aston Villa:Liverpool",
        "season_file": "2425-E0.csv",
        "date": "19/02/2025",
        "home": "Aston Villa",
        "away": "Liverpool",
        "external_sources": [],
    },
    "ur1-278d83c5ab06bc9e067ff65d": {
        "identity_kind": "local_schedule_exact",
        "canonical_match_id": "fd:E0:2425:2024-12-22:Tottenham:Liverpool",
        "season_file": "2425-E0.csv",
        "date": "22/12/2024",
        "home": "Tottenham",
        "away": "Liverpool",
        "external_sources": [],
    },
    "ur1-30ed95e68efde468b44075ea": {
        "identity_kind": "runtime_and_official_exact",
        "canonical_match_id": "spftll:match:3003917",
        "canonical_name_id": "footballmind:premier-league:2026-27:2026-10-18:nottingham-forest:arsenal",
        "provider_match_ids": {
            "espn": "espn:match:401878769",
            "premier_league": "2645263",
        },
        "kickoff": "2026-10-18T15:30:00+00:00",
        "home": "Nottingham Forest",
        "away": "Arsenal",
        "external_sources": [
            "https://www.premierleague.com/en/news/4675097/all-380-fixtures-for-season-2026-27",
            "https://www.premierleague.com/en/match/2645263/nottingham-forest-vs-arsenal/overview",
        ],
    },
    "ur1-4cd71f9f33133a22c7448145": {
        "identity_kind": "official_and_provider_exact_project_alias_pending",
        "canonical_name_id": "footballmind:premier-league:2026-27:2026-10-18:brighton:crystal-palace",
        "provider_match_ids": {
            "espn": "espn:match:401879261",
            "premier_league": "2645257",
        },
        "kickoff": "2026-10-18T13:00:00+00:00",
        "home": "Brighton & Hove Albion",
        "away": "Crystal Palace",
        "external_sources": [
            "https://www.premierleague.com/en/news/4675097/all-380-fixtures-for-season-2026-27",
            "https://www.premierleague.com/es/match/2645257/brighton-and-hove-albion-vs-crystal-palace/overview",
        ],
    },
    "ur1-efe647982f41ba4a345be862": {
        "identity_kind": "official_and_provider_exact_project_alias_pending",
        "canonical_name_id": "footballmind:premier-league:2026-27:2026-10-18:brighton:crystal-palace",
        "provider_match_ids": {
            "espn": "espn:match:401879261",
            "premier_league": "2645257",
        },
        "kickoff": "2026-10-18T13:00:00+00:00",
        "home": "Brighton & Hove Albion",
        "away": "Crystal Palace",
        "external_sources": [
            "https://www.premierleague.com/en/news/4675097/all-380-fixtures-for-season-2026-27",
            "https://www.premierleague.com/es/match/2645257/brighton-and-hove-albion-vs-crystal-palace/overview",
        ],
    },
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["record_id"])] = row
    return rows


def local_schedule_evidence(spec: dict[str, Any]) -> dict[str, Any]:
    filename = spec.get("season_file")
    if not filename:
        return {"status": "not_applicable"}
    path = SCHEDULE_ROOT / filename
    matches: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="", errors="replace") as handle:
            for row in csv.reader(handle):
                if len(row) >= 8 and row[1] == spec["date"] and row[3] == spec["home"] and row[4] == spec["away"]:
                    matches.append({
                        "competition": row[0], "date": row[1], "time": row[2],
                        "home": row[3], "away": row[4], "home_goals": row[5],
                        "away_goals": row[6], "result": row[7],
                    })
    return {
        "status": "exact_one_row" if len(matches) == 1 else ("no_row" if not matches else "multiple_rows"),
        "path": str(path),
        "matches": matches,
    }


def classify(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    times = row.get("times") or {}
    body = str(row.get("body") or "")
    body_truncated_possible = len(body) >= 1600
    published = times.get("source_published_at")
    schedule = local_schedule_evidence(spec)
    kickoff = spec.get("kickoff")
    if not kickoff and schedule.get("status") == "exact_one_row":
        item = schedule["matches"][0]
        try:
            local = datetime.strptime(f"{item['date']} {item['time']}", "%d/%m/%Y %H:%M").replace(tzinfo=ZoneInfo("Europe/London"))
            kickoff = local.astimezone(timezone.utc).isoformat()
        except (KeyError, TypeError, ValueError):
            kickoff = None
    if not kickoff:
        # For local historical rows, derive UTC kickoff from source schedule only as a
        # date/time check is outside the candidate package. Keep this explicitly unknown.
        published_before_kickoff = "unknown_without_timezone_binding"
    else:
        published_before_kickoff = None
        try:
            published_before_kickoff = datetime.fromisoformat(published.replace("Z", "+00:00")) < datetime.fromisoformat(kickoff)
        except (TypeError, ValueError):
            published_before_kickoff = "invalid_or_unknown"
    if spec["identity_kind"] == "local_schedule_exact":
        identity_status = "resolved_local_schedule_exact"
    elif spec["identity_kind"] == "runtime_and_official_exact":
        identity_status = "resolved_project_runtime_plus_official_and_provider"
    else:
        identity_status = "resolved_official_and_provider_project_alias_pending"
    return {
        "record_id": row["record_id"],
        "article_title": row.get("article_title"),
        "source_url": row.get("source_url"),
        "source_domain": row.get("source_domain"),
        "body_present": bool(body),
        "body_chars": len(body),
        "body_sha256": sha256_text(body) if body else None,
        "body_completeness": "suspected_1600_char_cap" if body_truncated_possible else ("present_untruncated_signal" if body else "missing"),
        "source_published_at": published,
        "source_updated_at": times.get("source_updated_at"),
        "system_available_at": times.get("system_available_at"),
        "capture_at": times.get("capture_at"),
        "canonical_match_id": spec.get("canonical_match_id"),
        "canonical_name_id": spec.get("canonical_name_id"),
        "provider_match_ids": spec.get("provider_match_ids", {}),
        "home_team": spec.get("home"),
        "away_team": spec.get("away"),
        "kickoff": kickoff,
        "identity_status": identity_status,
        "local_schedule_evidence": schedule,
        "external_sources": spec.get("external_sources", []),
        "published_before_kickoff": published_before_kickoff,
        "source_available_before_kickoff": "unknown" if not times.get("system_available_at") else "requires_clock_comparison",
        "same_match_cluster": spec.get("canonical_name_id") or spec.get("canonical_match_id"),
        "second_review_decision": (row.get("second_review") or {}).get("decision"),
        "second_review_rationale": (row.get("second_review") or {}).get("rationale"),
        "training_authority": "research_review_only",
        "f_candidate_status": "needs_body_recovery_before_f" if body_truncated_possible else "candidate_pending_f_run_and_source_assertion",
        "prospective_eligible": False,
        "historical_research_eligible": bool(body) and not body_truncated_possible and published_before_kickoff is True,
        "g_label_eligible": False,
        "exclusion_or_next_step": ("Recover the complete article body before f; current text is capped at 1600 characters. "
                                    if body_truncated_possible else "") +
                                   "Need fixed-f before/after plus independent result; system_available_at is UNKNOWN, so do not call this prospective evidence or empirical g label.",
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    records = [classify(rows[rid], TARGETS[rid]) for rid in TARGETS]
    clusters: dict[str, list[str]] = {}
    for record in records:
        clusters.setdefault(record["same_match_cluster"], []).append(record["record_id"])
    summary = {
        "schema": "preannotation-v3-canonical-candidate-audit-v1",
        "status": "research_audit_only",
        "input": str(INPUT),
        "reviewed_records": len(records),
        "unique_identity_clusters": len(clusters),
        "body_records": sum(bool(r["body_present"]) for r in records),
        "published_before_kickoff": sum(r["published_before_kickoff"] is True for r in records),
        "source_available_before_kickoff_known": sum(r["source_available_before_kickoff"] != "unknown" for r in records),
        "prospective_eligible": sum(bool(r["prospective_eligible"]) for r in records),
        "historical_research_eligible": sum(bool(r["historical_research_eligible"]) for r in records),
        "body_truncation_suspected": sum(r["body_completeness"] == "suspected_1600_char_cap" for r in records),
        "g_label_eligible": sum(bool(r["g_label_eligible"]) for r in records),
        "clusters": {k: sorted(v) for k, v in sorted(clusters.items())},
        "important_boundary": "Published-before is not the same as system-available-before; no empirical f/g row is created.",
    }
    write_json(OUT / "canonical-candidate-audit.json", {"summary": summary, "records": records})
    with (OUT / "canonical-candidate-table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["record_id", "article_title", "canonical_match_id", "canonical_name_id", "identity_status", "body_chars", "source_published_at", "system_available_at", "published_before_kickoff", "prospective_eligible", "historical_research_eligible", "g_label_eligible", "same_match_cluster"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})
    write_json(OUT / "external-verification.json", {
        "schema": "preannotation-v3-external-identity-verification-v1",
        "status": "identity_and_schedule_only",
        "sources": sorted({url for record in records for url in record["external_sources"]}),
        "facts_checked": ["fixture order", "date", "kickoff clock", "official/provider match reference", "final result where shown"],
        "not_used_for": ["f claims", "prediction_after", "realized_gain", "training labels", "source_available_at"],
        "note": "External pages corroborate identity and schedule. The article body and model labels remain governed by the local archived record.",
    })
    write_json(OUT / "body-recovery-queue.json", {
        "schema": "preannotation-v3-body-recovery-queue-v1",
        "status": "research_queue_only",
        "records": [
            {
                "record_id": record["record_id"],
                "source_url": record["source_url"],
                "canonical_name_id": record["canonical_name_id"],
                "reason": "archived candidate body is exactly 1600 characters and ends mid-sentence",
                "required_action": "retrieve the same URL once and preserve raw/body SHA plus current retrieval clock",
                "f_admission": "blocked_until_complete_body",
                "prospective_status": "unknown",
            }
            for record in records if record["body_completeness"] == "suspected_1600_char_cap"
        ],
        "rules": [
            "Do not infer missing paragraphs from title, URL, search snippets, or current match result.",
            "A current retrieval can support historical reconstruction but cannot establish source_available_at before the old cutoff.",
        ],
    })
    write_json(OUT / "network-probe.json", {
        "schema": "preannotation-v3-network-probe-ledger-v1",
        "status": "probe_only_no_archive_no_model_call",
        "probe_run_at_utc": "2026-10-24T10:07:07Z",
        "attempts": [
            {
                "source_url": "https://www.cpfc.co.uk/news/first-team/sage-sarr-wharton-brighton-preview-palace/",
                "method": "GET",
                "observed_status": "tls_failed",
                "response_saved": False,
                "body_used": False,
            },
            {
                "source_url": "https://www.cpfc.co.uk/news/match-previews/the-opposite-view-hurzeler-on-sage-palace-rivalry/",
                "method": "GET",
                "observed_status": 200,
                "response_bytes": 169063,
                "response_saved": False,
                "body_used": False,
            },
        ],
        "note": "These probes only test whether current retrieval is possible. They do not change the archived record or create f/g labels.",
    })
    readme = f"""# Preannotation v3 canonical candidate audit\n\n- Schema: `preannotation-v3-canonical-candidate-audit-v1`\n- Status: research audit only; no Argilla writeback, no model call, no training.\n- Input: `{INPUT}`\n- Candidate records: {summary['reviewed_records']}\n- Unique match clusters: {summary['unique_identity_clusters']}\n- Bodies present: {summary['body_records']}\n- Suspected 1600-character body caps: {summary['body_truncation_suspected']}\n- Published before kickoff: {summary['published_before_kickoff']}\n- `system_available_at` known before kickoff: {summary['source_available_before_kickoff_known']}\n- Prospective eligible: {summary['prospective_eligible']}\n- Historical research candidates with an untruncated body: {summary['historical_research_eligible']}\n- Empirical g label rows: {summary['g_label_eligible']}\n\nThe two Palace–Brighton articles are one physical match cluster and both are capped at 1600 characters mid-sentence, so they need body recovery before f. The Forest–Arsenal article resolves to the existing project runtime match. The two older Liverpool articles resolve against the local 2024/25 schedule, but their availability clocks remain unknown. Published-before is retained as evidence; it is not promoted to prospective availability.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    manifest_files = []
    for path in sorted(OUT.iterdir()):
        if path.name == "manifest.json":
            continue
        manifest_files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(OUT / "manifest.json", {"schema": "preannotation-v3-canonical-candidate-audit-manifest-v1", "files": manifest_files})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
