"""Report projection for canonical streams; reuse the existing validator.

Called only while constructing a new export. Raw responses are never written
here. Hash checks use the stream validator and the exporter's file digest.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from football_platform.information.stream_validator import _safe_raw, validate_stream


REUSED_COMPONENTS = [
    "existing spftll collector", "RequestTransport", "RawArchive",
    "AcquisitionDecisionJournal", "RefreshService", "LiveEvidenceEnvironment",
    "LearnedVOIRuntime", "NewsDiscoveryTools", "RoutedOpenWebTools",
    "existing stream validator",
]
CONTEXT_FIELDS = ("ranking", "points", "qualification_status", "schedule",
                  "injuries", "lineup_status", "odds", "weather")


def finalize_stream_package(target: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Finish a newly created package; do not call on a published old export."""
    from .prospective_stream import DATA_FILES, _sha256, _time

    target = Path(target).resolve()
    rows = {name: [json.loads(line) for line in (target / name).read_text(encoding="utf-8").splitlines()
                   if line.strip()] for name in DATA_FILES}
    fixtures = {row["physical_match_id"]: row for row in rows["fixtures.jsonl"]}
    kickoff = {mid: _time(row.get("kickoff_at") or row.get("kickoff")) for mid, row in fixtures.items()}
    def before(row, field):
        stamp = _time(row.get(field)); end = kickoff.get(row.get("physical_match_id"))
        return stamp is not None and end is not None and stamp < end
    selected = [row for row in rows["attempts.jsonl"] if row.get("selected_by_policy") is True]
    documents = rows["documents.jsonl"]
    discovery_documents = [row for row in documents if row.get("document_kind") == "discovery"]
    nonempty = [row for row in documents if (row.get("body_text") or row.get("body") or "").strip()]
    payload_digests = Counter(row.get("raw_sha256") for row in nonempty if row.get("raw_sha256"))
    references = {}
    for filename, values in rows.items():
        for number, row in enumerate(values, 1):
            if row.get("raw_path") is not None:
                key = (row.get("raw_path"), row.get("raw_sha256"))
                references.setdefault(key, []).append(f"{filename}:{number}")
    raw_checks = []
    for (path, digest), locations in references.items():
        issues = []
        _safe_raw(target, path, digest, locations[0] + ".raw_path", issues)
        raw_checks.append({"raw_path": path, "raw_sha256": digest,
                           "verified": not issues, "references": locations,
                           "issues": [item.as_dict() for item in issues]})
    raw_reference_coverage = {}
    for filename, values in rows.items():
        if filename == "labels/results.jsonl":
            category = "outcome"
        elif filename == "fixtures.jsonl":
            category = "fixture"
        elif filename == "discovery.jsonl":
            category = "menu"
        elif filename == "attempts.jsonl":
            category = "action"
        elif filename == "context_snapshots.jsonl":
            category = "context"
        else:
            continue
        referenced = sum(bool(row.get("raw_path") and row.get("raw_sha256")) for row in values)
        raw_reference_coverage[category] = {"rows": len(values), "with_raw_reference": referenced,
                                            "raw_reference_missing": len(values) - referenced,
                                            "note": "context raw is optional when no separate response exists" if category == "context" else None}
    statuses = Counter(row.get("status", "unknown") for row in selected)
    missing = {}
    for filename, values in rows.items():
        keys = sorted({key for row in values for key in row})
        missing[filename] = {key: sum(row.get(key) is None for row in values)
                             for key in keys if any(row.get(key) is None for row in values)}
    derived_path = target / "derived/g-training-candidates.jsonl"
    derived = [json.loads(line) for line in derived_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    timelines = []
    for mid, fixture in fixtures.items():
        contexts = [row for row in rows["context_snapshots.jsonl"] if row.get("physical_match_id") == mid]
        menus = [row for row in rows["discovery.jsonl"] if row.get("physical_match_id") == mid]
        actions = [row for row in selected if row.get("physical_match_id") == mid]
        reasons = []
        if fixture.get("information_mode") != "prospective_capture": reasons.append("historical_reconstruction_not_online_trajectory")
        if not menus or not all(row.get("original_agent_menu_proven") is True for row in menus): reasons.append("original_agent_menu_not_proven")
        if not actions: reasons.append("no_selected_action")
        if not contexts or not all(all(before(row, field) for field in ("captured_at", "available_at", "prediction_cutoff")) for row in contexts): reasons.append("context_availability_or_clock_unproven")
        if not any(row.get("snapshot_kind") == "final" and row.get("stop_reason") for row in contexts): reasons.append("stop_reason_unknown")
        if not any(row.get("physical_match_id") == mid for row in rows["labels/results.jsonl"]): reasons.append("independent_outcome_missing")
        timelines.append({"physical_match_id": mid, "home_team": fixture.get("home_team"),
                          "away_team": fixture.get("away_team"), "timeline_complete": not reasons,
                          "incomplete_reasons": reasons, "menus": len(menus), "selected_actions": len(actions),
                          "contexts": len(contexts)})
    enriched = dict(report)
    enriched.update({
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "independent_matches": len(fixtures),
        "teams_covered": len({row.get(key) for row in fixtures.values() for key in ("home_team", "away_team") if row.get(key)}),
        "home_teams_covered": len({row.get("home_team") for row in fixtures.values()}),
        "away_teams_covered": len({row.get("away_team") for row in fixtures.values()}),
        "document_records": len(documents), "nonempty_bodies": len(nonempty),
        "duplicate_body_ratio": ((len(nonempty) - len(payload_digests)) / len(nonempty)) if nonempty else None,
        "duplicate_basis": "exact_nonempty_raw_payload_digest_only; semantic duplication not assessed",
        "time_precision": dict(Counter(row.get("time_precision", "unknown") for row in documents)),
        "information_mode_counts": {mode: sum(row.get("information_mode") == mode for row in documents)
            for mode in ("prospective_capture", "archived_snapshot", "historical_reconstruction")},
        "selected_action_statuses": dict(statuses),
        "failed_results": sum(statuses.get(status, 0) for status in ("failed", "timeout", "http_error", "parse_error", "blocked", "cancelled")),
        "empty_results": statuses.get("empty", 0),
        "discovery_document_statuses": dict(Counter(row.get("status", "not_recorded") for row in discovery_documents)),
        "retrieval_no_data_boundary": "NO_DATA/empty describes the saved post-filter projection; it does not prove the upstream provider returned zero items. Raw provider items are not reconstructed as Agent-visible candidates during export.",
        "documents_with_links_truncated": sum(row.get("links_truncated") is True for row in documents),
        "documents_with_link_truncation_unknown": sum(row.get("links_truncated") is None for row in documents),
        "visible_document_link_observations": sum(len(row["visible_document_links"])
            for row in rows["discovery.jsonl"] if isinstance(row.get("visible_document_links"), list)),
        "document_link_observation_basis": "Repeated pre-decision document-link snapshots, not unique provider results or additional actions. visible_candidates remains the separate actual action menu.",
        "attempts_before_kickoff": sum(before(row, "decision_at") for row in rows["attempts.jsonl"]),
        "completed_attempts_with_both_clocks_before_kickoff": sum(
            before(row, "requested_at") and before(row, "response_observed_at")
            for row in selected
        ),
        "unselected_decisions_before_kickoff": sum(before(row, "decision_at") for row in rows["attempts.jsonl"] if row.get("selected_by_policy") is False),
        "contexts_before_kickoff": sum(before(row, "observed_at") for row in rows["context_snapshots.jsonl"]),
        "contexts_with_proven_availability_before_kickoff": sum(
            all(before(row, field) for field in ("captured_at", "available_at", "prediction_cutoff"))
            for row in rows["context_snapshots.jsonl"]
        ),
        "observed_contexts_before_kickoff": sum(before(row, "observed_at") for row in rows["context_snapshots.jsonl"]),
        "original_agent_menus_proven": sum(row.get("original_agent_menu_proven") is True for row in rows["discovery.jsonl"]),
        "raw_digest_checks": f"{sum(row['verified'] for row in raw_checks)}/{len(raw_checks)}",
        "raw_reference_coverage": raw_reference_coverage,
        "g_training_eligible_rows": sum(row.get("label_status") == "eligible" for row in derived),
        "g_training_ineligible_rows": sum(row.get("label_status") == "ineligible" for row in derived),
        "g_training_count_basis": "derived/g-training-candidates.jsonl.label_status",
        "empirical_g": "deferred", "missing_fields": missing,
        "context_field_observation_missing": {key: sum(not isinstance(row.get(key), dict) or row[key].get("observed_at") is None
            for row in rows["context_snapshots.jsonl"]) for key in CONTEXT_FIELDS},
        "per_match_timeline": timelines,
        "reused_components": REUSED_COMPONENTS,
        "new_components": ["missing fixture/outcome/report adapters", "canonical export fields", "compatibility and integrity tests"],
        "live_catalog_boundary": "fixture_history and odds only; no claim of structured injuries, lineups, weather or tactics coverage",
    })
    reports = target / "reports"
    reports.mkdir(exist_ok=True)
    def save(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    save(reports / "prospective-capture-report.json", enriched)
    save(reports / "audit-report.json", {"schema": "footballmind-prospective-stream-audit-v2",
        "audited_at": enriched["report_generated_at"], "raw_checks": raw_checks,
        "inventory_provenance": enriched.get("inventory_provenance"),
        "missing_fields": missing, "per_match_timeline": timelines,
        "empirical_g": "deferred", "hash_scope": "all package files except manifest.json and its external checksum sidecar"})
    current_manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    quality = ["Prospective Information Stream Status", "", "schema_version: " + str(current_manifest.get("schema_version", current_manifest.get("schema")))]
    quality.extend(f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}" for key, value in enriched.items())
    quality.extend(["", "available prospective fixtures: " + str(enriched.get("available_prospective_fixtures", 0)),
                    "shortage reason: " + str(enriched.get("shortage_reason")),
                    "All unknown timestamps, probabilities and costs remain null.",
                    "Historical rows are excluded from online empirical_g; an empty package is not a completed trajectory."])
    (target / "quality-report.txt").write_text("\n".join(quality) + "\n", encoding="utf-8")
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    modes = {row.get("information_mode") for values in rows.values() for row in values
             if row.get("information_mode")}
    manifest["information_mode"] = (next(iter(modes)) if len(modes) == 1
                                     else "mixed" if modes else manifest.get("information_mode", "unknown"))
    manifest.update(report=enriched, empirical_g="deferred",
        hash_scope="every package file except manifest.json and manifest.sha256; manifest.json digest is in manifest.sha256")
    if enriched.get("inventory_provenance") is not None:
        manifest["external_provenance"] = {"fixture_inventory": enriched["inventory_provenance"]}
    def index_files():
        files = {}
        for path in sorted(target.rglob("*")):
            if not path.is_file() or path.name in {"manifest.json", "manifest.sha256"}: continue
            relative = path.relative_to(target).as_posix()
            entry = {"path": relative, "sha256": _sha256(path), "byte_size": path.stat().st_size}
            if path.suffix == ".jsonl": entry["row_count"] = sum(bool(line.strip()) for line in path.read_bytes().splitlines())
            files[relative] = entry
        manifest["files"] = files
        save(manifest_path, manifest)
    index_files()
    validation = validate_stream(target)
    save(reports / "validation-report.json", validation)
    index_files()
    final = validate_stream(target)
    if final != validation:
        raise ValueError("Final manifest validation changed after sealing report files")
    (target / "manifest.sha256").write_text(_sha256(manifest_path) + "  manifest.json\n", encoding="ascii")
    return enriched
