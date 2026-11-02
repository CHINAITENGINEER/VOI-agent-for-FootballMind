"""Reproducible local cup-context readiness inventory; no network by default."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from football_spider.models.contracts import Observation
from football_spider.normalization.sources import stamp
from football_spider.storage.raw_archive import RawArchive, json_bytes

CUP_PATTERN = re.compile(r"杯|亚运|亞運|Cup|Asian.?Games|Champions|Europa", re.I)
CONTEXT_KEYS = {"qualification", "rank_objective", "rank_objectives", "rotation", "lineup", "lineups",
                "tactical_intent", "formation", "starting_xi", "qualification_clinched"}


def _facts(path: Path, workspace: Path) -> dict:
    body = path.read_bytes()
    return {"path": path.relative_to(workspace).as_posix(), "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def _walk(value, path="$", nested=False):
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}", nested)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]", nested)


def audit(workspace: Path) -> dict:
    workspace = workspace.resolve()
    archive_root = workspace / "football_spider/platform_data"
    archive = RawArchive(archive_root)
    rows, cup_candidates, source_counts, matched_payloads = [], [], Counter(), set()
    for metadata in sorted((archive_root / "raw").rglob("*.observation.json")):
        observation = Observation.model_validate_json(metadata.read_bytes())
        source_counts[observation.source] += 1
        item = {**_facts(metadata, workspace), "kind": "raw_observation", "observation_id": observation.id,
                "source": observation.source, "observed_at": observation.observed_at.isoformat(),
                "http_status": observation.http_status, "content_sha256": observation.content_hash,
                "source_published_at": None, "source_publication_missing_reason": "not_a_verified_field_in_raw_observation",
                "context_key_candidates": [], "cup_row_candidates": 0}
        payload = (archive_root / observation.payload_path).resolve()
        matched_payloads.add(payload)
        try:
            body = archive.read_body(observation)
            item["raw_payload"] = _facts(payload, workspace)
            item["raw_content_hash_verified"] = True
            try:
                parsed = json.loads(body)
            except (ValueError, UnicodeError):
                parsed = None
            for path, record in _walk(parsed):
                keys = sorted(set(record) & CONTEXT_KEYS)
                if keys:
                    item["context_key_candidates"].append({"json_path": path, "keys": keys})
                league = record.get("leagueChs") or record.get("leaguechs") or record.get("league")
                match_id = record.get("matchId")
                if not isinstance(league, str) or match_id is None or not CUP_PATTERN.search(league):
                    continue
                kickoff = record.get("matchTime") or record.get("matchtime")
                try:
                    before = observation.observed_at < stamp(kickoff, "Asia/Shanghai") if kickoff else None
                except (TypeError, ValueError):
                    before = None
                candidate = {"observation_id": observation.id, "metadata_path": item["path"], "json_path": path,
                             "source_match_id": str(match_id), "competition_name": league, "source_kickoff": kickoff,
                             "home_source_id": record.get("homeId") or record.get("hometeamid"),
                             "away_source_id": record.get("awayId") or record.get("awayteamid"),
                             "home_name": record.get("homeChs") or record.get("home") or record.get("hometeamchs"),
                             "away_name": record.get("awayChs") or record.get("away") or record.get("awayteamchs"),
                             "observed_at": observation.observed_at.isoformat(), "observed_before_source_kickoff": before,
                             "path_is_historical_form_or_h2h": bool(re.search(r"battle|history|recent|future", path, re.I)),
                             "record_location": "nested_schedule" if "ScheduleList" in path else (
                                 "main_fixture_list" if re.fullmatch(r"\$\.data\[\d+\]\.list\[\d+\]", path) else "other_nested_context"),
                             "time_interpretation": "source_normalizer_Asia_Shanghai; no_verified_publication_time",
                             "eligibility": "raw_candidate_only_no_qualification_lineup_tactical_evidence_chain"}
                cup_candidates.append(candidate)
                item["cup_row_candidates"] += 1
        except (ValueError, OSError) as error:
            item.update(raw_content_hash_verified=False, integrity_error=type(error).__name__)
        rows.append(item)
    orphans = [_facts(path, workspace) for path in sorted((archive_root / "raw").rglob("*.body.gz"))
               if path.resolve() not in matched_payloads]
    collector_root = workspace / "var/parallel/collectors"
    exports = []
    export_matches = set()
    normalized_cup = 0
    for path in sorted(collector_root.rglob("matches.jsonl")):
        if "round7" in path.parts or "pg" in path.parts:
            continue
        matches = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        leagues = Counter(record.get("competition_name", "unknown") for record in matches)
        cup_count = sum(count for league, count in leagues.items() if CUP_PATTERN.search(league))
        normalized_cup += cup_count
        for record in matches:
            export_matches.add((record.get("source"), record.get("source_id") or record.get("match_id")))
        files = {}
        for name in ("manifest.json", "COMPLETE.json", "features.jsonl", "feature_exclusions.jsonl", "result_versions.jsonl"):
            candidate = path.parent / name
            if candidate.exists():
                entry = _facts(candidate, workspace)
                if candidate.suffix == ".jsonl":
                    entry["row_count"] = sum(bool(line.strip()) for line in candidate.read_bytes().splitlines())
                files[name] = entry
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes()) if manifest_path.exists() else {}
        exports.append({**_facts(path, workspace), "row_count": len(matches), "competition_counts": dict(leagues),
                        "cup_row_candidates": cup_count, "information_mode_counts": dict(Counter(r.get("information_mode", "unknown") for r in matches)),
                        "available_context_topics": sorted({key for record in matches for key in CONTEXT_KEYS if key in record}),
                        "source_observed_at": manifest.get("source_observed_at"), "scope": manifest.get("scope"),
                        "source_revision_bias": manifest.get("source_revision_bias"), "companion_files": files,
                        "readiness_reason": "no_registered_source_timed_cup_context_acquisition_trajectories"})
    legacy = []
    for path in sorted((workspace / "football_spider/data").glob("match_*.json")):
        document = json.loads(path.read_bytes())
        legacy.append({**_facts(path, workspace), "top_level_keys": sorted(document) if isinstance(document, dict) else [],
                       "eligibility": "legacy_snapshot_no_context_evidence_registration_or_source_timed_policy_trace"})
    return {"schema": "round7-local-cup-context-readiness-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": {"raw_archive": "football_spider/platform_data/raw", "prior_exports": "var/parallel/collectors/* excluding round7 and pg",
                      "legacy_snapshots": "football_spider/data/match_*.json", "credentials_and_cookies": "excluded",
                      "test_and_synthetic_agent_fixtures": "excluded_from_real_evidence_counts",
                      "new_round7_official_capture": "reported_separately_not_preexisting_evidence"},
            "summary": {"committed_raw_observations": len(rows), "source_counts": dict(source_counts),
                        "raw_hash_verified": sum(row.get("raw_content_hash_verified", False) for row in rows),
                        "uncommitted_raw_payloads": len(orphans), "export_match_files": len(exports),
                        "export_rows_including_versions": sum(row["row_count"] for row in exports),
                        "unique_source_match_keys": len(export_matches), "normalized_cup_row_candidates": normalized_cup,
                        "raw_cup_row_occurrences": len(cup_candidates),
                        "raw_cup_unique_source_match_ids": len({row["source_match_id"] for row in cup_candidates}),
                        "raw_cup_occurrences_observed_before_kickoff": sum(row["observed_before_source_kickoff"] is True for row in cup_candidates),
                        "raw_cup_prematch_unique_source_ids": len({row["source_match_id"] for row in cup_candidates
                                                                  if row["observed_before_source_kickoff"] is True}),
                        "context_key_candidate_occurrences": sum(len(row["context_key_candidates"]) for row in rows),
                        "verified_operational_cup_context_policy_trajectories": 0,
                        "policy_gain_research_ready": False},
            "method_limits": ["Cup-name/key scans identify candidates, not semantically verified facts.",
                              "Repeated archive/export versions are not independent matches or policy runs.",
                              "A historical match row inside a current H2H response is not a pre-match context snapshot.",
                              "No registered qualification/rank/rotation/lineup/tactical acquisition trajectory was located in the audited files."],
            "raw_observations": rows, "uncommitted_payloads": orphans, "raw_cup_candidates": cup_candidates,
            "exports": exports, "legacy_snapshots": legacy,
            "required_next_fields": ["canonical_fixture_and_team_ids", "kickoff_and_cutoff_with_timezone", "competition_stage_and_group",
                                     "source_timed_standings_and_tie_break_rules", "qualification_and_rank_objective_per_target",
                                     "source_timed_lineup_and_reference_lineup", "actual_observed_at_and_operational_available_at",
                                     "raw_content_hash_and_claim_anchor", "action_catalog_cost_receipt_and_acquisition_order",
                                     "visible_state_and_prediction_before_each_action", "later_separate_result_label_and_label_available_at"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(json_bytes(report))
    print(json.dumps(report["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()
