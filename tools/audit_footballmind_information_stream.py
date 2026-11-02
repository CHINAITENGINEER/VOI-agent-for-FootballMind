"""Audit evidence quality of a FootballMind information-stream package.

This audit keeps replayability, actual pre-kickoff observation, and prospective
online-OPE eligibility separate. Unknown provenance is never silently inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_time(row: dict[str, Any], *keys: str) -> datetime | None:
    for key in keys:
        value = _time(row.get(key))
        if value is not None:
            return value
    return None


def _before(value: datetime | None, cutoff: datetime | None) -> bool | None:
    if value is None or cutoff is None:
        return None
    return value <= cutoff


def _count_timing(rows: Iterable[dict[str, Any]], cutoff_by_id: dict[str, datetime | None], *keys: str) -> dict[str, int]:
    counts = {"before_kickoff": 0, "after_kickoff": 0, "unknown": 0}
    for row in rows:
        result = _before(_first_time(row, *keys), cutoff_by_id.get(row.get("physical_match_id")))
        if result is True:
            counts["before_kickoff"] += 1
        elif result is False:
            counts["after_kickoff"] += 1
        else:
            counts["unknown"] += 1
    return counts


def _all_before(timing: dict[str, int]) -> bool | None:
    if timing["after_kickoff"]:
        return False
    if timing["unknown"]:
        return None
    return timing["before_kickoff"] > 0


def _none_after(timing: dict[str, int]) -> bool | None:
    if timing["after_kickoff"]:
        return False
    if timing["unknown"]:
        return None
    return True


def _raw_digest_audit(package: Path, rows: Iterable[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify supplied raw hashes without following source-machine paths."""
    checks: list[tuple[str, str]] = []
    file_hashes = manifest.get("file_hashes") if isinstance(manifest, dict) else None
    if isinstance(file_hashes, dict):
        checks.extend((str(path), str(expected)) for path, expected in file_hashes.items())
    for row in rows:
        raw_path, expected = row.get("raw_path"), row.get("raw_sha256")
        if isinstance(raw_path, str) and isinstance(expected, str) and (raw_path, expected) not in checks:
            checks.append((raw_path, expected))
    missing: list[str] = []
    mismatched: list[str] = []
    unverifiable: list[str] = []
    checked = 0
    for raw_path, expected in checks:
        path = Path(raw_path)
        if path.is_absolute():
            unverifiable.append(raw_path)
            continue
        try:
            resolved = (package / path).resolve()
            resolved.relative_to(package.resolve())
        except (OSError, ValueError):
            unverifiable.append(raw_path)
            continue
        if not resolved.exists():
            missing.append(raw_path)
            continue
        checked += 1
        if _sha(resolved) != expected:
            mismatched.append(raw_path)
    if mismatched:
        status = "failed"
    elif missing or unverifiable:
        status = "unknown"
    elif checks:
        status = "verified"
    else:
        status = "unknown"
    return {
        "status": status,
        "manifest_entries": len(file_hashes) if isinstance(file_hashes, dict) else None,
        "checks_declared": len(checks),
        "checks_completed": checked,
        "missing": missing,
        "mismatched": mismatched,
        "unverifiable": unverifiable,
        "raw_package_unchanged_by_adapter": None,
        "unchanged_claim_basis": "A post-adapter audit cannot prove that no file changed without a pre-adapter snapshot; it can only verify current bytes against declared digests.",
    }


def _overlap(documents: list[dict[str, Any]], fixtures: list[dict[str, Any]], prior_cases: Any) -> dict[str, Any]:
    if not isinstance(prior_cases, list):
        return {"status": "unknown", "reason": "prior_cases_unavailable"}
    old_fixtures = {case.get("fixture", {}).get("physical_match_id") for case in prior_cases if isinstance(case, dict)}
    old_fixtures.discard(None)
    old_docs = [doc for case in prior_cases if isinstance(case, dict) for doc in (case.get("documents") or [])]
    old_urls = {doc.get("url") for doc in old_docs if doc.get("url")}
    old_hashes = {doc.get("raw_sha256") for doc in old_docs if doc.get("raw_sha256")}
    fixture_ids = {row.get("physical_match_id") for row in fixtures if row.get("physical_match_id")}
    urls = {doc.get("url") for doc in documents if doc.get("url")}
    hashes = {doc.get("raw_sha256") for doc in documents if doc.get("raw_sha256")}
    return {
        "status": "measured",
        "prior_fixture_count": len(old_fixtures),
        "package_fixture_count": len(fixture_ids),
        "new_fixture_ids": sorted(fixture_ids - old_fixtures),
        "independent_fixture_increase": len(fixture_ids - old_fixtures),
        "new_document_urls_vs_prior_corpus": len(urls - old_urls),
        "new_document_hashes_vs_prior_corpus": len(hashes - old_hashes),
        "overlapping_document_hashes": len(hashes & old_hashes),
        "interpretation": "Overlap is computed from physical_match_id, document URL and raw_sha256; menus and HTTP receipts are not counted as new fixtures.",
    }


def _model_summary(path: Path) -> dict[str, Any]:
    report = _json(path, {}) or {}
    return {
        "status": report.get("status"),
        "b_gain_signs": report.get("B_gain_signs"),
        "c_mean_actual_gain": report.get("C_mean_actual_gain"),
        "f_selected_regularization": report.get("f_selection", {}).get("selected_regularization"),
        "g_selected_regularization": report.get("g_selection", {}).get("selected_regularization"),
        "g_constant_baseline_won": report.get("g_selection", {}).get("constant_baseline_won"),
    }


def _derived_cases_path(derived: Path) -> Path:
    """Accept either a cases directory or the adapter's JSON output file."""
    if derived.is_file():
        return derived
    return derived / "cases.json"


def audit(package: Path, derived: Path, fixed_report: Path, selected_report: Path, prior_cases: Path) -> dict[str, Any]:
    fixtures = _rows(package / "fixtures.jsonl")
    documents = _rows(package / "documents.jsonl")
    discovery = _rows(package / "discovery.jsonl")
    attempts = _rows(package / "attempts.jsonl")
    snapshots = _rows(package / "context_snapshots.jsonl")
    results = _rows(package / "labels" / "results.jsonl")
    manifest = _json(package / "manifest.json", {}) or {}
    cases_path = _derived_cases_path(derived)
    cases_raw = _json(cases_path, None)
    cases = cases_raw if isinstance(cases_raw, list) else []
    prior = _json(prior_cases, None)
    kickoff = {row.get("physical_match_id"): _time(row.get("kickoff")) for row in fixtures}
    doc_modes = Counter(row.get("information_mode") or "unknown" for row in documents)
    menu_modes = Counter(row.get("menu_status") or row.get("information_mode") or "unknown" for row in discovery)
    attempts_status = Counter(row.get("status") or "unknown" for row in attempts)
    # ``captured_at`` is a transport timestamp, not proof that the body was
    # visible to the Agent.  A missing ``available_at`` therefore remains
    # unknown even when a later capture happened before kickoff by accident.
    doc_timing = _count_timing(documents, kickoff, "available_at")
    # This is only a source-publication proxy.  Do not promote a field named
    # ``source_availability_text`` into an actual visibility timestamp.
    publication_timing = _count_timing(documents, kickoff, "source_published_at", "published_at")
    discovery_timing = _count_timing(discovery, kickoff, "observed_at", "discovered_at", "captured_at")
    attempt_timing = _count_timing(attempts, kickoff, "requested_at", "response_at")
    snapshot_timing = _count_timing(snapshots, kickoff, "snapshot_observed_at", "observed_at")
    result_timing = _count_timing(results, kickoff, "result_observed_at", "source_observed_at")
    snapshot_proven = sum(bool(row.get("historical_snapshot_proven")) for row in documents + discovery + snapshots)
    menu_original_proven = sum(bool(row.get("original_agent_menu_proven")) for row in discovery)
    raw_audit = _raw_digest_audit(package, documents + discovery + attempts + results, manifest)
    overlap = _overlap(documents, fixtures, prior)
    all_case_results_bound = bool(cases) and len(cases) == len(fixtures)

    prospective_requirements = {
        "document_visibility_before_kickoff": _all_before(doc_timing),
        "discovery_observed_before_kickoff": _all_before(discovery_timing),
        "attempts_requested_before_kickoff": _all_before(attempt_timing),
        "context_observed_before_kickoff": _all_before(snapshot_timing),
        "original_agent_menu_proven": (None if not discovery else menu_original_proven == len(discovery)),
        "no_postkickoff_documents_or_menus": (None if doc_timing["unknown"] or discovery_timing["unknown"] else _none_after(doc_timing) and _none_after(discovery_timing)),
        "raw_digests_verified": raw_audit["status"] == "verified",
        "results_observed_after_kickoff": (None if not results or result_timing["unknown"] else result_timing["after_kickoff"] == len(results)),
    }
    eligible_online_ope = all(prospective_requirements.values())
    unknown_requirements = [name for name, value in prospective_requirements.items() if value is None]
    failed_requirements = [name for name, value in prospective_requirements.items() if value is False]
    clubs = {row.get("home_team") for row in fixtures} | {row.get("away_team") for row in fixtures}
    return {
        "schema": "footballmind-information-stream-audit-v2",
        "package": str(package.resolve()),
        "derived_cases": str(cases_path.resolve()),
        "counts": {"fixtures": len(fixtures), "documents": len(documents), "discovery_menus": len(discovery), "attempts": len(attempts), "context_snapshots": len(snapshots), "results": len(results), "derived_cases": len(cases), "clubs": len(clubs)},
        "integrity": {"attempt_status_counts": dict(attempts_status), "document_information_modes": dict(doc_modes), "menu_information_modes": dict(menu_modes), "all_case_results_bound": all_case_results_bound, "raw_digest_verification": raw_audit},
        "provenance_limits": {
            "declared_information_modes": dict(doc_modes), "publication_timestamp_timing": publication_timing,
            "actual_document_visibility_timing": doc_timing, "discovery_actual_timing": discovery_timing,
            "attempt_actual_timing": attempt_timing, "context_snapshot_timing": snapshot_timing,
            "result_observation_timing": result_timing, "historical_snapshot_proven_count": snapshot_proven,
            "original_agent_menu_proven_count": menu_original_proven, "postkickoff_discovery_rows": discovery_timing["after_kickoff"],
            "document_body_version_proven_at_original_time": all(bool(row.get("historical_snapshot_proven")) for row in documents) if documents else None,
            "menus_are_original_agent_menus": menu_original_proven == len(discovery) if discovery else None,
            "selected_by_collector_is_policy_trace": False if discovery else None,
        },
        "corpus_overlap": overlap,
        "fixed_f_g": _model_summary(fixed_report), "selected_f_g": _model_summary(selected_report),
        "eligibility": {
            "historical_replay": "eligible" if all_case_results_bound and raw_audit["status"] == "verified" else "unknown",
            # A known post-kickoff/collector-only failure is decisive even if
            # another field is unknown; unknown is reserved for otherwise
            # clean evidence whose missing provenance prevents a conclusion.
            "online_ope": "eligible" if eligible_online_ope else ("ineligible" if failed_requirements else "unknown"),
            "online_ope_requirements": prospective_requirements, "failed_requirements": failed_requirements, "unknown_requirements": unknown_requirements,
        },
        "decision": {
            "use_empirical_g_in_runtime": False,
            "reason": "Empirical g may be used only after conditional action-choice evidence is identified from a prospective or otherwise proven pre-kickoff trajectory; reconstructed menus and publication dates do not establish that evidence.",
            "valid_use": "replayable offline reveal, f-response probe, provenance, digest and leakage audit",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("derived", type=Path)
    parser.add_argument("--fixed-report", type=Path, required=True)
    parser.add_argument("--selected-report", type=Path, required=True)
    parser.add_argument("--prior-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.package, args.derived, args.fixed_report, args.selected_report, args.prior_cases)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
