"""Audit the Round86 schema, independent holdout and parameter-freeze gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_research.evaluation.policy_experiment_protocol import (  # noqa: E402
    audit_episode_schema,
    audit_holdout_manifest,
    audit_parameter_freeze,
    audit_policy_experiment,
)


EPISODES = ROOT / "var" / "parallel" / "baselines" / "round9" / "residual-evsi-v1" / "episodes.jsonl"
REPORT = ROOT / "var" / "parallel" / "baselines" / "round9" / "residual-evsi-v1" / "report.json"
DESIGN = ROOT / "var" / "parallel" / "baselines" / "round9" / "residual-evsi-v1" / "design.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(str(value) for value in values if value is not None)


def build_audit() -> dict[str, Any]:
    episodes = load_jsonl(EPISODES)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    design = json.loads(DESIGN.read_text(encoding="utf-8"))

    schema = audit_episode_schema(episodes)
    episode_match_ids = sorted({str(row.get("match_id")) for row in episodes if row.get("match_id")})
    folds = report.get("folds") or []
    latest = folds[-1] if folds else {}
    split = latest.get("split") or {}
    residual = latest.get("residual_split") or {}
    partitions = {
        "train": _ids(latest.get("training_match_ids")) or _ids(residual.get("training_match_ids")),
        "validation": _ids(latest.get("validation_match_ids")),
        "test": _ids(latest.get("test_match_ids")),
    }
    holdout = audit_holdout_manifest(
        partitions,
        # Distinguish supervised fit exposure from prior policy exploration.
        prior_calibration_match_ids=sorted(set(partitions["train"]) | set(partitions["validation"])),
        prior_policy_search_match_ids=episode_match_ids,
        episode_match_ids=episode_match_ids,
        declared_independent=False,
    )
    test_ids = set(partitions["test"])
    test_rows = [row for row in episodes if row.get("fold") == latest.get("fold")
                 and row.get("match_id") in test_ids]
    first_cutoff = min((row["prediction_cutoff"] for row in test_rows), default=None)
    freeze = audit_parameter_freeze(design.get("policy_freeze") or {}, earliest_test_prediction_cutoff=first_cutoff)

    checks = {
        "raw_episode_rows_15080": len(episodes) == 15080,
        "schema_review_expected_for_old_archive": schema["status"] == "SCHEMA_REVIEW_REQUIRED",
        "required_metadata_gaps_are_reported": {
            "dataset_version", "competition_id", "league", "lead_hours",
            "fixture_available_at", "strategy_version", "candidate_set_version",
            "source_id", "source_version", "feature_manifest_hash",
            "policy_context_hash", "evaluator_sidecar_ref",
        }.issubset(set(schema["missing_field_counts"])),
        "latest_fold_has_three_partitions": all(partitions[name] for name in ("train", "validation", "test")),
        "holdout_review_expected_for_old_policy_archive": holdout["status"] == "HOLDOUT_REVIEW_REQUIRED",
        "policy_archive_test_overlap_is_visible": holdout["test_overlap_prior_policy_search_count"] == len(partitions["test"]),
        "latest_model_training_does_not_overlap_test": holdout["test_overlap_prior_count"] == 0,
        "full_model_training_inventory_used": len(partitions["train"]) == 689,
        "costs_are_explicitly_simulated": "simulated currency" in str(design.get("cost_interpretation", "")),
        "freeze_review_expected_without_manifest": freeze["status"] == "FREEZE_REVIEW_REQUIRED",
        "design_is_exploratory_history": "exploratory" in str(design.get("classification", "")).lower()
        and "previously_explored" in str(design.get("classification", "")).lower(),
        "no_runtime_labels_in_schema_issues": all(
            field not in json.dumps(schema, ensure_ascii=False).lower()
            for field in ('"home_goals"', '"away_goals"', '"outcome"')
        ),
    }
    output = {
        "schema_version": "round86-policy-experiment-protocol-audit-v2",
        "status": "ROUND86_PROTOCOL_GATE_AUDIT",
        "audit_mode": "legacy_archive_diagnostic",
        "protocol_consistent": False,
        "checks": checks,
        "schema": {
            "status": schema["status"],
            "rows": schema["rows"],
            "matches": schema["matches"],
            "missing_field_counts": schema["missing_field_counts"],
            "issue_counts": schema["issue_counts"],
            "errors": schema["errors"],
            "warnings": schema["warnings"],
        },
        "holdout": {
            **holdout,
            "issues": holdout["issues"],
        },
        "parameter_freeze": freeze,
        "latest_fold": {
            "fold": latest.get("fold"),
            "train_matches": len(partitions["train"]),
            "validation_matches": len(partitions["validation"]),
            "test_matches": len(partitions["test"]),
            "test_start": split.get("test_start"),
            "test_end": split.get("test_end"),
        },
        "evidence_boundary": {
            "current_archive": "previously explored residual-EVSI history",
            "independent_confirmation": False,
            "cost_interpretation": design.get("cost_interpretation"),
            "diagnostic_success_meaning": "expected historical deficiencies were detected, not that the archive passed the protocol",
            "required_future_run": "new episode schema + declared holdout + frozen parameter manifest before test",
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, help="New-run JSONL; requires --manifest")
    parser.add_argument("--manifest", type=Path, help="Single-fold frozen run contract")
    parser.add_argument("--output", type=Path, help="Write the audit JSON to this path")
    args = parser.parse_args()
    if bool(args.episodes) != bool(args.manifest):
        parser.error("--episodes and --manifest must be supplied together")
    try:
        if args.episodes:
            output = audit_policy_experiment(load_jsonl(args.episodes), json.loads(args.manifest.read_text(encoding="utf-8")))
            passed = output["protocol_consistent"]
        else:
            output = build_audit()
            passed = all(output["checks"].values())
    except (OSError, ValueError, TypeError) as exc:
        output = {"status": "INPUT_ERROR", "protocol_consistent": False, "error": str(exc)}
        passed = False
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
