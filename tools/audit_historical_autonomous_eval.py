#!/usr/bin/env python3
"""Run the historical as-of audit against the current frozen exports.

Example (from the repository root)::

    python tools/audit_historical_autonomous_eval.py \
      --matches var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-9b0135105553a7a7/matches.jsonl \
      --features var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-9b0135105553a7a7/features.jsonl \
      --trajectories var/parallel/baselines/round9/residual-evsi-v1/trajectories.jsonl \
      --episodes var/parallel/baselines/round9/residual-evsi-v1/episodes.jsonl \
      --out var/parallel/agent/historical-protocol/report.json

The trajectory and episode files are optional.  If they are supplied, the
trajectory is audited as policy-visible state and the episode metrics are
reported as evaluator-side descriptive statistics only.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys
from typing import Any

# Running a file below ``tools/`` sets ``sys.path[0]`` to that directory.  Add
# the repository root explicitly so the documented ``python tools/...`` entry
# point behaves the same way as ``python -m`` and as the test runner.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from football_research.evaluation.historical_protocol import (
    build_historical_snapshots,
    load_jsonl,
    protocol_report,
    rolling_episode_folds,
    summarize_policy_metrics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", required=True, type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--trajectories", type=Path)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--initial-validation-start", default="2025-10-01T00:00:00Z")
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--lead-hours", nargs="+", type=float, default=(24.0, 1.0))
    parser.add_argument("--information-mode", default="historical_reconstruction")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    matches = load_jsonl(args.matches)
    snapshots, exclusions = build_historical_snapshots(matches, lead_hours=args.lead_hours)
    features = load_jsonl(args.features) if args.features else []
    trajectories = load_jsonl(args.trajectories) if args.trajectories else []
    folds = rolling_episode_folds(
        snapshots,
        initial_validation_start=args.initial_validation_start,
        validation_window=timedelta(days=args.validation_days),
        test_window=timedelta(days=args.test_days),
        step=None if args.step_days is None else timedelta(days=args.step_days),
        max_folds=args.max_folds,
    )
    report = protocol_report(
        match_rows=snapshots,
        feature_rows=features,
        trajectory_rows=trajectories,
        folds=folds,
        expected_information_mode=args.information_mode,
    )
    report["input"] = {
        "matches_path": str(args.matches),
        "features_path": None if args.features is None else str(args.features),
        "trajectories_path": None if args.trajectories is None else str(args.trajectories),
        "episodes_path": None if args.episodes is None else str(args.episodes),
        "requested_lead_hours": list(args.lead_hours),
        "snapshot_exclusions": exclusions,
    }
    if args.episodes:
        report["policy_metrics"] = summarize_policy_metrics(load_jsonl(args.episodes))
        comparison_eligible = (
            report["status"] == "PASS"
            and report["counts"].get("errors", 0) == 0
            and report["counts"].get("warnings", 0) == 0
        )
        report["policy_metrics"]["eligible_for_comparison"] = comparison_eligible
        report["policy_metrics"]["eligibility_reason"] = (
            "protocol_pass_no_warnings" if comparison_eligible else "protocol_audit_failed_or_warned"
        )
    return report


def main() -> int:
    args = _parser().parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "evidence_grade": report["evidence_grade"],
        "counts": report["counts"],
        "issue_counts": report["issue_counts"],
        "folds": len(report["folds"]),
        "out": str(args.out),
    }, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
