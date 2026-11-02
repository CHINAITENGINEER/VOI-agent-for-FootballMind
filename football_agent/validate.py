"""Reproducible offline acceptance, keeping real and synthetic evidence separate.

Run from workspace root: python -m football_agent.validate
No collector or database mutation is involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .adapters import ArchiveReplayTools, LocalArchiveTools
from .audit import atomic_write
from .engine import Workflow
from .testing import SCENARIOS, SyntheticTools, TEST_CUTOFF, TEST_MATCH_ID
from .types import now


EXPECTED = {"normal": ("STOPPED", "NO_NEW_VALID_EVIDENCE"), "empty": ("DEFERRED", "NO_DATA"),
            "timeout": ("DEFERRED", "TIMEOUT"), "rate_limited": ("WAITING", "RATE_LIMITED"),
            "duplicate": ("STOPPED", "NO_NEW_VALID_EVIDENCE"), "stale": ("DEFERRED", "NO_VALID_EVIDENCE"),
            "conflict": ("DEFERRED", "CONFLICTING_EVIDENCE"), "budget": ("STOPPED", "COST_BUDGET_EXHAUSTED"),
            "model_unavailable": ("DEFERRED", "MODEL_UNAVAILABLE")}


def exercise(tools, folder, name, match_id, cutoff):
    workflow = Workflow(tools, folder / "runs")
    run = workflow.start(match_id, cutoff)
    run = workflow.run_to_completion(run["run_id"])
    output = folder / f"{name}.json"
    atomic_write(output, run)
    predicted = [e["data"]["after"] for e in run["events"] if e["type"] == "prediction_updated"]
    summary = {"run_id": run["run_id"], "mode": run["mode"], "state": run["state"], "reason": run["reason"],
               "report_path": str(output.resolve()), "run_path": str((folder / "runs" / run["run_id"] / "run.json").resolve()),
               "match": run["match"], "prediction_cutoff": cutoff, "budget": run["budget"],
               "events": dict(Counter(e["type"] for e in run["events"])),
               "prediction_history": [{k: p[k] for k in ("home_win_probability", "draw_probability", "away_win_probability", "model_version")}
                                      for p in predicted], "audit": workflow.store.verify(run["run_id"])}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="var/parallel/agent/acceptance")
    parser.add_argument("--archive-root", default="football_spider/platform_data/raw")
    parser.add_argument("--match-id", default="spftll:3013707")
    parser.add_argument("--cutoff", default="2026-09-20T11:10:00Z")
    args = parser.parse_args(argv)
    folder = Path(args.output_dir)
    report = {"generated_at": now(), "external_requests": 0, "synthetic": {}, "real": {}, "failures": []}
    for scenario in SCENARIOS:
        result = exercise(SyntheticTools(scenario), folder, "synthetic-" + scenario, TEST_MATCH_ID, TEST_CUTOFF)
        report["synthetic"][scenario] = result
        if (result["state"], result["reason"]) != EXPECTED[scenario]:
            report["failures"].append("synthetic-" + scenario)
    real = exercise(ArchiveReplayTools(args.archive_root), folder, "real-archive-replay", args.match_id, args.cutoff)
    report["real"]["archive_replay"] = real
    if real["state"] != "STOPPED" or not real["prediction_history"]:
        report["failures"].append("real-archive-replay")
    disconnected = exercise(LocalArchiveTools(args.archive_root), folder, "real-refresh-unconfigured", args.match_id, args.cutoff)
    report["real"]["refresh_unconfigured"] = disconnected
    if (disconnected["state"], disconnected["reason"]) != ("DEFERRED", "NEEDS_CONFIGURATION"):
        report["failures"].append("real-refresh-unconfigured")
    report["capabilities"] = LocalArchiveTools(args.archive_root).capabilities()
    report["status"] = "OK" if not report["failures"] else "FAILED"
    atomic_write(folder / "validation.json", report)
    print(json.dumps({"status": report["status"], "report": str((folder / "validation.json").resolve()),
                      "synthetic_cases": len(SCENARIOS), "real_cases": len(report["real"]), "failures": report["failures"]}, ensure_ascii=False, indent=2))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
