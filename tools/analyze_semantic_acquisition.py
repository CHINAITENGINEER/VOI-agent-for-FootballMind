"""Audit assumption-sensitive information choices in one frozen semantic run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.semantic_acquisition_diagnostics import analyze_acquisition, render_acquisition_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    readable = args.output.with_suffix(".txt")
    if args.output.resolve() == readable.resolve() or args.output.exists() or readable.exists():
        parser.error("Choose a new JSON output path; existing artifacts are immutable")
    try:
        raw = args.run.read_bytes()
        report = analyze_acquisition(json.loads(raw))
        report["source_file"] = {"path": str(args.run.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
        if args.run.read_bytes() != raw:
            raise ValueError("Source run changed during analysis")
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    readable.write_text(render_acquisition_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "states": len(report["cycles"]), "grid_points": sum(c.get("grid_points", 0) for c in report["cycles"]),
                      "network_calls": 0, "semantic_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
