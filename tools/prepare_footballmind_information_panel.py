"""Prepare a validated raw information stream for the historical f/g adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_research.data_adapters.information_stream_panel import build_historical_cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-documents", type=int, default=3)
    parser.add_argument("--result-clock", choices=["observed", "source_reconstruction"], default="source_reconstruction")
    args = parser.parse_args(argv)
    cases, report = build_historical_cases(args.package, maximum_documents=args.maximum_documents, result_clock=args.result_clock)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(cases), "reasons": report["reasons"]}, ensure_ascii=False))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
