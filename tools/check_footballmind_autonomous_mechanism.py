"""Run FootballMind's offline autonomous-information mechanism audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.autonomous_mechanism_acceptance import (
    run_autonomous_mechanism_acceptance,
)


def main() -> int:
    # Windows desktop terminals may default to a legacy code page.  The
    # report is a JSON artifact, so always emit UTF-8 for pipes and files.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="optional JSON report path; parent directories are created")
    args = parser.parse_args()
    report = run_autonomous_mechanism_acceptance()
    body = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
