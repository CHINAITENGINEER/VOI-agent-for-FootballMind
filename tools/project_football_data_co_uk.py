"""Build the local football-data.co.uk historical projection without HTTP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_research.data_adapters.football_data_co_uk import build_projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="football-data.co.uk crawl directory")
    parser.add_argument("--output", type=Path, default=None, help="projection directory; defaults to ROOT/projected")
    args = parser.parse_args()
    report = build_projection(args.root, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
