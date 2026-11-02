"""Prepare a match-only FootballMind run without asking the user for a topic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.autonomous_match_contract import build_match_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, help="例如：Santos vs Flamengo")
    parser.add_argument("--competition")
    parser.add_argument("--kickoff", help="带时区 ISO 时间；缺少时保留为待解析")
    parser.add_argument("--prediction-cutoff", dest="cutoff", help="带时区 ISO 时间")
    parser.add_argument("--match-id")
    parser.add_argument("--mode", choices=("prematch", "retrospective"), default="prematch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = {"home": None, "away": None, "competition": args.competition,
             "kickoff": args.kickoff, "prediction_cutoff": args.cutoff,
             "match_id": args.match_id, "mode": args.mode}
    text = args.match
    for separator in (" vs ", " VS ", " v ", " V ", " 对阵 ", "对阵", " 对 "):
        if separator in text:
            value["home"], value["away"] = (part.strip() for part in text.split(separator, 1))
            break
    else:
        value = args.match
    value = {key: item for key, item in value.items()} if isinstance(value, dict) else value
    bundle = build_match_bundle(value)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    (output / "request.json").write_text(json.dumps(bundle["request"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "initial-state.json").write_text(json.dumps(bundle["state"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "output": str(output),
                      "target_key": bundle["request"]["target_key"],
                      "temporal_status": bundle["request"]["temporal"]["status"],
                      "unknown_count": len(bundle["state"]["unknown_information"]),
                      "policy_mode": bundle["request"]["policy"]["mode"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
