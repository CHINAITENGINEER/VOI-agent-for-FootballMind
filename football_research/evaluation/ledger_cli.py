"""python -m football_research.evaluation.ledger_cli --help"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

from football_research.baselines.api import PredictionService, canonical_json
from .ledger import ForecastLedger


def _rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Immutable forecast registration and versioned result scoring")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register-forecasts", help="Import existing predictions as replay")
    register.add_argument("--predictions", required=True)
    register.add_argument("--fixtures", help="Research dataset JSONL or fixture JSONL, if kickoff absent in predictions")
    issue = commands.add_parser("issue-forecasts", help="Generate and freeze fresh prospective predictions")
    issue.add_argument("--model", required=True)
    issue.add_argument("--fixtures", required=True)
    issue.add_argument("--match-id", action="append", required=True)
    issue.add_argument("--prediction-cutoff")
    issue.add_argument("--features", help="Exact match/cutoff feature mappings in JSONL")
    issue.add_argument("--max-latency-seconds", type=float, default=300)
    settle = commands.add_parser("settle-forecasts", help="Append immutable result versions")
    settle.add_argument("--results", required=True)
    score = commands.add_parser("score-forecasts", help="Score active labels, separately by provenance and mode")
    score.add_argument("--as-of")
    score.add_argument("--result-grace-hours", type=float, default=3)
    for command in (register, issue, settle, score):
        command.add_argument("--ledger", required=True)
        command.add_argument("--output", help="Write command report JSON")
    args = parser.parse_args(argv)
    ledger = ForecastLedger(args.ledger)
    if args.command == "score-forecasts":
        report = ledger.score(as_of=args.as_of, result_grace_hours=args.result_grace_hours)
    else:
        operations = []
        if args.command == "register-forecasts":
            fixtures = {r["match_id"]: r for r in _rows(args.fixtures) if r.get("kind") in (None, "match")} if args.fixtures else {}
            for row in _rows(args.predictions):
                kickoff = row.get("kickoff", fixtures.get(row["match_id"], {}).get("kickoff"))
                if kickoff is None:
                    raise ValueError(f"No kickoff for {row['match_id']}")
                operations.append(ledger.import_forecast(row, kickoff=kickoff, source_reference=str(Path(args.predictions).resolve())))
        elif args.command == "settle-forecasts":
            operations = ledger.settle_results(_rows(args.results))
        else:
            service = PredictionService.load(args.model)
            features = _rows(args.features) if args.features else []
            fixtures = [r for r in _rows(args.fixtures) if r.get("kind") in (None, "match") and r["match_id"] in args.match_id]
            if {r["match_id"] for r in fixtures} != set(args.match_id):
                raise ValueError("Some requested match IDs are absent from fixtures")
            for fixture in fixtures:
                selected = [r for r in features if r["match_id"] == fixture["match_id"] and r["information_mode"] == service.information_mode]
                if len(selected) > 1:
                    raise ValueError("Supply one exact feature snapshot per requested match")
                feature = selected[0] if selected else None
                cutoff = args.prediction_cutoff or (feature["prediction_cutoff"] if feature else None)
                operations.append(ledger.issue_forecast(service, fixture, prediction_cutoff=cutoff,
                                  features=feature, max_latency_seconds=args.max_latency_seconds))
        report = {"command": args.command, "status_counts": dict(Counter(r["status"] for r in operations)),
                  "operations": operations}
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(canonical_json({"command": args.command, "status_counts": report["status_counts"], "output": args.output}))
    return report


if __name__ == "__main__":
    main()
