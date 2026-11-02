"""Partial prediction and round-four risk-aware acquisition CLIs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Exploratory partial prediction and empirical acquisition")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--platform-export", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--design")
    round4 = commands.add_parser("round4")
    round4.add_argument("--platform-export", required=True)
    round4.add_argument("--round3", required=True)
    round4.add_argument("--output", required=True)
    round4.add_argument("--design")
    args = parser.parse_args()
    design = json.loads(Path(args.design).read_bytes()) if args.design else None
    if args.command == "round4":
        from .experiment_round4 import run as experiment
        report = experiment(args.platform_export, args.output, round3_dir=args.round3, design=design)
    else:
        from .experiment import run as experiment
        report = experiment(args.platform_export, args.output, design=design)
    print(json.dumps({k: report[k] for k in ("status", "input_matches", "test_matches", "episode_runs", "estimator_runtime")}))


if __name__ == "__main__":
    main()
