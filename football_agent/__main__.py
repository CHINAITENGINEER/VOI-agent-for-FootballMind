"""Small standalone entrypoint; shared project CLI remains integration-owned."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .audit import RunStore, atomic_write
from .engine import Workflow
from .policy import FixedOrderPolicy
from .types import Budget


def parser():
    root = argparse.ArgumentParser(description="FootballMind Agent workflow and auditable replay")
    root.add_argument("--store-dir", default="var/parallel/agent/round2/runs")
    root.add_argument("--model-catalog", help="Server-owned model allowlist JSON; clients select registered IDs only")
    root.add_argument("--enable-platform-refresh", action="store_true", help="Enable bounded existing spftll collector (2 requests maximum per refresh)")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="List server-registered model IDs and requirements")
    for name in ("capabilities", "matches", "run"):
        command = sub.add_parser(name)
        command.add_argument("--mode", choices=("real", "test"), default="real")
        command.add_argument("--scenario", default="normal")
        command.add_argument("--archive-root", default="football_spider/platform_data/raw")
        command.add_argument("--model-id", default="market")
        command.add_argument("--time-mode", choices=("archive_replay", "live"), default="archive_replay")
        command.add_argument("--information-mode", choices=("operational", "historical_reconstruction"), default="operational")
        command.add_argument("--platform-export", help="Completed platform export path, configured by operator")
        command.add_argument("--archive-replay", action="store_true", help="Replay existing archive arrivals without an external refresh")
        if name != "capabilities":
            command.add_argument("--cutoff")
        if name == "run":
            command.add_argument("--match-id")
            command.add_argument("--max-calls", type=int, default=20)
            command.add_argument("--max-queries", type=int, default=3)
            command.add_argument("--max-requests", type=int, default=6)
            command.add_argument("--max-cost", type=float, default=3.0)
            command.add_argument("--max-elapsed-seconds", type=float, default=60.0)
            command.add_argument("--max-tool-seconds", type=float, default=10.0)
            command.add_argument("--topics", default="odds,lineups,injuries")
            command.add_argument("--initial-only", action="store_true", help="Execute match/evidence/features/predict then user stop")
            command.add_argument("--output", help="Optional complete report JSON file")
    for name in ("show", "verify"):
        command = sub.add_parser(name)
        command.add_argument("run_id")
    command = sub.add_parser("resume", help="Restore exact persisted tool configuration and continue a run")
    command.add_argument("run_id")
    command.add_argument("--action", choices=("next", "run", "resume", "stop", "defer", "query"), default="run")
    command.add_argument("--topic")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from .model_catalog import ModelCatalog
        catalog = ModelCatalog(args.model_catalog)
        service = None
        if args.enable_platform_refresh:
            from .collector_refresh import PlatformRefreshAdapter
            service = PlatformRefreshAdapter(getattr(args, "archive_root", "football_spider/platform_data/raw"))
        if args.command == "verify":
            result = RunStore(args.store_dir).verify(args.run_id)
        elif args.command == "show":
            result = RunStore(args.store_dir).load(args.run_id)
        elif args.command == "models":
            result = {"status": "OK", "models": catalog.list_models()}
        elif args.command == "resume":
            from demo.server import DemoApplication
            result = DemoApplication(args.store_dir, model_catalog=catalog, service=service).post(
                f"/api/runs/{args.run_id}/action", {"action": args.action, "topic": args.topic})
        else:
            from .adapters import LocalArchiveTools
            from .testing import SyntheticTools, TEST_CUTOFF, TEST_MATCH_ID
            if args.mode == "test":
                tools = SyntheticTools(args.scenario)
            elif args.platform_export or args.time_mode == "live" or service:
                from .frozen_features import platform_tools
                tools = platform_tools(model_id=args.model_id, model_catalog=catalog, information_mode=args.information_mode,
                                      time_mode=args.time_mode, export_path=args.platform_export, archive_root=args.archive_root, service=service)
            elif args.archive_replay:
                from .adapters import ArchiveReplayTools
                tools = ArchiveReplayTools(args.archive_root, model_id=args.model_id, model_catalog=catalog, information_mode=args.information_mode)
            else:
                tools = LocalArchiveTools(args.archive_root, model_id=args.model_id, model_catalog=catalog, information_mode=args.information_mode)
            workflow = Workflow(tools, args.store_dir,
                                FixedOrderPolicy(args.topics.split(",")) if args.command == "run" else None)
            if args.command == "capabilities":
                result = workflow.capabilities()
            elif args.command == "matches":
                from dataclasses import asdict
                result = asdict(workflow.list_matches(args.cutoff))
            else:
                match_id = args.match_id or (TEST_MATCH_ID if args.mode == "test" else None)
                cutoff = args.cutoff or (TEST_CUTOFF if args.mode == "test" else None)
                if match_id is None or cutoff is None:
                    raise ValueError("Real runs require --match-id and --cutoff with explicit timezone")
                run = workflow.start(match_id, cutoff, Budget(max_calls=args.max_calls, max_queries=args.max_queries, max_requests=args.max_requests,
                    max_cost=args.max_cost, max_elapsed_seconds=args.max_elapsed_seconds, max_tool_seconds=args.max_tool_seconds))
                if args.initial_only:
                    while run["state"] not in {"PREDICTED", "DEFERRED", "STOPPED", "WAITING"}:
                        run = workflow.step(run["run_id"])
                    if run["state"] == "PREDICTED":
                        run = workflow.step(run["run_id"], action="stop")
                else:
                    run = workflow.run_to_completion(run["run_id"])
                if args.output:
                    atomic_write(Path(args.output), run)
                result = {**run["report"], "run_path": str(workflow.store.root / run["run_id"] / "run.json"),
                          "events": len(run["events"]), "audit": workflow.store.verify(run["run_id"])}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result.get("state") not in {"DEFERRED", "WAITING"} and result.get("status", "OK") == "OK" else 2
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(json.dumps({"status": "FAILED", "category": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
