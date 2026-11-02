"""CLI for the same visible-state workflow served by demo.acquisition_server."""
import argparse
import json
from pathlib import Path

from .acquisition import AcquisitionWorkflow, STRATEGY_ALIASES
from .proposer import LLMProposer
from .research_bridge import make_backend


def summary(run):
    return {key: run.get(key) for key in ("run_id", "match_id", "prediction_cutoff", "strategy", "state",
            "reason", "configuration", "budget", "prediction", "visible_state", "risk_mode", "evidence", "proposal_audit", "stop_audit")}


def validate_synthetic(output):
    """Persist flow evidence, explicitly separated from trained-model evidence."""
    records = []
    for scenario, limits in (("immediate_stop", {}), ("normal", {}), ("no_data", {}),
                             ("normal", {"max_actions": 1}), ("cost_exceeds_gain", {})):
        backend = make_backend(mode="synthetic_test", scenario=scenario)
        workflow = AcquisitionWorkflow(backend, output / "runs")
        run = workflow.start("synthetic:alpha-beta", "2025-01-01T12:00:00Z", budget=limits)
        run = workflow.run_to_completion(run["run_id"])
        audit = workflow.store.verify(run["run_id"])
        records.append({"scenario": scenario, "limits": limits, **summary(run),
                        "selected_actions": [h["action_id"] for h in run["history"]], "audit": audit})
    workflow = AcquisitionWorkflow(make_backend(mode="synthetic_test"), output / "runs")
    run = workflow.start("synthetic:alpha-beta", "2025-01-01T12:00:00Z", strategy="llm_gate")
    missing = workflow.run_to_completion(run["run_id"])
    assert records[0]["budget"]["actions"] == 0
    assert records[1]["selected_actions"] == ["fixture.reveal.strength", "fixture.reveal.xg"]
    assert records[2]["budget"]["actions"] == 3
    assert records[3]["budget"]["actions"] == 1
    assert records[4]["budget"]["actions"] == 0
    assert missing["reason"] == "LLM_NEEDS_CONFIGURATION" and missing["budget"]["llm_calls"] == 0
    result = {"mode": "synthetic_test", "status": "PASS", "real_external_requests": 0,
              "cases": records, "unconfigured_llm": summary(missing)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "PASS", "mode": "synthetic_test", "artifact": str(output / "validation.json"),
            "run_ids": [r["run_id"] for r in records], "unconfigured_run_id": missing["run_id"]}


def main():
    parser = argparse.ArgumentParser(description="FootballMind public-state acquisition workflow")
    parser.add_argument("command", choices=("run", "resume", "show", "validate-synthetic"))
    parser.add_argument("--mode", choices=("real", "synthetic_test"), default="real")
    parser.add_argument("--bundle")
    parser.add_argument("--research-artifact")
    parser.add_argument("--store-dir", default="var/parallel/agent/round3/runs")
    parser.add_argument("--output", default="var/parallel/agent/round3/synthetic-acceptance")
    parser.add_argument("--match-id")
    parser.add_argument("--cutoff")
    parser.add_argument("--run-id")
    parser.add_argument("--strategy", choices=("fixed", "pure_llm", "all_candidates", "llm_gate",
                                                 *STRATEGY_ALIASES.keys()), default="all_candidates")
    parser.add_argument("--risk-mode", choices=("mean", "conservative"), default="mean")
    parser.add_argument("--budget", default="{}", help="JSON acquisition budget limits")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "validate-synthetic":
        result = validate_synthetic(Path(args.output))
    elif args.command == "show":
        from .audit import RunStore
        result = summary(RunStore(args.store_dir).load(args.run_id))
    else:
        backend = make_backend(bundle_path=args.bundle, model_artifact=args.research_artifact, mode=args.mode)
        proposer = LLMProposer.from_env(ledger_path=str(Path(args.store_dir).parent / "llm" / "cli-ledger.json"))
        workflow = AcquisitionWorkflow(backend, args.store_dir, proposer=proposer)
        if args.command == "run":
            if not args.match_id or not args.cutoff: parser.error("run requires --match-id and --cutoff")
            run = workflow.start(args.match_id, args.cutoff, args.strategy, json.loads(args.budget), args.seed, args.risk_mode)
            args.run_id = run["run_id"]
        elif not args.run_id: parser.error("resume requires --run-id")
        else: workflow.step(args.run_id)
        result = summary(workflow.run_to_completion(args.run_id))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
