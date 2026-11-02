"""Audit two continuation policies, planned totals and actual query receipts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from .acquisition_evaluation import summarize_runs
from .audit import RunStore
from .calibrated_continuation_bridge import ESTIMATOR_KIND, GATE_KIND, GAIN_SEMANTICS, COST_SEMANTICS, CalibratedContinuationAcquisitionWorkflow, build_calibrated_continuation_cards, make_calibrated_continuation_backend
from .verify_round12_state_ablation import reused_conditional_cards
from .verify_round7_contextual import artifact_hashes
from .verify_round8_joint import _equal, _hash, _no_hypothetical_runtime, _require
from .verify_round9_calibrated import compare_candidate_trace
from .verify_round11_frozen import compare_prediction_updates


POLICIES = {"calibrated-adaptive-two-step": "adaptive", "calibrated-commit-two-step": "commit"}
DESIGN_SHA256 = "9739f3f33abdb7fcdefd532b2f63f2f027a93ced692cb785ef01114e4e8eefc7"


def select_cases(report, episodes, schedule_episodes):
    cases = []
    for fold in report["folds"]:
        number = fold["fold"]
        _require(number in range(2,7) and fold["reference_fold"] == number-2, "Unexpected frozen predictor fold")
        scheduled = [r for r in schedule_episodes if r["fold"] == number and r["strategy"] == "base-only" and r["budget_money"] == .05]
        _require(bool(scheduled), "Frozen base-only test schedule missing")
        first = min(scheduled,key=lambda r:(r["kickoff"],r["match_id"]))
        for strategy,mode in POLICIES.items():
            rows = [r for r in episodes if r["fold"] == number and r["strategy"] == strategy
                    and r["match_id"] == first["match_id"] and r["budget_money"] == .05]
            _require(len(rows) == 1, "Exactly one fixed-first continuation reference required")
            _require(rows[0]["lambda_money"] == .5, "Continuation lambda must be fixed at .5")
            _require(all(rows[0][k] == first[k] for k in ("kickoff","prediction_cutoff")), "Reference timeline differs")
            cases.append({"fold":number,"reference_fold":fold["reference_fold"],"first":first,"expected":rows[0],
                          "strategy":strategy,"mode":mode,"planner":fold["planners"][mode],
                          **{k:fold[k] for k in ("joint_model","source_calibrator")}})
    _require(len(cases) == 10 and {c["fold"] for c in cases} == set(range(2,7)), "Five folds and two modes required")
    return cases


def reused_marginal_cards(directory,cases):
    root=Path(directory).resolve()
    acceptance=json.loads((root/"acceptance.json").read_text(encoding="utf-8"))
    _require(acceptance["status"] == "ROUND13_CONTENT_MARGINAL_AGENT_QUERY_PARITY_VERIFIED", "Marginal reference is unverified")
    records=[]
    store=RunStore(root/"runs")
    for number in range(2,7):
        case=next(c for c in cases if c["fold"] == number)
        found=[r for r in acceptance["checks"] if r["fold"] == number and r["strategy"] == "content-marginal-mean"]
        _require(len(found)==1 and found[0]["match_id"]==case["first"]["match_id"], "Marginal reference fixture differs")
        ref=found[0]
        path=root/ref["decision_card"]["path"]
        _require(_hash(path)==ref["decision_card"]["sha256"], "Frozen marginal card changed")
        card=json.loads(path.read_text(encoding="utf-8"))
        audit=store.verify(ref["run_id"])
        run=store.load(ref["run_id"])
        _require(audit["status"]=="OK" and card["audit_head_hash"]==audit["head_hash"]==ref["decision_card"]["audit_head_hash"], "Marginal audit differs")
        _require(card["artifact_hashes"]["predictor"]==case["joint_model"]["sha256"] and
                 card["artifact_hashes"]["source_calibrator"]==case["source_calibrator"]["sha256"], "Marginal source artifacts differ")
        _require(run["prediction_cutoff"].replace("+00:00","Z")==case["first"]["prediction_cutoff"].replace("+00:00","Z"), "Marginal cutoff differs")
        records.append({"fold":number,"match_id":case["first"]["match_id"],"path":str(path),"sha256":_hash(path),
                        "run_id":ref["run_id"],"audit_head_hash":audit["head_hash"],"actions":card["actual_actions"],"execution_reused_not_rerun":True})
    return {"acceptance_path":str(root/"acceptance.json"),"acceptance_sha256":_hash(root/"acceptance.json"),"cards":records}


def verify(research, output, bundle, conditional_agent, marginal_agent):
    research, output, bundle = Path(research).resolve(), Path(output).resolve(), Path(bundle).resolve()
    root = Path(__file__).resolve().parents[1]
    _require(_hash(root / "docs/research/acquisition/round14-design.md") == DESIGN_SHA256, "Preregistered round-fourteen design changed")
    report = json.loads((research / "report.json").read_text(encoding="utf-8"))
    _require(report.get("status") == "OK", "Research release incomplete")
    episodes = [json.loads(s) for s in (research / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    schedule_path = Path(report["source_round11"]) / "episodes.jsonl"
    _require(_hash(schedule_path) == report["source_artifacts_sha256"]["episodes.jsonl"], "Frozen test schedule hash mismatch")
    schedule = [json.loads(s) for s in schedule_path.read_text(encoding="utf-8").splitlines()]
    cases = select_cases(report, episodes, schedule)
    reused = reused_conditional_cards(conditional_agent, cases)
    reused_marginal = reused_marginal_cards(marginal_agent, cases)
    for item in reused["cards"]:
        card=json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        case=next(c for c in cases if c["fold"]==item["fold"])
        _require(card["artifact_hashes"]["estimator"]==case["source_calibrator"]["sha256"], "Conditional source calibrator differs")
    keys = {(c["fold"], c["first"]["match_id"], c["strategy"], .05) for c in cases}
    trajectories = {}
    with (research / "trajectories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["fold"], row["match_id"], row["strategy"], row["budget_money"]
            if key in keys:
                _require(key not in trajectories, "Duplicate matching trajectory")
                trajectories[key] = row["events"]
    _require(set(trajectories) == keys, "Reference trajectories missing")
    hashes = artifact_hashes(report["model_artifacts"])
    template = json.loads((root / "var/parallel/agent/round4/backend-v3-fold-2.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "decision-cards").mkdir()
    runs, checks = [], []
    for case in cases:
        first, number, strategy = case["first"], case["fold"], case["strategy"]
        paths = {}
        for name in ("joint_model", "source_calibrator", "planner"):
            item = case[name]
            path = (research / item["path"]).resolve()
            _require(_hash(path) == item["sha256"] == hashes[item["path"].replace("\\", "/")], name + " artifact hash mismatch")
            paths[name] = path
        model_path, source_path, value_path = paths["joint_model"], paths["source_calibrator"], paths["planner"]
        config = {"partial_predictor": str(model_path), "partial_predictor_kind": "learned-joint-returns-v1",
                  "value_estimator": str(value_path), "value_estimator_kind": ESTIMATOR_KIND,
                  "source_calibrator": os.path.relpath(source_path, output), "source_calibrator_kind": "residual-evsi-calibrator-v1",
                  "gate": {"kind": GATE_KIND, "risk_level": "mean", "mode": case["mode"]},
                  "lambda_cost": {"money": .5}, "experiment": deepcopy(template["experiment"]),
                  "fixtures": [{k: first[k] for k in ("match_id", "kickoff", "prediction_cutoff")}],
                  "selection_basis": "first scheduled test fixture by kickoff then ID; no result/gain/action filtering",
                  "research_release": {"version": report["experiment_version"], "fold": number, "reference_fold": case["reference_fold"],
                      "strategy": strategy, "continuation_mode": case["mode"], "lambda_source": "fixed_reporting_utility_coefficient_0.5_no_policy_tuning",
                      "reporting_utility_money_weight": .5, "budget_money": .05, "request_budget": 2,
                      "evidence_grade": report["evidence_grade"], "report_sha256": _hash(research / "report.json")}}
        config_path = output / f"backend-fold-{number}-{strategy}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        backend = make_calibrated_continuation_backend(bundle_path=bundle, model_artifact=config_path)
        workflow = CalibratedContinuationAcquisitionWorkflow(backend, output / "runs")
        run = workflow.start(first["match_id"], first["prediction_cutoff"], budget={"max_simulated_cost": .05, "max_actions": 2})
        final = workflow.run_to_completion(run["run_id"])
        expected = case["expected"]
        _equal([final["prediction"]["probabilities"][k] for k in ("home", "draw", "away")], expected["final_probabilities"], "Final prediction")
        _require([h["action_id"] for h in final["history"]] == expected["actions"], "Continuation action sequence differs")
        _equal(final["budget"]["simulated_cost"], expected["cost"]["money"], "Physical money spent")
        _equal(final["budget"]["actions"], expected["cost"]["requests"], "Projection calls spent")
        _require(final["budget"]["llm_calls"] == 0 and final["state"] == "STOPPED", "Unexpected LLM use or nonterminal replay")
        trace = trajectories[(number, first["match_id"], strategy, .05)]
        parity = compare_candidate_trace(final, trace)
        parity["planned_totals_and_branches_equal"] = parity.pop("single_step_calibration_equal")
        updates = compare_prediction_updates(final, trace)
        audit = workflow.store.verify(final["run_id"])
        _require(audit["status"] == "OK", "Event/blob audit failed")
        loaded = backend.configuration()
        _require(loaded["artifact_hashes"] == {"configuration": _hash(config_path), "predictor": _hash(model_path), "estimator": _hash(value_path), "source_calibrator": _hash(source_path)}, "Loaded artifact identity mismatch")
        _require(loaded["reference_joint"]["joint_artifact_sha256"] == _hash(model_path), "Estimator reference joint hash mismatch")
        _require(loaded["reference_joint"]["joint_model_version"] == loaded["predictor_version"] == case["joint_model"]["version"], "Frozen joint version mismatch")
        _require(loaded["value_estimator_kind"] == ESTIMATOR_KIND and loaded["continuation_gate"]["mode"] == case["mode"], "Continuation estimator identity lost")
        _no_hypothetical_runtime(final["visible_state"])
        _no_hypothetical_runtime(final["prediction"])
        _require(loaded["reference_source_calibrator"] == {"version": case["source_calibrator"]["version"],
                    "sha256": _hash(source_path), "kind": "residual-evsi-calibrator-v1"}, "Source calibrator binding differs")
        card = build_calibrated_continuation_cards(final)
        branch_count = sum(len(candidate["model_only_value_audit"]["branches"]) for c in card["cards"] for candidate in c["candidates"])
        immediate_branch_count = sum(len(candidate["model_only_value_audit"]["immediate_valuation_branches"])
                                     for c in card["cards"] for candidate in c["candidates"])
        card_path = output / "decision-cards" / f"fold-{number}-{strategy}.json"
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        runs.append(final)
        checks.append({"fold": number, "reference_fold": case["reference_fold"], "strategy": strategy, "risk_mode": "mean",
                       "estimator_kind": ESTIMATOR_KIND, "continuation_mode": case["mode"], "model_evsi": None,
                       "match_id": first["match_id"], "run_id": final["run_id"], "actions": expected["actions"], "reason": final["reason"],
                       "simulated_cost": final["budget"]["simulated_cost"], "fixed_lambda_money": .5,
                       "probabilities_equal": True, "artifact_hashes_equal": True, "audit_events": audit["events_verified"],
                       "decision_card": {"path": str(card_path.relative_to(output)), "sha256": _hash(card_path), "audit_head_hash": card["audit_head_hash"]},
                       "model_only_branches_checked": branch_count + immediate_branch_count,
                       "continuation_branches_checked": branch_count, "immediate_valuation_branches_checked": immediate_branch_count,
                       **parity, **updates})
        print(f"fold {number} {strategy}: actions={expected['actions']}, {final['reason']}, {parity['candidate_values_checked']} candidates", flush=True)
    names = ("football_agent/calibrated_continuation_bridge.py", "football_agent/verify_round14_continuation.py",
             "football_agent/verify_round12_state_ablation.py",
             "football_agent/query_decision_card.py", "football_agent/verify_round11_frozen.py", "football_agent/verify_round9_calibrated.py",
             "football_agent/calibrated_joint_bridge.py", "football_agent/calibrated_joint_workflow.py", "football_agent/research_bridge.py",
             "football_agent/joint_workflow.py", "football_agent/joint_bridge.py", "football_agent/acquisition.py", "football_agent/audit.py",
             "football_research/acquisition/calibrated_continuation.py", "football_research/acquisition/content_marginal.py", "football_research/acquisition/joint_returns.py", "football_research/acquisition/round9_residual.py")
    acceptance = {"status": "ROUND14_CALIBRATED_CONTINUATION_AGENT_PARITY_VERIFIED", "checks": checks, "conditional_reference": reused, "marginal_reference": reused_marginal,
                  "design_sha256": DESIGN_SHA256, "reference_report_sha256": _hash(research / "report.json"),
                  "reference_episodes_sha256": _hash(research / "episodes.jsonl"), "reference_trajectories_sha256": _hash(research / "trajectories.jsonl"),
                  "frozen_test_schedule": {"path": str(schedule_path.resolve()), "sha256": _hash(schedule_path)},
                  "implementation_sha256": {name: _hash(root / name) for name in names},
                  "distinct_existing_matches": len({c["match_id"] for c in checks}), "new_independent_matches": 0,
                  "new_external_requests": 0, "llm_calls": 0, "actual_provider_money": None,
                  "scope": "Two mean-only continuation surrogates; expected plan rewards/costs are separate from actual historical receipts. Model-only branches are not queries or independent games."}
    (output / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(summarize_runs(runs), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("research", "output", "bundle", "conditional-agent", "marginal-agent"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = verify(args.research, args.output, args.bundle, args.conditional_agent, args.marginal_agent)
    print(json.dumps({"status": result["status"], "checks": len(result["checks"])}))


if __name__ == "__main__":
    main()
