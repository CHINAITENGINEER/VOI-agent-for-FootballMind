"""Check mean/LCB calibrated one-step gates in the actual frozen Agent workflow."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .acquisition_evaluation import summarize_runs
from .calibrated_joint_bridge import CALIBRATOR_KIND, GATE_KIND, make_calibrated_joint_backend
from .calibrated_joint_workflow import CalibratedJointAcquisitionWorkflow
from .verify_round7_contextual import artifact_hashes
from .verify_round8_joint import _equal, _hash, _no_hypothetical_runtime, _require


RISK_LEVELS = {"calibrated-mean": "mean", "calibrated-lcb": "conservative"}


def select_cases(report, episodes):
    cases = []
    for fold in report["folds"]:
        number = fold["fold"]
        scheduled = [r for r in episodes if r["fold"] == number and r["strategy"] == "base-only" and r["budget_money"] == .05]
        _require(bool(scheduled), "Fixed reference schedule is missing")
        first = min(scheduled, key=lambda r: (r["kickoff"], r["match_id"]))
        for strategy, risk in RISK_LEVELS.items():
            rows = [r for r in episodes if r["fold"] == number and r["strategy"] == strategy
                    and r["match_id"] == first["match_id"] and r["budget_money"] == .05]
            _require(len(rows) == 1, "Exactly one matching calibrated reference episode is required")
            _require(all(first[k] == rows[0][k] for k in ("kickoff", "prediction_cutoff")), "Reference timeline differs")
            cases.append({"fold": number, "first": first, "expected": rows[0], "strategy": strategy, "risk_level": risk,
                          "joint_model": fold["joint_model"], "calibrator": fold["calibrator"],
                          "lambda_money": fold["lambda_selection"][strategy + "/0.05"]["selected"]})
    _require(bool(cases), "No calibrated references found")
    return cases


def compare_candidate_trace(run, research_events):
    """Include unsupported candidates: missing support is itself a gate result."""
    by_step = {e["step"]: e for e in research_events}
    acquired = steps = count = unsupported = 0
    for event in run["events"]:
        if event["type"] == "acquisition_result":
            acquired += 1
        elif event["type"] == "value_estimated":
            _require(acquired in by_step, "Research trace lacks the corresponding acquisition step")
            plan = by_step[acquired]["decision"]
            expected = {c["action_id"]: c for c in plan["candidates"] if "expected_gain" in c}
            actual = {c["action_id"]: c["value"] for c in event["data"]["candidates"] if "value" in c}
            _require(set(actual) == set(expected), "The Agent and research evaluated different legal candidates")
            for aid, value in actual.items():
                reference, native = expected[aid], value["research_estimate"]
                _equal({k: native[k] for k in reference}, reference, f"step-{acquired}/{aid}/native_candidate")
                _equal(value["expected_loss_improvement"], reference["expected_gain"], aid + "/g_hat")
                _equal(value["lower_gain"], reference["lower_gain"], aid + "/descriptive_lcb")
                _equal(value["cost_in_loss_units"], reference["cost_penalty"], aid + "/single_action_cost")
                for name in ("support", "parent_support", "effective_support", "support_status", "fallback_reason", "standard_error"):
                    _equal(value[name], reference.get(name), aid + "/" + name)
                _require(native["planned_action"] == plan["action_id"], "Calibrated gate action differs")
                _equal(native["physical_budget"], plan["physical_budget"], aid + "/physical_budget")
                _equal(native["lambda_cost"], plan["lambda_cost"], aid + "/lambda")
                if value["status"] == "OK":
                    score = value.get("conservative_net_gain") if run["risk_mode"] == "conservative" else value["net_gain"]
                    _equal(score, reference["selection_score"], aid + "/gate_score")
                else:
                    unsupported += 1
                    _require(reference.get("decision_gain") is None, "Supported calibrated candidate was dropped by the Agent")
                count += 1
            steps += 1
    return {"value_steps": steps, "candidate_values_checked": count, "unsupported_candidates_checked": unsupported,
            "single_step_calibration_equal": True, "support_and_fallback_equal": True}


def verify(research, output, bundle):
    research, output, bundle = Path(research).resolve(), Path(output).resolve(), Path(bundle).resolve()
    report = json.loads((research / "report.json").read_text(encoding="utf-8"))
    _require(report.get("status") == "OK", "Research release incomplete")
    episodes = [json.loads(s) for s in (research / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = select_cases(report, episodes)
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
    root = Path(__file__).resolve().parents[1]
    template = json.loads((root / "var/parallel/agent/round4/backend-v3-fold-2.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    runs, checks = [], []
    for case in cases:
        first, number, strategy = case["first"], case["fold"], case["strategy"]
        model_paths = {}
        for name in ("joint_model", "calibrator"):
            item = case[name]
            path = (research / item["path"]).resolve()
            _require(_hash(path) == item["sha256"] == hashes[item["path"].replace("\\", "/")], name + " artifact hash mismatch")
            model_paths[name] = path
        model_path, value_path = model_paths["joint_model"], model_paths["calibrator"]
        config = {"partial_predictor": str(model_path), "partial_predictor_kind": "learned-joint-returns-v1",
                  "value_estimator": str(value_path), "value_estimator_kind": CALIBRATOR_KIND,
                  "gate": {"kind": GATE_KIND, "risk_level": case["risk_level"]},
                  "lambda_cost": {"money": case["lambda_money"]}, "experiment": deepcopy(template["experiment"]),
                  "fixtures": [{k: first[k] for k in ("match_id", "kickoff", "prediction_cutoff")}],
                  "selection_basis": "first scheduled test fixture by kickoff then ID; no gain/result/action filtering",
                  "lambda_validation_budget": {"money": .05, "requests": 2, "selected_lambda_money": case["lambda_money"]},
                  "research_release": {"version": report["experiment_version"], "fold": number, "strategy": strategy,
                                       "evidence_grade": report["evidence_grade"], "report_sha256": _hash(research / "report.json")}}
        config_path = output / f"backend-fold-{number}-{strategy}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        backend = make_calibrated_joint_backend(bundle_path=bundle, model_artifact=config_path)
        workflow = CalibratedJointAcquisitionWorkflow(backend, output / "runs")
        run = workflow.start(first["match_id"], first["prediction_cutoff"], risk_mode=case["risk_level"],
                             budget={"max_simulated_cost": .05, "max_actions": 2})
        final = workflow.run_to_completion(run["run_id"])
        expected = case["expected"]
        _equal([final["prediction"]["probabilities"][k] for k in ("home", "draw", "away")], expected["final_probabilities"], "Final prediction")
        _require([h["action_id"] for h in final["history"]] == expected["actions"], "Acquired actions differ")
        _equal(final["budget"]["simulated_cost"], expected["cost"]["money"], "Physical money spent")
        _equal(final["budget"]["actions"], expected["cost"]["requests"], "Projection calls spent")
        _require(final["budget"]["llm_calls"] == 0, "Unexpected LLM invocation")
        parity = compare_candidate_trace(final, trajectories[(number, first["match_id"], strategy, .05)])
        audit = workflow.store.verify(final["run_id"])
        _require(audit["status"] == "OK", "Event audit failed")
        loaded = backend.configuration()
        _require(loaded["artifact_hashes"] == {"configuration": _hash(config_path), "predictor": _hash(model_path),
                                              "estimator": _hash(value_path)}, "Loaded artifact identity mismatch")
        _require(loaded["reference_joint"]["joint_artifact_sha256"] == _hash(model_path), "Calibrator reference model hash mismatch")
        _require(loaded["reference_joint"]["joint_model_version"] == loaded["predictor_version"], "Calibrator reference model version mismatch")
        _no_hypothetical_runtime(final["visible_state"])
        _no_hypothetical_runtime(final["prediction"])
        runs.append(final)
        checks.append({"fold": number, "strategy": strategy, "risk_mode": case["risk_level"], "horizon": 1,
                       "match_id": first["match_id"], "run_id": final["run_id"], "actions": expected["actions"],
                       "reason": final["reason"], "simulated_cost": final["budget"]["simulated_cost"],
                       "selected_lambda_money": case["lambda_money"], "probabilities_equal": True,
                       "artifact_hashes_equal": True, "calibrator_joint_binding_equal": True,
                       "hypothetical_runtime_tokens": False, "audit_events": audit["events_verified"], **parity})
        print(f"fold {number} {strategy}: {final['reason']}, {parity['candidate_values_checked']} candidate values checked", flush=True)
    acceptance = {"status": "ROUND9_CALIBRATED_JOINT_AGENT_CANDIDATE_AND_EXECUTION_PARITY_VERIFIED", "checks": checks,
                  "reference_report_sha256": _hash(research / "report.json"), "reference_episodes_sha256": _hash(research / "episodes.jsonl"),
                  "reference_trajectories_sha256": _hash(research / "trajectories.jsonl"),
                  "implementation_sha256": {name: _hash(root / name) for name in (
                      "football_agent/calibrated_joint_bridge.py", "football_agent/calibrated_joint_workflow.py",
                      "football_agent/verify_round9_calibrated.py", "football_agent/research_bridge.py", "football_agent/joint_workflow.py",
                      "football_agent/joint_bridge.py", "football_agent/acquisition.py", "football_agent/verify_round7_contextual.py",
                      "football_agent/verify_round8_joint.py", "football_research/acquisition/round9_residual.py",
                      "football_research/acquisition/joint_returns.py", "football_research/acquisition/joint_planner.py")},
                  "distinct_existing_matches": len({c["match_id"] for c in checks}), "new_independent_matches": 0,
                  "new_external_requests": 0, "llm_calls": 0, "actual_provider_money": None,
                  "scope": "Fixed historical single-step calibrated gate parity; no new independent data or confidence-coverage claim."}
    (output / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(summarize_runs(runs), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("research", "output", "bundle"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = verify(args.research, args.output, args.bundle)
    print(json.dumps({"status": result["status"], "checks": len(result["checks"])}))


if __name__ == "__main__":
    main()
