"""Fixed-schedule Agent parity and auditable query cards for round eleven."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .acquisition_evaluation import summarize_runs
from .calibrated_joint_bridge import CALIBRATOR_KIND, GATE_KIND, make_calibrated_joint_backend
from .calibrated_joint_workflow import CalibratedJointAcquisitionWorkflow
from .query_decision_card import build_query_decision_cards
from .verify_round7_contextual import artifact_hashes
from .verify_round8_joint import _equal, _hash, _no_hypothetical_runtime, _require
from .verify_round9_calibrated import compare_candidate_trace


POLICIES = {"frozen-version-mean": ("same_version", "mean"), "frozen-version-lcb": ("same_version", "conservative"),
            "matched-mixed-mean": ("matched_mixed", "mean")}
FIXED_LAMBDA = .5


def select_cases(report, episodes):
    cases = []
    for fold in report["folds"]:
        number = fold["fold"]
        _require(number in range(2, 7) and fold["reference_fold"] == number - 2, "Unexpected frozen predictor fold assignment")
        scheduled = [r for r in episodes if r["fold"] == number and r["strategy"] == "base-only" and r["budget_money"] == .05]
        _require(bool(scheduled), "Fixed base-only schedule missing")
        first = min(scheduled, key=lambda r: (r["kickoff"], r["match_id"]))
        for strategy, (calibration, risk) in POLICIES.items():
            rows = [r for r in episodes if r["fold"] == number and r["strategy"] == strategy
                    and r["match_id"] == first["match_id"] and r["budget_money"] == .05]
            _require(len(rows) == 1, "Exactly one fixed-schedule policy reference required")
            _require(all(first[k] == rows[0][k] for k in ("kickoff", "prediction_cutoff")), "Reference timeline differs")
            _require(rows[0]["lambda_money"] == FIXED_LAMBDA, "Round-eleven utility conversion must be fixed at .5")
            cases.append({"fold": number, "reference_fold": fold["reference_fold"], "first": first, "expected": rows[0],
                          "strategy": strategy, "risk_level": risk, "calibration": calibration,
                          "joint_model": fold["joint_model"], "calibrator": fold["calibrators"][calibration]})
    _require(len(cases) == 15 and {c["fold"] for c in cases} == set(range(2, 7)), "All five target folds and three policies required")
    return cases


def compare_prediction_updates(run, research_events):
    """Check each actual revealed return has the same updated research posterior."""
    reference = {e["step"]: e for e in research_events}
    acquired = checked = 0
    for event in run["events"]:
        if event["type"] == "acquisition_result":
            acquired += 1
        if event["type"] == "prediction_updated":
            _require(acquired in reference, "Reference missing actual-return posterior")
            actual = event["snapshot"]["prediction"]
            expected = reference[acquired]["prediction"]
            _equal([actual["probabilities"][k] for k in ("home", "draw", "away")], expected["probabilities"], "Intermediate actual-return prediction")
            _equal(actual["observed_return_categories"], expected["observed_return_categories"], "Acquired return encoding")
            _require(actual["model_version"] == expected["model_version"], "Intermediate frozen model version mismatch")
            checked += 1
    _require(checked == len(run["history"]) + 1, "Every actual return must refresh the joint posterior")
    return {"actual_return_prediction_updates_checked": checked}


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
    (output / "decision-cards").mkdir()
    runs, checks = [], []
    for case in cases:
        first, number, strategy = case["first"], case["fold"], case["strategy"]
        paths = {}
        for name in ("joint_model", "calibrator"):
            item = case[name]
            path = (research / item["path"]).resolve()
            _require(_hash(path) == item["sha256"] == hashes[item["path"].replace("\\", "/")], name + " artifact hash mismatch")
            paths[name] = path
        model_path, value_path = paths["joint_model"], paths["calibrator"]
        config = {"partial_predictor": str(model_path), "partial_predictor_kind": "learned-joint-returns-v1",
                  "value_estimator": str(value_path), "value_estimator_kind": CALIBRATOR_KIND,
                  "gate": {"kind": GATE_KIND, "risk_level": case["risk_level"]}, "lambda_cost": {"money": FIXED_LAMBDA},
                  "experiment": deepcopy(template["experiment"]),
                  "fixtures": [{k: first[k] for k in ("match_id", "kickoff", "prediction_cutoff")}],
                  "selection_basis": "first scheduled test fixture by kickoff then ID; no gain/result/action filtering",
                  "research_release": {"version": report["experiment_version"], "fold": number,
                      "reference_fold": case["reference_fold"], "strategy": strategy, "calibration": case["calibration"],
                      "lambda_source": "fixed_reporting_utility_coefficient_0.5_no_policy_tuning",
                      "reporting_utility_money_weight": FIXED_LAMBDA, "budget_money": .05, "request_budget": 2,
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
        _require([h["action_id"] for h in final["history"]] == expected["actions"], "Autonomously acquired actions differ")
        _equal(final["budget"]["simulated_cost"], expected["cost"]["money"], "Physical money spent")
        _equal(final["budget"]["actions"], expected["cost"]["requests"], "Projection calls spent")
        _require(final["budget"]["llm_calls"] == 0, "Unexpected LLM invocation")
        _require(final["state"] == "STOPPED", "Reference replay did not finish with a policy or budget STOP")
        trace = trajectories[(number, first["match_id"], strategy, .05)]
        parity = compare_candidate_trace(final, trace)
        updates = compare_prediction_updates(final, trace)
        audit = workflow.store.verify(final["run_id"])
        _require(audit["status"] == "OK", "Full event/blob audit failed")
        loaded = backend.configuration()
        _require(loaded["artifact_hashes"] == {"configuration": _hash(config_path), "predictor": _hash(model_path),
                                              "estimator": _hash(value_path)}, "Loaded artifact identity mismatch")
        _require(loaded["reference_joint"]["joint_artifact_sha256"] == _hash(model_path), "Calibrator reference model hash mismatch")
        _require(loaded["reference_joint"]["joint_model_version"] == loaded["predictor_version"] == case["joint_model"]["version"], "Frozen predictor version mismatch")
        _no_hypothetical_runtime(final["visible_state"])
        _no_hypothetical_runtime(final["prediction"])
        card = build_query_decision_cards(final)
        card_path = output / "decision-cards" / f"fold-{number}-{strategy}.json"
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        runs.append(final)
        checks.append({"fold": number, "reference_fold": case["reference_fold"], "strategy": strategy, "risk_mode": case["risk_level"],
                       "calibration": case["calibration"], "match_id": first["match_id"], "run_id": final["run_id"],
                       "actions": expected["actions"], "reason": final["reason"], "simulated_cost": final["budget"]["simulated_cost"],
                       "fixed_lambda_money": FIXED_LAMBDA, "probabilities_equal": True, "artifact_hashes_equal": True,
                       "calibrator_joint_binding_equal": True, "hypothetical_runtime_tokens": False, "audit_events": audit["events_verified"],
                       "decision_card": {"path": str(card_path.relative_to(output)), "sha256": _hash(card_path), "audit_head_hash": card["audit_head_hash"]},
                       **parity, **updates})
        print(f"fold {number} {strategy}: actions={expected['actions']}, {final['reason']}, {parity['candidate_values_checked']} candidates", flush=True)
    acceptance = {"status": "ROUND11_FROZEN_PREDICTOR_AGENT_QUERY_DECISION_PARITY_VERIFIED", "checks": checks,
                  "reference_report_sha256": _hash(research / "report.json"), "reference_episodes_sha256": _hash(research / "episodes.jsonl"),
                  "reference_trajectories_sha256": _hash(research / "trajectories.jsonl"),
                  "implementation_sha256": {name: _hash(root / name) for name in (
                      "football_agent/verify_round11_frozen.py", "football_agent/query_decision_card.py",
                      "football_agent/calibrated_joint_bridge.py", "football_agent/calibrated_joint_workflow.py",
                      "football_agent/verify_round9_calibrated.py", "football_agent/research_bridge.py", "football_agent/joint_workflow.py",
                      "football_agent/joint_bridge.py", "football_agent/acquisition.py", "football_agent/audit.py",
                      "football_agent/verify_round7_contextual.py", "football_agent/verify_round8_joint.py",
                      "football_research/acquisition/round9_residual.py", "football_research/acquisition/joint_returns.py",
                      "football_research/acquisition/joint_planner.py")},
                  "distinct_existing_matches": len({c["match_id"] for c in checks}), "new_independent_matches": 0,
                  "new_external_requests": 0, "llm_calls": 0, "actual_provider_money": None,
                  "scope": "Fixed historical same-domain query autonomy and execution parity; two views of one response, no open tool discovery or prospective effectiveness claim."}
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
