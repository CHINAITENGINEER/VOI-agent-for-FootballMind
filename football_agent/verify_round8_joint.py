"""Replay fixed historical fixtures through the actual joint-planning Agent."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .joint_workflow import JointAcquisitionWorkflow
from .acquisition_evaluation import summarize_runs
from .research_bridge import make_backend
from .verify_round7_contextual import artifact_hashes


HORIZONS = {"evsi-one-step": 1, "evsi-two-step": 2}
NUMERIC_FIELDS = ("immediate_evsi", "expected_risk_after_observation", "immediate_cost_vector", "immediate_penalty",
                  "expected_terminal_risk", "expected_plan_cost_vector", "expected_plan_penalty",
                  "expected_loss_improvement", "planning_net_gain", "selection_score", "branches")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _equal(actual, expected, name):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        _require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and abs(actual - expected) <= 1e-10,
                 name + ": numerical mismatch")
    elif isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), name + ": mapping differs")
        for key in expected:
            _equal(actual[key], expected[key], name + "/" + key)
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), name + ": list differs")
        for index, item in enumerate(expected):
            _equal(actual[index], item, name + f"/{index}")
    else:
        _require(actual == expected, name + ": value differs")


def compare_candidate_trace(run, research_events):
    """Compare every actually evaluated candidate, including contingent costs."""
    by_step = {event["step"]: event for event in research_events}
    acquired, value_steps, values_checked = 0, 0, 0
    for event in run["events"]:
        if event["type"] == "acquisition_result":
            acquired += 1
        elif event["type"] == "value_estimated":
            _require(acquired in by_step, "Research trace lacks the corresponding acquisition step")
            plan = by_step[acquired]["decision"]
            expected = {c["action_id"]: c for c in plan["candidates"] if c["feasible"]}
            actual = {c["action_id"]: c["value"] for c in event["data"]["candidates"] if c.get("value", {}).get("status") == "OK"}
            _require(set(actual) == set(expected), "The Agent and research evaluated different feasible candidates")
            for aid, value in actual.items():
                native = value["research_estimate"]
                for field in NUMERIC_FIELDS:
                    _equal(native[field], expected[aid][field], f"step-{acquired}/{aid}/{field}")
                _equal(value["net_gain"], expected[aid]["selection_score"], f"step-{acquired}/{aid}/execution_score")
                _equal(value["cost_in_loss_units"], expected[aid]["expected_plan_penalty"], f"step-{acquired}/{aid}/plan_cost")
                _require(native["planned_action"] == plan["action_id"], "Shared planner action differs")
                _require(native["horizon"] == plan["horizon"], "Finite horizon changed across bridge")
                _require(value["lower_gain"] is None and value["uncertainty"] is None, "Unmodeled parameter confidence was fabricated")
                values_checked += 1
            value_steps += 1
    return {"value_steps": value_steps, "candidate_values_checked": values_checked,
            "contingent_branches_equal": True, "expected_plan_costs_equal": True}


def _no_hypothetical_runtime(value):
    if isinstance(value, dict):
        _require("joint_return_hypothesis" not in value, "Hypothetical response token entered runtime evidence")
        for item in value.values():
            _no_hypothetical_runtime(item)
    elif isinstance(value, list):
        for item in value:
            _no_hypothetical_runtime(item)


def select_cases(report, episodes):
    cases = []
    for fold in report["folds"]:
        number = fold["fold"]
        schedule = [r for r in episodes if r["fold"] == number and r["strategy"] == "base-only" and r["budget_money"] == .05]
        _require(bool(schedule), "Reference schedule missing")
        first = min(schedule, key=lambda r: (r["kickoff"], r["match_id"]))
        for strategy, horizon in HORIZONS.items():
            rows = [r for r in episodes if r["fold"] == number and r["strategy"] == strategy
                    and r["match_id"] == first["match_id"] and r["budget_money"] == .05]
            _require(len(rows) == 1, "Exactly one matching one/two-step reference is required")
            _require(all(first[key] == rows[0][key] for key in ("kickoff", "prediction_cutoff")), "Reference timestamps differ")
            cases.append({"fold": number, "strategy": strategy, "horizon": horizon, "first": first,
                          "expected": rows[0], "joint_model": fold["joint_model"],
                          "lambda_money": fold["lambda_selection"][strategy + "/0.05"]["selected"]})
    return cases


def verify(research, output, bundle):
    research, output, bundle = Path(research).resolve(), Path(output).resolve(), Path(bundle).resolve()
    report = json.loads((research / "report.json").read_text(encoding="utf-8"))
    _require(report.get("status") == "OK", "Research release incomplete")
    episodes = [json.loads(line) for line in (research / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = select_cases(report, episodes)
    keys = {(c["fold"], c["first"]["match_id"], c["strategy"], .05) for c in cases}
    trajectories = {}
    for line in (research / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = row["fold"], row["match_id"], row["strategy"], row["budget_money"]
        if key in keys:
            _require(key not in trajectories, "Duplicate matching trajectory")
            trajectories[key] = row["events"]
    _require(set(trajectories) == keys, "Reference trajectories missing")
    release_hashes = artifact_hashes(report["model_artifacts"])
    root = Path(__file__).resolve().parents[1]
    template = json.loads((root / "var/parallel/agent/round4/backend-v3-fold-2.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    runs, checks = [], []
    for case in cases:
        number, first, strategy = case["fold"], case["first"], case["strategy"]
        joint = case["joint_model"]
        path = research / joint["path"]
        _require(_hash(path) == joint["sha256"] == release_hashes[joint["path"].replace("\\", "/")], "Joint artifact hash mismatch")
        config = {"partial_predictor": str(path), "partial_predictor_kind": "learned-joint-returns-v1",
                  "value_estimator": str(path), "value_estimator_kind": "joint-risk-planner-v1",
                  "planner": {"kind": "finite-joint-bellman-v1", "horizon": case["horizon"]},
                  "lambda_cost": {"money": case["lambda_money"]}, "experiment": deepcopy(template["experiment"]),
                  "fixtures": [{k: first[k] for k in ("match_id", "kickoff", "prediction_cutoff")}],
                  "selection_basis": "first scheduled test fixture by kickoff then ID; no gain/result/action filtering",
                  "lambda_validation_budget": {"money": .05, "requests": 2, "selected_lambda_money": case["lambda_money"]},
                  "research_release": {"version": report["experiment_version"], "fold": number, "strategy": strategy,
                                       "evidence_grade": report["evidence_grade"], "report_sha256": _hash(research / "report.json")}}
        config_path = output / f"backend-fold-{number}-{strategy}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        backend = make_backend(mode="real", bundle_path=bundle, model_artifact=config_path)
        workflow = JointAcquisitionWorkflow(backend, output / "runs")
        run = workflow.start(first["match_id"], first["prediction_cutoff"], risk_mode="mean",
                             budget={"max_simulated_cost": .05, "max_actions": 2})
        final = workflow.run_to_completion(run["run_id"])
        expected = case["expected"]
        probabilities = [final["prediction"]["probabilities"][k] for k in ("home", "draw", "away")]
        _equal(probabilities, expected["final_probabilities"], "Final prediction")
        _require([h["action_id"] for h in final["history"]] == expected["actions"], "Acquired actions differ")
        _equal(final["budget"]["simulated_cost"], expected["cost"]["money"], "Physical money spent")
        _equal(final["budget"]["actions"], expected["cost"]["requests"], "Projection calls spent")
        _require(final["budget"]["llm_calls"] == 0, "Unexpected LLM invocation")
        key = number, first["match_id"], strategy, .05
        parity = compare_candidate_trace(final, trajectories[key])
        audit = workflow.store.verify(final["run_id"])
        _require(audit["status"] == "OK", "Event audit failed")
        hashes = backend.configuration()["artifact_hashes"]
        _require(hashes == {"configuration": _hash(config_path), "predictor": _hash(path), "estimator": _hash(path)}, "Loaded artifact identity mismatch")
        _no_hypothetical_runtime(final["visible_state"])
        _no_hypothetical_runtime(final["prediction"])
        runs.append(final)
        checks.append({"fold": number, "strategy": strategy, "horizon": case["horizon"], "risk_mode": "mean",
                       "match_id": first["match_id"], "run_id": final["run_id"], "actions": expected["actions"],
                       "reason": final["reason"], "simulated_cost": final["budget"]["simulated_cost"],
                       "selected_lambda_money": case["lambda_money"], "probabilities_equal": True,
                       "artifact_hashes_equal": True, "hypothetical_runtime_tokens": False,
                       "audit_events": audit["events_verified"], **parity})
        print(f"fold {number} {strategy}: {final['reason']}, {parity['candidate_values_checked']} candidate values checked", flush=True)
    acceptance = {"status": "ROUND8_JOINT_AGENT_CANDIDATE_AND_EXECUTION_PARITY_VERIFIED", "checks": checks,
                  "reference_report_sha256": _hash(research / "report.json"), "reference_episodes_sha256": _hash(research / "episodes.jsonl"),
                  "reference_trajectories_sha256": _hash(research / "trajectories.jsonl"),
                  "implementation_sha256": {name: _hash(root / name) for name in ("football_agent/research_bridge.py", "football_agent/joint_bridge.py", "football_agent/joint_workflow.py",
                      "football_agent/verify_round8_joint.py", "football_research/acquisition/joint_planner.py", "football_research/acquisition/joint_returns.py")},
                  "distinct_existing_matches": len({c["match_id"] for c in checks}), "new_independent_matches": 0,
                  "new_external_requests": 0, "llm_calls": 0, "actual_provider_money": None,
                  "scope": "Fixed historical workflow parity under the fitted joint model; no new data, live effectiveness, or parameter-confidence claim."}
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
