"""Replay two average-gain policies; reference frozen conditional Agent cards."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .acquisition_evaluation import summarize_runs
from .audit import RunStore
from .gain_prior_bridge import ESTIMATOR_KIND, GATE_KIND, GainPriorAcquisitionWorkflow, build_gain_prior_cards, make_gain_prior_backend
from .verify_round7_contextual import artifact_hashes
from .verify_round8_joint import _equal, _hash, _no_hypothetical_runtime, _require
from .verify_round9_calibrated import compare_candidate_trace
from .verify_round11_frozen import compare_prediction_updates


POLICIES = {"tool-prior-mean": "action_only", "stage-prior-mean": "acquired_set"}
DESIGN_SHA256 = "884ac4fe147f006855acbd959b21e824d78e7169c001f1f70a41759dfe2cf8a8"


def select_cases(report, episodes, schedule_episodes=None):
    schedule_episodes = episodes if schedule_episodes is None else schedule_episodes
    cases = []
    for fold in report["folds"]:
        number = fold["fold"]
        _require(number in range(2, 7) and fold["reference_fold"] == number - 2, "Unexpected frozen predictor fold")
        schedule = [r for r in schedule_episodes if r["fold"] == number and r["strategy"] == "base-only" and r["budget_money"] == .05]
        _require(bool(schedule), "Frozen base-only test schedule missing")
        first = min(schedule, key=lambda r: (r["kickoff"], r["match_id"]))
        for strategy, mode in POLICIES.items():
            matches = [r for r in episodes if r["fold"] == number and r["strategy"] == strategy
                       and r["match_id"] == first["match_id"] and r["budget_money"] == .05]
            _require(len(matches) == 1, "Expected exactly one fixed-first average-gain reference")
            expected = matches[0]
            _require(expected["lambda_money"] == .5, "Average-gain execution lambda must be fixed at .5")
            _require(all(expected[k] == first[k] for k in ("kickoff", "prediction_cutoff")), "Reference timeline differs")
            cases.append({"fold": number, "reference_fold": fold["reference_fold"], "first": first, "expected": expected,
                          "strategy": strategy, "mode": mode, "joint_model": fold["joint_model"], "gain_prior": fold["gain_priors"][mode]})
    _require(len(cases) == 10 and {c["fold"] for c in cases} == set(range(2, 7)), "Five folds and two mean controls required")
    return cases


def reused_conditional_cards(directory, cases):
    """Verify and reference five old cards, without re-executing their policies."""
    root = Path(directory).resolve()
    acceptance = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
    _require(acceptance["status"] == "ROUND11_FROZEN_PREDICTOR_AGENT_QUERY_DECISION_PARITY_VERIFIED", "Conditional Agent reference is unverified")
    store = RunStore(root / "runs")
    output = []
    for number in range(2, 7):
        case = next(c for c in cases if c["fold"] == number)
        found = [r for r in acceptance["checks"] if r["fold"] == number and r["strategy"] == "frozen-version-mean"]
        _require(len(found) == 1 and found[0]["match_id"] == case["first"]["match_id"], "Conditional card does not use the same scheduled fixture")
        record = found[0]
        path = (root / record["decision_card"]["path"]).resolve()
        _require(_hash(path) == record["decision_card"]["sha256"], "Frozen conditional card changed")
        card = json.loads(path.read_text(encoding="utf-8"))
        run = store.load(record["run_id"])
        audit = store.verify(record["run_id"])
        _require(card["audit_head_hash"] == audit["head_hash"] == record["decision_card"]["audit_head_hash"], "Conditional card/run audit mismatch")
        _require(card["artifact_hashes"]["predictor"] == case["joint_model"]["sha256"], "Conditional comparison uses a different predictor")
        _require(run["prediction_cutoff"] == case["first"]["prediction_cutoff"] or
                 run["prediction_cutoff"].replace("+00:00", "Z") == case["first"]["prediction_cutoff"].replace("+00:00", "Z"), "Conditional prediction cutoff differs")
        output.append({"fold": number, "match_id": case["first"]["match_id"], "run_id": record["run_id"],
                       "path": str(path), "sha256": _hash(path), "audit_head_hash": audit["head_hash"],
                       "actions": card["actual_actions"], "execution_reused_not_rerun": True})
    return {"acceptance_path": str(root / "acceptance.json"), "acceptance_sha256": _hash(root / "acceptance.json"), "cards": output}


def verify(research, output, bundle, conditional_agent):
    research, output, bundle = Path(research).resolve(), Path(output).resolve(), Path(bundle).resolve()
    root = Path(__file__).resolve().parents[1]
    _require(_hash(root / "docs/research/acquisition/round12-design.md") == DESIGN_SHA256, "Preregistered round-twelve design changed")
    report = json.loads((research / "report.json").read_text(encoding="utf-8"))
    _require(report.get("status") == "OK", "Research release incomplete")
    episodes = [json.loads(s) for s in (research / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    schedule_path = Path(report["source_round11"]) / "episodes.jsonl"
    _require(_hash(schedule_path) == report["source_artifacts_sha256"]["episodes.jsonl"], "Frozen test schedule hash mismatch")
    schedule = [json.loads(s) for s in schedule_path.read_text(encoding="utf-8").splitlines()]
    cases = select_cases(report, episodes, schedule)
    reused = reused_conditional_cards(conditional_agent, cases)
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
        for name in ("joint_model", "gain_prior"):
            item = case[name]
            path = (research / item["path"]).resolve()
            _require(_hash(path) == item["sha256"] == hashes[item["path"].replace("\\", "/")], name + " artifact hash mismatch")
            paths[name] = path
        model_path, value_path = paths["joint_model"], paths["gain_prior"]
        config = {"partial_predictor": str(model_path), "partial_predictor_kind": "learned-joint-returns-v1",
                  "value_estimator": str(value_path), "value_estimator_kind": ESTIMATOR_KIND,
                  "gate": {"kind": GATE_KIND, "risk_level": "mean", "erasure_mode": case["mode"]},
                  "lambda_cost": {"money": .5}, "experiment": deepcopy(template["experiment"]),
                  "fixtures": [{k: first[k] for k in ("match_id", "kickoff", "prediction_cutoff")}],
                  "selection_basis": "first scheduled test fixture by kickoff then ID; no result/gain/action filtering",
                  "research_release": {"version": report["experiment_version"], "fold": number, "reference_fold": case["reference_fold"],
                      "strategy": strategy, "erasure_mode": case["mode"], "lambda_source": "fixed_reporting_utility_coefficient_0.5_no_policy_tuning",
                      "reporting_utility_money_weight": .5, "budget_money": .05, "request_budget": 2,
                      "evidence_grade": report["evidence_grade"], "report_sha256": _hash(research / "report.json")}}
        config_path = output / f"backend-fold-{number}-{strategy}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        backend = make_gain_prior_backend(bundle_path=bundle, model_artifact=config_path)
        workflow = GainPriorAcquisitionWorkflow(backend, output / "runs")
        run = workflow.start(first["match_id"], first["prediction_cutoff"], budget={"max_simulated_cost": .05, "max_actions": 2})
        final = workflow.run_to_completion(run["run_id"])
        expected = case["expected"]
        _equal([final["prediction"]["probabilities"][k] for k in ("home", "draw", "away")], expected["final_probabilities"], "Final prediction")
        _require([h["action_id"] for h in final["history"]] == expected["actions"], "Average-gain action sequence differs")
        _equal(final["budget"]["simulated_cost"], expected["cost"]["money"], "Physical money spent")
        _equal(final["budget"]["actions"], expected["cost"]["requests"], "Projection calls spent")
        _require(final["budget"]["llm_calls"] == 0 and final["state"] == "STOPPED", "Unexpected LLM use or nonterminal replay")
        trace = trajectories[(number, first["match_id"], strategy, .05)]
        parity = compare_candidate_trace(final, trace)
        updates = compare_prediction_updates(final, trace)
        audit = workflow.store.verify(final["run_id"])
        _require(audit["status"] == "OK", "Event/blob audit failed")
        loaded = backend.configuration()
        _require(loaded["artifact_hashes"] == {"configuration": _hash(config_path), "predictor": _hash(model_path), "estimator": _hash(value_path)}, "Loaded artifact identity mismatch")
        _require(loaded["reference_joint"]["joint_artifact_sha256"] == _hash(model_path), "Estimator reference joint hash mismatch")
        _require(loaded["reference_joint"]["joint_model_version"] == loaded["predictor_version"] == case["joint_model"]["version"], "Frozen joint version mismatch")
        _require(loaded["value_estimator_kind"] == ESTIMATOR_KIND and loaded["gain_prior_gate"]["erasure_mode"] == case["mode"], "Average-gain estimator identity lost")
        _no_hypothetical_runtime(final["visible_state"])
        _no_hypothetical_runtime(final["prediction"])
        card = build_gain_prior_cards(final)
        card_path = output / "decision-cards" / f"fold-{number}-{strategy}.json"
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        runs.append(final)
        checks.append({"fold": number, "reference_fold": case["reference_fold"], "strategy": strategy, "risk_mode": "mean",
                       "estimator_kind": ESTIMATOR_KIND, "erasure_mode": case["mode"], "model_evsi": None,
                       "match_id": first["match_id"], "run_id": final["run_id"], "actions": expected["actions"], "reason": final["reason"],
                       "simulated_cost": final["budget"]["simulated_cost"], "fixed_lambda_money": .5,
                       "probabilities_equal": True, "artifact_hashes_equal": True, "audit_events": audit["events_verified"],
                       "decision_card": {"path": str(card_path.relative_to(output)), "sha256": _hash(card_path), "audit_head_hash": card["audit_head_hash"]},
                       **parity, **updates})
        print(f"fold {number} {strategy}: actions={expected['actions']}, {final['reason']}, {parity['candidate_values_checked']} candidates", flush=True)
    names = ("football_agent/gain_prior_bridge.py", "football_agent/verify_round12_state_ablation.py",
             "football_agent/query_decision_card.py", "football_agent/verify_round11_frozen.py", "football_agent/verify_round9_calibrated.py",
             "football_agent/calibrated_joint_bridge.py", "football_agent/calibrated_joint_workflow.py", "football_agent/research_bridge.py",
             "football_agent/joint_workflow.py", "football_agent/joint_bridge.py", "football_agent/acquisition.py", "football_agent/audit.py",
             "football_research/acquisition/gain_prior.py", "football_research/acquisition/joint_returns.py", "football_research/acquisition/round9_residual.py")
    acceptance = {"status": "ROUND12_STATE_ERASURE_AGENT_QUERY_PARITY_VERIFIED", "checks": checks, "conditional_reference": reused,
                  "design_sha256": DESIGN_SHA256, "reference_report_sha256": _hash(research / "report.json"),
                  "reference_episodes_sha256": _hash(research / "episodes.jsonl"), "reference_trajectories_sha256": _hash(research / "trajectories.jsonl"),
                  "frozen_test_schedule": {"path": str(schedule_path.resolve()), "sha256": _hash(schedule_path)},
                  "implementation_sha256": {name: _hash(root / name) for name in names},
                  "distinct_existing_matches": len({c["match_id"] for c in checks}), "new_independent_matches": 0,
                  "new_external_requests": 0, "llm_calls": 0, "actual_provider_money": None,
                  "scope": "Two mean-only state-erased value controls; unchanged evidence-conditioned predictor, historical controlled views, no live effectiveness claim."}
    (output / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(summarize_runs(runs), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("research", "output", "bundle", "conditional-agent"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = verify(args.research, args.output, args.bundle, args.conditional_agent)
    print(json.dumps({"status": result["status"], "checks": len(result["checks"])}))


if __name__ == "__main__":
    main()
