"""Run fitted contextual policies through the real frozen Agent execution path.

Selection is the first scheduled test fixture in every fold, independently of
policy gain, eventual match result, and realized action count. This checks bridge
parity, not a new policy evaluation or live/LLM effectiveness.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .acquisition import AcquisitionWorkflow
from .acquisition_evaluation import summarize_runs
from .research_bridge import make_backend


KINDS = {"visible-context": "contextual-value-v1", "source-identity": "source-identity-value-v1"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def select_cases(report, episodes):
    """Freeze selection from schedule alone and require one matched result row."""
    cases = []
    for fold in report["folds"]:
        number = fold["fold"]
        scheduled = [r for r in episodes if r["fold"] == number and r["strategy"] == "base-only" and r["budget_money"] == .05]
        _require(bool(scheduled), f"Fold {number} has no matching reference schedule")
        first = min(scheduled, key=lambda r: (r["kickoff"], r["match_id"]))
        for mode, kind in KINDS.items():
            if mode not in fold["models"]:
                continue
            strategy = "contextual-" + mode
            matched = [r for r in episodes if r["fold"] == number and r["match_id"] == first["match_id"]
                       and r["strategy"] == strategy and r["budget_money"] == .05]
            _require(len(matched) == 1, f"Fold {number}/{strategy} needs exactly one reference episode")
            _require(first["prediction_cutoff"] == matched[0]["prediction_cutoff"] and first["kickoff"] == matched[0]["kickoff"],
                     "Reference timeline differs")
            cases.append({"fold": number, "mode": mode, "kind": kind, "strategy": strategy,
                          "first": first, "expected": matched[0],
                          "lambda_money": fold["lambda_selection"][strategy + "/0.05"]["selected"],
                          "model": fold["models"][mode]})
    _require(bool(cases), "No supported contextual artifacts in the report")
    return cases


def verify(research, output, bundle):
    research, output, bundle = Path(research).resolve(), Path(output).resolve(), Path(bundle).resolve()
    report = json.loads((research / "report.json").read_text(encoding="utf-8"))
    _require(report.get("status") == "OK", "Research release is incomplete")
    episodes = [json.loads(s) for s in (research / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = select_cases(report, episodes)
    # Reuse the fixed platform costs/bindings that underlie the existing frozen
    # experiments. Read only the experiment contract, never its old models.
    root = Path(__file__).resolve().parents[1]
    template_path = root / "var/parallel/agent/round4/backend-v3-fold-2.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    checks, runs = [], []
    reference_hash = _hash(research / "report.json")
    for case in cases:
        number, mode, first = case["fold"], case["mode"], case["first"]
        predictor_path = research / "models" / f"fold-{number}-predictor.json"
        value_path = research / case["model"]["path"]
        _require(_hash(value_path) == case["model"]["sha256"], "Value-model artifact hash changed")
        config = {"partial_predictor": str(predictor_path), "value_estimator": str(value_path),
                  "value_estimator_kind": case["kind"], "lambda_cost": {"money": case["lambda_money"]},
                  "experiment": deepcopy(template["experiment"]),
                  "fixtures": [{k: first[k] for k in ("match_id", "kickoff", "prediction_cutoff")}],
                  "selection_basis": "first scheduled test fixture per fold; no gain, result, or action-count filtering",
                  "lambda_validation_budget": {"money": .05, "requests": 2, "selected_lambda_money": case["lambda_money"]},
                  "research_release": {"version": report["experiment_version"], "fold": number,
                                       "strategy": case["strategy"], "directory": str(research),
                                       "evidence_grade": report["evidence_grade"], "report_sha256": reference_hash}}
        config_path = output / f"backend-fold-{number}-{mode}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        backend = make_backend(mode="real", bundle_path=bundle, model_artifact=config_path)
        workflow = AcquisitionWorkflow(backend, output / "runs")
        run = workflow.start(first["match_id"], first["prediction_cutoff"], risk_mode="mean",
                             budget={"max_simulated_cost": .05, "max_actions": 2})
        final = workflow.run_to_completion(run["run_id"])
        expected = case["expected"]
        probabilities = [final["prediction"]["probabilities"][k] for k in ("home", "draw", "away")]
        actual_actions = [h["action_id"] for h in final["history"]]
        _require(max(abs(a - b) for a, b in zip(probabilities, expected["final_probabilities"])) < 1e-10,
                 f"Fold {number}/{mode}: final probability mismatch")
        _require(actual_actions == expected["actions"], f"Fold {number}/{mode}: acquired action mismatch")
        _require(abs(final["budget"]["simulated_cost"] - expected["cost"]["money"]) < 1e-10,
                 f"Fold {number}/{mode}: simulated cost mismatch")
        _require(final["budget"]["llm_calls"] == 0, "Unexpected LLM invocation")
        audit = workflow.store.verify(final["run_id"])
        _require(audit["status"] == "OK", "Execution audit failed")
        hashes = backend.configuration()["artifact_hashes"]
        _require(hashes == {"configuration": _hash(config_path), "predictor": _hash(predictor_path), "estimator": _hash(value_path)},
                 "Loaded artifact/configuration identity mismatch")
        runs.append(final)
        checks.append({"fold": number, "strategy": case["strategy"], "estimator_kind": case["kind"], "risk_mode": "mean",
                       "match_id": first["match_id"], "prediction_cutoff": first["prediction_cutoff"],
                       "run_id": final["run_id"], "state": final["state"], "reason": final["reason"],
                       "actions": actual_actions, "simulated_cost": final["budget"]["simulated_cost"],
                       "selected_lambda_money": case["lambda_money"], "research_probabilities_equal": True,
                       "artifact_hashes_equal": True, "audit_events": audit["events_verified"]})
        print(f"fold {number} {mode}: {final['reason']} ({len(actual_actions)} actions)", flush=True)
    acceptance = {"status": "ROUND6_CONTEXTUAL_AGENT_BRIDGE_PARITY_VERIFIED", "checks": checks,
                  "reference_report_sha256": reference_hash, "reference_episodes_sha256": _hash(research / "episodes.jsonl"),
                  "template_experiment_sha256": sha256(json.dumps(template["experiment"], sort_keys=True).encode()).hexdigest(),
                  "implementation_sha256": {name: _hash(root / name) for name in (
                      "football_agent/research_bridge.py", "football_agent/verify_round6_contextual.py")},
                  "distinct_existing_matches": len({c["match_id"] for c in checks}), "new_independent_matches": 0,
                  "llm_provider": "NOT_CONFIGURED_OR_CALLED", "new_external_requests": 0,
                  "scope": "Frozen historical bridge parity, no live benefit or LLM effectiveness claim; repeated fixtures are not new test data."}
    (output / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(summarize_runs(runs), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    result = verify(args.research, args.output, args.bundle)
    print(json.dumps({"status": result["status"], "checks": len(result["checks"])}))


if __name__ == "__main__":
    main()
