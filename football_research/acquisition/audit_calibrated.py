"""Read-only replay and boundary audit of frozen calibrated acquisition runs.

Coefficient sensitivity holds logged initial states and validation lambdas fixed;
it is not a policy rollout or a test-set hyperparameter selection procedure.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean

from .lookahead_calibrated import select_two_step_v2
from .partial import PartialPredictor
from .policy import select_or_stop
from .risk_calibrated import CalibratedRiskAwareValueEstimator
from .state import GROUP_FIELDS


def _rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _poison(state):
    result = deepcopy(state)
    result.update(label="POISON", result={"home_goals": 99},
                  feature_index="UNOPENED_POISON_PATH", future_snapshot={"xg": -999})
    for group, value in result["groups"].items():
        value["label"] = "POISON"
        if value["status"] != "observed":
            value["values"] = {field: -999999 for field in GROUP_FIELDS[group]}
    return result


def _check_probability(prediction):
    values = prediction["probabilities"]
    assert len(values) == 3 and all(math.isfinite(p) and 0 <= p <= 1 for p in values)
    assert abs(sum(values) - 1) < 1e-10


def audit(run):
    run = Path(run)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    design = json.loads((run / "design.json").read_text(encoding="utf-8"))
    catalog = json.loads((run / "catalog.json").read_text(encoding="utf-8"))["actions"]
    models = {fold["fold"]: (
        PartialPredictor.load(run / "models" / f"fold-{fold['fold']}-predictor.json"),
        CalibratedRiskAwareValueEstimator.load(run / "models" / f"fold-{fold['fold']}-risk.json")
    ) for fold in report["folds"]}
    episodes = [r for r in _rows(run / "episodes.jsonl") if r["budget_money"] == .05 and r["seed"] == 0]
    by_episode = {(r["fold"], r["match_id"], r["strategy"]): r for r in episodes}
    checked = Counter()
    sensitivities = defaultdict(Counter)
    signatures = defaultdict(set)
    groups = tuple(GROUP_FIELDS)

    for trace in _rows(run / "trajectories.jsonl"):
        if trace["budget_money"] != .05 or trace["seed"] != 0:
            continue
        predictor, estimator = models[trace["fold"]]
        if trace["strategy"] == "all-available":
            full = trace["events"][-1]["visible_state"]
            assert all(v["status"] == "observed" for v in full["groups"].values())
            for mask in range(8):
                state = deepcopy(full)
                for index, group in enumerate(groups):
                    if not mask & (1 << index):
                        state["groups"][group] = {"status": "not_acquired"}
                prediction = predictor.predict_partial(state)
                _check_probability(prediction)
                assert prediction["mask_key"] == mask
                assert predictor.predict_partial(_poison(state)) == prediction
                checked["mask_predictions"] += 1
            for group in groups:
                for status in ("source_missing", "stale"):
                    state = deepcopy(full)
                    state["groups"][group]["status"] = status
                    prediction = predictor.predict_partial(state)
                    _check_probability(prediction)
                    assert group not in prediction["mask"]
                    assert predictor.predict_partial(_poison(state)) == prediction
                    checked["unusable_group_predictions"] += 1
            continue
        if trace["strategy"] not in {"risk-mean", "risk-conservative", "two-step"}:
            continue
        episode = by_episode[trace["fold"], trace["match_id"], trace["strategy"]]
        spent = {"money": 0., "requests": 0.}
        acquired = set()
        conservative = trace["strategy"] == "risk-conservative"
        for event in trace["events"]:
            state = event["visible_state"]
            prediction = predictor.predict_partial(state)
            assert prediction == event["prediction"]
            assert predictor.predict_partial(_poison(state)) == prediction
            candidates = [a for a in catalog if a["action_id"] not in acquired]
            budget = {"money": .05 - spent["money"], "requests": design["request_budget"] - spent["requests"]}
            weights = {"money": episode["lambda_money"]}
            risk_level = "conservative" if conservative else "mean"
            values = estimator.estimate_value(state, candidates, prediction=prediction, risk_level=risk_level)
            poisoned_candidates = [dict(a, result={"xg": 999}, label="POISON") for a in candidates]
            assert estimator.estimate_value(_poison(state), poisoned_candidates,
                prediction=prediction, risk_level=risk_level) == values

            def decide(visible, actions, estimates):
                if not actions:
                    return {"decision": "stop", "action_id": None, "reason": "no_legal_action"}
                if trace["strategy"] == "two-step":
                    return select_two_step_v2(visible, actions, predictor=predictor.predict_partial,
                        estimator=estimator, budget=budget, lambda_cost=weights,
                        min_support=design["min_support"])
                return select_or_stop(visible, estimates, budget, weights,
                    conservative=conservative, min_support=design["min_support"])

            decision = decide(state, candidates, values)
            assert decide(_poison(state), poisoned_candidates, values) == decision
            for key in ("decision", "action_id", "reason"):
                assert decision[key] == event["decision"][key], (trace["match_id"], key)
            checked["replayed_decisions"] += 1
            if trace["strategy"] == "risk-mean" and event["step"] == 0:
                signatures[trace["fold"]].add(tuple((v["action_id"], v["expected_gain"]) for v in values))
                for coefficient in (0., .5, 1., 1.64, 2., 3.):
                    adjusted = deepcopy(values)
                    for value in adjusted:
                        value["lower_gain"] = (value["gain"] - coefficient * value["uncertainty"]
                            if value["gain"] is not None and value["uncertainty"] is not None else None)
                    choice = select_or_stop(state, adjusted, budget, weights,
                        conservative=True, min_support=design["min_support"])
                    sensitivities[coefficient][choice["action_id"] or "STOP"] += 1
            if "charged_cost" in event:
                acquired.add(event["decision"]["action_id"])
                for unit in spent:
                    spent[unit] += event["charged_cost"][unit]

    calibration = {}
    for strategy in sorted({r["strategy"] for r in episodes}):
        selected = [r for r in episodes if r["strategy"] == strategy]
        bins = []
        for index in range(10):
            cell = [r for r in selected if min(9, int(max(r["final_probabilities"]) * 10)) == index]
            if cell:
                bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                    "matches": len(cell), "confidence": mean(max(r["final_probabilities"]) for r in cell),
                    "accuracy": mean(r["accuracy"] for r in cell)})
        calibration[strategy] = {"bins": bins, "independent_matches": len(selected),
            "top_label_ece": sum(b["matches"] * abs(b["confidence"] - b["accuracy"]) for b in bins) / len(selected)}
    points = [{"key": key, "money": value["money"], "log_loss": value["log_loss"]}
              for key, value in report["metrics"].items()]
    frontier = [p for p in points if not any(q["money"] <= p["money"] and q["log_loss"] <= p["log_loss"]
                and (q["money"] < p["money"] or q["log_loss"] < p["log_loss"]) for q in points)]
    return {"status": "PASS", "run": str(run), "checks": dict(checked),
        "initial_value_vectors_per_fold": {str(k): len(v) for k, v in signatures.items()},
        "coefficient_sensitivity": {str(k): dict(v) for k, v in sensitivities.items()},
        "sensitivity_scope": "fixed risk-mean initial states/lambdas; descriptive gates only, no rollouts or coefficient selection",
        "probability_calibration": calibration,
        "calibration_scope": "top-label confidence versus frozen episode accuracy, ten equal-width bins, budget .05 seed 0; not classwise calibration; random seed 0 only",
        "empirical_cost_quality_frontier": frontier,
        "frontier_scope": "observed point estimates, including repeated equivalent points; no superiority or uncertainty-adjusted frontier claim",
        "source_quality_scope": "missing/stale exclusion and hidden-field invariance; no learned reliability or quality-effect validation",
        "failed_scope": "public FAILED is source_missing plus reason in platform_bridge; no distinct failed subset model",
        "hashes": {name: sha256((run / name).read_bytes()).hexdigest()
                   for name in ("report.json", "episodes.jsonl", "trajectories.jsonl", "catalog.json")},
        "audit_implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.run)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Append a new named audit artifact; frozen experiments are never rewritten.
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("status", "checks", "initial_value_vectors_per_fold", "coefficient_sensitivity")}))


if __name__ == "__main__":
    main()
