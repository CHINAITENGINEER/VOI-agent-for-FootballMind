"""Independent replay/accounting audit for contextual acquisition experiments.

Outcome-bearing rows are consumed only by this offline verifier. Replaying a
record here does not expose labels to the policy that produced that record.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

from .data import ControlledEpisode, load_episodes
from .experiment import losses, outcome
from .partial import PartialPredictor
from .policy import select_or_stop
from .state import parse_time
from .verify_round4 import FORBIDDEN, _rows, _walk

KEYS = ("fold", "match_id", "strategy", "budget_money", "seed")
UNITS = ("money", "requests", "latency_ms", "tokens")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, tolerance=1e-10):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual), f"Non-finite/non-numeric {name}")
    require(abs(actual - expected) <= tolerance, f"Mismatch in {name}: {actual} vs {expected}")


def verify_episode_trace(row, trace, source, predictor, actions, request_budget,
                         estimator=None, min_support=12):
    """Reconstruct every reveal, probability, charge and final score from source."""
    require(tuple(row[k] for k in KEYS) == tuple(trace[k] for k in KEYS), "Trace identity mismatch")
    require(row["match_id"] == source["match_id"], "Source identity mismatch")
    require(parse_time(row["prediction_cutoff"]) == parse_time(source["prediction_cutoff"]), "Cutoff mismatch")
    require(parse_time(row["kickoff"]) == parse_time(source["kickoff"]), "Kickoff mismatch")
    env = ControlledEpisode(source, actions)
    charged = dict.fromkeys(UNITS, 0.0)
    events = trace["events"]
    require(bool(events), "Empty trace")
    count = 0
    for step, event in enumerate(events):
        require(event["step"] == step, "Non-contiguous trace step")
        state = event["visible_state"]
        require(not FORBIDDEN.intersection(_walk(state)), "Hidden label/path in visible state")
        expected_state = {**env.visible_state(), "action_history": [
            {"action_id": aid, "status": "acquired_or_source_missing"}
            for aid in env.performed_actions]}
        require(state == expected_state, "Visible state differs from prior permitted reveals")
        prediction = predictor.predict_partial(state)
        observed = event["prediction"]["probabilities"]
        require(len(observed) == 3, "Prediction must have three outcomes")
        for a, b in zip(observed, prediction["probabilities"]):
            close(a, b, "replayed probability")
        decision = event["decision"]
        if estimator is not None and env.legal_actions():
            public = {a["action_id"]: a for a in actions}
            candidates = [{**a, **{k: public[a["action_id"]][k] for k in
                          ("dependency_key", "cache_policy", "cost_scope", "quality_policy", "prerequisites")
                          if k in public[a["action_id"]]}} for a in env.legal_actions()]
            estimates = estimator.estimate_value(state, candidates, prediction=prediction, risk_level="mean")
            replayed = select_or_stop(state, estimates,
                                      {"money": max(0.0, row["budget_money"] - charged["money"]),
                                       "requests": max(0.0, request_budget - charged["requests"])},
                                      {"money": row["lambda_money"]}, conservative=False, min_support=min_support)
            for key in ("decision", "action_id", "reason"):
                require(decision.get(key) == replayed.get(key), "Model-replayed decision mismatch: " + key)
        if decision["decision"] == "stop":
            require(step == len(events) - 1, "Trace continues after stop")
            require("charged_cost" not in event, "Stop charged as acquisition")
            require(row["stop_reason"] == decision["reason"], "Stop reason mismatch")
            break
        require(decision["decision"] == "action", "Unknown decision type")
        available = {a["action_id"]: a for a in env.legal_actions()}
        aid = decision["action_id"]
        require(aid in available, "Illegal or repeated action")
        require("charged_cost" in event, "Acquisition lacks charge")
        for unit in UNITS:
            charge = float(available[aid]["cost"].get(unit, 0.0))
            close(event["charged_cost"].get(unit, 0.0), charge, "action charge " + unit)
            charged[unit] += charge
        require(charged["money"] <= row["budget_money"] + 1e-10, "Money budget exceeded")
        require(charged["requests"] <= request_budget, "Request budget exceeded")
        env.reveal(aid)
        count += 1
    require(events[-1]["decision"]["decision"] == "stop", "Missing terminal stop")
    require(list(env.performed_actions) == row["actions"], "Final action sequence mismatch")
    final = predictor.predict_partial(env.visible_state())
    require(row["final_mask"] == final["mask"], "Final mask mismatch")
    require(len(row["final_probabilities"]) == 3, "Final outcome count mismatch")
    for a, b in zip(row["final_probabilities"], final["probabilities"]):
        close(a, b, "final probability")
    for unit in UNITS:
        close(row["cost"][unit], charged[unit], "accumulated " + unit)
    for metric, value in losses(final["probabilities"], outcome(source)).items():
        close(row[metric], value, "source-recomputed " + metric)
    require(row["actual_external_requests"] == row["actual_llm_tokens"] == 0, "Unexpected external execution")
    return {"events": len(events), "actions": count}


def verify_aggregate_metrics(rows, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[f"{row['strategy']}/money={row['budget_money']:g}"].append(row)
    require(set(grouped) == set(metrics), "Aggregate strategy coverage mismatch")
    for key, members in grouped.items():
        per_match = defaultdict(list)
        for row in members:
            per_match[row["match_id"]].append(row)
        require(metrics[key]["independent_matches"] == len(per_match), "Independent match count mismatch")
        for name in ("log_loss", "brier_score", "accuracy", "money", "requests"):
            means = []
            for items in per_match.values():
                means.append(sum((v["cost"][name] if name in ("money", "requests") else v[name])
                                 for v in items) / len(items))
            close(metrics[key][name], sum(means) / len(means), "aggregate " + name)


def verify(run, bundle):
    from .contextual_value import ContextualValueEstimator, MODES
    from .risk_calibrated import CalibratedRiskAwareValueEstimator

    run = Path(run)
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    report, design, implementation = read("report.json"), read("design.json"), read("implementation.json")
    require(implementation == report["implementation_sha256"], "Implementation manifest mismatch")
    for name, digest in implementation.items():
        require(sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == digest, "Source code changed: " + name)
    source = Path(design["source_round5"])
    for name, digest in read("source-artifacts.json").items():
        require(sha256((source / name).read_bytes()).hexdigest() == digest, "Round-five artifact changed: " + name)
    if (run / "identity-source-artifacts.json").exists():
        identity_sources = read("identity-source-artifacts.json")
        source6 = Path(identity_sources["round6_source"])
        for name, digest in identity_sources["round6_files"].items():
            require(sha256((source6 / name).read_bytes()).hexdigest() == digest, "Original round-six artifact changed")
        for mapping in identity_sources["mapping_inputs"]:
            require(sha256(Path(mapping["path"]).read_bytes()).hexdigest() == mapping["sha256"], "Identity mapping input changed")
        from .contextual_identity import validate_crosswalk
        crosswalk = read("identity-crosswalk.json")
        validate_crosswalk(crosswalk)
        require(crosswalk["version"] == identity_sources["crosswalk_version"], "Crosswalk version mismatch")
    for artifact in report["model_artifacts"]:
        require(sha256((run / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"], "Model artifact changed")
    source_rows, audit = load_episodes(bundle, lead_hours=24)
    require(audit["platform_manifest_sha256"] == read("data-audit.json")["current"]["platform_manifest_sha256"], "Historical export changed")
    by_match = {r["match_id"]: r for r in source_rows}
    models, all_test_ids, fold_ids = {}, set(), {}
    for fold in report["folds"]:
        number = fold["fold"]
        train, val, test = (set(fold[k]) for k in ("training_match_ids", "validation_match_ids", "test_match_ids"))
        require(not (train & val or train & test or val & test or test & all_test_ids), "Fold partition overlap")
        all_test_ids.update(test)
        fold_ids[number] = test
        predictor = PartialPredictor.load(run / "models" / f"fold-{number}-predictor.json")
        require((run / "models" / f"fold-{number}-predictor.json").read_bytes() ==
                (source / "models" / f"fold-{number}-predictor.json").read_bytes(), "Predictor changed from reference")
        require(all(parse_time(predictor.metadata["training_as_of"]) <= parse_time(by_match[m]["prediction_cutoff"])
                    for m in test), "Predictor postdates test prediction")
        values = {"risk-mean": CalibratedRiskAwareValueEstimator.load(run / "models" / f"fold-{number}-risk.json")}
        require(set(MODES) <= set(fold["models"]), "Missing prespecified ablation")
        for mode, artifact in fold["models"].items():
            if mode == "source-identity":
                from .contextual_identity import SourceIdentityValueEstimator
                model = SourceIdentityValueEstimator.load(run / artifact["path"])
                require(model.mode == "source-identity" and model.model.mode == "visible-context", "Identity correction changes feature set")
                require(model.crosswalk == read("identity-crosswalk.json"), "Model crosswalk differs from frozen identity mapping")
            else:
                require(mode in MODES, "Unknown ablation")
                model = ContextualValueEstimator.load(run / artifact["path"])
                require(model.mode == mode, "Incorrect ablation artifact")
            mtrain, mcal = (set(model.manifest[k]) for k in ("training_match_ids", "calibration_match_ids"))
            require(mtrain == set(fold["value_training_match_ids"]) and mcal == set(fold["value_calibration_match_ids"]), "Value split mismatch")
            require(not mtrain & mcal and mtrain | mcal <= train, "Value training/calibration leaks")
            require(parse_time(model.manifest["trained_as_of"]) == parse_time(predictor.metadata["training_as_of"]), "Value/predictor time mismatch")
            times = model.manifest["temporal_boundary"]
            if times["calibration_min_prediction_cutoff"]:
                require(parse_time(times["train_max_result_available_at"]) < parse_time(times["calibration_min_prediction_cutoff"]), "Calibration precedes training labels")
            values["contextual-" + mode] = model
        models[number] = predictor, values
        for selection in fold["lambda_selection"].values():
            require(set(selection["validation_match_ids"]) == val, "Lambda tuned on wrong matches")
            choices = selection["candidates"]
            require({v["lambda_money"] for v in choices} == set(design["lambda_money_grid"]), "Lambda search grid changed")
            optimum = min(choices, key=lambda v: (v["validation_utility"], -v["lambda_money"]))["lambda_money"]
            close(selection["selected"], optimum, "selected validation lambda")
    rows = list(_rows(run / "episodes.jsonl"))
    require(len(rows) == report["episode_runs"], "Episode count mismatch")
    require(len(all_test_ids) == report["test_matches"], "Test match count mismatch")
    keyed = {tuple(row[k] for k in KEYS): row for row in rows}
    require(len(keyed) == len(rows), "Duplicate episode")
    require(set(row["match_id"] for row in rows) == all_test_ids, "Episode test membership mismatch")
    for row in rows:
        require(row["match_id"] in fold_ids[row["fold"]], "Episode in wrong fold")
        if row["strategy"].startswith("contextual-"):
            fold = next(f for f in report["folds"] if f["fold"] == row["fold"])
            selection = fold["lambda_selection"][f"{row['strategy']}/{row['budget_money']:g}"]
            close(row["lambda_money"], selection["selected"], "episode lambda")
    verify_aggregate_metrics(rows, report["metrics"])
    seen, events, actions, policy_replays = set(), 0, 0, 0
    catalog = read("catalog.json")["actions"]
    for trace in _rows(run / "trajectories.jsonl"):
        key = tuple(trace[k] for k in KEYS)
        require(key in keyed and key not in seen, "Missing or duplicate trace identity")
        seen.add(key)
        row = keyed[key]
        predictor, estimators = models[row["fold"]]
        estimator = estimators.get(row["strategy"])
        counts = verify_episode_trace(row, trace, by_match[row["match_id"]], predictor, catalog,
                                      design["request_budget"], estimator, design["min_support"])
        events += counts["events"]
        actions += counts["actions"]
        policy_replays += int(estimator is not None)
    require(seen == set(keyed), "Incomplete trajectory coverage")
    return {"status": "PASS", "models_loaded": len(report["model_artifacts"]), "episode_runs": len(rows),
            "independent_test_matches": len(all_test_ids), "reconstructed_visible_states": events,
            "reconstructed_acquisition_charges": actions, "numerical_policy_replayed_episodes": policy_replays,
            "source_outcome_losses_recomputed": True, "temporal_splits_checked": True,
            "artifact_hashes_checked": True, "aggregate_metrics_recomputed": True,
            "scope": "offline artifact, source, numerical policy and accounting integrity; no superiority or prospective claim",
            "artifacts": {name: sha256((run / name).read_bytes()).hexdigest() for name in
                          ("report.json", "design.json", "episodes.jsonl", "trajectories.jsonl", "initial-values.jsonl")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.run, args.bundle)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
