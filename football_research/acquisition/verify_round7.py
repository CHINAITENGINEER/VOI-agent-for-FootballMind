"""Independent source, OOF and policy audit of equal-free-context acquisition.

Outcome-bearing records enter this offline audit only. They never enter the
runtime policy or an inference request. Old experiment artifacts stay frozen.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import timedelta
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import random

from .data import ControlledEpisode, load_episodes
from .experiment import losses, outcome, state_after
from .state import parse_time
from .verify_round4 import _rows
from .verify_round6 import (KEYS, close, require, verify_aggregate_metrics,
                            verify_episode_trace)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify_oof_sample(sample, source, predictor, actions, training_ids, cutoff):
    """Rebuild both sides of a supervised transition, including missing sources."""
    require(sample["match_id"] == source["match_id"], "OOF source identity mismatch")
    require(set(sample["oof_train_ids"]) == set(training_ids), "OOF training IDs mismatch")
    require(sample["match_id"] not in training_ids, "OOF self training")
    require(parse_time(sample["oof_train_cutoff"]) == parse_time(cutoff), "OOF cutoff mismatch")
    require(parse_time(cutoff) < parse_time(source["prediction_cutoff"]), "OOF future training")
    for name in ("kickoff", "result_available_at"):
        require(parse_time(sample[name]) == parse_time(source[name]), "OOF source time mismatch: " + name)
    catalog = {a["action_id"]: a for a in actions}
    aid = sample["action"]["action_id"]
    require(aid in catalog and sample["action"] == catalog[aid], "OOF action catalog mismatch")
    ids = list(catalog)
    matches = []
    for count in range(len(ids)):
        for acquired in itertools.combinations(ids, count):
            if aid not in acquired and state_after(source, actions, acquired) == sample["state"]:
                matches.append(acquired)
    require(len(matches) == 1, "OOF state is not one unique permitted reveal")
    acquired = matches[0]
    before = predictor.predict_partial(sample["state"])["probabilities"]
    after = predictor.predict_partial(state_after(source, actions, (*acquired, aid)))["probabilities"]
    if "acquired_action_ids" in sample:
        require(set(sample["acquired_action_ids"]) == set(acquired), "OOF acquired action mismatch")
    if "oof_predictor_version" in sample:
        require(sample["oof_predictor_version"] == predictor.model_version, "OOF predictor version mismatch")
    require(len(sample["prediction_before"]) == 3, "OOF prediction length")
    for actual, expected in zip(sample["prediction_before"], before):
        close(actual, expected, "OOF before probability")
    if "prediction_after" in sample:
        require(len(sample["prediction_after"]) == 3, "OOF after probability length")
        for actual, expected in zip(sample["prediction_after"], after):
            close(actual, expected, "OOF after probability")
    expected = losses(before, outcome(source))["log_loss"] - losses(after, outcome(source))["log_loss"]
    close(sample["delta"], expected, "recomputed OOF delta")
    return source["match_id"], tuple(acquired), aid


def verify_heuristic_trace(row, trace, source, actions, design):
    """Recompute nonlearned policy decisions, separately from the runner."""
    strategy = row["strategy"]
    require(strategy in {"base-only", "all-available", "fixed-order", "random", "entropy"}, "Unknown heuristic")
    env = ControlledEpisode(source, actions)
    rng = random.Random(int(sha256(f"round4|{row['seed']}|{row['match_id']}".encode()).hexdigest()[:16], 16))
    money, requests = 0.0, 0.0
    for event in trace["events"]:
        candidates = env.legal_actions()
        affordable = [a for a in candidates
                      if a["cost"].get("money", math.inf) <= row["budget_money"] - money + 1e-12
                      and a["cost"].get("requests", math.inf) <= design["request_budget"] - requests]
        expected_action = None
        if strategy == "base-only":
            reason = "base_only"
        elif not candidates:
            reason = "no_legal_action"
        elif not affordable:
            reason = "budget"
        elif strategy == "entropy" and -sum(p * math.log(max(p, 1e-15))
                for p in event["prediction"]["probabilities"]) <= design["entropy_threshold"]:
            reason = "entropy_below_threshold"
        else:
            selected = rng.choice(affordable) if strategy == "random" else affordable[0]
            expected_action, reason = selected["action_id"], strategy
        decision = event["decision"]
        require(decision.get("action_id") == expected_action, "Heuristic action mismatch")
        require(decision.get("reason") == reason, "Heuristic reason mismatch")
        require(decision.get("decision") == ("action" if expected_action else "stop"), "Heuristic stopping mismatch")
        if expected_action is None:
            break
        charge = next(a["cost"] for a in candidates if a["action_id"] == expected_action)
        money += charge.get("money", 0.0)
        requests += charge.get("requests", 0.0)
        env.reveal(expected_action)


def verify_episode_coverage(rows, fold_ids, strategies, budgets, seeds):
    """Seed multiplicity may not silently overweight games or drop a strategy."""
    actual = [tuple(r[k] for k in KEYS) for r in rows]
    require(len(set(actual)) == len(actual), "Duplicate episode identity")
    expected = {(fold, match, strategy, budget, seed)
                for fold, ids in fold_ids.items() for match in ids
                for strategy in strategies for budget in budgets
                for seed in (seeds if strategy == "random" else [0])}
    require(set(actual) == expected, "Incomplete episode/seed/budget coverage")


def verify(run, bundle, earlier_bundle):
    from football_research.evaluation.splits import rolling_origin_splits
    from .contextual_identity import SourceIdentityValueEstimator, build_crosswalk, validate_crosswalk
    from .contextual_partial import ContextualPartialPredictor
    from .contextual_value import ContextualValueEstimator
    from .experiment_round4 import run_episode

    run = Path(run)
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    report, design = read("report.json"), read("design.json")
    implementation = read("implementation.json")
    require(implementation == report["implementation_sha256"], "Implementation manifest mismatch")
    for name, value in implementation.items():
        require(digest(Path(__file__).parent / name) == value, "Source changed: " + name)
    for artifact in report["model_artifacts"]:
        require(digest(run / artifact["path"]) == artifact["sha256"], "Model artifact changed")
    crosswalk = read("identity-crosswalk.json")
    validate_crosswalk(crosswalk)
    require(build_crosswalk([earlier_bundle, bundle]) == crosswalk, "Identity mapping differs from frozen source IDs")
    current, current_audit = load_episodes(bundle, lead_hours=design["lead_hours"])
    earlier, earlier_audit = load_episodes(earlier_bundle, lead_hours=design["lead_hours"])
    audit = read("data-audit.json")
    for key, recomputed in (("current", current_audit), ("earlier", earlier_audit)):
        require(audit[key]["platform_manifest_sha256"] == recomputed["platform_manifest_sha256"], "Source export changed")
    sources = {r["match_id"]: r for r in earlier + current}
    require(len(sources) == len(earlier) + len(current) == report["input_matches"], "Input membership/overlap mismatch")
    actions = read("catalog.json")["actions"]
    prior = Path(design["source_round5"])
    if (run / "source-artifacts.json").exists():
        for name, value in read("source-artifacts.json").items():
            require(digest(prior / name) == value, "Reference changed: " + name)
    prior_folds = {f["fold"]: f for f in json.loads((prior / "report.json").read_text(encoding="utf-8"))["folds"]}

    # Numerical labels are recomputed from NEW context-aware OOF predictors.
    oof_manifest = read("oof-manifest.json")
    oof_models, oof_targets = {}, set()
    for fold in oof_manifest["folds"]:
        artifact = Path(fold["artifact"])
        if not artifact.is_file():
            artifact = run / artifact
        model = ContextualPartialPredictor.load(artifact)
        require(digest(artifact) == fold["sha256"], "OOF manifest artifact hash mismatch")
        require(model.model_version == fold["model_version"], "OOF manifest model version mismatch")
        require(model.crosswalk == crosswalk, "OOF free-context identity differs")
        train, target = set(fold["training_match_ids"]), set(fold["prediction_match_ids"])
        require(not train & target and not target & oof_targets, "OOF membership overlap")
        require(train == set(model.metadata["training_match_ids"]), "OOF artifact membership mismatch")
        require(parse_time(model.metadata["training_as_of"]) == parse_time(fold["cutoff"]), "OOF artifact cutoff mismatch")
        require(all(parse_time(sources[mid]["result_available_at"]) <= parse_time(fold["cutoff"]) for mid in train), "OOF future labels")
        require(all(parse_time(fold["cutoff"]) < parse_time(sources[mid]["prediction_cutoff"]) for mid in target), "OOF train after prediction")
        reference = "sha256:" + digest(artifact)
        require(reference not in oof_models, "Duplicate OOF artifact")
        oof_models[reference] = model, fold
        oof_targets.update(target)
    oof_seen, oof_count = set(), 0
    for sample in _rows(run / "oof-supervision.jsonl"):
        require(sample["oof_manifest_ref"] in oof_models, "Unknown OOF model reference")
        model, fold = oof_models[sample["oof_manifest_ref"]]
        require(sample["match_id"] in fold["prediction_match_ids"], "Wrong OOF target chunk")
        key = verify_oof_sample(sample, sources[sample["match_id"]], model, actions,
                                fold["training_match_ids"], fold["cutoff"])
        require(key not in oof_seen, "Duplicate OOF transition")
        oof_seen.add(key)
        oof_count += 1
    require(oof_count == oof_manifest["samples"], "OOF sample count mismatch")
    require({k[0] for k in oof_seen} == oof_targets, "OOF target coverage mismatch")
    expected_transitions = len(actions) * 2 ** (len(actions) - 1)
    require(oof_count == len(oof_targets) * expected_transitions, "Missing OOF transition coverage")

    source_folds = rolling_origin_splits(earlier + current, initial_validation_start=design["validation_start"],
            validation_window=timedelta(days=design.get("validation_days", 30)),
            test_window=timedelta(days=design.get("test_days", 30)))
    predictors, values, fold_ids, all_test_ids = {}, {}, {}, set()
    validation_replays = 0
    for fold in report["folds"]:
        number = fold["fold"]
        train, validation, test = (set(fold[k]) for k in ("training_match_ids", "validation_match_ids", "test_match_ids"))
        require(not (train & validation or train & test or validation & test or test & all_test_ids), "Outer split overlap")
        for key, members in (("training_match_ids", train), ("validation_match_ids", validation), ("test_match_ids", test)):
            require(members == set(prior_folds[number][key]), "Changed explored reference split")
        recomputed_fold = source_folds[number]
        require(train == {r["match_id"] for r in recomputed_fold.train}, "Source train split differs")
        require(validation == {r["match_id"] for r in recomputed_fold.validation}, "Source validation split differs")
        require(test == {r["match_id"] for r in recomputed_fold.test}, "Source test split differs")
        all_test_ids.update(test)
        fold_ids[number] = test
        predictor = ContextualPartialPredictor.load(run / "models" / f"fold-{number}-predictor.json")
        require(predictor.crosswalk == crosswalk, "Outcome free-context identity differs")
        require(set(predictor.metadata["training_match_ids"]) == train, "Outcome predictor membership mismatch")
        require(parse_time(predictor.metadata["training_as_of"]) == recomputed_fold.train_cutoff, "Outcome predictor cutoff mismatch")
        predictors[number] = predictor
        require(set(fold["models"]) == {"no-context", "prediction-only", "source-identity"}, "Missing prespecified value ablation")
        for mode, artifact in fold["models"].items():
            cls = SourceIdentityValueEstimator if mode == "source-identity" else ContextualValueEstimator
            model = cls.load(run / artifact["path"])
            require(model.mode == mode, "Value mode mismatch")
            if mode == "source-identity":
                require(model.crosswalk == crosswalk, "Predictor and acquisition free identities differ")
            mtrain, mcal = (set(model.manifest[k]) for k in ("training_match_ids", "calibration_match_ids"))
            require(mtrain == set(fold["value_training_match_ids"]) and mcal == set(fold["value_calibration_match_ids"]), "Value split mismatch")
            require(not mtrain & mcal and mtrain | mcal <= train & oof_targets, "Value OOF membership leakage")
            require(parse_time(model.manifest["trained_as_of"]) == recomputed_fold.train_cutoff, "Value cutoff mismatch")
            times = model.manifest["temporal_boundary"]
            if times["calibration_min_prediction_cutoff"]:
                require(parse_time(times["train_max_result_available_at"]) < parse_time(times["calibration_min_prediction_cutoff"]), "Calibration time leakage")
            values[number, "contextual-" + mode] = model
        for key, selection in fold["lambda_selection"].items():
            strategy, budget = key.split("/")
            estimator = values[number, strategy]
            require(set(selection["validation_match_ids"]) == validation, "Lambda uses nonvalidation games")
            close(selection["utility_money_weight"], design["validation_utility_money_weight"], "Validation objective weight")
            candidates = selection["candidates"]
            require({c["lambda_money"] for c in candidates} == set(design["lambda_money_grid"]), "Lambda grid differs")
            for candidate in candidates:
                results = [run_episode(sources[mid], predictor, estimator, estimator, actions, strategy="risk-mean",
                        budget_money=float(budget), lambda_money=candidate["lambda_money"], seed=0, design=design)[0]
                        for mid in selection["validation_match_ids"]]
                observed = sum(r["log_loss"] + design["validation_utility_money_weight"] * r["cost"]["money"]
                               for r in results) / len(results)
                close(candidate["validation_utility"], observed, "Recomputed validation utility")
                require(candidate["validation_matches"] == len(validation), "Validation count mismatch")
                validation_replays += len(results)
            close(selection["selected"], min(candidates, key=lambda c: (c["validation_utility"], -c["lambda_money"]))["lambda_money"], "Selected lambda")

    rows = list(_rows(run / "episodes.jsonl"))
    require(len(rows) == report["episode_runs"] and len(all_test_ids) == report["test_matches"], "Episode/test count differs")
    strategies = {"base-only", "all-available", "entropy", "fixed-order", "random",
                  "contextual-no-context", "contextual-prediction-only", "contextual-source-identity"}
    verify_episode_coverage(rows, fold_ids, strategies, design["money_budgets"], design["random_seeds"])
    verify_aggregate_metrics(rows, report["metrics"])
    keyed = {tuple(r[k] for k in KEYS): r for r in rows}
    seen, events, charges, numerical_policies = set(), 0, 0, 0
    for trace in _rows(run / "trajectories.jsonl"):
        key = tuple(trace[k] for k in KEYS)
        require(key in keyed and key not in seen, "Trace membership/duplication mismatch")
        seen.add(key)
        row = keyed[key]
        predictor, estimator = predictors[row["fold"]], values.get((row["fold"], row["strategy"]))
        require(row["model_version"] == predictor.model_version, "Episode predictor version mismatch")
        if estimator is not None:
            fold = next(f for f in report["folds"] if f["fold"] == row["fold"])
            close(row["lambda_money"], fold["lambda_selection"][f"{row['strategy']}/{row['budget_money']:g}"]["selected"], "Episode lambda")
            require(row["gain_version"] == estimator.version, "Episode value version mismatch")
        counts = verify_episode_trace(row, trace, sources[row["match_id"]], predictor, actions,
                                      design["request_budget"], estimator, design["min_support"])
        if estimator is None:
            verify_heuristic_trace(row, trace, sources[row["match_id"]], actions, design)
        events += counts["events"]
        charges += counts["actions"]
        numerical_policies += int(estimator is not None)
    require(seen == set(keyed), "Missing policy trace")
    return {"status": "VERIFIED", "round": 7, "test_matches": len(all_test_ids), "episode_runs": len(rows),
            "oof_models": len(oof_models), "oof_transitions_recomputed": oof_count,
            "model_artifacts": len(report["model_artifacts"]), "visible_states_replayed": events,
            "acquisition_charges_verified": charges, "numerical_policy_episodes_replayed": numerical_policies,
            "heuristic_policy_episodes_replayed": len(rows) - numerical_policies,
            "validation_episodes_recomputed": validation_replays,
            "external_requests": 0, "llm_tokens": 0,
            "checks": ["source export and model/code hashes", "unchanged outer partitions",
                "new free-context OOF predictions and deltas", "train/calibration/test time separation",
                "equal exact source identity mapping", "all budgets and random seeds present",
                "visible states and unpurchased masking", "model forecasts and source outcome scores",
                "all learned and heuristic policy decisions", "validation-only lambda scores and selection"],
            "artifacts": {n: digest(run / n) for n in ("report.json", "design.json", "episodes.jsonl",
                "trajectories.jsonl", "oof-supervision.jsonl", "oof-manifest.json", "identity-crosswalk.json")},
            "verifier_sha256": digest(__file__),
            "scope": "integrity and reproducibility audit, not independent confirmation of predictive benefit"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--earlier-bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result = verify(args.run, args.bundle, args.earlier_bundle)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
