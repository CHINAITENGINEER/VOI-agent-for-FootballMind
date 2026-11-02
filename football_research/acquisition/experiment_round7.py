"""Equal-free-context active information experiment; all OOF gains recomputed.

Outcome predictors for every strategy and acquisition mask receive exactly the
same legal fixture identity/calendar context. All rounds remain exploratory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import timedelta
import itertools
import json
from pathlib import Path

import numpy as np

from football_research.baselines.api import canonical_json
from football_research.evaluation.splits import rolling_origin_splits
from .contextual_identity import SourceIdentityValueEstimator, build_crosswalk
from .contextual_partial import ContextualPartialPredictor
from .contextual_value import ContextualValueEstimator
from .data import load_episodes
from .experiment import DEFAULT_DESIGN as OOF_DESIGN, losses, outcome, state_after
from .experiment_round4 import DEFAULT_DESIGN, _select_lambda, _sem_time, run_episode, write_json
from .experiment_round5 import split_calibration, diagnostics, calibration_summary
from .experiment_round6 import _hash

VERSION = "equal-free-context-recomputed-oof-round7-v1"
MODES = ("no-context", "prediction-only", "source-identity")
STRATEGIES = ("base-only", "all-available", "entropy", "fixed-order", "random",
              "contextual-no-context", "contextual-prediction-only", "contextual-source-identity")


def forward_oof(rows, actions, design, crosswalk, output):
    """Fit every delta-generating predictor anew under the new free context."""
    rows = sorted(rows, key=lambda r: (_sem_time(r["kickoff"]), r["match_id"]))
    samples, chunks = [], []
    ids = [a["action_id"] for a in actions]
    for start in range(design["oof_warmup"], len(rows), design["oof_chunk_matches"]):
        targets = rows[start:start+design["oof_chunk_matches"]]
        cutoff = min(_sem_time(r["prediction_cutoff"]) for r in targets)-timedelta(microseconds=1)
        train = [r for r in rows if _sem_time(r["result_available_at"]) < cutoff and _sem_time(r["kickoff"]) < cutoff]
        if len(train) < design["min_train_matches"]:
            continue
        model = ContextualPartialPredictor(crosswalk).fit(train, as_of=cutoff, dataset_version=rows[0]["dataset_version"],
                                                       regularization=design["regularization"])
        artifact = output/"oof-models"/f"chunk-{start}.json"
        model.save(artifact)
        train_ids = [r["match_id"] for r in train]
        digest = _hash(artifact)
        chunks.append({"artifact": str(artifact.relative_to(output)), "sha256": digest, "cutoff": cutoff.isoformat(),
                       "model_version": model.model_version, "training_match_ids": train_ids,
                       "prediction_match_ids": [r["match_id"] for r in targets]})
        for row in targets:
            cached = {}
            for count in range(len(ids)+1):
                for acquired in itertools.combinations(ids, count):
                    public = state_after(row, actions, acquired)
                    cached[frozenset(acquired)] = (public, model.predict_partial(public)["probabilities"])
            for acquired, (state, before) in cached.items():
                for action in actions:
                    if action["action_id"] in acquired:
                        continue
                    after = cached[acquired | {action["action_id"]}][1]
                    samples.append({"match_id": row["match_id"], "kickoff": row["kickoff"],
                        "result_available_at": row["result_available_at"], "state": state, "action": action,
                        "prediction_before": before, "prediction_after": after,
                        "acquired_action_ids": sorted(acquired),
                        "delta": losses(before,outcome(row))["log_loss"]-losses(after,outcome(row))["log_loss"],
                        "oof_train_cutoff": cutoff.isoformat(), "oof_train_ids": train_ids,
                        "oof_manifest_ref": "sha256:"+digest, "oof_predictor_version": model.model_version})
        print(f"OOF chunk {start}: train={len(train)} targets={len(targets)}", flush=True)
    write_json(output/"oof-manifest.json", {"folds": chunks, "samples": len(samples),
        "independent_matches": len({r["match_id"] for r in samples}),
        "training_target": "new_contextual_predictor_forward_oof_loss_before_minus_after; negative gains preserved",
        "old_oof_reused": False})
    (output/"oof-supervision.jsonl").write_text("".join(canonical_json(r)+"\n" for r in samples), encoding="utf-8")
    return samples


def summarize(results, design):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in results:
        grouped[(row["strategy"],row["budget_money"])][row["match_id"]].append(row)
    means, metrics = {}, {}
    for (strategy,budget), matches in grouped.items():
        per_match = {}
        for mid, rows in matches.items():
            per_match[mid] = {metric: float(np.mean([r[metric] for r in rows])) for metric in ("log_loss","brier_score","accuracy")}
            per_match[mid].update({unit: float(np.mean([r["cost"][unit] for r in rows])) for unit in ("money","requests")})
            per_match[mid]["kickoff"] = rows[0]["kickoff"]
            per_match[mid]["utility_loss"] = per_match[mid]["log_loss"]+design["validation_utility_money_weight"]*per_match[mid]["money"]
        means[(strategy,budget)] = per_match
        records = [r for rows in matches.values() for r in rows]
        metrics[f"{strategy}/money={budget:g}"] = {"independent_matches": len(per_match), "episodes": len(records),
            **{metric: float(np.mean([r[metric] for r in per_match.values()])) for metric in ("log_loss","brier_score","accuracy","money","requests","utility_loss")},
            "stop_reasons": dict(Counter(r["stop_reason"] for r in records)),
            "action_counts": dict(Counter(a for r in records for a in r["actions"])),
            "seed_metrics": {str(seed): {metric: float(np.mean([r[metric] for r in records if r["seed"] == seed]))
                              for metric in ("log_loss","brier_score")} for seed in sorted({r["seed"] for r in records})},
            "aggregation": "average seeds within each match first, then equal weight matches"}
    comparisons = {}
    for budget in design["money_budgets"]:
        for strategy in STRATEGIES:
            current = means[(strategy,budget)]
            refs = ("base-only",) if not strategy.startswith("contextual-") else tuple(s for s in STRATEGIES if s != strategy)
            for reference in refs:
                if strategy == reference:
                    continue
                prior = means[(reference,budget)]
                if current.keys() != prior.keys():
                    raise ValueError("Paired match sets differ")
                blocks = defaultdict(list)
                for mid in sorted(current):
                    now, old = current[mid], prior[mid]
                    year, week, _ = _sem_time(now["kickoff"]).isocalendar()
                    blocks[(year,week)].append([now[k]-old[k] for k in ("log_loss","brier_score","money","utility_loss")])
                arrays = [np.asarray(rows) for rows in blocks.values()]
                sums, counts = np.asarray([a.sum(axis=0) for a in arrays]), np.asarray([len(a) for a in arrays])
                draws = np.random.default_rng(design["bootstrap_seed"]).integers(0,len(arrays),size=(design["bootstrap_samples"],len(arrays)))
                boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
                point = sums.sum(axis=0)/counts.sum()
                comparisons[f"{strategy}-vs-{reference}/money={budget:g}"] = {
                    "strategy": strategy, "reference": reference, "independent_matches": int(counts.sum()), "weeks": len(arrays),
                    "mean_log_loss_difference": float(point[0]), "confidence_interval_95": np.quantile(boot[:,0],[.025,.975]).tolist(),
                    "mean_brier_difference": float(point[1]), "mean_money_difference": float(point[2]),
                    "mean_utility_loss_difference": float(point[3]), "utility_confidence_interval_95": np.quantile(boot[:,3],[.025,.975]).tolist(),
                    "method": "paired UTC-week percentile bootstrap; seeds averaged within match before bootstrap",
                    "limitations": "already explored season; no multiplicity, adaptive-research, fitted-policy or long-term dependence correction"}
    return metrics, comparisons


def run(bundle, earlier_bundle, source_round5, output):
    output, source = Path(output), Path(source_round5)
    output.mkdir(parents=True, exist_ok=False)
    config = {**DEFAULT_DESIGN, "experiment_version": VERSION, "classification": "exploratory_equal_free_context_previously_seen_history",
        "strategies": STRATEGIES, "modes": MODES, "outcome_regularization": 10., "value_regularization": 20.,
        "outcome_model": "contextual-partial-v1", "free_context_all_strategies": "exact source identity, kickoff calendar and lead time",
        "source_round5": str(source.resolve()), "source_round5_usage": "catalog and outer split identity only; no predictions/deltas/trajectories copied",
        "oof_recompute": True, "oof_design": OOF_DESIGN,
        "hyperparameters": "fixed before test; lambda selected only on temporal validation, no test parameter search"}
    write_json(output/"design.json", config)
    earlier, earlier_audit = load_episodes(earlier_bundle, lead_hours=24)
    current, current_audit = load_episodes(bundle, lead_hours=24)
    if {r["match_id"] for r in earlier} & {r["match_id"] for r in current}:
        raise ValueError("Duplicate matches across exports")
    episodes = earlier+current
    crosswalk = build_crosswalk([earlier_bundle,bundle])
    write_json(output/"identity-crosswalk.json", crosswalk)
    write_json(output/"data-audit.json", {"current": current_audit, "earlier": earlier_audit, "input_matches": len(episodes),
        "oof_reused": False, "old_predictors_reused": False, "historical_group_motivation_features": "absent_not_fabricated"})
    catalog = json.loads((source/"catalog.json").read_text(encoding="utf-8"))
    actions = catalog["actions"]
    write_json(output/"catalog.json", catalog)
    source_report = json.loads((source/"report.json").read_text(encoding="utf-8"))
    source_artifacts = {name: _hash(source/name) for name in ("catalog.json","report.json","design.json")}
    write_json(output/"source-artifacts.json", source_artifacts)
    names = ("contextual_partial.py","experiment_round7.py","contextual_value.py","contextual_identity.py","experiment_round5.py",
             "experiment_round4.py","experiment_round6.py","experiment.py","policy.py","state.py","value.py","data.py","risk_calibrated.py")
    implementation = {name: _hash(Path(__file__).parent/name) for name in names}
    write_json(output/"implementation.json", implementation)
    oof_config = {**OOF_DESIGN, "regularization": config["outcome_regularization"]}
    oof = forward_oof(episodes,actions,oof_config,crosswalk,output)
    folds = rolling_origin_splits(episodes, initial_validation_start=config["validation_start"], validation_window=timedelta(days=30), test_window=timedelta(days=30))
    results, diagnostic_rows, initial_rows, fold_reports = [], [], [], []
    with (output/"trajectories.jsonl").open("w",encoding="utf-8") as traces:
        for number,fold in enumerate(folds):
            old = next((r for r in source_report["folds"] if r["fold"] == number),None)
            if old is None:
                continue
            for key,rows in (("training_match_ids",fold.train),("validation_match_ids",fold.validation),("test_match_ids",fold.test)):
                if {r["match_id"] for r in rows} != set(old[key]):
                    raise ValueError("Frozen outer split mismatch: "+key)
            print(f"outer fold {number}: train={len(fold.train)} test={len(fold.test)}",flush=True)
            predictor = ContextualPartialPredictor(crosswalk).fit(fold.train,as_of=fold.train_cutoff,dataset_version=episodes[0]["dataset_version"],regularization=config["outcome_regularization"])
            predictor_path = output/"models"/f"fold-{number}-predictor.json"
            predictor.save(predictor_path)
            predictor = ContextualPartialPredictor.load(predictor_path)
            train_ids = {r["match_id"] for r in fold.train}
            prior = [s for s in oof if s["match_id"] in train_ids and _sem_time(s["result_available_at"]) <= fold.train_cutoff]
            train,tail,split = split_calibration(prior,config["calibration_fraction"])
            fold_report = {"fold": number,"split": fold.manifest(),"value_split": split,
                "training_match_ids": sorted(train_ids),"validation_match_ids": sorted(r["match_id"] for r in fold.validation),
                "test_match_ids": sorted(r["match_id"] for r in fold.test),"value_training_match_ids": sorted({r["match_id"] for r in train}),
                "value_calibration_match_ids": sorted({r["match_id"] for r in tail}),"models": {},"lambda_selection": {},"initial_value_diagnostics": {},
                "predictor": {"path": str(predictor_path.relative_to(output)),"sha256": _hash(predictor_path),"version": predictor.model_version,"metadata": predictor.metadata}}
            models,selected = {},{}
            for mode in MODES:
                model = (SourceIdentityValueEstimator(crosswalk,regularization=config["value_regularization"],min_support=config["min_support"])
                         if mode == "source-identity" else ContextualValueEstimator(mode=mode,regularization=config["value_regularization"],min_support=config["min_support"]))
                model.fit(train,as_of=fold.train_cutoff,dataset_version=episodes[0]["dataset_version"],calibration_samples=tail,
                    metadata={"fold": number,"oof_sha256": _hash(output/"oof-supervision.jsonl"),"outcome_model": predictor.version})
                path = output/"models"/f"fold-{number}-{mode}.json"
                model.save(path)
                models[mode] = type(model).load(path)
                fold_report["models"][mode] = {"path": str(path.relative_to(output)),"sha256": _hash(path),"version": model.version,"manifest": model.manifest,"calibration": model.calibration}
                for budget in config["money_budgets"]:
                    selected[(mode,budget)] = _select_lambda(fold.validation,predictor,model,model,actions,"risk-mean",budget,config)
                    fold_report["lambda_selection"][f"contextual-{mode}/{budget:g}"] = selected[(mode,budget)][1]
            initial_p,unknown_slots = set(),0
            vectors = {mode:set() for mode in MODES}
            for row in fold.test:
                state = state_after(row,actions,())
                before = predictor.predict_partial(state)
                initial_p.add(tuple(round(v,12) for v in before["probabilities"]))
                unknown_slots += len(set(before["unknown_identity_fields"]) & {"home_team_id","away_team_id"})
                for mode,model in models.items():
                    estimates = model.estimate_value(state,actions,prediction=before)
                    vector = tuple(round(e["expected_gain"],12) if e["expected_gain"] is not None else None for e in estimates)
                    vectors[mode].add(vector)
                    initial_rows.append({"fold": number,"match_id": row["match_id"],"strategy": "contextual-"+mode,
                        "prediction": before,"initial_gain_vector": list(vector),"estimates": estimates})
                for budget,strategy in itertools.product(config["money_budgets"],STRATEGIES):
                    mode = strategy.removeprefix("contextual-") if strategy.startswith("contextual-") else "source-identity"
                    model = models[mode]
                    lam = selected[(mode,budget)][0] if strategy.startswith("contextual-") else 0.
                    for seed in config["random_seeds"] if strategy == "random" else [0]:
                        result,trace = run_episode(row,predictor,model,model,actions,
                            strategy="risk-mean" if strategy.startswith("contextual-") else strategy,
                            budget_money=budget,lambda_money=lam,seed=seed,design=config)
                        result.update(strategy=strategy,fold=number)
                        results.append(result)
                        traces.write(canonical_json({"fold": number,"match_id": row["match_id"],"strategy": strategy,
                            "budget_money": budget,"seed": seed,"events": trace})+"\n")
            for mode,model in models.items():
                fold_report["initial_value_diagnostics"][mode] = {"matches": len(fold.test),"distinct_gain_vectors_rounded_12dp": len(vectors[mode])}
                diagnostic_rows.extend({**r,"fold": number,"strategy": "contextual-"+mode} for r in diagnostics(fold.test,predictor,model,actions))
            fold_report["initial_prediction_diagnostics"] = {"distinct_probability_vectors_rounded_12dp": len(initial_p),"matches": len(fold.test),
                "unknown_team_slots": unknown_slots,"team_slots": 2*len(fold.test),"unknown_team_fraction": unknown_slots/(2*len(fold.test))}
            fold_reports.append(fold_report)
            write_json(output/"fold-progress.json",fold_reports)
    metrics,comparisons = summarize(results,config)
    artifacts = [{"path": str(path.relative_to(output)),"sha256": _hash(path)} for directory in ("models","oof-models") for path in sorted((output/directory).glob("*.json"))]
    report = {"status": "OK","experiment_version": VERSION,"evidence_grade": config["classification"],"input_matches": len(episodes),
        "test_matches": len({r["match_id"] for r in results}),"episode_runs": len(results),"folds": fold_reports,"metrics": metrics,
        "paired_comparisons": comparisons,"model_artifacts": artifacts,"implementation_sha256": implementation,
        "source_artifacts_sha256": source_artifacts,"oof_sha256": _hash(output/"oof-supervision.jsonl"),"old_oof_reused": False,
        "risk_calibration": {mode: calibration_summary([r for r in diagnostic_rows if r["strategy"] == "contextual-"+mode]) for mode in MODES},
        "cost_interpretation": config["cost_interpretation"],"external_requests": 0,"llm_tokens": 0,
        "limitations": ["Same historical 290-match test season used in earlier rounds; exploratory only",
            "All strategies and every outcome-model mask now receive same free identity/calendar context",
            "All OOF outcome models and gains recomputed; old round scores are different predictive problems",
            "Two simulated projection actions, not two independent provider API bills",
            "Exact provider identity is offline reconstructed metadata, not historical verification",
            "Unknown identities fall back to zero coefficients; no unseen-team calibration guarantee",
            "No real qualification/rotation/lineup/tactic inputs in this EPL export",
            "No finite-sample, adaptive-stopping or conditional interval coverage guarantee",
            "Negative realized gain permitted; fixed predictor misspecification can make acquired evidence harmful",
            "Five random seeds do not increase independent match sample size"]}
    write_json(output/"report.json",report)
    for name,rows in (("episodes",results),("initial-values",initial_rows),("risk-diagnostics",diagnostic_rows)):
        (output/(name+".jsonl")).write_text("".join(canonical_json(r)+"\n" for r in rows),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("bundle","earlier-bundle","source-round5","output"):
        parser.add_argument("--"+argument,required=True)
    args = parser.parse_args()
    report = run(args.bundle,args.earlier_bundle,args.source_round5,args.output)
    print(json.dumps({k:report[k] for k in ("status","input_matches","test_matches","episode_runs")}))
