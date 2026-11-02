"""Corrected-identity sensitivity, preserving all prespecified round-six results."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import timedelta
import json
from pathlib import Path
import shutil

import numpy as np

from football_research.baselines.api import canonical_json
from football_research.evaluation.splits import rolling_origin_splits
from .contextual_identity import SourceIdentityValueEstimator, build_crosswalk
from .data import load_episodes
from .experiment import state_after
from .experiment_round4 import _select_lambda, _sem_time, run_episode, write_json
from .experiment_round5 import split_calibration, diagnostics, calibration_summary
from .experiment_round6 import _hash, _jsonl, summarize
from .partial import PartialPredictor

VERSION = "round6-exact-source-identity-sensitivity-v1"
STRATEGY = "contextual-source-identity"


def identity_comparisons(results, design):
    output = {}
    references = sorted({r["strategy"] for r in results}-{STRATEGY})
    for budget in design["money_budgets"]:
        current = {r["match_id"]: r for r in results if r["strategy"] == STRATEGY and r["budget_money"] == budget}
        for reference in references:
            before = {r["match_id"]: r for r in results if r["strategy"] == reference and r["budget_money"] == budget}
            if current.keys() != before.keys():
                raise ValueError("Comparison test IDs differ")
            records, blocks = [], defaultdict(list)
            for mid, row in current.items():
                old = before[mid]
                delta = row["log_loss"]-old["log_loss"]
                year, week, _ = _sem_time(row["kickoff"]).isocalendar()
                blocks[(year, week)].append(delta)
                records.append((row, old))
            values = list(blocks.values())
            sums, counts = np.asarray([sum(v) for v in values]), np.asarray([len(v) for v in values])
            rng = np.random.default_rng(design["bootstrap_seed"])
            draws = rng.integers(0, len(values), size=(design["bootstrap_samples"], len(values)))
            boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
            delta = np.asarray([a["log_loss"]-b["log_loss"] for a,b in records])
            match_draws = rng.integers(0, len(delta), size=(design["bootstrap_samples"], len(delta)))
            output[f"{STRATEGY}-vs-{reference}/money={budget:g}"] = {
                "strategy": STRATEGY, "reference": reference, "independent_matches": len(records), "weeks": len(values),
                "mean_log_loss_difference": float(delta.mean()), "confidence_interval_95": np.quantile(boot, [.025,.975]).tolist(),
                "match_bootstrap_interval_95": np.quantile(delta[match_draws].mean(axis=1), [.025,.975]).tolist(),
                "mean_brier_difference": float(np.mean([a["brier_score"]-b["brier_score"] for a,b in records])),
                "mean_money_difference": float(np.mean([a["cost"]["money"]-b["cost"]["money"] for a,b in records])),
                "method": "paired UTC-week percentile bootstrap; individual matches appear once",
                "limitations": "post-audit identity-correction sensitivity on previously explored test season; no multiplicity or fitted-policy correction"}
    return output


def run(bundle, earlier_bundle, source_round6, output):
    source, output = Path(source_round6), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((source/"report.json").read_text(encoding="utf-8"))
    design = json.loads((source/"design.json").read_text(encoding="utf-8"))
    source_round5 = Path(design["source_round5"])
    design.update(experiment_version=VERSION, classification="exploratory_post_audit_exact_source_identity_sensitivity",
        source_round6=str(source.resolve()), additional_strategy=STRATEGY,
        sensitivity_rationale="same provider IDs had different canonical UUIDs across export snapshots; corrected without target outcomes",
        comparison_rule="preserve v2 as original primary exploratory ablations; append correction regardless of performance")
    write_json(output/"design.json", design)
    crosswalk = build_crosswalk([earlier_bundle, bundle])
    write_json(output/"identity-crosswalk.json", crosswalk)
    write_json(output/"identity-source-artifacts.json", {"round6_source": str(source.resolve()),
        "round6_files": {name: _hash(source/name) for name in ("report.json", "episodes.jsonl", "trajectories.jsonl")},
        "crosswalk_version": crosswalk["version"], "mapping_inputs": crosswalk["sources"]})
    for name in ("catalog.json", "source-artifacts.json", "data-audit.json"):
        shutil.copy2(source/name, output/name)
    shutil.copytree(source/"models", output/"models")
    implementation = json.loads((source/"implementation.json").read_text(encoding="utf-8"))
    implementation.update({name: _hash(Path(__file__).parent/name) for name in ("contextual_identity.py", "experiment_round6_identity.py")})
    write_json(output/"implementation.json", implementation)
    episodes, _ = load_episodes(earlier_bundle, lead_hours=24)
    current, _ = load_episodes(bundle, lead_hours=24)
    episodes += current
    actions = json.loads((source/"catalog.json").read_text(encoding="utf-8"))["actions"]
    oof = list(_jsonl(source_round5/"oof-supervision.jsonl"))
    results, initial_rows, diagnostic_rows = list(_jsonl(source/"episodes.jsonl")), list(_jsonl(source/"initial-values.jsonl")), list(_jsonl(source/"risk-diagnostics.jsonl"))
    fold_reports = {r["fold"]: r for r in report["folds"]}
    folds = rolling_origin_splits(episodes, initial_validation_start=design["validation_start"],
        validation_window=timedelta(days=30), test_window=timedelta(days=30))
    shutil.copy2(source/"trajectories.jsonl", output/"trajectories.jsonl")
    with (output/"trajectories.jsonl").open("a", encoding="utf-8") as traces:
        for number, fold in enumerate(folds):
            if number not in fold_reports:
                continue
            print(f"identity sensitivity fold {number}: {len(fold.test)} test matches", flush=True)
            fold_report = fold_reports[number]
            train_ids = {r["match_id"] for r in fold.train}
            if train_ids != set(fold_report["training_match_ids"]) or {r["match_id"] for r in fold.test} != set(fold_report["test_match_ids"]):
                raise ValueError("Frozen split changed")
            predictor = PartialPredictor.load(output/"models"/f"fold-{number}-predictor.json")
            prior = [s for s in oof if s["match_id"] in train_ids and _sem_time(s["result_available_at"]) <= fold.train_cutoff]
            train, tail, _ = split_calibration(prior, design["calibration_fraction"])
            model = SourceIdentityValueEstimator(crosswalk, regularization=design["regularization"], min_support=design["min_support"],
                risk_multiplier=design["risk_multiplier"]).fit(train, as_of=fold.train_cutoff,
                dataset_version=episodes[0]["dataset_version"], calibration_samples=tail,
                metadata={"fold": number, "classification": design["classification"], "oof_source_sha256": _hash(source_round5/"oof-supervision.jsonl")})
            path = output/"models"/f"fold-{number}-source-identity.json"
            model.save(path)
            model = SourceIdentityValueEstimator.load(path)
            fold_report["models"]["source-identity"] = {"path": str(path.relative_to(output)), "sha256": _hash(path), "version": model.version,
                "manifest": model.manifest, "calibration": model.calibration}
            selected = {budget: _select_lambda(fold.validation, predictor, model, model, actions, "risk-mean", budget, design)
                        for budget in design["money_budgets"]}
            fold_report["lambda_selection"].update({f"{STRATEGY}/{budget:g}": info for budget, (_, info) in selected.items()})
            vectors, unknown_team_slots, unknown_matches, unmapped_fields = set(), 0, 0, 0
            for row in fold.test:
                state = state_after(row, actions, ())
                prediction = predictor.predict_partial(state)
                estimates = model.estimate_value(state, actions, prediction=prediction)
                vector = tuple(round(e["expected_gain"], 12) if e["expected_gain"] is not None else None for e in estimates)
                vectors.add(vector)
                unknown = {k for e in estimates for k in e["unknown_identity_fields"]}
                unmapped = {k for e in estimates for k in e["identity_projection"]["unmapped_fields"]}
                unknown_matches += int(bool(unknown))
                unknown_team_slots += len(unknown & {"home_team_id", "away_team_id"})
                unmapped_fields += len(unmapped)
                initial_rows.append({"fold": number, "match_id": row["match_id"], "strategy": STRATEGY, "initial_gain_vector": list(vector),
                    "unknown_identity_fields": sorted(unknown), "unmapped_identity_fields": sorted(unmapped),
                    "estimator_version": model.version, "estimates": estimates})
                for budget in design["money_budgets"]:
                    result, trace = run_episode(row, predictor, model, model, actions, strategy="risk-mean", budget_money=budget,
                        lambda_money=selected[budget][0], seed=0, design=design)
                    result.update(strategy=STRATEGY, fold=number)
                    results.append(result)
                    traces.write(canonical_json({"fold": number, "match_id": row["match_id"], "strategy": STRATEGY,
                        "budget_money": budget, "seed": 0, "events": trace})+"\n")
            fold_report["initial_value_diagnostics"]["source-identity"] = {"matches": len(fold.test), "distinct_gain_vectors_rounded_12dp": len(vectors),
                "unknown_identity_matches": unknown_matches, "unknown_team_slots": unknown_team_slots, "team_slots": 2*len(fold.test),
                "unknown_team_fraction": unknown_team_slots/(2*len(fold.test)), "in_training_team_fraction": 1-unknown_team_slots/(2*len(fold.test)),
                "unmapped_identity_fields": unmapped_fields, "scope": "train-only provider identity vocabulary; exact crosswalk is offline reconstruction"}
            diagnostic_rows.extend({**r, "fold": number, "strategy": STRATEGY} for r in diagnostics(fold.test, predictor, model, actions))
            write_json(output/"fold-progress.json", list(fold_reports.values()))
    metrics, comparisons = summarize(results, design)
    comparisons.update(identity_comparisons(results, design))
    report.update(status="OK", experiment_version=VERSION, evidence_grade=design["classification"],
        episode_runs=len(results), new_policy_episode_runs=report["new_policy_episode_runs"]+sum(r["strategy"] == STRATEGY for r in results),
        additional_identity_policy_episode_runs=sum(r["strategy"] == STRATEGY for r in results),
        folds=list(fold_reports.values()), metrics=metrics, paired_comparisons=comparisons,
        model_artifacts=[{"path": str(path.relative_to(output)), "sha256": _hash(path)} for path in sorted((output/"models").glob("*.json"))],
        implementation_sha256=implementation, identity_crosswalk_version=crosswalk["version"],
        original_prespecified_result=str((source/"report.json").resolve()))
    report["risk_calibration"]["source-identity"] = calibration_summary([r for r in diagnostic_rows if r["strategy"] == STRATEGY])
    report["limitations"] += ["Identity correction is a post-audit sensitivity, not a new independent confirmation", "Exact source-ID crosswalk is reconstructed offline, not proof of historical verification"]
    write_json(output/"report.json", report)
    for name, records in (("episodes", results), ("initial-values", initial_rows), ("risk-diagnostics", diagnostic_rows)):
        (output/(name+".jsonl")).write_text("".join(canonical_json(row)+"\n" for row in records), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("bundle", "earlier-bundle", "source-round6", "output"):
        parser.add_argument("--"+argument, required=True)
    args = parser.parse_args()
    report = run(args.bundle, args.earlier_bundle, args.source_round6, args.output)
    print(json.dumps({k: report[k] for k in ("status", "test_matches", "episode_runs", "additional_identity_policy_episode_runs")}))
