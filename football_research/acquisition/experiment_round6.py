"""Prespecified contextual-value ablations on the frozen round-five experiment.

Reuse the exact 760-match input, forward OOF labels, predictors, and 290 test
matches. The season has already been explored; all comparisons are exploratory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import timedelta
from hashlib import sha256
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from football_research.baselines.api import canonical_json
from football_research.evaluation.splits import rolling_origin_splits
from .contextual_value import ContextualValueEstimator, MODES
from .data import load_episodes
from .experiment import state_after
from .experiment_round4 import DEFAULT_DESIGN, _select_lambda, _sem_time, run_episode, write_json
from .experiment_round5 import split_calibration, diagnostics, calibration_summary
from .partial import PartialPredictor

VERSION = "contextual-identifiability-round6-v1"
REFERENCES = ("base-only", "all-available", "entropy", "risk-mean")


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def summarize(results, design):
    grouped = defaultdict(list)
    for row in results:
        grouped[f"{row['strategy']}/money={row['budget_money']:g}"].append(row)
    metrics, paired = {}, {}
    for key, rows in sorted(grouped.items()):
        metrics[key] = {"independent_matches": len({r["match_id"] for r in rows}), "episodes": len(rows),
            **{metric: float(np.mean([r[metric] for r in rows])) for metric in ("log_loss", "brier_score", "accuracy")},
            **{unit: float(np.mean([r["cost"][unit] for r in rows])) for unit in ("money", "requests")},
            "action_counts": dict(Counter(a for r in rows for a in r["actions"])),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in rows))}
    for budget, mode in itertools.product(design["money_budgets"], MODES):
        strategy = "contextual-"+mode
        current = {r["match_id"]: r for r in grouped[f"{strategy}/money={budget:g}"]}
        for reference in (*REFERENCES, "contextual-no-context", "contextual-prediction-only"):
            if reference == strategy:
                continue
            before = {r["match_id"]: r for r in grouped[f"{reference}/money={budget:g}"]}
            if current.keys() != before.keys():
                raise ValueError("Paired comparisons must use identical test matches")
            records = [(r, before[mid]) for mid, r in current.items()]
            blocks = defaultdict(list)
            for a, b in records:
                year, week, _ = _sem_time(a["kickoff"]).isocalendar()
                blocks[(year, week)].append(a["log_loss"]-b["log_loss"])
            values = list(blocks.values())
            sums, counts = np.asarray([sum(v) for v in values]), np.asarray([len(v) for v in values])
            rng = np.random.default_rng(design["bootstrap_seed"])
            draws = rng.integers(0, len(values), size=(design["bootstrap_samples"], len(values)))
            boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
            # Each match contributes once. An additional match bootstrap is
            # descriptive and does not replace the more conservative week unit.
            delta = np.asarray([a["log_loss"]-b["log_loss"] for a,b in records])
            match_draws = rng.integers(0, len(delta), size=(design["bootstrap_samples"], len(delta)))
            paired[f"{strategy}-vs-{reference}/money={budget:g}"] = {
                "strategy": strategy, "reference": reference, "independent_matches": len(records), "weeks": len(values),
                "mean_log_loss_difference": float(delta.mean()),
                "confidence_interval_95": np.quantile(boot, [.025, .975]).tolist(),
                "match_bootstrap_interval_95": np.quantile(delta[match_draws].mean(axis=1), [.025, .975]).tolist(),
                "mean_brier_difference": float(np.mean([a["brier_score"]-b["brier_score"] for a,b in records])),
                "mean_money_difference": float(np.mean([a["cost"]["money"]-b["cost"]["money"] for a,b in records])),
                "method": "paired UTC-week percentile bootstrap; individual matches appear once",
                "limitations": "already explored test season; no multiplicity, adaptive research, fitted-policy or long-range dependence correction"}
    return metrics, paired


def run(bundle, earlier_bundle, source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    design = {**DEFAULT_DESIGN, "experiment_version": VERSION,
        "classification": "exploratory_previously_seen_history_contextual_ablation", "modes": MODES,
        "regularization": 20.0, "regularization_selection": "fixed_before_test_run_no_test_search",
        "value_fit": "separate_action_mask_ridge_intercept_unpenalized", "references": REFERENCES,
        "source_round5": str(source.resolve()), "new_qualification_features": "not_available_in_frozen_EPL_export_not_used"}
    write_json(output/"design.json", design)
    current, current_audit = load_episodes(bundle, lead_hours=24)
    earlier, earlier_audit = load_episodes(earlier_bundle, lead_hours=24)
    episodes = earlier+current
    if {r["match_id"] for r in current} & {r["match_id"] for r in earlier}:
        raise ValueError("Earlier and current season IDs overlap")
    catalog = json.loads((source/"catalog.json").read_text(encoding="utf-8"))
    source_report = json.loads((source/"report.json").read_text(encoding="utf-8"))
    source_design = json.loads((source/"design.json").read_text(encoding="utf-8"))
    for key in ("money_budgets", "lambda_money_grid", "validation_start", "validation_utility_money_weight", "request_budget"):
        if design[key] != source_design[key]:
            raise ValueError("Frozen round-five design mismatch: "+key)
    actions = catalog["actions"]
    write_json(output/"catalog.json", catalog)
    oof = list(_jsonl(source/"oof-supervision.jsonl"))
    write_json(output/"data-audit.json", {"current": current_audit, "earlier": earlier_audit,
        "input_matches": len(episodes), "oof_source": str((source/"oof-supervision.jsonl").resolve()),
        "oof_sha256": _hash(source/"oof-supervision.jsonl"), "oof_samples": len(oof),
        "new_external_requests": 0, "motivation_lineup_tactical_features": "absent_not_fabricated"})
    source_files = ("contextual_value.py", "experiment_round6.py", "experiment_round5.py", "experiment_round4.py",
                    "partial.py", "policy.py", "state.py", "value.py", "data.py", "risk_calibrated.py", "experiment.py")
    implementation = {name: _hash(Path(__file__).parent/name) for name in source_files}
    write_json(output/"implementation.json", implementation)
    source_artifacts = {name: _hash(source/name) for name in ("report.json", "episodes.jsonl", "trajectories.jsonl", "catalog.json", "oof-supervision.jsonl")}
    write_json(output/"source-artifacts.json", source_artifacts)
    results = [r for r in _jsonl(source/"episodes.jsonl") if r["strategy"] in REFERENCES]
    diagnostics_all, initial_rows, fold_reports = [], [], []
    folds = rolling_origin_splits(episodes, initial_validation_start=design["validation_start"],
        validation_window=timedelta(days=30), test_window=timedelta(days=30))
    with (output/"trajectories.jsonl").open("w", encoding="utf-8") as traces:
        for row in _jsonl(source/"trajectories.jsonl"):
            if row["strategy"] in REFERENCES:
                traces.write(canonical_json(row)+"\n")
        for number, fold in enumerate(folds):
            prior_report = next((r for r in source_report["folds"] if r["fold"] == number), None)
            if prior_report is None:
                continue
            train_ids = {r["match_id"] for r in fold.train}
            for key, values in (("training_match_ids", train_ids), ("validation_match_ids", {r["match_id"] for r in fold.validation}),
                                ("test_match_ids", {r["match_id"] for r in fold.test})):
                if values != set(prior_report[key]):
                    raise ValueError("Frozen fold ID mismatch: "+key)
            print(f"fold {number}: train={len(fold.train)} test={len(fold.test)}", flush=True)
            model_dir = output/"models"
            model_dir.mkdir(exist_ok=True)
            for kind in ("predictor", "risk"):
                shutil.copy2(source/"models"/f"fold-{number}-{kind}.json", model_dir/f"fold-{number}-{kind}.json")
            predictor = PartialPredictor.load(model_dir/f"fold-{number}-predictor.json")
            prior = [s for s in oof if s["match_id"] in train_ids and _sem_time(s["result_available_at"]) <= fold.train_cutoff]
            train, tail, split = split_calibration(prior, design["calibration_fraction"])
            fold_report = {"fold": number, "split": fold.manifest(), "value_split": split,
                "training_match_ids": sorted(train_ids), "validation_match_ids": sorted(r["match_id"] for r in fold.validation),
                "test_match_ids": sorted(r["match_id"] for r in fold.test),
                "value_training_match_ids": sorted({r["match_id"] for r in train}),
                "value_calibration_match_ids": sorted({r["match_id"] for r in tail}),
                "models": {}, "lambda_selection": {}, "initial_value_diagnostics": {}}
            for mode in MODES:
                strategy = "contextual-"+mode
                model = ContextualValueEstimator(mode=mode, regularization=design["regularization"], min_support=design["min_support"],
                    risk_multiplier=design["risk_multiplier"]).fit(train, as_of=fold.train_cutoff,
                    dataset_version=episodes[0]["dataset_version"], calibration_samples=tail,
                    metadata={"fold": number, "oof_source_sha256": source_artifacts["oof-supervision.jsonl"]})
                path = model_dir/f"fold-{number}-{mode}.json"
                model.save(path)
                # Test actual serialization before using the artifact in policy.
                model = ContextualValueEstimator.load(path)
                fold_report["models"][mode] = {"path": str(path.relative_to(output)), "sha256": _hash(path), "version": model.version,
                    "calibration": model.calibration, "manifest": model.manifest}
                selected = {budget: _select_lambda(fold.validation, predictor, model, model, actions, "risk-mean", budget, design)
                            for budget in design["money_budgets"]}
                fold_report["lambda_selection"].update({f"{strategy}/{budget:g}": record for budget, (_, record) in selected.items()})
                vectors, unknown_matches, team_total, unknown_teams = set(), 0, 0, 0
                for row in fold.test:
                    state = state_after(row, actions, ())
                    prediction = predictor.predict_partial(state)
                    estimates = model.estimate_value(state, actions, prediction=prediction)
                    vector = tuple(round(e["expected_gain"], 12) if e["expected_gain"] is not None else None for e in estimates)
                    vectors.add(vector)
                    unknown = set(k for e in estimates for k in e["unknown_identity_fields"])
                    unknown_matches += int(bool(unknown))
                    if mode == "visible-context":
                        team_total += 2
                        unknown_teams += len(unknown & {"home_team_id", "away_team_id"})
                    initial_rows.append({"fold": number, "match_id": row["match_id"], "strategy": strategy,
                        "initial_gain_vector": list(vector), "unknown_identity_fields": sorted(unknown),
                        "estimator_version": model.version, "estimates": estimates})
                    for budget in design["money_budgets"]:
                        result, trace = run_episode(row, predictor, model, model, actions, strategy="risk-mean",
                            budget_money=budget, lambda_money=selected[budget][0], seed=0, design=design)
                        result.update(strategy=strategy, fold=number)
                        results.append(result)
                        traces.write(canonical_json({"fold": number, "match_id": row["match_id"], "strategy": strategy,
                            "budget_money": budget, "seed": 0, "events": trace})+"\n")
                diag = diagnostics(fold.test, predictor, model, actions)
                diagnostics_all.extend({**r, "fold": number, "strategy": strategy} for r in diag)
                fold_report["initial_value_diagnostics"][mode] = {"matches": len(fold.test), "distinct_gain_vectors_rounded_12dp": len(vectors),
                    "unknown_identity_matches": unknown_matches, "unknown_team_slots": unknown_teams, "team_slots": team_total,
                    "unknown_team_fraction": unknown_teams/team_total if team_total else None,
                    "in_training_team_fraction": 1-unknown_teams/team_total if team_total else None,
                    "scope": "known identity in same role in value-training vocabulary; no causal or current-strength interpretation"}
                print(f"  {mode}: {len(vectors)} initial vectors", flush=True)
            fold_reports.append(fold_report)
            write_json(output/"fold-progress.json", fold_reports)
    metrics, comparisons = summarize(results, design)
    artifacts = [{"path": str(path.relative_to(output)), "sha256": _hash(path)} for path in sorted((output/"models").glob("*.json"))]
    report = {"status": "OK", "experiment_version": VERSION, "evidence_grade": design["classification"],
        "input_matches": len(episodes), "test_matches": len({r["match_id"] for r in results}), "episode_runs": len(results),
        "new_policy_episode_runs": sum(r["strategy"].startswith("contextual-") for r in results),
        "reference_episode_source": "exact frozen round5 rows and traces, not retrained", "folds": fold_reports,
        "metrics": metrics, "paired_comparisons": comparisons, "model_artifacts": artifacts,
        "risk_calibration": {mode: calibration_summary([r for r in diagnostics_all if r["strategy"] == "contextual-"+mode]) for mode in MODES},
        "source_artifacts_sha256": source_artifacts, "implementation_sha256": implementation,
        "external_requests": 0, "llm_tokens": 0, "cost_interpretation": design["cost_interpretation"],
        "limitations": ["Already-explored 290-match test season; no confirmatory claim", "Value variation alone does not establish benefit",
            "Ridge bias/misspecification not included in descriptive mean SE", "Predictive interval coverage is not mean CI coverage",
            "Unknown team effects fall back to zero; identity effects can become stale", "No group-qualification/lineup/tactical features in EPL export",
            "Only two simulated evidence actions, not independent provider requests", "Empty-mask outcome predictor still uses same fold prior",
            "Free fixture identity learns historical association; no direct causal rotation or motivation inference",
            "Fixed ridge alpha chosen before test run; lambda chosen only on temporal validation"]}
    write_json(output/"report.json", report)
    for name, records in (("episodes", results), ("risk-diagnostics", diagnostics_all), ("initial-values", initial_rows)):
        (output/(name+".jsonl")).write_text("".join(canonical_json(r)+"\n" for r in records), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--earlier-bundle", required=True)
    parser.add_argument("--source-round5", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.bundle, args.earlier_bundle, args.source_round5, args.output)
    print(json.dumps({k: report[k] for k in ("status", "test_matches", "episode_runs", "new_policy_episode_runs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
