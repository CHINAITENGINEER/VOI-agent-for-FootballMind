"""Round-five recovery: temporal calibration and independent cost/risk audits.

No external requests. All costs are simulated. Earlier-season training is an
exploratory sensitivity analysis; already examined test matches stay exploratory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import timedelta
from hashlib import sha256
import itertools
import json
from pathlib import Path

import numpy as np

from football_research.baselines.api import canonical_json
from football_research.evaluation.splits import rolling_origin_splits
from .data import load_episodes
from .experiment import DEFAULT_DESIGN as OOF_DESIGN, forward_oof, losses, outcome, state_after
from .experiment_round4 import (DEFAULT_DESIGN, STRATEGIES, _select_lambda, _summarize,
    _sem_time, run_episode as legacy_run_episode, write_json)
from .lookahead_calibrated import select_two_step_v2
from .partial import PartialPredictor
from .value import ValueEstimator
from .risk_calibrated import CalibratedRiskAwareValueEstimator

VERSION = "risk-calibration-separated-intervals-round5-v1"


def run_episode(*args, **kwargs):
    return legacy_run_episode(*args, **kwargs, lookahead=select_two_step_v2)


def make_plots(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for strategy in STRATEGIES:
        rows = sorted((v for k,v in report['metrics'].items() if k.startswith(strategy+'/')), key=lambda r:r['money'])
        for axis, metric in zip(axes, ('log_loss','brier_score')):
            axis.plot([r['money'] for r in rows], [r[metric] for r in rows], 'o-', label=strategy)
            axis.set(xlabel='Mean simulated cost', ylabel=metric)
    axes[1].legend(fontsize=7)
    fig.suptitle('Round 5: exploratory historical replay; 290 previously seen matches')
    fig.tight_layout()
    fig.savefig(output/'risk-cost-quality-frontier.png', dpi=160)
    plt.close(fig)
    bins = report['risk_calibration']['bins']
    fig, ax = plt.subplots(figsize=(6,5))
    x, y = [b['mean_expected_gain'] for b in bins], [b['mean_realized_delta'] for b in bins]
    if bins:
        ax.plot(x,y,'o-')
        bounds = [min(x+y),max(x+y)]
        ax.plot(bounds,bounds,'k--')
    ax.set(xlabel='Estimated mean gain', ylabel='Mean realized loss gain', title='Round 5: descriptive gain calibration')
    fig.tight_layout()
    fig.savefig(output/'risk-gain-calibration.png',dpi=160)
    plt.close(fig)


def split_calibration(samples, fraction=.2):
    """Keep same-kickoff matches together; purge labels crossing tail cutoff."""
    times = sorted({_sem_time(s["kickoff"]) for s in samples})
    if len(times) < 3:
        return samples, [], {"purged_matches": 0, "reason": "insufficient_time_blocks"}
    boundary = times[min(len(times)-1, max(1, int(len(times)*(1-fraction))))]
    tail = [s for s in samples if _sem_time(s["kickoff"]) >= boundary]
    first_prediction = min(_sem_time(s["state"]["prediction_cutoff"]) for s in tail)
    train = [s for s in samples if _sem_time(s["kickoff"]) < boundary
             and _sem_time(s["result_available_at"]) < first_prediction]
    kept = {s["match_id"] for s in train + tail}
    return train, tail, {"boundary": boundary.isoformat(),
        "first_tail_prediction": first_prediction.isoformat(),
        "purged_matches": len({s["match_id"] for s in samples} - kept)}


def diagnostics(rows, predictor, estimator, actions):
    records = []
    ids = [a["action_id"] for a in actions]
    for row in rows:
        for count in range(len(ids)):
            for acquired in itertools.combinations(ids, count):
                visible = state_after(row, actions, acquired)
                before = predictor.predict_partial(visible)
                estimates = estimator.estimate_value(visible, [a for a in actions if a["action_id"] not in acquired], prediction=before)
                for e in estimates:
                    after = predictor.predict_partial(state_after(row, actions, (*acquired, e["action_id"])))
                    records.append({"match_id": row["match_id"], "mask": before["mask_key"], "action_id": e["action_id"],
                        "estimated_gain": e["expected_gain"], "realized_delta": losses(before["probabilities"], outcome(row))["log_loss"] - losses(after["probabilities"], outcome(row))["log_loss"],
                        **{k: e.get(k) for k in ("uncertainty", "lower_gain", "upper_gain", "predictive_lower_gain", "predictive_upper_gain", "support", "parent_support", "gate_reason", "fallback_reason")}})
    return records


def calibration_summary(records):
    valid = [r for r in records if r["estimated_gain"] is not None]
    if not valid:
        return {"transitions": 0}
    def avg(key, rows):
        return float(np.mean([r[key] for r in rows]))
    bins = []
    for bucket in np.array_split(sorted(valid, key=lambda r: r["estimated_gain"]), 5):
        rows = list(bucket)
        if rows:
            bins.append({"transitions": len(rows), "matches": len({r["match_id"] for r in rows}),
                         "predicted_mean": avg("estimated_gain", rows), "realized_mean": avg("realized_delta", rows)})
    intervals = [r for r in valid if r["predictive_lower_gain"] is not None]
    return {"transitions": len(valid), "independent_matches": len({r["match_id"] for r in valid}),
        "sign_accuracy_realized": float(np.mean([(r["estimated_gain"] > 0) == (r["realized_delta"] > 0) for r in valid])),
        "sign_accuracy_majority_baseline": max(sum(r["realized_delta"] > 0 for r in valid), sum(r["realized_delta"] <= 0 for r in valid))/len(valid),
        "predictive_interval_coverage": float(np.mean([r["predictive_lower_gain"] <= r["realized_delta"] <= r["predictive_upper_gain"] for r in intervals])) if intervals else None,
        "predictive_interval_count": len(intervals), "mean_bias": float(np.mean([r["estimated_gain"]-r["realized_delta"] for r in valid])),
        "bins": bins, "mean_ci_coverage_against_single_game": "NOT_APPLICABLE_different_estimands",
        "scope": "already_seen_historical_test_diagnostics; realized signs are noisy labels for expected gain"}


def stopping_regret(row, result, predictor, actions, design):
    """Evaluator-only feasible hindsight continuations; never passed to policy."""
    remaining = [a for a in actions if a["action_id"] not in result["actions"]]
    best = result["log_loss"]
    for count in range(1, len(remaining)+1):
        for subset in itertools.combinations(remaining, count):
            money = sum(a["cost"]["money"] for a in subset)
            requests = sum(a["cost"]["requests"] for a in subset)
            if (money + result["cost"]["money"] > result["budget_money"] + 1e-12 or
                    requests + result["cost"]["requests"] > design["request_budget"]):
                continue
            after = predictor.predict_partial(state_after(row, actions, result["actions"] + [a["action_id"] for a in subset]))
            utility_loss = losses(after["probabilities"], outcome(row))["log_loss"] + design["validation_utility_money_weight"]*money
            best = min(best, utility_loss)
    return result["log_loss"] - best


def run(bundle, source, output, earlier_bundle=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = {**DEFAULT_DESIGN, "experiment_version": VERSION,
              "classification": "exploratory_already_seen_history_separated_mean_and_predictive_intervals"}
    # Freeze before reading outcome-dependent experiment results.
    write_json(output / "design.json", config)
    episodes, audit = load_episodes(bundle, lead_hours=24)
    source = Path(source)
    catalog = json.loads((source / "catalog.json").read_text(encoding="utf-8"))
    actions = catalog["actions"]
    oof = [json.loads(line) for line in (source / "oof-supervision.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if earlier_bundle:
        earlier, earlier_audit = load_episodes(earlier_bundle, lead_hours=24)
        assert not {r["match_id"] for r in earlier} & {r["match_id"] for r in episodes}
        assert max(_sem_time(r["result_available_at"]) for r in earlier) < min(_sem_time(r["prediction_cutoff"]) for r in episodes)
        episodes = earlier + episodes
        audit["earlier_training_export"] = earlier_audit
        # Recompute labels under predictors trained on the SAME expanded history.
        # Reusing single-season OOF deltas here would mismatch model populations.
        oof = forward_oof(episodes, actions, OOF_DESIGN, output)
    write_json(output / "data-audit.json", audit)
    write_json(output / "catalog.json", catalog)
    names = ("experiment_round5.py", "experiment_round4.py", "experiment.py", "risk_calibrated.py", "risk.py", "policy.py", "partial.py", "value.py", "data.py", "state.py", "lookahead.py", "lookahead_calibrated.py")
    implementation = {n: sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in names}
    write_json(output / "implementation.json", implementation)
    folds = rolling_origin_splits(episodes, initial_validation_start=config["validation_start"],
        validation_window=timedelta(days=30), test_window=timedelta(days=30))
    results, diagnostic_rows, fold_reports = [], [], []
    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as traces:
        for number, fold in enumerate(folds):
            train_ids = {r["match_id"] for r in fold.train}
            prior = [s for s in oof if s["match_id"] in train_ids and _sem_time(s["result_available_at"]) <= fold.train_cutoff]
            if len(train_ids) < 30 or not fold.test or not prior:
                continue
            print(f"fold {number}: train={len(train_ids)} test={len(fold.test)}", flush=True)
            predictor = PartialPredictor().fit(fold.train, as_of=fold.train_cutoff, dataset_version=episodes[0]["dataset_version"])
            predictor.save(output / "models" / f"fold-{number}-predictor.json")
            v1 = ValueEstimator(min_support=config["min_support"], shrinkage=config["shrinkage"]).fit(prior, as_of=fold.train_cutoff, dataset_version=episodes[0]["dataset_version"])
            train, tail, split = split_calibration(prior, config["calibration_fraction"])
            if not train:
                raise ValueError("Temporal calibration purge left no value training data")
            risk = CalibratedRiskAwareValueEstimator(min_support=config["min_support"], shrinkage=config["shrinkage"], risk_multiplier=config["risk_multiplier"]).fit(train, as_of=fold.train_cutoff, dataset_version=episodes[0]["dataset_version"], calibration_samples=tail)
            risk.save(output / "models" / f"fold-{number}-risk.json")
            v1.save(output / "models" / f"fold-{number}-v1-gain.json")
            selected = {(s,b): _select_lambda(fold.validation, predictor, v1, risk, actions, s, b, config, episode_runner=run_episode)
                        for s,b in itertools.product(("learned-net-gain-v1", "risk-mean", "risk-conservative", "two-step"), config["money_budgets"])}
            fold_reports.append({"fold": number, "split": fold.manifest(), "value_split": split,
                "training_match_ids": sorted(train_ids), "validation_match_ids": sorted(r["match_id"] for r in fold.validation),
                "test_match_ids": sorted(r["match_id"] for r in fold.test), "risk_version": risk.version,
                "calibration": risk.calibration, "lambda_selection": {f"{s}/{b:g}": v[1] for (s,b),v in selected.items()}})
            for row in fold.test:
                for budget, strategy in itertools.product(config["money_budgets"], STRATEGIES):
                    lam = selected.get((strategy, budget), (0.0,))[0]
                    for seed in config["random_seeds"] if strategy == "random" else [0]:
                        result, trace = run_episode(row, predictor, v1, risk, actions, strategy=strategy,
                            budget_money=budget, lambda_money=lam, seed=seed, design=config)
                        result.update(fold=number, hindsight_stop_regret=stopping_regret(row, result, predictor, actions, config))
                        results.append(result)
                        traces.write(canonical_json({"fold": number, "match_id": row["match_id"], "strategy": strategy, "budget_money": budget, "seed": seed, "events": trace}) + "\n")
            diagnostic_rows.extend(diagnostics(fold.test, predictor, risk, actions))
            write_json(output / "fold-progress.json", fold_reports)
    metrics, paired, per_match = _summarize(results, config)
    for key, metric in metrics.items():
        subset = [r for r in results if f"{r['strategy']}/money={r['budget_money']:g}" == key]
        metric["hindsight_stop_regret"] = float(np.mean([r["hindsight_stop_regret"] for r in subset]))
    report = {"status": "OK", "experiment_version": VERSION, "evidence_grade": config["classification"],
        "input_matches": len(episodes), "test_matches": len({r["match_id"] for r in results}), "episode_runs": len(results),
        "earlier_training_used": bool(earlier_bundle), "folds": fold_reports, "metrics": metrics, "paired_comparisons": paired,
        "risk_calibration": calibration_summary(diagnostic_rows), "implementation_sha256": implementation,
        "cost_interpretation": config["cost_interpretation"], "external_requests": 0, "llm_tokens": 0,
        "limitations": ["Previously examined test season: exploratory only", "Simulated view costs, not provider bills",
            "Mean cluster intervals have no finite-sample/conditional guarantee", "Predictive coverage is not mean CI coverage",
            "Hindsight regret uses outcome-known continuations, not an executable oracle", "Only two historical acquisition actions",
            "Dynamic synthetic scenarios are validated separately and do not establish real-source value learning"]}
    write_json(output / "report.json", report)
    for name, data in (("episodes", results), ("risk-diagnostics", diagnostic_rows)):
        (output / f"{name}.jsonl").write_text("".join(canonical_json(r)+"\n" for r in data), encoding="utf-8")
    # Reuse the risk-cost plotting routine, supplying its explicit diagnostic schema.
    plot_report = {**report, "risk_calibration": {"bins": [{"mean_expected_gain": b["predicted_mean"], "mean_realized_delta": b["realized_mean"]} for b in report["risk_calibration"].get("bins", [])]}}
    make_plots(plot_report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--round3-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--earlier-bundle")
    args = parser.parse_args()
    report = run(args.bundle, args.round3_dir, args.output, args.earlier_bundle)
    print(json.dumps({k:report[k] for k in ("status", "test_matches", "episode_runs", "earlier_training_used")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
