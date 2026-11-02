"""Round four risk-aware value and finite lookahead experiment.

This runner consumes the frozen round-three real export and OOF supervision,
then fits a new risk estimator inside each outer training window.  It never
passes a full episode row or a label to runtime prediction or policy code.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from football_research.baselines.api import canonical_json, utc
from football_research.evaluation.splits import rolling_origin_splits
from .data import ControlledEpisode, load_episodes
from .experiment import losses, outcome, state_after
from .lookahead import select_two_step
from .partial import PartialPredictor
from .policy import select_or_stop
from .risk import RiskAwareValueEstimator
from .value import ValueEstimator

VERSION = "risk-aware-conditional-forward-v1"
STRATEGIES = ("base-only", "all-available", "random", "fixed-order", "entropy", "disagreement",
              "learned-net-gain-v1", "risk-mean", "risk-conservative", "two-step")
DEFAULT_DESIGN = {
    "experiment_version": VERSION,
    "classification": "exploratory_already_seen_season_risk_aware_controlled_reveal",
    "lead_hours": 24, "validation_start": "2025-10-01T00:00:00Z", "validation_days": 30, "test_days": 30,
    "request_budget": 2, "money_budgets": [0.0, 0.02, 0.03, 0.05],
    "lambda_money_grid": [0.0, 0.25, 0.5, 1.0, 2.0],
    "validation_utility_money_weight": 0.5,
    "min_support": 12, "shrinkage": 20.0, "risk_multiplier": 1.64,
    "risk_target_coverage": 0.8, "calibration_fraction": 0.2,
    "bootstrap_samples": 2000, "bootstrap_seed": 1729,
    "random_seeds": [7, 19, 41, 73, 101], "entropy_threshold": 1.0,
    "disagreement_threshold_l1": 0.04,
    "llm_policies": {"pure-LLM": "SKIPPED_NO_AGENT_REAL_LLM_RUNNER", "hybrid": "SKIPPED_NO_AGENT_REAL_LLM_RUNNER"},
    "cost_interpretation": "simulated currency; utility 0.5 log-loss nats per simulated currency unit; not actual API prices",
    "environment_randomness": "deterministic frozen reveal; same rows/actions/seeds for all strategies",
}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sem_time(value):
    return utc(value)


def _episode_prediction(model, state):
    # This function accepts only the sanitized state and returns the public
    # predictor projection; it has no episode-row argument.
    return model.predict_partial(state)


def _heuristic_decision(strategy, state, prediction, initial, candidates, budget, design, rng):
    affordable = [a for a in candidates if a["cost"].get("money", float("inf")) <= budget.get("money", 0.0) + 1e-12
                   and a["cost"].get("requests", float("inf")) <= budget.get("requests", 0.0) + 1e-12]
    if not affordable:
        return {"decision": "stop", "action_id": None, "reason": "budget"}
    probabilities = prediction["probabilities"]
    entropy = -sum(p * math.log(max(p, 1e-15)) for p in probabilities)
    disagreement = sum(abs(a - b) for a, b in zip(probabilities, initial["probabilities"]))
    if strategy == "entropy" and entropy <= design["entropy_threshold"]:
        return {"decision": "stop", "action_id": None, "reason": "entropy_below_threshold"}
    if strategy == "disagreement" and disagreement <= design["disagreement_threshold_l1"]:
        return {"decision": "stop", "action_id": None, "reason": "disagreement_below_threshold"}
    selected = rng.choice(affordable) if strategy == "random" else affordable[0]
    return {"decision": "action", "action_id": selected["action_id"], "reason": strategy}


def run_episode(row, predictor, estimator_v1, estimator_risk, actions, *, strategy,
                budget_money, lambda_money, seed, design, lookahead=None):
    env = ControlledEpisode(row, actions)
    episode_seed = int(sha256(f"round4|{seed}|{row['match_id']}".encode()).hexdigest()[:16], 16)
    rng = random.Random(episode_seed)
    spent = {key: 0.0 for key in ("requests", "money", "latency_ms", "tokens")}
    initial = _episode_prediction(predictor, env.visible_state())
    prediction = initial
    events = []
    stopped = "base_only"
    predictor_ms = estimator_ms = 0.0
    estimator_calls = 0
    for step in range(len(actions) + 1):
        state = {**env.visible_state(), "action_history": [{"action_id": aid, "status": "acquired_or_source_missing"}
                                                              for aid in env.performed_actions]}
        if step:
            started = time.perf_counter()
            prediction = _episode_prediction(predictor, state)
            predictor_ms += (time.perf_counter() - started) * 1000
        public_metadata = {item["action_id"]: item for item in actions}
        candidates = []
        for candidate in env.legal_actions():
            original = public_metadata.get(candidate["action_id"], {})
            candidates.append({**candidate, **{key: deepcopy(original[key]) for key in
                ("dependency_key", "cache_policy", "cost_scope", "quality_policy", "prerequisites") if key in original}})
        budget = {"money": max(0.0, budget_money - spent["money"]),
                  "requests": max(0.0, design["request_budget"] - spent["requests"])}
        if strategy == "base-only":
            decision = {"decision": "stop", "action_id": None, "reason": "base_only"}
        elif not candidates:
            decision = {"decision": "stop", "action_id": None, "reason": "no_legal_action"}
        elif strategy in {"learned-net-gain-v1", "risk-mean", "risk-conservative"}:
            started = time.perf_counter()
            if strategy == "learned-net-gain-v1":
                estimates = estimator_v1.estimate(state, candidates, prediction=prediction)
                decision = select_or_stop(state, estimates, budget, {"money": lambda_money},
                                          conservative=False, min_support=design["min_support"])
            else:
                risk_level = "conservative" if strategy == "risk-conservative" else "mean"
                estimates = estimator_risk.estimate_value(state, candidates, prediction=prediction, risk_level=risk_level)
                decision = select_or_stop(state, estimates, budget, {"money": lambda_money},
                                          conservative=risk_level == "conservative", min_support=design["min_support"])
            estimator_ms += (time.perf_counter() - started) * 1000
            estimator_calls += 1
        elif strategy == "two-step":
            started = time.perf_counter()
            decision = (lookahead or select_two_step)(state, candidates,
                predictor=lambda public: _episode_prediction(predictor, public),
                estimator=estimator_risk, budget=budget, lambda_cost={"money": lambda_money},
                risk_level="mean", min_support=design["min_support"])
            estimator_ms += (time.perf_counter() - started) * 1000
            estimator_calls += 1
        else:
            decision = _heuristic_decision(strategy, state, prediction, initial, candidates, budget, design, rng)
        event = {"step": step, "visible_state": deepcopy(state), "prediction": deepcopy(prediction),
                 "decision": deepcopy(decision)}
        events.append(event)
        if decision["decision"] == "stop":
            stopped = decision["reason"]
            break
        selected = next(candidate for candidate in candidates if candidate["action_id"] == decision["action_id"])
        env.reveal(selected["action_id"])
        event["charged_cost"] = deepcopy(selected["cost"])
        for unit in spent:
            spent[unit] += float(selected["cost"].get(unit, 0.0))
    result = {"match_id": row["match_id"], "kickoff": row["kickoff"],
        "prediction_cutoff": row["prediction_cutoff"], "strategy": strategy, "seed": seed,
        "budget_money": budget_money, "lambda_money": lambda_money, "cost": spent,
        "predictor_latency_ms": predictor_ms, "estimator_latency_ms": estimator_ms,
        "estimator_calls": estimator_calls, "actual_external_requests": 0, "actual_llm_tokens": 0,
        "compute_money": None, "compute_money_status": "not_metered_local_cpu",
        "final_probabilities": prediction["probabilities"], "final_mask": prediction["mask"],
        "actions": list(env.performed_actions), "stop_reason": stopped,
        "model_version": prediction["model_version"],
        "gain_version": estimator_risk.version if strategy in {"risk-mean", "risk-conservative", "two-step"} else estimator_v1.version,
        **losses(prediction["probabilities"], outcome(row))}
    return result, events


def _split_calibration(samples, fraction):
    ordered = sorted(samples, key=lambda sample: (_sem_time(sample["kickoff"]), sample["match_id"]))
    cut = int(len(ordered) * (1.0 - fraction))
    if cut <= 0 or cut >= len(ordered):
        return ordered, []
    train_ids = {sample["match_id"] for sample in ordered[:cut]}
    train = [sample for sample in ordered if sample["match_id"] in train_ids]
    calibration = [sample for sample in ordered if sample["match_id"] not in train_ids]
    return train, calibration


def _select_lambda(validation, predictor, estimator_v1, estimator_risk, actions, strategy, budget, design, episode_runner=None):
    scores = []
    for value in design["lambda_money_grid"]:
        rows = [(episode_runner or run_episode)(row, predictor, estimator_v1, estimator_risk, actions,
                            strategy=strategy, budget_money=budget, lambda_money=value, seed=0, design=design)[0]
                for row in validation]
        utility = float(np.mean([r["log_loss"] + design["validation_utility_money_weight"] * r["cost"]["money"]
                                for r in rows])) if rows else None
        scores.append({"lambda_money": value, "validation_utility": utility,
                       "validation_matches": len(rows)})
    selected = min(scores, key=lambda item: (item["validation_utility"], -item["lambda_money"]))["lambda_money"]
    return selected, {"selected": selected, "candidates": scores,
                      "validation_match_ids": [row["match_id"] for row in validation],
                      "utility_money_weight": design["validation_utility_money_weight"]}


def _summarize(results, design):
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["strategy"], row["budget_money"])].append(row)
    summaries, per_match = {}, {}
    for (strategy, budget), rows in sorted(grouped.items()):
        grouped_match = defaultdict(list)
        for row in rows:
            grouped_match[row["match_id"]].append(row)
        means = {}
        for match_id, entries in grouped_match.items():
            means[match_id] = {key: float(np.mean([entry[key] for entry in entries]))
                               for key in ("log_loss", "brier_score", "accuracy", "predictor_latency_ms", "estimator_latency_ms")}
            means[match_id].update(money=float(np.mean([entry["cost"]["money"] for entry in entries])),
                                   requests=float(np.mean([entry["cost"]["requests"] for entry in entries])),
                                   kickoff=entries[0]["kickoff"])
        key = f"{strategy}/money={budget:g}"
        per_match[key] = means
        summaries[key] = {"independent_matches": len(means), "episodes": len(rows),
            **{metric: float(np.mean([row[metric] for row in means.values()]))
               for metric in ("log_loss", "brier_score", "accuracy", "money", "requests", "predictor_latency_ms", "estimator_latency_ms")},
            "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
            "action_counts": dict(Counter(action for row in rows for action in row["actions"])),
            "seed_metrics": {str(seed): {metric: float(np.mean([row[metric] for row in rows if row["seed"] == seed]))
                                          for metric in ("log_loss", "brier_score")}
                             for seed in sorted({row["seed"] for row in rows})},
            "cost_basis": "simulated_money_and_requests; local_compute_ms_measured_separately"}
    pairs = {}
    for budget in design["money_budgets"]:
        base = per_match[f"base-only/money={budget:g}"]
        for strategy in STRATEGIES:
            if strategy == "base-only":
                continue
            current = per_match[f"{strategy}/money={budget:g}"]
            blocks = defaultdict(list)
            for match_id in sorted(base.keys() & current.keys()):
                year, week, _ = _sem_time(current[match_id]["kickoff"]).isocalendar()
                blocks[(year, week)].append(current[match_id]["log_loss"] - base[match_id]["log_loss"])
            values = list(blocks.values())
            observed = np.concatenate(values) if values else np.array([])
            ci = None
            if len(observed) >= 30 and len(values) >= 8:
                rng = np.random.default_rng(design["bootstrap_seed"])
                sums = np.array([sum(value) for value in values])
                counts = np.array([len(value) for value in values])
                draws = rng.integers(0, len(values), size=(design["bootstrap_samples"], len(values)))
                sample = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
                ci = np.quantile(sample, [0.025, 0.975]).tolist()
            pairs[f"{strategy}/money={budget:g}"] = {
                "reference": "base-only", "mean_log_loss_difference": float(observed.mean()) if len(observed) else None,
                "confidence_interval_95": ci, "independent_matches": len(observed), "weeks": len(values),
                "method": "paired UTC week cluster percentile bootstrap; random seeds averaged within each match",
                "limitations": "exploratory; no multiple-comparison or fitted-policy uncertainty correction"}
    return summaries, pairs, per_match


def _risk_diagnostics(test_rows, predictor, estimator, actions):
    diagnostics = []
    for row in test_rows:
        ids = [action["action_id"] for action in actions]
        for count in range(len(ids)):
            for acquired in itertools.combinations(ids, count):
                state = state_after(row, actions, acquired)
                prediction = predictor.predict_partial(state)
                legal = [action for action in actions if action["action_id"] not in acquired]
                estimates = estimator.estimate_value(state, legal, prediction=prediction, risk_level="mean")
                for estimate in estimates:
                    after_state = state_after(row, actions, (*acquired, estimate["action_id"]))
                    after = predictor.predict_partial(after_state)
                    realized = losses(prediction["probabilities"], outcome(row))["log_loss"] - losses(after["probabilities"], outcome(row))["log_loss"]
                    diagnostics.append({"match_id": row["match_id"], "mask": prediction["mask_key"],
                        "action_id": estimate["action_id"], "estimated_gain": estimate["expected_gain"],
                        "uncertainty": estimate["uncertainty"], "lower_gain": estimate["lower_gain"],
                        "support": estimate["support"], "realized_delta": realized,
                        "quality_context": estimate["quality_context"], "fallback_reason": estimate["fallback_reason"]})
    return diagnostics


def run(platform_export: str | Path, output_dir: str | Path, *, round3_dir: str | Path,
        design: dict | None = None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = {**DEFAULT_DESIGN, **(design or {})}
    write_json(output / "design.json", config)
    source = Path(round3_dir)
    implementation_files = ("experiment_round4.py", "risk.py", "lookahead.py", "partial.py", "state.py", "data.py", "policy.py", "value.py")
    implementation = {name: sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in implementation_files}
    write_json(output / "implementation.json", implementation)
    episodes, audit = load_episodes(platform_export, lead_hours=config["lead_hours"])
    write_json(output / "data-audit.json", audit)
    catalog = json.loads((source / "catalog.json").read_text(encoding="utf-8"))
    actions = catalog["actions"]
    write_json(output / "catalog.json", catalog)
    oof = list(_load_jsonl(source / "oof-supervision.jsonl"))
    folds = rolling_origin_splits(episodes, initial_validation_start=config["validation_start"],
        validation_window=timedelta(days=config["validation_days"]), test_window=timedelta(days=config["test_days"]))
    results, diagnostics, fold_reports = [], [], []
    trajectory_file = (output / "trajectories.jsonl").open("w", encoding="utf-8")
    try:
        for fold_number, fold in enumerate(folds):
            train_ids = {row["match_id"] for row in fold.train}
            prior = [sample for sample in oof if sample["match_id"] in train_ids and
                     _sem_time(sample["result_available_at"]) <= fold.train_cutoff]
            if len(fold.train) < 30 or not fold.test or not prior:
                continue
            predictor = PartialPredictor().fit(list(fold.train), as_of=fold.train_cutoff,
                                               dataset_version=episodes[0]["dataset_version"], regularization=10.0)
            v1 = ValueEstimator(min_support=config["min_support"], shrinkage=config["shrinkage"],
                                uncertainty_multiplier=config["risk_multiplier"]).fit(
                prior, as_of=fold.train_cutoff, dataset_version=episodes[0]["dataset_version"])
            risk_train, risk_calibration = _split_calibration(prior, config["calibration_fraction"])
            risk = RiskAwareValueEstimator(min_support=config["min_support"], shrinkage=config["shrinkage"],
                                           risk_multiplier=config["risk_multiplier"]).fit(
                risk_train, as_of=fold.train_cutoff, dataset_version=episodes[0]["dataset_version"],
                calibration_samples=risk_calibration, calibration_target_coverage=config["risk_target_coverage"],
                metadata={"third_round_failure": "mask_action_point_estimate_unreliable", "calibration_role": "inner_train_tail"})
            risk.save(output / "models" / f"fold-{fold_number}-risk.json")
            v1.save(output / "models" / f"fold-{fold_number}-v1-gain.json")
            selected = {}
            for budget in config["money_budgets"]:
                for strategy in ("learned-net-gain-v1", "risk-mean", "risk-conservative", "two-step"):
                    selected[strategy, budget] = _select_lambda(fold.validation, predictor, v1, risk, actions,
                        strategy, budget, config)
            fold_reports.append({"fold": fold_number, "split": fold.manifest(),
                "training_match_ids": sorted(train_ids), "oof_training_matches": len({s["match_id"] for s in prior}),
                "risk_training_matches": len({s["match_id"] for s in risk_train}),
                "risk_calibration_matches": len({s["match_id"] for s in risk_calibration}),
                "validation_match_ids": sorted(row["match_id"] for row in fold.validation),
                "test_match_ids": sorted(row["match_id"] for row in fold.test),
                "v1_gain_version": v1.version, "risk_version": risk.version,
                "risk_multiplier": risk.risk_multiplier,
                "lambda_selection": {f"{strategy}/{budget:g}": value[1] for (strategy, budget), value in selected.items()}})
            write_json(output / "fold-progress.json", fold_reports)
            for row in fold.test:
                for budget, strategy in itertools.product(config["money_budgets"], STRATEGIES):
                    lambda_value = selected[strategy, budget][0] if strategy in {"learned-net-gain-v1", "risk-mean", "risk-conservative", "two-step"} else 0.0
                    seeds = config["random_seeds"] if strategy == "random" else [0]
                    for seed in seeds:
                        result, trace = run_episode(row, predictor, v1, risk, actions, strategy=strategy,
                            budget_money=budget, lambda_money=lambda_value, seed=seed, design=config)
                        results.append({"fold": fold_number, **result})
                        trajectory_file.write(canonical_json({"fold": fold_number, "match_id": row["match_id"],
                            "strategy": strategy, "budget_money": budget, "seed": seed, "events": trace}) + "\n")
            diagnostics.extend(_risk_diagnostics(fold.test, predictor, risk, actions))
    finally:
        trajectory_file.close()
    metrics, paired, per_match = _summarize(results, config)
    valid = [row for row in diagnostics if row["estimated_gain"] is not None]
    bins = []
    for values in np.array_split(sorted(valid, key=lambda row: row["estimated_gain"]), 5):
        values = list(values)
        if values:
            bins.append({"transitions": len(values), "independent_matches": len({row["match_id"] for row in values}),
                         "mean_expected_gain": float(np.mean([row["estimated_gain"] for row in values])),
                         "mean_realized_delta": float(np.mean([row["realized_delta"] for row in values])),
                         "mean_uncertainty": float(np.mean([row["uncertainty"] for row in values if row["uncertainty"] is not None])) if any(row["uncertainty"] is not None for row in values) else None,
                         "mean_support": float(np.mean([row["support"] for row in values]))})
    support_bins = {}
    for label, rows in [("low_lt_12", [row for row in valid if row["support"] < 12]),
                        ("supported_ge_12", [row for row in valid if row["support"] >= 12])]:
        support_bins[label] = {"transitions": len(rows), "sign_accuracy": float(np.mean([
            (row["estimated_gain"] > 0) == (row["realized_delta"] > 0) for row in rows])) if rows else None,
            "mean_error": float(np.mean([row["estimated_gain"] - row["realized_delta"] for row in rows])) if rows else None}
    rank_states = defaultdict(list)
    for row in valid:
        rank_states[(row["match_id"], row["mask"])].append(row)
    rank_states = [rows for rows in rank_states.values() if len(rows) >= 2]
    report = {"status": "OK", "experiment_version": VERSION,
        "evidence_grade": config["classification"], "dataset_version": episodes[0]["dataset_version"],
        "input_matches": len(episodes), "test_matches": len({row["match_id"] for row in results}),
        "episode_runs": len(results), "strategies": STRATEGIES, "implementation_sha256": implementation,
        "folds": fold_reports, "metrics": metrics, "paired_comparisons": paired,
        "risk_calibration": {"bins": bins, "support_bins": support_bins,
            "transitions": len(valid), "independent_matches": len({row["match_id"] for row in valid}),
            "sign_accuracy": float(np.mean([(row["estimated_gain"] > 0) == (row["realized_delta"] > 0) for row in valid])) if valid else None,
            "top_action_realized_agreement": float(np.mean([
                max(rows, key=lambda row: row["estimated_gain"])["action_id"] == max(rows, key=lambda row: row["realized_delta"])["action_id"]
                for rows in rank_states])) if rank_states else None,
            "ranking_states": len(rank_states), "realized_negative_fraction": float(np.mean([row["realized_delta"] < 0 for row in valid])) if valid else None,
            "scope": "OOF diagnostic over 290 previously explored matches; not an independent confirmation set"},
        "failure_cases": sorted(valid, key=lambda row: abs(row["estimated_gain"] - row["realized_delta"]), reverse=True)[:20],
        "estimator_runtime": {"risk_batches": sum(row["estimator_calls"] for row in results if row["strategy"] in {"risk-mean", "risk-conservative", "two-step"}),
            "mean_episode_estimator_ms": float(np.mean([row["estimator_latency_ms"] for row in results if row["estimator_calls"]])) if any(row["estimator_calls"] for row in results) else 0.0,
            "external_requests": 0, "llm_tokens": 0, "money": None, "money_status": "not_metered_local_cpu"},
        "llm_policies": config["llm_policies"], "cost_interpretation": config["cost_interpretation"],
        "limitations": ["2025/26 test matches were already explored in round3", "historical reconstruction uses delayed-availability assumptions and revised source snapshots", "risk intervals are empirical residual intervals without strict coverage", "finite lookahead uses mask transition projections and is not global planning", "simulated costs are not real provider savings", "one season and early OOF support limit transport"]}
    write_json(output / "report.json", report)
    (output / "episodes.jsonl").write_text("".join(canonical_json(row) + "\n" for row in results), encoding="utf-8")
    (output / "risk-diagnostics.jsonl").write_text("".join(canonical_json(row) + "\n" for row in diagnostics), encoding="utf-8")
    make_plots(report, output)
    return report


def make_plots(report, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for strategy in STRATEGIES:
        values = [value for key, value in report["metrics"].items() if key.startswith(strategy + "/")]
        values.sort(key=lambda value: value["money"])
        axes[0].plot([value["money"] for value in values], [value["log_loss"] for value in values], "o-", label=strategy)
        axes[1].plot([value["money"] for value in values], [value["brier_score"] for value in values], "o-", label=strategy)
    axes[0].set(xlabel="Mean simulated currency cost", ylabel="log loss")
    axes[1].set(xlabel="Mean simulated currency cost", ylabel="Brier score")
    axes[1].legend(fontsize=7)
    fig.suptitle("Round four risk-aware acquisition; 290 previously explored matches")
    fig.tight_layout()
    fig.savefig(output / "risk-cost-quality-frontier.png", dpi=160)
    plt.close(fig)
    bins = report["risk_calibration"]["bins"]
    if bins:
        fig, ax = plt.subplots(figsize=(6, 5))
        x = [row["mean_expected_gain"] for row in bins]
        y = [row["mean_realized_delta"] for row in bins]
        ax.plot(x, y, "o-")
        bounds = [min(x + y), max(x + y)]
        ax.plot(bounds, bounds, "k--")
        ax.set(xlabel="Expected conditional gain", ylabel="Realized OOF loss gain",
               title="Risk-aware gain calibration; diagnostic only")
        fig.tight_layout()
        fig.savefig(output / "risk-gain-calibration.png", dpi=160)
        plt.close(fig)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform-export", required=True)
    parser.add_argument("--round3", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--design")
    args = parser.parse_args(argv)
    design = json.loads(Path(args.design).read_text(encoding="utf-8")) if args.design else None
    report = run(args.platform_export, args.output, round3_dir=args.round3, design=design)
    print(json.dumps({key: report[key] for key in ("status", "input_matches", "test_matches", "episode_runs", "risk_calibration")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
