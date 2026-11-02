"""Exploratory forward-time controlled reveal experiment; no external requests.

Labels/full feature rows stay in this offline supervisor. Decision functions see
serialized visible states, catalog metadata and a frozen model only.
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
from .partial import PartialPredictor
from .value import ValueEstimator
from .policy import select_or_stop

VERSION = "controlled-reveal-forward-v1"
STRATEGIES = ("base-only", "all-available", "random", "fixed-order", "entropy", "disagreement",
              "learned-net-gain", "conservative-net-gain")
DEFAULT_DESIGN = {
    "experiment_version": VERSION, "classification": "exploratory_already_seen_season_controlled_reveal",
    "lead_hours": 24, "initial_validation_start": "2025-10-01T00:00:00Z", "validation_days": 30, "test_days": 30,
    "regularization": 10.0, "min_train_matches": 30, "oof_warmup": 40, "oof_chunk_matches": 30,
    "gain_min_support": 12, "gain_shrinkage": 20.0, "uncertainty_multiplier": 1.64,
    "money_budgets": [0.0, 0.02, 0.03, 0.05], "request_budget": 2,
    "lambda_money_grid": [0.0, 0.25, 0.5, 1.0, 2.0], "validation_utility_money_weight": 0.5,
    "cost_interpretation": "simulated currency; utility 0.5 log-loss nats per simulated currency unit; not actual API prices",
    "random_seeds": [7, 19, 41, 73, 101], "bootstrap_samples": 2000, "bootstrap_seed": 1729,
    "entropy_threshold": 1.0, "disagreement_threshold_l1": 0.04,
    "llm_policies": {"pure-LLM": "SKIPPED_NO_AGENT_REAL_LLM_RUNNER", "hybrid": "SKIPPED_NO_AGENT_REAL_LLM_RUNNER"},
    "calibration": "diagnostic_only_no_test_set_fitting",
    "environment_randomness": "deterministic_frozen_reveal; identical source_missing outcomes across all strategies/seeds",
}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(canonical_json(value) + "\n", encoding="utf-8")


def outcome(row):
    return 0 if row["home_goals"] > row["away_goals"] else 1 if row["home_goals"] == row["away_goals"] else 2


def losses(probabilities, label):
    return {"log_loss": -math.log(max(probabilities[label], 1e-15)),
            "brier_score": sum((p - int(i == label)) ** 2 for i, p in enumerate(probabilities)),
            "accuracy": int(max(range(3), key=lambda i: probabilities[i]) == label)}


def state_after(row, actions, acquired):
    environment = ControlledEpisode(row, actions)
    for aid in acquired:
        environment.reveal(aid)
    return environment.visible_state()


def forward_oof(rows, actions, design, output):
    """Every Delta is from a predictor trained before its target chunk cutoff."""
    rows = sorted(rows, key=lambda r: (utc(r["kickoff"]), r["match_id"]))
    samples, folds = [], []
    ids = [a["action_id"] for a in actions]
    for start in range(design["oof_warmup"], len(rows), design["oof_chunk_matches"]):
        targets = rows[start:start + design["oof_chunk_matches"]]
        cutoff = min(utc(r["prediction_cutoff"]) for r in targets) - timedelta(microseconds=1)
        train = [r for r in rows if utc(r["result_available_at"]) <= cutoff and utc(r["kickoff"]) < cutoff]
        if len(train) < design["min_train_matches"]:
            continue
        model = PartialPredictor().fit(train, as_of=cutoff, dataset_version=rows[0]["dataset_version"], regularization=design["regularization"])
        artifact = output / "oof-models" / f"chunk-{start}.json"
        model.save(artifact)
        train_ids = [r["match_id"] for r in train]
        manifest_ref = "sha256:" + sha256(artifact.read_bytes()).hexdigest()
        folds.append({"artifact": str(artifact), "cutoff": cutoff.isoformat(), "training_match_ids": train_ids,
                      "prediction_match_ids": [r["match_id"] for r in targets]})
        for row in targets:
            cached = {}
            for count in range(len(ids) + 1):
                for acquired in itertools.combinations(ids, count):
                    state = state_after(row, actions, acquired)
                    cached[frozenset(acquired)] = (state, model.predict_partial(state)["probabilities"])
            for acquired, (state, before) in cached.items():
                for action in actions:
                    if action["action_id"] in acquired:
                        continue
                    after = cached[acquired | {action["action_id"]}][1]
                    samples.append({"match_id": row["match_id"], "kickoff": row["kickoff"],
                        "result_available_at": row["result_available_at"], "state": state, "action": action,
                        "prediction_before": before,
                        "delta": losses(before, outcome(row))["log_loss"] - losses(after, outcome(row))["log_loss"],
                        "oof_train_cutoff": cutoff.isoformat(), "oof_train_ids": train_ids, "oof_manifest_ref": manifest_ref})
    write_json(output / "oof-manifest.json", {"folds": folds, "samples": len(samples),
        "independent_matches": len({s["match_id"] for s in samples}),
        "training_target": "realized_oof_loss_before_minus_after; negative outcomes preserved"})
    # Training supervision is deliberately separate from runtime artifact/state.
    (output / "oof-supervision.jsonl").write_text("".join(canonical_json(s) + "\n" for s in samples), encoding="utf-8")
    return samples


def run_episode(row, predictor, estimator, actions, *, strategy, budget_money, lambda_money, seed, design):
    env = ControlledEpisode(row, actions)
    # Match ID/seed schedule never depends on hidden values or outcome.
    episode_seed = int(sha256(f"{seed}|{row['match_id']}".encode()).hexdigest()[:16], 16)
    rng = random.Random(episode_seed)
    spent = {k: 0.0 for k in ("requests", "money", "latency_ms", "tokens")}
    trace, history = [], []
    initial_started = time.perf_counter()
    initial = predictor.predict_partial(env.visible_state())
    initial_ms = (time.perf_counter() - initial_started) * 1000
    prediction = initial
    predictor_ms, estimator_ms, estimator_calls = initial_ms, 0.0, 0
    stopped = "base_only"
    for step in range(len(actions) + 1):
        state = {**env.visible_state(), "action_history": deepcopy(history)}
        if step:
            start = time.perf_counter()
            prediction = predictor.predict_partial(state)
            predictor_ms += (time.perf_counter() - start) * 1000
        candidates = env.legal_actions()
        budget = {"money": max(0.0, budget_money - spent["money"]),
                  "requests": max(0.0, design["request_budget"] - spent["requests"])}
        if strategy == "base-only":
            stopped = "base_only"
            break
        if not candidates:
            stopped = "no_legal_action"
            break
        if strategy in {"learned-net-gain", "conservative-net-gain"}:
            estimates = estimator.estimate(state, candidates, prediction=prediction)
            elapsed = estimates[0]["estimator_batch_latency_ms"] if estimates else 0.0
            estimator_ms += elapsed
            estimator_calls += 1
            decision = select_or_stop(state, estimates, budget, {"money": lambda_money},
                                      conservative=strategy == "conservative-net-gain", min_support=design["gain_min_support"])
        else:
            affordable = [a for a in candidates if a["cost"]["money"] <= budget["money"] + 1e-12
                          and a["cost"]["requests"] <= budget["requests"]]
            if not affordable:
                decision = {"decision": "stop", "action_id": None, "reason": "budget"}
            else:
                p = prediction["probabilities"]
                entropy = -sum(v * math.log(max(v, 1e-15)) for v in p)
                disagree = sum(abs(a - b) for a, b in zip(p, initial["probabilities"]))
                reason = "entropy_below_threshold" if strategy == "entropy" and entropy <= design["entropy_threshold"] else (
                    "disagreement_below_threshold" if strategy == "disagreement" and step > 0 and disagree <= design["disagreement_threshold_l1"] else None)
                if reason:
                    decision = {"decision": "stop", "action_id": None, "reason": reason}
                else:
                    selected = rng.choice(affordable) if strategy == "random" else affordable[0]
                    decision = {"decision": "action", "action_id": selected["action_id"], "reason": strategy}
        event = {"step": step, "visible_state": state, "prediction": prediction, "decision": decision}
        trace.append(event)
        if decision["decision"] == "stop":
            stopped = decision["reason"]
            break
        selected = next(a for a in candidates if a["action_id"] == decision["action_id"])
        env.reveal(selected["action_id"])
        for unit in spent:
            spent[unit] += selected["cost"][unit]
        history.append({"action_id": selected["action_id"], "status": "acquired_or_source_missing"})
        event["charged_cost"] = selected["cost"]
    result = {"match_id": row["match_id"], "kickoff": row["kickoff"], "prediction_cutoff": row["prediction_cutoff"],
        "strategy": strategy, "seed": seed, "budget_money": budget_money, "lambda_money": lambda_money,
        "cost": spent, "predictor_latency_ms": predictor_ms, "estimator_latency_ms": estimator_ms,
        "estimator_calls": estimator_calls, "actual_external_requests": 0, "actual_llm_tokens": 0,
        "compute_money": None, "compute_money_status": "not_metered_local_cpu",
        "final_probabilities": prediction["probabilities"], "final_mask": prediction["mask"],
        "actions": list(env.performed_actions), "stop_reason": stopped,
        "model_version": prediction["model_version"], "gain_version": estimator.version,
        **losses(prediction["probabilities"], outcome(row))}
    return result, trace


def select_lambda(validation, predictor, estimator, actions, *, strategy, budget_money, design):
    scores = []
    for value in design["lambda_money_grid"]:
        results = [run_episode(r, predictor, estimator, actions, strategy=strategy, budget_money=budget_money,
                               lambda_money=value, seed=0, design=design)[0] for r in validation]
        utility = float(np.mean([r["log_loss"] + design["validation_utility_money_weight"] * r["cost"]["money"] for r in results])) if results else None
        scores.append({"lambda_money": value, "validation_utility": utility, "validation_matches": len(results)})
    candidates = [r for r in scores if r["validation_utility"] is not None]
    selected = min(candidates, key=lambda r: (r["validation_utility"], -r["lambda_money"]))["lambda_money"] if candidates else max(design["lambda_money_grid"])
    return selected, {"selected": selected, "candidates": scores, "utility_money_weight": design["validation_utility_money_weight"],
        "validation_match_ids": [r["match_id"] for r in validation], "fallback": None if candidates else "no_validation_choose_highest_cost_penalty"}


def gain_diagnostics(rows, predictor, estimator, actions):
    """Offline evaluation only: unseen action returns/labels never reach estimate."""
    result = []
    ids = [a["action_id"] for a in actions]
    for row in rows:
        for n in range(len(ids)):
            for acquired in itertools.combinations(ids, n):
                state = state_after(row, actions, acquired)
                before = predictor.predict_partial(state)
                legal = [a for a in actions if a["action_id"] not in acquired]
                estimates = estimator.estimate(state, legal, prediction=before)
                for estimate in estimates:
                    action = estimate["action_id"]
                    after_state = state_after(row, actions, (*acquired, action))
                    after = predictor.predict_partial(after_state)
                    delta = losses(before["probabilities"], outcome(row))["log_loss"] - losses(after["probabilities"], outcome(row))["log_loss"]
                    result.append({"match_id": row["match_id"], "kickoff": row["kickoff"], "mask": before["mask_key"],
                        "action_id": action, "estimated_gain": estimate["gain"], "realized_delta": delta,
                        "support_matches": estimate["support_matches"], "reason": estimate["reason"],
                        "lower_gain": estimate["lower_gain"]})
    return result


def summarize(results, design):
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["strategy"], row["budget_money"])].append(row)
    summaries, averaged = {}, {}
    for (strategy, budget), rows in sorted(grouped.items()):
        per_match = defaultdict(list)
        for row in rows:
            per_match[row["match_id"]].append(row)
        means = {}
        for mid, sample in per_match.items():
            means[mid] = {k: float(np.mean([r[k] for r in sample])) for k in ("log_loss", "brier_score", "accuracy", "estimator_latency_ms", "predictor_latency_ms")}
            means[mid].update(money=float(np.mean([r["cost"]["money"] for r in sample])), requests=float(np.mean([r["cost"]["requests"] for r in sample])), kickoff=sample[0]["kickoff"])
        key = f"{strategy}/money={budget:g}"
        averaged[key] = means
        summaries[key] = {"independent_matches": len(means), "episodes": len(rows),
            **{k: float(np.mean([r[k] for r in means.values()])) for k in ("log_loss", "brier_score", "accuracy", "money", "requests", "estimator_latency_ms", "predictor_latency_ms")},
            "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
            "action_counts": dict(Counter(a for r in rows for a in r["actions"])),
            "seed_metrics": {str(seed): {k: float(np.mean([r[k] for r in rows if r["seed"] == seed])) for k in ("log_loss", "brier_score")}
                             for seed in sorted({r["seed"] for r in rows})},
            "cost_basis": "simulated_money_and_requests; local_compute_ms_measured_separately"}
    pairs = {}
    for budget in design["money_budgets"]:
        reference = averaged[f"base-only/money={budget:g}"]
        for strategy in STRATEGIES:
            if strategy == "base-only":
                continue
            key = f"{strategy}/money={budget:g}"
            lhs = averaged[key]
            buckets = defaultdict(list)
            for mid in sorted(lhs.keys() & reference.keys()):
                kickoff = utc(lhs[mid]["kickoff"])
                year, week, _ = kickoff.isocalendar()
                buckets[(year, week)].append(lhs[mid]["log_loss"] - reference[mid]["log_loss"])
            block_values = list(buckets.values())
            observed = np.concatenate(block_values) if block_values else np.array([])
            ci = None
            if len(observed) >= 30 and len(block_values) >= 8:
                # resample the same week indices for every strategy at this budget
                rng = np.random.default_rng(design["bootstrap_seed"])
                sums = np.array([sum(v) for v in block_values])
                counts = np.array([len(v) for v in block_values])
                draws = rng.integers(0, len(block_values), size=(design["bootstrap_samples"], len(block_values)))
                sampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
                ci = np.quantile(sampled, [0.025, 0.975]).tolist()
            pairs[key] = {"reference": "base-only", "mean_log_loss_difference": float(observed.mean()) if len(observed) else None,
                "confidence_interval_95": ci, "independent_matches": len(observed), "weeks": len(block_values),
                "method": "paired UTC week cluster percentile bootstrap; random seeds averaged within each match",
                "limitations": "exploratory, no multiple comparison or fitted-policy uncertainty correction"}
    return summaries, pairs


def run(export_dir, output_dir, *, design=None, actions=None, catalog_manifest=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = {**DEFAULT_DESIGN, **(design or {})}
    write_json(output / "design.json", config)  # Freeze before looking at experiment metrics.
    implementation = {name: sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("experiment.py", "partial.py", "state.py", "value.py", "policy.py", "data.py", "platform_bridge.py")}
    write_json(output / "implementation.json", implementation)
    if actions is None:
        from .platform_bridge import experiment_actions
        actions, catalog_manifest = experiment_actions()
    write_json(output / "catalog.json", {"actions": actions, "platform_binding": catalog_manifest})
    episodes, audit = load_episodes(export_dir, lead_hours=config["lead_hours"])
    write_json(output / "data-audit.json", audit)
    if not episodes:
        raise ValueError("No eligible real episodes")
    samples = forward_oof(episodes, actions, config, output)
    folds = rolling_origin_splits(episodes, initial_validation_start=config["initial_validation_start"],
        validation_window=timedelta(days=config["validation_days"]), test_window=timedelta(days=config["test_days"]))
    all_results, all_diagnostics, fold_reports = [], [], []
    traces = (output / "trajectories.jsonl").open("w", encoding="utf-8")
    try:
        for i, fold in enumerate(folds):
            train = list(fold.train)
            if len(train) < config["min_train_matches"] or not fold.test:
                continue
            train_ids = {r["match_id"] for r in train}
            prior_samples = [s for s in samples if s["match_id"] in train_ids and utc(s["result_available_at"]) <= fold.train_cutoff]
            if not prior_samples:
                continue
            predictor = PartialPredictor().fit(train, as_of=fold.train_cutoff,
                dataset_version=episodes[0]["dataset_version"], regularization=config["regularization"])
            estimator = ValueEstimator(min_support=config["gain_min_support"], shrinkage=config["gain_shrinkage"],
                uncertainty_multiplier=config["uncertainty_multiplier"]).fit(prior_samples, as_of=fold.train_cutoff, dataset_version=episodes[0]["dataset_version"])
            predictor.save(output / "models" / f"fold-{i}-partial.json")
            estimator.save(output / "models" / f"fold-{i}-gain.json")
            selections = {}
            for budget in config["money_budgets"]:
                for strategy in ("learned-net-gain", "conservative-net-gain"):
                    selections[strategy, budget] = select_lambda(fold.validation, predictor, estimator, actions,
                        strategy=strategy, budget_money=budget, design=config)
            fold_reports.append({"fold": i, "split": fold.manifest(), "training_match_ids": sorted(train_ids),
                "gain_training_matches": len({s["match_id"] for s in prior_samples}),
                "gain_training_ids": sorted({s["match_id"] for s in prior_samples}),
                "validation_match_ids": sorted({r["match_id"] for r in fold.validation}),
                "test_match_ids": sorted({r["match_id"] for r in fold.test}),
                "model_version": predictor.model_version, "gain_version": estimator.version,
                "lambda_selection": {f"{s}/{b:g}": v[1] for (s, b), v in selections.items()}})
            write_json(output / "fold-progress.json", fold_reports)
            for row in fold.test:
                for budget, strategy in itertools.product(config["money_budgets"], STRATEGIES):
                    penalty = selections[strategy, budget][0] if strategy in {"learned-net-gain", "conservative-net-gain"} else 0
                    seeds = config["random_seeds"] if strategy == "random" else [0]
                    for seed in seeds:
                        result, trace = run_episode(row, predictor, estimator, actions, strategy=strategy,
                            budget_money=budget, lambda_money=penalty, seed=seed, design=config)
                        all_results.append({"fold": i, **result})
                        traces.write(canonical_json({"fold": i, "match_id": row["match_id"], "strategy": strategy,
                            "budget_money": budget, "seed": seed, "events": trace}) + "\n")
            all_diagnostics.extend(gain_diagnostics(fold.test, predictor, estimator, actions))
    finally:
        traces.close()
    if not all_results:
        raise ValueError("No eligible held-forward exploratory episodes")
    metrics, comparisons = summarize(all_results, config)
    valid = [d for d in all_diagnostics if d["estimated_gain"] is not None]
    bins = []
    for values in np.array_split(sorted(valid, key=lambda r: r["estimated_gain"]), 5):
        values = list(values)
        if values:
            bins.append({"transitions": len(values), "independent_matches": len({r["match_id"] for r in values}),
                "mean_estimated_gain": float(np.mean([r["estimated_gain"] for r in values])),
                "mean_realized_delta": float(np.mean([r["realized_delta"] for r in values]))})
    failures = sorted(valid, key=lambda r: r["estimated_gain"] - r["realized_delta"], reverse=True)[:20]
    ranked = defaultdict(list)
    for diagnostic in valid:
        ranked[diagnostic["match_id"], diagnostic["mask"]].append(diagnostic)
    rankings = [values for values in ranked.values() if len(values) >= 2]
    latencies = [r["estimator_latency_ms"] / r["estimator_calls"] for r in all_results if r["estimator_calls"]]
    report = {"status": "OK", "experiment_version": VERSION, "evidence_grade": config["classification"],
        "dataset_version": episodes[0]["dataset_version"], "input_matches": len(episodes),
        "implementation_sha256": implementation,
        "test_matches": len({r["match_id"] for r in all_results}), "episode_runs": len(all_results),
        "folds": fold_reports, "metrics": metrics, "paired_comparisons": comparisons,
        "gain_calibration": {"bins": bins, "sign_accuracy": float(np.mean([(d["estimated_gain"] > 0) == (d["realized_delta"] > 0) for d in valid])),
            "transitions": len(valid), "independent_matches": len({d["match_id"] for d in valid}),
            "realized_negative_fraction": float(np.mean([d["realized_delta"] < 0 for d in valid])),
            "top_action_realized_agreement": float(np.mean([max(v, key=lambda d: d["estimated_gain"])["action_id"] == max(v, key=lambda d: d["realized_delta"])["action_id"] for v in rankings])) if rankings else None,
            "ranking_states": len(rankings),
            "scope": "offline diagnostic across all reachable states; not independent samples and not policy fitting"},
        "failure_examples": failures,
        "estimator_runtime": {"mean_ms_per_batch": float(np.mean(latencies)), "p95_ms_per_episode_average_batch": float(np.quantile(latencies, 0.95)),
                              "external_requests": 0, "llm_tokens": 0, "money": None, "money_status": "not_metered_local_cpu"},
        "llm_policies": config["llm_policies"], "cost_interpretation": config["cost_interpretation"],
        "limitations": ["290 test matches were already explored in round2", "historical reconstruction timing and revised-snapshot biases remain",
            "simulated reveal costs are not actual API savings", "one-step rule may miss complementary evidence", "one season and small forward OOF support",
            "no test tuning or test probability calibration", "CPU latency measured separately; money for local compute is unknown"]}
    write_json(output / "report.json", report)
    (output / "episodes.jsonl").write_text("".join(canonical_json(r) + "\n" for r in all_results), encoding="utf-8")
    (output / "gain-diagnostics.jsonl").write_text("".join(canonical_json(r) + "\n" for r in all_diagnostics), encoding="utf-8")
    make_plots(report, output)
    return report


def make_plots(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for strategy in STRATEGIES:
        values = [v for k, v in report["metrics"].items() if k.startswith(strategy + "/")]
        values.sort(key=lambda v: v["money"])
        for axis, metric in zip(axes, ("log_loss", "brier_score")):
            axis.plot([v["money"] for v in values], [v[metric] for v in values], "o-", label=strategy)
            axis.set(xlabel="Mean simulated currency cost", ylabel=metric)
    axes[1].legend(fontsize=7)
    fig.suptitle("Exploratory historical controlled reveal (290 previously seen matches)")
    fig.tight_layout()
    fig.savefig(output / "loss-cost-curves.png", dpi=160)
    plt.close(fig)
    bins = report["gain_calibration"]["bins"]
    fig, ax = plt.subplots(figsize=(6, 5))
    x, y = [v["mean_estimated_gain"] for v in bins], [v["mean_realized_delta"] for v in bins]
    ax.plot(x, y, "o-")
    bounds = [min(x + y), max(x + y)]
    ax.plot(bounds, bounds, "k--")
    ax.set(xlabel="Estimated log-loss gain (bin mean)", ylabel="Realized out-of-time loss gain",
           title="Empirical gain calibration; diagnostic only")
    fig.tight_layout()
    fig.savefig(output / "gain-calibration.png", dpi=160)
    plt.close(fig)
