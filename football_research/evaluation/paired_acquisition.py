"""Paired acquisition comparisons on reused, controlled historical evidence.

This consolidates the existing round-eight reference policies and strict VoI
runner. A model factory belongs outside this module: every policy in one call
uses exactly that frozen predictor, evidence catalogue, initial state and
resource limit. Counterfactual return diagnostics are evaluator-only and are
never supplied to a policy. These are development results, not a new holdout.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from ..acquisition.experiment import losses, outcome, state_after
from ..acquisition.experiment_round8 import run_episode as run_reference
from ..acquisition.joint_planner import UNITS, _budget, _cost
from ..acquisition.joint_returns import entropy
from ..acquisition.joint_voi import LearnedJointVOIPolicy
from ..acquisition.state import parse_time
from .learned_voi_replay import run_learned_voi_episode, score_reconstructed_trajectory

STRATEGIES = ("no-query", "fixed-order", "all-within-budget", "uniform-random", "strict-voi")
REFERENCE_STRATEGIES = {"no-query": "base-only", "fixed-order": "fixed-order",
                        "all-within-budget": "all-available", "uniform-random": "random"}
EVIDENCE_GRADE = "reused_historical_reconstruction_development"


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _entropy_band(p):
    h = entropy(p)
    return "lower" if h < .9 else "intermediate" if h < 1.03 else "higher"


def _context(p):
    return {"entropy": entropy(p), "entropy_band": _entropy_band(p),
            "favorite": ("home", "draw", "away")[int(np.argmax(p))],
            "favorite_probability": max(p),
            "confidence_band": "strong_favorite" if max(p) >= .6 else "no_strong_favorite"}


def _validate(rows, actions, limits, seeds, lambda_money):
    limits, _ = _budget({"limits": limits, "spent": {}})
    if set(limits) - {"money", "requests"}:
        raise ValueError("Existing reference policies support money/request limits only")
    if set(limits) != {"money", "requests"}:
        raise ValueError("Declare identical money and request limits for every policy")
    if not rows or len({r["match_id"] for r in rows}) != len(rows):
        raise ValueError("Require nonempty unique paired match identities")
    if not actions or len({a["action_id"] for a in actions}) != len(actions):
        raise ValueError("Require nonempty unique action identities")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("Declare distinct integer random seeds")
    if isinstance(lambda_money, bool) or not math.isfinite(lambda_money) or lambda_money < 0:
        raise ValueError("Nonnegative finite cost conversion required")
    for action in actions:
        _cost(action)
    if any(r["information_mode"] != "historical_reconstruction" for r in rows):
        raise ValueError("This evaluator requires explicit historical reconstruction")
    return limits


def diagnostic_transitions(rows, model, actions, *, policy, limits, model_name, fold):
    """Enumerate all declared one-step contexts after actors have finished.

    Every match contributes an empty state and each one-action state. Outcomes
    are used only here to diagnose model error; no hindsight best action is
    presented as an executable or unbiased oracle strategy.
    """
    transitions = []
    ids = [a["action_id"] for a in actions]
    acquired_sets = [()] + [(aid,) for aid in ids]
    for row in rows:
        label = outcome(row)
        for acquired in acquired_sets:
            before = state_after(row, actions, acquired)
            before["action_history"] = [{"action_id": aid} for aid in acquired]
            spent = {u: sum(_cost(a)[u] for a in actions if a["action_id"] in acquired) for u in UNITS}
            # A state reachable only under a larger budget is not a feasible
            # choice point in this experiment, and is not included.
            if any(spent[u] > cap + 1e-12 for u, cap in limits.items()):
                continue
            decision = policy.plan(before, actions, {"limits": limits, "spent": spent},
                                   candidate_generation_complete=False)
            prior = decision["prediction"]["probabilities"]
            for action in actions:
                aid = action["action_id"]
                if aid in acquired:
                    continue
                candidate = next(c for c in decision["candidates"] if c["action_id"] == aid)
                if not candidate["feasible"]:
                    continue
                after = state_after(row, actions, (*acquired, aid))
                posterior = model.predict_partial(after)["probabilities"]
                token = model.observed_categories(after)[aid]
                value = candidate["value"]
                branches = candidate.get("branches")
                branch = next((b for b in branches or [] if b["branch_id"] == token), None)
                realized = losses(prior, label)["log_loss"] - losses(posterior, label)["log_loss"]
                predicted = value["gross_evsi"]
                transitions.append({"model_name": model_name, "fold": fold,
                    "match_id": row["match_id"], "kickoff": row["kickoff"],
                    "model_version": model.model_version, "limits": limits,
                    "action_id": aid, "acquired_before": list(acquired),
                    "state_context": _context(prior), "return_category": token,
                    "value_status": value["status"], "value_reason": value["reason"],
                    "predicted_evsi": predicted, "net_voi": value["net_voi"],
                    "actual_log_loss_reduction": realized,
                    "evsi_minus_realized_gain": None if predicted is None else predicted - realized,
                    "return_probability": None if branch is None else branch["probability"],
                    "return_log_loss": None if branch is None else -math.log(max(branch["probability"], 1e-15)),
                    "selected_at_this_context": decision["selection"].get("candidate_id") == aid,
                    "observation_kind": "source_missing" if token == "missing" else "observed_numeric_summary",
                    "scope": "evaluator_only_all_feasible_declared_transitions_not_selected_query_calibration"})
    return transitions


def evaluate_acquisition_policies(rows, model, actions, *, limits, lambda_money,
                                  policy_factory=None, seeds=(0, 1, 2), output=None,
                                  model_name="frozen", fold=None, diagnostics=True):
    """Stable integration entrypoint for a frozen predictor and policy factory.

    ``policy_factory(model, lambda_cost={"money": lambda_money})`` defaults to
    LearnedJointVOIPolicy. Its actor interface is the existing plan/verify_update
    protocol. New probability models can subclass JointReturnPredictor without
    changing the simulator, scorers or baseline policies.
    """
    rows, actions, seeds = list(rows), deepcopy(list(actions)), tuple(seeds)
    limits = _validate(rows, actions, limits, seeds, lambda_money)
    factory = policy_factory or LearnedJointVOIPolicy
    policy = factory(model, lambda_cost={"money": lambda_money})
    if policy.model is not model:
        raise ValueError("All strategies must use the same supplied frozen predictor")
    results = []
    writer = None
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        writer = (output / "policy-trajectories.jsonl").open("w", encoding="utf-8")
    try:
        for row in sorted(rows, key=lambda r: (parse_time(r["kickoff"]), r["match_id"])):
            # The strict actor completes before any result is used for scoring.
            trajectory = run_learned_voi_episode(row, policy, actions, limits=limits)
            evaluation = score_reconstructed_trajectory(trajectory, row)
            prior = trajectory["initial_probabilities"]
            identity = {"model_name": model_name, "fold": fold, "match_id": row["match_id"],
                "kickoff": row["kickoff"], "prediction_cutoff": row["prediction_cutoff"],
                "model_version": model.model_version, "limits": limits, "lambda_money": lambda_money,
                "initial_probabilities": prior, "state_context": _context(prior),
                "evidence_grade": EVIDENCE_GRADE, "catalog_sha256": _digest(actions),
                "actual_external_requests": 0, "actual_llm_tokens": 0}
            strict = {**identity, "strategy": "strict-voi", "seed": None,
                "final_probabilities": trajectory["final_probabilities"],
                "actions": trajectory["actions"], "cost": trajectory["cost"],
                "stop_reason": trajectory["terminal_selection"]["reason"],
                **evaluation["after_loss"]}
            strict["utility_loss"] = strict["log_loss"] + lambda_money * strict["cost"]["money"]
            results.append(strict)
            if writer:
                writer.write(json.dumps({**identity, "strategy": "strict-voi", "seed": None,
                                         "trajectory": trajectory}, allow_nan=False) + "\n")
            for strategy, original in REFERENCE_STRATEGIES.items():
                for seed in seeds if strategy == "uniform-random" else (0,):
                    record, events = run_reference(row, model, actions, strategy=original,
                        budget_money=limits["money"], lambda_money=lambda_money, seed=seed,
                        design={"request_budget": limits["requests"]})
                    if record["initial_probabilities"] != prior:
                        raise ValueError("Paired policies changed the initial forecast")
                    if any(record["cost"][u] > cap + 1e-12 for u, cap in limits.items()):
                        raise ValueError("Reference strategy exceeded the shared physical budget")
                    result = {**identity, "strategy": strategy,
                        "seed": seed if strategy == "uniform-random" else None,
                        **{k: record[k] for k in ("final_probabilities", "actions", "cost", "stop_reason",
                                                 "log_loss", "brier_score", "accuracy")}}
                    result["utility_loss"] = result["log_loss"] + lambda_money * result["cost"]["money"]
                    results.append(result)
                    if writer:
                        writer.write(json.dumps({**identity, "strategy": strategy, "seed": result["seed"],
                            "trajectory": events}, allow_nan=False) + "\n")
    finally:
        if writer:
            writer.close()
    transitions = diagnostic_transitions(rows, model, actions, policy=policy, limits=limits,
        model_name=model_name, fold=fold) if diagnostics else []
    summary = summarize_acquisition_evaluations(results, transitions)
    if output is not None:
        _write(output / "paired-episodes.json", results)
        _write(output / "response-diagnostics.json", transitions)
        _write(output / "report.json", summary)
    return {"episodes": results, "transitions": transitions, "summary": summary}


def summarize_acquisition_evaluations(records, transitions=()):
    """Equal-weight matches; random seeds are averaged within each match."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["model_name"], _digest(r["limits"]), r["strategy"])
        grouped[key][r["match_id"]].append(r)
    if not grouped:
        raise ValueError("No policy records")
    average, metrics = {}, []
    for key, matches in sorted(grouped.items()):
        model_name, _, strategy = key
        flat = [r for rows in matches.values() for r in rows]
        seeds_expected = {r["seed"] for r in next(iter(matches.values()))}
        for rows in matches.values():
            if len(rows) != len({r["seed"] for r in rows}) or {r["seed"] for r in rows} != seeds_expected:
                raise ValueError("Every paired match requires the same unique seed set")
            if len({_digest(r["initial_probabilities"]) for r in rows}) != 1:
                raise ValueError("Random seeds changed the initial prediction")
        by_match = {}
        for mid, rows in matches.items():
            by_match[mid] = {k: float(np.mean([r[k] for r in rows]))
                            for k in ("log_loss", "brier_score", "accuracy", "utility_loss")}
            by_match[mid].update({u: float(np.mean([r["cost"][u] for r in rows])) for u in UNITS})
            by_match[mid].update(kickoff=rows[0]["kickoff"], initial=rows[0]["initial_probabilities"],
                                 catalog=rows[0]["catalog_sha256"], lambda_money=rows[0]["lambda_money"])
        average[key] = by_match
        metric = {"model_name": model_name, "strategy": strategy, "limits": flat[0]["limits"],
                  "unique_matches": len(matches), "episodes": len(flat),
                  "mean_log_loss": float(np.mean([r["log_loss"] for r in by_match.values()])),
                  "mean_brier_score": float(np.mean([r["brier_score"] for r in by_match.values()])),
                  "mean_utility_loss": float(np.mean([r["utility_loss"] for r in by_match.values()])),
                  "mean_cost": float(np.mean([r["money"] for r in by_match.values()])),
                  "mean_queries": float(np.mean([r["requests"] for r in by_match.values()])),
                  "paths": dict(Counter(" -> ".join(r["actions"]) or "no_query" for r in flat)),
                  "first_actions": dict(Counter(r["actions"][0] if r["actions"] else "no_query" for r in flat)),
                  "stop_reasons": dict(Counter(r["stop_reason"] for r in flat)),
                  "aggregation": "seed mean within match, then equal-weight matches"}
        metrics.append(metric)
    comparisons = []
    families = {(m, b) for m, b, _ in average}
    equivalences = []
    for model_name, budget_key in sorted(families):
        keys = {(m, b, s) for m, b, s in average if (m, b) == (model_name, budget_key)}
        if {k[2] for k in keys} != set(STRATEGIES):
            raise ValueError("All five required policies must be present")
        a = average[(model_name, budget_key, "strict-voi")]
        for strategy in STRATEGIES:
            b = average[(model_name, budget_key, strategy)]
            if set(a) != set(b):
                raise ValueError("Paired policy match identities differ")
            blocks = defaultdict(list)
            for mid in sorted(a):
                if any(a[mid][k] != b[mid][k] for k in ("kickoff", "initial", "catalog", "lambda_money")):
                    raise ValueError("Paired policies changed state, catalogue or cost conversion")
                blocks[parse_time(a[mid]["kickoff"]).isocalendar()[:2]].append([
                    a[mid]["log_loss"] - b[mid]["log_loss"], a[mid]["utility_loss"] - b[mid]["utility_loss"]])
            if strategy == "strict-voi":
                continue
            arrays = [np.asarray(blocks[k]) for k in sorted(blocks)]
            sums, counts = np.array([x.sum(axis=0) for x in arrays]), np.array([len(x) for x in arrays])
            draws = np.random.default_rng(1729).integers(0, len(arrays), size=(1000, len(arrays)))
            boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
            diff = sums.sum(axis=0) / counts.sum()
            sample = next(r for r in records if r["model_name"] == model_name and _digest(r["limits"]) == budget_key)
            comparisons.append({"model_name": model_name, "limits": sample["limits"],
                "strategy": "strict-voi", "reference": strategy, "unique_matches": len(a),
                "week_blocks": len(arrays), "mean_log_loss_difference": float(diff[0]),
                "mean_utility_loss_difference": float(diff[1]),
                "log_loss_interval_95_descriptive": np.quantile(boot[:, 0], [.025, .975]).tolist(),
                "utility_interval_95_descriptive": np.quantile(boot[:, 1], [.025, .975]).tolist(),
                "scope": "paired UTC-week bootstrap; reused development data; no adaptive-selection or multiplicity correction"})
        fixed = {r["match_id"]: r for r in records if r["model_name"] == model_name
                 and _digest(r["limits"]) == budget_key and r["strategy"] == "fixed-order"}
        all_budget = {r["match_id"]: r for r in records if r["model_name"] == model_name
                      and _digest(r["limits"]) == budget_key and r["strategy"] == "all-within-budget"}
        equivalent = all(all(fixed[mid][k] == all_budget[mid][k] for k in
                           ("actions", "cost", "final_probabilities")) for mid in fixed)
        equivalences.append({"model_name": model_name, "limits": next(iter(fixed.values()))["limits"],
                             "policies": ["fixed-order", "all-within-budget"], "identical": equivalent,
                             "interpretation": "Both existing reference policies traverse the declared affordable catalogue order; not independent algorithm evidence"})
    diagnostics = []
    by_transition = defaultdict(list)
    for r in transitions:
        group = (r["model_name"], _digest(r["limits"]), r["action_id"],
                 " -> ".join(r["acquired_before"]) or "initial", r["state_context"]["entropy_band"], r["return_category"])
        by_transition[group].append(r)
    for key, rows in sorted(by_transition.items()):
        computed = [r for r in rows if r["predicted_evsi"] is not None]
        diagnostics.append({"model_name": key[0], "limits": rows[0]["limits"], "action_id": key[2],
            "acquired_before": key[3], "entropy_band": key[4], "return_category": key[5],
            "unique_matches": len({r["match_id"] for r in rows}), "computed": len(computed),
            "mean_predicted_evsi": float(np.mean([r["predicted_evsi"] for r in computed])) if computed else None,
            "mean_realized_gain_same_computed_rows": float(np.mean([r["actual_log_loss_reduction"] for r in computed])) if computed else None,
            "mean_realized_gain_all_rows": float(np.mean([r["actual_log_loss_reduction"] for r in rows])),
            "mean_evsi_minus_realized_gain": float(np.mean([r["evsi_minus_realized_gain"] for r in computed])) if computed else None,
            "negative_realized_gain_fraction": float(np.mean([r["actual_log_loss_reduction"] < 0 for r in rows])),
            "unknown_reasons": dict(Counter(r["value_reason"] for r in rows if r["predicted_evsi"] is None)),
            "scope": "descriptive actual-return strata; return type is hindsight-only and not a prequery state feature"})
    behavior = []
    behavior_groups = defaultdict(list)
    for r in records:
        if r["strategy"] == "strict-voi":
            behavior_groups[(r["model_name"], _digest(r["limits"]), r["state_context"]["entropy_band"],
                             r["state_context"]["confidence_band"])].append(r)
    for key, rows in sorted(behavior_groups.items()):
        behavior.append({"model_name": key[0], "limits": rows[0]["limits"], "entropy_band": key[2],
                         "confidence_band": key[3], "matches": len(rows),
                         "mean_queries": float(np.mean([r["cost"]["requests"] for r in rows])),
                         "first_actions": dict(Counter(r["actions"][0] if r["actions"] else "no_query" for r in rows)),
                         "mean_log_loss": float(np.mean([r["log_loss"] for r in rows]))})
    return {"schema": "paired-acquisition-evaluation-v1", "evidence_grade": EVIDENCE_GRADE,
            "metrics": metrics, "paired_comparisons": comparisons, "policy_equivalences": equivalences,
            "state_adaptiveness": behavior, "return_mismatch_diagnostics": diagnostics,
            "not_run": [{"strategy": "llm-only", "reason": "No actual frozen LLM policy executed"},
                        {"strategy": "oracle", "reason": "Hindsight return analysis is not an executable policy"}],
            "scope": {"costs": "identical simulated projection costs; not provider bills",
                      "outcomes": "same-provider reconstructed regulation results, not independent gold",
                      "open_world_benefit_proven": False, "new_holdout": False,
                      "numeric_voi_training_eligible": False, "external_network_requests": 0,
                      "calibration": "Observed failures diagnose model misspecification; one realization is not a true conditional EVSI label"}}
