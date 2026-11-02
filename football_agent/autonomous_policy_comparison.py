"""Evaluator-side comparison of information-acquisition policies.

The comparison is deliberately separate from the deployable policy surface.  A
policy receives the public scenario manifest and returns a trace through
``run_policy``.  This module then scores that trace against the evaluator-only
realisation to measure risk, cost, stopping efficiency, hindsight regret, and
the aggregate cost-risk Pareto frontier.  The ``oracle_upper_bound`` is a
hindsight upper bound: it enumerates candidate subsets using the private
realisation and is never an executable Agent policy.

The distinction matters for FootballMind's claim.  A lower Brier score by a
clairvoyant oracle is a measurement reference, not evidence that the Agent saw
the future.  Public manifests, query requests, context transitions, and this
module's public projection contain no outcome or response tape.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import permutations
import json
import math
from typing import Any

from .autonomous_acquisition_eval import (
    OUTCOMES,
    POLICIES,
    QuerySpec,
    Scenario,
    build_scenarios,
    run_policy,
    _brier,
)


VERSION = "autonomous-policy-comparison-v1"
COMPARISON_POLICIES = ("fixed_all", "heuristic", "voi_gate", "oracle_upper_bound")
_EPSILON = 1e-12
_LOG_EPSILON = 1e-15


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _response_status(scenario: Scenario, query: QuerySpec) -> str:
    """Mirror envelope completion semantics in the evaluator-only layer."""
    response = query.responses[scenario.private_response_labels[query.candidate_id]]
    if response.status is not None:
        return response.status
    if response.error_code:
        return "FAILED"
    available = response.operational_available_at or response.observed_at
    if available is not None and available > scenario.prediction_cutoff:
        return "LATE"
    if not response.context_updates:
        return "AMBIGUOUS"
    return "SUCCEEDED"


def _simulate_sequence(scenario: Scenario, selected: tuple[str, ...]) -> dict[str, Any]:
    """Score a private candidate sequence without putting truth in policy state."""
    by_id = {query.candidate_id: query for query in scenario.queries}
    prediction = tuple(scenario.initial_prediction)
    statuses: list[str] = []
    cost = 0.0
    for candidate_id in selected:
        query = by_id[candidate_id]
        response = query.responses[scenario.private_response_labels[candidate_id]]
        status = _response_status(scenario, query)
        statuses.append(status)
        cost += float(query.cost)
        if status == "SUCCEEDED" and response.prediction is not None:
            prediction = tuple(response.prediction)
    initial_loss = _brier(scenario.initial_prediction, scenario.private_outcome)
    final_loss = _brier(prediction, scenario.private_outcome)
    utility = initial_loss - final_loss - cost
    return {
        "selected_candidates": selected,
        "statuses": tuple(statuses),
        "final_prediction": prediction,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "acquisition_cost": cost,
        "query_count": len(selected),
        "net_utility": utility,
    }


def _oracle_run(scenario: Scenario) -> dict[str, Any]:
    """Return the hindsight best sequence, including the empty sequence."""
    candidate_ids = tuple(query.candidate_id for query in scenario.queries)
    candidates: list[dict[str, Any]] = []
    for length in range(len(candidate_ids) + 1):
        if length == 0:
            candidates.append(_simulate_sequence(scenario, ()))
            continue
        candidates.extend(_simulate_sequence(scenario, sequence) for sequence in permutations(candidate_ids, length))
    best = max(
        candidates,
        key=lambda row: (
            row["net_utility"],
            -row["acquisition_cost"],
            -row["query_count"],
            tuple(reversed(row["selected_candidates"])),
        ),
    )
    output = dict(best)
    output["policy"] = "oracle_upper_bound"
    output["stop_reason"] = "hindsight_exhaustive_subset_upper_bound"
    output["replan_reasons"] = ()
    output["evaluator_only"] = True
    return output


def _policy_run(scenario: Scenario, policy: str) -> dict[str, Any]:
    if policy == "heuristic":
        return run_policy(scenario, "first_query").as_dict()
    if policy in ("fixed_all", "voi_gate"):
        return run_policy(scenario, policy).as_dict()
    if policy == "oracle_upper_bound":
        return _oracle_run(scenario)
    raise ValueError(f"unsupported comparison policy: {policy}")


def _log_loss(prediction: Iterable[float], outcome: str) -> float:
    index = OUTCOMES.index(outcome)
    probability = max(float(tuple(prediction)[index]), _LOG_EPSILON)
    return -math.log(probability)


def _metric_row(scenario: Scenario, policy: str, run: Mapping[str, Any], oracle: Mapping[str, Any]) -> dict[str, Any]:
    prediction = tuple(float(item) for item in run["final_prediction"])
    if len(prediction) != len(OUTCOMES):
        raise ValueError("final_prediction must cover all outcome classes")
    confidence = max(prediction)
    outcome_index = OUTCOMES.index(scenario.private_outcome)
    correct = prediction.index(max(prediction)) == outcome_index
    initial_loss = _finite(run["initial_loss"], "initial_loss")
    final_loss = _finite(run["final_loss"], "final_loss")
    cost = _finite(run["acquisition_cost"], "acquisition_cost")
    gain = initial_loss - final_loss
    oracle_utility = _finite(oracle["net_utility"], "oracle.net_utility")
    utility = _finite(run["net_utility"], "net_utility")
    return {
        "scenario_id": scenario.scenario_id,
        "policy": policy,
        "initial_brier": initial_loss,
        "final_brier": final_loss,
        "log_loss": _log_loss(prediction, scenario.private_outcome),
        "confidence": confidence,
        "correct_top_class": correct,
        "confidence_error": abs(confidence - float(correct)),
        "acquisition_cost": cost,
        "query_count": int(run["query_count"]),
        "query_fraction": len(run["selected_candidates"]) / max(len(scenario.queries), 1),
        "cost_fraction": cost / max(sum(query.cost for query in scenario.queries), _EPSILON),
        "brier_gain": gain,
        "net_utility": utility,
        "regret_vs_oracle": max(0.0, oracle_utility - utility),
        "oracle_utility": oracle_utility,
        "oracle_optimal": abs(oracle_utility - utility) <= 1e-10,
        "selected_candidates": list(run["selected_candidates"]),
        "statuses": list(run["statuses"]),
        "stop_reason": str(run["stop_reason"]),
        "evaluator_only_metric": True,
        "hidden_result_included": False,
    }


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    return math.fsum(float(row[key]) for row in rows) / max(len(rows), 1)


def _calibration_bins(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Small-sample reliability summary; Brier remains the primary risk score."""
    bins: list[dict[str, Any]] = []
    for lower, upper in ((0.0, 0.5), (0.5, 0.7), (0.7, 1.0 + _EPSILON)):
        selected = [row for row in rows if lower <= row["confidence"] < upper]
        if not selected:
            continue
        mean_confidence = _mean(selected, "confidence")
        accuracy = math.fsum(float(row["correct_top_class"]) for row in selected) / len(selected)
        bins.append({
            "lower": lower, "upper": upper, "count": len(selected),
            "mean_confidence": mean_confidence, "empirical_accuracy": accuracy,
            "absolute_gap": abs(mean_confidence - accuracy),
        })
    return bins


def _aggregate(policy: str, rows: list[Mapping[str, Any]], fixed_cost: float) -> dict[str, Any]:
    total_cost = math.fsum(float(row["acquisition_cost"]) for row in rows)
    total_queries = math.fsum(float(row["query_count"]) for row in rows)
    total_gain = math.fsum(float(row["brier_gain"]) for row in rows)
    return {
        "policy": policy,
        "scenario_count": len(rows),
        "mean_final_brier": _mean(rows, "final_brier"),
        "mean_log_loss": _mean(rows, "log_loss"),
        "mean_confidence_error": _mean(rows, "confidence_error"),
        "top_class_accuracy": math.fsum(float(row["correct_top_class"]) for row in rows) / max(len(rows), 1),
        "mean_acquisition_cost": _mean(rows, "acquisition_cost"),
        "total_acquisition_cost": total_cost,
        "mean_query_count": _mean(rows, "query_count"),
        "mean_query_fraction": _mean(rows, "query_fraction"),
        "mean_cost_fraction": _mean(rows, "cost_fraction"),
        "zero_query_rate": sum(row["query_count"] == 0 for row in rows) / max(len(rows), 1),
        "mean_brier_gain": _mean(rows, "brier_gain"),
        "mean_net_utility": _mean(rows, "net_utility"),
        "mean_regret_vs_oracle": _mean(rows, "regret_vs_oracle"),
        "oracle_optimal_rate": sum(bool(row["oracle_optimal"]) for row in rows) / max(len(rows), 1),
        "cost_savings_vs_fixed_all": None if fixed_cost <= _EPSILON else 1.0 - total_cost / fixed_cost,
        "brier_gain_per_cost": None if total_cost <= _EPSILON else total_gain / total_cost,
        "calibration_bins": _calibration_bins(rows),
        "evaluator_only_metric": True,
    }


def _pareto(aggregates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = tuple(aggregates)
    dominated_by: dict[str, list[str]] = {}
    for name in names:
        point = aggregates[name]
        dominators = []
        for other in names:
            if other == name:
                continue
            candidate = aggregates[other]
            no_worse = (
                candidate["mean_acquisition_cost"] <= point["mean_acquisition_cost"] + _EPSILON
                and candidate["mean_final_brier"] <= point["mean_final_brier"] + _EPSILON
            )
            strictly_better = (
                candidate["mean_acquisition_cost"] < point["mean_acquisition_cost"] - _EPSILON
                or candidate["mean_final_brier"] < point["mean_final_brier"] - _EPSILON
            )
            if no_worse and strictly_better:
                dominators.append(other)
        dominated_by[name] = sorted(dominators)
    frontier = [name for name in names if not dominated_by[name]]
    return {
        "objective": "minimize_mean_acquisition_cost_and_mean_final_brier",
        "frontier": frontier,
        "dominated_by": dominated_by,
        "points": {
            name: {
                "mean_acquisition_cost": aggregates[name]["mean_acquisition_cost"],
                "mean_final_brier": aggregates[name]["mean_final_brier"],
            }
            for name in names
        },
    }


def compare_scenarios(scenarios: Iterable[Scenario] | None = None) -> dict[str, Any]:
    scenarios = tuple(build_scenarios() if scenarios is None else scenarios)
    if not scenarios:
        raise ValueError("at least one scenario is required")
    public_rows: list[dict[str, Any]] = []
    metric_rows: dict[str, list[dict[str, Any]]] = {policy: [] for policy in COMPARISON_POLICIES}
    for scenario in scenarios:
        oracle = _oracle_run(scenario)
        policy_rows = {}
        for policy in COMPARISON_POLICIES:
            run = _policy_run(scenario, policy)
            row = _metric_row(scenario, policy, run, oracle)
            metric_rows[policy].append(row)
            policy_rows[policy] = row
        public_rows.append({
            "scenario": scenario.public_manifest(),
            "policies": policy_rows,
            "hidden_result_included": False,
        })
    fixed_cost = math.fsum(float(row["acquisition_cost"]) for row in metric_rows["fixed_all"])
    aggregates = {
        policy: _aggregate(policy, rows, fixed_cost)
        for policy, rows in metric_rows.items()
    }
    return {
        "schema_version": VERSION,
        "policy_order": list(COMPARISON_POLICIES),
        "scenarios": public_rows,
        "aggregate": aggregates,
        "pareto": _pareto(aggregates),
        "evaluator_only": {
            "scope": "synthetic mechanism evaluation",
            "truth_in_policy_view": False,
            "oracle_definition": "hindsight exhaustive subset upper bound using private evaluator realisation",
            "calibration_note": "confidence error and reliability bins are evaluator metrics; Brier is the primary proper risk score",
        },
        "hidden_result_included": False,
    }


def reviewer_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a reviewer-safe scorecard, explicitly separated from runtime input.

    Brier, calibration, and regret are computed with evaluator-only labels.  A
    reviewer may inspect this projection, while a deployed Agent must consume
    :func:`runtime_projection` instead.
    """
    if report.get("schema_version") != VERSION:
        raise ValueError("unexpected comparison schema")
    projection = {
        "schema_version": VERSION,
        "projection_scope": "reviewer_evaluation_only",
        "runtime_safe": False,
        "policy_order": list(report.get("policy_order", [])),
        "scenarios": report.get("scenarios", []),
        "aggregate": report.get("aggregate", {}),
        "pareto": report.get("pareto", {}),
        "hidden_result_included": False,
    }
    encoded = json.dumps(projection, ensure_ascii=False).lower()
    for forbidden in ('"outcome"', '"score"', '"result"', '"home_score"', '"away_score"', '"sampled_result"'):
        if forbidden in encoded:
            raise ValueError("public comparison projection contains evaluator-only key")
    return projection


def runtime_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only projection suitable for a policy/runtime input.

    It contains public manifests and the declared policy registry, but excludes
    every outcome-derived score, hindsight choice, and evaluator annotation.
    """
    if report.get("schema_version") != VERSION:
        raise ValueError("unexpected comparison schema")
    scenarios = []
    for row in report.get("scenarios", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("scenario"), Mapping):
            raise ValueError("comparison scenario row is malformed")
        scenarios.append({"scenario": dict(row["scenario"])})
    projection = {
        "schema_version": VERSION,
        "projection_scope": "runtime_public_context",
        "runtime_safe": True,
        "policy_order": [policy for policy in report.get("policy_order", []) if policy != "oracle_upper_bound"],
        "scenarios": scenarios,
        "truth_in_policy_view": False,
        "hidden_result_included": False,
    }
    encoded = json.dumps(projection, ensure_ascii=False).lower()
    for forbidden in ('"true_outcome"', '"private_outcome"', '"private_response_labels"', '"outcome"',
                      '"final_brier"', '"log_loss"', '"regret_vs_oracle"', '"oracle_upper_bound"'):
        if forbidden in encoded:
            raise ValueError("runtime comparison projection contains evaluator-only field")
    return projection


# Backward-compatible name used by the first Round83 replay; reviewer output is
# intentionally retained as a separate scorecard and is marked runtime_safe=False.
public_projection = reviewer_projection


__all__ = [
    "VERSION", "COMPARISON_POLICIES", "compare_scenarios", "reviewer_projection",
    "runtime_projection", "public_projection",
]
