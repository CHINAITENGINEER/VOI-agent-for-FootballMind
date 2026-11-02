"""Evaluator-only paired policy metrics for historical autonomous replays.

The historical as-of protocol answers a visibility question: could a policy
have seen each fact at its prediction cutoff?  This module answers the paired
evaluation question that follows it: on the *same* bound match/cutoff, how did
different acquisition policies trade proper scoring loss against acquisition
cost?

The input is deliberately an episode sidecar.  It is never a runtime policy
input.  Outcome labels are read from the separately audited historical
snapshot index, probabilities are rescored, and any exported loss that does
not agree with that recomputation is reported as an issue.  Rows without an
exact ``match_id + prediction_cutoff`` binding are excluded explicitly rather
than silently joining by team names or row order.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from .historical_protocol import EpisodeFold


VERSION = "historical-policy-metrics-v1"
LOG_EPSILON = 1e-15
METRIC_TOLERANCE = 1e-8


@dataclass(frozen=True)
class MetricIssue:
    code: str
    severity: str
    match_id: str | None
    path: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "match_id": self.match_id,
            "path": self.path,
            "detail": self.detail,
        }


def _utc(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _issue(issues: list[MetricIssue], code: str, severity: str, match_id: str | None,
           path: str, detail: str) -> None:
    issues.append(MetricIssue(code, severity, match_id, path, detail))


def _key(match_id: Any, cutoff: Any) -> tuple[str, datetime]:
    if match_id is None or not str(match_id).strip():
        raise ValueError("match_id must be nonempty")
    return str(match_id), _utc(cutoff, "prediction_cutoff")


def _outcome_index(row: Mapping[str, Any]) -> int:
    home, away = row.get("home_goals"), row.get("away_goals")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (home, away)):
        raise ValueError("audited snapshot needs nonnegative integer home_goals/away_goals")
    return 0 if home > away else 1 if home == away else 2


def _probabilities(row: Mapping[str, Any], path: str) -> tuple[float, float, float]:
    raw = row.get("final_probabilities", row.get("probabilities"))
    if raw is None:
        names = ("home_win_probability", "draw_probability", "away_win_probability")
        raw = [row.get(name) for name in names]
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{path}.final_probabilities must contain three values")
    values = tuple(_finite(value, f"{path}.final_probabilities[{index}]") for index, value in enumerate(raw))
    if any(value < 0 or value > 1 for value in values):
        raise ValueError(f"{path}.final_probabilities values must be in [0, 1]")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-8, rel_tol=0.0):
        raise ValueError(f"{path}.final_probabilities must sum to one")
    return values  # type: ignore[return-value]


def _losses(probabilities: tuple[float, float, float], target: int) -> tuple[float, float]:
    brier = sum((value - int(index == target)) ** 2 for index, value in enumerate(probabilities))
    log_loss = -math.log(max(probabilities[target], LOG_EPSILON))
    return brier, log_loss


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _mean(values: Iterable[float]) -> float | None:
    values = tuple(float(value) for value in values)
    return None if not values else math.fsum(values) / len(values)


def _cost_receipt(row: Mapping[str, Any], path: str) -> tuple[float, float, str]:
    nested = row.get("cost")
    if isinstance(nested, Mapping):
        money = nested.get("money", 0.0)
        requests = nested.get("requests", 0.0)
        if money is not None and requests is not None:
            return _finite(money, f"{path}.cost.money"), _finite(requests, f"{path}.cost.requests"), "cost"
    # Compatibility with old exports.  The field is retained in the report as
    # a fallback marker; it is never confused with ``budget_money``.
    return (
        0.0,
        _finite(row.get("actual_external_requests", 0.0), f"{path}.actual_external_requests"),
        "actual_external_requests",
    )


def _snapshot_index(snapshots: Iterable[Mapping[str, Any]], issues: list[MetricIssue]) -> dict[tuple[str, datetime], dict[str, Any]]:
    result: dict[tuple[str, datetime], dict[str, Any]] = {}
    labels: dict[str, tuple[int, int]] = {}
    for index, row in enumerate(snapshots):
        path = f"snapshots[{index}]"
        try:
            key = _key(row.get("match_id") or row.get("canonical_match_id"), row.get("prediction_cutoff"))
            kickoff = _utc(row.get("kickoff"), f"{path}.kickoff")
            outcome = _outcome_index(row)
        except (TypeError, ValueError) as exc:
            _issue(issues, "INVALID_SNAPSHOT", "error", None, path, str(exc))
            continue
        label = (row.get("home_goals"), row.get("away_goals"))
        previous_label = labels.get(key[0])
        if previous_label is not None and previous_label != label:
            _issue(issues, "CONFLICTING_SNAPSHOT_LABEL", "error", key[0], path, "same match has conflicting labels")
        labels[key[0]] = label  # type: ignore[assignment]
        existing = result.get(key)
        if existing is not None:
            if existing["kickoff"] != kickoff or existing["outcome"] != outcome:
                _issue(issues, "CONFLICTING_SNAPSHOT_BINDING", "error", key[0], path, "duplicate key has conflicting kickoff or label")
            continue
        result[key] = {
            "match_id": key[0], "cutoff": key[1], "kickoff": kickoff,
            "outcome": outcome, "home_goals": row.get("home_goals"), "away_goals": row.get("away_goals"),
        }
    return result


def _test_fold_map(folds: Sequence[EpisodeFold], issues: list[MetricIssue]) -> dict[str, int]:
    result: dict[str, int] = {}
    for fold in folds:
        for match_id in fold.test_match_ids:
            previous = result.get(str(match_id))
            if previous is not None and previous != fold.index:
                _issue(issues, "MATCH_IN_MULTIPLE_TEST_FOLDS", "error", str(match_id), "folds", "test partitions overlap")
            result[str(match_id)] = fold.index
    return result


def _bootstrap_ci(
    values_by_match: Mapping[str, float], *, samples: int, seed: int, confidence: float,
) -> list[float] | None:
    if not values_by_match or samples < 1 or not 0 < confidence < 1:
        return None
    match_ids = sorted(values_by_match)
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values_by_match[match_ids[rng.randrange(len(match_ids))]] for _ in match_ids]
        means.append(math.fsum(draw) / len(draw))
    tail = (1.0 - confidence) / 2.0
    return [_percentile(means, tail), _percentile(means, 1.0 - tail)]


def _metric_summary(rows_by_match: Mapping[str, Mapping[str, float]], *, samples: int, seed: int,
                    confidence: float) -> dict[str, Any]:
    metrics = ("brier_score", "log_loss", "cost_money", "requests")
    output: dict[str, Any] = {"matches": len(rows_by_match)}
    for metric_index, metric in enumerate(metrics):
        values = {match_id: float(row[metric]) for match_id, row in rows_by_match.items()}
        output[f"mean_{metric}"] = _mean(values.values())
        output[f"{metric}_confidence_interval"] = _bootstrap_ci(
            values, samples=samples, seed=seed + metric_index * 1009, confidence=confidence,
        )
    return output


def compare_historical_policies(
    episodes: Iterable[Mapping[str, Any]],
    snapshots: Iterable[Mapping[str, Any]],
    *,
    folds: Sequence[EpisodeFold] = (),
    baseline_strategy: str = "base-only",
    test_only: bool = True,
    budget_money: float | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 1729,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare policies on exactly bound match/cutoff keys.

    Duplicate exports are collapsed by ``(match_id, cutoff, strategy, seed)``
    when their rescored values agree.  Replicate seeds (the ``random`` policy
    in the current archive) are then averaged within each key.  All confidence
    intervals resample whole matches, so repeated snapshots/replicates do not
    masquerade as independent matches.
    """
    if not isinstance(baseline_strategy, str) or not baseline_strategy.strip():
        raise ValueError("baseline_strategy must be nonempty")
    if bootstrap_samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap parameters")
    episode_rows = tuple(episodes)
    budget_values: set[float] = set()
    for index, row in enumerate(episode_rows):
        if not isinstance(row, Mapping):
            continue
        raw_budget = row.get("budget_money", 0.0)
        try:
            budget_values.add(_finite(raw_budget, f"episodes[{index}].budget_money"))
        except ValueError:
            # The normal row validation below emits the path-specific issue.
            pass
    if budget_money is not None:
        budget_money = _finite(budget_money, "budget_money")
    elif len(budget_values) > 1:
        raise ValueError("episode archive contains multiple budget_money values; evaluate one budget slice at a time")
    selected_budget = next(iter(budget_values), 0.0) if budget_money is None else budget_money
    issues: list[MetricIssue] = []
    snapshot_index = _snapshot_index(snapshots, issues)
    test_fold_by_match = _test_fold_map(tuple(folds), issues)
    raw_count = 0
    excluded = Counter()
    duplicate_count = 0
    accepted: dict[tuple[tuple[str, datetime], str], list[dict[str, Any]]] = defaultdict(list)
    seen_replicates: dict[tuple[tuple[str, datetime], str, Any], tuple[Any, ...]] = {}
    source_fold_values: Counter[int] = Counter()

    for index, row in enumerate(episode_rows):
        raw_count += 1
        path = f"episodes[{index}]"
        if not isinstance(row, Mapping):
            _issue(issues, "ROW_NOT_MAPPING", "error", None, path, "episode must be a mapping")
            continue
        try:
            row_budget = _finite(row.get("budget_money", 0.0), f"{path}.budget_money")
        except ValueError as exc:
            _issue(issues, "INVALID_BUDGET", "error", str(row.get("match_id")) if row.get("match_id") else None, path, str(exc))
            continue
        if not math.isclose(row_budget, selected_budget, abs_tol=1e-12, rel_tol=0.0):
            excluded["other_budget_slice"] += 1
            continue
        match_id = str(row.get("match_id")) if row.get("match_id") is not None else None
        try:
            key = _key(match_id, row.get("prediction_cutoff"))
        except (TypeError, ValueError) as exc:
            _issue(issues, "INVALID_EPISODE_KEY", "error", match_id, path, str(exc))
            continue
        snapshot = snapshot_index.get(key)
        if snapshot is None:
            excluded["unbound_snapshot"] += 1
            _issue(issues, "EPISODE_SNAPSHOT_UNBOUND", "warning", key[0], path,
                   "episode has no exact audited match_id/prediction_cutoff snapshot")
            continue
        if row.get("kickoff") is not None:
            try:
                if _utc(row["kickoff"], f"{path}.kickoff") != snapshot["kickoff"]:
                    _issue(issues, "EPISODE_KICKOFF_MISMATCH", "error", key[0], path,
                           "episode kickoff differs from audited snapshot")
                    continue
            except ValueError as exc:
                _issue(issues, "INVALID_EPISODE_KICKOFF", "error", key[0], path, str(exc))
                continue
        fold_index = test_fold_by_match.get(key[0])
        if test_only and fold_index is None:
            excluded["not_in_test_fold"] += 1
            continue
        strategy = row.get("strategy")
        if not isinstance(strategy, str) or not strategy.strip():
            _issue(issues, "MISSING_STRATEGY", "error", key[0], path, "strategy must be nonempty")
            continue
        try:
            probabilities = _probabilities(row, path)
            brier, log_loss = _losses(probabilities, snapshot["outcome"])
            cost_money, requests, cost_source = _cost_receipt(row, path)
            declared_brier = row.get("brier_score")
            declared_log = row.get("log_loss")
            if declared_brier is not None and abs(_finite(declared_brier, f"{path}.brier_score") - brier) > METRIC_TOLERANCE:
                _issue(issues, "EXPORTED_BRIER_MISMATCH", "error", key[0], path, "exported brier differs from rescored label")
            if declared_log is not None and abs(_finite(declared_log, f"{path}.log_loss") - log_loss) > METRIC_TOLERANCE:
                _issue(issues, "EXPORTED_LOG_LOSS_MISMATCH", "error", key[0], path, "exported log loss differs from rescored label")
        except (TypeError, ValueError) as exc:
            _issue(issues, "INVALID_EPISODE_METRIC", "error", key[0], path, str(exc))
            continue
        seed = row.get("seed")
        fingerprint = (brier, log_loss, cost_money, requests, probabilities)
        replicate_key = (key, strategy, seed)
        previous = seen_replicates.get(replicate_key)
        if previous is not None:
            if previous != fingerprint:
                _issue(issues, "DUPLICATE_REPLICATE_CONFLICT", "error", key[0], path,
                       "same strategy/seed is exported with different metrics")
            else:
                duplicate_count += 1
            continue
        seen_replicates[replicate_key] = fingerprint
        accepted[(key, strategy)].append({
            "brier_score": brier, "log_loss": log_loss, "cost_money": cost_money,
            "requests": requests, "seed": seed, "cost_source": cost_source,
            "fold": fold_index,
        })
        if row.get("fold") is not None:
            try:
                source_fold_values[int(row["fold"])] += 1
            except (TypeError, ValueError):
                _issue(issues, "INVALID_SOURCE_FOLD", "warning", key[0], path, "episode fold is not an integer")

    # Average replicate seeds within a bound key.  This creates one paired
    # observation per strategy/match/cutoff, while preserving replicate counts.
    by_strategy_key: dict[str, dict[tuple[str, datetime], dict[str, float]]] = defaultdict(dict)
    replicate_counts: Counter[str] = Counter()
    cost_sources: Counter[str] = Counter()
    for (key, strategy), values in accepted.items():
        replicate_counts[strategy] += len(values)
        for value in values:
            cost_sources[value["cost_source"]] += 1
        by_strategy_key[strategy][key] = {
            metric: math.fsum(float(value[metric]) for value in values) / len(values)
            for metric in ("brier_score", "log_loss", "cost_money", "requests")
        }

    strategies = sorted(by_strategy_key)
    baseline_keys = set(by_strategy_key.get(baseline_strategy, {}))
    paired_keys_by_strategy: dict[str, set[tuple[str, datetime]]] = {
        strategy: set(keys) & baseline_keys for strategy, keys in by_strategy_key.items()
    }
    if baseline_strategy not in by_strategy_key:
        _issue(issues, "BASELINE_MISSING", "error", None, "episodes", f"baseline {baseline_strategy!r} has no accepted rows")

    aggregate: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for strategy in strategies:
        keys = paired_keys_by_strategy[strategy]
        rows_by_match: dict[str, dict[str, float]] = {}
        for key in sorted(keys):
            value = by_strategy_key[strategy][key]
            rows_by_match.setdefault(key[0], defaultdict(float))
            # A match may have multiple cutoffs; average them before CI.
            target = rows_by_match[key[0]]
            for metric in ("brier_score", "log_loss", "cost_money", "requests"):
                target[metric] = target.get(metric, 0.0) + value[metric]
            target["_n"] = target.get("_n", 0.0) + 1.0
        for target in rows_by_match.values():
            n = target.pop("_n")
            for metric in ("brier_score", "log_loss", "cost_money", "requests"):
                target[metric] /= n
        summary = _metric_summary(
            rows_by_match, samples=bootstrap_samples, seed=bootstrap_seed + len(aggregate) * 10000,
            confidence=confidence,
        )
        summary.update({
            "strategy": strategy,
            "bound_prediction_keys": len(keys),
            "replicate_rows": replicate_counts[strategy],
            "paired_with_baseline": strategy != baseline_strategy,
        })
        aggregate[strategy] = summary
        if strategy == baseline_strategy:
            continue
        delta_by_match: dict[str, dict[str, float]] = {}
        for key in sorted(keys):
            if key not in by_strategy_key[baseline_strategy]:
                continue
            left, right = by_strategy_key[strategy][key], by_strategy_key[baseline_strategy][key]
            target = delta_by_match.setdefault(key[0], defaultdict(float))
            for metric in ("brier_score", "log_loss", "cost_money", "requests"):
                target[metric] = target.get(metric, 0.0) + left[metric] - right[metric]
            target["_n"] = target.get("_n", 0.0) + 1.0
        for target in delta_by_match.values():
            n = target.pop("_n")
            for metric in ("brier_score", "log_loss", "cost_money", "requests"):
                target[metric] /= n
        delta_summary = _metric_summary(
            delta_by_match, samples=bootstrap_samples, seed=bootstrap_seed + 500000 + len(paired) * 10000,
            confidence=confidence,
        )
        delta_summary.update({
            "strategy": strategy, "baseline_strategy": baseline_strategy,
            "difference_definition": "strategy minus baseline; negative loss/cost is favorable",
            "paired_prediction_keys": len(keys), "paired_matches": len(delta_by_match),
        })
        paired[strategy] = delta_summary

    points = {
        strategy: {
            "mean_cost_money": aggregate[strategy].get("mean_cost_money"),
            "mean_brier_score": aggregate[strategy].get("mean_brier_score"),
        }
        for strategy in strategies
    }
    frontier = []
    for strategy, point in points.items():
        dominated = False
        for other, candidate in points.items():
            if other == strategy or candidate["mean_cost_money"] is None or candidate["mean_brier_score"] is None:
                continue
            if point["mean_cost_money"] is None or point["mean_brier_score"] is None:
                continue
            if (candidate["mean_cost_money"] <= point["mean_cost_money"]
                    and candidate["mean_brier_score"] <= point["mean_brier_score"]
                    and (candidate["mean_cost_money"] < point["mean_cost_money"]
                         or candidate["mean_brier_score"] < point["mean_brier_score"])):
                dominated = True
                break
        if not dominated:
            frontier.append(strategy)

    issue_counts = Counter(issue.code for issue in issues)
    severity = Counter(issue.severity for issue in issues)
    fold_summary: dict[str, Any] = {}

    def _match_averaged_rows(strategy: str, keys: Iterable[tuple[str, datetime]]) -> dict[str, dict[str, float]]:
        by_match: dict[str, dict[str, float]] = {}
        for key in keys:
            value = by_strategy_key[strategy].get(key)
            if value is None:
                continue
            target = by_match.setdefault(key[0], defaultdict(float))
            target["_n"] = target.get("_n", 0.0) + 1.0
            for metric in ("brier_score", "log_loss", "cost_money", "requests"):
                target[metric] = target.get(metric, 0.0) + value[metric]
        for target in by_match.values():
            n = target.pop("_n")
            for metric in ("brier_score", "log_loss", "cost_money", "requests"):
                target[metric] /= n
        return by_match

    for fold_index in sorted(set(test_fold_by_match.values())):
        fold_matches = {mid for mid, value in test_fold_by_match.items() if value == fold_index}
        fold_keys = {key for strategy in strategies for key in by_strategy_key[strategy] if key[0] in fold_matches}
        fold_summary[str(fold_index)] = {
            "test_matches": len(fold_matches),
            "bound_keys": len(fold_keys),
            "strategies": {
                strategy: _metric_summary(
                    _match_averaged_rows(
                        strategy,
                        (key for key in paired_keys_by_strategy[strategy] if key[0] in fold_matches),
                    ),
                    samples=0, seed=0, confidence=confidence,
                )
                for strategy in strategies
            },
        }

    comparison_eligible = (
        severity["error"] == 0
        and severity["warning"] == 0
        and not excluded
    )

    return {
        "schema_version": VERSION,
        "status": "PASS" if severity["error"] == 0 else "FAIL",
        "evaluator_only": True,
        "runtime_safe": False,
        "descriptive_only": True,
        "comparison_eligible": comparison_eligible,
        "eligibility_reason": (
            "exact_bound_complete_no_exclusions" if comparison_eligible
            else "evaluator_slice_has_binding_or_partition_exclusions"
        ),
        "independent_confirmation": False,
        "scope": "historical replay of previously explored residual-EVSI episodes",
        "baseline_strategy": baseline_strategy,
        "test_only": test_only,
        "budget_money": selected_budget,
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "coverage": {
            "raw_episode_rows": raw_count,
            "accepted_replicate_rows": sum(len(values) for values in accepted.values()),
            "duplicate_rows_collapsed": duplicate_count,
            "bound_prediction_keys": len({key for strategy in by_strategy_key for key in by_strategy_key[strategy]}),
            "bound_matches": len({key[0] for strategy in by_strategy_key for key in by_strategy_key[strategy]}),
            "test_fold_matches": len(test_fold_by_match),
            "excluded_rows": dict(sorted(excluded.items())),
            "cost_source_counts": dict(sorted(cost_sources.items())),
            "source_fold_counts": dict(sorted(source_fold_values.items())),
        },
        "issues": [issue.as_dict() for issue in issues],
        "issue_counts": dict(sorted(issue_counts.items())),
        "counts": {"errors": severity["error"], "warnings": severity["warning"]},
        "aggregate": aggregate,
        "paired_vs_baseline": paired,
        "cost_risk_frontier": {
            "objective": "minimize mean realized money cost and mean Brier score",
            "frontier": sorted(frontier),
            "points": points,
        },
        "folds": fold_summary,
        "protocol_notes": {
            "labels": "read only from audited evaluator sidecars; never included in runtime state",
            "pairing": "exact match_id + prediction_cutoff; replicate seeds averaged within a pair",
            "interval": "deterministic percentile bootstrap resampling whole matches",
            "cost": "nested cost.money and cost.requests preferred; budget_money excluded from realized cost",
            "oracle": "no hindsight oracle is included in this historical runtime comparison",
        },
    }


__all__ = ["VERSION", "MetricIssue", "compare_historical_policies"]
