"""Strict, reproducible scoring for regulation-time three-way forecasts.

Rows are mappings so the scorer can consume exported public contracts without
introducing another public prediction type. Order is home win, draw, away win.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable, Mapping

Row = Mapping[str, Any]
PROBABILITY_FIELDS = (
    "home_win_probability", "draw_probability", "away_win_probability"
)
CLASS_NAMES = ("home_win", "draw", "away_win")
LOG_LOSS_EPSILON = 1e-15


def _utc(value: Any, field: str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def outcome_index(row: Row) -> int:
    """Return 0/1/2 from final regulation-time goals; missing labels fail closed."""
    home, away = row.get("home_goals"), row.get("away_goals")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (home, away)):
        raise ValueError("home_goals and away_goals must be nonnegative integers")
    return 0 if home > away else 1 if home == away else 2


def probabilities(row: Row) -> tuple[float, float, float]:
    values = tuple(float(row[name]) for name in PROBABILITY_FIELDS)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-8, rel_tol=0):
        raise ValueError("probabilities must sum to one")
    return values  # type: ignore[return-value]


def per_prediction_losses(row: Row) -> dict[str, float]:
    target = outcome_index(row)
    values = probabilities(row)
    return {
        "log_loss": -math.log(max(values[target], LOG_LOSS_EPSILON)),
        "brier_score": sum((p - int(i == target)) ** 2 for i, p in enumerate(values)),
        "accuracy": float(max(range(3), key=values.__getitem__) == target),
    }


def score_predictions(rows: Iterable[Row]) -> dict[str, Any]:
    """Snapshot-weighted scores; multiclass Brier is sum of 3 squared errors.

    Ties for accuracy resolve home/draw/away in that fixed order. Zero target
    probabilities are clipped only inside the logarithm at 1e-15; probabilities
    are never silently normalized. Empty input has null scores, not zero losses.
    """
    rows = list(rows)
    losses = [per_prediction_losses(row) for row in rows]
    return {
        "n_predictions": len(rows),
        "n_matches": len({str(row["match_id"]) for row in rows}),
        "log_loss": mean(x["log_loss"] for x in losses) if losses else None,
        "brier_score": mean(x["brier_score"] for x in losses) if losses else None,
        "accuracy": mean(x["accuracy"] for x in losses) if losses else None,
        "brier_definition": "sum_over_three_classes; range [0,2]",
        "weighting": "equal prediction snapshots",
        "log_loss_epsilon": LOG_LOSS_EPSILON,
    }


def reliability_table(rows: Iterable[Row], n_bins: int = 10) -> list[dict[str, Any]]:
    """Return all class/bin cells, including empty cells with null averages."""
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    buckets: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        target = outcome_index(row)
        for class_index, probability in enumerate(probabilities(row)):
            bucket_index = min(int(probability * n_bins), n_bins - 1)
            buckets[class_index, bucket_index].append((probability, int(target == class_index)))
    result = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        for bucket_index in range(n_bins):
            cells = buckets[class_index, bucket_index]
            result.append({
                "class": class_name,
                "bin": bucket_index,
                "lower": bucket_index / n_bins,
                "upper": (bucket_index + 1) / n_bins,
                "count": len(cells),
                "mean_probability": mean(p for p, _ in cells) if cells else None,
                "observed_frequency": mean(y for _, y in cells) if cells else None,
            })
    return result


def save_calibration_plot(rows: Iterable[Row], path: str | Path, n_bins: int = 10) -> Path:
    """Write a noninteractive PNG with reliability curves and bin counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = reliability_table(rows, n_bins=n_bins)
    figure, axes = plt.subplots(2, 3, figsize=(12, 6), height_ratios=(3, 1))
    for index, name in enumerate(CLASS_NAMES):
        class_cells = [cell for cell in cells if cell["class"] == name]
        observed = [cell for cell in class_cells if cell["count"]]
        axes[0, index].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        axes[0, index].plot(
            [cell["mean_probability"] for cell in observed],
            [cell["observed_frequency"] for cell in observed], "o-",
        )
        axes[0, index].set(xlim=(0, 1), ylim=(0, 1), title=name.replace("_", " "),
                           xlabel="Mean predicted probability", ylabel="Observed frequency")
        axes[1, index].bar([(c["lower"] + c["upper"]) / 2 for c in class_cells],
                           [c["count"] for c in class_cells], width=0.9 / n_bins)
        axes[1, index].set(xlim=(0, 1), xlabel="Probability bin", ylabel="Count")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="png", dpi=150)
    plt.close(figure)
    return destination


def score_by_lead_time(
    rows: Iterable[Row], boundaries_hours: tuple[float, ...] = (0, 1, 6, 24, 72, math.inf)
) -> list[dict[str, Any]]:
    """Score half-open lead-hour buckets independently (including empty bins)."""
    if len(boundaries_hours) < 2 or boundaries_hours[0] < 0 or any(
        not a < b for a, b in zip(boundaries_hours, boundaries_hours[1:])
    ):
        raise ValueError("lead time boundaries must be strictly increasing and nonnegative")
    buckets: list[list[Row]] = [[] for _ in range(len(boundaries_hours) - 1)]
    for row in rows:
        hours = (_utc(row["kickoff"], "kickoff") - _utc(row["prediction_cutoff"], "prediction_cutoff")).total_seconds() / 3600
        if hours < 0:
            raise ValueError("prediction_cutoff must not follow kickoff")
        for index, (lower, upper) in enumerate(zip(boundaries_hours, boundaries_hours[1:])):
            if lower <= hours < upper:
                buckets[index].append(row)
                break
        else:
            raise ValueError("lead time is outside supplied boundaries")
    return [{
        "lead_hours_min": lower,
        "lead_hours_max_exclusive": upper if math.isfinite(upper) else None,
        **score_predictions(bucket),
    } for lower, upper, bucket in zip(boundaries_hours, boundaries_hours[1:], buckets)]


def _prediction_key(row: Row) -> tuple[str, datetime]:
    return str(row["match_id"]), _utc(row["prediction_cutoff"], "prediction_cutoff")


def _index_predictions(rows: Iterable[Row]) -> dict[tuple[str, datetime], Row]:
    result = {}
    for row in rows:
        key = _prediction_key(row)
        if key in result:
            raise ValueError(f"duplicate prediction for match_id/cutoff: {key}")
        result[key] = row
    return result


def _percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_model_comparison(
    left: Iterable[Row], right: Iterable[Row], *, bootstrap_samples: int = 2000,
    seed: int = 0, min_matches: int = 30, min_weeks: int = 8,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Pair identical match/cutoff keys; average snapshots per match before scoring.

    The estimand is the equal-match mean of (left loss - right loss). Negative
    means the left model is better. Percentile intervals resample whole UTC
    calendar weeks, retaining all matches and their paired snapshots in a week.
    This preserves both within-match and same-week dependence. The interval is
    descriptive and is suppressed below BOTH configured sample thresholds;
    it does not account for dependence extending beyond a week.
    """
    if bootstrap_samples < 1 or min_matches < 1 or min_weeks < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap parameters")
    left_index, right_index = _index_predictions(left), _index_predictions(right)
    keys = sorted(left_index.keys() & right_index.keys())
    match_values: dict[str, list[dict[str, float]]] = defaultdict(list)
    match_weeks: dict[str, datetime] = {}
    paired_rows = []
    for key in keys:
        lhs, rhs = left_index[key], right_index[key]
        if outcome_index(lhs) != outcome_index(rhs):
            raise ValueError(f"paired predictions disagree on outcome: {key}")
        kickoff = _utc(lhs["kickoff"], "kickoff")
        if kickoff != _utc(rhs["kickoff"], "kickoff"):
            raise ValueError(f"paired predictions disagree on kickoff: {key}")
        week = kickoff.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=kickoff.weekday())
        if key[0] in match_weeks and match_weeks[key[0]] != week:
            raise ValueError(f"match snapshots disagree on kickoff week: {key[0]}")
        match_weeks[key[0]] = week
        lhs_loss, rhs_loss = per_prediction_losses(lhs), per_prediction_losses(rhs)
        differences = {metric: lhs_loss[metric] - rhs_loss[metric] for metric in ("log_loss", "brier_score")}
        match_values[key[0]].append(differences)
        paired_rows.append({"match_id": key[0], "prediction_cutoff": key[1].isoformat(), **differences})
    per_match = {
        match_id: {metric: mean(value[metric] for value in values) for metric in ("log_loss", "brier_score")}
        for match_id, values in match_values.items()
    }
    week_matches: dict[datetime, list[str]] = defaultdict(list)
    for match_id, week in match_weeks.items():
        week_matches[week].append(match_id)
    eligible = len(per_match) >= min_matches and len(week_matches) >= min_weeks
    intervals = {"log_loss": None, "brier_score": None}
    if eligible:
        rng = random.Random(seed)
        weeks = sorted(week_matches)
        samples: dict[str, list[float]] = {"log_loss": [], "brier_score": []}
        for _ in range(bootstrap_samples):
            selected_matches = [match_id for week in rng.choices(weeks, k=len(weeks)) for match_id in week_matches[week]]
            for metric in samples:
                samples[metric].append(mean(per_match[match_id][metric] for match_id in selected_matches))
        tail = (1 - confidence) / 2
        intervals = {metric: [_percentile(values, tail), _percentile(values, 1 - tail)] for metric, values in samples.items()}
    return {
        "difference_definition": "left minus right; negative favors left",
        "weighting": "equal matches after averaging paired snapshots within each match",
        "paired_predictions": len(keys), "paired_matches": len(per_match), "utc_calendar_weeks": len(week_matches),
        "left_only_predictions": len(left_index.keys() - right_index.keys()),
        "right_only_predictions": len(right_index.keys() - left_index.keys()),
        "left_prediction_coverage": len(keys) / len(left_index) if left_index else None,
        "right_prediction_coverage": len(keys) / len(right_index) if right_index else None,
        "mean_log_loss_difference": mean(v["log_loss"] for v in per_match.values()) if per_match else None,
        "mean_brier_score_difference": mean(v["brier_score"] for v in per_match.values()) if per_match else None,
        "confidence_intervals": intervals,
        "interval_method": "percentile bootstrap of UTC calendar-week clusters; matches remain intact",
        "interval_status": "estimated" if eligible else "insufficient_matches_or_weeks",
        "interval_limitations": "Does not cover model-selection uncertainty or dependence extending beyond a week.",
        "confidence": confidence, "bootstrap_samples": bootstrap_samples if eligible else 0,
        "seed": seed, "minimum_matches": min_matches, "minimum_weeks": min_weeks,
        "per_prediction_differences": paired_rows,
    }
