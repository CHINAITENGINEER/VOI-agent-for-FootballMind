"""Episode-level paired bootstrap summaries for adaptive-policy replay.

The unit of resampling is an episode block, never an individual query.  The
input is the evaluator-side ``paired_deltas`` emitted by
``counterfactual_policy_replay.compare_counterfactual_policies``.  Unknown
losses remain missing throughout the calculation and are reported with their
own denominator.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import random
from typing import Any


VERSION = "paired-bootstrap-v1"
_METRICS = ("final_loss_delta", "net_utility_delta")


def _finite(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _rows(report: Mapping[str, Any], policy: str, baseline: str | None, metric: str) -> list[dict[str, Any]]:
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")
    paired = report.get("paired_deltas")
    if not isinstance(paired, Mapping) or policy not in paired:
        raise ValueError(f"paired deltas for policy {policy!r} are required")
    raw = paired[policy]
    if not isinstance(raw, list):
        raise ValueError("paired delta rows must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ValueError(f"paired_deltas[{policy}][{index}] must be a mapping")
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("paired delta episode_id must be nonempty")
        episode_id = episode_id.strip()
        if episode_id in seen:
            raise ValueError("paired delta episode IDs must be unique")
        seen.add(episode_id)
        value = row.get(metric)
        if value is not None:
            value = _finite(value, f"{episode_id}.{metric}")
        eligible = row.get("counterfactual_eligible")
        if not isinstance(eligible, bool):
            raise ValueError(f"{episode_id}.counterfactual_eligible must be boolean")
        result.append({"episode_id": episode_id, "value": value, "eligible": eligible})
    if baseline is not None and baseline == policy:
        raise ValueError("baseline and policy must differ")
    return result


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires observations")
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return math.fsum(values) / len(values)


def paired_bootstrap(
    report: Mapping[str, Any], *, policy: str, baseline: str | None = None,
    metric: str = "final_loss_delta", block_by: Mapping[str, str] | None = None,
    resamples: int = 2000, seed: int = 0, confidence: float = .95,
) -> dict[str, Any]:
    """Return a reproducible percentile CI over paired episode differences.

    ``block_by`` maps episode IDs to a public grouping label (for example a
    league or context stratum).  Bootstrap sampling draws whole blocks with
    replacement, preserving all observed pairs inside each sampled block.  A
    missing delta never becomes zero and is excluded only from the observed
    metric denominator.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("resamples must be an integer >= 100")
    confidence = _finite(confidence, "confidence", minimum=.5, maximum=1.0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    rows = _rows(report, policy, baseline, metric)
    if block_by is not None and not isinstance(block_by, Mapping):
        raise ValueError("block_by must be a mapping")
    for row in rows:
        episode_id = row["episode_id"]
        if block_by is not None and episode_id not in block_by:
            raise ValueError(f"block_by missing episode {episode_id}")
    observed = [row for row in rows if row["value"] is not None and row["eligible"]]
    unknown = len(rows) - len(observed)
    if not observed:
        return {
            "schema_version": VERSION, "policy": policy, "baseline": baseline,
            "metric": metric, "estimate": None, "confidence_interval": None,
            "observed_denominator": 0, "total_episode_count": len(rows),
            "unknown_denominator": unknown, "block_count": 0,
            "resamples": 0, "seed": seed, "unknown_metrics_are_not_zero": True,
        }
    blocks: dict[str, list[float]] = {}
    for row in observed:
        key = "episode:" + row["episode_id"] if block_by is None else str(block_by[row["episode_id"]])
        blocks.setdefault(key, []).append(row["value"])
    block_values = list(blocks.values())
    estimate = _mean(value for values in block_values for value in values)
    generator = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(resamples):
        sampled = [generator.choice(block_values) for _ in block_values]
        flattened = [value for values in sampled for value in values]
        bootstrap_means.append(_mean(flattened))
    alpha = (1.0 - confidence) / 2.0
    return {
        "schema_version": VERSION, "policy": policy, "baseline": baseline,
        "metric": metric, "estimate": estimate,
        "confidence_interval": [_percentile(bootstrap_means, alpha),
                                 _percentile(bootstrap_means, 1.0 - alpha)],
        "observed_denominator": len(observed), "total_episode_count": len(rows),
        "unknown_denominator": unknown, "block_count": len(block_values),
        "resamples": resamples, "seed": seed,
        "confidence": confidence, "unknown_metrics_are_not_zero": True,
        "unit": "episode_block_mean",
    }


__all__ = ["VERSION", "paired_bootstrap"]
