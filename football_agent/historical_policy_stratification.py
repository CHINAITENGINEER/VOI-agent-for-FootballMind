"""Context-stratified historical evaluation of autonomous information choice.

This layer keeps the Round139 policy comparator unchanged and aggregates its
paired traces by public match context.  Context labels are as-of metadata (for
example qualification state, tactical layout, or rotation risk); evaluator
losses and outcomes never participate in group assignment.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

from .historical_policy_comparison import (
    POLICIES,
    VERSION as COMPARISON_VERSION,
    compare_historical_episodes,
)

VERSION = "historical-policy-stratification-v1"
DEFAULT_CONTEXT_FIELDS = (
    "qualification_state",
    "tactical_layout",
    "rotation_risk",
    "dominant_uncertainty",
)
_EPS = 1e-12


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _context(episode: Mapping[str, Any], field: str, index: int) -> str:
    public = _mapping(episode.get("public"), f"episodes[{index}].public")
    block = public.get("context", episode.get("context", {}))
    if block is None:
        block = {}
    block = _mapping(block, f"episodes[{index}].context")
    value = block.get(field, "unknown")
    if value is None or (isinstance(value, str) and not value.strip()):
        return "unknown"
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"episodes[{index}].context.{field} must be scalar")
    return str(value)


def _mean(values: Sequence[float | None]) -> float | None:
    observed = [float(value) for value in values if value is not None]
    return math.fsum(observed) / len(observed) if observed else None


def _group_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    aggregate: dict[str, Any] = {}
    for policy in POLICIES:
        policy_rows = list(rows[policy])
        selected = [row.get("selected_candidates") for row in policy_rows]
        costs = [row.get("spent_cost") for row in policy_rows]
        losses = [row.get("final_loss") for row in policy_rows]
        gains = [row.get("decision_loss_reduction") for row in policy_rows]
        aggregate[policy] = {
            "policy_id": policy,
            "episode_count": len(policy_rows),
            "mean_resource_cost": _mean(costs),
            "mean_final_loss": _mean(losses),
            "mean_decision_loss_reduction": _mean(gains),
            "mean_query_count": _mean([None if value is None else float(len(value)) for value in selected]),
            "zero_query_rate": None if not selected else sum(value == [] for value in selected) / len(selected),
            "selection_counts": dict(sorted(Counter(
                candidate for value in selected if isinstance(value, list) for candidate in value
            ).items())),
            "unknown_final_loss_count": sum(value is None for value in losses),
            "unknown_resource_cost_count": sum(value is None for value in costs),
        }
    baseline = {row["episode_id"]: row for row in rows["fixed_all"]}
    paired: dict[str, Any] = {}
    for policy in ("autonomous_voi", "fixed_order"):
        deltas = []
        for row in rows[policy]:
            base = baseline[row["episode_id"]]
            deltas.append({
                "episode_id": row["episode_id"],
                "resource_cost_delta_vs_fixed_all": None if row.get("spent_cost") is None or base.get("spent_cost") is None else row["spent_cost"] - base["spent_cost"],
                "final_loss_delta_vs_fixed_all": None if row.get("final_loss") is None or base.get("final_loss") is None else row["final_loss"] - base["final_loss"],
            })
        paired[policy] = {"baseline": "fixed_all", "episodes": deltas,
                          "mean_resource_cost_delta": _mean([item["resource_cost_delta_vs_fixed_all"] for item in deltas]),
                          "mean_final_loss_delta": _mean([item["final_loss_delta_vs_fixed_all"] for item in deltas])}
    return {"aggregate": aggregate, "paired_deltas": paired}


def compare_by_context(
    episodes: Iterable[Mapping[str, Any]],
    *,
    context_fields: Sequence[str] = DEFAULT_CONTEXT_FIELDS,
) -> dict[str, Any]:
    raw = list(episodes)
    if not raw:
        raise ValueError("at least one historical episode is required")
    fields = tuple(_text(field, "context_fields[]") for field in context_fields)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("context_fields must be nonempty and unique")
    global_report = compare_historical_episodes(raw)
    groups: dict[str, dict[str, Any]] = {}
    assignments: list[dict[str, Any]] = []
    for field in fields:
        buckets: dict[str, list[Mapping[str, Any]]] = {}
        for index, episode in enumerate(raw):
            value = _context(episode, field, index)
            group_id = f"{field}={value}"
            buckets.setdefault(group_id, []).append(episode)
            assignments.append({"episode_id": episode["episode_id"], "field": field,
                                "value": value, "group_id": group_id})
        for group_id, members in buckets.items():
            comparison = compare_historical_episodes(members)
            summary = _group_summary(comparison)
            groups[group_id] = {
                "group_id": group_id,
                "field": field,
                "value": group_id.split("=", 1)[1],
                "episode_ids": list(comparison["episode_ids"]),
                "episode_count": len(members),
                "summary": summary,
                "resource_risk_pareto": comparison["resource_risk_pareto"],
            }
    return {
        "schema_version": VERSION,
        "comparison_schema": COMPARISON_VERSION,
        "policy_order": list(POLICIES),
        "episode_ids": list(global_report["episode_ids"]),
        "global": global_report,
        "groups": groups,
        "assignments": assignments,
        "context_fields": list(fields),
        "contract": {
            "group_assignment_uses_public_context_only": True,
            "matched_episode_set_within_group": True,
            "evaluator_sidecar_required": True,
            "runtime_policy_reads_evaluator_labels": False,
            "unknown_context_is_explicit_group": True,
            "unknown_metrics_are_not_zero": True,
            "real_source_requests": 0,
            "real_model_calls": 0,
            "numeric_voi_training_eligible": False,
        },
    }


__all__ = ["VERSION", "DEFAULT_CONTEXT_FIELDS", "compare_by_context"]
