"""Paired offline comparison for autonomous information acquisition.

Round139 turns the existing evaluator-side policy benchmark into a historical
episode scorecard.  The actor sees only the public, as-of snapshot; the
evaluator sidecar is consumed only after selection.  The three named arms are:

``autonomous_voi``
    The current state-conditioned ``current_voi_gate`` policy.
``fixed_all``
    The public-menu budget baseline.
``fixed_order``
    A predeclared menu-order policy which attempts every listed candidate while
    the public budget permits it.

This is a mechanism benchmark, not a match-prediction claim.  It reports the
paired resource/loss points and their Pareto frontier while preserving unknown
metrics as unknown.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import datetime
import json
import math
from typing import Any

from .policy_benchmark import VERSION as POLICY_BENCHMARK_VERSION
from .policy_benchmark import compare_policies

VERSION = "historical-policy-comparison-v1"
POLICIES = ("autonomous_voi", "fixed_all", "fixed_order")
_SOURCE_POLICIES = {
    "autonomous_voi": "current_voi_gate",
    "fixed_all": "fixed_all",
    "fixed_order": "fixed_subset",
}
_EPS = 1e-12


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed


def _public_leaks(value: Any, path: str = "public") -> list[str]:
    forbidden = {"result", "outcome", "score", "home_score", "away_score",
                 "final_score", "decision_loss_after", "realized_loss"}
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in forbidden:
                found.append(f"{path}.{key_text}")
            found.extend(_public_leaks(child, f"{path}.{key_text}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_public_leaks(child, f"{path}[{index}]"))
    return found


def _historical_guard(episode: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Validate the historical/as-of boundary without reading evaluator data."""
    public = _mapping(episode.get("public"), f"episodes[{index}].public")
    cutoff = public.get("prediction_cutoff", episode.get("prediction_cutoff"))
    kickoff = public.get("kickoff", episode.get("kickoff"))
    if cutoff is None or kickoff is None:
        raise ValueError("historical episode requires prediction_cutoff and kickoff")
    cutoff_time = _time(cutoff, f"episodes[{index}].prediction_cutoff")
    kickoff_time = _time(kickoff, f"episodes[{index}].kickoff")
    if cutoff_time > kickoff_time:
        raise ValueError("prediction_cutoff must not follow kickoff")
    # These fields are labels or post-match payloads and must stay sidecar-only.
    leaked = _public_leaks(public, f"episodes[{index}].public")
    if leaked:
        raise ValueError("public historical snapshot contains evaluator fields: " + ",".join(leaked))
    return {"prediction_cutoff": cutoff_time.isoformat(), "kickoff": kickoff_time.isoformat()}


def _fixed_order_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Pin fixed_order to all public candidates, retaining menu order."""
    cloned = deepcopy(dict(episode))
    public = dict(_mapping(cloned.get("public"), "episode.public"))
    menu = public.get("candidate_menu", public.get("candidates"))
    if isinstance(menu, Mapping):
        candidate_ids = list(menu)
    elif isinstance(menu, list):
        candidate_ids = [item if isinstance(item, str) else item.get("candidate_id") for item in menu]
    else:
        raise ValueError("public candidate menu is required")
    if not candidate_ids or any(not isinstance(item, str) or not item.strip() for item in candidate_ids):
        raise ValueError("public candidate menu contains an invalid candidate")
    public["fixed_subset"] = list(candidate_ids)
    cloned["public"] = public
    return cloned


def _remap(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = {policy: [] for policy in POLICIES}
    aggregate = {policy: {} for policy in POLICIES}
    for public_policy, source_policy in _SOURCE_POLICIES.items():
        for row in report["rows"][source_policy]:
            projected = dict(row)
            projected["policy_id"] = public_policy
            projected["source_policy_id"] = source_policy
            rows[public_policy].append(projected)
        aggregate[public_policy] = dict(report["aggregate"][source_policy])
        aggregate[public_policy]["policy_id"] = public_policy
        aggregate[public_policy]["source_policy_id"] = source_policy
    return {"rows": rows, "aggregate": aggregate}


def _pareto(aggregate: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    points = {
        policy: {
            "mean_resource_cost": value.get("mean_spent_cost"),
            "mean_final_loss": value.get("mean_final_loss"),
            "mean_decision_loss_reduction": value.get("mean_decision_loss_reduction"),
            "episode_count": value.get("episode_count"),
        }
        for policy, value in aggregate.items()
    }
    frontier: list[str] = []
    dominated_by: dict[str, list[str]] = {}
    for policy, point in points.items():
        dominators: list[str] = []
        cost, loss = point["mean_resource_cost"], point["mean_final_loss"]
        if cost is None or loss is None:
            dominated_by[policy] = []
            continue
        for other, candidate in points.items():
            if other == policy or candidate["mean_resource_cost"] is None or candidate["mean_final_loss"] is None:
                continue
            no_worse = (candidate["mean_resource_cost"] <= cost + _EPS and
                        candidate["mean_final_loss"] <= loss + _EPS)
            strict = (candidate["mean_resource_cost"] < cost - _EPS or
                      candidate["mean_final_loss"] < loss - _EPS)
            if no_worse and strict:
                dominators.append(other)
        dominated_by[policy] = sorted(dominators)
        if not dominators:
            frontier.append(policy)
    return {"objective": "minimize_mean_resource_cost_and_mean_final_loss",
            "frontier": frontier, "dominated_by": dominated_by, "points": points}


def compare_historical_episodes(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    raw = list(episodes)
    if not raw:
        raise ValueError("at least one historical episode is required")
    guards = []
    ids = []
    for index, episode in enumerate(raw):
        _mapping(episode, f"episodes[{index}]")
        ids.append(_text(episode.get("episode_id"), f"episodes[{index}].episode_id"))
        guards.append(_historical_guard(episode, index))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate historical episode_id")
    fixed_order = [_fixed_order_episode(episode) for episode in raw]
    # All three arms replay the same paired episodes.  The fixed-order arm is
    # represented by a copy whose public fixed subset contains every candidate;
    # the benchmark's stable menu traversal then preserves the declared order.
    source_report = compare_policies(
        fixed_order,
        policies=("current_voi_gate", "fixed_all", "fixed_subset"),
    )
    remapped = _remap(source_report)
    paired: dict[str, dict[str, Any]] = {}
    for policy in ("autonomous_voi", "fixed_order"):
        target_rows = {row["episode_id"]: row for row in remapped["rows"][policy]}
        baseline_rows = {row["episode_id"]: row for row in remapped["rows"]["fixed_all"]}
        deltas = []
        for episode_id in ids:
            row, base = target_rows[episode_id], baseline_rows[episode_id]
            deltas.append({
                "episode_id": episode_id,
                "resource_cost_delta_vs_fixed_all": None if row.get("spent_cost") is None or base.get("spent_cost") is None else row["spent_cost"] - base["spent_cost"],
                "final_loss_delta_vs_fixed_all": None if row.get("final_loss") is None or base.get("final_loss") is None else row["final_loss"] - base["final_loss"],
                "net_utility_delta_vs_fixed_all": None if row.get("net_utility") is None or base.get("net_utility") is None else row["net_utility"] - base["net_utility"],
            })
        paired[policy] = {"baseline": "fixed_all", "episodes": deltas}
    aggregate = remapped["aggregate"]
    # The source benchmark returns mean_spent_cost under a stable name only in
    # its aggregate.  Keep an explicit alias used by the resource-risk chart.
    for value in aggregate.values():
        value["mean_resource_cost"] = value.get("mean_spent_cost")
    return {
        "schema_version": VERSION,
        "policy_benchmark_schema": POLICY_BENCHMARK_VERSION,
        "policy_order": list(POLICIES),
        "episode_ids": ids,
        "as_of_guards": guards,
        "rows": remapped["rows"],
        "aggregate": aggregate,
        "paired_deltas": paired,
        "resource_risk_pareto": _pareto(aggregate),
        "contract": {
            "matched_historical_episode_set": True,
            "actor_uses_public_snapshot_only": True,
            "evaluator_sidecar_required": True,
            "runtime_policy_reads_evaluator_labels": False,
            "unknown_metrics_are_not_zero": True,
            "fixed_order_is_predeclared_public_menu_order": True,
            "numeric_voi_training_eligible": False,
            "real_source_requests": 0,
            "real_model_calls": 0,
        },
    }


def runtime_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Project only the actor-visible policy contract for later UI wiring."""
    if report.get("schema_version") != VERSION:
        raise ValueError("unexpected historical comparison schema")
    projection = {
        "schema_version": VERSION,
        "projection_scope": "runtime_public_context",
        "policy_order": list(POLICIES),
        "episode_ids": list(report.get("episode_ids", [])),
        "as_of_guards": list(report.get("as_of_guards", [])),
        "truth_in_policy_view": False,
        "hidden_result_included": False,
    }
    encoded = json.dumps(projection, ensure_ascii=False).lower()
    for forbidden in ('"result"', '"outcome"', '"final_loss"', '"decision_loss"', '"realized_loss"'):
        if forbidden in encoded:
            raise ValueError("runtime projection contains evaluator-only fields")
    return projection


__all__ = ["VERSION", "POLICIES", "compare_historical_episodes", "runtime_projection"]
