"""Round162 adapter for paired counterfactual policy replay.

The adapter keeps one public as-of episode fixed while replaying three query
policies: a declared menu order, a canonical full-query control, and an
interval-based VoI policy.  It is evaluator-only.  The actor trace is safe to
show to a runtime or reviewer; sequence losses and paired loss deltas remain
in the evaluator projection.

Independent-gold episodes can support performance reporting.  Synthetic
offline episodes are accepted only when the caller opts into the adapter's
offline mode and are reported as structural mechanism evidence.  They never
become numeric VoI training data or a real-world performance claim.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from .counterfactual_policy_replay import (
    INDEPENDENT_GOLD,
    INDEPENDENT_GOLD_RECONSTRUCTED,
    POLICIES as REPLAY_POLICIES,
    compare_counterfactual_policies,
)


VERSION = "round162-counterfactual-policy-adapter-v1"
POLICIES = ("fixed_order", "full_query", "interval_voi")
SYNTHETIC_OFFLINE = "SYNTHETIC_OFFLINE"
_EPS = 1e-12
_FORBIDDEN_PUBLIC_KEYS = frozenset({
    "result", "outcome", "score", "scores", "goals", "home_score", "away_score",
    "final_score", "label", "loss", "decision_loss", "realized_gain", "realized_loss",
    "evaluator", "sidecar", "sequence_losses", "action_returns", "post_state", "after_state",
})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _public_key_scan(value: Any, path: str = "public") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in _FORBIDDEN_PUBLIC_KEYS:
                found.append(f"{path}.{key_text}")
            found.extend(_public_key_scan(child, f"{path}.{key_text}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_public_key_scan(child, f"{path}[{index}]"))
    return found


def _sidecar(episode: Mapping[str, Any]) -> Mapping[str, Any]:
    sidecar = episode.get("evaluation_sidecar", episode.get("evaluator_sidecar"))
    if not isinstance(sidecar, Mapping):
        raise ValueError("evaluation_sidecar is required")
    return sidecar


def _label_status(episode: Mapping[str, Any]) -> str:
    return _text(_sidecar(episode).get("label_status"), "evaluation_sidecar.label_status")


def _validate_batch(episodes: Sequence[Mapping[str, Any]], *, allow_offline: bool) -> tuple[str, list[dict[str, Any]]]:
    if not episodes:
        raise ValueError("at least one episode is required")
    statuses: list[str] = []
    manifests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, episode in enumerate(episodes):
        _mapping(episode, f"episodes[{index}]")
        episode_id = _text(episode.get("episode_id"), f"episodes[{index}].episode_id")
        if episode_id in seen:
            raise ValueError("duplicate episode_id")
        seen.add(episode_id)
        public = _mapping(episode.get("public"), f"episodes[{index}].public")
        leaked = _public_key_scan(public, f"episodes[{index}].public")
        if leaked:
            raise ValueError("public episode contains evaluator fields: " + ",".join(leaked))
        status = _label_status(episode)
        if status not in {INDEPENDENT_GOLD, INDEPENDENT_GOLD_RECONSTRUCTED, SYNTHETIC_OFFLINE}:
            raise ValueError("unsupported evaluation label status")
        statuses.append(status)
        manifests.append({
            "episode_id": episode_id,
            "public_snapshot_digest": _digest(public),
            "target_key": public.get("target_key"),
            "prediction_cutoff": public.get("prediction_cutoff"),
            "kickoff": public.get("kickoff"),
            "budget": public.get("budget"),
            "candidate_ids": [
                item if isinstance(item, str) else item.get("candidate_id")
                for item in public.get("candidate_menu", public.get("candidates", []))
            ],
        })
    if len(set(statuses)) != 1:
        raise ValueError("a paired batch must use one evaluation label status")
    status = statuses[0]
    if status == SYNTHETIC_OFFLINE and not allow_offline:
        raise ValueError("synthetic offline replay requires allow_offline=True")
    return status, manifests


def _mean(values: Sequence[float | None]) -> float | None:
    known = [float(value) for value in values if value is not None]
    return None if not known else math.fsum(known) / len(known)


def _paired_summary(comparison: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    paired = comparison.get("paired_deltas", {})
    summary: dict[str, Any] = {}
    for policy, rows in paired.items():
        eligible = [row for row in rows if row.get("counterfactual_eligible") is True]
        summary[policy] = {
            "paired_episode_count": len(rows),
            "paired_eligible_count": len(eligible),
            "paired_ineligible_count": len(rows) - len(eligible),
            "mean_final_loss_delta": _mean([row.get("final_loss_delta") for row in eligible]),
            "mean_net_utility_delta": _mean([row.get("net_utility_delta") for row in eligible]),
            "unknown_deltas_are_excluded": True,
        }
    # Synthetic losses are useful to check replay mechanics, but keeping their
    # numeric means in the report could be mistaken for measured performance.
    if status == SYNTHETIC_OFFLINE:
        for row in summary.values():
            row["numeric_performance_reportable"] = False
            row["mean_final_loss_delta"] = None
            row["mean_net_utility_delta"] = None
    else:
        for row in summary.values():
            row["numeric_performance_reportable"] = True
    return summary


def compare_policies(
    episodes: Iterable[Mapping[str, Any]], *,
    interval_provider: Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]] | None = None,
    allow_offline: bool = False,
    allow_reconstructed: bool = False,
) -> dict[str, Any]:
    """Replay a matched episode set under the Round162 policy contract."""
    raw = list(episodes)
    status, manifests = _validate_batch(raw, allow_offline=allow_offline)
    if status == INDEPENDENT_GOLD_RECONSTRUCTED and not allow_reconstructed:
        raise ValueError("reconstructed result replay requires allow_reconstructed=True")
    accepted = (status,)
    comparison = compare_counterfactual_policies(
        raw,
        interval_provider=interval_provider,
        policies=POLICIES,
        accepted_label_statuses=accepted,
    )
    results = comparison["results"]
    eligible_by_policy = {
        policy: sum(bool(row["evaluator"]["counterfactual_eligible"]) for row in rows)
        for policy, rows in results.items()
    }
    actor_contracts = []
    for policy in POLICIES:
        for row in results[policy]:
            trace = row["actor_trace"]
            if trace.get("hidden_result_included") is not False:
                raise ValueError("actor trace must explicitly exclude hidden result")
            actor_contracts.append({
                "episode_id": row["episode_id"],
                "policy": policy,
                "selected_actions": list(trace.get("selected_actions", [])),
                "query_count": trace.get("query_count"),
                "spent_cost": trace.get("spent_cost"),
                "stop_reason": trace.get("stop_reason"),
                "hidden_result_included": False,
            })
    return {
        "schema_version": VERSION,
        "policy_order": list(POLICIES),
        "evaluation_label_status": status,
        "evaluation_scope": (
            "independent_gold_performance" if status == INDEPENDENT_GOLD
            else "retrospective_reconstructed_utility" if status == INDEPENDENT_GOLD_RECONSTRUCTED
            else "synthetic_structural_mechanism"
        ),
        "synthetic_or_offline_only": status == SYNTHETIC_OFFLINE,
        "operational_availability_claim": status == INDEPENDENT_GOLD,
        "numeric_voi_training_eligible": False,
        "real_source_requests": 0 if status == SYNTHETIC_OFFLINE else None,
        "real_model_calls": 0 if status == SYNTHETIC_OFFLINE else None,
        "independent_gold_labels": len(raw) if status == INDEPENDENT_GOLD else 0,
        "reconstructed_result_labels": len(raw) if status == INDEPENDENT_GOLD_RECONSTRUCTED else 0,
        "episode_count": len(raw),
        "manifests": manifests,
        "actor_contracts": actor_contracts,
        "results": results,
        "paired_deltas": comparison["paired_deltas"],
        "paired_summary": _paired_summary(comparison, status=status),
        "eligible_episode_count_by_policy": eligible_by_policy,
        "contract": {
            "same_public_episode_menu_cutoff_budget": True,
            "actor_reads_public_snapshot_only": True,
            "sidecar_is_evaluator_only": True,
            "selected_transition_only": True,
            "complete_sequence_required": True,
            "unknown_metrics_are_not_zero": True,
            "paired_by_episode": True,
            "runtime_changes": False,
        },
        "limitations": [
            "The interval policy receives declared state-bound intervals; this round does not calibrate those intervals.",
            "Synthetic offline episodes validate policy-control and leakage contracts, not real match performance.",
            "Independent-gold performance requires complete, cutoff-safe, source-audited sequence losses.",
            "Reconstructed result labels support retrospective utility only; they do not prove historical online source availability.",
        ],
    }


def runtime_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return actor-visible replay evidence without evaluator annotations."""
    if report.get("schema_version") != VERSION:
        raise ValueError("unexpected Round162 report schema")
    actor_contracts = []
    for row in report.get("actor_contracts", []):
        if not isinstance(row, Mapping):
            raise ValueError("actor contract row is malformed")
        actor_contracts.append({
            "episode_id": row.get("episode_id"),
            "policy": row.get("policy"),
            "selected_actions": list(row.get("selected_actions", [])),
            "query_count": row.get("query_count"),
            "spent_cost": row.get("spent_cost"),
            "stop_reason": row.get("stop_reason"),
            "hidden_result_included": False,
        })
    projection = {
        "schema_version": VERSION,
        "projection_scope": "runtime_actor_trace",
        "policy_order": list(report.get("policy_order", [])),
        "episode_count": report.get("episode_count", 0),
        "actor_contracts": actor_contracts,
        "truth_in_policy_view": False,
        "hidden_result_included": False,
    }
    encoded = _canonical(projection).casefold()
    for forbidden in ('"result"', '"outcome"', '"score"', '"loss"', '"sidecar"', '"sequence_losses"', '"evaluator"'):
        if forbidden in encoded:
            raise ValueError("runtime projection contains evaluator-only fields")
    return projection


__all__ = ["VERSION", "POLICIES", "SYNTHETIC_OFFLINE", "compare_policies", "runtime_projection"]
