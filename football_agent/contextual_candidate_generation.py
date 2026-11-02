"""Context-conditioned proposal generation for autonomous acquisition.

The module ranks registered actions using actor-visible, cutoff-safe context.
It never turns a match label, a declared expected value, or an evaluator
sidecar into a value estimate. Actual acquisition remains bound to the
state_digest in :mod:`autonomous_acquisition_step`.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any, Sequence

from .autonomous_acquisition_step import prepare_acquisition_state
from .historical_policy_comparison import compare_historical_episodes

VERSION = "contextual-candidate-generation-v2"
CONTEXT_FIELDS = frozenset({
    "qualification_state", "tactical_layout", "rotation_risk",
    "dominant_uncertainty", "weather_state", "lineup_status",
    "match_importance", "competition_stage", "motivation_state",
    "schedule_congestion",
})
_FORBIDDEN = frozenset({
    "result", "outcome", "score", "scores", "goals", "home_score",
    "away_score", "home_goals", "away_goals", "final_score", "label",
    "labels", "target", "decision_loss", "decision_loss_after", "realized_loss",
    "realized_gain", "baseline_loss", "evaluator", "sidecar",
    "future_result", "actual_result", "actual_outcome", "result_payload",
    "status", "label_status", "settlement_status", "loss", "final_loss",
    "hidden_return", "hidden_returns", "post_state", "after_state",
})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _private_paths(value: Any, path: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in _FORBIDDEN:
                found.append(f"{path}.{key_text}")
            found.extend(_private_paths(child, f"{path}.{key_text}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_private_paths(child, f"{path}[{index}]"))
    return found


def _scalar(value: Any, name: str) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "unknown"
    if type(value) in (int, float):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    elif not isinstance(value, (str, bool)):
        raise ValueError(f"{name} must be scalar")
    return str(value)


def _context(public: Mapping[str, Any]) -> dict[str, str]:
    raw = public.get("context", {})
    raw = {} if raw is None else _mapping(raw, "public.context")
    private = _private_paths(raw, "public.context")
    if private:
        raise ValueError("public context contains evaluator fields: " + ",".join(private))
    result: dict[str, str] = {}
    for key, value in raw.items():
        name = _text(str(key), "context field")
        if name not in CONTEXT_FIELDS:
            raise ValueError(f"unsupported public.context field: {name}")
        result[name] = _scalar(value, f"public.context.{name}")
    return result


def _affinity(item: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    raw = item.get("context_affinity", item.get("context_relevance", {}))
    raw = {} if raw is None else _mapping(raw, f"candidate_menu[{index}].context_affinity")
    private = _private_paths(raw, f"candidate_menu[{index}].context_affinity")
    if private:
        raise ValueError("candidate affinity contains evaluator fields: " + ",".join(private))
    for field, values in raw.items():
        field_name = _text(str(field), "candidate affinity field")
        if field_name not in CONTEXT_FIELDS:
            raise ValueError(f"unsupported candidate affinity field: {field_name}")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"candidate_menu[{index}].context_affinity values must be scalar lists")
        for value in values:
            _scalar(value, f"candidate_menu[{index}].context_affinity.{field_name}")
    return raw


def _candidate_rows(public: Mapping[str, Any]) -> list[Mapping[str, Any] | str]:
    raw_menu = public.get("candidate_menu", public.get("candidates"))
    if not isinstance(raw_menu, list) or not raw_menu:
        raise ValueError("public candidate_menu must be a nonempty list")
    return raw_menu


def generate_contextual_candidates(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Return a public-only, auditable priority proposal."""
    if not isinstance(episode, Mapping):
        raise ValueError("episode must be a mapping")
    public = _mapping(episode.get("public"), "episode.public")
    public_private = _private_paths(public, "episode.public")
    if public_private:
        raise ValueError("public proposal contains evaluator fields: " + ",".join(public_private))
    context = _context(public)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(_candidate_rows(public)):
        item = {"candidate_id": raw} if isinstance(raw, str) else dict(_mapping(raw, f"candidate_menu[{index}]"))
        candidate_id = _text(item.get("candidate_id"), f"candidate_menu[{index}].candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        private = _private_paths(item, f"candidate_menu[{index}]")
        if private:
            raise ValueError("candidate contains evaluator fields: " + ",".join(private))
        affinity = _affinity(item, index)
        matches: list[str] = []
        unknown_fields: list[str] = []
        score = 0
        for field, accepted in affinity.items():
            field_name = str(field)
            current = context.get(field_name, "unknown")
            if current == "unknown":
                unknown_fields.append(field_name)
                continue
            values = [_scalar(value, f"candidate_menu[{index}].context_affinity.{field_name}") for value in accepted]
            if current in values:
                matches.append(f"{field_name}={current}")
                score += 1
        declared_priority = item.get("declared_priority", item.get("priority", 0))
        if isinstance(declared_priority, bool) or type(declared_priority) not in (int, float) or not math.isfinite(float(declared_priority)):
            raise ValueError(f"candidate_menu[{index}].declared_priority must be finite")
        rows.append({
            "candidate_id": candidate_id,
            "context_match_score": score,
            "declared_priority": float(declared_priority),
            "matched_context": sorted(matches),
            "unknown_context_fields": sorted(set(unknown_fields)),
            "candidate_generation_status": "matched" if matches else ("context_unknown" if unknown_fields else "unmatched"),
        })
    ordered = sorted(rows, key=lambda row: (-row["context_match_score"], -row["declared_priority"], row["candidate_id"]))
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return {
        "schema_version": VERSION,
        "episode_id": episode.get("episode_id"),
        "context": context,
        "candidate_order": [row["candidate_id"] for row in ordered],
        "candidates": ordered,
        "actor_visible": True,
        "evaluator_fields_used": False,
        "numeric_voi_estimated": False,
        "generation_basis": "public_context_affinity_then_declared_priority_then_candidate_id",
    }


def apply_contextual_candidate_order(episode: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reorder only the public menu; preserve the evaluator sidecar."""
    generated = generate_contextual_candidates(episode)
    cloned = deepcopy(dict(episode))
    public = dict(_mapping(cloned.get("public"), "episode.public"))
    raw_menu = _candidate_rows(public)
    by_id: dict[str, Any] = {}
    for index, raw in enumerate(raw_menu):
        if isinstance(raw, str):
            key = _text(raw, f"candidate_menu[{index}]")
        else:
            key = _text(_mapping(raw, f"candidate_menu[{index}]").get("candidate_id"), "candidate_id")
        by_id[key] = raw
    public["candidate_menu"] = [by_id[candidate_id] for candidate_id in generated["candidate_order"]]
    public["candidate_generation_trace"] = {
        "schema_version": VERSION,
        "candidate_order": list(generated["candidate_order"]),
        "context": dict(generated["context"]),
        "evaluator_fields_used": False,
        "numeric_voi_estimated": False,
    }
    cloned["public"] = public
    return cloned, generated


def compare_contextual_candidates(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep the historical baseline unchanged and attach proposal diagnostics."""
    raw = list(episodes)
    if not raw:
        raise ValueError("at least one historical episode is required")
    traces = [generate_contextual_candidates(episode) for episode in raw]
    report = compare_historical_episodes(raw)
    return {
        "schema_version": VERSION,
        "comparison": report,
        "candidate_generation": traces,
        "proposal_diagnostic": {
            "contextual_order_is_proposal_only": True,
            "dynamic_replanning_evaluated": False,
            "numeric_values_reestimated": False,
            "baseline_replayed_on_original_menu": True,
        },
        "contract": {
            "candidate_generation_uses_public_context_only": True,
            "all_original_candidates_preserved": True,
            "evaluator_sidecar_untouched": True,
            "runtime_policy_reads_evaluator_labels": False,
            "fixed_order_baseline_unchanged": True,
            "real_source_requests": 0,
            "real_model_calls": 0,
            "numeric_voi_training_eligible": False,
        },
    }


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include timezone")
    return parsed.astimezone(timezone.utc)


def _bound_context(actor_context: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, list[str]], list[str], list[str]]:
    """Project only usable, non-expired ``values.context`` observations."""
    cutoff = _time(actor_context["prediction_cutoff"], "prediction_cutoff")
    usable_ids = {sid for sid, summary in (state.get("provenance_gate", {}).get("sources", {}) or {}).items()
                  if summary.get("state") in {"available", "duplicate"}}
    values: dict[str, list[tuple[str, str]]] = {}
    ignored: list[str] = []
    # The acquisition state hashes ``snapshot.source_observations``.  Read
    # that canonical copy rather than the convenience top-level projection so
    # an un-hashed mutation cannot change the proposed context.
    snapshot = state.get("snapshot")
    rows = snapshot.get("source_observations", []) if isinstance(snapshot, Mapping) else []
    if not isinstance(rows, list):
        raise ValueError("actor_context.source_observations must be a list")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        source_id = row.get("source_id")
        payload = row.get("values")
        has_context_payload = isinstance(payload, Mapping) and isinstance(payload.get("context"), Mapping)
        if not isinstance(source_id, str) or source_id not in usable_ids:
            if has_context_payload and isinstance(source_id, str):
                ignored.append(source_id)
            continue
        expiry = row.get("valid_until", row.get("expires_at"))
        if expiry is not None and _time(expiry, "observation expiry") < cutoff:
            ignored.append(source_id)
            continue
        context = payload.get("context") if isinstance(payload, Mapping) else None
        if not isinstance(context, Mapping):
            continue
        for field, raw_value in context.items():
            name = _text(str(field), "observed context field")
            if name not in CONTEXT_FIELDS:
                ignored.append(f"{source_id}:{name}")
                continue
            values.setdefault(name, []).append((source_id, _scalar(raw_value, f"observed context.{name}")))
    bound: dict[str, str] = {}
    support: dict[str, list[str]] = {}
    conflicts: list[str] = []
    for field, entries in sorted(values.items()):
        distinct = {value for _, value in entries}
        support[field] = sorted(source_id for source_id, _ in entries)
        if len(distinct) != 1:
            conflicts.append(field)
            bound[field] = "unknown"
        else:
            bound[field] = next(iter(distinct))
    return bound, support, sorted(set(ignored)), sorted(conflicts)


def prepare_contextual_acquisition(
    actor_context: Mapping[str, Any], registry: Sequence[Mapping[str, Any]], *, registry_complete: bool = False,
) -> dict[str, Any]:
    """Bind public context proposals to the real state and provenance gate."""
    if not isinstance(actor_context, Mapping) or "actor_context" in actor_context:
        raise ValueError("pass only the public actor_context, not its envelope")
    if type(registry_complete) is not bool:
        raise ValueError("registry_complete must be boolean")
    state = prepare_acquisition_state(actor_context, registry, candidate_generation_complete=registry_complete)
    context, support, ignored, conflicts = _bound_context(actor_context, state)
    proposal_episode = {"episode_id": actor_context.get("match_id"),
                        "public": {"context": context, "candidate_menu": list(registry)}}
    generation = generate_contextual_candidates(proposal_episode)
    return {
        "schema_version": VERSION,
        "acquisition_state": state,
        "candidate_generation": generation,
        "context_binding": {
            "context": context,
            "supporting_source_ids": support,
            "ignored_source_ids": ignored,
            "conflicting_fields": conflicts,
        },
        "contract": {
            "proposal_only": True,
            "numeric_voi_estimated": False,
            "state_bound_selection_required": True,
            "replan_after_completed_receipt": True,
        },
    }


__all__ = ["VERSION", "CONTEXT_FIELDS", "generate_contextual_candidates",
           "apply_contextual_candidate_order", "compare_contextual_candidates",
           "prepare_contextual_acquisition"]
