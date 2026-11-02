"""Bridge historical as-of envelopes into adaptive replay episodes.

This adapter is the seam between the collector-facing historical contract and
the Round143 counterfactual replay contract.  It copies only the public
pre-action snapshot, requires explicit candidate routes and costs, and accepts
the evaluator sidecar as a separate argument so labels never become actor
context by accident.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any

from .historical_asof_episode import validate_historical_asof_envelope


VERSION = "historical-replay-adapter-v1"
_FORBIDDEN_PUBLIC_KEYS = frozenset({
    "result", "outcome", "score", "scores", "goals",
    "home_score", "away_score", "final_score", "label",
    "loss", "decision_loss", "realized_loss", "realized_gain",
    "evaluator", "sidecar", "sequence_losses", "action_returns",
    "post_state", "after_state", "prediction", "probability",
})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    else:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _public(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = _text(str(raw_key), f"{path}.key")
            if key.casefold() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"{path} contains an evaluator-only field: {key}")
            result[key] = _public(child, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_public(child, f"{path}[]") for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} contains an unsupported value")


def build_replay_episode_from_historical_envelope(
    envelope: Mapping[str, Any], *,
    candidate_routes: Mapping[str, Mapping[str, Any]],
    budget: float,
    evaluator_sidecar: Mapping[str, Any],
    initial_context: Mapping[str, Any] | None = None,
    loss_unit: str = "decision_loss",
) -> dict[str, Any]:
    """Create a Round143 replay episode from a validated historical envelope.

    ``candidate_routes`` is deliberately explicit because the historical
    envelope stores candidate identity but does not invent query costs or tool
    routes.  ``evaluator_sidecar`` is copied as a private sibling of ``public``
    and must carry independent gold labels before replay can score a policy.
    """
    if not isinstance(envelope, Mapping):
        raise ValueError("envelope must be a mapping")
    validation = validate_historical_asof_envelope(envelope)
    if not validation.get("valid"):
        raise ValueError("historical envelope is invalid: " + "; ".join(validation.get("reasons", ())))
    episode = envelope.get("episode")
    if not isinstance(episode, Mapping):
        raise ValueError("envelope.episode is required")
    pre_action = episode.get("pre_action")
    if not isinstance(pre_action, Mapping):
        raise ValueError("episode.pre_action is required")
    snapshot = pre_action.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("episode.pre_action.snapshot is required")
    candidate_ids = pre_action.get("candidate_ids")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise ValueError("historical envelope must contain candidate_ids")
    normalized_ids = [_text(item, "candidate_ids[]") for item in candidate_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("candidate_ids must be unique")
    if not isinstance(candidate_routes, Mapping):
        raise ValueError("candidate_routes must be a mapping")
    if set(candidate_routes) != set(normalized_ids):
        missing = sorted(set(normalized_ids) - set(candidate_routes))
        extra = sorted(set(candidate_routes) - set(normalized_ids))
        raise ValueError(f"candidate route set mismatch: missing={missing}, extra={extra}")
    menu: list[dict[str, Any]] = []
    for candidate_id in normalized_ids:
        route = candidate_routes[candidate_id]
        if not isinstance(route, Mapping):
            raise ValueError(f"candidate_routes[{candidate_id}] must be a mapping")
        source_id = _text(route.get("source_id"), f"{candidate_id}.source_id")
        action_id = _text(route.get("action_id"), f"{candidate_id}.action_id")
        cost = _finite(route.get("cost"), f"{candidate_id}.cost", minimum=0)
        row = {"candidate_id": candidate_id, "source_id": source_id,
               "action_id": action_id, "cost": cost,
               "action_kind": route.get("action_kind", "query")}
        menu.append(row)
    if isinstance(budget, bool):
        raise ValueError("budget must be finite")
    budget_value = _finite(budget, "budget", minimum=0)
    if not isinstance(evaluator_sidecar, Mapping):
        raise ValueError("evaluator_sidecar must be a mapping")
    if evaluator_sidecar.get("label_status") != "INDEPENDENT_GOLD":
        raise ValueError("evaluator_sidecar.label_status must be INDEPENDENT_GOLD")
    target_key = _text(episode.get("target_key") or snapshot.get("target_key"), "target_key")
    episode_id = _text(episode.get("episode_id"), "episode_id")
    cutoff = _time(episode.get("prediction_cutoff") or snapshot.get("prediction_cutoff"), "prediction_cutoff")
    kickoff = _time(snapshot.get("kickoff"), "kickoff")
    if cutoff > kickoff:
        raise ValueError("prediction_cutoff must not follow kickoff")
    if not isinstance(loss_unit, str) or not loss_unit.strip():
        raise ValueError("loss_unit must be a nonempty string")
    context_source = snapshot.get("known", {}) if initial_context is None else initial_context
    if not isinstance(context_source, Mapping):
        raise ValueError("initial_context must be a mapping")
    context = _public(context_source, "initial_context")
    public = {
        "target_key": target_key, "kickoff": kickoff,
        "prediction_cutoff": cutoff, "initial_context": context,
        "candidate_generation_complete": pre_action.get("candidate_generation_complete") is True,
        "candidate_menu": menu, "budget": budget_value,
        "loss_unit": loss_unit,
        "snapshot_digest": pre_action.get("snapshot_digest"),
        "information_mode": snapshot.get("information_mode", "historical_reconstruction"),
    }
    public = _public(public, "public")
    return {
        "episode_id": episode_id,
        "public": public,
        "evaluation_sidecar": deepcopy(dict(evaluator_sidecar)),
        "adapter": {"schema_version": VERSION,"historical_envelope_validation": validation,
                    "runtime_can_read_sidecar": False,"candidate_route_count": len(menu)},
    }


__all__ = ["VERSION", "build_replay_episode_from_historical_envelope"]
