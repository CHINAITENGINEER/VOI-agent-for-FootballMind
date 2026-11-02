"""Independent outcome and action-loss sidecar contract.

This module is deliberately evaluator-only.  Match results are accepted only
from a source that is demonstrably separate from the actor evidence.  A result
may be observed after the prediction cutoff, but it is never projected into an
actor/public state.  Action losses are computed only when the action binding,
pre/post distributions, outcome label mapping, and all relevant clocks are
complete; otherwise the result is explicitly ``UNKNOWN`` or ``PARTIAL``.

No network/provider calls are made here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping, Sequence

VERSION = "round167-independent-outcome-loss-v1"
COMPUTED = "COMPUTED"
UNKNOWN = "UNKNOWN"
PARTIAL = "PARTIAL"
REJECTED = "REJECTED"
INDEPENDENT_GOLD = "INDEPENDENT_GOLD"

_FORBIDDEN_PUBLIC = frozenset({
    "result", "outcome", "outcome_label", "label", "score", "scores",
    "home_score", "away_score", "home_goals", "away_goals", "goals",
    "actual_result", "target_label", "loss", "decision_loss",
    "log_loss", "brier", "realized_loss", "result_source", "source_payload",
})
_EPS = 1e-12


class IndependentOutcomeError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IndependentOutcomeError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise IndependentOutcomeError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise IndependentOutcomeError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _probabilities(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise IndependentOutcomeError(f"{name} must be a nonempty probability vector")
    values = tuple(float(item) if not isinstance(item, bool) else math.nan for item in value)
    if any(not math.isfinite(v) or v < -_EPS for v in values):
        raise IndependentOutcomeError(f"{name} contains invalid probability")
    if abs(sum(values) - 1.0) > 1e-8:
        raise IndependentOutcomeError(f"{name} must sum to one")
    return values


def _digest_list(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        raise IndependentOutcomeError(f"{name} must be a sequence")
    result = []
    for item in value:
        text = _text(item, name)
        if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
            raise IndependentOutcomeError(f"{name} contains an invalid SHA256")
        result.append(text)
    return tuple(result)


def _text_list(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        raise IndependentOutcomeError(f"{name} must be a sequence")
    return tuple(_text(item, name) for item in value)


def _public_leaks(value: Any, path: str = "public") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in _FORBIDDEN_PUBLIC:
                leaks.append(f"{path}.{key_text}")
            leaks.extend(_public_leaks(child, f"{path}.{key_text}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            leaks.extend(_public_leaks(child, f"{path}[{index}]"))
    return leaks


def _actor_binding(actor: Mapping[str, Any] | None) -> dict[str, Any]:
    if actor is None:
        return {"source_ids": (), "source_digests": (), "sources": ()}
    if not isinstance(actor, Mapping):
        raise IndependentOutcomeError("actor_binding must be a mapping")
    source_ids_value = actor.get("source_ids") if actor.get("source_ids") is not None else actor.get("source_id")
    source_digests_value = actor.get("source_digests") if actor.get("source_digests") is not None else actor.get("content_digests")
    sources_value = actor.get("sources") if actor.get("sources") is not None else actor.get("source")
    source_ids = _text_list(source_ids_value, "actor.source_ids")
    source_digests = _digest_list(source_digests_value, "actor.source_digests")
    sources = _text_list(sources_value, "actor.sources")
    cutoff = actor.get("prediction_cutoff")
    if cutoff is not None:
        cutoff = _iso(_time(cutoff, "actor.prediction_cutoff"))
    return {
        "episode_id": actor.get("episode_id"),
        "target_key": actor.get("target_key"),
        "prediction_cutoff": cutoff,
        "source_ids": source_ids,
        "source_digests": source_digests,
        "sources": sources,
        "actor_id": actor.get("actor_id"),
    }


def _binding_digest(binding: Mapping[str, Any]) -> str:
    """Digest only the actor-side identity that an outcome is joined to."""
    return digest({
        "episode_id": binding.get("episode_id"),
        "target_key": binding.get("target_key"),
        "prediction_cutoff": binding.get("prediction_cutoff"),
        "source_ids": list(binding.get("source_ids", ())),
        "source_digests": list(binding.get("source_digests", ())),
        "sources": list(binding.get("sources", ())),
        "actor_id": binding.get("actor_id"),
    })


def build_independent_outcome(
    outcome: Mapping[str, Any], *, actor_binding: Mapping[str, Any] | None = None,
    evaluator_id: str = "independent-outcome-v1",
) -> dict[str, Any]:
    """Normalize one post-cutoff result into an evaluator-only record.

    ``outcome`` must contain a distinct source, digest and availability clock.
    The returned record retains the score/label in the sidecar but its public
    projection is result-free.
    """
    if not isinstance(outcome, Mapping):
        raise IndependentOutcomeError("outcome must be a mapping")
    source = _text(outcome.get("source", outcome.get("result_source")), "outcome.source")
    source_id = _text(outcome.get("source_id", outcome.get("result_source_id")), "outcome.source_id")
    source_digest = _digest_list(outcome.get("source_payload_digest", outcome.get("content_digest")), "outcome.source_payload_digest")
    if len(source_digest) != 1:
        raise IndependentOutcomeError("outcome.source_payload_digest is required")
    source_digest = source_digest[0]
    evaluator_id = _text(evaluator_id, "evaluator_id")
    if actor_binding is None:
        raise IndependentOutcomeError("OUTCOME_ACTOR_BINDING_REQUIRED")
    actor = _actor_binding(actor_binding)
    if not actor["sources"] and not actor["source_ids"]:
        raise IndependentOutcomeError("OUTCOME_ACTOR_SOURCES_REQUIRED")
    actor_sources = set(actor["sources"])
    actor_source_ids = set(actor["source_ids"])
    actor_digests = set(actor["source_digests"])
    actor_id_tokens = set(actor_source_ids)
    actor_id_tokens.update(item.rsplit(":", 1)[-1] for item in actor_source_ids)
    if source in actor_sources or source_id in actor_id_tokens or source_digest in actor_digests:
        raise IndependentOutcomeError("OUTCOME_SOURCE_NOT_INDEPENDENT")
    if outcome.get("actor_id") is not None and outcome.get("actor_id") == actor.get("actor_id"):
        raise IndependentOutcomeError("OUTCOME_EVALUATOR_ACTOR_ID_COLLISION")
    episode_id = _text(outcome.get("episode_id", actor.get("episode_id")), "outcome.episode_id")
    target_key = _text(outcome.get("target_key", actor.get("target_key")), "outcome.target_key")
    cutoff_value = outcome.get("prediction_cutoff", actor.get("prediction_cutoff"))
    cutoff = _time(cutoff_value, "outcome.prediction_cutoff")
    observed = _time(outcome.get("observed_at", outcome.get("result_observed_at")), "outcome.observed_at")
    available = _time(outcome.get("available_at", outcome.get("result_available_at")), "outcome.available_at")
    if available <= cutoff:
        raise IndependentOutcomeError("OUTCOME_AVAILABLE_BEFORE_CUTOFF")
    if observed > available:
        raise IndependentOutcomeError("OUTCOME_OBSERVED_AFTER_AVAILABLE")
    kickoff = _time(outcome.get("kickoff"), "outcome.kickoff")
    if kickoff > observed:
        raise IndependentOutcomeError("OUTCOME_OBSERVED_BEFORE_KICKOFF")
    label = _text(outcome.get("outcome_label", outcome.get("label")), "outcome.outcome_label")
    mapping = outcome.get("label_mapping")
    if not isinstance(mapping, Mapping) or not mapping:
        raise IndependentOutcomeError("OUTCOME_LABEL_MAPPING_REQUIRED")
    label_mapping = {str(k): int(v) for k, v in mapping.items()}
    if label not in label_mapping:
        raise IndependentOutcomeError("OUTCOME_LABEL_NOT_IN_MAPPING")
    if len(set(label_mapping.values())) != len(label_mapping) or min(label_mapping.values()) < 0:
        raise IndependentOutcomeError("OUTCOME_LABEL_MAPPING_INVALID")
    score = {
        key: outcome[key] for key in ("home_goals", "away_goals", "home_score", "away_score")
        if key in outcome
    }
    if not score:
        raise IndependentOutcomeError("OUTCOME_RESULT_FIELDS_REQUIRED")
    record = {
        "schema_version": VERSION,
        "status": INDEPENDENT_GOLD,
        "evaluator_id": evaluator_id,
        "independent_from_actor": True,
        "actor_binding_digest": _binding_digest(actor),
        "actor_source_ids": list(actor["source_ids"]),
        "actor_source_digests": list(actor["source_digests"]),
        "actor_sources": list(actor["sources"]),
        "episode_id": episode_id,
        "target_key": target_key,
        "prediction_cutoff": _iso(cutoff),
        "source": source,
        "source_id": source_id,
        "source_payload_digest": source_digest,
        "kickoff": _iso(kickoff),
        "observed_at": _iso(observed),
        "available_at": _iso(available),
        "outcome_label": label,
        "label_mapping": label_mapping,
        **score,
        "actor_public_projection": {"episode_id": episode_id, "target_key": target_key,
                                    "prediction_cutoff": _iso(cutoff),
                                    "outcome_available_at": _iso(available),
                                    "outcome_visible_to_actor": False},
    }
    record["record_digest"] = digest(record)
    return record


def _verify_outcome(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise IndependentOutcomeError("outcome record must be a mapping")
    if record.get("schema_version") != VERSION or record.get("status") != INDEPENDENT_GOLD:
        raise IndependentOutcomeError("OUTCOME_RECORD_SCHEMA_MISMATCH")
    if record.get("independent_from_actor") is not True:
        raise IndependentOutcomeError("OUTCOME_INDEPENDENCE_REQUIRED")
    copy = dict(record)
    digest_value = copy.pop("record_digest", None)
    if digest_value != digest(copy):
        raise IndependentOutcomeError("OUTCOME_RECORD_DIGEST_MISMATCH")
    # Protect the evaluator record from a forged public projection.
    projection = record.get("actor_public_projection")
    if not isinstance(projection, Mapping) or projection.get("outcome_visible_to_actor") is not False:
        raise IndependentOutcomeError("OUTCOME_PUBLIC_PROJECTION_INVALID")


def _loss(metric: str, probabilities: tuple[float, ...], index: int) -> float:
    if metric == "log_loss":
        return -math.log(max(probabilities[index], _EPS))
    if metric == "brier":
        return sum((p - (1.0 if i == index else 0.0)) ** 2 for i, p in enumerate(probabilities))
    raise IndependentOutcomeError("LOSS_METRIC_UNSUPPORTED")


def compute_action_loss(
    action: Mapping[str, Any], outcome: Mapping[str, Any], *, actor_binding: Mapping[str, Any] | None = None,
    metric: str = "log_loss",
) -> dict[str, Any]:
    """Compute before/after loss only for a fully bound, cutoff-safe action."""
    if not isinstance(action, Mapping):
        return {"status": UNKNOWN, "reasons": ["ACTION_MAPPING_MISSING"]}
    try:
        _verify_outcome(outcome)
    except IndependentOutcomeError as exc:
        return {"status": UNKNOWN, "reasons": [str(exc)]}
    reasons: list[str] = []
    if actor_binding is None:
        return {"status": UNKNOWN, "reasons": ["ACTOR_BINDING_MISSING"], "numeric_voi_training_eligible": False}
    actor = _actor_binding(actor_binding)
    if outcome.get("actor_binding_digest") != _binding_digest(actor):
        reasons = ["OUTCOME_ACTOR_BINDING_MISMATCH"]
        return {"status": UNKNOWN, "reasons": reasons, "numeric_voi_training_eligible": False}
    for key in ("episode_id", "target_key", "candidate_id", "request_id", "action_id"):
        if not isinstance(action.get(key), str) or not action[key].strip():
            reasons.append(f"ACTION_{key.upper()}_MISSING")
    if action.get("episode_id") != outcome.get("episode_id") or action.get("target_key") != outcome.get("target_key"):
        reasons.append("ACTION_OUTCOME_BINDING_MISMATCH")
    if actor.get("episode_id") is not None and action.get("episode_id") != actor["episode_id"]:
        reasons.append("ACTION_ACTOR_EPISODE_MISMATCH")
    if actor.get("target_key") is not None and action.get("target_key") != actor["target_key"]:
        reasons.append("ACTION_ACTOR_TARGET_MISMATCH")
    cutoff = _time(outcome["prediction_cutoff"], "outcome.prediction_cutoff")
    if actor.get("prediction_cutoff") is not None and _time(actor["prediction_cutoff"], "actor.prediction_cutoff") != cutoff:
        reasons.append("ACTION_CUTOFF_BINDING_MISMATCH")
    try:
        captured = _time(action.get("captured_at"), "action.captured_at")
    except IndependentOutcomeError as exc:
        reasons.append("ACTION_CAPTURE_TIME_MISSING")
        captured = None
    if captured is not None and captured > cutoff:
        reasons.append("ACTION_AFTER_CUTOFF")
    try:
        before = _probabilities(action.get("prediction_before"), "prediction_before")
        after = _probabilities(action.get("prediction_after"), "prediction_after")
    except IndependentOutcomeError as exc:
        reasons.append(str(exc))
        before = after = None
    mapping = outcome.get("label_mapping")
    label = outcome.get("outcome_label")
    index = mapping.get(label) if isinstance(mapping, Mapping) else None
    if not isinstance(index, int) or index < 0:
        reasons.append("OUTCOME_LABEL_MAPPING_INCOMPLETE")
    elif before is not None and after is not None and (index >= len(before) or len(before) != len(after)):
        reasons.append("PREDICTION_LABEL_DIMENSION_MISMATCH")
    if reasons:
        status = PARTIAL if any(r not in {"ACTION_MAPPING_MISSING", "ACTION_OUTCOME_BINDING_MISMATCH", "ACTION_ACTOR_EPISODE_MISMATCH", "ACTION_ACTOR_TARGET_MISMATCH"} for r in reasons) else UNKNOWN
        return {"status": status, "reasons": sorted(set(reasons)), "numeric_voi_training_eligible": False}
    before_loss = _loss(metric, before, index)
    after_loss = _loss(metric, after, index)
    result = {
        "status": COMPUTED,
        "metric": metric,
        "episode_id": action["episode_id"], "target_key": action["target_key"],
        "candidate_id": action["candidate_id"], "request_id": action["request_id"],
        "action_id": action["action_id"],
        "captured_at": _iso(captured), "prediction_cutoff": _iso(cutoff),
        "outcome_available_at": outcome["available_at"],
        "decision_loss_before": round(before_loss, 12),
        "decision_loss_after": round(after_loss, 12),
        "loss_reduction": round(before_loss - after_loss, 12),
        "outcome_record_digest": outcome["record_digest"],
        "numeric_voi_training_eligible": True,
    }
    result["loss_digest"] = digest(result)
    return result


def build_loss_registry(actions: Iterable[Mapping[str, Any]], outcome: Mapping[str, Any], *, actor_binding: Mapping[str, Any] | None = None, metric: str = "log_loss") -> dict[str, Any]:
    """Compute a sidecar registry; failures remain explicit and auditable."""
    try:
        _verify_outcome(outcome)
    except IndependentOutcomeError as exc:
        return {"schema_version": VERSION, "status": UNKNOWN, "results": [], "reasons": [str(exc)], "numeric_voi_training_eligible": False}
    results = [compute_action_loss(action, outcome, actor_binding=actor_binding, metric=metric) for action in actions]
    statuses = {row["status"] for row in results}
    status = (COMPUTED if results and statuses == {COMPUTED}
              else UNKNOWN if results and statuses == {UNKNOWN}
              else PARTIAL if results else UNKNOWN)
    return {
        "schema_version": VERSION, "status": status,
        "outcome_record_digest": outcome["record_digest"],
        "results": results,
        "numeric_voi_training_eligible": status == COMPUTED,
        "public_projection": {
            "episode_id": outcome["episode_id"],
            "target_key": outcome["target_key"],
            "prediction_cutoff": outcome["prediction_cutoff"],
            "outcome_visible_to_actor": False,
            "loss_visible_to_actor": False,
        },
    }


def public_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return only a result-free projection suitable for actor/public state."""
    if not isinstance(value, Mapping):
        raise IndependentOutcomeError("public projection requires a mapping")
    forbidden = _public_leaks(value)
    if forbidden:
        raise IndependentOutcomeError("OUTCOME_FIELDS_IN_PUBLIC_STATE:" + ",".join(forbidden))
    projection = deepcopy(dict(value))
    projection.pop("outcome_record_digest", None)
    projection.pop("results", None)
    projection["outcome_visible_to_actor"] = False
    projection["loss_visible_to_actor"] = False
    return projection


__all__ = [
    "VERSION", "COMPUTED", "UNKNOWN", "PARTIAL", "REJECTED", "INDEPENDENT_GOLD",
    "IndependentOutcomeError", "digest", "build_independent_outcome",
    "compute_action_loss", "build_loss_registry", "public_projection",
]
