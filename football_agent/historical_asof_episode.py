"""Build actor-safe historical as-of episodes from frozen match records.

This module is deliberately an adapter, rather than a collector.  It accepts
already materialized rows from the platform/research exports and constructs a
single public pre-action episode at an explicit prediction cutoff.  Every
source observation is filtered by its *availability* clock, not by kickoff or
the time the row happened to be read.  Result fields are retained only in a
separate evaluator sidecar.

The adapter is conservative by design:

* a missing availability timestamp is reported as unknown and is not promoted
  into the actor snapshot; ``observed_at`` is never treated as system
  availability in strict mode;
* a missing finite candidate menu yields an incomplete sentinel menu rather
  than invented tools;
* fields that look like post-match labels are removed recursively from the
  public snapshot;
* this module never opens a database, calls a provider, trains a model, or
  derives numeric information value.

The returned envelope has three independent parts: ``episode`` (the public
contract), ``evaluator_sidecar`` (future label metadata), and ``audit`` (the
adapter's exclusions and timing decisions).  Callers should pass only
``envelope["episode"]`` to an actor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .value_evaluation_contract import VERSION as VALUE_CONTRACT_VERSION


VERSION = "historical-asof-episode-v1"
ADAPTER_VERSION = VERSION
EVALUATOR_ID = "historical-outcome-sidecar-v1"
UNKNOWN_MENU_ID = "__candidate_menu_unknown__"

FORBIDDEN_PUBLIC_KEYS = frozenset({
    "outcome", "outcomes", "label", "labels", "target_label", "result",
    "score", "scores", "goals", "home_goals", "away_goals", "final_score",
    "future_result", "hidden_return", "hidden_returns", "hidden_features",
    "full_snapshot", "post_state", "after_state", "response_body", "raw_data",
    "actual_result", "actual_outcome", "realized_loss", "decision_loss_after",
    "realized_gain", "target", "status", "score_period", "label_status",
    "result_available_at", "label_available_at", "settlement_status",
})

LABEL_KEYS = frozenset({
    "result", "outcome", "score", "home_score", "away_score", "home_goals",
    "away_goals", "final_score", "status", "score_period", "label_status",
    "result_available_at", "label_available_at", "settlement_status",
})

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(body).hexdigest()


def _time(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _optional_time(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    return _time(value, field)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any, field: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _row(value: Any, field: str = "row") -> dict[str, Any]:
    """Convert a JSON-like row or dataclass without guessing source fields."""
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if is_dataclass(value) and not isinstance(value, type):
        return deepcopy(asdict(value))
    raise ValueError(f"{field} must be a mapping or dataclass")


def _public_project(value: Any, *, path: str = "snapshot") -> Any:
    """Drop post-match fields recursively while preserving actor evidence."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in FORBIDDEN_PUBLIC_KEYS:
                continue
            result[key] = _public_project(child, path=f"{path}/{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_public_project(child, path=f"{path}/{index}")
                for index, child in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {path}")
    return deepcopy(value)


def _find_forbidden(value: Any, path: str = "snapshot") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in FORBIDDEN_PUBLIC_KEYS:
                found.append(f"{path}/{key_text}")
            found.extend(_find_forbidden(child, f"{path}/{key_text}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_find_forbidden(child, f"{path}/{index}"))
    return found


def _availability(observation: Mapping[str, Any], *, allow_observed_at_fallback: bool = False) -> tuple[datetime | None, str | None]:
    """Return the strongest explicit availability clock without backdating.

    ``available_at`` is preferred, followed by system operational availability
    and ingestion.  A source observation clock is only used when the caller
    explicitly enables ``allow_observed_at_fallback`` for a legacy replay.  The
    selected basis is retained in the audit.
    """
    keys = ["available_at", "operational_available_at", "ingested_at"]
    if allow_observed_at_fallback:
        keys.append("observed_at")
    for key in keys:
        if observation.get(key) is not None:
            return _time(observation[key], f"source_observations.{key}"), key
    return None, None


def _normalise_source_observations(row: Mapping[str, Any], cutoff: datetime, *, allow_observed_at_fallback: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    raw = row.get("source_observations", row.get("observations", row.get("sources", [])))
    issues: list[dict[str, Any]] = []
    unknown: list[str] = []
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("source_observations must be a list")
    visible: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            issues.append({"index": index, "reason": "row_not_mapping"})
            continue
        source_id = item.get("source_id") or item.get("evidence_id")
        if not isinstance(source_id, str) or not source_id.strip():
            issues.append({"index": index, "reason": "missing_source_id"})
            unknown.append(f"source_observations[{index}]")
            continue
        source_id = source_id.strip()
        if source_id in seen_source_ids:
            issues.append({"source_id": source_id, "reason": "duplicate_source_id"})
            unknown.append(source_id)
            continue
        seen_source_ids.add(source_id)
        observed: datetime | None = None
        if item.get("observed_at") is not None:
            try:
                observed = _time(item["observed_at"], f"source_observations[{index}].observed_at")
            except ValueError:
                issues.append({"source_id": source_id, "reason": "invalid_observed_at"})
                unknown.append(source_id)
                continue
            if observed > cutoff:
                issues.append({"source_id": source_id, "reason": "observed_after_cutoff", "observed_at": _iso(observed)})
                unknown.append(source_id)
                continue
        try:
            available_at, basis = _availability(item, allow_observed_at_fallback=allow_observed_at_fallback)
        except ValueError as exc:
            issues.append({"source_id": source_id, "reason": "invalid_availability", "detail": str(exc)})
            unknown.append(source_id)
            continue
        if available_at is None:
            issues.append({"source_id": source_id, "reason": "missing_availability"})
            unknown.append(source_id)
            continue
        if available_at > cutoff:
            issues.append({"source_id": source_id, "reason": "after_cutoff", "available_at": _iso(available_at)})
            continue
        content_digest = item.get("content_digest") or item.get("content_hash")
        if content_digest is not None and (not isinstance(content_digest, str) or not _DIGEST_RE.fullmatch(content_digest)):
            issues.append({"source_id": source_id, "reason": "invalid_content_digest"})
            unknown.append(source_id)
            continue
        source_status = item.get("status", "observed")
        # A provider's post-match status is not an actor-visible source
        # quality state.  Keep the observation but redact the semantic leak.
        if source_status in {"finished", "completed", "settled", "post_match"}:
            issues.append({"source_id": source_id, "reason": "post_match_status_redacted"})
            source_status = "unknown"
            unknown.append(source_id)
        projected: dict[str, Any] = {
            "source_id": source_id,
            "kind": item.get("kind", item.get("source", "unknown")),
            # ``quality`` is an actor-visible source quality annotation.  The
            # generic ``status`` key is intentionally forbidden because some
            # providers use it for post-match settlement state.
            "quality": source_status,
            "available_at": _iso(available_at),
            "availability_basis": basis,
        }
        if observed is not None:
            projected["observed_at"] = _iso(observed)
        if content_digest is not None:
            projected["content_digest"] = content_digest
        if isinstance(item.get("values"), Mapping):
            values = _public_project(item["values"], path=f"source_observations[{index}]/values")
            if values:
                projected["values"] = values
        visible.append(projected)
    visible.sort(key=lambda item: (item["available_at"], item["source_id"]))
    return visible, issues, sorted(set(unknown))


def _normalise_candidates(row: Mapping[str, Any], cutoff: datetime, source_ids: set[str], *, candidate_catalog_complete: bool | None) -> tuple[list[dict[str, Any]], bool, list[str]]:
    raw = row.get("candidate_menu", row.get("candidates"))
    unknown: list[str] = []
    if raw is None:
        unknown.append("candidate_menu")
        return ([{"candidate_id": UNKNOWN_MENU_ID, "kind": "unknown", "source_ids": [],
                  "declared_cost": {}, "availability": "unknown"}], False, unknown)
    if not isinstance(raw, list) or not raw:
        raise ValueError("candidate_menu must be a nonempty list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"candidate_menu[{index}] must be an object")
        candidate_id = _text(item.get("candidate_id"), f"candidate_menu[{index}].candidate_id")
        assert candidate_id is not None
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        source_list = item.get("source_ids", [])
        if not isinstance(source_list, list) or any(not isinstance(source, str) or not source.strip() for source in source_list):
            raise ValueError(f"candidate_menu[{index}].source_ids must be a list of strings")
        source_list = list(dict.fromkeys(source_list))
        missing_sources = sorted(set(source_list) - source_ids)
        if missing_sources:
            unknown.extend(f"{candidate_id}:source:{source}" for source in missing_sources)
        cost = item.get("declared_cost", {})
        if not isinstance(cost, Mapping) or any(not isinstance(key, str) or not key.strip() or type(value) is not int or value < 0 for key, value in cost.items()):
            raise ValueError(f"candidate_menu[{index}].declared_cost must be a nonnegative integer mapping")
        availability = item.get("availability")
        if availability is None:
            availability = "available" if source_list and not missing_sources else "unknown"
        if availability not in {"available", "unavailable", "unknown", "stale", "duplicate"}:
            raise ValueError(f"candidate_menu[{index}].availability is invalid")
        projected: dict[str, Any] = {
            "candidate_id": candidate_id,
            "kind": item.get("kind", "unknown"),
            "source_ids": source_list,
            "declared_cost": dict(cost),
            "availability": availability,
        }
        for key in ("description", "requires"):
            if key in item:
                projected[key] = deepcopy(item[key])
        result.append(projected)
    raw_complete = row.get("candidate_generation_complete")
    if candidate_catalog_complete is None and raw_complete is not None and type(raw_complete) is not bool:
        raise ValueError("candidate_generation_complete must be boolean when provided")
    complete = candidate_catalog_complete if candidate_catalog_complete is not None else raw_complete is True
    if type(complete) is not bool:
        raise ValueError("candidate_catalog_complete must be boolean when provided")
    return result, complete, sorted(set(unknown))


def _label_sidecar(row: Mapping[str, Any], match_id: str, kickoff: datetime) -> dict[str, Any]:
    home_goals = row.get("home_goals", row.get("home_score"))
    away_goals = row.get("away_goals", row.get("away_score"))
    has_score = type(home_goals) is int and home_goals >= 0 and type(away_goals) is int and away_goals >= 0
    label_time = row.get("label_available_at", row.get("result_available_at"))
    available_at = None
    if label_time is not None:
        try:
            available_at = _time(label_time, "label_available_at")
        except ValueError:
            available_at = None
    label_status = row.get("label_status")
    if label_status not in {"INDEPENDENT_EVALUATOR_GOLD", "ACTOR_PROVISIONAL", "UNAVAILABLE"}:
        label_status = "UNAVAILABLE"
    outcome: dict[str, Any] = {
        "match_id": match_id,
        "kickoff": _iso(kickoff),
        "label_status": label_status,
        "evaluator_id": row.get("evaluator_id", EVALUATOR_ID),
        "available_at": None if available_at is None else _iso(available_at),
        "score_period": row.get("score_period", "unknown"),
        "status": row.get("status", "unknown"),
    }
    if has_score:
        outcome["home_goals"] = home_goals
        outcome["away_goals"] = away_goals
        outcome["outcome"] = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
    outcome["outcome_digest"] = _digest({key: outcome[key] for key in ("match_id", "home_goals", "away_goals", "score_period", "status") if key in outcome})
    return outcome


def build_historical_asof_episode(
    match: Mapping[str, Any] | Any,
    *,
    prediction_cutoff: datetime | str | None = None,
    candidate_catalog_complete: bool | None = None,
    allow_observed_at_fallback: bool = False,
) -> dict[str, Any]:
    """Build one actor-safe historical episode envelope.

    The match row must carry an explicit kickoff and stable match ID.  A cutoff
    may come from the row or the function argument and must be no later than
    kickoff.  The returned ``episode`` contains no score/result fields.  In
    strict mode a source with only ``observed_at`` remains unknown; callers
    replaying legacy exports must opt into the fallback explicitly.
    """
    row = _row(match)
    match_id = _text(row.get("match_id", row.get("canonical_match_id")), "match_id")
    assert match_id is not None
    kickoff = _time(row.get("kickoff"), "kickoff")
    cutoff = _time(prediction_cutoff if prediction_cutoff is not None else row.get("prediction_cutoff"), "prediction_cutoff")
    if cutoff > kickoff:
        raise ValueError("prediction_cutoff must not follow kickoff")
    if row.get("status") in {"cancelled", "postponed", "suspended"}:
        raise ValueError("inactive fixture cannot form a historical as-of episode")

    if type(allow_observed_at_fallback) is not bool:
        raise ValueError("allow_observed_at_fallback must be boolean")
    observations, observation_issues, unknown_sources = _normalise_source_observations(
        row, cutoff, allow_observed_at_fallback=allow_observed_at_fallback)
    visible_source_ids = {item["source_id"] for item in observations}
    candidates, menu_complete, unknown_candidates = _normalise_candidates(
        row, cutoff, visible_source_ids, candidate_catalog_complete=candidate_catalog_complete)

    target_key = _text(row.get("target_key"), "target_key", required=False) or f"match:{match_id}"
    episode_id = f"historical:{_digest({'match_id': match_id, 'prediction_cutoff': _iso(cutoff), 'adapter': VERSION})[:32]}"
    snapshot: dict[str, Any] = {
        "match_id": match_id,
        "target_key": target_key,
        "competition_id": row.get("competition_id"),
        "competition_name": row.get("competition_name"),
        "home_team_id": row.get("home_team_id"),
        "away_team_id": row.get("away_team_id"),
        "home_team_name": row.get("home_team_name"),
        "away_team_name": row.get("away_team_name"),
        "kickoff": _iso(kickoff),
        "prediction_cutoff": _iso(cutoff),
        "information_mode": row.get("information_mode", "historical_reconstruction"),
        "source": row.get("source", "unknown"),
        "source_observations": observations,
        "known": _public_project(row.get("known", {}), path="known") if isinstance(row.get("known", {}), Mapping) else {},
    }
    snapshot = {key: value for key, value in snapshot.items() if value is not None}
    # Public rows supplied by a collector may have legacy fields that were not
    # copied above.  Do not copy them implicitly; the explicit projection is
    # the point where the historical boundary is enforced.
    snapshot = json.loads(json.dumps(_public_project(snapshot), ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":"), allow_nan=False))
    leaks = _find_forbidden(snapshot)
    if leaks:
        raise ValueError("public snapshot contains forbidden fields: " + ",".join(leaks))
    snapshot_digest = _digest(snapshot)
    operation_id = f"historical-selection:{episode_id}"
    sidecar = _label_sidecar(row, match_id, kickoff)
    audit = {
        "adapter_version": ADAPTER_VERSION,
        "episode_id": episode_id,
        "match_id": match_id,
        "prediction_cutoff": _iso(cutoff),
        "kickoff": _iso(kickoff),
        "included_source_count": len(observations),
        "excluded_source_count": len(observation_issues),
        "excluded_sources": observation_issues,
        "unknown_fields": sorted(set(unknown_sources + unknown_candidates)),
        "candidate_generation_complete": menu_complete,
        "candidate_menu_digest": _digest([item["candidate_id"] for item in candidates]),
        "public_snapshot_digest": snapshot_digest,
        "label_is_sidecar_only": True,
        "real_source_requests": 0,
        "numeric_voi_calibrated": False,
        "allow_observed_at_fallback": allow_observed_at_fallback,
    }
    episode = {
        "schema_version": VALUE_CONTRACT_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "episode_id": episode_id,
        "target_key": target_key,
        "prediction_cutoff": _iso(cutoff),
        "pre_action": {
            "snapshot": snapshot,
            "snapshot_digest": snapshot_digest,
            "candidate_ids": [item["candidate_id"] for item in candidates],
            "candidate_generation_complete": menu_complete,
            "selection_operation_id": operation_id,
            "decision": "defer",
            "selected_candidate_id": None,
            "selected_at": _iso(cutoff),
        },
        "candidate_evaluations": [],
        "evaluation_outcome": None,
        "evaluator_sidecar": {
            "status": "not_attached",
            "required": True,
            "runtime_can_read": False,
        },
        "historical_asof": {
            "adapter_version": ADAPTER_VERSION,
            "source_observation_count": len(observations),
            "candidate_generation_complete": menu_complete,
            "unknown_fields": audit["unknown_fields"],
            "label_sidecar_digest": _digest(sidecar),
            "runtime_safe": True,
            "allow_observed_at_fallback": allow_observed_at_fallback,
        },
    }
    return {"episode": episode, "evaluator_sidecar": sidecar, "audit": audit}


def build_historical_asof_episodes(
    matches: Iterable[Mapping[str, Any] | Any],
    *,
    prediction_cutoff: datetime | str | None = None,
    candidate_catalog_complete: bool | None = None,
    allow_observed_at_fallback: bool = False,
) -> dict[str, Any]:
    """Build a deterministic batch and retain row-level exclusions."""
    episodes: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        try:
            episodes.append(build_historical_asof_episode(
                match, prediction_cutoff=prediction_cutoff,
                candidate_catalog_complete=candidate_catalog_complete,
                allow_observed_at_fallback=allow_observed_at_fallback))
        except (TypeError, ValueError) as exc:
            exclusions.append({"index": index, "reason": str(exc)})
    return {
        "schema_version": VERSION,
        "episode_count": len(episodes),
        "episodes": episodes,
        "exclusions": exclusions,
        "runtime_safe": True,
        "numeric_voi_calibrated": False,
    }


def validate_historical_asof_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Audit the actor/evaluator separation without reading external state."""
    reasons: list[str] = []
    if not isinstance(envelope, Mapping):
        return {"valid": False, "reasons": ["envelope_not_mapping"]}
    episode = envelope.get("episode")
    if not isinstance(episode, Mapping):
        return {"valid": False, "reasons": ["episode_missing"]}
    snapshot = episode.get("pre_action", {}).get("snapshot") if isinstance(episode.get("pre_action"), Mapping) else None
    if not isinstance(snapshot, Mapping):
        reasons.append("public_snapshot_missing")
    else:
        reasons.extend(f"public_result_leak:{path}" for path in _find_forbidden(snapshot))
    cutoff = None
    try:
        cutoff = _time(episode.get("prediction_cutoff"), "prediction_cutoff")
    except ValueError:
        reasons.append("invalid_prediction_cutoff")
    if isinstance(snapshot, Mapping) and cutoff is not None:
        for item in snapshot.get("source_observations", []):
            try:
                available = _time(item.get("available_at"), "available_at")
                if available > cutoff:
                    reasons.append(f"source_after_cutoff:{item.get('source_id')}")
            except (TypeError, ValueError):
                reasons.append(f"invalid_source_time:{item.get('source_id')}")
            observed_value = item.get("observed_at")
            if observed_value is not None:
                try:
                    observed = _time(observed_value, "observed_at")
                    if observed > cutoff:
                        reasons.append(f"source_observed_after_cutoff:{item.get('source_id')}")
                except (TypeError, ValueError):
                    reasons.append(f"invalid_source_observed_time:{item.get('source_id')}")
    if envelope.get("evaluator_sidecar") is None:
        reasons.append("evaluator_sidecar_missing")
    return {"valid": not reasons, "episode_id": episode.get("episode_id"), "reasons": reasons}


__all__ = [
    "VERSION", "ADAPTER_VERSION", "EVALUATOR_ID", "UNKNOWN_MENU_ID",
    "build_historical_asof_episode", "build_historical_asof_episodes",
    "validate_historical_asof_envelope",
]
