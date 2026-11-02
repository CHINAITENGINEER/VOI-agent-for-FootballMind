"""Convert real collector observations into append-only ledger metadata.

This adapter sits after ``football_spider.collect`` and before semantic
normalization.  It records when a request was made, what the source returned,
which dependency it consumed, and whether entity mapping was usable.  Raw
payload bytes and source parsers stay in the spider archive; this module only
creates the metadata contract needed by source-return calibration and VoI
holdout evaluation.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .types import utc


VERSION = "collector-ledger-adapter-v1"
_SUCCESS = frozenset(range(200, 300))
_MAPPING_STATES = frozenset({"mapped", "unmapped", "ambiguous", "not_evaluated"})


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("collector result or observation must be a mapping/model")


def _text(value: Any, name: str, *, default: str | None = None) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _mapping_state(value: Any) -> str:
    if value is None:
        return "not_evaluated"
    value = str(value).strip().lower()
    aliases = {"valid": "mapped", "stable": "mapped", "ok": "mapped",
               "missing": "unmapped", "unknown": "not_evaluated"}
    value = aliases.get(value, value)
    if value not in _MAPPING_STATES:
        raise ValueError(f"Unsupported entity mapping state: {value}")
    return value


def normalized_batch_mapping_state(batch: Mapping[str, Any] | Any) -> str:
    """Infer mapping usability from a normalized batch without reading the DB.

    ``stable`` source identifiers are usable for source-level joins.  A
    ``source_name`` identifier is retained as ambiguous because a name-only
    join can collide across teams; ``match_scoped`` identifiers are treated as
    unmapped.  Parser/validation issues that mention identity relations are
    surfaced as ambiguity rather than silently upgrading the response.
    """
    payload = _dump(batch)
    matches = payload.get("matches", []) or []
    issues = payload.get("issues", []) or []
    if not matches:
        return "not_evaluated"
    identity_quality = []
    for match in matches:
        for key in ("competition", "home", "away"):
            ref = match.get(key) or {}
            identity_quality.append(ref.get("id_quality", "unknown"))
    if any(value == "match_scoped" for value in identity_quality):
        return "unmapped"
    if any(value in {"source_name", "unknown"} for value in identity_quality):
        return "ambiguous"
    for issue in issues:
        rule = str(issue.get("rule", ""))
        if "CONFLICT" in rule or "AMBIG" in rule:
            return "ambiguous"
        if "UNRESOLVED" in rule or "IDENTITY" in rule:
            return "unmapped"
    return "mapped"


def _quality(obs: Mapping[str, Any], result_status: str, mapping_state: str) -> str:
    if obs.get("error_class") or obs.get("http_status") not in _SUCCESS:
        return "PROVIDER_ERROR"
    if mapping_state == "ambiguous":
        return "CONFLICT"
    if mapping_state == "unmapped":
        return "SOURCE_MISSING"
    if str(result_status).upper() == "PARTIAL":
        return "PARTIAL"
    return "VALIDATED" if mapping_state == "mapped" else "UNVERIFIED"


def _availability(quality: str, observed_at: str | None, cutoff: str) -> str:
    if quality in {"PROVIDER_ERROR", "SOURCE_MISSING", "INVALID", "STALE", "CONFLICT"}:
        return "failed"
    if observed_at is None:
        return "failed"
    return "available" if utc(observed_at) <= utc(cutoff) else "late"


def observation_event(
    observation: Mapping[str, Any] | Any,
    *,
    result_status: str,
    source_version: str,
    prediction_cutoff: str,
    candidate_id: str | None = None,
    entity_mapping_state: str | None = None,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Return one normalized metadata event for one archived request attempt."""
    obs = _dump(observation)
    source = _text(obs.get("source"), "source")
    observation_id = _text(obs.get("id"), "observation.id")
    request_id = _text(obs.get("request_id"), "observation.request_id")
    observed_at = obs.get("observed_at")
    observed_at = utc(observed_at).isoformat() if observed_at else None
    requested_at = (obs.get("context") or {}).get("requested_at")
    if requested_at is None:
        requested_at = (obs.get("context") or {}).get("request_started_at")
    requested_at = utc(requested_at).isoformat() if requested_at else None
    captured = utc(captured_at or observed_at or prediction_cutoff).isoformat()
    state = _mapping_state(entity_mapping_state)
    quality = _quality(obs, result_status, state)
    availability = _availability(quality, observed_at, prediction_cutoff)
    # A late response still has a real operational arrival timestamp. The
    # Ledger needs that timestamp to distinguish a late result from a failed
    # request; the point-in-time bridge decides whether it was usable.
    operational = observed_at if availability in {"available", "late"} else None
    return {
        "schema_version": VERSION,
        "candidate_id": _text(candidate_id, "candidate_id", default=f"observation:{observation_id}"),
        "source_id": source,
        "source_version": _text(source_version, "source_version"),
        "dependency_key": f"{source}:request:{request_id}",
        "observation_id": observation_id,
        "request_id": request_id,
        "endpoint": _text(obs.get("endpoint"), "observation.endpoint"),
        "quality": quality,
        "availability": availability,
        "observed_at": observed_at,
        "operational_available_at": operational,
        "requested_at": requested_at,
        "captured_at": captured,
        "prediction_cutoff": utc(prediction_cutoff).isoformat(),
        "http_status": obs.get("http_status"),
        "error_class": obs.get("error_class"),
        "entity_mapping_state": state,
        "evidence_refs": [f"raw-observation:{observation_id}", _text(obs.get("payload_path"), "observation.payload_path")],
        "content_hash": _text(obs.get("content_hash"), "observation.content_hash"),
    }


def collection_events(
    result: Mapping[str, Any] | Any,
    observations: Iterable[Mapping[str, Any] | Any],
    *,
    source_version: str,
    prediction_cutoff: str,
    candidate_by_observation: Mapping[str, str] | None = None,
    entity_mapping_by_observation: Mapping[str, str] | None = None,
    normalized_by_observation: Mapping[str, Any] | None = None,
    captured_at: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a collection run into deterministic event rows.

    ``observations`` is intentionally supplied by the raw archive reader.  A
    collection result contains IDs and status, while the archive contains the
    immutable timing, response and checksum fields needed here.
    """
    payload = _dump(result)
    status = str(payload.get("status", "FAILED"))
    by_id = {str(item.get("id")): item for item in (_dump(obs) for obs in observations)}
    expected = [str(item) for item in payload.get("observations", [])]
    ids = expected or sorted(by_id)
    candidate_by_observation = candidate_by_observation or {}
    entity_mapping_by_observation = entity_mapping_by_observation or {}
    normalized_by_observation = normalized_by_observation or {}
    events = []
    for observation_id in ids:
        if observation_id not in by_id:
            continue
        events.append(observation_event(
            by_id[observation_id], result_status=status, source_version=source_version,
            prediction_cutoff=prediction_cutoff,
            candidate_id=candidate_by_observation.get(observation_id),
            entity_mapping_state=entity_mapping_by_observation.get(
                observation_id,
                normalized_batch_mapping_state(normalized_by_observation[observation_id])
                if observation_id in normalized_by_observation else None,
            ),
            captured_at=captured_at,
        ))
    return events


def events_to_ledger_entries(
    events: Iterable[Mapping[str, Any]],
    *,
    episode_id: str,
    target_key: str,
    action_id: str,
    prediction_cutoff: str,
    resolution_status: str = "UNRESOLVED",
    sequence_start: int = 0,
):
    """Materialize collector events as append-only ``EvidenceLedgerEntry`` objects."""
    from .evidence_ledger import EvidenceLedgerEntry

    entries = []
    for offset, event in enumerate(events):
        row = dict(event)
        entries.append(EvidenceLedgerEntry.from_row(
            row, episode_id=episode_id, target_key=target_key, action_id=action_id,
            request_id=row["request_id"], sequence=sequence_start + offset,
            prediction_cutoff=prediction_cutoff, captured_at=row.get("captured_at"),
            resolution_status=resolution_status, requested_at=row.get("requested_at"),
            source_version=row["source_version"],
        ))
    return tuple(entries)


def append_collection_events(ledger, events: Iterable[Mapping[str, Any]], **kwargs):
    """Append collector events through the ledger's checksum/lock boundary."""
    return ledger.append(events_to_ledger_entries(events, **kwargs))


__all__ = ["VERSION", "normalized_batch_mapping_state", "observation_event", "collection_events",
           "events_to_ledger_entries", "append_collection_events"]
