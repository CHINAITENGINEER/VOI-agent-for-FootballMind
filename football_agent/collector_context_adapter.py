"""Project real collector metadata into an agent-visible pre-query context.

The source normalizers intentionally retain both pre-match fields and result
fields in one ``NormalizedBatch``.  Passing that batch directly to an Agent
would make it very easy to leak a score or a post-match status into the
context used for a query decision.  This adapter therefore uses a small,
explicit whitelist of structural pre-query facts and keeps provenance in a
separate evidence record.

The adapter does not infer qualification, rotation, or tactical intent.  A
collector can establish that a fixture exists and when it starts; those
football states remain unresolved until a suitable source is queried.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import math
from typing import Any

from .context_query_readiness import ContextState
from .evidence_binding import AVAILABILITY, QUALITIES
from .types import utc


VERSION = "collector-context-adapter-v1"
_ACCEPTED_QUALITIES = frozenset({"VALIDATED", "PARTIAL", "UNVERIFIED"})
_HIDDEN_KEYS = frozenset({
    "result", "outcome", "score", "home_score", "away_score", "half_score",
    "final_score", "hidden_result", "probability", "realized_loss",
})
_PRE_MATCH_STATUSES = frozenset({"scheduled", "postponed", "cancelled", "suspended"})
_CONTEXT_KEYS = frozenset({
    "fixture_id", "competition_id", "competition_name", "home_team_id",
    "away_team_id", "kickoff_at", "stage", "pre_match_status",
    "competition_id_quality", "home_team_id_quality", "away_team_id_quality",
    "home_rank_known", "away_rank_known", "odds_available", "quote_count",
    "lead_time_seconds",
})


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("collector event or normalized batch must be a mapping/model")


def _text(value: Any, name: str, *, default: str | None = None) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str, *, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    return utc(value).isoformat()


def _scalar(value: Any, name: str) -> Any:
    """Accept only finite scalar values in the Agent-visible context."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{name} must be a finite scalar")


def _entity(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"normalized match is missing {name}")
    return value


def _quality_usable(event: Mapping[str, Any], cutoff: str) -> tuple[bool, str]:
    quality = _text(event.get("quality"), "event.quality", default="UNVERIFIED")
    availability = _text(event.get("availability"), "event.availability", default="failed")
    if quality not in QUALITIES:
        raise ValueError(f"Unsupported event quality: {quality}")
    if availability not in AVAILABILITY:
        raise ValueError(f"Unsupported event availability: {availability}")
    observed = event.get("observed_at")
    operational = event.get("operational_available_at")
    if availability != "available" or quality not in _ACCEPTED_QUALITIES:
        return False, f"event_not_usable:{availability}:{quality}"
    if observed is None or operational is None:
        return False, "event_missing_operational_timestamp"
    if utc(observed) > utc(cutoff) or utc(operational) > utc(cutoff):
        return False, "event_arrived_after_prediction_cutoff"
    return True, "source_available_at_cutoff"


def _event_metadata(event: Mapping[str, Any], cutoff: str) -> dict[str, Any]:
    """Copy only provenance fields; payload/result fields are never copied."""
    # An explicit hidden-result marker is an invalid collector event.  This is
    # intentionally strict so a caller cannot silently label a contaminated
    # record as public context.
    if bool(event.get("hidden_result_included", False)):
        raise ValueError("collector event is marked as containing a hidden result")
    source_id = _text(event.get("source_id"), "event.source_id")
    source_version = _text(event.get("source_version"), "event.source_version")
    observation_id = _text(event.get("observation_id"), "event.observation_id")
    dependency_key = _text(event.get("dependency_key"), "event.dependency_key")
    captured = _time(event.get("captured_at") or cutoff, "event.captured_at")
    observed = _time(event.get("observed_at"), "event.observed_at", required=False)
    operational = _time(event.get("operational_available_at"), "event.operational_available_at", required=False)
    refs = tuple(_text(ref, "event.evidence_ref") for ref in (event.get("evidence_refs") or ()))
    return {
        "source_id": source_id,
        "source_version": source_version,
        "observation_id": observation_id,
        "dependency_key": dependency_key,
        "quality": _text(event.get("quality"), "event.quality", default="UNVERIFIED"),
        "availability": _text(event.get("availability"), "event.availability", default="failed"),
        "observed_at": observed,
        "operational_available_at": operational,
        "captured_at": captured,
        "prediction_cutoff": _time(event.get("prediction_cutoff") or cutoff, "prediction_cutoff"),
        "evidence_refs": refs,
        "content_hash": _text(event.get("content_hash"), "event.content_hash", default=""),
    }


@dataclass(frozen=True)
class ContextEvidence:
    """Provenance for one projected context key.

    ``accepted`` is separate from ``quality``: a failed/late source can be
    retained for audit without becoming an Agent-visible fact.
    """

    context_key: str
    value: Any
    target_key: str
    source_id: str
    source_version: str
    observation_id: str
    dependency_key: str
    quality: str
    availability: str
    observed_at: str | None
    operational_available_at: str | None
    captured_at: str
    prediction_cutoff: str
    evidence_refs: tuple[str, ...] = ()
    content_hash: str = ""
    accepted: bool = False
    reason: str = ""
    visible_to_agent: bool = True
    hidden_result_included: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_key", _text(self.context_key, "context_key"))
        if self.context_key not in _CONTEXT_KEYS or self.context_key.lower() in _HIDDEN_KEYS:
            raise ValueError("context key is outside the public pre-query whitelist")
        object.__setattr__(self, "target_key", _text(self.target_key, "target_key"))
        for field in ("source_id", "source_version", "observation_id", "dependency_key"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if self.quality not in QUALITIES or self.availability not in AVAILABILITY:
            raise ValueError("unsupported evidence quality or availability")
        object.__setattr__(self, "observed_at", _time(self.observed_at, "observed_at", required=False))
        object.__setattr__(self, "operational_available_at",
                           _time(self.operational_available_at, "operational_available_at", required=False))
        object.__setattr__(self, "captured_at", _time(self.captured_at, "captured_at"))
        object.__setattr__(self, "prediction_cutoff", _time(self.prediction_cutoff, "prediction_cutoff"))
        if self.observed_at and self.operational_available_at and utc(self.observed_at) > utc(self.operational_available_at):
            raise ValueError("observed_at cannot follow operational_available_at")
        object.__setattr__(self, "value", _scalar(self.value, "context evidence value"))
        object.__setattr__(self, "evidence_refs", tuple(_text(ref, "evidence_ref") for ref in self.evidence_refs))
        object.__setattr__(self, "content_hash", str(self.content_hash))
        if not isinstance(self.accepted, bool) or not isinstance(self.visible_to_agent, bool):
            raise ValueError("accepted and visible_to_agent must be booleans")
        if self.hidden_result_included is not False:
            raise ValueError("context evidence must exclude hidden results")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "context_key": self.context_key,
            "value": self.value,
            "target_key": self.target_key,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "observation_id": self.observation_id,
            "dependency_key": self.dependency_key,
            "quality": self.quality,
            "availability": self.availability,
            "observed_at": self.observed_at,
            "operational_available_at": self.operational_available_at,
            "captured_at": self.captured_at,
            "prediction_cutoff": self.prediction_cutoff,
            "evidence_refs": list(self.evidence_refs),
            "content_hash": self.content_hash,
            "accepted": self.accepted,
            "reason": self.reason,
            "visible_to_agent": self.visible_to_agent,
            "hidden_result_included": False,
        }


@dataclass(frozen=True)
class CollectorContextSnapshot:
    """Agent-visible context plus an immutable provenance trail."""

    target_key: str
    prediction_cutoff: str
    context: ContextState
    evidence: tuple[ContextEvidence, ...]
    conflict_keys: tuple[str, ...] = ()
    hidden_result_included: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_key", _text(self.target_key, "target_key"))
        object.__setattr__(self, "prediction_cutoff", _time(self.prediction_cutoff, "prediction_cutoff"))
        if not isinstance(self.context, ContextState):
            raise ValueError("context must be a ContextState")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "conflict_keys", tuple(sorted(set(self.conflict_keys))))
        if self.hidden_result_included is not False:
            raise ValueError("context snapshots must exclude hidden results")
        for row in self.evidence:
            if row.target_key != self.target_key or row.prediction_cutoff != self.prediction_cutoff:
                raise ValueError("context evidence target/cutoff differs from snapshot")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "target_key": self.target_key,
            "prediction_cutoff": self.prediction_cutoff,
            "context": self.context.as_dict(),
            "evidence": [row.as_dict() for row in self.evidence],
            "conflict_keys": list(self.conflict_keys),
            "hidden_result_included": False,
        }


def _match_rows(batch: Any) -> tuple[dict[str, Any], ...]:
    payload = _dump(batch)
    rows = payload.get("matches", ())
    if not isinstance(rows, (list, tuple)):
        raise ValueError("normalized batch matches must be a sequence")
    return tuple(_dump(row) for row in rows)


def _quote_rows(batch: Any) -> tuple[dict[str, Any], ...]:
    payload = _dump(batch)
    rows = payload.get("quotes", ())
    if not isinstance(rows, (list, tuple)):
        raise ValueError("normalized batch quotes must be a sequence")
    return tuple(_dump(row) for row in rows)


def _safe_projection(match: Mapping[str, Any], quotes: tuple[dict[str, Any], ...], cutoff: str) -> dict[str, Any]:
    """Build structural context without reading score/result fields."""
    competition = _entity(match, "competition")
    home = _entity(match, "home")
    away = _entity(match, "away")
    kickoff = _time(match.get("kickoff"), "match.kickoff")
    evidence = match.get("evidence") if isinstance(match.get("evidence"), Mapping) else {}
    status = str(match.get("status", "")).strip().lower()
    projection: dict[str, Any] = {
        "fixture_id": _text(match.get("source_id"), "match.source_id"),
        "competition_id": _text(competition.get("source_id"), "competition.source_id"),
        "competition_name": _text(competition.get("name"), "competition.name"),
        "home_team_id": _text(home.get("source_id"), "home.source_id"),
        "away_team_id": _text(away.get("source_id"), "away.source_id"),
        "kickoff_at": kickoff,
        "competition_id_quality": _text(competition.get("id_quality"), "competition.id_quality", default="unknown"),
        "home_team_id_quality": _text(home.get("id_quality"), "home.id_quality", default="unknown"),
        "away_team_id_quality": _text(away.get("id_quality"), "away.id_quality", default="unknown"),
        "home_rank_known": evidence.get("home_rank_raw") not in (None, ""),
        "away_rank_known": evidence.get("away_rank_raw") not in (None, ""),
        "odds_available": any(str(row.get("source_match_id")) == str(match.get("source_id")) for row in quotes),
        "quote_count": sum(1 for row in quotes if str(row.get("source_match_id")) == str(match.get("source_id"))),
    }
    if match.get("stage") not in (None, ""):
        projection["stage"] = str(match["stage"])
    # A scheduled/postponed state is public pre-match context.  Finished/live
    # status is deliberately excluded because it can act as an outcome proxy.
    if status in _PRE_MATCH_STATUSES:
        projection["pre_match_status"] = status
    lead = (utc(kickoff) - utc(cutoff)).total_seconds()
    if math.isfinite(lead):
        projection["lead_time_seconds"] = lead
    return projection


def build_context_snapshot(
    event: Mapping[str, Any] | Any,
    normalized_batch: Mapping[str, Any] | Any,
    *,
    target_source_match_id: str | None = None,
    target_key: str | None = None,
    prediction_cutoff: str | None = None,
) -> CollectorContextSnapshot:
    """Project one archived collection response into a ContextState.

    The target match is selected by its source ID.  If the batch has exactly
    one match, the ID may be omitted.  Quote-only responses can still project
    ``odds_available`` when a target ID is supplied; they do not manufacture
    fixture identity fields.
    """
    event_payload = _dump(event)
    cutoff = _time(prediction_cutoff or event_payload.get("prediction_cutoff"), "prediction_cutoff")
    metadata = _event_metadata(event_payload, cutoff)
    usable, usability_reason = _quality_usable(event_payload, cutoff)
    matches = _match_rows(normalized_batch)
    quotes = _quote_rows(normalized_batch)
    target_id = str(target_source_match_id) if target_source_match_id is not None else None
    selected = [row for row in matches if target_id is None or str(row.get("source_id")) == target_id]
    if target_id is None and len(matches) == 1:
        selected = list(matches)
    if len(selected) > 1:
        raise ValueError("target_source_match_id is required when a batch has multiple matching fixtures")
    if selected:
        match = selected[0]
        target_id = str(match.get("source_id"))
    elif target_id is None:
        raise ValueError("normalized batch has no target fixture")
    target = _text(target_key or event_payload.get("candidate_id") or f"match:{target_id}", "target_key")
    values: dict[str, Any] = {}
    evidence_rows: list[ContextEvidence] = []
    projection = _safe_projection(selected[0], quotes, cutoff) if selected else {
        "odds_available": any(str(row.get("source_match_id")) == target_id for row in quotes),
        "quote_count": sum(1 for row in quotes if str(row.get("source_match_id")) == target_id),
    }
    for key, value in projection.items():
        accepted = usable
        reason = usability_reason if not accepted else "projected_public_pre_query_fact"
        evidence_rows.append(ContextEvidence(
            context_key=key, value=value if accepted else None, target_key=target,
            source_id=metadata["source_id"], source_version=metadata["source_version"],
            observation_id=metadata["observation_id"], dependency_key=metadata["dependency_key"],
            quality=metadata["quality"], availability=metadata["availability"],
            observed_at=metadata["observed_at"], operational_available_at=metadata["operational_available_at"],
            captured_at=metadata["captured_at"], prediction_cutoff=metadata["prediction_cutoff"],
            evidence_refs=metadata["evidence_refs"], content_hash=metadata["content_hash"],
            accepted=accepted, reason=reason,
        ))
        if accepted:
            values[key] = value
    return CollectorContextSnapshot(
        target_key=target, prediction_cutoff=cutoff,
        context=ContextState(values), evidence=tuple(evidence_rows),
    )


def merge_context_snapshots(snapshots: Iterable[CollectorContextSnapshot]) -> CollectorContextSnapshot:
    """Merge source snapshots while retaining conflicts as unresolved values."""
    materialized = tuple(snapshots)
    if not materialized:
        raise ValueError("At least one context snapshot is required")
    target = materialized[0].target_key
    cutoff = materialized[0].prediction_cutoff
    if any(row.target_key != target or row.prediction_cutoff != cutoff for row in materialized):
        raise ValueError("Context snapshots must share target and prediction cutoff")
    all_evidence = tuple(item for snapshot in materialized for item in snapshot.evidence)
    by_key: dict[str, list[Any]] = {}
    for row in all_evidence:
        if row.accepted:
            by_key.setdefault(row.context_key, []).append(row.value)
    values: dict[str, Any] = {}
    conflicts: list[str] = []
    for key, candidates in by_key.items():
        unique = []
        for value in candidates:
            if value not in unique:
                unique.append(value)
        # Coverage facts are monotone across source endpoints: an odds-only
        # response can add quotes to a fixture snapshot that previously had no
        # quote rows.  Treating ``False`` and ``True`` as a semantic conflict
        # would discard exactly the update the second collector supplied.
        if key == "odds_available":
            values[key] = any(bool(value) for value in candidates)
            continue
        if key == "quote_count":
            values[key] = max(int(value) for value in candidates)
            continue
        if len(unique) == 1:
            values[key] = unique[0]
        else:
            conflicts.append(key)
    return CollectorContextSnapshot(
        target_key=target, prediction_cutoff=cutoff, context=ContextState(values),
        evidence=all_evidence, conflict_keys=tuple(conflicts),
    )


__all__ = [
    "VERSION", "ContextEvidence", "CollectorContextSnapshot",
    "build_context_snapshot", "merge_context_snapshots",
]
