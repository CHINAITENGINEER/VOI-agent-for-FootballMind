"""Source observation freshness and availability contract for Round134.

The contract keeps source provenance separate from outcome labels and from any
probability or reliability estimate.  It answers a narrower question:
*could this observation have been used at the supplied prediction cutoff?*

The module is deliberately offline.  A collector or a scenario bridge can
pass an observation manifest, and an Agent can use the returned status to
choose a refresh/alternative-source action.  No status is converted into a
probability, a gold label, or a claim about the truth of the underlying
football fact.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

from .types import utc


VERSION = "source-reliability-contract-v1"
REPORT_VERSION = "source-reliability-report-v1"

FRESHNESS = frozenset({"fresh", "stale", "unknown"})
AVAILABILITY = frozenset({
    "available", "missing", "blocked", "failed", "unknown", "late",
})
SOURCE_STATES = frozenset({
    "available", "stale", "future", "missing", "blocked", "failed",
    "unknown", "duplicate", "conflict",
})


class SourceManifestError(ValueError):
    """Raised when an observation manifest is structurally malformed."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceManifestError(f"{name}_REQUIRED")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _time(value: Any, name: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise SourceManifestError(f"{name}_REQUIRED")
        return None
    try:
        return utc(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise SourceManifestError(f"{name}_INVALID_TIME") from exc


def _digest(value: Any) -> str | None:
    """Keep source content hashes opaque but require a nonempty string.

    Production collectors may use SHA-256, while fixtures often use a short
    deterministic token.  The contract does not infer a hash algorithm.
    """
    if value is None:
        return None
    return _text(value, "digest")


def _canonical_digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(body).hexdigest()


@dataclass(frozen=True)
class SourceObservation:
    """One immutable source observation and its temporal provenance.

    ``observed_at`` is when the source observation was made and ``available_at``
    is when the Agent could operationally consume it.  ``valid_until`` is an
    exclusive freshness bound; a cutoff at or after that timestamp treats the
    observation as stale.  Missing timestamps remain explicit unknowns.
    """

    source_id: str
    observed_at: str | None = None
    available_at: str | None = None
    valid_until: str | None = None
    freshness: str = "unknown"
    availability: str = "unknown"
    conflict_group: str | None = None
    digest: str | None = None
    observation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        observed = _time(self.observed_at, "observed_at")
        available = _time(self.available_at, "available_at")
        valid_until = _time(self.valid_until, "valid_until")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "valid_until", valid_until)
        if self.freshness not in FRESHNESS:
            raise SourceManifestError("freshness_UNSUPPORTED")
        if self.availability not in AVAILABILITY:
            raise SourceManifestError("availability_UNSUPPORTED")
        object.__setattr__(self, "conflict_group",
                           _optional_text(self.conflict_group, "conflict_group"))
        object.__setattr__(self, "digest", _digest(self.digest))
        object.__setattr__(self, "observation_id",
                           _optional_text(self.observation_id, "observation_id"))
        if observed and available and utc(available) < utc(observed):
            raise SourceManifestError("available_at_BEFORE_observed_at")
        if available and valid_until and utc(valid_until) <= utc(available):
            raise SourceManifestError("valid_until_NOT_AFTER_available_at")
        if observed and valid_until and utc(valid_until) <= utc(observed):
            raise SourceManifestError("valid_until_NOT_AFTER_observed_at")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceObservation":
        if not isinstance(value, Mapping):
            raise SourceManifestError("observation_OBJECT_REQUIRED")
        return cls(
            source_id=value.get("source_id"),
            observed_at=value.get("observed_at"),
            available_at=value.get("available_at", value.get("operational_available_at")),
            valid_until=value.get("valid_until", value.get("expires_at")),
            freshness=value.get("freshness", "unknown"),
            availability=value.get("availability", "unknown"),
            conflict_group=value.get("conflict_group"),
            digest=value.get("digest", value.get("content_hash")),
            observation_id=value.get("observation_id"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "source_id": self.source_id,
            "observed_at": self.observed_at,
            "available_at": self.available_at,
            "valid_until": self.valid_until,
            "freshness": self.freshness,
            "availability": self.availability,
            "conflict_group": self.conflict_group,
            "digest": self.digest,
            "observation_id": self.observation_id,
        }


def _overlap(left: SourceObservation, right: SourceObservation) -> bool:
    """Return whether two observations have overlapping validity windows."""
    left_start = utc(left.available_at or left.observed_at) if (left.available_at or left.observed_at) else None
    right_start = utc(right.available_at or right.observed_at) if (right.available_at or right.observed_at) else None
    if left_start is None or right_start is None:
        return False
    left_end = utc(left.valid_until) if left.valid_until else None
    right_end = utc(right.valid_until) if right.valid_until else None
    start = max(left_start, right_start)
    if left_end is not None and start >= left_end:
        return False
    if right_end is not None and start >= right_end:
        return False
    return True


def _effective_state(observation: SourceObservation, cutoff: datetime) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if observation.observed_at and utc(observation.observed_at) > cutoff:
        reasons.append("OBSERVED_AFTER_CUTOFF")
    if observation.available_at and utc(observation.available_at) > cutoff:
        reasons.append("AVAILABLE_AFTER_CUTOFF")
    if reasons:
        return "future", tuple(reasons)
    if observation.valid_until and cutoff >= utc(observation.valid_until):
        return "stale", ("VALID_UNTIL_REACHED",)
    if observation.freshness == "stale":
        return "stale", ("DECLARED_STALE",)
    if observation.availability != "available":
        return observation.availability, (f"AVAILABILITY_{observation.availability.upper()}",)
    if observation.observed_at is None or observation.available_at is None:
        return "unknown", ("TIMESTAMP_UNKNOWN",)
    if observation.digest is None:
        return "unknown", ("DIGEST_UNKNOWN",)
    if observation.freshness != "fresh":
        return "unknown", ("FRESHNESS_UNKNOWN",)
    return "available", ()


def _source_groups(rows: list[SourceObservation]) -> tuple[set[int], set[int], set[int]]:
    """Return duplicate row indexes, conflicting row indexes and group indexes."""
    duplicate: set[int] = set()
    conflict: set[int] = set()
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (row.source_id, row.digest or "<unknown>")
        grouped.setdefault(key, []).append(index)
    for indexes in grouped.values():
        if len(indexes) > 1 and rows[indexes[0]].digest is not None:
            duplicate.update(indexes[1:])

    # A source itself may revise over time.  That is only a conflict when the
    # validity windows overlap; non-overlapping revisions are a normal history.
    for i, left in enumerate(rows):
        for j in range(i + 1, len(rows)):
            right = rows[j]
            if left.digest is None or right.digest is None or left.digest == right.digest:
                continue
            same_source = left.source_id == right.source_id
            same_group = (left.conflict_group is not None and
                          left.conflict_group == right.conflict_group)
            if (same_source or same_group) and _overlap(left, right):
                conflict.update({i, j})
    return duplicate, conflict, set(index for indexes in grouped.values() for index in indexes)


def _source_summary(
    source_id: str,
    rows: list[dict[str, Any]],
    *,
    duplicate: bool,
    conflict: bool,
) -> dict[str, Any]:
    states = [row["state"] for row in rows]
    reasons = sorted({reason for row in rows for reason in row["reasons"]})
    usable = "available" in states and not conflict
    if conflict:
        state = "conflict"
        action = "resolve_conflict"
        reasons.append("CONFLICTING_DIGESTS_OR_SOURCES")
    elif usable and duplicate:
        state = "duplicate"
        action = "use_one_canonical_observation"
        reasons.append("DUPLICATE_OBSERVATION_DEDUP_REQUIRED")
    elif usable:
        state = "available"
        action = "use"
    elif "stale" in states:
        state = "stale"
        action = "refresh_source"
    elif "future" in states:
        state = "future"
        action = "wait_until_cutoff_or_replay"
    elif states and all(item in {"missing", "blocked", "failed"} for item in states):
        state = next(item for item in ("missing", "blocked", "failed") if item in states)
        action = "query_alternative_source"
    else:
        state = "unknown"
        action = "defer_unknown"
    return {
        "source_id": source_id,
        "state": state,
        "action": action,
        "usable_at_cutoff": usable,
        "duplicate": duplicate,
        "conflict": conflict,
        "observation_count": len(rows),
        "observations": rows,
        "reasons": sorted(set(reasons)),
        # This field is deliberately explicit: state is provenance readiness,
        # never an outcome probability or evaluator gold label.
        "probability_gold": False,
    }


def _candidate_filter(
    candidate: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise SourceManifestError("candidate_id_REQUIRED")
    raw_ids = candidate.get("source_ids", ())
    if not isinstance(raw_ids, (list, tuple)):
        raise SourceManifestError("candidate_source_ids_LIST_REQUIRED")
    source_ids = tuple(dict.fromkeys(str(item).strip() for item in raw_ids if str(item).strip()))
    if not source_ids:
        return {
            "candidate_id": candidate_id,
            "source_ids": [],
            "eligibility": "defer_unknown",
            "reasons": ["NO_SOURCE_IDS"],
            "probability_gold": False,
        }
    rows = [sources.get(source_id) for source_id in source_ids]
    missing = [source_id for source_id, row in zip(source_ids, rows) if row is None]
    states = [row["state"] for row in rows if row is not None]
    reasons = list(missing and ["SOURCE_MANIFEST_MISSING"] or [])
    if missing:
        reasons.extend(f"SOURCE_ID_UNKNOWN:{source_id}" for source_id in missing)
    if any(state == "conflict" for state in states):
        reasons.append("SOURCE_CONFLICT_REQUIRES_RESOLUTION")
    if any(state == "stale" for state in states):
        reasons.append("SOURCE_STALE_REQUIRES_REFRESH")
    if any(state == "future" for state in states):
        reasons.append("SOURCE_FUTURE_NOT_VISIBLE_AT_CUTOFF")
    if any(state in {"missing", "blocked", "failed"} for state in states):
        reasons.append("SOURCE_UNAVAILABLE")
    if any(state == "duplicate" for state in states):
        reasons.append("DUPLICATE_SOURCE_DEDUP_REQUIRED")
    if any(state in {"available", "duplicate"} for state in states) and not any(state == "conflict" for state in states):
        eligibility = "eligible_with_warnings" if reasons else "eligible"
    else:
        eligibility = "defer_unknown"
    return {
        "candidate_id": candidate_id,
        "source_ids": list(source_ids),
        "eligibility": eligibility,
        "reasons": sorted(set(reasons)),
        "probability_gold": False,
    }


def evaluate_source_manifest(
    observations: Iterable[SourceObservation | Mapping[str, Any]],
    *,
    cutoff: str,
    candidates: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate source visibility and freshness at an as-of cutoff.

    The returned report is suitable for candidate filtering.  It has no
    numeric reliability score and no outcome label.  Unknown provenance stays
    ``unknown``/``defer_unknown`` so callers must query, refresh, or stop with
    an explicit reason instead of silently treating an unverified source as
    evidence gold.
    """
    try:
        cutoff_value = utc(cutoff)
    except (TypeError, ValueError) as exc:
        raise SourceManifestError("cutoff_INVALID_TIME") from exc
    rows = [item if isinstance(item, SourceObservation) else SourceObservation.from_mapping(item)
            for item in observations]
    duplicate_indexes, conflict_indexes, _ = _source_groups(rows)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        state, reasons = _effective_state(row, cutoff_value)
        if index in conflict_indexes:
            state = "conflict"
            reasons = tuple(sorted(set(reasons) | {"CONFLICTING_DIGESTS_OR_SOURCES"}))
        by_source.setdefault(row.source_id, []).append({
            **row.as_dict(),
            "state": state,
            "reasons": list(reasons),
            "visible_at_cutoff": state in {"available", "duplicate"},
        })
    summaries: dict[str, dict[str, Any]] = {}
    for source_id, source_rows in sorted(by_source.items()):
        duplicate = any(rows[index].source_id == source_id for index in duplicate_indexes)
        conflict = any(rows[index].source_id == source_id for index in conflict_indexes)
        summaries[source_id] = _source_summary(source_id, source_rows,
                                                duplicate=duplicate, conflict=conflict)
    candidate_report = []
    if candidates is not None:
        candidate_report = [_candidate_filter(candidate, summaries) for candidate in candidates]
    core = {
        "schema_version": REPORT_VERSION,
        "cutoff": cutoff_value.isoformat(),
        "observations": [row for source_id in sorted(by_source) for row in by_source[source_id]],
        "sources": summaries,
        "duplicate_source_ids": sorted({rows[index].source_id for index in duplicate_indexes}),
        "conflict_source_ids": sorted({rows[index].source_id for index in conflict_indexes}),
        "candidate_filter": candidate_report,
        "unknown_is_not_gold": True,
        "probability_gold": False,
        "numeric_reliability_estimate": None,
    }
    report = dict(core)
    report["report_digest"] = _canonical_digest(core)
    return report


def validate_source_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a serialized report without assigning quality probabilities."""
    if not isinstance(report, Mapping) or report.get("schema_version") != REPORT_VERSION:
        raise SourceManifestError("report_SCHEMA_VERSION_INVALID")
    expected = report.get("report_digest")
    if not isinstance(expected, str) or expected != _canonical_digest({
        key: report[key] for key in report if key != "report_digest"
    }):
        raise SourceManifestError("report_DIGEST_MISMATCH")
    if report.get("unknown_is_not_gold") is not True or report.get("probability_gold") is not False:
        raise SourceManifestError("report_GOLD_BOUNDARY_INVALID")
    if report.get("numeric_reliability_estimate") is not None:
        raise SourceManifestError("report_NUMERIC_RELIABILITY_FORBIDDEN")
    return {
        "valid": True,
        "report_digest": expected,
        "source_count": len(report.get("sources", {})),
        "candidate_count": len(report.get("candidate_filter", [])),
    }


__all__ = [
    "VERSION", "REPORT_VERSION", "FRESHNESS", "AVAILABILITY", "SOURCE_STATES",
    "SourceManifestError", "SourceObservation", "evaluate_source_manifest",
    "validate_source_report",
]
