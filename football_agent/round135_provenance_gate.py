"""As-of provenance gate for autonomous candidate selection (Round135).

The gate is a small, offline bridge between a historical as-of episode and the
Round134 source manifest contract.  It answers two different questions:

* Is the evidence already visible and usable for this candidate?
* Is an action which could obtain or refresh that evidence executable?

Those questions are intentionally kept separate.  A missing observation must
not permanently disable the query action that could obtain it.  The module has
no provider access, no model calls, and no numeric value-of-information or
outcome fields.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

from .source_reliability import (
    SourceManifestError,
    SourceObservation,
    evaluate_source_manifest,
)
from .types import utc


VERSION = "round135-provenance-gate-v1"


class ProvenanceGateError(ValueError):
    """Raised when an episode or candidate catalog is malformed."""


_FORBIDDEN_KEYS = frozenset({
    "outcome", "result", "score", "scores", "goals", "label", "labels",
    "gold", "sidecar", "evaluator", "probability", "probabilities", "voi",
    "expected_value", "realized_loss", "decision_loss", "target_label",
})
_VISIBLE_STATES = frozenset({"available", "duplicate"})
_ACTION_TRUE = frozenset({"available", "executable", "ready", "callable", "ok"})
_ACTION_FALSE = frozenset({"blocked", "failed", "missing", "unavailable", "disabled"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceGateError(f"{name}_REQUIRED")
    return value.strip()


def _time(value: Any, name: str) -> datetime:
    try:
        return utc(value)
    except (TypeError, ValueError) as exc:
        raise ProvenanceGateError(f"{name}_INVALID_TIME") from exc


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(body).hexdigest()


def _safe(value: Any) -> Any:
    """Project public candidate metadata while dropping private result keys."""
    if isinstance(value, Mapping):
        return {
            str(key): _safe(child)
            for key, child in value.items()
            if str(key).casefold() not in _FORBIDDEN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_safe(child) for child in value]
    return value


def _episode_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceGateError("episode_REQUIRED")
    if isinstance(value.get("episode"), Mapping):
        return value["episode"]
    return value


def _candidate_action(value: Mapping[str, Any]) -> str:
    """Return action readiness without treating missing evidence as disabled."""
    raw = None
    for key in ("action_availability", "retrieval_availability", "tool_availability"):
        if key in value:
            raw = value[key]
            break
    if raw is None and isinstance(value.get("action"), Mapping):
        raw = value["action"].get("availability")
    if raw is None and "can_execute" in value:
        raw = value["can_execute"]
    if isinstance(raw, bool):
        return "executable" if raw else "unavailable"
    if isinstance(raw, str):
        lowered = raw.strip().casefold()
        if lowered in _ACTION_TRUE:
            return "executable"
        if lowered in _ACTION_FALSE:
            return "unavailable"
        if lowered in {"unknown", "deferred", "pending"}:
            return "unknown"
    return "unknown"


def _claim_key(row: Mapping[str, Any]) -> str | None:
    """Find explicit same-claim metadata; raw digest differences alone do not."""
    for key in ("claim_conflict_group", "claim_key", "claim_id", "claim"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}:{value.strip()}"
    # Round134's conflict_group is explicitly documented as a provider claim
    # group, so it is safe to preserve as conflict metadata.
    value = row.get("conflict_group")
    return f"conflict_group:{value.strip()}" if isinstance(value, str) and value.strip() else None


def _source_row(raw: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    source_id = raw.get("source_id", raw.get("evidence_id"))
    source_id = _text(source_id, "source_id")
    digest = raw.get("digest", raw.get("content_digest", raw.get("content_hash")))
    if digest is not None and (not isinstance(digest, str) or not digest.strip()):
        digest = None
    freshness = raw.get("freshness")
    if freshness is None:
        quality = raw.get("quality")
        freshness = "stale" if quality in {"stale", "expired"} else "unknown"
    if freshness not in {"fresh", "stale", "unknown"}:
        freshness = "unknown"
    availability = raw.get("availability")
    if availability is None:
        quality = raw.get("quality")
        if quality in {"available", "observed", "fresh", "ok"}:
            availability = "available"
        elif quality in {"blocked", "failed", "missing"}:
            availability = quality
        else:
            availability = "unknown"
    if availability not in {"available", "missing", "blocked", "failed", "unknown", "late"}:
        availability = "unknown"
    row = {
        "source_id": source_id,
        "observed_at": raw.get("observed_at"),
        "available_at": raw.get("available_at", raw.get("operational_available_at", raw.get("ingested_at"))),
        "valid_until": raw.get("valid_until", raw.get("expires_at")),
        "freshness": freshness,
        "availability": availability,
        "digest": digest,
        "observation_id": raw.get("observation_id"),
    }
    claim = _claim_key(raw)
    if claim:
        row["conflict_group"] = claim
    return row, claim


def _is_future(row: Mapping[str, Any], cutoff: datetime) -> bool:
    for key in ("observed_at", "available_at", "operational_available_at", "ingested_at"):
        value = row.get(key)
        if value is None:
            continue
        try:
            if _time(value, key) > cutoff:
                return True
        except ProvenanceGateError:
            # Malformed clocks stay in the manifest as unknown; they are not
            # silently treated as evidence available before the cutoff.
            continue
    return False


def _canonical_asof_rows(raw_rows: Iterable[Mapping[str, Any]], cutoff: datetime) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Filter future rows before source_reliability sees them, then canonicalise revisions."""
    grouped: dict[str, list[tuple[dict[str, Any], str | None, int]]] = {}
    future: set[str] = set()
    unknown: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            continue
        try:
            row, claim = _source_row(raw)
        except ProvenanceGateError:
            sid = raw.get("source_id")
            if isinstance(sid, str) and sid.strip():
                unknown.add(sid.strip())
            continue
        # Keep malformed clocks as explicit unknown observations instead of
        # allowing one bad provider row to abort the complete gate.
        malformed_clock = False
        for key in ("observed_at", "available_at", "valid_until"):
            value = row.get(key)
            if value is None:
                continue
            try:
                _time(value, key)
            except ProvenanceGateError:
                row[key] = None
                malformed_clock = True
        if malformed_clock:
            row["freshness"] = "unknown"
            row["availability"] = "unknown"
            unknown.add(row["source_id"])
        observed_value = row.get("observed_at")
        available_value = row.get("available_at")
        if observed_value is not None and available_value is not None:
            try:
                if _time(available_value, "available_at") < _time(observed_value, "observed_at"):
                    row["freshness"] = "unknown"
                    row["availability"] = "unknown"
                    unknown.add(row["source_id"])
            except ProvenanceGateError:
                pass
        if _is_future(raw, cutoff):
            future.add(row["source_id"])
            continue
        # Preserve malformed/missing clocks as explicit unknown observations.
        grouped.setdefault(row["source_id"], []).append((row, claim, index))
    result: list[dict[str, Any]] = []
    for source_id in sorted(grouped):
        rows = grouped[source_id]
        distinct_digests = {row.get("digest") for row, _, _ in rows}
        claim_digests: dict[str, set[Any]] = {}
        for row, claim, _ in rows:
            if claim is not None:
                claim_digests.setdefault(claim, set()).add(row.get("digest"))
        # A raw digest change is a normal revision.  It becomes a conflict
        # only when the input explicitly says that the same claim was observed
        # with different content.
        has_explicit_conflict = any(len(digests) > 1 for digests in claim_digests.values())
        if has_explicit_conflict:
            chosen = rows
        elif len(distinct_digests) <= 1:
            # Identical observations remain visible so the contract can report
            # duplicate evidence rather than double-counting it.
            chosen = rows
        else:
            # A different digest with no same-claim marker is a normal source
            # revision.  Keep only the latest visible revision.
            def rank(item: tuple[dict[str, Any], str | None, int]) -> tuple[datetime, str, int]:
                row, _, index = item
                stamp = row.get("available_at") or row.get("observed_at")
                parsed = utc(stamp) if stamp is not None else datetime.min.replace(tzinfo=timezone.utc)
                return (parsed, str(row.get("observation_id") or ""), index)
            chosen = [max(rows, key=rank)]
        result.extend(row for row, _, _ in sorted(
            chosen,
            key=lambda item: (
                str(item[0].get("available_at") or ""),
                str(item[0].get("observed_at") or ""),
                str(item[0].get("digest") or ""),
                str(item[0].get("observation_id") or ""),
                item[2],
            ),
        ))
    return result, sorted(future), sorted(unknown)


def _candidate_rows(episode: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    if candidates is not None:
        rows = list(candidates)
    else:
        pre = episode.get("pre_action")
        ids = pre.get("candidate_ids", []) if isinstance(pre, Mapping) else []
        rows = [{"candidate_id": value, "source_ids": []} for value in ids if isinstance(value, str)]
    if not rows:
        raise ProvenanceGateError("candidate_catalog_REQUIRED")
    return rows


def build_provenance_gate(
    episode: Mapping[str, Any],
    *,
    candidates: Iterable[Mapping[str, Any]] | None = None,
    observations: Iterable[Mapping[str, Any]] | None = None,
    cutoff: str | None = None,
    aggregation_mode: str = "all_required",
) -> dict[str, Any]:
    """Build an actor-safe provenance state at one prediction cutoff."""
    public_episode = _episode_mapping(episode)
    cutoff_value = _time(cutoff or public_episode.get("prediction_cutoff"), "cutoff")
    if aggregation_mode not in {"all_required", "any_usable"}:
        raise ProvenanceGateError("aggregation_mode_UNSUPPORTED")
    if observations is None:
        pre = public_episode.get("pre_action")
        snapshot = pre.get("snapshot", {}) if isinstance(pre, Mapping) else {}
        observations = snapshot.get("source_observations", []) if isinstance(snapshot, Mapping) else []
    raw_rows = list(observations)
    asof_rows, future_ids, unknown_ids = _canonical_asof_rows(raw_rows, cutoff_value)
    try:
        manifest = evaluate_source_manifest(asof_rows, cutoff=cutoff_value.isoformat())
    except (SourceManifestError, TypeError, ValueError) as exc:
        raise ProvenanceGateError(f"manifest_INVALID:{exc}") from exc
    safe_sources: dict[str, Any] = {}
    for source_id, summary in manifest["sources"].items():
        safe_sources[source_id] = {
            "state": summary["state"],
            "action": summary["action"],
            "usable_at_cutoff": bool(summary["usable_at_cutoff"]),
            "duplicate": bool(summary["duplicate"]),
            "conflict": bool(summary["conflict"]),
            "observation_count": int(summary["observation_count"]),
            "reasons": sorted(str(item) for item in summary["reasons"]),
        }
    future_ids = sorted(set(future_ids))
    unknown_ids = sorted(set(unknown_ids))
    states: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for raw in _candidate_rows(public_episode, candidates):
        if not isinstance(raw, Mapping):
            raise ProvenanceGateError("candidate_OBJECT_REQUIRED")
        candidate_id = _text(raw.get("candidate_id"), "candidate_id")
        if candidate_id in seen_candidates:
            raise ProvenanceGateError(f"duplicate_candidate_id:{candidate_id}")
        seen_candidates.add(candidate_id)
        source_ids = raw.get("source_ids", [])
        if not isinstance(source_ids, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in source_ids):
            raise ProvenanceGateError(f"candidate_source_ids_INVALID:{candidate_id}")
        source_ids = list(dict.fromkeys(item.strip() for item in source_ids))
        source_states: dict[str, str] = {}
        reasons: set[str] = set()
        for source_id in source_ids:
            summary = safe_sources.get(source_id)
            if summary is None:
                # Future and absent evidence have the same actor-visible
                # state.  The future/late distinction belongs to audit only.
                source_states[source_id] = "unknown"
                reasons.add("SOURCE_NOT_VISIBLE_OR_UNKNOWN")
            else:
                source_states[source_id] = summary["state"]
                if summary["state"] == "stale":
                    reasons.add("SOURCE_STALE_REQUIRES_REFRESH")
                elif summary["state"] == "unknown":
                    reasons.add("SOURCE_NOT_VISIBLE_OR_UNKNOWN")
                elif summary["state"] == "conflict":
                    reasons.add("SOURCE_CONFLICT_REQUIRES_RESOLUTION")
                elif summary["state"] in {"missing", "blocked", "failed"}:
                    reasons.add("SOURCE_UNAVAILABLE")
                elif summary["state"] == "duplicate":
                    reasons.add("DUPLICATE_SOURCE_DEDUP_REQUIRED")
        usable = [state in _VISIBLE_STATES for state in source_states.values()]
        mode = raw.get("aggregation_mode", aggregation_mode)
        if mode not in {"all_required", "any_usable"}:
            raise ProvenanceGateError(f"aggregation_mode_UNSUPPORTED:{candidate_id}")
        if not source_ids:
            eligibility = "defer_unknown"
            reasons.add("NO_SOURCE_IDS")
        elif mode == "all_required":
            blocking = reasons - {"DUPLICATE_SOURCE_DEDUP_REQUIRED"}
            eligibility = "eligible" if all(usable) and not blocking else "defer_unknown"
            if eligibility == "eligible" and reasons:
                eligibility = "eligible_with_warnings"
        else:
            if any(usable):
                eligibility = "eligible_with_warnings" if reasons else "eligible"
            else:
                eligibility = "defer_unknown"
        action_eligibility = _candidate_action(raw)
        next_actions: set[str] = set()
        for state in source_states.values():
            if state == "stale":
                next_actions.add("refresh_source")
            elif state in {"unknown", "future"}:
                next_actions.add("query_source")
            elif state in {"missing", "blocked", "failed"}:
                next_actions.add("query_alternative_source")
            elif state == "conflict":
                next_actions.add("resolve_conflict")
        if action_eligibility == "unavailable":
            next_actions.add("query_alternative_source")
        # If no evidence is visible and action metadata is absent, keep the
        # action unknown rather than turning a missing manifest row into a ban.
        state = {
            "candidate_id": candidate_id,
            "source_ids": source_ids,
            "source_states": {key: source_states[key] for key in source_ids},
            "available_source_ids": [key for key in source_ids if source_states.get(key) in _VISIBLE_STATES],
            "unknown_source_ids": [key for key in source_ids if source_states.get(key) == "unknown"],
            "aggregation_mode": mode,
            "eligibility": eligibility,
            "action_eligibility": action_eligibility,
            "reasons": sorted(reasons),
            "next_actions": sorted(next_actions),
        }
        for key in ("kind", "requires", "declared_cost"):
            if key in raw:
                state[key] = _safe(raw[key])
        states.append(state)
    result = {
        "schema_version": VERSION,
        "cutoff": cutoff_value.isoformat(),
        "manifest_digest": manifest["report_digest"],
        "aggregation_mode": aggregation_mode,
        "candidate_order": [row["candidate_id"] for row in states],
        "candidates": states,
        "sources": safe_sources,
        "audit": {"future_source_ids": future_ids, "future_source_count": len(future_ids)},
        "unknown_source_ids": sorted(set(unknown_ids) | {sid for state in states for sid in state["unknown_source_ids"]}),
        "as_of_filter_applied": True,
        "runtime_safe": True,
    }
    # Digest only public, deterministic fields; do not hash or expose private
    # episode/evaluator material.
    digest_body = {key: value for key, value in result.items() if key != "audit"}
    result["gate_digest"] = _digest(digest_body)
    return result


evaluate_episode_provenance = build_provenance_gate
gate_historical_candidates = build_provenance_gate


def validate_provenance_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(report, Mapping) or report.get("schema_version") != VERSION:
        raise ProvenanceGateError("report_SCHEMA_VERSION_INVALID")
    if report.get("runtime_safe") is not True or report.get("as_of_filter_applied") is not True:
        raise ProvenanceGateError("report_RUNTIME_BOUNDARY_INVALID")
    expected = report.get("gate_digest")
    if not isinstance(expected, str):
        raise ProvenanceGateError("report_DIGEST_MISSING")
    body = {key: value for key, value in report.items() if key not in {"gate_digest", "audit"}}
    if expected != _digest(body):
        raise ProvenanceGateError("report_DIGEST_MISMATCH")
    def private_key(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).casefold() in _FORBIDDEN_KEYS or private_key(child):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(private_key(child) for child in value)
        return False
    if private_key(report):
        raise ProvenanceGateError("report_PRIVATE_FIELD_LEAK")
    return {"valid": True, "candidate_count": len(report.get("candidates", [])), "manifest_digest": report["manifest_digest"]}


__all__ = [
    "VERSION", "ProvenanceGateError", "build_provenance_gate",
    "evaluate_episode_provenance", "gate_historical_candidates",
    "validate_provenance_gate",
]
