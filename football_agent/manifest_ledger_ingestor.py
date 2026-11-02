"""Rebind collector-run events to a decision episode and cutoff.

Pipeline manifests are created at collector-run close, which is a useful
arrival audit but is not a prediction cutoff.  This module is the explicit
boundary where an Agent episode supplies its own cutoff before events become
Evidence Ledger entries.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .collector_ledger_adapter import events_to_ledger_entries
from .types import utc


VERSION = "manifest-ledger-ingestor-v1"
_FAILED_QUALITIES = frozenset({"PROVIDER_ERROR", "SOURCE_MISSING", "INVALID", "STALE", "CONFLICT"})


def rebind_event_cutoff(event: Mapping[str, Any], *, prediction_cutoff: str) -> dict[str, Any]:
    """Recompute point-in-time availability while preserving source metadata."""
    row = dict(event)
    quality = str(row.get("quality", "UNVERIFIED")).upper()
    observed_at = row.get("observed_at")
    original_availability = str(row.get("availability", "")).lower()
    arrival_at = row.get("operational_available_at") or observed_at
    if original_availability == "duplicate":
        availability, operational = "duplicate", arrival_at
    elif quality in _FAILED_QUALITIES or original_availability == "failed" or arrival_at is None:
        availability, operational = "failed", None
    else:
        arrival = utc(arrival_at)
        cutoff = utc(prediction_cutoff)
        availability = "available" if arrival <= cutoff else "late"
        operational = arrival.isoformat()
    row["availability"] = availability
    row["operational_available_at"] = operational
    row["prediction_cutoff"] = utc(prediction_cutoff).isoformat()
    row["ingestor_version"] = VERSION
    return row


def manifest_events_to_ledger_entries(
    manifest_entry: Mapping[str, Any],
    *,
    episode_id: str,
    target_key: str,
    action_id: str,
    prediction_cutoff: str,
    action_request_id: str | None = None,
    candidate_id: str | None = None,
    resolution_status: str = "UNRESOLVED",
    sequence_start: int = 0,
):
    """Turn one pipeline source entry into cutoff-correct Ledger entries."""
    if not isinstance(manifest_entry, Mapping):
        raise ValueError("manifest_entry must be a mapping")
    if manifest_entry.get("collector_events_cutoff_kind") != "collector_run_close_not_prediction_cutoff":
        raise ValueError("Manifest collector event cutoff semantics are not explicit")
    raw_events = manifest_entry.get("collector_events", ())
    if not isinstance(raw_events, Iterable) or isinstance(raw_events, (str, bytes, Mapping)):
        raise ValueError("collector_events must be a sequence")
    events = [rebind_event_cutoff(event, prediction_cutoff=prediction_cutoff) for event in raw_events]
    if action_request_id is not None:
        if not isinstance(action_request_id, str) or not action_request_id.strip():
            raise ValueError("action_request_id must be a nonempty string")
        if len(events) != 1:
            raise ValueError("Action binding requires exactly one collector event; select or aggregate explicitly")
        for event in events:
            source_request_id = event.get("request_id")
            event["evidence_refs"] = list(dict.fromkeys(
                [*event.get("evidence_refs", []), f"source-request:{source_request_id}"]))
            event["request_id"] = action_request_id
            if candidate_id is not None:
                event["candidate_id"] = candidate_id
    events.sort(key=lambda event: (
        event.get("requested_at") or event.get("observed_at") or "",
        event.get("request_id", ""), event.get("observation_id", ""),
    ))
    return events_to_ledger_entries(
        events, episode_id=episode_id, target_key=target_key, action_id=action_id,
        prediction_cutoff=prediction_cutoff, resolution_status=resolution_status,
        sequence_start=sequence_start,
    )


def append_manifest_events(ledger, manifest_entry: Mapping[str, Any], **kwargs):
    """Append cutoff-rebound manifest events through the immutable Ledger."""
    return ledger.append(manifest_events_to_ledger_entries(manifest_entry, **kwargs))


__all__ = ["VERSION", "rebind_event_cutoff", "manifest_events_to_ledger_entries", "append_manifest_events"]
