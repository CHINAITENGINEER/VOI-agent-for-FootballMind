"""Project already received collector metadata into observable channel signals.

A successful request establishes transport success, not factual accuracy or
content freshness. This adapter exposes observations for later calibration,
never supplies the synthetic health labels expected by source_health_voi.
Only records visible at the policy cutoff are projected; future response
metadata is omitted entirely, including error status, identity and hashes.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from .types import utc


VERSION = "source-health-observations-v1"
_CLOCKS = ("requested_at", "observed_at", "operational_available_at", "captured_at")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def source_health_observations(
    events: Iterable[Mapping[str, Any]], *, as_of: str | datetime,
) -> dict[str, Any]:
    """Return time-filtered signals from ``collector_ledger_adapter`` events.

    The caller supplies normalized archive metadata. Request latency is
    unknown if request start was never recorded. ``receipt_age_seconds``
    measures age of the local receipt, not age of the source's content.
    Duplicate request identities count once. Identical response hashes are
    kept as a shared-content cluster, without classifying the source as stale.
    No truth label, likelihood estimate, or acquisition policy is fitted here.
    """
    cutoff = utc(as_of)
    visible = []
    for raw in events:
        if not isinstance(raw, Mapping):
            raise ValueError("events must contain mappings")
        times = {key: utc(raw[key]) if raw.get(key) is not None else None for key in _CLOCKS}
        # Failure events may have no usable-content timestamp, but still need
        # a recorded return/receipt and archive clock before their status is known.
        if times["captured_at"] is None or times["observed_at"] is None:
            continue
        if any(value is not None and value > cutoff for value in times.values()):
            continue
        if times["requested_at"] is not None and times["requested_at"] > times["observed_at"]:
            raise ValueError("request start follows response observation")
        if times["observed_at"] > times["captured_at"]:
            raise ValueError("archive capture precedes response observation")
        if times["operational_available_at"] is not None and times["operational_available_at"] < times["observed_at"]:
            raise ValueError("operational availability precedes response observation")

        status = raw.get("http_status")
        transport = ("ERROR" if raw.get("error_class") or status is not None and (
            type(status) is not int or not 200 <= status < 300
        ) else "SUCCESS" if status is not None else "UNKNOWN")
        published = utc(raw["source_published_at"]) if raw.get("source_published_at") is not None else None
        # A future date in a received document is a temporal inconsistency,
        # not proof of freshness and not a future record entering the policy.
        publication_state = ("UNKNOWN" if published is None else
                             "FUTURE_TIMESTAMP" if published > times["observed_at"] else "DECLARED_TIMESTAMP")
        source_id = _text(raw.get("source_id"), "source_id")
        source_version = raw.get("source_version") or "UNVERSIONED"
        source_version = _text(source_version, "source_version")
        content_hash = raw.get("content_hash")
        if content_hash is not None and not isinstance(content_hash, str):
            raise ValueError("content_hash must be a string when present")
        signal = {
            "source_id": source_id,
            "source_version": source_version,
            "endpoint": _text(raw.get("endpoint"), "endpoint"),
            "request_id": _text(raw.get("request_id"), "request_id"),
            "dependency_key": _text(raw.get("dependency_key"), "dependency_key"),
            "observed_at": times["observed_at"].isoformat(),
            "captured_at": times["captured_at"].isoformat(),
            "requested_at": None if times["requested_at"] is None else times["requested_at"].isoformat(),
            "operational_available_at": None if times["operational_available_at"] is None else times["operational_available_at"].isoformat(),
            "transport_state": transport,
            "quality": _text(raw.get("quality", "UNVERIFIED"), "quality"),
            "entity_mapping_state": _text(raw.get("entity_mapping_state", "not_evaluated"), "entity_mapping_state"),
            "receipt_age_seconds": (cutoff - times["observed_at"]).total_seconds(),
            "latency_seconds": None if times["requested_at"] is None else (times["observed_at"] - times["requested_at"]).total_seconds(),
            "content_hash": content_hash or None,
            "content_timestamp_state": publication_state,
            "source_published_at": None if published is None else published.isoformat(),
            "declared_content_age_seconds": (cutoff - published).total_seconds() if publication_state == "DECLARED_TIMESTAMP" else None,
            "source_health_label": None,
            "semantic_accuracy": None,
            "acquisition_value": None,
        }
        visible.append(signal)

    visible.sort(key=lambda row: (row["captured_at"], row["source_id"], row["source_version"], row["request_id"]))
    by_request = {}
    for row in visible:
        key = (row["source_id"], row["source_version"], row["request_id"])
        if key in by_request and by_request[key] != row:
            raise ValueError("one request identity has conflicting health metadata")
        by_request[key] = row
    rows = list(by_request.values())
    groups = {}
    for row in rows:
        if row["content_hash"] and row["transport_state"] == "SUCCESS":
            key = (row["source_id"], row["source_version"], row["endpoint"], row["content_hash"])
            groups.setdefault(key, []).append(row["request_id"])
    repeated = [{"source_id": key[0], "source_version": key[1], "endpoint": key[2],
                 "content_hash": key[3], "request_ids": ids}
                for key, ids in sorted(groups.items()) if len(ids) > 1]
    return {
        "schema_version": VERSION,
        "as_of": cutoff.isoformat(),
        "scope": "observable_metadata_only",
        "observations": rows,
        "visible_request_count": len(rows),
        "repeated_content_groups": repeated,
        "health_model_support": "UNMEASURED",
        "likelihood_fitted": False,
        "numeric_voi_supported": False,
        "limitations": [
            "Transport success and recent receipt do not label content as correct or fresh.",
            "Equal content hashes may represent unchanged facts; repetition alone is not staleness.",
            "Future records and their status/identity are excluded from this policy-facing projection.",
            "These observations alone do not identify P(content | match state, source health).",
        ],
    }


__all__ = ["VERSION", "source_health_observations"]
