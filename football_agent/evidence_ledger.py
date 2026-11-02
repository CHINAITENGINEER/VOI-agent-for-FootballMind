"""Append-only evidence ledger for platform acquisition records.

The ledger preserves one immutable record per acquisition attempt. A later
refresh receives a new entry; earlier timing, quality and dependency metadata
remain available for point-in-time replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from filelock import FileLock

from .audit import json_bytes
from .evidence_binding import AVAILABILITY, QUALITIES
from .platform_evidence_adapter import acquisition_result_to_evidence_rows
from .types import utc


VERSION = "evidence-ledger-v1"
RESOLUTION_STATUSES = frozenset({"UNRESOLVED", "RESOLVED", "CONTESTED"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str, *, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    return utc(value).isoformat()


@dataclass(frozen=True)
class EvidenceLedgerEntry:
    entry_id: str
    episode_id: str
    target_key: str
    candidate_id: str
    action_id: str
    request_id: str
    sequence: int
    source_id: str
    dependency_key: str
    quality: str
    availability: str
    observed_at: str | None
    operational_available_at: str | None
    prediction_cutoff: str
    captured_at: str
    evidence_refs: tuple[str, ...] = ()
    content_hash: str = ""
    resolution_status: str = "UNRESOLVED"
    requested_at: str | None = None
    source_version: str = ""

    def __post_init__(self) -> None:
        for field in ("entry_id", "episode_id", "target_key", "candidate_id", "action_id",
                      "request_id", "source_id", "dependency_key"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("sequence must be a nonnegative integer")
        if self.quality not in QUALITIES:
            raise ValueError(f"Unsupported ledger quality: {self.quality}")
        if self.availability not in AVAILABILITY:
            raise ValueError(f"Unsupported ledger availability: {self.availability}")
        if self.resolution_status not in RESOLUTION_STATUSES:
            raise ValueError("Unsupported ledger resolution status")
        observed = _time(self.observed_at, "observed_at", required=False)
        available = _time(self.operational_available_at, "operational_available_at", required=False)
        cutoff = _time(self.prediction_cutoff, "prediction_cutoff")
        captured = _time(self.captured_at, "captured_at")
        requested = _time(self.requested_at, "requested_at", required=False)
        if observed and available and utc(observed) > utc(available):
            raise ValueError("observed_at cannot follow operational_available_at")
        if self.availability == "available" and (observed is None or available is None):
            raise ValueError("available evidence needs observed and operational timestamps")
        if self.availability == "available" and utc(available) > utc(cutoff):
            raise ValueError("available evidence must precede prediction cutoff")
        if self.availability == "late" and (observed is None or available is None):
            raise ValueError("late evidence needs observed and operational timestamps")
        if requested and utc(requested) > utc(captured):
            raise ValueError("requested_at cannot follow captured_at")
        if not isinstance(self.source_version, str):
            raise ValueError("source_version must be a string")
        refs = tuple(_text(ref, "evidence_ref") for ref in self.evidence_refs)
        if isinstance(self.content_hash, bool) or not isinstance(self.content_hash, str):
            raise ValueError("content_hash must be a string")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "operational_available_at", available)
        object.__setattr__(self, "prediction_cutoff", cutoff)
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "requested_at", requested)
        object.__setattr__(self, "source_version", self.source_version.strip())
        object.__setattr__(self, "evidence_refs", refs)

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        *,
        episode_id: str,
        target_key: str,
        action_id: str,
        request_id: str,
        sequence: int,
        prediction_cutoff: str,
        captured_at: str | None = None,
        resolution_status: str = "UNRESOLVED",
        requested_at: str | None = None,
        source_version: str = "",
    ) -> "EvidenceLedgerEntry":
        if not isinstance(row, Mapping):
            raise ValueError("Evidence row must be a mapping")
        body = {
            "episode_id": episode_id,
            "target_key": target_key,
            "candidate_id": row.get("candidate_id"),
            "action_id": action_id,
            "request_id": request_id,
            "sequence": sequence,
            "source_id": row.get("source_id"),
            "dependency_key": row.get("dependency_key"),
            "quality": row.get("quality", "UNVERIFIED"),
            "availability": row.get("availability", "failed"),
            "observed_at": row.get("observed_at"),
            "operational_available_at": row.get("operational_available_at", row.get("available_at")),
            "prediction_cutoff": prediction_cutoff,
            "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
            "requested_at": requested_at,
            "source_version": source_version,
            "evidence_refs": tuple(row.get("evidence_refs", ()) or ()),
            "content_hash": str(row.get("content_hash", "")),
            "resolution_status": resolution_status,
        }
        identity = {key: value for key, value in body.items() if key != "captured_at"}
        body["entry_id"] = hashlib.sha256(json_bytes(identity)).hexdigest()
        return cls(**body)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "entry_id": self.entry_id,
            "episode_id": self.episode_id,
            "target_key": self.target_key,
            "candidate_id": self.candidate_id,
            "action_id": self.action_id,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "source_id": self.source_id,
            "dependency_key": self.dependency_key,
            "quality": self.quality,
            "availability": self.availability,
            "observed_at": self.observed_at,
            "operational_available_at": self.operational_available_at,
            "prediction_cutoff": self.prediction_cutoff,
            "captured_at": self.captured_at,
            "requested_at": self.requested_at,
            "source_version": self.source_version,
            "evidence_refs": list(self.evidence_refs),
            "content_hash": self.content_hash,
            "resolution_status": self.resolution_status,
        }


class EvidenceLedger:
    """Append-only JSONL evidence ledger with per-record checksums."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.path = self.root / "evidence-ledger.jsonl"
        self.lock_path = self.root / "evidence-ledger.lock"

    def _read_unlocked(self) -> list[EvidenceLedgerEntry]:
        if not self.path.exists():
            return []
        entries = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            envelope = json.loads(line)
            record = envelope.get("record")
            if envelope.get("schema_version") != VERSION or not isinstance(record, Mapping):
                raise ValueError(f"Invalid evidence ledger envelope at line {line_number}")
            expected = hashlib.sha256(json_bytes(record)).hexdigest()
            if envelope.get("record_sha256") != expected:
                raise ValueError(f"Evidence ledger checksum mismatch at line {line_number}")
            entry = EvidenceLedgerEntry(**{key: value for key, value in record.items() if key != "schema_version"})
            if entry.entry_id != record["entry_id"]:
                raise ValueError("Evidence ledger identity mismatch")
            entries.append(entry)
        return entries

    def entries(self) -> tuple[EvidenceLedgerEntry, ...]:
        with FileLock(str(self.lock_path)):
            return tuple(self._read_unlocked())

    def append(self, entries: Iterable[EvidenceLedgerEntry]) -> tuple[EvidenceLedgerEntry, ...]:
        incoming = tuple(entries)
        if not incoming:
            raise ValueError("At least one evidence ledger entry is required")
        if len({entry.entry_id for entry in incoming}) != len(incoming):
            raise ValueError("Evidence ledger entry IDs must be unique")
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            existing = self._read_unlocked()
            existing_ids = {entry.entry_id for entry in existing}
            if any(entry.entry_id in existing_ids for entry in incoming):
                raise ValueError("Evidence ledger is append-only; entry identity already exists")
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                for entry in incoming:
                    record = entry.as_dict()
                    envelope = {
                        "schema_version": VERSION,
                        "record": record,
                        "record_sha256": hashlib.sha256(json_bytes(record)).hexdigest(),
                    }
                    stream.write(json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return incoming

    def append_acquisition_result(
        self,
        result: Mapping[str, Any] | Any,
        *,
        action_to_candidate: Mapping[str, str],
        target_key: str,
        prediction_cutoff: str,
        captured_at: str | None = None,
        resolution_status: str = "UNRESOLVED",
        requested_at: str | None = None,
        source_version: str = "",
    ) -> tuple[EvidenceLedgerEntry, ...]:
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)
        rows = acquisition_result_to_evidence_rows(result, action_to_candidate=action_to_candidate)
        entries = tuple(EvidenceLedgerEntry.from_row(
            row, episode_id=payload.get("episode_id"), target_key=target_key,
            action_id=payload.get("action_id"), request_id=payload.get("request_id"),
            sequence=payload.get("sequence", 0), prediction_cutoff=prediction_cutoff,
            captured_at=captured_at, resolution_status=resolution_status,
            requested_at=requested_at, source_version=source_version,
        ) for row in rows)
        return self.append(entries)

    def snapshot_rows(
        self,
        *,
        target_key: str,
        prediction_cutoff: str,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the latest append-only record per candidate for a target."""
        allowed = None if candidate_ids is None else set(candidate_ids)
        cutoff = utc(prediction_cutoff)
        latest: dict[str, EvidenceLedgerEntry] = {}
        for entry in self.entries():
            if entry.target_key != target_key:
                continue
            if allowed is not None and entry.candidate_id not in allowed:
                continue
            # A point-in-time replay may only see an acquisition attempt that
            # had already been captured by the decision cutoff. A late result
            # remains visible when the attempt predates the cutoff, while a
            # post-cutoff attempt stays outside the decision state entirely.
            if utc(entry.captured_at) > cutoff:
                continue
            previous = latest.get(entry.candidate_id)
            if previous is None or (utc(entry.captured_at), entry.sequence) > (utc(previous.captured_at), previous.sequence):
                latest[entry.candidate_id] = entry
        return [self._row_from_entry(latest[candidate_id]) for candidate_id in sorted(latest)]

    @staticmethod
    def _row_from_entry(entry: EvidenceLedgerEntry) -> dict[str, Any]:
        return {
            "candidate_id": entry.candidate_id,
            "source_id": entry.source_id,
            "dependency_key": entry.dependency_key,
            "quality": entry.quality,
            "availability": entry.availability,
            "observed_at": entry.observed_at,
            "operational_available_at": entry.operational_available_at,
            "evidence_refs": list(entry.evidence_refs),
            "content_hash": entry.content_hash,
            "resolution_status": entry.resolution_status,
            "requested_at": entry.requested_at,
            "source_version": entry.source_version,
        }


__all__ = ["VERSION", "RESOLUTION_STATUSES", "EvidenceLedgerEntry", "EvidenceLedger"]
