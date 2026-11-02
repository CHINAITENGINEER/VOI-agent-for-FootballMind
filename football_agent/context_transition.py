"""Append-only, hash-chained context transitions for Agent replay."""
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
from .contextual_adaptive_voi import normalize_context
from .context_query_readiness import ContextState
from .types import utc


VERSION = "context-transition-ledger-v1"
EVENT_TYPES = frozenset({"INITIALIZED", "QUERY_SELECTED", "CONTEXT_UPDATED", "REPLANNED", "STOPPED"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _context(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if isinstance(value, Mapping) and not value:
        return {}
    try:
        return normalize_context(value)
    except ValueError as exc:
        raise ValueError(f"{name} is invalid: {exc}") from exc


def _time(value: Any, name: str) -> str:
    return utc(value).isoformat()


@dataclass(frozen=True)
class ContextTransition:
    event_id: str
    episode_id: str
    target_key: str
    sequence: int
    event_type: str
    previous_context: Mapping[str, Any]
    next_context: Mapping[str, Any]
    changed_keys: tuple[str, ...]
    trigger: str
    reason: str
    prediction_cutoff: str
    captured_at: str
    selected_candidate_id: str | None = None
    action: str | None = None
    source_id: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id"))
        object.__setattr__(self, "episode_id", _text(self.episode_id, "episode_id"))
        object.__setattr__(self, "target_key", _text(self.target_key, "target_key"))
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        if self.event_type not in EVENT_TYPES:
            raise ValueError("Unsupported context transition event type")
        previous = _context(self.previous_context, "previous_context")
        following = _context(self.next_context, "next_context")
        object.__setattr__(self, "previous_context", previous)
        object.__setattr__(self, "next_context", following)
        changed = tuple(sorted(set(previous) | set(following)))
        changed = tuple(key for key in changed
                        if key not in previous or key not in following or previous[key] != following[key])
        declared = tuple(sorted(set(self.changed_keys)))
        if declared != changed:
            raise ValueError("changed_keys must equal the context diff")
        object.__setattr__(self, "changed_keys", declared)
        object.__setattr__(self, "trigger", _text(self.trigger, "trigger"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "prediction_cutoff", _time(self.prediction_cutoff, "prediction_cutoff"))
        object.__setattr__(self, "captured_at", _time(self.captured_at, "captured_at"))
        object.__setattr__(self, "selected_candidate_id", _optional_text(self.selected_candidate_id, "selected_candidate_id"))
        object.__setattr__(self, "action", _optional_text(self.action, "action"))
        object.__setattr__(self, "source_id", _optional_text(self.source_id, "source_id"))
        refs = tuple(_text(ref, "evidence_ref") for ref in self.evidence_refs)
        object.__setattr__(self, "evidence_refs", refs)
        if self.event_type == "INITIALIZED" and self.sequence != 1:
            raise ValueError("An initialized context must be the first sequence event")
        if self.event_type == "QUERY_SELECTED" and not self.selected_candidate_id:
            raise ValueError("A query-selected event needs a candidate ID")
        if self.event_type == "QUERY_SELECTED" and self.action != "QUERY":
            raise ValueError("A query-selected event needs QUERY action")

    @classmethod
    def from_transition(
        cls,
        *,
        episode_id: str,
        target_key: str,
        sequence: int,
        event_type: str,
        previous_context: Mapping[str, Any],
        next_context: Mapping[str, Any],
        trigger: str,
        reason: str,
        prediction_cutoff: str,
        captured_at: str | None = None,
        selected_candidate_id: str | None = None,
        action: str | None = None,
        source_id: str | None = None,
        evidence_refs: Iterable[str] = (),
    ) -> "ContextTransition":
        previous = _context(previous_context, "previous_context")
        following = _context(next_context, "next_context")
        changed = tuple(sorted(key for key in set(previous) | set(following)
                              if key not in previous or key not in following or previous[key] != following[key]))
        captured = captured_at or datetime.now(timezone.utc).isoformat()
        identity = {
            "episode_id": episode_id, "target_key": target_key, "sequence": sequence,
            "event_type": event_type, "previous_context": previous, "next_context": following,
            "changed_keys": list(changed), "trigger": trigger, "reason": reason,
            "prediction_cutoff": _time(prediction_cutoff, "prediction_cutoff"),
            "selected_candidate_id": selected_candidate_id, "action": action,
            "source_id": source_id, "evidence_refs": list(evidence_refs),
        }
        event_id = hashlib.sha256(json_bytes(identity)).hexdigest()
        return cls(
            event_id=event_id, episode_id=episode_id, target_key=target_key, sequence=sequence,
            event_type=event_type, previous_context=previous, next_context=following,
            changed_keys=changed, trigger=trigger, reason=reason,
            prediction_cutoff=prediction_cutoff, captured_at=captured,
            selected_candidate_id=selected_candidate_id, action=action,
            source_id=source_id, evidence_refs=tuple(evidence_refs),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "event_id": self.event_id, "episode_id": self.episode_id,
            "target_key": self.target_key, "sequence": self.sequence,
            "event_type": self.event_type,
            "previous_context": dict(self.previous_context), "next_context": dict(self.next_context),
            "changed_keys": list(self.changed_keys), "trigger": self.trigger, "reason": self.reason,
            "prediction_cutoff": self.prediction_cutoff, "captured_at": self.captured_at,
            "selected_candidate_id": self.selected_candidate_id, "action": self.action,
            "source_id": self.source_id, "evidence_refs": list(self.evidence_refs),
            "hidden_result_included": False,
        }


class ContextTransitionLedger:
    """Append-only JSONL ledger with a hash chain and context replay."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.path = self.root / "context-transitions.jsonl"
        self.lock_path = self.root / "context-transitions.lock"

    def _read_unlocked(self) -> list[ContextTransition]:
        if not self.path.exists():
            return []
        entries = []
        previous_hash = None
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            envelope = json.loads(line)
            record = envelope.get("record")
            if envelope.get("schema_version") != VERSION or not isinstance(record, Mapping):
                raise ValueError(f"Invalid context transition envelope at line {line_number}")
            if envelope.get("previous_hash") != previous_hash:
                raise ValueError(f"Context transition chain predecessor mismatch at line {line_number}")
            expected_record = hashlib.sha256(json_bytes(record)).hexdigest()
            if envelope.get("record_sha256") != expected_record:
                raise ValueError(f"Context transition record checksum mismatch at line {line_number}")
            chain_body = {"previous_hash": previous_hash, "record": record}
            expected_chain = hashlib.sha256(json_bytes(chain_body)).hexdigest()
            if envelope.get("chain_hash") != expected_chain:
                raise ValueError(f"Context transition chain checksum mismatch at line {line_number}")
            entry = ContextTransition(**{key: value for key, value in record.items()
                                        if key not in {"schema_version", "hidden_result_included"}})
            if record.get("hidden_result_included") is not False or entry.event_id != record["event_id"]:
                raise ValueError("Context transition hidden-result or identity flag is invalid")
            entries.append(entry)
            previous_hash = expected_chain
        return entries

    def entries(self) -> tuple[ContextTransition, ...]:
        with FileLock(str(self.lock_path)):
            return tuple(self._read_unlocked())

    def append(self, entries: Iterable[ContextTransition]) -> tuple[ContextTransition, ...]:
        incoming = tuple(entries)
        if not incoming:
            raise ValueError("At least one context transition is required")
        if len({entry.event_id for entry in incoming}) != len(incoming):
            raise ValueError("Context transition IDs must be unique")
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            existing = self._read_unlocked()
            existing_ids = {entry.event_id for entry in existing}
            if any(entry.event_id in existing_ids for entry in incoming):
                raise ValueError("Context transition ledger is append-only; event identity already exists")
            previous_hash = None
            if self.path.exists():
                lines = self.path.read_text(encoding="utf-8").splitlines()
                if lines:
                    previous_hash = json.loads(lines[-1])["chain_hash"]
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                for entry in incoming:
                    record = entry.as_dict()
                    chain_body = {"previous_hash": previous_hash, "record": record}
                    chain_hash = hashlib.sha256(json_bytes(chain_body)).hexdigest()
                    envelope = {
                        "schema_version": VERSION, "previous_hash": previous_hash,
                        "record": record,
                        "record_sha256": hashlib.sha256(json_bytes(record)).hexdigest(),
                        "chain_hash": chain_hash,
                    }
                    stream.write(json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                    previous_hash = chain_hash
                stream.flush()
                os.fsync(stream.fileno())
        return incoming

    def replay_episode(self, episode_id: str, *, target_key: str | None = None) -> dict[str, Any]:
        episode_id = _text(episode_id, "episode_id")
        selected = [entry for entry in self.entries()
                    if entry.episode_id == episode_id and (target_key is None or entry.target_key == target_key)]
        selected.sort(key=lambda entry: entry.sequence)
        if not selected:
            raise ValueError("No context transitions found for episode")
        previous_context: Mapping[str, Any] = {}
        for expected_sequence, entry in enumerate(selected, 1):
            if entry.sequence != expected_sequence:
                raise ValueError("Context transition sequence has a gap")
            if dict(entry.previous_context) != dict(previous_context):
                raise ValueError("Context transition previous snapshot does not match replay state")
            previous_context = entry.next_context
        return {
            "schema_version": VERSION,
            "episode_id": episode_id,
            "target_key": selected[-1].target_key,
            "event_count": len(selected),
            "events": [entry.as_dict() for entry in selected],
            "final_context": dict(previous_context),
            "context_state": ContextState(previous_context).as_dict(),
            "hidden_result_included": False,
        }

    def verify(self) -> dict[str, Any]:
        entries = self.entries()
        return {
            "schema_version": VERSION,
            "status": "OK",
            "events_verified": len(entries),
            "head_event_id": entries[-1].event_id if entries else None,
        }


__all__ = ["VERSION", "EVENT_TYPES", "ContextTransition", "ContextTransitionLedger"]
