"""Auditable local execution envelopes for Agent information queries.

``QueryValueGate`` selects a candidate, while ``ContextTransition`` records
the resulting context.  This module is the narrow seam between those two
layers.  It hashes the request and public result, records timing/quality
states, and turns a completed envelope into either a context update or a
replanning transition.  No network client lives here; callers provide a
fixture result from a local tool adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from collections.abc import Iterable, Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from types import MappingProxyType

from filelock import FileLock

from .audit import json_bytes
from .context_query_readiness import ContextState
from .context_transition import ContextTransition
from .contextual_adaptive_voi import normalize_context
from .query_value_contract import QUERY, QueryValueDecision
from .state_query_contract import StateQueryBoard
from .types import utc


VERSION = "query-execution-envelope-v1"
PLANNED = "PLANNED"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
LATE = "LATE"
AMBIGUOUS = "AMBIGUOUS"
STATUSES = frozenset({PLANNED, SUCCEEDED, FAILED, LATE, AMBIGUOUS})
FINAL_STATUSES = frozenset({SUCCEEDED, FAILED, LATE, AMBIGUOUS})

_HIDDEN_KEYS = frozenset({
    "result", "outcome", "score", "home_score", "away_score", "half_score",
    "final_score", "hidden_result", "probability", "sampled_result", "realized_loss",
})
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _time(value: Any, name: str, *, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    return utc(value).isoformat()


def _cost(value: Any, name: str = "cost") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _public(value: Any, path: str = "payload") -> Any:
    """Validate and copy a local tool payload without result-like fields."""
    if isinstance(value, Mapping):
        output = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"{path}.key must be a string")
            key = _text(raw_key, f"{path}.key")
            if key.lower() in _HIDDEN_KEYS:
                raise ValueError(f"{path} contains a hidden result field")
            output[key] = _public(raw_value, f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [_public(item, f"{path}[]") for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    raise ValueError(f"{path} contains a non-serializable value")


def _keys(value: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of keys")
    result = tuple(_text(item, name) for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique nonempty keys")
    if any(item.lower() in _HIDDEN_KEYS for item in result):
        raise ValueError(f"{name} contains a hidden result key")
    return result


def _refs(value: Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("evidence_refs must be a sequence")
    result = tuple(_text(item, "evidence_ref") for item in value)
    if len(set(result)) != len(result):
        raise ValueError("evidence_refs must be unique")
    return result


def _identity_hash(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(json_bytes(body)).hexdigest()


def _envelope_identity(
    *,
    query_id: str,
    status: str,
    request_hash: str,
    result_hash: str | None,
    context_updates: Mapping[str, Any],
    evidence_refs: Iterable[str],
) -> str:
    """Return the content identity used by both creation and replay.

    The ledger checksum protects a line from accidental corruption, but it is
    not enough to prove that the ``envelope_id`` still describes that line.
    Keeping this calculation in one place lets deserialisation reject a
    record whose fields were changed together with its checksum.
    """
    if status == PLANNED:
        body = {"query_id": query_id, "status": status, "request_hash": request_hash}
    else:
        body = {
            "query_id": query_id, "status": status, "request_hash": request_hash,
            "result_hash": result_hash, "context_updates": dict(context_updates),
            "evidence_refs": list(evidence_refs),
        }
    return _identity_hash(body)


@dataclass(frozen=True)
class QueryExecutionEnvelope:
    """One immutable planning or execution state for a selected query."""

    envelope_id: str
    query_id: str
    target_key: str
    candidate_id: str
    prediction_cutoff: str
    expected_context_keys: tuple[str, ...]
    cost: float
    selected_by: str
    tool_version: str
    request_hash: str
    result_hash: str | None
    status: str
    requested_at: str
    observed_at: str | None
    operational_available_at: str | None
    captured_at: str
    evidence_refs: tuple[str, ...] = ()
    context_updates: Mapping[str, Any] | None = None
    error_code: str | None = None
    replanning_required: bool = False
    hidden_result_included: bool = False

    def __post_init__(self) -> None:
        for field in ("envelope_id", "query_id", "target_key", "candidate_id", "selected_by", "tool_version"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "prediction_cutoff", _time(self.prediction_cutoff, "prediction_cutoff"))
        object.__setattr__(self, "expected_context_keys", _keys(self.expected_context_keys, "expected_context_keys"))
        object.__setattr__(self, "cost", _cost(self.cost))
        object.__setattr__(self, "request_hash", _hash(self.request_hash, "request_hash"))
        if self.result_hash is not None:
            object.__setattr__(self, "result_hash", _hash(self.result_hash, "result_hash"))
        if self.status not in STATUSES:
            raise ValueError("Unsupported query execution status")
        requested = _time(self.requested_at, "requested_at")
        observed = _time(self.observed_at, "observed_at", required=False)
        available = _time(self.operational_available_at, "operational_available_at", required=False)
        captured = _time(self.captured_at, "captured_at")
        if observed and available and utc(observed) > utc(available):
            raise ValueError("observed_at cannot follow operational_available_at")
        if observed and utc(observed) > utc(captured):
            raise ValueError("observed_at cannot follow captured_at")
        if available and utc(available) > utc(captured):
            raise ValueError("operational_available_at cannot follow captured_at")
        if utc(requested) > utc(captured):
            raise ValueError("requested_at cannot follow captured_at")
        if utc(requested) > utc(self.prediction_cutoff):
            raise ValueError("requested_at cannot follow prediction cutoff")
        object.__setattr__(self, "requested_at", requested)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "operational_available_at", available)
        object.__setattr__(self, "captured_at", captured)
        updates = {} if self.context_updates is None else _public(self.context_updates, "context_updates")
        if not isinstance(updates, Mapping):
            raise ValueError("context_updates must be a mapping")
        # ContextTransition supports scalar public state.  Reject a nested
        # value here, before a SUCCEEDED entry can be persisted but not replayed.
        updates = normalize_context(updates) if updates else {}
        unknown = set(updates) - set(self.expected_context_keys)
        if unknown:
            raise ValueError("context_updates contains keys outside expected_context_keys")
        # A frozen dataclass does not freeze a nested dict.  Expose a read-only
        # mapping so a caller cannot alter the identity after construction or
        # append a record whose later replay has different semantics.
        object.__setattr__(self, "context_updates", MappingProxyType(updates))
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))
        object.__setattr__(self, "error_code", _optional_text(self.error_code, "error_code"))
        if not isinstance(self.replanning_required, bool):
            raise ValueError("replanning_required must be boolean")
        if self.hidden_result_included is not False:
            raise ValueError("query envelopes must exclude hidden results")
        if self.status == PLANNED:
            if self.result_hash is not None or updates or observed or available or self.replanning_required or self.error_code:
                raise ValueError("planned envelope cannot contain execution result fields")
            if captured != requested:
                raise ValueError("planned capture must equal requested_at")
        else:
            if not self.evidence_refs:
                raise ValueError("completed envelope needs evidence references")
            if self.result_hash is None:
                raise ValueError("completed envelope needs a result hash")
            if self.status == SUCCEEDED:
                if self.result_hash is None or not updates or self.replanning_required:
                    raise ValueError("successful envelope needs a result and context updates")
                if self.error_code is not None:
                    raise ValueError("successful envelope cannot contain an error code")
                if observed is None or available is None:
                    raise ValueError("successful envelope needs observed and operational timestamps")
                if available is not None and utc(available) > utc(self.prediction_cutoff):
                    raise ValueError("successful result arrived after prediction cutoff")
                if utc(captured) > utc(self.prediction_cutoff):
                    raise ValueError("successful result was captured after prediction cutoff")
            else:
                if not self.replanning_required:
                    raise ValueError("failed, late, or ambiguous result must trigger replanning")
                if self.status == LATE and (
                    (available is None or utc(available) <= utc(self.prediction_cutoff))
                    and utc(captured) <= utc(self.prediction_cutoff)
                ):
                    raise ValueError("late result needs an operational or capture timestamp after cutoff")
                if self.status == LATE and observed is None:
                    raise ValueError("late envelope needs an observed timestamp")
                if updates:
                    raise ValueError("non-successful result cannot update context")
        expected_identity = _envelope_identity(
            query_id=self.query_id, status=self.status, request_hash=self.request_hash,
            result_hash=self.result_hash, context_updates=updates,
            evidence_refs=self.evidence_refs,
        )
        if self.envelope_id != expected_identity:
            raise ValueError("query execution envelope identity mismatch")

    @property
    def success(self) -> bool:
        return self.status == SUCCEEDED

    @property
    def late(self) -> bool:
        return self.status == LATE

    @property
    def ambiguous(self) -> bool:
        return self.status == AMBIGUOUS

    @classmethod
    def from_decision(
        cls,
        decision: QueryValueDecision,
        board: StateQueryBoard,
        request_payload: Mapping[str, Any],
        *,
        tool_version: str,
        expected_context_keys: Iterable[str] | None = None,
        query_id: str | None = None,
        requested_at: str | None = None,
        evidence_refs: Iterable[str] = (),
    ) -> "QueryExecutionEnvelope":
        if not isinstance(decision, QueryValueDecision) or decision.action != QUERY or not decision.candidate_id:
            raise ValueError("an execution envelope requires a QUERY decision")
        if not isinstance(board, StateQueryBoard):
            raise ValueError("board must be a StateQueryBoard")
        candidate = next((item for item in board.candidates if item.candidate_id == decision.candidate_id), None)
        if candidate is None or candidate not in board.actionable_candidates():
            raise ValueError("selected candidate is not currently actionable")
        if not isinstance(request_payload, Mapping):
            raise ValueError("request_payload must be a mapping")
        request = _public(request_payload, "request_payload")
        request_hash = _identity_hash(request)
        requested = _time(requested_at or board.prediction_cutoff, "requested_at")
        keys = _keys(expected_context_keys or (candidate.state_variable,), "expected_context_keys")
        body = {
            "target_key": board.target_key, "candidate_id": candidate.candidate_id,
            "prediction_cutoff": board.prediction_cutoff, "expected_context_keys": list(keys),
            "request_hash": request_hash, "tool_version": _text(tool_version, "tool_version"),
        }
        qid = _text(query_id, "query_id") if query_id is not None else "query-" + _identity_hash(body)
        envelope_id = _identity_hash({"query_id": qid, "status": PLANNED, "request_hash": request_hash})
        assessment = next((item for item in decision.assessments if item.candidate_id == candidate.candidate_id), None)
        cost = candidate.cost if assessment is None or assessment.cost is None else assessment.cost
        return cls(
            envelope_id=envelope_id, query_id=qid, target_key=board.target_key,
            candidate_id=candidate.candidate_id, prediction_cutoff=board.prediction_cutoff,
            expected_context_keys=keys, cost=cost, selected_by=decision.selection_basis,
            tool_version=_text(tool_version, "tool_version"), request_hash=request_hash,
            result_hash=None, status=PLANNED, requested_at=requested,
            observed_at=None, operational_available_at=None, captured_at=requested,
            evidence_refs=evidence_refs, context_updates={},
        )

    def complete(
        self,
        result_payload: Mapping[str, Any],
        *,
        context_updates: Mapping[str, Any] | None = None,
        evidence_refs: Iterable[str] = (),
        observed_at: str | None = None,
        operational_available_at: str | None = None,
        captured_at: str | None = None,
        status: str | None = None,
        error_code: str | None = None,
    ) -> "QueryExecutionEnvelope":
        """Complete using an explicit capture supplied by the receiving boundary.

        Source observation and request clocks cannot stand in for capture,
        including for failed executions.  A caller recording a local failure
        must timestamp that failure at its own recording boundary.
        """
        if self.status != PLANNED:
            raise ValueError("only a planned envelope can be completed")
        if not isinstance(result_payload, Mapping):
            raise ValueError("result_payload must be a mapping")
        result = _public(result_payload, "result_payload")
        updates = {} if context_updates is None else _public(context_updates, "context_updates")
        if not isinstance(updates, Mapping):
            raise ValueError("context_updates must be a mapping")
        result_hash = _identity_hash(result)
        observed = _time(observed_at, "observed_at", required=False)
        available = _time(
            observed if operational_available_at is None else operational_available_at,
            "operational_available_at", required=False,
        )
        # Source observation is not execution capture.  The receiving adapter
        # (or historical record) must supply capture explicitly; this object
        # must never invent a provider completion time from observation/now.
        captured = _time(captured_at, "captured_at")
        late_clock = (
            (available is not None and utc(available) > utc(self.prediction_cutoff))
            or utc(captured) > utc(self.prediction_cutoff)
        )
        if status is None:
            if error_code:
                status = FAILED
            elif late_clock:
                status = LATE
            elif not updates:
                status = AMBIGUOUS
            else:
                status = SUCCEEDED
        if status not in FINAL_STATUSES:
            raise ValueError("completion status must be a final execution status")
        if status == SUCCEEDED and error_code is not None:
            raise ValueError("successful completion cannot contain an error code")
        if status == SUCCEEDED and late_clock:
            raise ValueError("successful result arrived or was captured after prediction cutoff")
        if status == LATE and available is None:
            raise ValueError("late completion needs an operational timestamp")
        if status != SUCCEEDED:
            updates = {}
        refs = tuple(dict.fromkeys((*self.evidence_refs, *_refs(evidence_refs), f"query-result:{result_hash}")))
        envelope_id = _envelope_identity(
            query_id=self.query_id, status=status, request_hash=self.request_hash,
            result_hash=result_hash, context_updates=updates, evidence_refs=refs,
        )
        return QueryExecutionEnvelope(
            envelope_id=envelope_id, query_id=self.query_id, target_key=self.target_key,
            candidate_id=self.candidate_id, prediction_cutoff=self.prediction_cutoff,
            expected_context_keys=self.expected_context_keys, cost=self.cost,
            selected_by=self.selected_by, tool_version=self.tool_version,
            request_hash=self.request_hash, result_hash=result_hash, status=status,
            requested_at=self.requested_at, observed_at=observed,
            operational_available_at=available, captured_at=captured,
            evidence_refs=refs, context_updates=dict(updates), error_code=error_code,
            replanning_required=status != SUCCEEDED,
        )

    def to_context_transition(
        self,
        *,
        episode_id: str,
        sequence: int,
        previous_context: Mapping[str, Any] | ContextState,
        captured_at: str | None = None,
    ) -> ContextTransition:
        if self.status == PLANNED:
            raise ValueError("planned envelope cannot create a context transition")
        captured = _time(self.captured_at if captured_at is None else captured_at, "captured_at")
        if utc(captured) < utc(self.captured_at):
            raise ValueError("transition capture cannot precede execution capture")
        if self.status == SUCCEEDED and utc(captured) > utc(self.prediction_cutoff):
            raise ValueError("successful transition was captured after prediction cutoff")
        previous = previous_context.values if isinstance(previous_context, ContextState) else dict(previous_context)
        if self.status == SUCCEEDED:
            following = {**previous, **dict(self.context_updates or {})}
            event_type, trigger, reason = "CONTEXT_UPDATED", "QUERY_RESULT", "query_result_applied"
        else:
            following = dict(previous)
            event_type = "REPLANNED"
            trigger = f"QUERY_RESULT_{self.status}"
            reason = f"query_{self.status.lower()}_requires_replanning"
        refs = tuple(dict.fromkeys((*self.evidence_refs, f"query:{self.query_id}")))
        return ContextTransition.from_transition(
            episode_id=episode_id, target_key=self.target_key, sequence=sequence,
            event_type=event_type, previous_context=previous, next_context=following,
            trigger=trigger, reason=reason, prediction_cutoff=self.prediction_cutoff,
            captured_at=captured,
            selected_candidate_id=self.candidate_id, action=QUERY,
            source_id=self.tool_version, evidence_refs=refs,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "envelope_id": self.envelope_id, "query_id": self.query_id,
            "target_key": self.target_key, "candidate_id": self.candidate_id,
            "prediction_cutoff": self.prediction_cutoff,
            "expected_context_keys": list(self.expected_context_keys), "cost": self.cost,
            "selected_by": self.selected_by, "tool_version": self.tool_version,
            "request_hash": self.request_hash, "result_hash": self.result_hash,
            "status": self.status, "requested_at": self.requested_at,
            "observed_at": self.observed_at,
            "operational_available_at": self.operational_available_at,
            "captured_at": self.captured_at, "evidence_refs": list(self.evidence_refs),
            "context_updates": dict(self.context_updates or {}), "error_code": self.error_code,
            "replanning_required": self.replanning_required,
            "hidden_result_included": False,
        }


class QueryExecutionLedger:
    """Append-only local journal for planned and completed envelopes."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.path = self.root / "query-executions.jsonl"
        self.lock_path = self.root / "query-executions.lock"

    def _read_unlocked(self) -> list[QueryExecutionEnvelope]:
        if not self.path.exists():
            return []
        entries = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            envelope = json.loads(line)
            record = envelope.get("record")
            if envelope.get("schema_version") != VERSION or not isinstance(record, Mapping):
                raise ValueError(f"Invalid query execution envelope at line {line_number}")
            if record.get("schema_version") != VERSION:
                raise ValueError(f"Invalid query execution record schema at line {line_number}")
            expected = hashlib.sha256(json_bytes(record)).hexdigest()
            if envelope.get("record_sha256") != expected:
                raise ValueError(f"Query execution checksum mismatch at line {line_number}")
            if record.get("hidden_result_included") is not False:
                raise ValueError("Query execution ledger hidden-result flag is invalid")
            entry = QueryExecutionEnvelope(**{key: value for key, value in record.items()
                                              if key not in {"schema_version", "hidden_result_included"}})
            if entry.envelope_id != record["envelope_id"]:
                raise ValueError("Query execution envelope identity mismatch")
            entries.append(entry)
        self._validate_lifecycle(entries)
        return entries

    @staticmethod
    def _validate_lifecycle(entries: Iterable[QueryExecutionEnvelope]) -> None:
        """One durable intent and at most one terminal receipt per query."""
        planned: dict[str, QueryExecutionEnvelope] = {}
        finished: set[str] = set()
        identities: set[str] = set()
        bound_fields = (
            "target_key", "candidate_id", "prediction_cutoff", "expected_context_keys",
            "cost", "selected_by", "tool_version", "request_hash", "requested_at",
        )
        for entry in entries:
            if entry.envelope_id in identities:
                raise ValueError("query execution ledger contains duplicate envelope identity")
            identities.add(entry.envelope_id)
            if entry.status == PLANNED:
                if entry.query_id in planned:
                    raise ValueError("query execution already has a planned intent")
                planned[entry.query_id] = entry
                continue
            original = planned.get(entry.query_id)
            if original is None:
                raise ValueError("query execution completion needs a preceding planned intent")
            if entry.query_id in finished:
                raise ValueError("query execution already has a terminal receipt")
            if any(getattr(original, name) != getattr(entry, name) for name in bound_fields):
                raise ValueError("query execution completion differs from its planned intent")
            if not set(original.evidence_refs).issubset(entry.evidence_refs):
                raise ValueError("query execution completion lost planned evidence references")
            finished.add(entry.query_id)

    def entries(self) -> tuple[QueryExecutionEnvelope, ...]:
        with FileLock(str(self.lock_path)):
            return tuple(self._read_unlocked())

    def append(self, envelopes: Iterable[QueryExecutionEnvelope]) -> tuple[QueryExecutionEnvelope, ...]:
        incoming = tuple(envelopes)
        if not incoming:
            raise ValueError("At least one query execution envelope is required")
        if any(not isinstance(item, QueryExecutionEnvelope) for item in incoming):
            raise ValueError("query execution ledger accepts execution envelopes only")
        # Re-run dataclass validation before any bytes are written.  Callers
        # must not be able to persist mutated or half-valid frozen instances.
        incoming = tuple(replace(item) for item in incoming)
        if len({item.envelope_id for item in incoming}) != len(incoming):
            raise ValueError("Query execution envelope IDs must be unique")
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            existing = self._read_unlocked()
            existing_ids = {item.envelope_id for item in existing}
            if any(item.envelope_id in existing_ids for item in incoming):
                raise ValueError("Query execution ledger is append-only; envelope identity already exists")
            self._validate_lifecycle((*existing, *incoming))
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                for item in incoming:
                    record = item.as_dict()
                    envelope = {
                        "schema_version": VERSION, "record": record,
                        "record_sha256": hashlib.sha256(json_bytes(record)).hexdigest(),
                    }
                    stream.write(json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return incoming

    def for_query(self, query_id: str) -> tuple[QueryExecutionEnvelope, ...]:
        query_id = _text(query_id, "query_id")
        return tuple(item for item in self.entries() if item.query_id == query_id)


__all__ = [
    "VERSION", "PLANNED", "SUCCEEDED", "FAILED", "LATE", "AMBIGUOUS", "STATUSES",
    "QueryExecutionEnvelope", "QueryExecutionLedger",
]
