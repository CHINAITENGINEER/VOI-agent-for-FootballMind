"""Append-only persistence for action-level prediction episodes.

This module stores the actor's *declared* state at each selection boundary.  It
is deliberately independent from result/evaluator data: a missing posterior is
stored as ``None`` and reported as ``PARTIAL`` rather than being filled with a
default distribution.  The journal is content addressed and chained so that a
later edit, a result leak, or a cross-action binding cannot be silently
accepted during replay.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from filelock import FileLock


VERSION = "episode-persistence-v1"
STATUS_COMPLETE = "COMPLETE"
STATUS_PARTIAL = "PARTIAL"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_EPS = 1e-8
_FORBIDDEN = frozenset({
    "result", "outcome", "score", "scores", "goals", "home_score",
    "away_score", "final_score", "label", "loss", "decision_loss",
    "realized_gain", "realized_loss", "future_result", "actual_result",
    "target_label", "sequence_losses", "action_returns", "evaluator", "sidecar",
    "post_state", "after_state", "hidden_result",
})


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str) -> str:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _prediction(value: Any, name: str) -> list[float] | None:
    # None is an explicit unknown, never a zero-filled prediction.
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty probability vector or None")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{name} must contain finite numbers")
        item = float(item)
        if item < 0:
            raise ValueError(f"{name} must contain nonnegative numbers")
        result.append(item)
    if abs(sum(result) - 1.0) > _EPS:
        raise ValueError(f"{name} must sum to one")
    return result


def _forbidden(value: Any, path: str = "record") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if name.casefold() in _FORBIDDEN:
                found.append(f"{path}.{name}")
            found.extend(_forbidden(child, f"{path}.{name}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_forbidden(child, f"{path}[{index}]"))
    return found


def _normalise_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("episode action must be a mapping")
    leaks = _forbidden(record)
    if leaks:
        raise ValueError("episode action contains evaluator fields: " + ",".join(leaks))
    row = deepcopy(dict(record))
    row.pop("record_digest", None)
    row.pop("envelope_digest", None)
    row["schema_version"] = row.get("schema_version", VERSION)
    if row["schema_version"] != VERSION:
        raise ValueError("unsupported episode persistence schema")
    # Keep unknown values explicitly unknown.  Do not synthesize clocks or
    # predictions from another field.
    if "prediction_before" in row:
        row["prediction_before"] = _prediction(row["prediction_before"], "prediction_before")
    else:
        row["prediction_before"] = None
    if "prediction_after" in row:
        row["prediction_after"] = _prediction(row["prediction_after"], "prediction_after")
    else:
        row["prediction_after"] = None
    if "prediction_before" not in row or "prediction_after" not in row:
        raise AssertionError("prediction fields are explicitly represented")
    # ``state_digest`` is the compact name used by the runtime ledger.  Keep
    # it as an alias while storing the before/after boundary explicitly.
    if row.get("state_before_digest") is None and row.get("state_digest") is not None:
        row["state_before_digest"] = row["state_digest"]
    if row.get("state_digest") is None and row.get("state_before_digest") is not None:
        row["state_digest"] = row["state_before_digest"]
    if "prediction_cutoff" in row and row["prediction_cutoff"] is not None:
        row["prediction_cutoff"] = _time(row["prediction_cutoff"], "prediction_cutoff")
    for key in ("captured_at", "selected_at"):
        if key in row and row[key] is not None:
            row[key] = _time(row[key], key)
    # Accept either the compact top-level binding names or nested bindings,
    # but persist both in a deterministic form for easy downstream joins.
    selection = row.get("selection_binding") if isinstance(row.get("selection_binding"), Mapping) else {}
    if row.get("selection_operation_id") is not None:
        if selection.get("operation_id") is not None and selection.get("operation_id") != row["selection_operation_id"]:
            raise ValueError("selection binding identity conflict")
        selection = {**selection, "operation_id": row["selection_operation_id"]}
    if row.get("selection_digest") is not None:
        if selection.get("digest") is not None and selection.get("digest") != row["selection_digest"]:
            raise ValueError("selection binding digest conflict")
        selection = {**selection, "digest": row["selection_digest"]}
    if selection:
        row["selection_binding"] = deepcopy(selection)
        row["selection_operation_id"] = selection.get("operation_id")
        row["selection_digest"] = selection.get("digest")
    acquisition = row.get("acquisition_binding") if isinstance(row.get("acquisition_binding"), Mapping) else {}
    if row.get("acquisition_id") is not None:
        if acquisition.get("acquisition_id") is not None and acquisition.get("acquisition_id") != row["acquisition_id"]:
            raise ValueError("acquisition binding identity conflict")
        acquisition = {**acquisition, "acquisition_id": row["acquisition_id"]}
    if row.get("acquisition_digest") is not None:
        if acquisition.get("digest") is not None and acquisition.get("digest") != row["acquisition_digest"]:
            raise ValueError("acquisition binding digest conflict")
        acquisition = {**acquisition, "digest": row["acquisition_digest"]}
    if acquisition:
        row["acquisition_binding"] = deepcopy(acquisition)
        row["acquisition_id"] = acquisition.get("acquisition_id")
        row["acquisition_digest"] = acquisition.get("digest")
    admission = row.get("admission_binding") if isinstance(row.get("admission_binding"), Mapping) else {}
    if row.get("admission_status") is not None:
        if admission.get("status") is not None and admission.get("status") != row["admission_status"]:
            raise ValueError("admission binding status conflict")
        admission = {**admission, "status": row["admission_status"]}
    if row.get("admission_digest") is not None:
        if admission.get("digest") is not None and admission.get("digest") != row["admission_digest"]:
            raise ValueError("admission binding digest conflict")
        admission = {**admission, "digest": row["admission_digest"]}
    if admission:
        row["admission_binding"] = deepcopy(admission)
        row["admission_status"] = admission.get("status")
        row["admission_digest"] = admission.get("digest")
    return row


def _record_digest(record: Mapping[str, Any]) -> str:
    # The digest is over the immutable payload, never over its own field.
    payload = deepcopy(dict(record))
    payload.pop("record_digest", None)
    return _digest(payload)


def _reason(report: dict[str, Any], reason: str) -> None:
    report.setdefault("reasons", []).append(reason)


def _binding(report: dict[str, Any], row: Mapping[str, Any], prefix: str,
             required: bool) -> None:
    value = row.get(prefix + "_digest")
    if value is not None and not _valid_digest(value):
        _reason(report, prefix + "_digest_invalid")
    if required and not _valid_digest(value):
        _reason(report, prefix + "_binding_missing")


def validate_episode_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate immutable records without introducing missing values.

    The report is ``PARTIAL`` whenever required scientific evidence is absent;
    callers may persist and inspect such episodes.  Structural tampering raises
    from the journal reader before this function is called.
    """
    rows = [row for row in records if isinstance(row, Mapping)]
    report: dict[str, Any] = {"schema_version": VERSION, "status": STATUS_COMPLETE,
                              "eligible_for_replay": True, "numeric_voi_training_eligible": False,
                              "reasons": [], "record_count": len(rows)}
    seen_selection: dict[str, tuple[Any, Any]] = {}
    seen_acquisition: dict[str, tuple[Any, Any]] = {}
    previous_after: list[float] | None = None
    for position, row in enumerate(rows):
        if _forbidden(row):
            _reason(report, f"record_{position}_evaluator_fields_present")
        if row.get("schema_version") != VERSION:
            _reason(report, f"record_{position}_schema_version_invalid")
        if row.get("sequence") != position:
            _reason(report, f"record_{position}_sequence_not_contiguous")
        for field in ("episode_id", "target_key", "action_id", "candidate_id", "state_before_digest"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                _reason(report, f"record_{position}_{field}_missing")
        if row.get("prediction_cutoff") is None:
            _reason(report, f"record_{position}_prediction_cutoff_missing")
        else:
            try:
                _time(row["prediction_cutoff"], f"record[{position}].prediction_cutoff")
            except ValueError:
                _reason(report, f"record_{position}_prediction_cutoff_invalid")
        if not _valid_digest(row.get("state_before_digest")):
            _reason(report, f"record_{position}_state_before_digest_invalid")
        if row.get("state_digest") is not None:
            if not _valid_digest(row.get("state_digest")):
                _reason(report, f"record_{position}_state_digest_invalid")
            elif row.get("state_digest") != row.get("state_before_digest"):
                _reason(report, f"record_{position}_state_digest_mismatch")
        if row.get("state_after_digest") is not None and not _valid_digest(row["state_after_digest"]):
            _reason(report, f"record_{position}_state_after_digest_invalid")
        if row.get("prediction_before") is None:
            _reason(report, f"record_{position}_prediction_before_unknown")
        if row.get("prediction_after") is None:
            _reason(report, f"record_{position}_prediction_after_unknown")
        before, after = row.get("prediction_before"), row.get("prediction_after")
        if previous_after is not None and before is not None and before != previous_after:
            _reason(report, f"record_{position}_prediction_continuity_mismatch")
        if after is not None:
            previous_after = deepcopy(after)
        selection_id = row.get("selection_operation_id")
        if not isinstance(selection_id, str) or not selection_id.strip():
            _reason(report, f"record_{position}_selection_binding_missing")
        else:
            if selection_id in seen_selection:
                _reason(report, f"record_{position}_selection_binding_duplicate")
            seen_selection[selection_id] = (row.get("candidate_id"), row.get("action_id"))
        _binding(report, row, "selection", required=True)
        acquisition_id = row.get("acquisition_id")
        is_terminal = str(row.get("action_id", "")).upper() in {"STOP", "DEFER", "USER_INPUT_REQUIRED"}
        if acquisition_id is not None:
            if not isinstance(acquisition_id, str) or not acquisition_id.strip():
                _reason(report, f"record_{position}_acquisition_id_invalid")
            elif acquisition_id in seen_acquisition:
                previous = seen_acquisition[acquisition_id]
                current = (row.get("selection_operation_id"), row.get("candidate_id"))
                if previous != current:
                    _reason(report, f"record_{position}_cross_action_acquisition_binding")
                else:
                    _reason(report, f"record_{position}_acquisition_binding_duplicate")
                seen_acquisition[acquisition_id] = current
            else:
                seen_acquisition[acquisition_id] = (row.get("selection_operation_id"), row.get("candidate_id"))
            _binding(report, row, "acquisition", required=True)
            if row.get("admission_status") is None:
                _reason(report, f"record_{position}_admission_binding_missing")
            elif row.get("admission_status") not in {"admitted", "rejected", "pending", "unknown", "legacy_unspecified"}:
                _reason(report, f"record_{position}_admission_status_invalid")
            _binding(report, row, "admission", required=row.get("admission_status") not in {None, "unknown"})
        elif not is_terminal:
            _reason(report, f"record_{position}_acquisition_binding_missing")
        if row.get("future_result") is not None:
            _reason(report, f"record_{position}_future_result_present")
    if report["reasons"]:
        report["status"] = STATUS_PARTIAL
        report["eligible_for_replay"] = False
    report["reasons"] = sorted(set(report["reasons"]))
    return report


class EpisodePersistence:
    """Thread/process-safe append-only action episode journal."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.path = self.root / "episode-actions.jsonl"
        self.lock_path = self.root / "episode-actions.lock"

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        previous = None
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"EPISODE_PERSISTENCE_INVALID_JSON:{line_number}") from exc
            if not isinstance(envelope, Mapping) or envelope.get("schema_version") != VERSION:
                raise ValueError(f"EPISODE_PERSISTENCE_SCHEMA_MISMATCH:{line_number}")
            row = envelope.get("record")
            if not isinstance(row, Mapping):
                raise ValueError(f"EPISODE_PERSISTENCE_RECORD_MISSING:{line_number}")
            row = deepcopy(dict(row))
            expected = _record_digest(row)
            if envelope.get("record_digest") != expected:
                raise ValueError(f"EPISODE_PERSISTENCE_RECORD_TAMPERED:{line_number}")
            if envelope.get("previous_digest") != previous:
                raise ValueError(f"EPISODE_PERSISTENCE_CHAIN_TAMPERED:{line_number}")
            expected_envelope = {"schema_version": VERSION, "record": row,
                                 "record_digest": expected, "previous_digest": previous}
            if envelope.get("envelope_digest") != _digest(expected_envelope):
                raise ValueError(f"EPISODE_PERSISTENCE_ENVELOPE_TAMPERED:{line_number}")
            rows.append(row)
            previous = expected
        return rows

    def records(self) -> tuple[dict[str, Any], ...]:
        with FileLock(str(self.lock_path)):
            return tuple(self._read_unlocked())

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        row = _normalise_record(record)
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            existing = self._read_unlocked()
            sequence = row.get("sequence")
            if sequence is None:
                row["sequence"] = len(existing)
            elif type(sequence) is not int or sequence != len(existing):
                raise ValueError("EPISODE_PERSISTENCE_SEQUENCE_NOT_APPEND_ONLY")
            if existing:
                head = existing[0]
                for key in ("episode_id", "target_key", "prediction_cutoff"):
                    if row.get(key) is not None and head.get(key) is not None and row[key] != head[key]:
                        raise ValueError(f"EPISODE_PERSISTENCE_{key.upper()}_MISMATCH")
            identity = (row.get("selection_operation_id"), row.get("acquisition_id"))
            for old in existing:
                if row.get("selection_operation_id") is not None and old.get("selection_operation_id") == row["selection_operation_id"]:
                    raise ValueError("EPISODE_PERSISTENCE_SELECTION_IMMUTABLE")
                if row.get("acquisition_id") is not None and old.get("acquisition_id") == row["acquisition_id"]:
                    raise ValueError("EPISODE_PERSISTENCE_ACQUISITION_IMMUTABLE")
            row["record_digest"] = _record_digest(row)
            previous = existing[-1].get("record_digest") if existing else None
            envelope = {"schema_version": VERSION, "record": row,
                        "record_digest": row["record_digest"], "previous_digest": previous}
            envelope["envelope_digest"] = _digest(envelope)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
        return deepcopy(row)

    def validate(self) -> dict[str, Any]:
        return validate_episode_records(self.records())

    def snapshot(self) -> dict[str, Any]:
        rows = list(self.records())
        report = validate_episode_records(rows)
        body = {"schema_version": VERSION, "records": rows, "report": report}
        return {**body, "snapshot_digest": _digest(body)}


ActionEpisodeStore = EpisodePersistence


__all__ = ["VERSION", "STATUS_COMPLETE", "STATUS_PARTIAL", "EpisodePersistence",
           "ActionEpisodeStore", "validate_episode_records"]
