"""Auditable pre-action value-of-information traces.

The trace stores what the policy estimated before an acquisition and what
observation arrived afterwards. It deliberately omits evaluator-only hidden
states and realized match outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from filelock import FileLock

from .audit import json_bytes
from .types import utc


VERSION = "voi-trace-v1"
_TOLERANCE = 1e-8


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _optional_number(value: Any, name: str, *, minimum: float | None = None) -> float | None:
    return None if value is None else _number(value, name, minimum=minimum)


def _probabilities(values: Iterable[Any], name: str) -> tuple[float, ...]:
    result = tuple(_number(value, f"{name} item", minimum=0.0) for value in values)
    if not result or abs(sum(result) - 1.0) > _TOLERANCE:
        raise ValueError(f"{name} must be a nonempty normalized probability vector")
    return result


@dataclass(frozen=True)
class VoITrace:
    trace_id: str
    episode_id: str
    target_key: str
    step: int
    action_id: str
    candidate_id: str
    request_id: str
    selection_basis: str
    model_version: str
    expected_information_value: float | None
    expected_cost: float
    expected_net_value: float | None
    risk_before: float
    expected_risk_after: float | None
    observed_risk_after: float | None
    belief_before: tuple[float, ...]
    prediction_before: tuple[float, ...]
    belief_after: tuple[float, ...] | None
    prediction_after: tuple[float, ...] | None
    observed_outcome: str | None
    stop_reason: str | None
    captured_at: str

    def __post_init__(self) -> None:
        for field in ("trace_id", "episode_id", "target_key", "action_id", "candidate_id",
                      "request_id", "selection_basis", "model_version"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        object.__setattr__(self, "expected_information_value",
                           _optional_number(self.expected_information_value, "expected_information_value", minimum=0.0))
        object.__setattr__(self, "expected_cost", _number(self.expected_cost, "expected_cost", minimum=0.0))
        object.__setattr__(self, "expected_net_value",
                           _optional_number(self.expected_net_value, "expected_net_value"))
        object.__setattr__(self, "risk_before", _number(self.risk_before, "risk_before", minimum=0.0))
        object.__setattr__(self, "expected_risk_after",
                           _optional_number(self.expected_risk_after, "expected_risk_after", minimum=0.0))
        object.__setattr__(self, "observed_risk_after",
                           _optional_number(self.observed_risk_after, "observed_risk_after", minimum=0.0))
        object.__setattr__(self, "belief_before", _probabilities(self.belief_before, "belief_before"))
        object.__setattr__(self, "prediction_before", _probabilities(self.prediction_before, "prediction_before"))
        if self.belief_after is not None:
            object.__setattr__(self, "belief_after", _probabilities(self.belief_after, "belief_after"))
        if self.prediction_after is not None:
            object.__setattr__(self, "prediction_after", _probabilities(self.prediction_after, "prediction_after"))
        if self.observed_outcome is not None:
            object.__setattr__(self, "observed_outcome", _text(self.observed_outcome, "observed_outcome"))
        if self.stop_reason is not None:
            object.__setattr__(self, "stop_reason", _text(self.stop_reason, "stop_reason"))
        object.__setattr__(self, "captured_at", utc(self.captured_at).isoformat())
        if self.action_id == "STOP":
            if self.candidate_id != "STOP" or self.stop_reason is None:
                raise ValueError("STOP traces require candidate_id STOP and a stop_reason")
            if self.expected_information_value is not None or self.expected_risk_after is not None:
                raise ValueError("STOP traces do not claim an information value or post-query risk")
            if self.belief_after is not None or self.prediction_after is not None or self.observed_outcome is not None:
                raise ValueError("STOP traces cannot contain an observation")
        elif self.stop_reason is not None:
            raise ValueError("Acquisition traces cannot contain a stop_reason")
        if self.action_id != "STOP" and self.observed_outcome is None:
            raise ValueError("Acquisition traces require the observed outcome")

    @classmethod
    def from_episode_event(
        cls,
        event: Mapping[str, Any],
        *,
        episode_id: str,
        target_key: str,
        request_id: str,
        model_version: str,
        captured_at: str | None = None,
    ) -> "VoITrace":
        if not isinstance(event, Mapping) or not isinstance(event.get("decision"), Mapping):
            raise ValueError("Episode event must contain a decision mapping")
        decision = event["decision"]
        action_id = decision.get("action_id")
        if not isinstance(action_id, str):
            raise ValueError("Decision action_id is required")
        belief = event.get("belief_before")
        if not isinstance(belief, Mapping):
            raise ValueError("Episode event belief_before is required")
        prediction = decision.get("prediction")
        if not isinstance(prediction, (list, tuple)):
            raise ValueError("Decision prediction is required")
        step = event.get("step")
        captured = captured_at or datetime.now(timezone.utc).isoformat()
        if action_id == "STOP":
            return cls(
                trace_id=_trace_id(episode_id, target_key, request_id, step, action_id),
                episode_id=episode_id, target_key=target_key, step=step,
                action_id="STOP", candidate_id="STOP", request_id=request_id,
                selection_basis=f"stop:{decision.get('reason', 'unspecified')}",
                model_version=model_version, expected_information_value=None,
                expected_cost=0.0, expected_net_value=0.0,
                risk_before=decision.get("current_risk"), expected_risk_after=None, observed_risk_after=None,
                belief_before=tuple(belief.values()), prediction_before=tuple(prediction),
                belief_after=None, prediction_after=None, observed_outcome=None,
                stop_reason=decision.get("reason", "unspecified"), captured_at=captured,
            )
        candidates = decision.get("candidates", [])
        selected = next((item for item in candidates if item.get("action_id") == action_id), None)
        if not isinstance(selected, Mapping) or not selected.get("feasible", False):
            raise ValueError("Chosen action must have a feasible candidate evaluation")
        belief_after = event.get("belief_after")
        prediction_after = event.get("prediction_after")
        if not isinstance(belief_after, Mapping) or not isinstance(prediction_after, (list, tuple)):
            raise ValueError("Acquisition event must contain posterior belief and prediction")
        expected_risk = selected.get("expected_risk_after_observation")
        return cls(
            trace_id=_trace_id(episode_id, target_key, request_id, step, action_id),
            episode_id=episode_id, target_key=target_key, step=step,
            action_id=action_id, candidate_id=action_id, request_id=request_id,
            selection_basis=decision.get("reason", "policy_selection"), model_version=model_version,
            expected_information_value=selected.get("immediate_evsi"),
            expected_cost=selected.get("cost", event.get("cost", 0.0)),
            expected_net_value=selected.get("planning_net_value", selected.get("immediate_net_value")),
            risk_before=decision.get("current_risk"), expected_risk_after=expected_risk,
            observed_risk_after=event.get("observed_risk_after"),
            belief_before=tuple(belief.values()), prediction_before=tuple(prediction),
            belief_after=tuple(belief_after.values()), prediction_after=tuple(prediction_after),
            observed_outcome=event.get("observation"), stop_reason=None, captured_at=captured,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION, "trace_id": self.trace_id,
            "episode_id": self.episode_id, "target_key": self.target_key,
            "step": self.step, "action_id": self.action_id, "candidate_id": self.candidate_id,
            "request_id": self.request_id, "selection_basis": self.selection_basis,
            "model_version": self.model_version,
            "expected_information_value": self.expected_information_value,
            "expected_cost": self.expected_cost, "expected_net_value": self.expected_net_value,
            "risk_before": self.risk_before, "expected_risk_after": self.expected_risk_after,
            "observed_risk_after": self.observed_risk_after,
            "belief_before": list(self.belief_before), "prediction_before": list(self.prediction_before),
            "belief_after": None if self.belief_after is None else list(self.belief_after),
            "prediction_after": None if self.prediction_after is None else list(self.prediction_after),
            "observed_outcome": self.observed_outcome, "stop_reason": self.stop_reason,
            "captured_at": self.captured_at, "hidden_result_included": False,
        }


def _trace_id(episode_id: str, target_key: str, request_id: str, step: int, action_id: str) -> str:
    return hashlib.sha256(json_bytes({"episode_id": episode_id, "target_key": target_key,
                                      "request_id": request_id, "step": step, "action_id": action_id})).hexdigest()


class VoITraceLedger:
    """Append-only, checksummed policy-selection trace store."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.path = self.root / "voi-trace.jsonl"
        self.lock_path = self.root / "voi-trace.lock"

    def _read_unlocked(self) -> list[VoITrace]:
        if not self.path.exists():
            return []
        entries = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            envelope = json.loads(line)
            record = envelope.get("record")
            if envelope.get("schema_version") != VERSION or not isinstance(record, Mapping):
                raise ValueError(f"Invalid VoI trace envelope at line {line_number}")
            expected = hashlib.sha256(json_bytes(record)).hexdigest()
            if envelope.get("record_sha256") != expected:
                raise ValueError(f"VoI trace checksum mismatch at line {line_number}")
            entry = VoITrace(**{key: value for key, value in record.items()
                                if key not in {"schema_version", "hidden_result_included"}})
            if entry.trace_id != record["trace_id"]:
                raise ValueError("VoI trace identity mismatch")
            if record.get("hidden_result_included") is not False:
                raise ValueError("VoI trace hidden-result flag must be false")
            entries.append(entry)
        return entries

    def entries(self) -> tuple[VoITrace, ...]:
        with FileLock(str(self.lock_path)):
            return tuple(self._read_unlocked())

    def append(self, entries: Iterable[VoITrace]) -> tuple[VoITrace, ...]:
        incoming = tuple(entries)
        if not incoming:
            raise ValueError("At least one VoI trace is required")
        if len({entry.trace_id for entry in incoming}) != len(incoming):
            raise ValueError("VoI trace IDs must be unique")
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            existing = self._read_unlocked()
            existing_ids = {entry.trace_id for entry in existing}
            if any(entry.trace_id in existing_ids for entry in incoming):
                raise ValueError("VoI trace ledger is append-only; trace identity already exists")
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                for entry in incoming:
                    record = entry.as_dict()
                    envelope = {"schema_version": VERSION, "record": record,
                                "record_sha256": hashlib.sha256(json_bytes(record)).hexdigest()}
                    stream.write(json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
        return incoming

    def for_episode(self, episode_id: str, *, target_key: str | None = None) -> tuple[VoITrace, ...]:
        return tuple(entry for entry in self.entries()
                     if entry.episode_id == episode_id and (target_key is None or entry.target_key == target_key))


__all__ = ["VERSION", "VoITrace", "VoITraceLedger"]
