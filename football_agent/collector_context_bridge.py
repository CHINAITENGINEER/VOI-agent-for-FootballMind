"""Bridge collector context snapshots into incremental Agent context state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .collector_context_adapter import CollectorContextSnapshot
from .context_query_readiness import ContextState
from .context_transition import ContextTransition


VERSION = "collector-context-bridge-v1"
_MONOTONE_KEYS = frozenset({"odds_available", "quote_count"})


@dataclass(frozen=True)
class ContextProjection:
    target_key: str
    prediction_cutoff: str
    previous_context: ContextState
    next_context: ContextState
    accepted_context_keys: tuple[str, ...]
    rejected_context_keys: tuple[str, ...]
    conflict_keys: tuple[str, ...]
    accepted_evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.previous_context, ContextState) or not isinstance(self.next_context, ContextState):
            raise ValueError("Context projection states must be ContextState values")
        object.__setattr__(self, "accepted_context_keys", tuple(sorted(set(self.accepted_context_keys))))
        object.__setattr__(self, "rejected_context_keys", tuple(sorted(set(self.rejected_context_keys))))
        object.__setattr__(self, "conflict_keys", tuple(sorted(set(self.conflict_keys))))
        object.__setattr__(self, "accepted_evidence_refs", tuple(dict.fromkeys(self.accepted_evidence_refs)))

    @property
    def changed_keys(self) -> tuple[str, ...]:
        keys = set(self.previous_context.values) | set(self.next_context.values)
        return tuple(sorted(key for key in keys
                           if key not in self.previous_context.values
                           or key not in self.next_context.values
                           or self.previous_context.values[key] != self.next_context.values[key]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "target_key": self.target_key,
            "prediction_cutoff": self.prediction_cutoff,
            "previous_context": self.previous_context.as_dict(),
            "next_context": self.next_context.as_dict(),
            "accepted_context_keys": list(self.accepted_context_keys),
            "rejected_context_keys": list(self.rejected_context_keys),
            "conflict_keys": list(self.conflict_keys),
            "changed_keys": list(self.changed_keys),
            "accepted_evidence_refs": list(self.accepted_evidence_refs),
            "hidden_result_included": False,
        }


def apply_context_snapshot(
    previous: ContextState | Mapping[str, Any],
    snapshot: CollectorContextSnapshot,
) -> ContextProjection:
    """Apply accepted snapshot values while preserving previous conflicts."""
    if not isinstance(snapshot, CollectorContextSnapshot):
        raise ValueError("snapshot must be a CollectorContextSnapshot")
    previous_state = previous if isinstance(previous, ContextState) else ContextState(previous)
    values = dict(previous_state.values)
    conflicts = set(snapshot.conflict_keys)
    rejected = set()
    accepted = set()
    refs = []
    for evidence in snapshot.evidence:
        if not evidence.accepted or not evidence.visible_to_agent:
            rejected.add(evidence.context_key)
            continue
        if evidence.context_key not in snapshot.context.values:
            rejected.add(evidence.context_key)
            continue
        current = values.get(evidence.context_key)
        if evidence.context_key == "odds_available" and evidence.context_key in values:
            values[evidence.context_key] = bool(current) or bool(evidence.value)
            accepted.add(evidence.context_key)
            refs.extend(evidence.evidence_refs)
            continue
        if evidence.context_key == "quote_count" and evidence.context_key in values:
            values[evidence.context_key] = max(int(current), int(evidence.value))
            accepted.add(evidence.context_key)
            refs.extend(evidence.evidence_refs)
            continue
        if evidence.context_key in values and current != evidence.value:
            conflicts.add(evidence.context_key)
            rejected.add(evidence.context_key)
            continue
        values[evidence.context_key] = evidence.value
        accepted.add(evidence.context_key)
        refs.extend(evidence.evidence_refs)
    next_state = ContextState(values)
    return ContextProjection(
        target_key=snapshot.target_key,
        prediction_cutoff=snapshot.prediction_cutoff,
        previous_context=previous_state,
        next_context=next_state,
        accepted_context_keys=tuple(accepted),
        rejected_context_keys=tuple(rejected),
        conflict_keys=tuple(conflicts),
        accepted_evidence_refs=tuple(refs),
    )


def projection_transition(
    projection: ContextProjection,
    *,
    episode_id: str,
    sequence: int,
    reason: str,
    trigger: str = "collector_snapshot",
    captured_at: str | None = None,
) -> ContextTransition:
    """Create a hashable context event from a collector projection."""
    return ContextTransition.from_transition(
        episode_id=episode_id, target_key=projection.target_key, sequence=sequence,
        event_type="CONTEXT_UPDATED", previous_context=projection.previous_context.values,
        next_context=projection.next_context.values, trigger=trigger, reason=reason,
        prediction_cutoff=projection.prediction_cutoff, captured_at=captured_at,
        source_id="collector-context-bridge", evidence_refs=projection.accepted_evidence_refs,
    )


__all__ = ["VERSION", "ContextProjection", "apply_context_snapshot", "projection_transition"]
