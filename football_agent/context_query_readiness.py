"""Gate domain queries on the public context they explicitly require.

This is a semantic planning layer. It decides whether a route has enough
declared context to be considered by the numeric VoI gate; it does not assign
information value to context or domain routes.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any

from .state_query_contract import StateQueryBoard


VERSION = "context-query-readiness-v1"
ROUTE_KINDS = frozenset({"context", "domain"})
UNKNOWN_MARKERS = frozenset({"", "unknown", "unresolved", "UNKNOWN", "UNRESOLVED"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _keys(value: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if value is None:
        value = ()
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of keys")
    result = tuple(_text(item, name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique keys")
    if required and not result:
        raise ValueError(f"{name} must contain at least one key")
    return result


def _known(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value.strip() in UNKNOWN_MARKERS)


@dataclass(frozen=True)
class ContextState:
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise ValueError("Context values must be a mapping")
        normalized = {}
        for key, value in self.values.items():
            normalized[_text(key, "context key")] = value
        object.__setattr__(self, "values", {key: normalized[key] for key in sorted(normalized)})

    def missing(self, keys: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(key for key in keys if not _known(self.values.get(key))))

    def known(self, key: str) -> bool:
        return _known(self.values.get(key))

    def with_updates(self, updates: Mapping[str, Any]) -> "ContextState":
        if not isinstance(updates, Mapping):
            raise ValueError("Context updates must be a mapping")
        return ContextState({**self.values, **dict(updates)})

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": VERSION, "values": dict(self.values)}


@dataclass(frozen=True)
class ContextQueryPlan:
    board: StateQueryBoard
    context: ContextState
    route_audit: tuple[dict[str, Any], ...]
    eligible_candidate_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "context": self.context.as_dict(),
            "route_audit": [dict(row) for row in self.route_audit],
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "board": self.board.manifest(view="evaluator"),
        }


def filter_context_routes(
    board: StateQueryBoard,
    context: ContextState | Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Filter routes whose declared context prerequisites are unresolved."""
    if not isinstance(board, StateQueryBoard):
        raise ValueError("board must be a StateQueryBoard")
    state = context if isinstance(context, ContextState) else ContextState(context)
    board_ids = {candidate.candidate_id for candidate in board.candidates}
    rows = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Context route must be a mapping")
        candidate_id = _text(candidate.get("candidate_id"), "candidate_id")
        if candidate_id in seen:
            raise ValueError("Context route IDs must be unique")
        if candidate_id not in board_ids:
            raise ValueError("Context route must belong to the query board")
        seen.add(candidate_id)
        route_kind = candidate.get("route_kind", "domain")
        if route_kind not in ROUTE_KINDS:
            raise ValueError("Unsupported context route kind")
        required = _keys(candidate.get("required_context_keys", ()), "required_context_keys")
        provides = _keys(candidate.get("provides_context_keys", ()), "provides_context_keys",
                         required=route_kind == "context")
        missing_required = state.missing(required)
        missing_provided = tuple(sorted(key for key in provides if not state.known(key)))
        if missing_required:
            eligible, reason = False, "context_prerequisite_missing"
        elif route_kind == "context":
            eligible = bool(missing_provided)
            reason = "context_resolution_route" if eligible else "context_already_known"
        else:
            eligible = True
            reason = "context_ready_for_domain_query"
        rows.append({
            "candidate_id": candidate_id,
            "route_kind": route_kind,
            "required_context_keys": list(required),
            "provides_context_keys": list(provides),
            "missing_context_keys": list(sorted(set(missing_required) | set(missing_provided)
                                                 if route_kind == "context" else missing_required)),
            "eligible": eligible,
            "reason": reason,
            "expected_value": None,
        })
    return tuple(rows)


def prepare_context_query_board(
    board: StateQueryBoard,
    context: ContextState | Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
) -> ContextQueryPlan:
    state = context if isinstance(context, ContextState) else ContextState(context)
    route_audit = filter_context_routes(board, state, candidates)
    eligible = tuple(row["candidate_id"] for row in route_audit if row["eligible"])
    allowed = set(eligible)
    filtered = StateQueryBoard(
        target_key=board.target_key, prediction_cutoff=board.prediction_cutoff,
        hypotheses=board.hypotheses,
        candidates=[candidate for candidate in board.candidates if candidate.candidate_id in allowed],
        registry=board.registry,
    )
    return ContextQueryPlan(filtered, state, route_audit, eligible)


__all__ = [
    "VERSION", "ROUTE_KINDS", "ContextState", "ContextQueryPlan",
    "filter_context_routes", "prepare_context_query_board",
]
