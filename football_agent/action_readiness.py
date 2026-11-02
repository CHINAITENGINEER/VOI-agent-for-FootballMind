"""Machine-readable next-step hints for an aggregated information action.

The readiness layer describes evidence state only. Numeric VoI and cost gates
remain the authority for deciding whether a query is worth paying for.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import hashlib
import json

from .action_evidence_aggregate import ActionEvidenceAggregate


VERSION = "action-readiness-v1"
NEXT_STEPS = frozenset({
    "STOP_ACTION_SATISFIED",
    "QUERY_MISSING_DEPENDENCIES",
    "RETRY_FAILED_DEPENDENCIES",
    "QUERY_ALTERNATIVE_SOURCE",
    "WAIT_FOR_LATE_EVIDENCE",
})


@dataclass(frozen=True)
class ActionReadiness:
    aggregate_id: str
    mode: str
    availability: str
    quality: str
    next_step: str
    actionable_dependency_keys: tuple[str, ...]
    late_dependency_keys: tuple[str, ...]
    failed_dependency_keys: tuple[str, ...]
    missing_dependency_keys: tuple[str, ...]
    duplicate_dependency_keys: tuple[str, ...] = ()

    def __post_init__(self):
        if self.next_step not in NEXT_STEPS:
            raise ValueError("Unsupported action readiness next step")
        for name in ("actionable_dependency_keys", "late_dependency_keys", "failed_dependency_keys", "missing_dependency_keys", "duplicate_dependency_keys"):
            values = tuple(dict.fromkeys(getattr(self, name)))
            object.__setattr__(self, name, tuple(sorted(values)))

    def as_dict(self):
        return {
            "schema_version": VERSION,
            "aggregate_id": self.aggregate_id,
            "mode": self.mode,
            "availability": self.availability,
            "quality": self.quality,
            "next_step": self.next_step,
            "actionable_dependency_keys": list(self.actionable_dependency_keys),
            "late_dependency_keys": list(self.late_dependency_keys),
            "failed_dependency_keys": list(self.failed_dependency_keys),
            "missing_dependency_keys": list(self.missing_dependency_keys),
            "duplicate_dependency_keys": list(self.duplicate_dependency_keys),
        }


ROUTE_ROLES = frozenset({"primary", "retry", "alternative"})


def filter_candidate_routes(readiness: ActionReadiness, candidates: Iterable[Mapping]):
    """Filter tool routes using evidence readiness before VoI ranking.

    Each route declares ``candidate_id``, ``dependency_key`` and a role. The
    function returns an audit row for every route, so an Agent can explain why
    a tool was eligible or held back. It never assigns a value to a route.
    """
    rows = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Candidate route must be a mapping")
        candidate_id = candidate.get("candidate_id")
        dependency_key = candidate.get("dependency_key")
        role = candidate.get("route_role", "primary")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("Candidate route needs a candidate_id")
        if not isinstance(dependency_key, str) or not dependency_key.strip():
            raise ValueError("Candidate route needs a dependency_key")
        if role not in ROUTE_ROLES:
            raise ValueError("Unsupported candidate route role")
        if candidate_id in seen:
            raise ValueError("Candidate route IDs must be unique")
        seen.add(candidate_id)
        if readiness.next_step == "STOP_ACTION_SATISFIED":
            eligible, reason = False, "action_already_satisfied"
        elif readiness.next_step == "QUERY_MISSING_DEPENDENCIES":
            eligible = dependency_key in readiness.missing_dependency_keys and role == "primary"
            reason = "missing_dependency_route" if eligible else "dependency_not_missing"
        elif readiness.next_step == "RETRY_FAILED_DEPENDENCIES":
            eligible = dependency_key in readiness.failed_dependency_keys and role == "retry"
            reason = "failed_dependency_retry" if eligible else "route_not_retry_for_failed_dependency"
        elif readiness.next_step == "QUERY_ALTERNATIVE_SOURCE":
            eligible = dependency_key in (set(readiness.failed_dependency_keys) | set(readiness.duplicate_dependency_keys)) and role == "alternative"
            reason = "failed_dependency_alternative" if eligible else "route_not_alternative_for_failed_dependency"
        else:
            eligible, reason = False, "wait_for_late_evidence"
        rows.append({"candidate_id": candidate_id, "dependency_key": dependency_key,
                     "route_role": role, "eligible": eligible, "reason": reason})
    return tuple(rows)


def _missing_readiness(dependency_key: str) -> ActionReadiness:
    digest = hashlib.sha256(dependency_key.encode("utf-8")).hexdigest()
    return ActionReadiness(
        aggregate_id="missing-binding:" + digest, mode="any_usable",
        availability="failed", quality="SOURCE_MISSING",
        next_step="QUERY_MISSING_DEPENDENCIES",
        actionable_dependency_keys=(dependency_key,), late_dependency_keys=(),
        failed_dependency_keys=(), missing_dependency_keys=(dependency_key,),
        duplicate_dependency_keys=(),
    )


def filter_candidate_routes_by_bindings(bindings: Iterable, candidates: Iterable[Mapping]):
    """Filter routes using the current binding state for each dependency.

    Action-level ``any_usable`` may already be satisfied by one source. This
    function keeps that aggregate fact while still exposing source-level
    conflict/unverified/duplicate states to each candidate route.
    """
    binding_rows = tuple(bindings)
    by_dependency: dict[str, list] = {}
    for binding in binding_rows:
        dependency = getattr(binding, "dependency_key", None)
        if not isinstance(dependency, str) or not dependency.strip():
            raise ValueError("Evidence binding needs a dependency_key")
        by_dependency.setdefault(dependency, []).append(binding)
    readiness_by_dependency = {
        dependency: readiness_from_bindings(rows, mode="any_usable")
        for dependency, rows in by_dependency.items()
    }
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Candidate route must be a mapping")
        dependency = candidate.get("dependency_key")
        if not isinstance(dependency, str) or not dependency.strip():
            raise ValueError("Candidate route needs a dependency_key")
        readiness = readiness_by_dependency.get(dependency) or _missing_readiness(dependency)
        filtered = filter_candidate_routes(readiness, (candidate,))[0]
        rows.append({**filtered, "dependency_readiness": readiness.as_dict()})
    return tuple(rows), readiness_by_dependency


@dataclass(frozen=True)
class ReadinessQueryPlan:
    """A filtered StateQueryBoard ready for the existing VoI gate."""

    readiness: ActionReadiness
    board: object
    route_audit: tuple[dict, ...]
    eligible_candidate_ids: tuple[str, ...]

    def as_dict(self):
        return {
            "schema_version": VERSION,
            "readiness": self.readiness.as_dict(),
            "route_audit": [dict(row) for row in self.route_audit],
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "board": self.board.manifest(view="evaluator"),
        }


@dataclass(frozen=True)
class BindingReadinessQueryPlan:
    """A query board filtered per dependency binding."""

    board: object
    route_audit: tuple[dict, ...]
    eligible_candidate_ids: tuple[str, ...]
    readiness_by_dependency: tuple[tuple[str, ActionReadiness], ...]

    def as_dict(self):
        return {
            "schema_version": VERSION,
            "route_audit": [dict(row) for row in self.route_audit],
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "readiness_by_dependency": {
                key: value.as_dict() for key, value in self.readiness_by_dependency
            },
            "board": self.board.manifest(view="evaluator"),
        }


def prepare_query_board(board, readiness: ActionReadiness, candidates: Iterable[Mapping]):
    """Apply evidence readiness before calling ``QueryValueGate.decide``.

    The returned board preserves hypotheses, target and cutoff while exposing
    only routes eligible for the current evidence state. The caller still runs
    the normal numeric VoI/cost gate on this board.
    """
    from .state_query_contract import StateQueryBoard

    route_audit = filter_candidate_routes(readiness, candidates)
    board_ids = {candidate.candidate_id for candidate in board.candidates}
    route_ids = {row["candidate_id"] for row in route_audit}
    if not route_ids <= board_ids:
        raise ValueError("Readiness routes must belong to the query board")
    eligible_ids = tuple(row["candidate_id"] for row in route_audit if row["eligible"])
    allowed = set(eligible_ids)
    filtered_board = StateQueryBoard(
        target_key=board.target_key,
        prediction_cutoff=board.prediction_cutoff,
        hypotheses=board.hypotheses,
        candidates=[candidate for candidate in board.candidates if candidate.candidate_id in allowed],
        registry=board.registry,
    )
    return ReadinessQueryPlan(
        readiness=readiness, board=filtered_board, route_audit=route_audit,
        eligible_candidate_ids=eligible_ids,
    )


def prepare_query_board_from_bindings(board, bindings: Iterable, candidates: Iterable[Mapping]):
    """Build a VoI-ready board while preserving per-dependency readiness."""
    route_audit, readiness_by_dependency = filter_candidate_routes_by_bindings(bindings, candidates)
    board_ids = {candidate.candidate_id for candidate in board.candidates}
    route_ids = {row["candidate_id"] for row in route_audit}
    if not route_ids <= board_ids:
        raise ValueError("Binding routes must belong to the query board")
    eligible_ids = tuple(row["candidate_id"] for row in route_audit if row["eligible"])
    allowed = set(eligible_ids)
    from .state_query_contract import StateQueryBoard
    filtered_board = StateQueryBoard(
        target_key=board.target_key, prediction_cutoff=board.prediction_cutoff,
        hypotheses=board.hypotheses,
        candidates=[candidate for candidate in board.candidates if candidate.candidate_id in allowed],
        registry=board.registry,
    )
    return BindingReadinessQueryPlan(
        board=filtered_board, route_audit=route_audit,
        eligible_candidate_ids=eligible_ids,
        readiness_by_dependency=tuple(sorted(readiness_by_dependency.items())),
    )


def readiness_for(aggregate: ActionEvidenceAggregate) -> ActionReadiness:
    statuses = dict(aggregate.dependency_statuses)
    missing = tuple(aggregate.missing_dependency_keys)
    available = tuple(key for key, status in statuses.items() if status == "available")
    late = tuple(key for key, status in statuses.items() if status == "late")
    failed = tuple(key for key, status in statuses.items() if status == "failed")
    duplicate = tuple(key for key, status in statuses.items() if status == "duplicate")
    if aggregate.availability == "available":
        next_step = "STOP_ACTION_SATISFIED"
        actionable = ()
    elif missing:
        next_step = "QUERY_MISSING_DEPENDENCIES"
        actionable = missing
    elif aggregate.mode == "any_usable" and (failed or duplicate):
        next_step = "QUERY_ALTERNATIVE_SOURCE"
        actionable = failed + duplicate
    elif failed:
        next_step = "RETRY_FAILED_DEPENDENCIES"
        actionable = failed
    elif duplicate:
        next_step = "QUERY_ALTERNATIVE_SOURCE"
        actionable = duplicate
    elif late:
        next_step = "WAIT_FOR_LATE_EVIDENCE"
        actionable = ()
    else:
        next_step = "QUERY_ALTERNATIVE_SOURCE"
        actionable = ()
    return ActionReadiness(
        aggregate_id=aggregate.aggregate_id, mode=aggregate.mode,
        availability=aggregate.availability, quality=aggregate.quality,
        next_step=next_step, actionable_dependency_keys=actionable,
        late_dependency_keys=late, failed_dependency_keys=failed,
        missing_dependency_keys=missing, duplicate_dependency_keys=duplicate,
    )


def readiness_from_bindings(bindings: Iterable, *, mode: str,
                            required_dependency_keys: Iterable[str] | None = None,
                            aggregate_id: str | None = None) -> ActionReadiness:
    """Convert platform evidence bindings into readiness without adding VoI.

    This is the bridge for real collectors: operational availability and
    quality decide dependency state, while the numeric value gate remains a
    separate step.
    """
    if mode not in {"all_required", "any_usable"}:
        raise ValueError("Unsupported action readiness mode")
    rows = tuple(bindings)
    if not rows:
        raise ValueError("At least one evidence binding is required")
    required = None if required_dependency_keys is None else tuple(dict.fromkeys(required_dependency_keys))
    if required is not None and not required:
        raise ValueError("required_dependency_keys must not be empty")
    grouped: dict[str, list] = {}
    for binding in rows:
        dependency = getattr(binding, "dependency_key", None)
        availability = getattr(binding, "availability", None)
        quality = getattr(binding, "quality", None)
        if not isinstance(dependency, str) or not dependency.strip():
            raise ValueError("Evidence binding needs a dependency_key")
        if availability not in {"available", "late", "failed", "duplicate"}:
            raise ValueError("Evidence binding has unsupported availability")
        grouped.setdefault(dependency, []).append(binding)

    statuses = {}
    qualities = {}
    for dependency, candidates in grouped.items():
        usable = [item for item in candidates if item.availability == "available" and item.quality in {"VALIDATED", "PARTIAL"}]
        late = [item for item in candidates if item.availability == "late"]
        if usable:
            statuses[dependency] = "available"
            qualities[dependency] = "VALIDATED" if any(item.quality == "VALIDATED" for item in usable) else "PARTIAL"
        elif late:
            statuses[dependency] = "late"
            qualities[dependency] = "PARTIAL"
        elif any(item.availability == "duplicate" for item in candidates):
            statuses[dependency] = "duplicate"
            qualities[dependency] = "DUPLICATE"
        else:
            statuses[dependency] = "failed"
            quality_priority = ("SOURCE_MISSING", "CONFLICT", "STALE", "INVALID", "PROVIDER_ERROR", "UNVERIFIED", "NO_CHANGE")
            qualities[dependency] = next((value for value in quality_priority
                                          if any(item.quality == value for item in candidates)), "PROVIDER_ERROR")

    declared = set(required) if required is not None else set(grouped)
    missing = tuple(sorted(declared - set(grouped))) if mode == "all_required" and required is not None else ()
    relevant = [statuses[key] for key in sorted(declared & set(statuses))]
    if mode == "all_required":
        if missing or any(state == "failed" for state in relevant):
            failure_quality = next((qualities[key] for key in sorted(declared) if statuses.get(key) == "failed"), "PROVIDER_ERROR")
            availability, quality, next_step = "failed", "SOURCE_MISSING" if missing else failure_quality, "QUERY_MISSING_DEPENDENCIES" if missing else "RETRY_FAILED_DEPENDENCIES"
            actionable = missing or tuple(key for key in sorted(declared) if statuses.get(key) == "failed")
        elif any(state == "duplicate" for state in relevant):
            availability, quality, next_step = "failed", "DUPLICATE", "QUERY_ALTERNATIVE_SOURCE"
            actionable = tuple(key for key in sorted(declared) if statuses.get(key) == "duplicate")
        elif relevant and all(state == "available" for state in relevant):
            availability, quality, next_step, actionable = "available", "VALIDATED" if all(qualities[key] == "VALIDATED" for key in declared) else "PARTIAL", "STOP_ACTION_SATISFIED", ()
        else:
            availability, quality, next_step, actionable = "late", "PARTIAL", "WAIT_FOR_LATE_EVIDENCE", ()
    else:
        if any(state == "available" for state in relevant):
            availability, quality, next_step, actionable = "available", "VALIDATED" if any(qualities[key] == "VALIDATED" for key in declared if statuses.get(key) == "available") else "PARTIAL", "STOP_ACTION_SATISFIED", ()
        elif any(state == "late" for state in relevant):
            availability, quality, next_step, actionable = "late", "PARTIAL", "WAIT_FOR_LATE_EVIDENCE", ()
        else:
            duplicate_keys = tuple(key for key in sorted(declared) if statuses.get(key) == "duplicate")
            has_failed = any(statuses.get(key) == "failed" for key in declared)
            failure_quality = next((qualities[key] for key in sorted(declared) if statuses.get(key) == "failed"), "PROVIDER_ERROR")
            availability, quality, next_step = "failed", "DUPLICATE" if duplicate_keys and not has_failed else failure_quality, "QUERY_ALTERNATIVE_SOURCE"
            actionable = tuple(sorted(key for key in declared if statuses.get(key) in {"failed", "duplicate"}))
    if aggregate_id is None:
        body = {"mode": mode, "required": sorted(declared), "statuses": sorted(statuses.items())}
        aggregate_id = "binding-readiness:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return ActionReadiness(
        aggregate_id=aggregate_id, mode=mode, availability=availability, quality=quality,
        next_step=next_step, actionable_dependency_keys=tuple(actionable),
        late_dependency_keys=tuple(key for key in sorted(declared) if statuses.get(key) == "late"),
        failed_dependency_keys=tuple(key for key in sorted(declared) if statuses.get(key) == "failed"),
        missing_dependency_keys=missing,
        duplicate_dependency_keys=tuple(key for key in sorted(declared) if statuses.get(key) == "duplicate"),
    )


__all__ = ["VERSION", "NEXT_STEPS", "ROUTE_ROLES", "ActionReadiness", "ReadinessQueryPlan",
           "BindingReadinessQueryPlan", "readiness_for", "readiness_from_bindings",
           "filter_candidate_routes", "filter_candidate_routes_by_bindings",
           "prepare_query_board", "prepare_query_board_from_bindings"]
