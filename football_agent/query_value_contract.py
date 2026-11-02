"""Explicit boundary between query actionability and numeric information value.

``StateQueryBoard`` answers which state a route could inspect.  This module
answers what the current return model supports about that route.  An
uncalibrated or open-world route remains ``VALUE_UNMEASURED``; it never becomes
zero merely because a finite model did not register it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from .state_query_contract import StateQueryBoard


VERSION = "query-value-contract-v1"
TOLERANCE = 1e-12

VALUE_UNMEASURED = "VALUE_UNMEASURED"
MODELED_DECLARED_ASSUMPTIONS = "MODELED_DECLARED_ASSUMPTIONS"
CALIBRATION_DIAGNOSTIC_ONLY = "CALIBRATION_DIAGNOSTIC_ONLY"
VALUE_UNAVAILABLE_CUTOFF = "VALUE_UNAVAILABLE_CUTOFF"
VALUE_UNAVAILABLE_BUDGET = "VALUE_UNAVAILABLE_BUDGET"
VALUE_STATUSES = frozenset({
    VALUE_UNMEASURED, MODELED_DECLARED_ASSUMPTIONS, CALIBRATION_DIAGNOSTIC_ONLY,
    VALUE_UNAVAILABLE_CUTOFF, VALUE_UNAVAILABLE_BUDGET,
})

QUERY = "QUERY"
STOP = "STOP"
DEFER = "DEFER"
DECISION_ACTIONS = frozenset({QUERY, STOP, DEFER})

MODEL_SCOPED_STOP = "MODEL_SCOPED_STOP"
WITHHELD_OPEN_WORLD_VALUE_UNKNOWN = "WITHHELD_OPEN_WORLD_VALUE_UNKNOWN"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
CUTOFF_EXCLUDED = "CUTOFF_EXCLUDED"
NO_FEASIBLE_REGISTERED_ACTION = "NO_FEASIBLE_REGISTERED_ACTION"
STOP_KINDS = frozenset({
    MODEL_SCOPED_STOP, WITHHELD_OPEN_WORLD_VALUE_UNKNOWN, BUDGET_EXHAUSTED,
    CUTOFF_EXCLUDED, NO_FEASIBLE_REGISTERED_ACTION,
})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _budget_snapshot(value: Any, name: str) -> float | dict[str, float]:
    """Normalize either a scalar budget or independent named resource slots.

    A multi-resource runtime must not add incomparable units such as HTTP
    calls and local reads merely to fit a scalar certificate field.  The
    scalar form remains the native interface for ``QueryValueGate``; the
    mapping form preserves the runtime's declared slots without inventing a
    cross-resource exchange rate.
    """
    if isinstance(value, Mapping):
        normalized: dict[str, float] = {}
        for key, amount in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{name} keys must be nonempty strings")
            normalized[key] = _finite_nonnegative(amount, f"{name}[{key}]")
        return dict(sorted(normalized.items()))
    return _finite_nonnegative(value, name)


@dataclass(frozen=True)
class QueryValueAssessment:
    """Value support for one candidate, with unmeasured states explicit."""

    candidate_id: str
    value_status: str
    expected_value: float | None
    cost: float | None
    net_value: float | None
    support_scope: str
    calibration_status: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        if self.value_status not in VALUE_STATUSES:
            raise ValueError("Unsupported query value status")
        if self.expected_value is not None:
            object.__setattr__(self, "expected_value", _finite_nonnegative(self.expected_value, "expected_value"))
        if self.cost is not None:
            object.__setattr__(self, "cost", _finite_nonnegative(self.cost, "cost"))
        if self.net_value is not None:
            if self.value_status != MODELED_DECLARED_ASSUMPTIONS:
                raise ValueError("Only a declared finite model may provide net value")
            if not math.isfinite(self.net_value):
                raise ValueError("net_value must be finite")
            object.__setattr__(self, "net_value", float(self.net_value))
        if self.value_status == MODELED_DECLARED_ASSUMPTIONS:
            if self.expected_value is None or self.cost is None or self.net_value is None:
                raise ValueError("A modeled value needs expected value, cost and net value")
            if abs(self.net_value - (self.expected_value - self.cost)) > TOLERANCE:
                raise ValueError("net_value must equal expected_value minus cost")
            if self.support_scope != "declared_finite_model":
                raise ValueError("Modeled value needs declared_finite_model scope")
        else:
            if self.expected_value is not None:
                raise ValueError("Unmodeled value cannot carry expected value")
            if self.net_value is not None:
                raise ValueError("Unmodeled value cannot carry net value")
        object.__setattr__(self, "support_scope", _text(self.support_scope, "support_scope"))
        object.__setattr__(self, "calibration_status", _text(self.calibration_status, "calibration_status"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))

    @classmethod
    def unmeasured(cls, candidate_id: str, *, cost: float | None = None,
                   reason: str = "return_model_not_calibrated") -> "QueryValueAssessment":
        return cls(candidate_id, VALUE_UNMEASURED, None, None if cost is None else _finite_nonnegative(cost, "cost"),
                   None, "open_world_or_unregistered", "NOT_CALIBRATED", reason)

    @classmethod
    def modeled(cls, candidate_id: str, *, expected_value: float, cost: float,
                reason: str = "declared_finite_model") -> "QueryValueAssessment":
        expected_value, cost = _finite_nonnegative(expected_value, "expected_value"), _finite_nonnegative(cost, "cost")
        return cls(candidate_id, MODELED_DECLARED_ASSUMPTIONS, expected_value, cost,
                   expected_value - cost, "declared_finite_model", "ASSUMPTION_ONLY", reason)

    @classmethod
    def from_finite_model_row(cls, candidate_id: str, row: Mapping[str, Any]) -> "QueryValueAssessment":
        if not isinstance(row, Mapping) or not row.get("feasible"):
            return cls.unmeasured(candidate_id, reason="finite_model_row_unavailable")
        if "immediate_evsi" not in row or "cost" not in row:
            raise ValueError("A finite model row needs immediate_evsi and cost")
        return cls.modeled(candidate_id, expected_value=row["immediate_evsi"], cost=row["cost"],
                           reason="finite_model_candidate_row")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION, "candidate_id": self.candidate_id,
            "value_status": self.value_status, "expected_value": self.expected_value,
            "cost": self.cost, "net_value": self.net_value,
            "support_scope": self.support_scope, "calibration_status": self.calibration_status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StopCertificate:
    """A scoped stopping explanation; it never claims open-world optimality."""

    kind: str
    scope: str
    global_claim: bool
    reason: str
    actionable_unknown_count: int
    evaluated_candidate_ids: tuple[str, ...]
    remaining_budget: float | Mapping[str, float]
    maximum_modeled_net_value: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in STOP_KINDS:
            raise ValueError("Unsupported stop certificate kind")
        object.__setattr__(self, "scope", _text(self.scope, "scope"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if isinstance(self.actionable_unknown_count, bool) or not isinstance(self.actionable_unknown_count, int) or self.actionable_unknown_count < 0:
            raise ValueError("actionable_unknown_count must be a nonnegative integer")
        ids = tuple(_text(value, "candidate_id") for value in self.evaluated_candidate_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("evaluated candidate IDs must be unique")
        object.__setattr__(self, "evaluated_candidate_ids", ids)
        object.__setattr__(self, "remaining_budget", _budget_snapshot(self.remaining_budget, "remaining_budget"))
        if self.maximum_modeled_net_value is not None and not math.isfinite(self.maximum_modeled_net_value):
            raise ValueError("maximum_modeled_net_value must be finite")
        if self.maximum_modeled_net_value is not None:
            object.__setattr__(self, "maximum_modeled_net_value", float(self.maximum_modeled_net_value))
        if self.global_claim:
            raise ValueError("Stop certificates cannot claim an open-world global optimum")
        if self.kind == MODEL_SCOPED_STOP:
            if self.scope != "declared_finite_model" or self.maximum_modeled_net_value is None:
                raise ValueError("Model-scoped stop needs a finite model scope and maximum net value")
            if self.maximum_modeled_net_value > TOLERANCE:
                raise ValueError("Model-scoped stop requires no positive modeled net value")
        if self.kind == WITHHELD_OPEN_WORLD_VALUE_UNKNOWN and self.scope != "open_world":
            raise ValueError("Open-world withheld stop needs open_world scope")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION, "kind": self.kind, "scope": self.scope,
            "global_claim": self.global_claim, "reason": self.reason,
            "actionable_unknown_count": self.actionable_unknown_count,
            "evaluated_candidate_ids": list(self.evaluated_candidate_ids),
            "remaining_budget": (dict(self.remaining_budget) if isinstance(self.remaining_budget, Mapping)
                                 else self.remaining_budget),
            "maximum_modeled_net_value": self.maximum_modeled_net_value,
        }


@dataclass(frozen=True)
class QueryValueDecision:
    action: str
    candidate_id: str | None
    selection_basis: str
    assessments: tuple[QueryValueAssessment, ...]
    stop_certificate: StopCertificate | None

    def __post_init__(self) -> None:
        if self.action not in DECISION_ACTIONS:
            raise ValueError("Unsupported query value decision action")
        if self.action == QUERY and self.candidate_id is None:
            raise ValueError("A query decision needs a candidate ID")
        if self.action != QUERY and self.stop_certificate is None:
            raise ValueError("Stop or defer decisions need a scoped certificate")
        if self.action == QUERY and self.stop_certificate is not None:
            raise ValueError("A query decision cannot carry a stop certificate")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION, "action": self.action,
            "candidate_id": self.candidate_id, "selection_basis": self.selection_basis,
            "assessments": [assessment.as_dict() for assessment in self.assessments],
            "stop_certificate": None if self.stop_certificate is None else self.stop_certificate.as_dict(),
        }


class QueryValueGate:
    """Apply finite-model value only where its assumptions are declared."""

    version = VERSION

    @staticmethod
    def decide(board: StateQueryBoard, assessments: Iterable[QueryValueAssessment], *,
               remaining_budget: float, view: str = "evaluator") -> QueryValueDecision:
        remaining_budget = _finite_nonnegative(remaining_budget, "remaining_budget")
        actionable = board.actionable_candidates(view=view)
        assessments = tuple(assessments)
        by_id = {assessment.candidate_id: assessment for assessment in assessments}
        if len(by_id) != len(assessments):
            raise ValueError("Query value assessments need unique candidate IDs")
        unknown_ids = {candidate.candidate_id for candidate in actionable}
        if set(by_id) - unknown_ids:
            raise ValueError("Assessments must correspond to currently actionable candidates")
        normalized = tuple(by_id[candidate.candidate_id] if candidate.candidate_id in by_id else
                           QueryValueAssessment.unmeasured(candidate.candidate_id, cost=candidate.cost)
                           for candidate in actionable)
        if not normalized:
            certificate = StopCertificate(
                NO_FEASIBLE_REGISTERED_ACTION, "declared_candidate_registry", False,
                "No registered candidate addresses a currently visible unknown state.", 0, (), remaining_budget, 0.0)
            return QueryValueDecision(STOP, None, "semantic_candidate_scope", normalized, certificate)
        if remaining_budget <= TOLERANCE:
            certificate = StopCertificate(
                BUDGET_EXHAUSTED, "operational_budget", False,
                "The declared acquisition budget is exhausted while actionable unknowns remain.",
                len(normalized), tuple(item.candidate_id for item in normalized), remaining_budget, None)
            return QueryValueDecision(DEFER, None, "budget_gate", normalized, certificate)
        feasible = [item for item in normalized if item.cost is not None and item.cost <= remaining_budget + TOLERANCE]
        if not feasible:
            certificate = StopCertificate(
                BUDGET_EXHAUSTED, "operational_budget", False,
                "Every actionable candidate exceeds the remaining acquisition budget.",
                len(normalized), tuple(item.candidate_id for item in normalized), remaining_budget, None)
            return QueryValueDecision(DEFER, None, "budget_gate", normalized, certificate)
        positive = [item for item in feasible if item.value_status == MODELED_DECLARED_ASSUMPTIONS
                    and item.net_value is not None and item.net_value > TOLERANCE]
        if positive:
            selected = max(positive, key=lambda item: (item.net_value, item.candidate_id))
            return QueryValueDecision(QUERY, selected.candidate_id,
                                      "declared_finite_model_positive_net_value", normalized, None)
        unmeasured = [item for item in feasible if item.value_status != MODELED_DECLARED_ASSUMPTIONS]
        if unmeasured:
            certificate = StopCertificate(
                WITHHELD_OPEN_WORLD_VALUE_UNKNOWN, "open_world", False,
                "No supported positive finite-model action is available; remaining routes have unmeasured or non-deployable value.",
                len(normalized), tuple(item.candidate_id for item in normalized), remaining_budget, None)
            return QueryValueDecision(DEFER, None, "withhold_open_world_value_unknown", normalized, certificate)
        maximum = max(item.net_value for item in feasible if item.net_value is not None)
        certificate = StopCertificate(
            MODEL_SCOPED_STOP, "declared_finite_model", False,
            "All feasible candidates are covered by the declared finite model and have nonpositive net value.",
            len(normalized), tuple(item.candidate_id for item in normalized), remaining_budget, maximum)
        return QueryValueDecision(STOP, None, "declared_finite_model_no_positive_net_value", normalized, certificate)
