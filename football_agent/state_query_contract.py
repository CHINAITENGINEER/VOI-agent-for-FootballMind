"""Bind candidate information routes to explicit, independent state unknowns.

The contract is the semantic layer between an Agent's current evidence state
and its candidate tool menu.  It says which unknown a route could address and
whether that unknown is already annotated at the frozen cutoff.  It deliberately
does not manufacture a numeric value for an uncalibrated route.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping, Sequence

from .prequery_state import PreQueryStateAnnotation, PreQueryStateRegistry
from .types import utc


VERSION = "state-query-contract-v1"
UNKNOWN = "UNKNOWN"
KNOWN = "KNOWN"
CONTESTED = "CONTESTED"
HYPOTHESIS_STATUSES = frozenset({UNKNOWN, KNOWN, CONTESTED})
ACTIONABLE_UNKNOWN = "ACTIONABLE_UNKNOWN"
STATE_ALREADY_KNOWN = "STATE_ALREADY_KNOWN"
STATE_CONTESTED = "STATE_CONTESTED"
EXCLUDED_TARGET = "EXCLUDED_TARGET"
EXCLUDED_CUTOFF = "EXCLUDED_CUTOFF"
EXCLUDED_NO_HYPOTHESIS = "EXCLUDED_NO_HYPOTHESIS"
PRIVATE_EVALUATOR = "private_evaluator_only"
VIEWS = frozenset({"evaluator", "agent"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _values(values: Sequence[str], name: str = "possible_values") -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of unique strings")
    result = tuple(_text(value, name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique nonempty strings")
    return result


def _cost(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("cost must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class StateHypothesis:
    """One independently tracked state variable for a target."""

    hypothesis_id: str
    target_key: str
    state_variable: str
    possible_values: tuple[str, ...]
    status: str = UNKNOWN
    known_value: str | None = None
    annotation_id: str | None = None
    scope: str = "match"
    annotation_visibility: str = PRIVATE_EVALUATOR

    def __post_init__(self) -> None:
        for field in ("hypothesis_id", "target_key", "state_variable", "scope"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "annotation_visibility", _text(self.annotation_visibility, "annotation_visibility"))
        if self.annotation_visibility != PRIVATE_EVALUATOR:
            raise ValueError("Only private evaluator state annotations are supported in this contract")
        object.__setattr__(self, "possible_values", _values(self.possible_values))
        if self.status not in HYPOTHESIS_STATUSES:
            raise ValueError("Unsupported state hypothesis status")
        if self.known_value is not None:
            object.__setattr__(self, "known_value", _text(self.known_value, "known_value"))
        if self.annotation_id is not None:
            object.__setattr__(self, "annotation_id", _text(self.annotation_id, "annotation_id"))
        if self.status == KNOWN:
            if self.known_value is None or self.known_value not in self.possible_values:
                raise ValueError("A known hypothesis needs one declared possible value")
            if self.annotation_id is None:
                raise ValueError("A known hypothesis needs a pre-query annotation ID")
        elif self.known_value is not None:
            raise ValueError("Only a known hypothesis may carry a known value")
        if self.status == UNKNOWN and self.annotation_id is not None:
            raise ValueError("An unknown hypothesis cannot carry a known annotation ID")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StateHypothesis":
        fields = ("hypothesis_id", "target_key", "state_variable", "possible_values")
        if not isinstance(raw, Mapping) or any(field not in raw for field in fields):
            raise ValueError("State hypothesis is missing identity or possible values")
        return cls(**{field: raw[field] for field in fields}, status=raw.get("status", UNKNOWN),
                   known_value=raw.get("known_value"), annotation_id=raw.get("annotation_id"),
                   scope=raw.get("scope", "match"),
                   annotation_visibility=raw.get("annotation_visibility", PRIVATE_EVALUATOR))

    def as_dict(self, *, view: str = "evaluator") -> dict[str, Any]:
        if view not in VIEWS:
            raise ValueError("Unknown state hypothesis view")
        hidden_from_agent = view == "agent" and self.annotation_visibility == PRIVATE_EVALUATOR
        return {
            "hypothesis_id": self.hypothesis_id, "target_key": self.target_key,
            "state_variable": self.state_variable, "possible_values": list(self.possible_values),
            "status": UNKNOWN if hidden_from_agent and self.status == KNOWN else self.status,
            "known_value": None if hidden_from_agent else self.known_value,
            "annotation_id": None if hidden_from_agent else self.annotation_id,
            "scope": self.scope, "annotation_visibility": self.annotation_visibility,
        }


@dataclass(frozen=True)
class StateQueryCandidate:
    """A tool route labelled by the unknown it could inspect."""

    candidate_id: str
    target_key: str
    prediction_cutoff: str
    state_variable: str
    source_role: str
    query_kind: str
    cost: float
    endpoint: str | None = None
    action_id: str | None = None

    def __post_init__(self) -> None:
        for field in ("candidate_id", "target_key", "state_variable", "source_role", "query_kind"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "prediction_cutoff", utc(self.prediction_cutoff).isoformat())
        object.__setattr__(self, "cost", _cost(self.cost))
        for field in ("endpoint", "action_id"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _text(value, field))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StateQueryCandidate":
        fields = ("candidate_id", "target_key", "prediction_cutoff", "state_variable",
                  "source_role", "query_kind", "cost")
        if not isinstance(raw, Mapping) or any(field not in raw for field in fields):
            raise ValueError("State query candidate is missing target, state or cost fields")
        return cls(**{field: raw[field] for field in fields}, endpoint=raw.get("endpoint"),
                   action_id=raw.get("action_id"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "target_key": self.target_key,
            "prediction_cutoff": self.prediction_cutoff, "state_variable": self.state_variable,
            "source_role": self.source_role, "query_kind": self.query_kind,
            "cost": self.cost, "endpoint": self.endpoint, "action_id": self.action_id,
        }


class StateQueryBoard:
    """Evaluate route scope against explicitly declared state hypotheses."""

    version = VERSION

    def __init__(self, *, target_key: str, prediction_cutoff: str,
                 hypotheses: Iterable[StateHypothesis | Mapping[str, Any]],
                 candidates: Iterable[StateQueryCandidate | Mapping[str, Any]],
                 registry: PreQueryStateRegistry | None = None):
        self.target_key = _text(target_key, "target_key")
        self.prediction_cutoff = utc(prediction_cutoff).isoformat()
        self.hypotheses = tuple(item if isinstance(item, StateHypothesis) else StateHypothesis.from_mapping(item)
                                for item in hypotheses)
        self.candidates = tuple(item if isinstance(item, StateQueryCandidate) else StateQueryCandidate.from_mapping(item)
                                for item in candidates)
        if len({item.hypothesis_id for item in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("State hypothesis IDs must be unique")
        if len({item.state_variable for item in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("Each state variable needs one explicit hypothesis")
        if len({item.candidate_id for item in self.candidates}) != len(self.candidates):
            raise ValueError("State query candidate IDs must be unique")
        self.registry = registry
        for hypothesis in self.hypotheses:
            if hypothesis.target_key != self.target_key:
                raise ValueError("State hypothesis target differs from board target")
            if hypothesis.status == KNOWN:
                if registry is None:
                    raise ValueError("Known hypotheses require a pre-query state registry")
                annotation = registry.require(hypothesis.annotation_id)
                annotation.validate_for(target_key=self.target_key, prediction_cutoff=self.prediction_cutoff)
                if annotation.state_variable != hypothesis.state_variable or annotation.value != hypothesis.known_value:
                    raise ValueError("Known hypothesis differs from its pre-query annotation")
                if annotation.visibility != hypothesis.annotation_visibility:
                    raise ValueError("Known hypothesis visibility differs from its pre-query annotation")

    def _hypotheses_by_variable(self) -> dict[str, StateHypothesis]:
        result = {}
        for hypothesis in self.hypotheses:
            if hypothesis.state_variable in result:
                raise ValueError("Each state variable needs one explicit hypothesis")
            result[hypothesis.state_variable] = hypothesis
        return result

    @staticmethod
    def _status_for_view(hypothesis: StateHypothesis, view: str) -> str:
        if view not in VIEWS:
            raise ValueError("Unknown state query board view")
        if view == "agent" and hypothesis.annotation_visibility == PRIVATE_EVALUATOR and hypothesis.status == KNOWN:
            return UNKNOWN
        return hypothesis.status

    def assess_candidates(self, *, view: str = "evaluator") -> tuple[dict[str, Any], ...]:
        if view not in VIEWS:
            raise ValueError("Unknown state query board view")
        by_variable = self._hypotheses_by_variable()
        rows = []
        for candidate in self.candidates:
            if candidate.target_key != self.target_key:
                status = EXCLUDED_TARGET
                reason = "candidate_target_differs"
                hypothesis = None
            elif candidate.prediction_cutoff != self.prediction_cutoff:
                status = EXCLUDED_CUTOFF
                reason = "candidate_cutoff_differs"
                hypothesis = None
            else:
                hypothesis = by_variable.get(candidate.state_variable)
                if hypothesis is None:
                    status, reason = EXCLUDED_NO_HYPOTHESIS, "state_variable_not_declared"
                else:
                    effective_status = self._status_for_view(hypothesis, view)
                    if effective_status == UNKNOWN:
                        status = ACTIONABLE_UNKNOWN
                        reason = ("state_not_visible_to_agent" if view == "agent" and hypothesis.status == KNOWN
                                  else "state_is_unresolved")
                    elif effective_status == CONTESTED:
                        status, reason = STATE_CONTESTED, "state_has_conflicting_evidence"
                    else:
                        status, reason = STATE_ALREADY_KNOWN, "state_already_annotated"
            rows.append({
                "candidate_id": candidate.candidate_id, "state_variable": candidate.state_variable,
                "status": status, "reason": reason, "cost": candidate.cost,
                "action_id": candidate.action_id, "expected_value": None,
                "value_status": "UNMEASURED_UNTIL_RETURN_MODEL_CALIBRATED",
                "hypothesis_id": None if hypothesis is None else hypothesis.hypothesis_id,
            })
        return tuple(rows)

    def actionable_candidates(self, *, view: str = "evaluator") -> tuple[StateQueryCandidate, ...]:
        allowed = {row["candidate_id"] for row in self.assess_candidates(view=view)
                   if row["status"] in {ACTIONABLE_UNKNOWN, STATE_CONTESTED}}
        return tuple(candidate for candidate in self.candidates if candidate.candidate_id in allowed)

    def with_annotation(self, annotation: PreQueryStateAnnotation) -> "StateQueryBoard":
        annotation.validate_for(target_key=self.target_key, prediction_cutoff=self.prediction_cutoff)
        by_variable = self._hypotheses_by_variable()
        hypothesis = by_variable.get(annotation.state_variable)
        if hypothesis is None:
            raise ValueError("Annotation state variable is not declared on this board")
        if annotation.value not in hypothesis.possible_values:
            raise ValueError("Annotation value is outside the hypothesis support")
        if hypothesis.status == KNOWN and hypothesis.known_value != annotation.value:
            raise ValueError("A known hypothesis conflicts with the new annotation")
        replacement = replace(hypothesis, status=KNOWN, known_value=annotation.value,
                              annotation_id=annotation.annotation_id,
                              annotation_visibility=annotation.visibility)
        hypotheses = tuple(replacement if item.hypothesis_id == hypothesis.hypothesis_id else item
                           for item in self.hypotheses)
        return StateQueryBoard(target_key=self.target_key, prediction_cutoff=self.prediction_cutoff,
                               hypotheses=hypotheses, candidates=self.candidates, registry=self.registry)

    def manifest(self, *, view: str = "evaluator") -> dict[str, Any]:
        if view not in VIEWS:
            raise ValueError("Unknown state query board view")
        return {
            "schema_version": VERSION, "target_key": self.target_key,
            "prediction_cutoff": self.prediction_cutoff,
            "view": view,
            "hypotheses": [item.as_dict(view=view) for item in self.hypotheses],
            "candidates": [item.as_dict() for item in self.candidates],
            "candidate_assessments": list(self.assess_candidates(view=view)),
            "actionable_candidate_ids": [item.candidate_id for item in self.actionable_candidates(view=view)],
            "automatic_derivation": False,
            "value_status": "UNMEASURED_UNTIL_RETURN_MODEL_CALIBRATED",
            "agent_selection_remains_policy_decision": True,
        }
