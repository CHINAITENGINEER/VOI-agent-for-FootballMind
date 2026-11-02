"""Conservative, state-bound contextual value-of-information estimates.

The candidate generator only proposes a useful order.  This module is the
numeric boundary: it estimates the value of a registered action in the exact
actor-visible context, binds every estimate to the current acquisition state,
and exposes an interval-aware adapter to the existing selector.

The estimand is deliberately simple and auditable.  For one historical
episode/dependency cluster, ``Y`` is the observed signed risk reduction when a
usable result arrived, and zero when the request was missing, late, or
unusable.  Thus availability is included in the outcome instead of being
multiplied into a point estimate.  ``Y`` is bounded in [-1, 1], and a
Hoeffding interval is computed after clustering independent dependencies.  A
lower bound above cost authorizes acquisition; an interval crossing cost is
withheld and yields DEFER; an upper bound at or below cost can contribute to a
bounded registry stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from .autonomous_acquisition_step import select_acquisition_action
from .voi_evidence_bridge import VoIEvidenceLink


VERSION = "state-bound-context-value-v1"
SUPPORTED_INTERVAL = "supported_interval"
INSUFFICIENT_SUPPORT = "insufficient_support"
NO_MATCH = "no_matching_history"
CONTEXT_UNKNOWN = "context_incomplete"
ROUTE_MISMATCH = "route_mismatch"
_FORBIDDEN_CONTEXT_KEYS = frozenset({
    "result", "outcome", "score", "goals", "label", "loss", "decision_loss",
    "realized_gain", "realized_loss", "evaluator", "sidecar", "post_state",
    "after_state", "hidden_return", "settlement_status",
})


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include timezone")
    return parsed.astimezone(timezone.utc)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _context_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().casefold() in {"", "unknown", "unavailable", "missing"}:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return None


def normalize_observed_context(context: Mapping[str, Any], context_keys: tuple[str, ...]) -> tuple[tuple[str, str], ...] | None:
    """Return a complete actor-visible context; unknown fields are withheld."""
    if not isinstance(context, Mapping) or not context_keys:
        return None
    result: list[tuple[str, str]] = []
    for key in context_keys:
        value = _context_value(context.get(key))
        if value is None:
            return None
        result.append((key, value))
    return tuple(result)


def _context_has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_CONTEXT_KEYS or _context_has_forbidden_key(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_context_has_forbidden_key(child) for child in value)
    return False


def context_fingerprint(context: Mapping[str, Any], context_keys: tuple[str, ...]) -> str | None:
    normalized = normalize_observed_context(context, context_keys)
    return None if normalized is None else _digest(dict(normalized))


@dataclass(frozen=True)
class ContextualValueInterval:
    candidate_id: str
    context_fingerprint: str | None
    support_status: str
    reason: str
    dependency_count: int
    attempt_count: int
    mean_value: float | None
    lower_bound: float | None
    upper_bound: float | None
    cost: float | None
    cost_unit: str | None
    confidence: float
    family_size: int
    route: dict[str, str | None]

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        if self.dependency_count < 0 or self.attempt_count < 0 or self.family_size < 1:
            raise ValueError("support counts must be nonnegative and family_size positive")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        for field in ("mean_value", "lower_bound", "upper_bound", "cost"):
            value = getattr(self, field)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{field} must be finite")
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound cannot exceed upper_bound")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "candidate_id": self.candidate_id,
            "context_fingerprint": self.context_fingerprint,
            "support_status": self.support_status,
            "reason": self.reason,
            "dependency_count": self.dependency_count,
            "attempt_count": self.attempt_count,
            "mean_value": self.mean_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "cost": self.cost,
            "cost_unit": self.cost_unit,
            "confidence": self.confidence,
            "family_size": self.family_size,
            "route": dict(self.route),
        }


class StateBoundContextValueEstimator:
    """Estimate candidate value only within an exact context and episode cutoff."""

    def __init__(
        self,
        links: Iterable[VoIEvidenceLink],
        context_by_link_id: Mapping[str, Mapping[str, Any]],
        routes: Mapping[str, Mapping[str, Any]],
        *,
        context_keys: tuple[str, ...] = ("qualification_state", "tactical_layout"),
        min_support: int = 5,
        confidence: float = 0.9,
        calibration_as_of: str | None = None,
        family_size: int | None = None,
    ) -> None:
        if not context_keys or len(set(context_keys)) != len(context_keys):
            raise ValueError("context_keys must be a nonempty tuple of unique fields")
        if min_support < 1:
            raise ValueError("min_support must be positive")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if not isinstance(context_by_link_id, Mapping) or not isinstance(routes, Mapping):
            raise ValueError("context_by_link_id and routes must be mappings")
        self.context_keys = tuple(_text(key, "context_key") for key in context_keys)
        self.min_support = int(min_support)
        self.confidence = float(confidence)
        self.calibration_as_of = None if calibration_as_of is None else _time(calibration_as_of, "calibration_as_of")
        self.links = tuple(links)
        if any(not isinstance(link, VoIEvidenceLink) for link in self.links):
            raise ValueError("links must contain VoIEvidenceLink records")
        self.context_by_link_id = {}
        for key, value in context_by_link_id.items():
            if not isinstance(value, Mapping):
                continue
            if _context_has_forbidden_key(value):
                raise ValueError("context history contains evaluator-only fields")
            self.context_by_link_id[str(key)] = dict(value)
        self.routes = {str(key): dict(value) for key, value in routes.items()
                       if isinstance(value, Mapping)}
        self.family_size = max(1, int(family_size or len(self.routes) or 1))
        self.estimator_snapshot = _digest({
            "version": VERSION,
            "context_keys": self.context_keys,
            "min_support": self.min_support,
            "confidence": self.confidence,
            "calibration_as_of": None if self.calibration_as_of is None else self.calibration_as_of.isoformat(),
            "routes": self.routes,
            "links": [link.as_dict() for link in self.links],
            "contexts": self.context_by_link_id,
        })

    def _route(self, candidate_id: str) -> dict[str, Any] | None:
        route = self.routes.get(candidate_id)
        if route is None:
            return None
        return {
            "source_id": None if route.get("source_id") is None else _text(route.get("source_id"), "route.source_id"),
            "source_version": None if route.get("source_version") is None else _text(route.get("source_version"), "route.source_version"),
            "action_id": None if route.get("action_id") is None else _text(route.get("action_id"), "route.action_id"),
            "cost": None if route.get("cost") is None else _finite(route.get("cost"), "route.cost"),
            "unit": None if route.get("unit") is None else _text(route.get("unit"), "route.unit"),
        }

    def _candidate_links(self, candidate_id: str, context: tuple[tuple[str, str], ...], *, cutoff: datetime,
                         target_key: str | None) -> tuple[VoIEvidenceLink, ...]:
        route = self._route(candidate_id)
        if route is None:
            return ()
        wanted = dict(context)
        matches: list[VoIEvidenceLink] = []
        for link in self.links:
            if link.candidate_id != candidate_id or target_key is not None and link.target_key == target_key:
                continue
            if route["source_id"] is not None and link.source_id != route["source_id"]:
                continue
            if route["source_version"] is not None and link.source_version != route["source_version"]:
                continue
            if route["action_id"] is not None and link.action_id != route["action_id"]:
                continue
            if link.hidden_result_included:
                continue
            link_cutoff = _time(link.prediction_cutoff, "link.prediction_cutoff")
            if link_cutoff > cutoff:
                continue
            if self.calibration_as_of is not None:
                if link_cutoff > self.calibration_as_of:
                    continue
                if link.requested_at is None or _time(link.requested_at, "link.requested_at") > self.calibration_as_of:
                    continue
            observed = self.context_by_link_id.get(link.link_id)
            normalized = normalize_observed_context(observed or {}, self.context_keys)
            if normalized is None or dict(normalized) != wanted:
                continue
            matches.append(link)
        return tuple(matches)

    def estimate_candidate(self, candidate_id: str, context: Mapping[str, Any], *, cutoff: str,
                           target_key: str | None = None, cost: float | None = None,
                           cost_unit: str | None = None) -> ContextualValueInterval:
        candidate_id = _text(candidate_id, "candidate_id")
        cutoff_value = _time(cutoff, "cutoff")
        normalized = normalize_observed_context(context, self.context_keys)
        fp = None if normalized is None else _digest(dict(normalized))
        route = self._route(candidate_id)
        route_view = {} if route is None else {key: route.get(key) for key in ("source_id", "source_version", "action_id")}
        if normalized is None:
            return ContextualValueInterval(candidate_id, fp, CONTEXT_UNKNOWN, "current_context_incomplete", 0, 0,
                                           None, None, None, cost, cost_unit, self.confidence, self.family_size, route_view)
        if route is None:
            return ContextualValueInterval(candidate_id, fp, ROUTE_MISMATCH, "candidate_route_not_registered", 0, 0,
                                           None, None, None, cost, cost_unit, self.confidence, self.family_size, route_view)
        if cost is None:
            cost = route.get("cost")
        if cost is not None:
            cost = _finite(cost, "cost")
        if cost_unit is None:
            cost_unit = route.get("unit")
        matches = self._candidate_links(candidate_id, normalized, cutoff=cutoff_value, target_key=target_key)
        if not matches:
            return ContextualValueInterval(candidate_id, fp, NO_MATCH, "no_exact_context_history", 0, 0,
                                           None, None, None, cost, cost_unit, self.confidence, self.family_size, route_view)
        # A dependency identifies one independent source event.  Scope it by
        # episode and target so retries/candidates from another match do not
        # inflate support.
        by_dependency: dict[tuple[str, str, str], list[VoIEvidenceLink]] = {}
        for link in matches:
            dependency = link.dependency_key or link.request_id
            key = (link.episode_id, link.target_key, dependency)
            by_dependency.setdefault(key, []).append(link)
        values: list[float] = []
        for rows in by_dependency.values():
            available = [row.observed_risk_reduction for row in rows
                         if row.status == "linked_available" and row.observed_risk_reduction is not None]
            if available:
                value = sum(_finite(item, "observed_risk_reduction") for item in available) / len(available)
            else:
                value = 0.0
            if not -1.0 <= value <= 1.0:
                continue
            values.append(value)
        n = len(values)
        attempts = len(matches)
        if n < self.min_support:
            return ContextualValueInterval(candidate_id, fp, INSUFFICIENT_SUPPORT,
                                           f"independent_support_below_{self.min_support}", n, attempts,
                                           None if not values else sum(values) / n, None, None, cost, cost_unit,
                                           self.confidence, self.family_size, route_view)
        mean = sum(values) / n
        # Hoeffding for Y in [-1, 1], with a family-wise two-sided bound over
        # the registered candidate/context cells.
        delta_cell = (1.0 - self.confidence) / max(1, self.family_size)
        radius = math.sqrt(2.0 * math.log(2.0 / delta_cell) / n)
        lower = max(-1.0, mean - radius)
        upper = min(1.0, mean + radius)
        return ContextualValueInterval(candidate_id, fp, SUPPORTED_INTERVAL,
                                       "exact_context_hoeffding_interval", n, attempts, mean, lower, upper,
                                       cost, cost_unit, self.confidence, self.family_size, route_view)

    def estimate_state(self, state: Mapping[str, Any], context: Mapping[str, Any], *,
                       loss_unit: str, target_key: str | None = None) -> dict[str, ContextualValueInterval]:
        if not isinstance(state, Mapping):
            raise ValueError("state must be a mapping")
        cutoff = _time(state.get("prediction_cutoff"), "state.prediction_cutoff")
        intervals: dict[str, ContextualValueInterval] = {}
        for row in state.get("candidates", []):
            cid = _text(row.get("candidate_id"), "candidate_id")
            utility = row.get("utility_cost")
            cost = None if utility is None else _finite(utility.get("amount"), "utility_cost.amount")
            unit = None if utility is None else _text(utility.get("unit"), "utility_cost.unit")
            if unit is not None and unit != loss_unit:
                intervals[cid] = self.estimate_candidate(cid, context, cutoff=cutoff.isoformat(),
                                                          target_key=target_key, cost=cost, cost_unit=unit)
                continue
            intervals[cid] = self.estimate_candidate(cid, context, cutoff=cutoff.isoformat(),
                                                      target_key=target_key, cost=cost, cost_unit=unit)
        return intervals

    def gate_estimates(self, state: Mapping[str, Any], context: Mapping[str, Any], *,
                       loss_unit: str, target_key: str | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, ContextualValueInterval]]:
        intervals = self.estimate_state(state, context, loss_unit=loss_unit, target_key=target_key)
        estimates: dict[str, dict[str, Any]] = {}
        for cid, interval in intervals.items():
            if interval.support_status != SUPPORTED_INTERVAL or interval.lower_bound is None or interval.upper_bound is None:
                continue
            if interval.cost is None or interval.cost_unit != loss_unit:
                continue
            expected: float | None = None
            mode = ""
            if interval.lower_bound > interval.cost:
                expected, mode = interval.lower_bound, "lower_bound_above_cost"
            elif interval.upper_bound <= interval.cost:
                expected, mode = max(0.0, interval.upper_bound), "upper_bound_at_or_below_cost"
            if expected is None:
                continue
            estimates[cid] = {
                "candidate_id": cid,
                "state_digest": state.get("state_digest"),
                "expected_value": expected,
                "unit": loss_unit,
                "source": f"{VERSION}:{self.estimator_snapshot}",
                "training_cutoff": (self.calibration_as_of or _time(state["prediction_cutoff"], "prediction_cutoff")).isoformat(),
                "available_at": (self.calibration_as_of or _time(state["prediction_cutoff"], "prediction_cutoff")).isoformat(),
                "interval_lower": interval.lower_bound,
                "interval_upper": interval.upper_bound,
                "interval_decision": mode,
                "estimator_snapshot": self.estimator_snapshot,
            }
        return estimates, intervals

    def select(self, state: Mapping[str, Any], context: Mapping[str, Any], *, loss_unit: str,
               remaining_budget: Mapping[str, float] | None = None,
               target_key: str | None = None) -> dict[str, Any]:
        estimates, intervals = self.gate_estimates(state, context, loss_unit=loss_unit, target_key=target_key)
        selection = select_acquisition_action(state, loss_unit=loss_unit, estimates=estimates,
                                              remaining_budget=remaining_budget)
        return {
            "schema_version": VERSION,
            "numeric_voi_calibrated": True,
            "estimand": "availability_weighted_signed_risk_reduction",
            "state_digest": state.get("state_digest"),
            "estimator_snapshot": self.estimator_snapshot,
            "intervals": {cid: item.as_dict() for cid, item in intervals.items()},
            "gate_estimates": estimates,
            "selection": selection,
        }


def select_prepared_contextual_action(
    prepared: Mapping[str, Any], estimator: StateBoundContextValueEstimator, *, loss_unit: str,
    remaining_budget: Mapping[str, float] | None = None, target_key: str | None = None,
) -> dict[str, Any]:
    """Run the estimator from a ``prepare_contextual_acquisition`` bundle.

    This keeps the actor-visible context selected by the platform adapter in
    the call path, so a caller does not accidentally replace it with a
    hand-written question or a post-return observation.
    """
    if not isinstance(prepared, Mapping):
        raise ValueError("prepared must be a mapping")
    state = prepared.get("acquisition_state")
    binding = prepared.get("context_binding")
    if not isinstance(state, Mapping) or not isinstance(binding, Mapping):
        raise ValueError("prepared bundle lacks acquisition_state or context_binding")
    context = binding.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("prepared context_binding.context must be a mapping")
    return estimator.select(state, context, loss_unit=loss_unit,
                           remaining_budget=remaining_budget, target_key=target_key)


__all__ = [
    "VERSION", "SUPPORTED_INTERVAL", "INSUFFICIENT_SUPPORT", "NO_MATCH", "CONTEXT_UNKNOWN", "ROUTE_MISMATCH",
    "ContextualValueInterval", "StateBoundContextValueEstimator", "normalize_observed_context", "context_fingerprint",
    "select_prepared_contextual_action",
]
