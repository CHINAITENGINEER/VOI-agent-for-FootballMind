"""Conservatively calibrate candidate information value from past query returns.

This module calibrates operational query value in a declared bounded risk unit.
It never turns source returns into match-result probabilities and never treats an
unsupported candidate as zero value. A candidate enters ``QueryValueGate`` only
when future-safe, source-version-matched, independent support is sufficient.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Iterable, Mapping

from .query_value_contract import QueryValueAssessment
from .types import utc
from .voi_evidence_bridge import VoIEvidenceLink


VERSION = "adaptive-voi-calibration-v1"
SUPPORTED = "SUPPORTED_CONSERVATIVE_ESTIMATE"
UNSUPPORTED = frozenset({
    "NO_SOURCE_MATCH",
    "NO_FUTURE_SAFE_TRAINING_DATA",
    "INSUFFICIENT_INDEPENDENT_SUPPORT",
    "OBSERVED_VALUE_OUT_OF_RANGE",
})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _cost(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("cost must be finite and nonnegative")
    return float(value)


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("confidence must be finite")
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("confidence must be between zero and one")
    return value


def _minimum(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("min_independent_dependencies must be a positive integer")
    return value


def _lower_bound(mean: float, count: int, confidence: float) -> float:
    """One-sided Hoeffding lower bound for a value declared in [0, 1]."""
    radius = math.sqrt(math.log(2.0 / confidence) / (2.0 * count))
    return max(0.0, mean - radius)


@dataclass(frozen=True)
class AdaptiveVoIAssessment:
    candidate_id: str
    source_id: str
    source_version: str
    cost: float
    attempts: int
    available_attempts: int
    independent_dependency_count: int
    value_sample_count: int
    available_rate: float | None
    available_lower_bound: float | None
    mean_observed_risk_reduction: float | None
    risk_reduction_lower_bound: float | None
    expected_information_value: float | None
    support_status: str
    support_scope: str
    reason: str

    def __post_init__(self) -> None:
        for field in ("candidate_id", "source_id", "source_version", "support_status", "support_scope", "reason"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "cost", _cost(self.cost))
        for field in ("attempts", "available_attempts", "independent_dependency_count", "value_sample_count"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.available_attempts > self.attempts:
            raise ValueError("available_attempts cannot exceed attempts")
        for field in ("available_rate", "available_lower_bound", "mean_observed_risk_reduction",
                      "risk_reduction_lower_bound", "expected_information_value"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"{field} must be finite and nonnegative when present")
        if self.support_status == SUPPORTED and self.expected_information_value is None:
            raise ValueError("A supported assessment needs expected_information_value")

    @property
    def calibrated(self) -> bool:
        return self.support_status == SUPPORTED

    def as_query_assessment(self) -> QueryValueAssessment:
        if self.calibrated:
            return QueryValueAssessment.modeled(
                self.candidate_id,
                expected_value=float(self.expected_information_value),
                cost=self.cost,
                reason="future_safe_source_version_calibration_with_conservative_lower_bound",
            )
        return QueryValueAssessment.unmeasured(
            self.candidate_id,
            cost=self.cost,
            reason=f"adaptive_voi_calibration:{self.support_status.lower()}",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "cost": self.cost,
            "attempts": self.attempts,
            "available_attempts": self.available_attempts,
            "independent_dependency_count": self.independent_dependency_count,
            "value_sample_count": self.value_sample_count,
            "available_rate": self.available_rate,
            "available_lower_bound": self.available_lower_bound,
            "mean_observed_risk_reduction": self.mean_observed_risk_reduction,
            "risk_reduction_lower_bound": self.risk_reduction_lower_bound,
            "expected_information_value": self.expected_information_value,
            "support_status": self.support_status,
            "support_scope": self.support_scope,
            "reason": self.reason,
        }


def calibrate_candidate_value(
    candidate_id: str,
    *,
    source_id: str,
    source_version: str,
    cost: float,
    links: Iterable[VoIEvidenceLink],
    as_of: str | datetime,
    min_independent_dependencies: int = 5,
    confidence: float = 0.10,
) -> AdaptiveVoIAssessment:
    """Estimate one candidate's usable risk reduction from future-safe links.

    ``observed_risk_reduction`` must already be normalized to the declared
    bounded risk unit [0, 1] (for example a Brier-risk unit). Repeated links
    sharing one dependency form one value cluster, so retries do not inflate
    independent support. Requests after ``as_of`` are excluded to prevent
    temporal leakage.
    """
    candidate_id = _text(candidate_id, "candidate_id")
    source_id = _text(source_id, "source_id")
    source_version = _text(source_version, "source_version")
    cost = _cost(cost)
    cutoff = utc(as_of)
    minimum = _minimum(min_independent_dependencies)
    confidence = _confidence(confidence)
    matched = []
    for link in links:
        if not isinstance(link, VoIEvidenceLink):
            raise ValueError("links must contain VoIEvidenceLink values")
        if link.source_id != source_id or link.source_version != source_version:
            continue
        if link.action_id == "STOP" or link.status == "stop_recorded":
            continue
        if link.requested_at is None or utc(link.requested_at) > cutoff:
            continue
        matched.append(link)

    if not matched:
        return AdaptiveVoIAssessment(
            candidate_id, source_id, source_version, cost, 0, 0, 0, 0,
            None, None, None, None, None, "NO_SOURCE_MATCH",
            "future_safe_source_version_bounded_risk_unit", "no_matching_future_safe_links",
        )

    dependency_groups: dict[str, list[VoIEvidenceLink]] = defaultdict(list)
    for link in matched:
        dependency_groups[link.dependency_key or f"link:{link.link_id}"].append(link)
    independent_count = len(dependency_groups)
    available = [link for link in matched if link.status == "linked_available"]
    value_groups: list[float] = []
    out_of_range = False
    for group in dependency_groups.values():
        values = [link.observed_risk_reduction for link in group
                  if link.status == "linked_available" and link.observed_risk_reduction is not None]
        if not values:
            continue
        if any(value < 0.0 or value > 1.0 for value in values):
            out_of_range = True
            continue
        value_groups.append(math.fsum(values) / len(values))

    if independent_count < minimum:
        status = "INSUFFICIENT_INDEPENDENT_SUPPORT"
    elif out_of_range:
        status = "OBSERVED_VALUE_OUT_OF_RANGE"
    elif len(value_groups) < minimum:
        status = "INSUFFICIENT_INDEPENDENT_SUPPORT"
    else:
        status = SUPPORTED

    attempts = len(matched)
    available_attempts = len(available)
    available_rate = available_attempts / attempts
    available_lcb = _lower_bound(available_rate, attempts, confidence)
    mean_value = math.fsum(value_groups) / len(value_groups) if value_groups else None
    value_lcb = _lower_bound(mean_value, len(value_groups), confidence) if mean_value is not None else None
    expected = available_lcb * value_lcb if status == SUPPORTED and value_lcb is not None else None
    reason = (
        "future_safe_independent_support_and_bounded_value"
        if status == SUPPORTED else status.lower()
    )
    return AdaptiveVoIAssessment(
        candidate_id, source_id, source_version, cost, attempts, available_attempts,
        independent_count, len(value_groups), available_rate, available_lcb,
        mean_value, value_lcb, expected, status,
        "future_safe_source_version_bounded_risk_unit", reason,
    )


def calibrate_candidate_values(
    routes: Mapping[str, Mapping[str, Any]],
    links: Iterable[VoIEvidenceLink],
    *,
    as_of: str | datetime,
    min_independent_dependencies: int = 5,
    confidence: float = 0.10,
) -> tuple[AdaptiveVoIAssessment, ...]:
    """Calibrate a set of current candidate routes using one frozen history."""
    if not isinstance(routes, Mapping) or not routes:
        raise ValueError("routes must be a nonempty mapping")
    materialized = tuple(links)
    result = []
    for candidate_id, route in routes.items():
        if not isinstance(route, Mapping):
            raise ValueError("Each route must be a mapping")
        result.append(calibrate_candidate_value(
            candidate_id,
            source_id=route.get("source_id"),
            source_version=route.get("source_version"),
            cost=route.get("cost"),
            links=materialized,
            as_of=as_of,
            min_independent_dependencies=min_independent_dependencies,
            confidence=confidence,
        ))
    return tuple(result)


__all__ = [
    "VERSION", "SUPPORTED", "UNSUPPORTED", "AdaptiveVoIAssessment",
    "calibrate_candidate_value", "calibrate_candidate_values",
]
