"""Context-conditioned empirical VoI with an exact-context transfer boundary.

The same source can have different value in different pre-query match contexts.
This module keeps those contexts separate. A global source estimate is exposed
for diagnosis only; it never silently becomes the value for a new context.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Iterable, Mapping

from .adaptive_voi_calibration import AdaptiveVoIAssessment, SUPPORTED, calibrate_candidate_value
from .query_value_contract import QueryValueAssessment
from .voi_evidence_bridge import VoIEvidenceLink


VERSION = "contextual-adaptive-voi-v1"
CONTEXT_INSUFFICIENT = "CONTEXT_INSUFFICIENT_SUPPORT"
_FORBIDDEN_CONTEXT_KEYS = frozenset({
    "result", "outcome", "probability", "sampled_result", "hidden_result", "realized_loss",
})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _json_value(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{name} must contain scalar JSON values only")


def normalize_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize public pre-query context and reject evaluator-only fields."""
    if not isinstance(context, Mapping) or not context:
        raise ValueError("context must be a nonempty mapping")
    normalized = {}
    for key, value in context.items():
        key = _text(key, "context key")
        if key.lower() in _FORBIDDEN_CONTEXT_KEYS:
            raise ValueError("context contains an evaluator-only field")
        normalized[key] = _json_value(value, f"context[{key}]")
    return {key: normalized[key] for key in sorted(normalized)}


def context_fingerprint(context: Mapping[str, Any]) -> str:
    normalized = normalize_context(context)
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "ctx-" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ContextualVoIAssessment:
    candidate_id: str
    source_id: str
    source_version: str
    context: Mapping[str, Any]
    context_key: str
    exact: AdaptiveVoIAssessment
    global_reference: AdaptiveVoIAssessment
    support_status: str
    transfer_policy: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "source_version", _text(self.source_version, "source_version"))
        normalized = normalize_context(self.context)
        object.__setattr__(self, "context", normalized)
        if self.context_key != context_fingerprint(normalized):
            raise ValueError("context_key does not match context")
        if not isinstance(self.exact, AdaptiveVoIAssessment) or not isinstance(self.global_reference, AdaptiveVoIAssessment):
            raise ValueError("exact and global_reference must be AdaptiveVoIAssessment values")
        object.__setattr__(self, "support_status", _text(self.support_status, "support_status"))
        object.__setattr__(self, "transfer_policy", _text(self.transfer_policy, "transfer_policy"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))

    @property
    def calibrated(self) -> bool:
        return self.support_status == SUPPORTED

    def as_query_assessment(self) -> QueryValueAssessment:
        if self.calibrated:
            # The exact context estimate, rather than the global diagnostic, is
            # the only value allowed to enter the numeric gate.
            return self.exact.as_query_assessment()
        return QueryValueAssessment.unmeasured(
            self.candidate_id,
            cost=self.exact.cost,
            reason="contextual_adaptive_voi:exact_context_support_required",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "context": dict(self.context),
            "context_key": self.context_key,
            "exact_context_estimate": self.exact.as_dict(),
            "global_reference_estimate": self.global_reference.as_dict(),
            "support_status": self.support_status,
            "transfer_policy": self.transfer_policy,
            "reason": self.reason,
        }


def calibrate_contextual_candidate_value(
    candidate_id: str,
    *,
    source_id: str,
    source_version: str,
    cost: float,
    context: Mapping[str, Any],
    links: Iterable[VoIEvidenceLink],
    context_by_link_id: Mapping[str, Mapping[str, Any]],
    as_of: str | datetime,
    min_independent_dependencies: int = 5,
    confidence: float = 0.10,
) -> ContextualVoIAssessment:
    """Calibrate from exact context; expose global source value diagnostically."""
    normalized_context = normalize_context(context)
    key = context_fingerprint(normalized_context)
    materialized = tuple(links)
    if not isinstance(context_by_link_id, Mapping):
        raise ValueError("context_by_link_id must be a mapping")
    selected = []
    for link in materialized:
        if not isinstance(link, VoIEvidenceLink):
            raise ValueError("links must contain VoIEvidenceLink values")
        declared = context_by_link_id.get(link.link_id)
        if declared is None:
            continue
        if context_fingerprint(declared) == key:
            selected.append(link)
    exact = calibrate_candidate_value(
        candidate_id, source_id=source_id, source_version=source_version, cost=cost,
        links=selected, as_of=as_of, min_independent_dependencies=min_independent_dependencies,
        confidence=confidence,
    )
    global_reference = calibrate_candidate_value(
        candidate_id, source_id=source_id, source_version=source_version, cost=cost,
        links=materialized, as_of=as_of, min_independent_dependencies=min_independent_dependencies,
        confidence=confidence,
    )
    status = SUPPORTED if exact.calibrated else CONTEXT_INSUFFICIENT
    reason = "exact_context_support" if exact.calibrated else "global_reference_not_transferred_across_contexts"
    return ContextualVoIAssessment(
        candidate_id=candidate_id, source_id=source_id, source_version=source_version,
        context=normalized_context, context_key=key, exact=exact, global_reference=global_reference,
        support_status=status, transfer_policy="exact_context_only", reason=reason,
    )


def calibrate_contextual_candidate_values(
    routes: Mapping[str, Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    links: Iterable[VoIEvidenceLink],
    context_by_link_id: Mapping[str, Mapping[str, Any]],
    as_of: str | datetime,
    min_independent_dependencies: int = 5,
    confidence: float = 0.10,
) -> tuple[ContextualVoIAssessment, ...]:
    if not isinstance(routes, Mapping) or not routes:
        raise ValueError("routes must be a nonempty mapping")
    materialized = tuple(links)
    return tuple(
        calibrate_contextual_candidate_value(
            candidate_id,
            source_id=route.get("source_id"), source_version=route.get("source_version"),
            cost=route.get("cost"), context=context, links=materialized,
            context_by_link_id=context_by_link_id, as_of=as_of,
            min_independent_dependencies=min_independent_dependencies, confidence=confidence,
        )
        for candidate_id, route in routes.items()
        if isinstance(route, Mapping)
    )


__all__ = [
    "VERSION", "CONTEXT_INSUFFICIENT", "ContextualVoIAssessment",
    "normalize_context", "context_fingerprint", "calibrate_contextual_candidate_value",
    "calibrate_contextual_candidate_values",
]
