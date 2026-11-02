"""Evaluator-only observations for later empirical VOI calibration.

This module keeps the actor's public context separate from post-action loss
measurements.  A completed execution receipt can be paired with an independent
evaluator's before/after loss, then converted to the existing
``VoIEvidenceLink`` contract used by adaptive calibration.  Provisional actor
judgements remain visible as diagnostics but never become numeric training
support.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from .adaptive_voi_calibration import AdaptiveVoIAssessment, calibrate_candidate_value
from .execution_receipt_bridge import COMPLETED, _verify_receipt
from .voi_evidence_bridge import VoIEvidenceLink

VERSION = "round138-empirical-voi-observation-v1"
INDEPENDENT_GOLD = "INDEPENDENT_EVALUATOR_GOLD"
ACTOR_PROVISIONAL = "ACTOR_PROVISIONAL_NOT_GOLD"
UNAVAILABLE = "UNAVAILABLE"
_LABELS = {INDEPENDENT_GOLD, ACTOR_PROVISIONAL, UNAVAILABLE}


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


def _loss(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be finite in [0,1]")
    return float(value)


def build_empirical_voi_observation(receipt: Mapping[str, Any], *, evaluator_id: str,
                                    label_status: str, loss_unit: str,
                                    decision_loss_before: Any = None,
                                    decision_loss_after: Any = None,
                                    observed_at: str | datetime,
                                    source_id: str, source_version: str,
                                    dependency_key: str,
                                    candidate_id: str | None = None) -> dict[str, Any]:
    """Bind evaluator-only before/after loss to one completed receipt."""
    _verify_receipt(receipt)
    evaluator = _text(evaluator_id, "evaluator_id")
    status = _text(label_status, "label_status")
    if status not in _LABELS:
        raise ValueError("unsupported evaluator label status")
    unit = _text(loss_unit, "loss_unit")
    source = _text(source_id, "source_id")
    version = _text(source_version, "source_version")
    dependency = _text(dependency_key, "dependency_key")
    observed = _time(observed_at, "observed_at")
    accepted = _time(receipt.get("accepted_at"), "accepted_at")
    if observed < accepted:
        raise ValueError("evaluator observation precedes action acceptance")
    cid = _text(candidate_id or receipt.get("candidate_id"), "candidate_id")
    before = after = reduction = None
    eligible = False
    reason = "not_independent_gold"
    if status == INDEPENDENT_GOLD:
        before = _loss(decision_loss_before, "decision_loss_before")
        after = _loss(decision_loss_after, "decision_loss_after")
        if receipt.get("status") != COMPLETED:
            reason = "receipt_not_completed"
        elif unit not in {"decision_loss", "brier", "log_loss"}:
            raise ValueError("unsupported loss unit")
        else:
            reduction = before - after
            eligible = True
            reason = "independent_gold_completed_receipt"
    body = {
        "schema_version": VERSION, "receipt_digest": receipt["receipt_digest"],
        "receipt_id": receipt["receipt_id"], "candidate_id": cid,
        "evaluator_id": evaluator, "label_status": status, "loss_unit": unit,
        "decision_loss_before": before, "decision_loss_after": after,
        "observed_risk_reduction": reduction, "observed_at": observed.isoformat().replace("+00:00", "Z"),
        "prediction_cutoff": receipt["prediction_cutoff"],
        "source_id": source, "source_version": version, "dependency_key": dependency,
        "eligible_for_numeric_calibration": eligible, "reason": reason,
    }
    body["observation_digest"] = _digest(body)
    return body


def to_voi_link(observation: Mapping[str, Any]) -> VoIEvidenceLink:
    """Project one independent observation into the existing calibration link."""
    if not isinstance(observation, Mapping) or observation.get("schema_version") != VERSION:
        raise ValueError("invalid empirical observation")
    body = {key: value for key, value in observation.items() if key != "observation_digest"}
    if observation.get("observation_digest") != _digest(body):
        raise ValueError("empirical observation digest mismatch")
    derived_eligible = (
        observation.get("label_status") == INDEPENDENT_GOLD
        and observation.get("reason") == "independent_gold_completed_receipt"
        and type(observation.get("observed_risk_reduction")) in (int, float)
        and math.isfinite(float(observation["observed_risk_reduction"]))
    )
    if observation.get("eligible_for_numeric_calibration") is not derived_eligible:
        raise ValueError("empirical calibration eligibility mismatch")
    eligible = derived_eligible
    status = "linked_available" if eligible else "linked_unusable"
    link_body = {
        "observation_digest": observation["observation_digest"],
        "receipt_digest": observation["receipt_digest"],
        "candidate_id": observation["candidate_id"],
        "source_id": observation["source_id"], "source_version": observation["source_version"],
        "status": status,
    }
    link_id = _digest(link_body)
    return VoIEvidenceLink(
        link_id=link_id, trace_id=observation["receipt_id"],
        episode_id="receipt:" + observation["receipt_id"], target_key=observation["candidate_id"],
        request_id=observation["receipt_id"], action_id=observation["receipt_id"],
        candidate_id=observation["candidate_id"], evidence_entry_id=observation["receipt_id"],
        evidence_availability="available" if eligible else "unavailable",
        evidence_quality="INDEPENDENT_GOLD" if eligible else "ACTOR_PROVISIONAL",
        requested_at=observation["observed_at"], source_version=observation["source_version"],
        prediction_cutoff=observation["prediction_cutoff"],
        expected_information_value=None,
        observed_risk_reduction=observation.get("observed_risk_reduction") if eligible else None,
        calibration_error=None, status=status, source_id=observation["source_id"],
        dependency_key=observation["dependency_key"],
    )


def calibrate_observations(candidate_id: str, *, source_id: str, source_version: str,
                          cost: float, observations: Iterable[Mapping[str, Any]],
                          as_of: str | datetime, min_independent_dependencies: int = 5,
                          confidence: float = .10) -> AdaptiveVoIAssessment:
    links = [to_voi_link(item) for item in observations
             if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id]
    return calibrate_candidate_value(
        candidate_id, source_id=source_id, source_version=source_version,
        cost=cost, links=links, as_of=as_of,
        min_independent_dependencies=min_independent_dependencies,
        confidence=confidence,
    )


def actor_projection(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded fact that a receipt was evaluated, without loss data."""
    return {
        "schema_version": VERSION, "receipt_id": observation.get("receipt_id"),
        "candidate_id": observation.get("candidate_id"),
        "label_status": observation.get("label_status"),
        "eligible_for_numeric_calibration": False,
        "reason": "evaluator_metrics_are_sidecar_only",
    }


__all__ = [
    "VERSION", "INDEPENDENT_GOLD", "ACTOR_PROVISIONAL", "UNAVAILABLE",
    "build_empirical_voi_observation", "to_voi_link", "calibrate_observations",
    "actor_projection",
]
