"""Read-only adapter for the ``synthetic-g-teaching-v1`` mechanism prior.

The teaching model is useful for ordering a declared menu before an empirical
g estimate is available.  It is never an empirical value model: every score is
kept in a nested ``synthetic_prior`` record and the existing empirical support
gate remains the only authority for acquisition.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from football_research.acquisition.synthetic_g_teaching import (
    VERSION as MODEL_SCHEMA,
    SyntheticGModel,
)
from .g_prior_features import ALIAS, FIELD, TEACHING_SCHEMA, validate_teaching_prior_features


ADAPTER_SCHEMA = "footballmind-g-synthetic-teaching-prior-adapter-v1"
ESTIMATE_SCHEMA = "footballmind-g-synthetic-teaching-prior-estimate-v1"
SYNTHETIC_STATUS = "synthetic_teaching_only"


def _canonical_digest(value: Mapping[str, Any]) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(data).hexdigest()


def _empirical_status(value: Mapping[str, Any] | None,
                      explicit: str | None = None) -> str:
    if explicit is not None:
        return str(explicit)
    if not isinstance(value, Mapping):
        return "UNKNOWN"
    estimate = value.get("estimate")
    if isinstance(estimate, Mapping) and estimate.get("support_status") is not None:
        return str(estimate["support_status"])
    return str(value.get("support_status", "UNKNOWN"))


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


class SyntheticGTeachingPriorAdapter:
    """Expose the current synthetic teaching artifact as a ranking prior.

    Candidate features must be nested under ``synthetic_prior_features`` (or
    its existing alias ``g_prior_features``) and carry the teaching feature
    schema marker.  The adapter does not infer action, scenario, relevance or
    cost from URLs, titles, action IDs, or returned documents.
    """

    feature_field = FIELD

    def __init__(self, model: SyntheticGModel, *, source: Mapping[str, Any] | None = None,
                 artifact_bytes: bytes | None = None):
        if not isinstance(model, SyntheticGModel):
            raise ValueError("A validated synthetic-g-teaching-v1 model is required")
        self.model = model
        self.source = dict(source or {})
        self.artifact_bytes = bytes(artifact_bytes) if artifact_bytes is not None else None
        if self.artifact_bytes is not None:
            self.source.setdefault("sha256", sha256(self.artifact_bytes).hexdigest())
        self.source.setdefault("sha256", _canonical_digest(model.payload))
        self.source.setdefault("source_sha256", self.source["sha256"])
        self.source.setdefault("model_sha256", _canonical_digest(model.payload))
        self.source_sha256 = str(self.source["source_sha256"])
        self.model_sha256 = str(self.source["model_sha256"])

    @classmethod
    def from_path(cls, path: str | Path, *, declared_source_path: str | None = None):
        source_path = Path(path)
        raw = source_path.read_bytes()
        # Use the model's canonical loader so adapter wiring follows the same
        # validation path as direct ``SyntheticGModel.load(...).estimate(...)``
        # callers, while retaining the exact bytes for runtime replay.
        model = SyntheticGModel.load(source_path)
        payload = model.payload
        return cls(model,
                   source={"path": declared_source_path or str(source_path.resolve()),
                           "sha256": sha256(raw).hexdigest(),
                           "source_sha256": sha256(raw).hexdigest(),
                           "model_sha256": _canonical_digest(model.payload),
                           "container_schema": payload.get("schema"),
                           "model_section": "standalone_model"},
                   artifact_bytes=raw)

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "model_schema": MODEL_SCHEMA,
            "model_sha256": self.model_sha256,
            "source_sha256": self.source_sha256,
            "source_path": self.source.get("path"),
            "container_schema": self.source.get("container_schema", MODEL_SCHEMA),
            "model_section": self.source.get("model_section", "standalone_model"),
            "synthetic": True,
            "empirical": False,
            "support_status": SYNTHETIC_STATUS,
            "evidence_origin": "constructed_mechanism",
            "calibration_status": "not_established",
            "production_ready": False,
            "selection_mode": "prior_ranking_only_empirical_gate_unchanged",
        }

    def _features(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate must be a mapping")
        features = candidate.get(FIELD)
        if features is None:
            features = candidate.get(ALIAS)
        # ContextVOIPolicy passes the decision-local feature object to the
        # adapter's ``attach`` method.  Accept that object only when its
        # explicit schema marker is present; this is not semantic inference
        # from an action or URL.
        if features is None and candidate.get("schema") == TEACHING_SCHEMA:
            features = {key: value for key, value in candidate.items()
                        if key not in {"candidate_id", "action_id"}}
        if not isinstance(features, Mapping):
            raise ValueError("explicit synthetic teaching features are required")
        return validate_teaching_prior_features(features)

    def prior(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        candidate_id = candidate.get("candidate_id", candidate.get("action_id"))
        try:
            features = self._features(candidate)
            estimate = self.model.estimate(**{key: value for key, value in features.items()
                                              if key != "schema"})
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "schema": ESTIMATE_SCHEMA,
                "candidate_id": candidate_id,
                "status": "UNAVAILABLE",
                "prior_score": None,
                "prior_net_value": None,
                "reason": "teaching_prior_features_missing_or_invalid",
                "detail": str(exc),
                "support_status": SYNTHETIC_STATUS,
                "evidence_origin": "constructed_mechanism",
                "model_schema": MODEL_SCHEMA,
                "model_sha256": self.model_sha256,
                "calibration_status": "not_established",
                "production_ready": False,
            }
        return {
            **estimate,
            "candidate_id": candidate_id,
            "status": "PRIOR_ONLY",
            "prior_score": estimate["expected_signed_gain"],
            "prior_net_value": estimate["expected_net_voi"],
            "cost_deducted": True,
            "reason": "synthetic_teaching_prior_for_diagnostic_ranking_only",
            "model_sha256": self.model_sha256,
        }

    def assess(self, candidate: Mapping[str, Any], *,
               empirical_value: Mapping[str, Any] | None = None,
               empirical_support_status: str | None = None) -> dict[str, Any]:
        """Return separate synthetic and empirical lanes without promotion."""
        prior = self.prior(candidate)
        status = _empirical_status(empirical_value, empirical_support_status)
        supplied = dict(empirical_value) if isinstance(empirical_value, Mapping) else None
        gain = supplied.get("expected_signed_gain") if supplied else None
        net = supplied.get("net_voi") if supplied else None
        supported = status == "supported" and _finite(gain) and _finite(net)
        return {
            "schema": ADAPTER_SCHEMA,
            "candidate_id": candidate.get("candidate_id", candidate.get("action_id")),
            "synthetic_prior": prior,
            "empirical": {
                "support_status": status,
                "value_status": supplied.get("status") if supplied else None,
                "evidence_origin": ((supplied or {}).get("estimate") or {}).get("evidence_origin")
                    if isinstance((supplied or {}).get("estimate"), Mapping)
                    else (supplied or {}).get("evidence_origin"),
                "expected_signed_gain": float(gain) if supported else None,
                "net_voi": float(net) if supported else None,
            },
            "effective": {
                # Preserve UNKNOWN/prototype labels exactly; only the boolean
                # gate decides whether an empirical value can select.
                "support_status": status,
                "expected_signed_gain": float(gain) if supported else None,
                "net_voi": float(net) if supported else None,
                "selection_eligible": bool(supported),
                "gate_reason": None if supported else "empirical_support_required",
            },
            "provenance": {
                "synthetic": {
                    "evidence_origin": "constructed_mechanism",
                    "model_schema": MODEL_SCHEMA,
                    "model_sha256": self.model_sha256,
                    "support_status": SYNTHETIC_STATUS,
                    "production_ready": False,
                    "calibration_status": "not_established",
                },
                "empirical": {
                    "evidence_origin": ((supplied or {}).get("estimate") or {}).get("evidence_origin")
                        if isinstance((supplied or {}).get("estimate"), Mapping)
                        else (supplied or {}).get("evidence_origin"),
                    "support_status": status,
                },
            },
            "selection_allowed": bool(supported),
        }

    def attach(self, candidate: Mapping[str, Any],
               empirical_value: Mapping[str, Any] | None) -> dict[str, Any]:
        value = deepcopy(dict(empirical_value)) if isinstance(empirical_value, Mapping) else {}
        value["synthetic_prior"] = self.assess(candidate, empirical_value=empirical_value)
        return value

    def rank_diagnostic(self, candidates: list[Mapping[str, Any]], *,
                        empirical_values: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        empirical_values = empirical_values or {}
        rows = [self.assess(candidate, empirical_value=empirical_values.get(
            candidate.get("candidate_id", candidate.get("action_id")))) for candidate in candidates]
        ordered = sorted(rows, key=lambda row: (
            row["synthetic_prior"]["prior_net_value"] is None,
            -(row["synthetic_prior"]["prior_net_value"]
              if row["synthetic_prior"]["prior_net_value"] is not None else 0.0),
            str(row.get("candidate_id") or "")))
        return {
            "schema": ADAPTER_SCHEMA,
            "rows": ordered,
            "ordering_basis": "synthetic_teaching_prior_expected_net_voi_descending_none_last",
            "selection_allowed": bool(ordered) and all(row["selection_allowed"] for row in ordered),
            "selection_gate": "empirical_support_status_supported_required",
            "ranking_only": True,
            "provenance": self.manifest,
        }


def is_synthetic_prior_adapter(value: Any) -> bool:
    """Return whether ``value`` is one of the two validated prior adapters."""
    from .g_teacher_prior import SyntheticGTeacherPriorAdapter
    return isinstance(value, (SyntheticGTeacherPriorAdapter, SyntheticGTeachingPriorAdapter))


__all__ = ["ADAPTER_SCHEMA", "ESTIMATE_SCHEMA", "MODEL_SCHEMA",
           "SYNTHETIC_STATUS", "SyntheticGTeachingPriorAdapter",
           "is_synthetic_prior_adapter"]
