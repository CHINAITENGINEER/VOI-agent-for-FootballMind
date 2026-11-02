"""Optional synthetic g prior for information-value ranking.

The historical signed-gain model remains the empirical authority.  This
module is deliberately a separate provider: it can rank candidate information
families when the real g corpus is too small, while exposing provenance so the
caller cannot present the result as calibrated empirical evidence.
"""
from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "footballmind-g-synthetic-teacher-prior-v1"
BRIDGE_SCHEMA = "footballmind-g-training-bridge-v1"
ADAPTER_SCHEMA = "footballmind-g-prior-adapter-v1"
SYNTHETIC_STATUS = "SYNTHETIC_PRIOR"
UNKNOWN_STATUS = "UNKNOWN"
_CONTINUOUS = (
    "uncertainty", "decision_sensitivity", "source_reliability", "specificity",
    "redundancy", "misdirection_risk", "query_cost",
)

# This is intentionally a closed contract.  The bridge artifact is a teaching
# prior, so accepting a new feature silently would make a coefficient mean
# something different without a new artifact version.
_KIND_FEATURES = (
    "lineup_injury", "tactical_plan", "motivation_rotation", "odds_shock",
    "weather", "historical_h2h",
)
_EXPECTED_FEATURE_NAMES = ["intercept", *_CONTINUOUS,
                           *(f"kind:{kind}" for kind in _KIND_FEATURES)]


def load_prior(path: str | Path) -> dict[str, Any]:
    """Load the standalone synthetic model artifact.

    The phase-4 bridge also contains an empirical refit and a one-match
    probe.  When given that container, this helper extracts only
    ``synthetic_teacher_prior.model`` so a caller cannot accidentally treat
    the bridge's empirical section as the synthetic prior.  Use
    ``SyntheticGTeacherPriorAdapter`` when source/container provenance must
    be retained or when attaching a value to the runtime gate.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") == BRIDGE_SCHEMA:
        section = payload.get("synthetic_teacher_prior")
        if not isinstance(section, Mapping) or not isinstance(section.get("model"), Mapping):
            raise ValueError("phase-4 bridge has no synthetic teacher model")
        payload = section["model"]
    return _validate_model(payload)


def _validate_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported synthetic g prior schema")
    names = payload.get("feature_names")
    coefficients = payload.get("coefficients")
    if names != _EXPECTED_FEATURE_NAMES or not isinstance(coefficients, list) or len(names) != len(coefficients):
        raise ValueError("synthetic g prior feature contract is invalid")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in coefficients):
        raise ValueError("synthetic g prior coefficients must be finite numbers")
    if payload.get("label_origin") != "synthetic_domain_teacher":
        raise ValueError("synthetic g prior provenance is missing")
    if payload.get("calibration_status") != "not_established":
        raise ValueError("synthetic g prior must remain uncalibrated")
    if payload.get("production_ready") is not False:
        raise ValueError("synthetic g prior cannot be production-ready")
    if payload.get("usage_rule") != "prior_only_until_real_independent_gains_calibrate_it":
        raise ValueError("synthetic g prior usage rule is missing")
    return dict(payload)


def _load_model_or_bridge(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(model, source_metadata)`` from a model or bridge artifact."""
    source_path = Path(path)
    raw = source_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema") == SCHEMA:
        model = _validate_model(payload)
        section = "standalone_model"
    elif payload.get("schema") == BRIDGE_SCHEMA:
        section_payload = payload.get("synthetic_teacher_prior")
        if not isinstance(section_payload, Mapping) or not isinstance(section_payload.get("model"), Mapping):
            raise ValueError("phase-4 bridge has no synthetic teacher model")
        model = _validate_model(section_payload["model"])
        section = "synthetic_teacher_prior.model"
    else:
        raise ValueError("expected a synthetic prior model or phase-4 bridge artifact")
    canonical_model = json.dumps(model, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return model, {
        "path": str(source_path.resolve()),
        "sha256": sha256(raw).hexdigest(),  # backwards-compatible source digest
        "source_sha256": sha256(raw).hexdigest(),
        "model_sha256": sha256(canonical_model).hexdigest(),
        "container_schema": payload.get("schema"),
        "model_section": section,
    }


def _finite_unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _vector(model: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[float]:
    kind = candidate.get("information_kind")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("information_kind is required")
    values = [1.0]
    for name in _CONTINUOUS:
        value = _finite_unit(candidate.get(name), name)
        # query_cost is also bounded to [0, 1] for the prior; callers may use
        # an external currency/request budget separately.
        values.append(value)
    names = model["feature_names"]
    for name in names[8:]:
        values.append(float(name == f"kind:{kind}"))
    if len(values) != len(model["coefficients"]):
        raise ValueError("candidate vector does not match synthetic g prior")
    return values


def score_candidate(model: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Score one candidate with the synthetic teacher prior.

    ``expected_gain`` is a research prior for signed log-score improvement;
    ``net_value`` subtracts the declared query cost.  It is never a calibrated
    probability, monetary EVSI, or substitute for empirical g support.
    """
    vector = _vector(model, candidate)
    raw = sum(float(a) * float(b) for a, b in zip(vector, model["coefficients"]))
    cost = float(candidate["query_cost"])
    return {
        "expected_gain": raw,
        "expected_signed_gain": raw,
        "net_value": raw - cost,
        "cost_deducted": True,
        "support_status": "SYNTHETIC_PRIOR",
        "value_kind": "synthetic_teacher_expected_signed_gain",
        "calibration_status": model.get("calibration_status", "not_established"),
        "evidence_origin": "synthetic_domain_teacher",
        "production_ready": False,
    }


def rank_candidates(model: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for candidate in candidates:
        result = score_candidate(model, candidate)
        scored.append({"candidate_id": candidate.get("candidate_id"), **result})
    return sorted(scored, key=lambda item: (item["net_value"], item["candidate_id"] or ""), reverse=True)


def _empirical_status(value: Mapping[str, Any] | None, explicit: str | None) -> str:
    """Read empirical support without promoting any prior status.

    Only the exact value-model status ``supported`` is actionable.  In
    particular, ``SYNTHETIC_PRIOR``, ``prior_only`` and all unknown or
    prototype statuses remain outside the empirical selection gate.
    """
    if explicit is not None:
        return explicit
    if not isinstance(value, Mapping):
        return UNKNOWN_STATUS
    estimate = value.get("estimate")
    if isinstance(estimate, Mapping) and estimate.get("support_status") is not None:
        return str(estimate.get("support_status"))
    return str(value.get("support_status", UNKNOWN_STATUS))


class SyntheticGTeacherPriorAdapter:
    """Attach the phase-4 synthetic prior beside an empirical value.

    The adapter has two deliberately separate lanes:

    * ``synthetic_prior`` is a diagnostic ranking score produced by the
      uncalibrated teacher artifact;
    * ``effective`` is the empirical decision lane.  It is ``UNKNOWN`` with
      null gain/net value unless the supplied empirical model explicitly says
      ``support_status == 'supported'`` and supplies finite values.

    This object never writes synthetic numbers into ``expected_signed_gain``
    or ``net_voi``.  A caller may display or sort the diagnostic prior, but
    the existing UNKNOWN gate must make the acquisition decision.
    """

    def __init__(self, model: Mapping[str, Any], *, source: Mapping[str, Any] | None = None):
        self.model = _validate_model(model)
        self.source = dict(source or {})
        if "sha256" not in self.source:
            # A programmatically supplied model has no stable file identity;
            # retain an explicit digest of its canonical JSON instead.
            encoded = json.dumps(self.model, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            self.source["sha256"] = sha256(encoded).hexdigest()
        if "source_sha256" not in self.source:
            self.source["source_sha256"] = self.source["sha256"]
        if "model_sha256" not in self.source:
            encoded = json.dumps(self.model, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            self.source["model_sha256"] = sha256(encoded).hexdigest()
        self.source_sha256 = self.source["source_sha256"]
        self.model_sha256 = self.source["model_sha256"]

    @classmethod
    def from_path(cls, path: str | Path, *, declared_source_path: str | None = None) -> "SyntheticGTeacherPriorAdapter":
        model, source = _load_model_or_bridge(path)
        if declared_source_path is not None:
            source["path"] = declared_source_path
        adapter = cls(model, source=source)
        # Keep the exact source bytes available to the runtime so a run using
        # this prior can archive and replay the decision-local model without
        # relying on an external absolute path.
        adapter.artifact_bytes = Path(path).read_bytes()
        return adapter

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "model_schema": SCHEMA,
            "model_sha256": self.model_sha256,
            "source_sha256": self.source_sha256,
            "source_path": self.source.get("path"),
            "container_schema": self.source.get("container_schema", SCHEMA),
            "model_section": self.source.get("model_section", "standalone_model"),
            "synthetic": True,
            "empirical": False,
            "evidence_origin": "synthetic_domain_teacher",
            "calibration_status": "not_established",
            "production_ready": False,
            "selection_mode": "prior_ranking_only_empirical_gate_unchanged",
        }

    def prior(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        """Return the synthetic score, or an explicit unavailable record."""
        candidate_id = candidate.get("candidate_id", candidate.get("action_id"))
        try:
            scored = score_candidate(self.model, candidate)
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "schema": "footballmind-g-synthetic-prior-estimate-v1",
                "candidate_id": candidate_id,
                "status": "UNAVAILABLE",
                "prior_score": None,
                "prior_net_value": None,
                "reason": "prior_features_missing_or_invalid",
                "detail": str(exc),
                "support_status": SYNTHETIC_STATUS,
                "value_kind": "synthetic_teacher_expected_signed_gain",
                "evidence_origin": "synthetic_domain_teacher",
                "model_schema": SCHEMA,
                "model_sha256": self.model_sha256,
                "calibration_status": self.model.get("calibration_status", "not_established"),
                "production_ready": False,
            }
        return {
            "schema": "footballmind-g-synthetic-prior-estimate-v1",
            "candidate_id": candidate_id,
            "status": "PRIOR_ONLY",
            "prior_score": scored["expected_signed_gain"],
            "prior_net_value": scored["net_value"],
            "cost_deducted": True,
            "reason": "synthetic_prior_for_diagnostic_ranking_only",
            "support_status": SYNTHETIC_STATUS,
            "value_kind": "synthetic_teacher_expected_signed_gain",
            "evidence_origin": "synthetic_domain_teacher",
            "model_schema": SCHEMA,
            "model_sha256": self.model_sha256,
            "calibration_status": self.model.get("calibration_status", "not_established"),
            "production_ready": False,
        }

    def assess(self, candidate: Mapping[str, Any], *, empirical_value: Mapping[str, Any] | None = None,
               empirical_support_status: str | None = None) -> dict[str, Any]:
        """Return a provenance-preserving two-lane value envelope.

        ``effective`` intentionally carries nulls for unknown empirical
        support.  The synthetic score lives only in ``synthetic_prior`` and
        cannot make ``selection_allowed`` true.
        """
        prior = self.prior(candidate)
        status = _empirical_status(empirical_value, empirical_support_status)
        empirical = dict(empirical_value) if isinstance(empirical_value, Mapping) else None
        supported = status == "supported"
        gain = empirical.get("expected_signed_gain") if supported and empirical else None
        net = empirical.get("net_voi") if supported and empirical else None
        finite = (isinstance(gain, (int, float)) and not isinstance(gain, bool) and math.isfinite(float(gain))
                  and isinstance(net, (int, float)) and not isinstance(net, bool) and math.isfinite(float(net)))
        supported = bool(supported and finite)
        effective_status = "supported" if supported else UNKNOWN_STATUS
        return {
            "schema": ADAPTER_SCHEMA,
            "candidate_id": candidate.get("candidate_id", candidate.get("action_id")),
            "synthetic_prior": prior,
            "empirical": {
                "support_status": status,
                "value_status": empirical.get("status") if empirical else None,
                "evidence_origin": ((empirical or {}).get("estimate") or {}).get("evidence_origin")
                    if isinstance((empirical or {}).get("estimate"), Mapping) else (empirical or {}).get("evidence_origin"),
                "expected_signed_gain": gain if supported else None,
                "net_voi": net if supported else None,
            },
            "effective": {
                "support_status": effective_status,
                "expected_signed_gain": gain if supported else None,
                "net_voi": net if supported else None,
                "selection_eligible": supported,
                "gate_reason": None if supported else "empirical_support_unknown",
            },
            "provenance": {
                "synthetic": {
                    "evidence_origin": "synthetic_domain_teacher",
                    "model_schema": SCHEMA,
                    "model_sha256": self.model_sha256,
                    "production_ready": False,
                    "calibration_status": "not_established",
                },
                "empirical": {
                    "evidence_origin": ((empirical or {}).get("estimate") or {}).get("evidence_origin")
                        if isinstance((empirical or {}).get("estimate"), Mapping) else (empirical or {}).get("evidence_origin"),
                    "support_status": effective_status,
                },
            },
            "selection_allowed": supported,
        }

    def attach(self, candidate: Mapping[str, Any], empirical_value: Mapping[str, Any] | None) -> dict[str, Any]:
        """Copy an existing value and attach the prior without changing it."""
        value = deepcopy(dict(empirical_value)) if isinstance(empirical_value, Mapping) else {}
        value["synthetic_prior"] = self.assess(candidate, empirical_value=empirical_value)
        return value

    def rank_diagnostic(self, candidates: list[Mapping[str, Any]], *, empirical_values: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """Rank by prior while explicitly reporting whether selection is gated."""
        empirical_values = empirical_values or {}
        rows = [self.assess(candidate, empirical_value=empirical_values.get(
            candidate.get("candidate_id", candidate.get("action_id")))) for candidate in candidates]
        ordered = sorted(rows, key=lambda row: (
            row["synthetic_prior"]["prior_net_value"] is not None,
            row["synthetic_prior"]["prior_net_value"] if row["synthetic_prior"]["prior_net_value"] is not None else float("-inf"),
            str(row.get("candidate_id") or "")), reverse=True)
        return {
            "schema": ADAPTER_SCHEMA,
            "rows": ordered,
            "ordering_basis": "synthetic_prior_net_value_descending",
            # An empty diagnostic menu cannot authorize a request.  Keep the
            # explicit non-empty guard used by the teaching adapter.
            "selection_allowed": bool(ordered) and all(row["selection_allowed"] for row in ordered),
            "selection_gate": "empirical_support_status_supported_required",
            "provenance": self.manifest,
        }


__all__ = ["SCHEMA", "BRIDGE_SCHEMA", "ADAPTER_SCHEMA", "load_prior", "score_candidate",
           "rank_candidates", "SyntheticGTeacherPriorAdapter"]
