"""Research-grade one-step expected value of information (EVSI).

This module is deliberately separate from the operational acquisition loop.
It computes a model-based *expected* reduction in predictive risk from a
candidate observation.  It does not claim that the number is a realised
football gain: that requires an independent outcome sidecar and paired replay.

The contract is intentionally strict:

* the response probabilities and branch posteriors must be explicit;
* their mixture must reproduce the current prior within a declared tolerance;
* costs, delay and source reliability are charged separately from gross EVSI;
* missing or incoherent response models are ineligible for numeric selection;
* a stop is certified only when every candidate is computed and non-positive.

The default losses are proper scoring-rule risks for the three-way outcome
distribution.  A later decision-aware adapter can replace these with a
domain decision loss, but it must preserve the same branch and provenance
requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


VERSION = "expected-voi-v1"
OUTCOME_COUNT = 3
COMPUTED = "COMPUTED"
INELIGIBLE = "INELIGIBLE"


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _probabilities(values: Sequence[float] | Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(_finite(value, f"{name}[{index}]", minimum=0.0)
                   for index, value in enumerate(values))
    if len(result) != OUTCOME_COUNT:
        raise ValueError(f"{name} must contain exactly {OUTCOME_COUNT} probabilities")
    total = math.fsum(result)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must sum to one (got {total})")
    return result


def _risk(probabilities: Sequence[float], loss: str) -> float:
    if loss == "log_loss":
        # This is the Bayes risk of reporting the distribution itself under
        # logarithmic scoring, i.e. Shannon entropy in nats.
        return -math.fsum(p * math.log(p) for p in probabilities if p > 0.0)
    if loss == "brier":
        return 1.0 - math.fsum(p * p for p in probabilities)
    raise ValueError("loss must be 'log_loss' or 'brier'")


@dataclass(frozen=True)
class ObservationBranch:
    """One possible response of a candidate information action."""

    branch_id: str
    probability: float
    posterior: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.branch_id, str) or not self.branch_id.strip():
            raise ValueError("branch_id must be a nonempty string")
        _finite(self.probability, "branch.probability", minimum=0.0)
        _probabilities(self.posterior, "branch.posterior")


@dataclass(frozen=True)
class CandidateValue:
    """Auditable value decomposition for one candidate information action."""

    candidate_id: str
    status: str
    loss: str
    prior_risk: float | None
    expected_after_risk: float | None
    gross_evsi: float | None
    cost: float | None
    delay_penalty: float | None
    reliability_penalty: float | None
    net_voi: float | None
    coherence_error: float | None
    reason: str
    predictor_version: str | None = None
    response_model_version: str | None = None
    cost_model_version: str | None = None
    calibration_cutoff: str | None = None

    @property
    def eligible(self) -> bool:
        return self.status == COMPUTED

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": VERSION,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "loss": self.loss,
            "prior_risk": self.prior_risk,
            "expected_after_risk": self.expected_after_risk,
            "gross_evsi": self.gross_evsi,
            "cost": self.cost,
            "delay_penalty": self.delay_penalty,
            "reliability_penalty": self.reliability_penalty,
            "net_voi": self.net_voi,
            "coherence_error": self.coherence_error,
            "reason": self.reason,
            "predictor_version": self.predictor_version,
            "response_model_version": self.response_model_version,
            "cost_model_version": self.cost_model_version,
            "calibration_cutoff": self.calibration_cutoff,
        }


def evaluate_candidate(
    candidate_id: str,
    prior: Sequence[float] | Iterable[float],
    branches: Iterable[ObservationBranch | Mapping[str, object]],
    *,
    cost: float,
    delay: float = 0.0,
    delay_weight: float = 0.0,
    reliability: float = 1.0,
    reliability_penalty_weight: float = 0.0,
    loss: str = "log_loss",
    coherence_tolerance: float = 1e-6,
    predictor_version: str | None = None,
    response_model_version: str | None = None,
    cost_model_version: str | None = None,
    calibration_cutoff: str | None = None,
) -> CandidateValue:
    """Compute one-step model-based EVSI and its net value.

    ``branches`` is the response model ``q(z | state, action)`` together with
    ``p(y | state, z, action)``.  The mixture of branch posteriors must match
    ``prior``.  If that coherence check fails, the result is deliberately
    ``INELIGIBLE`` rather than silently normalising a misspecified model.
    """
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a nonempty string")
    prior_tuple = _probabilities(tuple(prior), "prior")
    cost_value = _finite(cost, "cost", minimum=0.0)
    delay_value = _finite(delay, "delay", minimum=0.0)
    delay_weight_value = _finite(delay_weight, "delay_weight", minimum=0.0)
    reliability_value = _finite(reliability, "reliability", minimum=0.0)
    if reliability_value > 1.0:
        raise ValueError("reliability must be <= 1")
    reliability_weight = _finite(reliability_penalty_weight,
                                 "reliability_penalty_weight", minimum=0.0)
    tolerance = _finite(coherence_tolerance, "coherence_tolerance", minimum=0.0)

    parsed: list[ObservationBranch] = []
    for index, raw in enumerate(branches):
        if isinstance(raw, ObservationBranch):
            parsed.append(raw)
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"branches[{index}] must be a mapping or ObservationBranch")
        posterior = raw.get("posterior")
        if not isinstance(posterior, (list, tuple)):
            raise ValueError(f"branches[{index}].posterior is required")
        parsed.append(ObservationBranch(
            branch_id=str(raw.get("branch_id", "")),
            probability=_finite(raw.get("probability"), f"branches[{index}].probability", minimum=0.0),
            posterior=tuple(posterior),
        ))
    if not parsed:
        raise ValueError("branches must not be empty")
    response_mass = math.fsum(item.probability for item in parsed)
    if not math.isclose(response_mass, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"branch probabilities must sum to one (got {response_mass})")

    mixture = tuple(math.fsum(item.probability * item.posterior[index]
                              for item in parsed)
                    for index in range(OUTCOME_COUNT))
    coherence_error = max(abs(mixture[index] - prior_tuple[index])
                          for index in range(OUTCOME_COUNT))
    if coherence_error > tolerance:
        return CandidateValue(
            candidate_id=candidate_id.strip(), status=INELIGIBLE, loss=loss,
            prior_risk=None, expected_after_risk=None, gross_evsi=None,
            cost=cost_value, delay_penalty=None, reliability_penalty=None,
            net_voi=None, coherence_error=coherence_error,
            reason="response_model_incoherent_with_prior",
            predictor_version=predictor_version,
            response_model_version=response_model_version,
            cost_model_version=cost_model_version,
            calibration_cutoff=calibration_cutoff,
        )

    prior_risk = _risk(prior_tuple, loss)
    expected_after = math.fsum(item.probability * _risk(item.posterior, loss)
                               for item in parsed)
    gross_evsi = prior_risk - expected_after
    delay_penalty = delay_value * delay_weight_value
    reliability_penalty = (1.0 - reliability_value) * reliability_weight
    net_voi = gross_evsi - cost_value - delay_penalty - reliability_penalty
    return CandidateValue(
        candidate_id=candidate_id.strip(), status=COMPUTED, loss=loss,
        prior_risk=prior_risk, expected_after_risk=expected_after,
        gross_evsi=gross_evsi, cost=cost_value,
        delay_penalty=delay_penalty, reliability_penalty=reliability_penalty,
        net_voi=net_voi, coherence_error=coherence_error,
        reason="computed_coherent_evsi",
        predictor_version=predictor_version,
        response_model_version=response_model_version,
        cost_model_version=cost_model_version,
        calibration_cutoff=calibration_cutoff,
    )


def unknown_candidate(
    candidate_id: str,
    prior: Sequence[float] | Iterable[float],
    *,
    reason: str,
    cost: float | None = None,
    loss: str = "log_loss",
    predictor_version: str | None = None,
    cost_model_version: str | None = None,
) -> CandidateValue:
    """Record missing response evidence without inventing a zero-value model.

    The current predictive risk may be known even when the effect of acquiring
    information is not.  Unknown cost is preserved too; neither absence is a
    certificate that acquisition is worthless.
    """
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a nonempty string")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("unknown candidate requires a nonempty reason")
    prior_tuple = _probabilities(prior, "prior")
    return CandidateValue(
        candidate_id=candidate_id.strip(), status=INELIGIBLE, loss=loss,
        prior_risk=_risk(prior_tuple, loss), expected_after_risk=None,
        gross_evsi=None, cost=None if cost is None else _finite(cost, "cost", minimum=0.0),
        delay_penalty=None, reliability_penalty=None, net_voi=None,
        coherence_error=None, reason=reason.strip(),
        predictor_version=predictor_version, cost_model_version=cost_model_version,
    )


def select_candidate(
    values: Iterable[CandidateValue],
    *,
    tolerance: float = 1e-12,
    defer_when_incomplete: bool = True,
) -> dict[str, object]:
    """Select the best positive net candidate or issue an auditable stop/defer.

    ``defer_when_incomplete`` is retained for call compatibility. It cannot
    authorize a stop certificate while any candidate value remains unknown.
    """
    rows = tuple(values)
    threshold = _finite(tolerance, "tolerance", minimum=0.0)
    eligible = tuple(row for row in rows if row.eligible and row.net_voi is not None)
    positive = tuple(row for row in eligible if row.net_voi > threshold)
    if positive:
        chosen = max(positive, key=lambda row: (row.net_voi, row.candidate_id))
        return {
            "decision": "acquire",
            "candidate_id": chosen.candidate_id,
            "reason": "best_positive_net_voi",
            "stop_certificate": None,
            "eligible_candidates": len(eligible),
            "ineligible_candidates": len(rows) - len(eligible),
        }
    if len(eligible) != len(rows):
        return {
            "decision": "defer",
            "candidate_id": None,
            "reason": "candidate_value_incomplete",
            "stop_certificate": None,
            "eligible_candidates": len(eligible),
            "ineligible_candidates": len(rows) - len(eligible),
        }
    return {
        "decision": "stop",
        "candidate_id": None,
        "reason": "all_computed_net_voi_nonpositive",
        "stop_certificate": {
            "kind": "finite_candidate_set_nonpositive_net_voi",
            "candidate_count": len(rows),
            "max_net_voi": None if not eligible else max(row.net_voi for row in eligible),
        },
        "eligible_candidates": len(eligible),
        "ineligible_candidates": len(rows) - len(eligible),
    }


__all__ = [
    "VERSION", "COMPUTED", "INELIGIBLE", "ObservationBranch", "CandidateValue",
    "evaluate_candidate", "unknown_candidate", "select_candidate",
]
