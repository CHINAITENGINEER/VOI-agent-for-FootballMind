"""Build a finite decision model with an explicit source-health latent factor.

The ordinary finite VoI model can express source quality when it is already
known as part of a single hypothesis.  This adapter makes the distinction
explicit for experiments: a cheap health probe observes the reliability of a
source, while the usefulness of a later content query depends on that health.
The two observations may be dependent after
marginalising the joint match/source hypothesis.

This is a mechanism-level reference implementation.  It does not infer
source health from a match result and it does not turn a health signal into a
football prediction by itself.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

from .decision_value import EVIDENCE_CLASS, FiniteDecisionModel, ObservationTool


VERSION = "source-health-voi-v1"


def _names(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique strings")
    return result


def _probability_vector(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a probability vector")
    result = tuple(values)
    if len(result) != size:
        raise ValueError(f"{name} has the wrong dimension")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 or value > 1
           for value in result):
        raise ValueError(f"{name} contains an invalid probability")
    if abs(math.fsum(result) - 1.0) > 1e-10:
        raise ValueError(f"{name} must sum to one")
    return tuple(float(value) for value in result)


def _nonnegative(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def build_source_health_model(
    match_hypotheses: Mapping[str, Sequence[float]],
    match_prior: Mapping[str, float],
    health_prior: Mapping[str, float],
    *,
    probe_likelihoods: Mapping[str, Sequence[float]],
    content_likelihoods: Mapping[tuple[str, str], Sequence[float]],
    probe_outcomes: Sequence[str] = ("healthy", "degraded"),
    content_outcomes: Sequence[str] = ("positive_signal", "no_answer", "negative_signal"),
    probe_cost: float = 0.0,
    content_cost: float = 0.0,
    outcome_labels: Sequence[str] = ("home", "draw", "away"),
) -> FiniteDecisionModel:
    """Create a joint ``match-state × source-health`` finite model.

    ``probe_likelihoods`` is keyed by health state.  ``content_likelihoods``
    is keyed by ``(match_state, health_state)``.  Thus a degraded source can
    return an apparently usable response with a declared probability, while
    the policy still has an opportunity to learn that the channel is degraded
    before spending the content-query cost.
    """
    if not isinstance(match_hypotheses, Mapping) or not match_hypotheses:
        raise ValueError("match_hypotheses must be a nonempty mapping")
    if not isinstance(match_prior, Mapping) or set(match_prior) != set(match_hypotheses):
        raise ValueError("match_prior must cover exactly the match hypotheses")
    if not isinstance(health_prior, Mapping) or not health_prior:
        raise ValueError("health_prior must be a nonempty mapping")
    health_states = _names(tuple(health_prior), "health states")
    match_states = _names(tuple(match_hypotheses), "match hypotheses")
    probe_outcomes = _names(probe_outcomes, "probe outcomes")
    content_outcomes = _names(content_outcomes, "content outcomes")
    probe_cost = _nonnegative(probe_cost, "probe_cost")
    content_cost = _nonnegative(content_cost, "content_cost")
    _probability_vector(tuple(match_prior[state] for state in match_states), len(match_states), "match_prior")
    _probability_vector(tuple(health_prior[state] for state in health_states), len(health_states), "health_prior")

    if not isinstance(probe_likelihoods, Mapping) or not isinstance(content_likelihoods, Mapping):
        raise ValueError("probe_likelihoods and content_likelihoods must be mappings")
    for health in health_states:
        if health not in probe_likelihoods:
            raise ValueError(f"probe_likelihoods is missing health state: {health}")
        _probability_vector(probe_likelihoods[health], len(probe_outcomes), f"probe_likelihoods[{health}]")
    if set(probe_likelihoods) != set(health_states):
        raise ValueError("probe_likelihoods must cover the health states exactly")
    expected_content_keys = {(match, health) for match in match_states for health in health_states}
    if set(content_likelihoods) != expected_content_keys:
        raise ValueError("content_likelihoods must cover every match/health pair exactly")
    for key, values in content_likelihoods.items():
        _probability_vector(values, len(content_outcomes), f"content_likelihoods[{key!r}]")

    hypotheses = {}
    prior = {}
    for match in match_states:
        for health in health_states:
            # JSON tuples remain distinct even when user names contain ::.
            joint = json.dumps([match, health], ensure_ascii=False, separators=(",", ":"))
            hypotheses[joint] = tuple(match_hypotheses[match])
            prior[joint] = float(match_prior[match]) * float(health_prior[health])

    probe_tool = ObservationTool(
        tuple(probe_outcomes),
        {json.dumps([match, health], ensure_ascii=False, separators=(",", ":")): tuple(probe_likelihoods[health])
         for match in match_states for health in health_states},
        probe_cost,
        "A source-health or freshness probe; it observes the channel, not the match outcome.",
        evidence_id="source-health-probe",
    )
    content_tool = ObservationTool(
        tuple(content_outcomes),
        {json.dumps([match, health], ensure_ascii=False, separators=(",", ":")): tuple(content_likelihoods[(match, health)])
         for match in match_states for health in health_states},
        content_cost,
        "A domain-content query whose reliability depends on the source-health state.",
        evidence_id="source-content-query",
    )
    return FiniteDecisionModel(
        hypotheses,
        prior,
        {"source_health_probe": probe_tool, "content_query": content_tool},
        outcome_labels=outcome_labels,
        assumptions=(
            "The joint latent state factors into a declared match state and source-health state.",
            "Match state and source health are independent in the initial prior; acquired content can couple them.",
            "Positive and negative content signals are domain responses; no_answer is distinct from a negative signal.",
            "The health probe is a declared observation of the source channel, not a match result.",
            "Content reliability is conditional on the complete joint state; no marginal independence is assumed.",
            "All priors, observation laws and costs are synthetic declarations for mechanism evaluation.",
        ),
    )


def manifest_with_health_factors(model: FiniteDecisionModel) -> dict[str, Any]:
    """Return a scoped manifest annotation for reports and audit tooling."""
    manifest = model.manifest()
    manifest["adapter_schema_version"] = VERSION
    manifest["model_schema_version"] = manifest["schema_version"]
    manifest["evidence_class"] = EVIDENCE_CLASS
    manifest["latent_factorization"] = "joint_match_state_and_source_health"
    manifest["policy_question"] = "whether to probe source health before acquiring domain content"
    manifest["limitations"] = list(manifest["limitations"]) + [
        "This adapter does not estimate source health from production returns.",
        "A health probe is useful only under the declared joint observation laws.",
    ]
    return manifest


__all__ = ["VERSION", "build_source_health_model", "manifest_with_health_factors"]
