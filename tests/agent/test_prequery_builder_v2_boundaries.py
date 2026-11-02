"""Boundary contract for the strict pre-query builder v2.

v1 remains a replay-compatible parser.  v2 is the opt-in online boundary and
must reject ambiguous or weakly typed public-state projections before any
feature is derived.
"""

import copy

import pytest

from football_agent.g_prior_features import (
    SCHEMA,
    TEACHING_SCHEMA,
    build_prequery_prior_features,
    build_strict_prequery_prior_features,
)


def _public(*, probabilities=(0.52, 0.23, 0.25), cutoff="2026-10-09T11:30:00Z"):
    return {
        "prediction": {"probabilities": list(probabilities)},
        "state": {
            "base": {"kickoff": "2026-10-10T11:30:00Z"},
            "prediction_cutoff": cutoff,
        },
    }


def _legacy_capability():
    return {
        "information_kind": "lineup_injury",
        "decision_sensitivity_multiplier": 0.9,
        "source_reliability": 0.8,
        "specificity": 0.85,
        "redundancy": 0.1,
        "misdirection_risk": 0.2,
        "query_cost": 0.02,
    }


def _teaching_capability():
    return {
        "action": "lineup",
        "scenario": "rotation_sensitive",
        "source_reliability": 0.8,
        "event_relevance": 0.9,
        "tactical_sensitivity": 0.7,
        "cost_units": 0.02,
    }


@pytest.mark.parametrize("schema, capability", [
    (SCHEMA, _legacy_capability()),
    (TEACHING_SCHEMA, _teaching_capability()),
])
def test_v1_keeps_legacy_weighted_probabilities_compatible(schema, capability):
    public = _public(probabilities=(52, 23, 25))
    result = build_prequery_prior_features(public, capability, schema=schema)
    if schema == TEACHING_SCHEMA:
        assert result["max_probability"] == pytest.approx(0.52)
    else:
        assert 0.0 <= result["uncertainty"] <= 1.0


@pytest.mark.parametrize("schema, capability", [
    (SCHEMA, _legacy_capability()),
    (TEACHING_SCHEMA, _teaching_capability()),
])
def test_v1_keeps_cutoff_equal_to_kickoff_compatible(schema, capability):
    public = _public(cutoff="2026-10-10T11:30:00Z")
    result = build_prequery_prior_features(public, capability, schema=schema)
    if schema == TEACHING_SCHEMA:
        assert result["time_to_kickoff_hours"] == pytest.approx(0.0)


@pytest.mark.parametrize("schema, capability", [
    (SCHEMA, _legacy_capability()),
    (TEACHING_SCHEMA, _teaching_capability()),
])
def test_v1_can_fallback_to_explicit_initial_probabilities(schema, capability):
    public = _public()
    public["prediction"] = None
    public["initial_probabilities"] = [0.52, 0.23, 0.25]
    result = build_prequery_prior_features(public, capability, schema=schema)
    if schema == TEACHING_SCHEMA:
        assert result["max_probability"] == pytest.approx(0.52)
    else:
        assert result["information_kind"] == "lineup_injury"


@pytest.mark.parametrize("bad_public, error", [
    (_public(probabilities=(52, 23, 25)), r"finite in \[0,1\]"),
    (_public(cutoff="2026-10-10T11:30:00Z"), "before kickoff"),
])
def test_v2_rejects_legacy_weighting_and_cutoff_at_kickoff(bad_public, error):
    with pytest.raises(ValueError, match=error):
        build_strict_prequery_prior_features(
            bad_public, _teaching_capability(), schema=TEACHING_SCHEMA)


def test_v2_rejects_null_prediction_fallback_to_initial_probabilities():
    public = _public()
    public["prediction"] = None
    public["initial_probabilities"] = [0.52, 0.23, 0.25]
    with pytest.raises(ValueError, match="requires prediction mapping"):
        build_strict_prequery_prior_features(
            public, _teaching_capability(), schema=TEACHING_SCHEMA)


@pytest.mark.parametrize("alias", [
    "probabilities", "initial_probabilities", "kickoff_at", "kickoff",
    "prediction_cutoff", "decision_at",
])
def test_v2_rejects_legacy_alias_conflicts(alias):
    public = _public()
    public[alias] = [0.52, 0.23, 0.25] if "probabilities" in alias else "2026-10-09T11:30:00Z"
    with pytest.raises(ValueError, match="legacy fallback aliases"):
        build_strict_prequery_prior_features(
            public, _teaching_capability(), schema=TEACHING_SCHEMA)


def test_v2_accepts_both_frozen_feature_schemas_with_canonical_state():
    public = _public()
    legacy = build_strict_prequery_prior_features(
        copy.deepcopy(public), _legacy_capability(), schema=SCHEMA)
    teaching = build_strict_prequery_prior_features(
        copy.deepcopy(public), _teaching_capability(), schema=TEACHING_SCHEMA)
    assert legacy["information_kind"] == "lineup_injury"
    assert teaching["schema"] == TEACHING_SCHEMA
