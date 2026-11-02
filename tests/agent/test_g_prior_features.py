from copy import deepcopy

import pytest

from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.g_prior_features import (
    ALIAS,
    FIELD,
    PREQUERY_BUILDER_SCHEMA,
    STRICT_PREQUERY_BUILDER_SCHEMA,
    TEACHING_SCHEMA,
    attach_prior_features,
    attach_prequery_prior_features,
    attach_strict_prequery_prior_features,
    build_prequery_feature_map,
    build_prequery_prior_features,
    build_strict_prequery_feature_map,
    build_strict_prequery_prior_features,
    prior_feature_manifest,
    strip_prior_features,
    validate_prior_features,
)
from football_agent.learned_voi_runtime import _actions
from football_agent.learned_voi_web import _web_action
from football_agent.match_source_profiles import resolve_profile


FEATURES = {
    "information_kind": "lineup_injury",
    "uncertainty": .8,
    "decision_sensitivity": .9,
    "source_reliability": .75,
    "specificity": .85,
    "redundancy": .1,
    "misdirection_risk": .2,
    "query_cost": .02,
}


def action(action_id="a"):
    return _web_action("read_url", "https://example.com/news/fixture", cost={"money": .01, "requests": 1.}) | {
        "action_id": action_id,
    }


def test_attach_requires_exact_prequery_declarations_and_does_not_mutate_menu():
    actions = [action("a"), action("b")]
    before = deepcopy(actions)
    result = attach_prior_features(actions, {"a": FEATURES, "b": {**FEATURES, "information_kind": "weather"}})
    assert actions == before
    assert result[0][FIELD] == validate_prior_features(FEATURES)
    assert FIELD not in actions[0]

    with pytest.raises(ValueError, match="exactly cover"):
        attach_prior_features(actions, {"a": FEATURES})
    with pytest.raises(ValueError, match="unknown fields"):
        attach_prior_features(actions, {"a": {**FEATURES, "title": "not a feature"}, "b": FEATURES})


def test_no_semantic_inference_and_transport_strips_only_declared_prior_fields():
    item = action("a")
    enriched = attach_prior_features([item], {"a": FEATURES})[0]
    assert enriched[FIELD]["information_kind"] == "lineup_injury"
    assert "fixture" in enriched["url"]  # URL is never inspected for features.
    transport = strip_prior_features(enriched)
    assert FIELD not in transport and ALIAS not in transport
    assert transport == item
    assert strip_prior_features({**enriched, "cost": {"money": .99, "requests": 1.}})["cost"]["money"] == .99


def test_runtime_action_contract_validates_prior_before_transport():
    enriched = attach_prior_features([action("a")], {"a": FEATURES})
    assert _actions(enriched)[0][FIELD]["query_cost"] == .02
    with pytest.raises(ValueError, match="only one"):
        _actions([{**enriched[0], ALIAS: FEATURES}])


def test_evidence_generator_has_explicit_feature_boundary_without_config_change():
    match = {"match_id": "fixture-one", "home_team": "Arsenal", "away_team": "Leeds",
             "kickoff_at": "2026-10-10T11:30:00Z", "prediction_cutoff": "2026-10-09T11:30:00Z"}
    profile = resolve_profile("Arsenal", "Leeds")
    generator = EvidenceDocumentCandidates(
        match_terms=["Arsenal", "Leeds"], seed_urls=profile["seed_urls"],
        allowed_hosts=profile["allowed_hosts"],
        numeric_actions=[{"action_id": "results_view", "group_ids": ["elo"],
                          "available": True, "cost_status": "simulated",
                          "cost": {"money": .01, "requests": 1.}}],
        web_cost={"money": .01, "requests": 1.}, match=match, profile=profile)
    public = {"state": {"base": {"kickoff": match["kickoff_at"]},
                          "prediction_cutoff": match["prediction_cutoff"]},
              "documents": [], "evidence": {"match_id": "fixture-one",
                          "schema_version": "match-evidence-extraction-v4",
                          "evidence": [], "state": {"followup_candidates": []}}}
    base = generator(public)
    features = {row["action_id"]: FEATURES for row in base}
    enriched = generator.actions_with_prior_features(public, features)
    assert [row["action_id"] for row in enriched] == [row["action_id"] for row in base]
    assert all(FIELD in row for row in enriched)
    assert generator.configuration() == generator.configuration()


def test_manifest_freezes_explicit_boundary():
    manifest = prior_feature_manifest()
    assert manifest["schema"] == "footballmind-g-prior-features-v1"
    assert manifest["prequery_builder_schema"] == PREQUERY_BUILDER_SCHEMA
    assert manifest["post_reveal_inference"] is False
    assert manifest["selection_authority"] == "empirical_g_only"


def _public_prediction():
    return {
        "prediction": {"probabilities": [0.52, 0.23, 0.25]},
        "state": {"base": {"kickoff": "2026-10-10T11:30:00Z"},
                  "prediction_cutoff": "2026-10-09T11:30:00Z"},
    }


def _legacy_capability(kind="lineup_injury"):
    return {
        "information_kind": kind,
        "decision_sensitivity_multiplier": .9,
        "source_reliability": .8,
        "specificity": .85,
        "redundancy": .1,
        "misdirection_risk": .2,
        "query_cost": .02,
    }


def _teaching_capability(action_name="lineup", scenario="rotation_sensitive"):
    return {
        "action": action_name,
        "scenario": scenario,
        "source_reliability": .8,
        "event_relevance": .9,
        "tactical_sensitivity": .7,
        "cost_units": .02,
    }


def test_prequery_builder_uses_only_public_prediction_and_explicit_capability():
    features = build_prequery_prior_features(_public_prediction(), _legacy_capability())
    assert features["information_kind"] == "lineup_injury"
    assert 0.0 <= features["uncertainty"] <= 1.0
    assert 0.0 <= features["decision_sensitivity"] <= 1.0
    assert features["query_cost"] == .02
    with pytest.raises(ValueError, match="missing fields"):
        build_prequery_prior_features(_public_prediction(), {"information_kind": "weather"})


def test_prequery_builder_supports_teaching_schema_and_frozen_time():
    features = build_prequery_prior_features(
        _public_prediction(), _teaching_capability(), schema=TEACHING_SCHEMA)
    assert features["schema"] == TEACHING_SCHEMA
    assert features["time_to_kickoff_hours"] == pytest.approx(24.0)
    assert features["max_probability"] == pytest.approx(.52 / 1.0)
    assert PREQUERY_BUILDER_SCHEMA == "footballmind-g-prequery-feature-builder-v1"


def test_prequery_builder_mapping_and_attach_are_exact_and_do_not_read_urls():
    actions = [action("a"), action("b")]
    before = deepcopy(actions)
    capabilities = {"a": _legacy_capability(), "b": _legacy_capability("weather")}
    declarations = build_prequery_feature_map(_public_prediction(), capabilities)
    enriched = attach_prequery_prior_features(actions, _public_prediction(), capabilities)
    assert set(declarations) == {"a", "b"}
    assert [row["action_id"] for row in enriched] == ["a", "b"]
    assert all(FIELD in row for row in enriched)
    assert actions == before
    # A title or URL cannot fill a missing capability: only the registry is
    # consumed by the builder.
    with pytest.raises(ValueError, match="missing fields"):
        build_prequery_feature_map(_public_prediction(), {"a": {"title": "lineup"}})


def test_strict_prequery_builder_requires_canonical_normalized_public_state():
    features = build_strict_prequery_prior_features(
        _public_prediction(), _teaching_capability(), schema=TEACHING_SCHEMA)
    assert features["schema"] == TEACHING_SCHEMA
    manifest = prior_feature_manifest()
    assert manifest["strict_prequery_builder_schema"] == STRICT_PREQUERY_BUILDER_SCHEMA
    assert manifest["strict_public_state_contract"]["cutoff_must_be_before_kickoff"] is True

    weighted = _public_prediction()
    weighted["prediction"]["probabilities"] = [52, 23, 25]
    with pytest.raises(ValueError, match=r"finite in \[0,1\]"):
        build_strict_prequery_prior_features(
            weighted, _teaching_capability(), schema=TEACHING_SCHEMA)

    legacy_fallback = _public_prediction()
    legacy_fallback["prediction"].pop("probabilities")
    legacy_fallback["initial_probabilities"] = [.52, .23, .25]
    with pytest.raises(ValueError, match="requires prediction.probabilities"):
        build_strict_prequery_prior_features(
            legacy_fallback, _teaching_capability(), schema=TEACHING_SCHEMA)

    at_kickoff = _public_prediction()
    at_kickoff["state"]["prediction_cutoff"] = at_kickoff["state"]["base"]["kickoff"]
    with pytest.raises(ValueError, match="before kickoff"):
        build_strict_prequery_prior_features(
            at_kickoff, _teaching_capability(), schema=TEACHING_SCHEMA)


def test_strict_prequery_mapping_is_exact_and_ignores_evidence_payloads():
    public = _public_prediction()
    # These fields may be carried by a runtime snapshot, but the builder must
    # not inspect them to derive a source prior.
    public["documents"] = [{"title": "late article", "url": "https://bad.example", "body": "post-query"}]
    public["evidence"] = {"snippet": "post-query", "result": "home"}
    capabilities = {"a": _teaching_capability(), "b": _teaching_capability("weather", "weather_sensitive")}
    declarations = build_strict_prequery_feature_map(
        public, capabilities, schema=TEACHING_SCHEMA)
    actions = [action("a"), action("b")]
    enriched = attach_strict_prequery_prior_features(
        actions, public, capabilities, schema=TEACHING_SCHEMA)
    assert set(declarations) == {"a", "b"}
    assert [row["action_id"] for row in enriched] == ["a", "b"]
    assert all(row[FIELD]["schema"] == TEACHING_SCHEMA for row in enriched)
    assert actions[0].get(FIELD) is None

    public["probabilities"] = [.52, .23, .25]
    with pytest.raises(ValueError, match="legacy fallback aliases"):
        build_strict_prequery_feature_map(public, capabilities, schema=TEACHING_SCHEMA)


def test_evidence_generator_exposes_strict_prequery_entrypoint():
    match = {"match_id": "fixture-strict", "home_team": "Arsenal", "away_team": "Leeds",
             "kickoff_at": "2026-10-10T11:30:00Z", "prediction_cutoff": "2026-10-09T11:30:00Z"}
    profile = resolve_profile("Arsenal", "Leeds")
    generator = EvidenceDocumentCandidates(
        match_terms=["Arsenal", "Leeds"], seed_urls=profile["seed_urls"],
        allowed_hosts=profile["allowed_hosts"],
        numeric_actions=[{"action_id": "results_view", "group_ids": ["elo"],
                          "available": True, "cost_status": "simulated",
                          "cost": {"money": .01, "requests": 1.}}],
        web_cost={"money": .01, "requests": 1.}, match=match, profile=profile)
    public = {
        "prediction": {"probabilities": [.52, .23, .25]},
        "state": {"base": {"kickoff": match["kickoff_at"]},
                  "prediction_cutoff": match["prediction_cutoff"]},
        "documents": [], "evidence": {"match_id": "fixture-strict",
                  "schema_version": "match-evidence-extraction-v4",
                  "evidence": [], "state": {"followup_candidates": []}},
    }
    base = generator(public)
    capabilities = {row["action_id"]: _teaching_capability() for row in base}
    enriched = generator.actions_with_strict_prequery_features(
        public, capabilities, schema=TEACHING_SCHEMA)
    assert [row["action_id"] for row in enriched] == [row["action_id"] for row in base]
    assert all(row[FIELD]["schema"] == TEACHING_SCHEMA for row in enriched)
