from __future__ import annotations

import math
import pytest

from football_agent.source_health_voi import (
    VERSION,
    build_source_health_model,
    manifest_with_health_factors,
)


MATCH = {
    "rotation_strong": (.75, .17, .08),
    "rotation_weak": (.35, .30, .35),
}
MATCH_PRIOR = {"rotation_strong": .5, "rotation_weak": .5}
HEALTH_PRIOR = {"fresh": .55, "stale": .45}
PROBE = {"fresh": (.92, .08), "stale": (.18, .82)}
CONTENT = {
    ("rotation_strong", "fresh"): (.86, .08, .06),
    ("rotation_weak", "fresh"): (.10, .12, .78),
    ("rotation_strong", "stale"): (.30, .30, .40),
    ("rotation_weak", "stale"): (.28, .32, .40),
}


def model(*, probe_cost=.003, content_cost=.01):
    return build_source_health_model(
        MATCH, MATCH_PRIOR, HEALTH_PRIOR,
        probe_likelihoods=PROBE, content_likelihoods=CONTENT,
        probe_cost=probe_cost, content_cost=content_cost,
    )


def test_health_probe_has_second_order_value_when_content_is_only_worthwhile_if_fresh():
    finite = model()
    one_step = finite.evaluate(finite.initial_state(), remaining_budget=.08, horizon=1)
    two_step = finite.evaluate(finite.initial_state(), remaining_budget=.08, horizon=2)

    assert one_step["action_id"] == "content_query"
    assert two_step["action_id"] == "source_health_probe"
    assert two_step["planning_net_value"] > 0
    probe = next(row for row in two_step["candidates"] if row["action_id"] == "source_health_probe")
    assert probe["immediate_net_value"] < 0
    assert {branch["observation"]: branch["continuation_action"] for branch in probe["branches"]} == {
        "healthy": "content_query", "degraded": "STOP",
    }


def test_health_observation_changes_the_next_information_choice():
    finite = model()
    healthy = finite.observe(finite.initial_state(), "source_health_probe", "healthy")
    degraded = finite.observe(finite.initial_state(), "source_health_probe", "degraded")

    assert finite.evaluate(healthy, remaining_budget=.077, horizon=1)["action_id"] == "content_query"
    assert finite.evaluate(degraded, remaining_budget=.077, horizon=1)["action_id"] == "STOP"
    # A probe learns about the channel only under the independent initial
    # prior. Its value comes from changed future queries, not reduced match risk.
    assert finite.prediction(healthy) == pytest.approx(finite.prediction(finite.initial_state()))
    assert finite.prediction(degraded) == pytest.approx(finite.prediction(finite.initial_state()))


def test_probe_then_conditional_content_value_matches_direct_joint_enumeration():
    finite = model()
    # Independently integrate the explicit policy: probe, read only if healthy,
    # then predict using the evidence. This does not call the Bellman recursion.
    leaves = [("degraded", None)] + [("healthy", j) for j in range(3)]
    expected_risk = .003  # probe always paid
    for probe_outcome, content_index in leaves:
        probe_index = ("healthy", "degraded").index(probe_outcome)
        joint_weights = {}
        for match, match_prob in MATCH_PRIOR.items():
            for health, health_prob in HEALTH_PRIOR.items():
                weight = match_prob * health_prob * PROBE[health][probe_index]
                if content_index is not None:
                    weight *= CONTENT[(match, health)][content_index]
                joint_weights[(match, health)] = weight
        probability = math.fsum(joint_weights.values())
        prediction = [math.fsum(weight * MATCH[match][y] for (match, _), weight in joint_weights.items())
                      / probability for y in range(3)]
        risk = -math.fsum(p * math.log(p) for p in prediction)
        expected_risk += probability * (risk + (.01 if content_index is not None else 0))
    decision = finite.evaluate(finite.initial_state(), remaining_budget=.08, horizon=2)
    assert decision["action_id"] == "source_health_probe"
    assert decision["expected_total_risk"] == pytest.approx(expected_risk, abs=1e-12)


def test_no_answer_is_distinct_from_a_domain_negative_signal():
    assert model().tools["content_query"].outcomes == ("positive_signal", "no_answer", "negative_signal")


def test_manifest_keeps_finite_model_schema_and_adds_adapter_scope():
    finite = model()
    manifest = manifest_with_health_factors(finite)
    assert manifest["adapter_schema_version"] == VERSION
    assert manifest["model_schema_version"] == finite.VERSION
    assert manifest["latent_factorization"] == "joint_match_state_and_source_health"
    assert "match result" not in manifest["policy_question"]
    restored = finite.from_manifest(manifest)
    assert restored.tools.keys() == finite.tools.keys()


def test_builder_rejects_incomplete_or_malformed_health_laws():
    with pytest.raises(ValueError, match="missing health state"):
        build_source_health_model(
            MATCH, MATCH_PRIOR, HEALTH_PRIOR,
            probe_likelihoods={"fresh": (.9, .1)}, content_likelihoods=CONTENT,
        )
    with pytest.raises(ValueError, match="cover every match/health pair"):
        build_source_health_model(
            MATCH, MATCH_PRIOR, HEALTH_PRIOR,
            probe_likelihoods=PROBE,
            content_likelihoods={("rotation_strong", "fresh"): (.86, .08, .06)},
        )
    with pytest.raises(ValueError, match="must be finite"):
        build_source_health_model(
            MATCH, MATCH_PRIOR, HEALTH_PRIOR,
            probe_likelihoods=PROBE, content_likelihoods=CONTENT,
            probe_cost=float("inf"),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.5", -0.1])
def test_invalid_priors_fail_before_joint_model_construction(value):
    with pytest.raises(ValueError):
        build_source_health_model(
            MATCH, MATCH_PRIOR, {"fresh": value, "stale": .45},
            probe_likelihoods=PROBE, content_likelihoods=CONTENT,
        )


def test_factor_names_with_delimiters_preserve_all_joint_states():
    matches = {"a::b": (.9, .1), "a": (.2, .8)}
    health = {"c": .5, "b::c": .5}
    finite = build_source_health_model(
        matches, {key: .5 for key in matches}, health,
        probe_likelihoods={key: (.5, .5) for key in health},
        content_likelihoods={(match, source): (.4, .2, .4) for match in matches for source in health},
        outcome_labels=("yes", "no"),
    )
    assert len(finite.hypothesis_names) == 4
    assert finite.prediction(finite.initial_state()) == pytest.approx((.55, .45))
