from copy import deepcopy
from dataclasses import replace
import math

import pytest

from football_agent.decision_value import (
    BeliefState, FiniteDecisionModel, ObservationTool, entropy, simulate_episode, synthetic_scenarios,
)
from football_agent.contextual_voi_report import build_contextual_report
from football_agent.verify_round6 import replay_trace, run_verification


def perfect_model(*, cost=.01, loss="log_loss"):
    return FiniteDecisionModel({"a": (1, 0, 0), "b": (0, 0, 1)}, {"a": .5, "b": .5},
                               {"reveal": ObservationTool(("a", "b", "impossible"), {"a": (1, 0, 0), "b": (0, 1, 0)}, cost)}, loss=loss)


def candidate(decision, action_id):
    return next(c for c in decision["candidates"] if c["action_id"] == action_id)


def test_exact_evsi_uses_every_possible_response_and_matches_information_identity():
    m = perfect_model()
    d = m.evaluate(m.initial_state(), remaining_budget=.02)
    c = candidate(d, "reveal")
    assert d["action_id"] == "reveal"
    assert c["immediate_evsi"] == pytest.approx(math.log(2))
    assert c["immediate_evsi"] == pytest.approx(m.target_mutual_information(m.initial_state(), "reveal"))
    assert c["planning_net_value"] == pytest.approx(math.log(2) - .01)
    assert [b["probability"] for b in c["branches"]] == [.5, .5, 0]
    assert c["branches"][-1]["status"] == "impossible"
    assert m.prior == (.5, .5)


def test_brier_is_a_distinct_loss_with_explicit_cost_units():
    m = perfect_model(loss="brier")
    d = m.evaluate(m.initial_state(), remaining_budget=.02)
    assert candidate(d, "reveal")["immediate_evsi"] == pytest.approx(.5)
    assert d["cost_unit"] == "expected_multiclass_brier"


def test_contextual_report_keeps_qualification_from_deterministically_implying_rotation():
    cases = synthetic_scenarios()
    report = build_contextual_report({
        "rotation_unknown": cases["qualified_rotation_unknown"],
        "reserve_strength_known": cases["qualified_reserves_known_strong"],
    })
    rows = {row["scenario_id"]: row for row in report["scenarios"]}
    assert rows["rotation_unknown"]["context"]["qualification"] == "already_qualified"
    assert rows["rotation_unknown"]["selected_action"] == "lineup"
    assert rows["reserve_strength_known"]["selected_action"] == "STOP"
    assert report["hidden_result_included"] is False


def test_contextual_report_prioritizes_tactics_when_draw_is_sufficient():
    scenario = synthetic_scenarios()["draw_sufficient_tactics_unknown"]
    report = build_contextual_report({"draw_sufficient": scenario})
    row = report["scenarios"][0]
    assert row["context"]["qualification_condition"] == "draw_sufficient"
    assert row["selected_action"] == "tactical_evidence"
    assert row["hidden_result_included"] is False


def test_contextual_report_stops_on_irrelevant_information_even_when_state_is_unknown():
    scenario = synthetic_scenarios()["irrelevant_high_entropy_news"]
    row = build_contextual_report({"irrelevant": scenario})["scenarios"][0]
    assert row["context"]["news_status"] == "unknown"
    assert row["selected_action"] == "STOP"
    assert row["candidates"][0]["immediate_evsi"] == 0.0


def test_budget_horizon_and_economic_stop_include_stop_option():
    m = perfect_model(cost=.8)
    assert m.evaluate(m.initial_state(), remaining_budget=1)["action_id"] == "STOP"
    assert m.evaluate(m.initial_state(), remaining_budget=.5)["reason"] == "no_feasible_actions"
    assert m.evaluate(m.initial_state(), remaining_budget=1, horizon=0)["reason"] == "horizon_exhausted"
    assert perfect_model(cost=0).evaluate(perfect_model(cost=0).initial_state(), remaining_budget=0)["action_id"] == "reveal"


@pytest.mark.parametrize("horizon", [-1, 3, True, 1.5])
def test_horizon_is_explicitly_bounded(horizon):
    m = perfect_model()
    with pytest.raises(ValueError):
        m.evaluate(m.initial_state(), remaining_budget=1, horizon=horizon)


def test_actual_observation_updates_only_after_acquisition_and_rejects_impossible_response():
    m = perfect_model()
    initial = m.initial_state()
    assert m.prediction(initial) == (.5, 0, .5)
    after = m.observe(initial, "reveal", "a")
    assert m.prediction(after) == (1, 0, 0)
    assert m.prediction(initial) == (.5, 0, .5)
    with pytest.raises(ValueError, match="Impossible observation"):
        m.observe(initial, "reveal", "impossible")
    with pytest.raises(ValueError, match="Unknown observation"):
        m.observe(initial, "reveal", "unknown")
    with pytest.raises(ValueError, match="Unknown tool"):
        m.observe(initial, "missing", "a")
    with pytest.raises(ValueError, match="one-shot"):
        m.observe(after, "reveal", "a")


def test_unearned_posterior_or_impossible_history_rejected():
    m = perfect_model()
    with pytest.raises(ValueError, match="Posterior must match"):
        m.prediction(BeliefState((1, 0)))
    with pytest.raises(ValueError, match="impossible"):
        m.prediction(BeliefState((1, 0), (("reveal", "impossible"),)))


@pytest.mark.parametrize("probabilities", [(.6, .6, 0), (-.1, .1, 1), (float("nan"), 0, 1), (True, 0, 0), (.5, .5)])
def test_invalid_probability_rows_rejected(probabilities):
    with pytest.raises(ValueError):
        FiniteDecisionModel({"a": probabilities}, {"a": 1}, {})


@pytest.mark.parametrize("cost", [-.1, float("inf"), True, "0.1"])
def test_invalid_costs_rejected(cost):
    with pytest.raises(ValueError):
        perfect_model(cost=cost)


def test_prior_likelihood_keys_and_unit_contracts():
    with pytest.raises(ValueError, match="Prior keys"):
        FiniteDecisionModel({"a": (1, 0, 0)}, {"b": 1}, {})
    with pytest.raises(ValueError, match="likelihoods"):
        FiniteDecisionModel({"a": (1, 0, 0)}, {"a": 1}, {"x": ObservationTool(("ok",), {"b": (1,)}, .01)})
    with pytest.raises(ValueError, match="cost unit"):
        FiniteDecisionModel({"a": (1, 0, 0)}, {"a": 1}, {}, cost_unit="dollars")
    with pytest.raises(ValueError, match="outcomes must be unique"):
        FiniteDecisionModel({"a": (1, 0, 0)}, {"a": 1}, {"x": ObservationTool(("ok", "ok"), {"a": (.5, .5)}, .01)})


def test_qualification_context_conditions_value_not_an_automatic_strength_penalty():
    cases = synthetic_scenarios()
    unknown = cases["qualified_rotation_unknown"]["model"]
    known = cases["qualified_reserves_known_strong"]["model"]
    a = unknown.evaluate(unknown.initial_state(), remaining_budget=.08)
    b = known.evaluate(known.initial_state(), remaining_budget=.08)
    assert a["action_id"] == "lineup"
    assert b["action_id"] == "STOP"
    assert candidate(a, "lineup")["immediate_evsi"] > candidate(b, "lineup")["immediate_evsi"]
    starters = unknown.observe(unknown.initial_state(), "lineup", "starters")
    reserves = unknown.observe(unknown.initial_state(), "lineup", "reserves")
    assert unknown.evaluate(starters, remaining_budget=.065)["action_id"] == "STOP"
    assert unknown.evaluate(reserves, remaining_budget=.065)["action_id"] == "reserve_quality"
    assert unknown.hypothesis_names[0] == "starters_strong"


def test_draw_sufficiency_is_not_an_observation_of_defensive_tactics():
    m = synthetic_scenarios()["draw_sufficient_tactics_unknown"]["model"]
    assert 0 < m.prior[0] < 1
    assert m.evaluate(m.initial_state(), remaining_budget=.08)["action_id"] == "tactical_evidence"
    after = m.observe(m.initial_state(), "tactical_evidence", "attacking_signal")
    assert 0 < after.belief[0] < m.prior[0]


def test_latent_entropy_reduction_is_not_target_information_value():
    m = synthetic_scenarios()["irrelevant_high_entropy_news"]["model"]
    initial = m.initial_state()
    assert entropy(initial.belief) == pytest.approx(math.log(2))
    after = m.observe(initial, "news", "a")
    assert entropy(after.belief) == 0
    assert m.prediction(initial) == m.prediction(after)
    assert m.target_mutual_information(initial, "news") == pytest.approx(0)
    assert m.evaluate(initial, remaining_budget=.08, horizon=2)["action_id"] == "STOP"


def test_correlated_reports_are_conditioned_on_shared_latent_not_multiplied_as_new_facts():
    m = synthetic_scenarios()["duplicate_shared_report"]["model"]
    first = m.evaluate(m.initial_state(), remaining_budget=.08)
    assert candidate(first, "report_a")["immediate_evsi"] > 0
    after = m.observe(m.initial_state(), "report_a", "positive")
    followup = m.evaluate(after, remaining_budget=.075, horizon=2)
    assert candidate(followup, "report_b")["immediate_evsi"] == pytest.approx(0)
    assert followup["action_id"] == "STOP"
    with pytest.raises(ValueError, match="Impossible"):
        m.observe(after, "report_b", "negative")


def test_same_evidence_id_is_blocked_even_for_differently_named_noisy_tools():
    tool = ObservationTool(("a", "b"), {"a": (.8, .2), "b": (.2, .8)}, .01, evidence_id="report:1")
    m = FiniteDecisionModel({"a": (1, 0, 0), "b": (0, 0, 1)}, {"a": .5, "b": .5}, {"source_1": tool, "source_2": tool})
    after = m.observe(m.initial_state(), "source_1", "a")
    d = m.evaluate(after, remaining_budget=.08, horizon=2)
    assert candidate(d, "source_2")["reason"] == "evidence_already_observed"
    with pytest.raises(ValueError, match="Evidence was already observed"):
        m.observe(after, "source_2", "a")
    assert FiniteDecisionModel.from_manifest(m.manifest()).manifest() == m.manifest()
    with pytest.raises(ValueError, match="same evidence ID"):
        FiniteDecisionModel({"a": (1, 0, 0), "b": (0, 0, 1)}, {"a": .5, "b": .5},
                            {"source_1": tool, "source_2": replace(tool, likelihoods={"a": (1, 0), "b": (0, 1)})})


def test_two_step_bellman_finds_complementarity_and_respects_total_budget():
    m = synthetic_scenarios()["complementary_xor_reference"]["model"]
    initial = m.initial_state()
    one = m.evaluate(initial, remaining_budget=.04, horizon=1)
    two = m.evaluate(initial, remaining_budget=.04, horizon=2)
    assert one["action_id"] == "STOP"
    assert two["action_id"] == "context_a"
    assert candidate(two, "context_a")["immediate_net_value"] == pytest.approx(-.02)
    assert candidate(two, "context_a")["planning_net_value"] > .4
    assert all(b["continuation_action"] == "context_b" for b in candidate(two, "context_a")["branches"])
    assert m.evaluate(initial, remaining_budget=.039, horizon=2)["action_id"] == "STOP"


def test_reproducible_sampling_replay_and_matched_outcomes_across_policies():
    m = synthetic_scenarios()["complementary_xor_reference"]["model"]
    a = simulate_episode(m, seed=73, budget=.04, horizon=1)
    b = simulate_episode(m, seed=73, budget=.04, horizon=2)
    assert b == simulate_episode(m, seed=73, budget=.04, horizon=2)
    assert a["evaluator_only"]["sampled_result"] == b["evaluator_only"]["sampled_result"]
    assert a["evaluator_only"]["sampled_hypothesis"] == b["evaluator_only"]["sampled_hypothesis"]
    assert replay_trace(m, b)["actions"] == 2
    assert b["spent"] == pytest.approx(.04)


def test_match_outcome_is_evaluator_only_and_cannot_rewrite_prior_or_acquisition_trace():
    m = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    manifest = deepcopy(m.manifest())
    original = simulate_episode(m, seed=7, budget=.08, horizon=2)
    altered = deepcopy(original)
    altered["evaluator_only"] = {"sampled_result": "away", "realized_loss": 1000}
    assert replay_trace(m, original) == replay_trace(m, altered)
    assert m.manifest() == manifest
    assert "result" not in original["trace"][0]["decision"]
    tampered = deepcopy(original)
    tampered["trace"][0]["decision"]["action_id"] = "STOP"
    with pytest.raises(ValueError, match="decision differs"):
        replay_trace(m, tampered)


def test_verifier_persists_manifest_and_sixty_constructed_replays(tmp_path):
    result = run_verification(tmp_path)
    assert result["scenarios"] == 6
    assert result["replayed_episodes"] == 60
    assert result["independent_real_matches"] == 0
    assert result["evidence_class"] == "SYNTHETIC_ASSUMPTIONS"
    assert (tmp_path / "manifest.json").is_file()
    assert len((tmp_path / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()) == 60
