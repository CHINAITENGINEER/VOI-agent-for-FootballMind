"""Constructed probability mechanisms, not evidence of football performance."""
from copy import deepcopy
import json
import math

import pytest

from football_agent.semantic_scenario_model import PLANNING_POLICY_VERSION, SemanticScenarioModel


def proposal():
    names = ("a0", "a1", "b0", "b1")
    forecasts = ((.85, .10, .05), (.10, .15, .75), (.75, .15, .10), (.05, .10, .85))
    laws = {
        "route": ((.8, .1, .1), (.8, .1, .1), (.1, .8, .1), (.1, .8, .1)),
        "read_a": ((.8, .1, .1), (.1, .8, .1), (.45, .45, .1), (.45, .45, .1)),
        "read_b": ((.45, .45, .1), (.45, .45, .1), (.8, .1, .1), (.1, .8, .1)),
    }
    return {"schema": "semantic-scenario-proposal-v1",
            "hypotheses": [{"id": name, "description": name, "prior": .25,
                            "outcome_probabilities": list(forecast)}
                           for name, forecast in zip(names, forecasts)],
            "actions": [{"action_id": name, "question": "Does " + name + " hold?",
                         "observation_group": name,
                         "likelihoods": [{"hypothesis_id": theta, "probabilities": list(law)}
                                         for theta, law in zip(names, rows)]}
                        for name, rows in laws.items()],
            "assumptions": ["Constructed mechanism for mathematical verification only."]}


BASELINE = [.4375, .125, .4375]


def test_alignment_preserves_prior_and_recovers_baseline_by_shared_logit_shift():
    raw = proposal()
    baseline = [.62, .25, .13]
    model = SemanticScenarioModel(raw, baseline)
    result = model.evaluate([], 3)
    assert model.prior == (.25,) * 4
    assert result["prediction_nominal"] == pytest.approx(baseline, abs=1e-9)
    assert result["alignment"]["max_marginal_error"] < 1e-9
    offsets = []
    for supplied, aligned in zip(raw["hypotheses"], model.aligned_outcome_probabilities):
        original = supplied["outcome_probabilities"]
        offsets.append([math.log(aligned[i] / original[i]) - math.log(aligned[2] / original[2])
                        for i in (0, 1)])
    for offset in offsets:
        assert offset == pytest.approx(offsets[0], abs=1e-10)
    assert raw == proposal()
    assert result["statistically_calibrated"] is False


def test_alignment_identity_and_asymmetric_priors():
    raw = proposal()
    weights = [.1, .2, .3, .4]
    for row, weight in zip(raw["hypotheses"], weights):
        row["prior"] = weight
    baseline = [sum(w * h["outcome_probabilities"][j] for w, h in zip(weights, raw["hypotheses"]))
                for j in range(3)]
    model = SemanticScenarioModel(raw, baseline)
    assert model.alignment["iterations"] == 0
    assert model.evaluate([], 3)["posterior_nominal"] == dict(zip(("a0", "a1", "b0", "b1"), weights))


def test_exact_entropy_value_matches_independent_mutual_information_and_lower_envelope():
    model = SemanticScenarioModel(proposal(), BASELINE)
    result = model.evaluate([], 3)
    for row in result["candidates"]:
        nets = []
        for model_id, finite in model.models.items():
            gross = finite.target_mutual_information(finite.initial_state(), row["action_id"])
            assert row["model_values"][model_id]["gross"] == pytest.approx(gross, abs=1e-12)
            nets.append(gross - model.request_loss_cost)
        assert row["lower"] == pytest.approx(min(nets), abs=1e-12)
        assert row["upper"] == pytest.approx(max(nets), abs=1e-12)
        assert row["cost"] == .001
    feasible = result["candidates"]
    expected = max(feasible, key=lambda row: (row["lower"], row["action_id"]))
    assert result["selected_action_id"] == expected["action_id"]
    assert result["lower_upper_interpretation"] == "net_value_envelope_not_confidence_interval"
    assert result["global_stop_certificate"] is False


def complementary_proposal():
    """A semantic wrapper version of the finite XOR continuation mechanism."""
    names = ("00", "01", "10", "11")
    hypotheses = [{"id": name, "description": name, "prior": .25,
                   "outcome_probabilities": list((.9, .05, .05)
                                                  if name[0] == name[1]
                                                  else (.05, .05, .9))}
                  for name in names]
    actions = []
    for index, action_id in enumerate(("context_a", "context_b")):
        likelihoods = []
        for name in names:
            likelihoods.append({"hypothesis_id": name,
                                "probabilities": list((.97, .02, .01)
                                                       if name[index] == "0"
                                                       else (.02, .97, .01))})
        actions.append({"action_id": action_id, "question": action_id,
                        "observation_group": action_id, "likelihoods": likelihoods})
    return {"schema": "semantic-scenario-proposal-v1", "hypotheses": hypotheses,
            "actions": actions, "assumptions": ["Constructed XOR continuation mechanism."]}


def test_bounded_two_step_planning_values_complementary_first_query():
    model = SemanticScenarioModel(complementary_proposal(), [.475, .05, .475],
                                  request_loss_cost=.02, contamination_levels=(0.,),
                                  planning_horizon=2)
    one = model.evaluate([], 4, horizon=1)
    two = model.evaluate([], 4)
    assert one["selected_action_id"] is None
    assert two["selected_action_id"] in {"context_a", "context_b"}
    first = next(row for row in two["candidates"] if row["action_id"] == two["selected_action_id"])
    assert first["net"] == pytest.approx(-.02)
    assert first["planning_net"] > 0
    assert first["planning_lower"] == pytest.approx(first["lower"])
    other = "context_a" if first["action_id"] == "context_b" else "context_b"
    assert set(first["continuation_actions"]) == {other, "STOP"}
    assert two["horizon"] == two["planning_horizon"] == 2


def test_two_step_planning_degrades_to_one_step_when_only_one_request_remains():
    model = SemanticScenarioModel(complementary_proposal(), [.475, .05, .475],
                                  request_loss_cost=.02, contamination_levels=(0.,),
                                  planning_horizon=2)
    one = model.evaluate([], 1, horizon=1)
    two = model.evaluate([], 1)
    assert two["selected_action_id"] == one["selected_action_id"] is None
    assert two["stop_reason"] == one["stop_reason"]
    for left, right in zip(two["candidates"], one["candidates"]):
        assert left["planning_net"] == pytest.approx(right["planning_net"])
        assert left["lower"] == pytest.approx(right["lower"])


def test_two_step_manifest_roundtrip_and_legacy_duplicate_group_guard():
    model = SemanticScenarioModel(complementary_proposal(), [.475, .05, .475],
                                  request_loss_cost=.02, contamination_levels=(0.,),
                                  planning_horizon=2)
    restored = SemanticScenarioModel.from_manifest(json.loads(json.dumps(model.manifest())))
    assert restored.planning_horizon == 2
    assert restored.evaluate([], 4) == model.evaluate([], 4)
    raw = complementary_proposal()
    raw["actions"][1]["observation_group"] = raw["actions"][0]["observation_group"]
    current = SemanticScenarioModel(raw, [.475, .05, .475], planning_horizon=2)
    assert current.evaluate([], 2)["selected_action_id"] is None
    with pytest.raises(ValueError, match="unique observation_group"):
        SemanticScenarioModel(raw, [.475, .05, .475], planning_horizon=2, planning_policy=None)


@pytest.mark.parametrize("restriction", ["unknown", "attempted", "blocked"])
def test_two_step_does_not_price_an_unavailable_complementary_continuation(restriction):
    model = SemanticScenarioModel(complementary_proposal(), [.475, .05, .475],
                                  request_loss_cost=.02, contamination_levels=(0.,), planning_horizon=2)
    arguments = {"observations": [], "remaining_requests": 2}
    if restriction == "unknown":
        arguments["observations"] = [["context_a", "unknown"]]
    elif restriction == "attempted":
        arguments["attempted_action_ids"] = ["context_a"]
    else:
        arguments["blocked_groups"] = ["context_a"]
    result = model.evaluate(**arguments)
    candidate = next(row for row in result["candidates"] if row["action_id"] == "context_b")
    assert result["selected_action_id"] is None
    assert candidate["planning_net"] == pytest.approx(-.02)
    assert candidate["continuation_actions"] == ["STOP"] * 3


@pytest.mark.parametrize("request_costs,remaining", [({}, 1), ({"context_a": 2, "context_b": 2}, 3)])
def test_two_step_request_budget_is_enforced_even_when_loss_cost_is_zero(request_costs, remaining):
    model = SemanticScenarioModel(complementary_proposal(), [.475, .05, .475],
                                  request_loss_cost=0., contamination_levels=(0.,),
                                  action_request_costs=request_costs, planning_horizon=2)
    result = model.evaluate([], remaining)
    assert result["selected_action_id"] is None
    for candidate in result["candidates"]:
        assert candidate["planning_net"] == pytest.approx(0.)
        assert candidate["continuation_actions"] == ["STOP"] * 3


@pytest.mark.parametrize("group_policy", ["single_informative_return", "first_attempt"])
def test_two_step_same_group_locks_on_information_and_respects_no_answer_policy(group_policy):
    raw = proposal()
    for action in raw["actions"]:
        action["observation_group"] = "shared_question"
    model = SemanticScenarioModel(raw, BASELINE, planning_horizon=2,
                                  observation_group_policy=group_policy)
    result = model.evaluate([], 3)
    for candidate in result["candidates"]:
        policy = {branch["observation"]: branch["action_id"] for branch in candidate["continuation_policy"]}
        assert policy["supports"] == policy["refutes"] == "STOP"
        if group_policy == "first_attempt":
            assert policy["no_answer"] == "STOP"
        else:
            assert policy["no_answer"] not in {"STOP", candidate["action_id"]}


def test_two_step_prices_one_public_policy_under_every_model_against_independent_risk_calculation():
    model = SemanticScenarioModel(proposal(), BASELINE, request_loss_cost=.02, planning_horizon=2)
    result = model.evaluate([], 3)
    for candidate in result["candidates"]:
        policy = {branch["observation"]: branch["action_id"] for branch in candidate["continuation_policy"]}
        for model_id, finite in model.models.items():
            initial = finite.initial_state()
            first_action = candidate["action_id"]
            total_risk = finite.tools[first_action].cost
            for first_observation, first_probability in finite.observation_distribution(initial, first_action).items():
                state = finite.observe(initial, first_action, first_observation)
                continuation = policy[first_observation]
                if continuation == "STOP":
                    branch_risk = finite.bayes_risk(state)
                else:
                    branch_risk = finite.tools[continuation].cost + sum(
                        probability * finite.bayes_risk(finite.observe(state, continuation, observation))
                        for observation, probability in finite.observation_distribution(state, continuation).items())
                total_risk += first_probability * branch_risk
            values = candidate["model_values"][model_id]
            assert values["planning_net"] == pytest.approx(finite.bayes_risk(initial) - total_risk, abs=1e-12)
            assert values["continuation_actions"] == [policy[outcome] for outcome in ("supports", "refutes", "no_answer")]
        assert candidate["planning_lower"] == pytest.approx(
            min(values["planning_net"] for values in candidate["model_values"].values()))
    # Nominal quality alone would continue here, but a common robust policy
    # stops under all quality models, so it cannot price an oracle continuation.
    read_a = next(row for row in result["candidates"] if row["action_id"] == "read_a")
    assert read_a["model_values"]["contamination_0"]["continuation_actions"] == ["STOP"] * 3
    assert result["global_policy_optimality_claim"] is False


def test_new_policy_marker_is_explicit_and_old_frozen_two_step_path_remains_replayable():
    current = SemanticScenarioModel(proposal(), BASELINE, request_loss_cost=.02, planning_horizon=2)
    assert current.manifest()["planning_policy"] == PLANNING_POLICY_VERSION
    assert SemanticScenarioModel(proposal(), BASELINE).manifest()["planning_policy"] == PLANNING_POLICY_VERSION
    old_manifest = current.manifest()
    old_manifest.pop("planning_policy")
    restored = SemanticScenarioModel.from_manifest(old_manifest)
    explicit_legacy = SemanticScenarioModel(proposal(), BASELINE, request_loss_cost=.02,
                                           planning_horizon=2, planning_policy=None)
    assert restored.manifest() == old_manifest
    assert restored.model_id == explicit_legacy.model_id
    assert restored.evaluate([], 3) == explicit_legacy.evaluate([], 3)
    old_read_a = next(row for row in restored.evaluate([], 3)["candidates"] if row["action_id"] == "read_a")
    assert old_read_a["model_values"]["contamination_0"]["continuation_actions"] == ["read_b"] * 3
    assert old_read_a["model_values"]["contamination_0.5"]["continuation_actions"] == ["STOP"] * 3
    assert "planning_policy" not in restored.evaluate([], 3)
    invalid = current.manifest()
    invalid["planning_policy"] = "unsupported"
    with pytest.raises(ValueError, match="planning_policy"):
        SemanticScenarioModel.from_manifest(invalid)


def test_two_step_action_extension_preserves_unknown_history_and_allows_group_fallback_only():
    raw = complementary_proposal()
    fallback = deepcopy(raw["actions"][0])
    fallback["action_id"] = "backup_a"
    raw["actions"].append(fallback)
    model = SemanticScenarioModel(raw, [.475, .05, .475], request_loss_cost=.02,
                                  contamination_levels=(0.,), planning_horizon=2)
    result = model.evaluate([["context_a", "unknown"]], 2)
    candidate = next(row for row in result["candidates"] if row["action_id"] == "context_b")
    assert candidate["planning_net"] > 0
    assert set(candidate["continuation_actions"]) == {"backup_a", "STOP"}
    assert all("context_a" not in row.get("continuation_actions", ()) for row in result["candidates"])
    informed = model.evaluate([["context_a", "supports"]], 2)
    assert next(row for row in informed["candidates"] if row["action_id"] == "backup_a")["feasible"] is False
    assert all("backup_a" not in row.get("continuation_actions", ()) for row in informed["candidates"])


def test_branch_risk_diagnostics_are_conditional_and_do_not_change_selection():
    model = SemanticScenarioModel(proposal(), BASELINE)
    result = model.evaluate([], 3)
    assert result["branch_risk_diagnostics"]["version"] == "branch-risk-diagnostics-v1"
    chosen = next(row for row in result["candidates"] if row["action_id"] == result["selected_action_id"])
    assert 0 <= chosen["branch_risk_increase_probability"] <= 1
    assert chosen["expected_risk_change"] == pytest.approx(-chosen["gross"], abs=1e-12)
    assert chosen["risk_diagnostics"]["nominal"]["branch_risk_increase_probability"] == pytest.approx(
        chosen["branch_risk_increase_probability"], abs=1e-12)
    assert chosen["risk_diagnostics"]["robust"]["branch_risk_increase_probability_lower"] <= chosen[
        "risk_diagnostics"]["robust"]["branch_risk_increase_probability_upper"]
    for model_id, values in chosen["model_values"].items():
        assert values["risk_increase_branches"] == [
            branch["observation"] for branch in next(
                row for row in model.models[model_id].evaluate(model.models[model_id].initial_state(), remaining_budget=3)["candidates"]
                if row["action_id"] == chosen["action_id"]
            )["branches"] if branch["probability"] > 0 and branch["terminal_risk"] >
            model.models[model_id].evaluate(model.models[model_id].initial_state(), remaining_budget=3)["current_risk"] + 1e-12
        ]
    # Diagnostics are metadata only: the action selected by the exact solver
    # remains the same as the maximum robust NetVoI candidate.
    expected = max((row for row in result["candidates"] if row["feasible"]),
                   key=lambda row: (row["lower"], row["action_id"]))
    assert result["selected_action_id"] == expected["action_id"]


def test_pre_extension_manifest_replays_without_new_diagnostic_fields():
    model = SemanticScenarioModel(proposal(), BASELINE)
    old_manifest = model.manifest()
    old_manifest.pop("branch_risk_diagnostics")
    restored = SemanticScenarioModel.from_manifest(old_manifest)
    result = restored.evaluate([], 3)
    assert "branch_risk_diagnostics" not in result
    assert all("branch_risk_increase_probability" not in row for row in result["candidates"])
    assert restored.manifest() == old_manifest


def test_no_answer_changes_neither_posterior_nor_prediction_in_any_declared_model():
    model = SemanticScenarioModel(proposal(), BASELINE)
    before = model.evaluate([["route", "supports"]], 2)
    after = model.evaluate([["route", "supports"], ["read_a", "no_answer"]], 1)
    for model_id in model.models:
        assert after["model_predictions"][model_id] == pytest.approx(before["model_predictions"][model_id])
        assert after["model_posteriors"][model_id] == pytest.approx(before["model_posteriors"][model_id])
    assert after["observations"][-1]["updates_belief"] is False
    assert next(row for row in after["candidates"] if row["action_id"] == "read_a")["feasible"] is False


def test_unknown_return_blocks_repeat_without_belief_update():
    model = SemanticScenarioModel(proposal(), BASELINE)
    result = model.evaluate([{"action_id": "route", "outcome": "unknown"}], 2)
    assert result["prediction_nominal"] == pytest.approx(BASELINE)
    assert result["posterior_nominal"] == {name: .25 for name in ("a0", "a1", "b0", "b1")}
    assert next(row for row in result["candidates"] if row["action_id"] == "route")["feasible"] is False


def test_different_real_observation_values_can_change_which_remaining_information_is_selected():
    model = SemanticScenarioModel(proposal(), BASELINE)
    supports = model.evaluate([["route", "supports"]], 2)
    refutes = model.evaluate([["route", "refutes"]], 2)
    assert supports["selected_action_id"] == "read_a"
    assert refutes["selected_action_id"] == "read_b"
    assert supports["prediction_nominal"] != refutes["prediction_nominal"]
    # Different declared quality laws must replay the observation separately.
    assert supports["model_posteriors"]["contamination_0"] != supports["model_posteriors"]["contamination_0.5"]


def test_declared_fully_uninformative_model_prevents_positive_robust_query():
    model = SemanticScenarioModel(proposal(), BASELINE, contamination_levels=(0., 1.))
    result = model.evaluate([], 3)
    assert result["selected_action_id"] is None
    assert result["stop_reason"] == "no_positive_declared_lower_net_value"
    assert result["stop_scope"] == "declared_model_set_and_visible_menu"
    for row in result["candidates"]:
        assert row["lower"] == pytest.approx(-.001)
    assert result["global_stop_certificate"] is False


def test_same_group_actions_can_have_different_laws_but_are_never_double_consumed():
    raw = proposal()
    raw["actions"][1]["observation_group"] = "route"
    model = SemanticScenarioModel(raw, BASELINE)
    result = model.evaluate([["route", "supports"]], 2)
    assert next(row for row in result["candidates"] if row["action_id"] == "read_a")["reason"] == (
        "observation_group_already_informative_or_blocked")
    with pytest.raises(ValueError, match="one-shot"):
        model.evaluate([["route", "supports"], ["read_a", "refutes"]], 1)
    with pytest.raises(ValueError, match="one-shot"):
        model.evaluate([["read_b", "unknown"], ["read_b", "supports"]], 1)


def test_attempted_and_blocked_inputs_prevent_retries_even_without_an_observation():
    model = SemanticScenarioModel(proposal(), BASELINE)
    result = model.evaluate([], 2, attempted_action_ids=["read_a"], blocked_groups=["read_b"])
    rows = {row["action_id"]: row for row in result["candidates"]}
    assert rows["read_a"]["feasible"] is False
    assert rows["read_b"]["feasible"] is False
    assert rows["route"]["feasible"] is True
    with pytest.raises(ValueError, match="unknown action"):
        model.evaluate([], 2, attempted_action_ids=["invented"])


def test_budget_stop_is_resource_stop_even_when_cost_is_zero():
    model = SemanticScenarioModel(proposal(), BASELINE, request_loss_cost=0.)
    result = model.evaluate([], 0)
    assert result["selected_action_id"] is None
    assert result["stop_reason"] == "request_budget_exhausted"
    assert not any(row["feasible"] for row in result["candidates"])


def test_manifest_json_roundtrip_reproduces_entire_decision_and_preserves_legacy_labels():
    model = SemanticScenarioModel(proposal(), BASELINE)
    manifest = json.loads(json.dumps(model.manifest(), allow_nan=False))
    rebuilt = SemanticScenarioModel.from_manifest(manifest)
    assert model.evaluate([["route", "supports"]], 2) == rebuilt.evaluate([["route", "supports"]], 2)
    assert model.models["contamination_0"].manifest()["evidence_class"] == "SYNTHETIC_ASSUMPTIONS"
    assert model.robust_model.evaluate(remaining_budget=.002)["evidence_class"] == "SYNTHETIC_ASSUMPTIONS"
    assert model.evaluate([], 3)["evidence_basis"] == "pretrained_semantic_model_assumptions_with_real_observations"


@pytest.mark.parametrize("value", [0., -0.1, float("nan"), float("inf"), True])
def test_invalid_probability_values_fail(value):
    raw = proposal()
    raw["hypotheses"][0]["outcome_probabilities"][0] = value
    with pytest.raises(ValueError):
        SemanticScenarioModel(raw, BASELINE)


def test_state_dependent_no_answer_and_missing_or_duplicate_hypotheses_fail():
    raw = proposal()
    raw["actions"][0]["likelihoods"][0]["probabilities"] = [.7, .1, .2]
    with pytest.raises(ValueError, match="equal across"):
        SemanticScenarioModel(raw, BASELINE)
    raw = proposal()
    raw["actions"][0]["likelihoods"][0]["hypothesis_id"] = "a1"
    with pytest.raises(ValueError, match="uniquely cover"):
        SemanticScenarioModel(raw, BASELINE)
    raw = proposal()
    raw["actions"][0]["likelihoods"].pop()
    with pytest.raises(ValueError, match="exactly cover"):
        SemanticScenarioModel(raw, BASELINE)


@pytest.mark.parametrize("levels", [(0.25, .5), (0., 0.), (0., 1.1), (True, 0.)])
def test_invalid_contamination_set_fails(levels):
    with pytest.raises(ValueError):
        SemanticScenarioModel(proposal(), BASELINE, contamination_levels=levels)


def test_contamination_uses_frozen_prior_marginal_after_observing_and_does_not_rebase():
    model = SemanticScenarioModel(proposal(), BASELINE)
    before = deepcopy(model.manifest())
    model.evaluate([["route", "supports"]], 2)
    assert model.manifest() == before
    for action_id, law in model.likelihoods.items():
        for theta in model.hypothesis_ids:
            for z in range(3):
                prior_marginal = sum(.25 * law[other][z] for other in model.hypothesis_ids)
                expected = .5 * law[theta][z] + .5 * prior_marginal
                actual = model.models["contamination_0.5"].tools[action_id].likelihoods[theta][z]
                assert actual == pytest.approx(expected)


def test_accumulated_action_menu_can_exceed_a_single_eight_action_batch():
    raw = proposal()
    source = deepcopy(raw["actions"][-1])
    for i in range(12):
        action = deepcopy(source)
        action["action_id"] = f"expanded_{i}"
        action["observation_group"] = f"expanded_group_{i}"
        raw["actions"].append(action)
    model = SemanticScenarioModel(raw, BASELINE)
    result = model.evaluate([["route", "supports"]], 2)
    rebuilt = SemanticScenarioModel.from_manifest(json.loads(json.dumps(model.manifest())))
    assert len(result["candidates"]) == 15
    assert rebuilt.evaluate([["route", "supports"]], 2) == result
    original = SemanticScenarioModel(proposal(), BASELINE)
    assert model.prior == original.prior
    assert model.aligned_outcome_probabilities == original.aligned_outcome_probabilities
    assert result["posterior_nominal"] == original.evaluate([["route", "supports"]], 2)["posterior_nominal"]


@pytest.mark.parametrize("unit_cost", [.001, 0.])
def test_two_request_action_is_infeasible_with_only_one_request_even_if_loss_cost_is_zero(unit_cost):
    model = SemanticScenarioModel(proposal(), BASELINE, request_loss_cost=unit_cost,
                                  action_request_costs={"read_a": 2})
    result = model.evaluate([["route", "supports"]], 1, blocked_groups=["read_b"])
    candidate = next(row for row in result["candidates"] if row["action_id"] == "read_a")
    assert candidate["request_cost"] == 2
    assert candidate["cost"] == unit_cost * 2
    assert candidate["feasible"] is False
    assert result["selected_action_id"] is None
    assert result["stop_reason"] == "insufficient_request_budget"
    assert result["global_stop_certificate"] is False


def test_heterogeneous_costs_match_actual_solver_cost_and_survive_manifest_roundtrip():
    model = SemanticScenarioModel(proposal(), BASELINE, action_request_costs={"read_a": 2, "read_b": 3})
    result = model.evaluate([["route", "supports"]], 3)
    for candidate in result["candidates"]:
        if candidate["feasible"]:
            for value in candidate["model_values"].values():
                assert value["net"] == pytest.approx(value["gross"] - candidate["cost"])
            for finite in model.models.values():
                assert finite.tools[candidate["action_id"]].cost == candidate["cost"]
    restored = SemanticScenarioModel.from_manifest(json.loads(json.dumps(model.manifest())))
    assert restored.evaluate([["route", "supports"]], 3) == result
    default = SemanticScenarioModel(proposal(), BASELINE)
    old_manifest = default.manifest()
    old_manifest.pop("action_request_costs")
    assert SemanticScenarioModel.from_manifest(old_manifest).evaluate([], 3) == default.evaluate([], 3)
    assert all(row["request_cost"] == 1 for row in default.evaluate([], 3)["candidates"])


@pytest.mark.parametrize("costs", [{"unknown": 1}, {"route": 0}, {"route": 2.0}, {"route": True}, {"route": -1}])
def test_invalid_request_cost_mapping_fails(costs):
    with pytest.raises(ValueError, match="action_request_costs"):
        SemanticScenarioModel(proposal(), BASELINE, action_request_costs=costs)


@pytest.mark.parametrize("inert_outcome", ["no_answer", "unknown"])
def test_different_action_in_same_group_can_follow_inert_return_then_group_locks_after_information(inert_outcome):
    raw = proposal()
    for action in raw["actions"]:
        action["observation_group"] = "shared_question"
    model = SemanticScenarioModel(raw, BASELINE)
    before = model.evaluate([], 3)
    failed = model.evaluate([["route", inert_outcome]], 2)
    assert failed["model_posteriors"] == before["model_posteriors"]
    rows = {row["action_id"]: row for row in failed["candidates"]}
    assert rows["route"]["feasible"] is False
    assert rows["read_a"]["feasible"] is True
    assert rows["read_b"]["feasible"] is True
    observed = model.evaluate([["route", inert_outcome], ["read_a", "supports"]], 1)
    direct = model.evaluate([["read_a", "supports"]], 1)
    for model_id in model.models:
        assert observed["model_posteriors"][model_id] == pytest.approx(direct["model_posteriors"][model_id])
    assert observed["posterior_nominal"] != before["posterior_nominal"]
    assert len(observed["observations"]) == 2
    assert not any(row["feasible"] for row in observed["candidates"])
    assert next(row for row in observed["candidates"] if row["action_id"] == "read_b")["reason"] == (
        "observation_group_already_informative_or_blocked")
    with pytest.raises(ValueError, match="already informative"):
        model.evaluate([["route", inert_outcome], ["read_a", "supports"], ["read_b", "refutes"]], 0)


def test_multiple_same_group_inert_actions_keep_all_history_without_duplicate_evidence():
    raw = proposal()
    for action in raw["actions"]:
        action["observation_group"] = "shared_question"
    model = SemanticScenarioModel(raw, BASELINE)
    initial = model.evaluate([], 3)
    result = model.evaluate([["route", "no_answer"], ["read_a", "no_answer"]], 1)
    assert len(result["observations"]) == 2
    for model_id in model.models:
        assert result["model_posteriors"][model_id] == pytest.approx(initial["model_posteriors"][model_id])
    assert next(row for row in result["candidates"] if row["action_id"] == "read_b")["feasible"] is True
    terminal = model.evaluate([["route", "no_answer"], ["read_a", "no_answer"], ["read_b", "refutes"]], 0)
    direct = model.evaluate([["read_b", "refutes"]], 0)
    assert terminal["prediction_nominal"] == pytest.approx(direct["prediction_nominal"])
    attempted = model.evaluate([], 2, attempted_action_ids=["route"])
    assert next(row for row in attempted["candidates"] if row["action_id"] == "read_a")["feasible"] is True


def test_old_manifest_preserves_first_attempt_group_rule_and_old_output_labels():
    raw = proposal()
    for action in raw["actions"]:
        action["observation_group"] = "same_group"
    legacy = SemanticScenarioModel(raw, BASELINE, observation_group_policy="first_attempt")
    manifest = legacy.manifest()
    assert "observation_group_policy" not in manifest
    restored = SemanticScenarioModel.from_manifest(manifest)
    assert restored.manifest() == manifest
    assert restored.model_id == legacy.model_id
    result = restored.evaluate([["route", "no_answer"]], 2)
    assert result == legacy.evaluate([["route", "no_answer"]], 2)
    assert not any(row["feasible"] for row in result["candidates"])
    assert next(row for row in result["candidates"] if row["action_id"] == "read_a")["reason"] == (
        "observation_group_already_attempted_or_blocked")
    assert "Each observation group is acquired at most once; residual cross-group dependence is not modeled." in result["assumptions"]
    current = SemanticScenarioModel(raw, BASELINE)
    assert current.manifest()["observation_group_policy"] == "single_informative_return"
    assert current.model_id != restored.model_id
