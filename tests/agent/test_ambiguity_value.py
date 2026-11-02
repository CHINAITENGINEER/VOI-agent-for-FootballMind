"""Shared executable policies, model mismatch and limited stopping claims."""
from copy import deepcopy
from itertools import product
import json
import math

import pytest

from football_agent.ambiguity_value import FiniteAmbiguityDecisionModel, LIMITS
from football_agent.decision_loss import FiniteDecisionLossModel
from football_agent.decision_value import ObservationTool


LOSSES = {"yes": (0, 1), "no": (1, 0), "defer": (.2, .2)}


def model(prior=None, likelihoods=None, *, cost=.05, losses=None, tool_name="check", evidence_id=None):
    return FiniteDecisionLossModel(
        {"yes": (1, 0), "no": (0, 1)}, prior or {"yes": .5, "no": .5},
        {tool_name: ObservationTool(("yes", "no", "failure"), likelihoods or
                                   {"yes": (1, 0, 0), "no": (0, 1, 0)}, cost, evidence_id=evidence_id)},
        outcome_labels=("yes", "no"), decision_losses=losses or LOSSES,
        cost_unit="expected_report_loss")


def candidate(result, action="check"):
    return next(c for c in result["candidates"] if c["action_id"] == action)


def identification():
    return FiniteAmbiguityDecisionModel({
        "zero": model({"yes": 1, "no": 0}),
        "one": model({"yes": 0, "no": 1}),
    })


@pytest.mark.parametrize("cost", [0, .05, .3])
def test_single_model_reduces_to_existing_terminal_evsi(cost):
    m = model(cost=cost)
    ordinary = m.evaluate(m.initial_state(), remaining_budget=1)
    robust = FiniteAmbiguityDecisionModel({"only": m}).evaluate((), remaining_budget=1)
    a, b = candidate(ordinary), candidate(robust)
    assert robust["current_risk"] == pytest.approx(ordinary["current_risk"])
    assert robust["action_id"] == ordinary["action_id"]
    assert b["robust_net_value"] == pytest.approx(a["immediate_net_value"])
    assert b["worst_expected_risk"] == pytest.approx(a["expected_total_risk"])
    assert b["dominance_upper_gain"] == pytest.approx(a["immediate_net_value"])


def test_model_identification_has_value_when_every_oracle_model_evsi_is_zero():
    m = identification()
    for single in m.models.values():
        assert candidate(single.evaluate(single.initial_state(), remaining_budget=1))["immediate_net_value"] == pytest.approx(-.05)
    result = m.evaluate((), remaining_budget=1)
    q = candidate(result)
    assert result["terminal_decision_if_stopping"]["decision"] == "defer"
    assert result["current_risk"] == pytest.approx(.2)
    assert q["response_policy"]["yes"] == "yes" and q["response_policy"]["no"] == "no"
    assert q["per_model_expected_risk"] == pytest.approx({"zero": .05, "one": .05})
    assert q["robust_net_value"] == pytest.approx(.15)
    assert q["dominance_upper_gain"] == pytest.approx(.15)
    failure = next(branch for branch in q["branches"] if branch["observation"] == "failure")
    assert failure["status"] == "IMPOSSIBLE_IN_ALL_ACTIVE_MODELS"
    assert result["action_id"] == "check"
    assert result["statistical_confidence_level"] is None
    assert not result["global_open_world_stop_proven"]


def test_worst_risk_improvement_is_not_a_positive_gain_in_each_model():
    loss = {"base": (1, .6), "report_a": (0, 2), "report_b": (2, 0)}
    likelihood = {"yes": (.625, .375, 0), "no": (.325, .675, 0)}
    m = FiniteAmbiguityDecisionModel({
        "m1": model({"yes": 1, "no": 0}, likelihood, losses=loss),
        "m2": model({"yes": 0, "no": 1}, likelihood, losses=loss),
    })
    result = m.evaluate((), remaining_budget=1)
    q = candidate(result)
    assert result["terminal_decision_if_stopping"]["decision"] == "base"
    assert q["response_policy"]["yes"] == "report_a" and q["response_policy"]["no"] == "report_b"
    assert q["per_model_expected_risk"] == pytest.approx({"m1": .8, "m2": .7})
    assert q["robust_net_value"] == pytest.approx(.2)
    assert q["per_model_net_gain"] == pytest.approx({"m1": .2, "m2": -.1})
    assert q["net_gain_spread"]["minimum"] == pytest.approx(-.1)


def failure_model(shared):
    hypotheses, prior, first, refresh = {}, {}, {}, {}
    for y, a, b in product((0, 1), repeat=3):
        if shared and a != b:
            continue
        theta = f"y{y}_a{a}_b{b}"
        hypotheses[theta] = (1-y, y)
        prior[theta] = .5*(.8 if a else .2)*(1 if shared else (.8 if b else .2))
        first[theta] = (int(a and not y), int(a and y), int(not a))
        refresh[theta] = (int(b and not y), int(b and y), int(not b))
    return FiniteDecisionLossModel(hypotheses, prior, {
        "check": ObservationTool(("yes", "no", "failure"), first, .05, evidence_id="source:first"),
        "refresh": ObservationTool(("yes", "no", "failure"), refresh, .05, evidence_id="source:refresh")},
        outcome_labels=("yes", "no"), decision_losses=LOSSES, cost_unit="expected_report_loss")


def test_failure_updates_model_specific_availability_without_fabricating_negative_answer():
    m = FiniteAmbiguityDecisionModel({"shared": failure_model(True), "independent": failure_model(False)})
    initial = m.evaluate((), remaining_budget=1)
    after = m.evaluate((("check", "failure"),), remaining_budget=.95)
    assert candidate(initial, "refresh")["robust_net_value"] == pytest.approx(.11)
    assert after["active_model_ids"] == ["shared", "independent"]
    for state in after["model_states"].values():
        assert state["prediction"] == pytest.approx([.5, .5])
        assert state["history_log_likelihood"] == pytest.approx(math.log(.2))
    q = candidate(after, "refresh")
    failure = next(b for b in q["branches"] if b["observation"] == "failure")
    assert failure["per_model_probability"] == pytest.approx({"shared": 1, "independent": .2})
    assert q["robust_net_value"] == pytest.approx(-.05)
    assert q["dominance_upper_gain"] == pytest.approx(.11)
    assert after["action_id"] == "STOP"
    assert after["stop_certificate"] == "NO_ROBUST_ONE_STEP_IMPROVEMENT_IN_DECLARED_MODEL_SET"
    assert after["dominance_stop_certificate"] is None


def test_exact_zero_excludes_one_model_and_does_not_restart_its_prior():
    result = identification().evaluate((("check", "yes"),), remaining_budget=1)
    assert result["active_model_ids"] == ["zero"]
    assert result["excluded_model_ids"] == ["one"]
    excluded = result["model_states"]["one"]
    assert excluded["status"] == "INCOMPATIBLE_WITH_OBSERVED_HISTORY"
    assert excluded["excluded_at_observation"] == 0 and excluded["posterior"] is None
    assert result["terminal_decision_if_stopping"]["decision"] == "yes"
    assert result["current_risk"] == 0
    assert candidate(result)["reason"] == "already_observed"
    assert result["dominance_stop_certificate"] is None  # No feasible query is an operational limit.


def test_all_models_rejecting_a_known_return_is_misspecification_not_value_stop():
    result = identification().evaluate((("check", "failure"),), remaining_budget=1)
    assert result["status"] == "MODEL_SET_MISSPECIFIED"
    assert result["action_id"] is None and result["current_risk"] is None
    assert result["terminal_decision_if_stopping"] is None
    assert result["stop_certificate"] is None and result["dominance_stop_certificate"] is None
    assert result["active_model_ids"] == []
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("history", [
    (("unknown", "yes"),), (("check", "unknown"),),
    (("check", "failure"), ("check", "yes")),
    (("check", "failure"), ("unknown", "yes")), (("check",),), ((True, "yes"),), "check/yes",
])
def test_invalid_history_is_validated_before_any_model_exclusion(history):
    with pytest.raises(ValueError):
        identification().evaluate(history, remaining_budget=1)


def test_same_evidence_alias_is_unavailable_and_duplicate_history_is_invalid():
    original = model(evidence_id="shared-report")
    raw = original.manifest()
    raw["tools"]["mirror"] = deepcopy(raw["tools"]["check"])
    m = FiniteAmbiguityDecisionModel({"only": FiniteDecisionLossModel.from_manifest(raw)})
    result = m.evaluate((("check", "yes"),), remaining_budget=1)
    assert candidate(result, "mirror")["reason"] == "evidence_already_observed"
    with pytest.raises(ValueError, match="same evidence"):
        m.evaluate((("check", "yes"), ("mirror", "yes")), remaining_budget=1)


def test_irrelevant_query_cost_can_certify_strong_one_step_stop():
    likelihood = {"yes": (.25, .5, .25), "no": (.25, .5, .25)}
    m = FiniteAmbiguityDecisionModel({"only": model(likelihoods=likelihood)})
    result = m.evaluate((), remaining_budget=1)
    q = candidate(result)
    assert q["robust_net_value"] == pytest.approx(-.05)
    assert q["dominance_upper_gain"] == pytest.approx(-.05)
    assert set(q["response_policy"].values()) == {"defer"}
    assert result["dominance_stop_certificate"] == "NO_POSITIVE_ONE_STEP_GAIN_RELATIVE_TO_SHARED_BASELINE_IN_ANY_DECLARED_MODEL"
    assert result["bounds_scope"]["horizon"] == 1
    assert not result["global_open_world_stop_proven"]


def test_budget_stop_does_not_claim_that_an_unaffordable_query_has_no_value():
    result = identification().evaluate((), remaining_budget=.049)
    assert candidate(result)["reason"] == "budget_exhausted"
    assert result["stop_certificate"] == "NO_FEASIBLE_REGISTERED_ACTION"
    assert result["dominance_stop_certificate"] is None


@pytest.mark.parametrize("horizon", [0, 2, True, 1.0, "1"])
def test_only_explicit_one_query_horizon_is_supported(horizon):
    with pytest.raises(ValueError, match="horizon"):
        identification().evaluate((), remaining_budget=1, horizon=horizon)


@pytest.mark.parametrize("budget", [-1, True, float("nan"), float("inf"), "1"])
def test_loss_budget_rejects_invalid_values(budget):
    with pytest.raises(ValueError):
        identification().evaluate((), remaining_budget=budget)


def test_manifest_json_round_trip_and_caller_mutation_preserve_computation():
    original = model()
    m = FiniteAmbiguityDecisionModel({"z-first": original, "a-second": model(cost=.05)})
    before = m.evaluate((), remaining_budget=1)
    serialized = json.loads(json.dumps(m.manifest(), sort_keys=True, allow_nan=False))
    restored = FiniteAmbiguityDecisionModel.from_manifest(serialized)
    assert restored.manifest() == m.manifest()
    assert restored.evaluate([], remaining_budget=1) == before
    original.prior = (1, 0)
    original.cost_unit = "changed_input"
    assert m.evaluate((), remaining_budget=1) == before
    serialized["model_order"] = ["z-first", "z-first"]
    with pytest.raises(ValueError):
        FiniteAmbiguityDecisionModel.from_manifest(serialized)


@pytest.mark.parametrize("field", ["target", "loss", "decision_order", "unit", "actions", "returns", "cost", "evidence"])
def test_models_must_describe_the_same_executable_problem(field):
    first = model()
    raw = first.manifest()
    if field == "target": raw["outcome_labels"] = ["different", "no"]
    if field == "loss": raw["decision_losses"]["defer"] = [.3, .3]
    if field == "decision_order": raw["decision_order"].reverse()
    if field == "unit": raw["cost_unit"] = "different_loss_unit"
    if field == "actions": raw["tools"]["other"] = raw["tools"].pop("check")
    if field == "returns": raw["tools"]["check"]["outcomes"] = ["no", "yes", "failure"]
    if field == "cost": raw["tools"]["check"]["cost"] = .06
    if field == "evidence": raw["tools"]["check"]["evidence_id"] = "other"
    with pytest.raises(ValueError, match="contract"):
        FiniteAmbiguityDecisionModel({"first": first, "other": FiniteDecisionLossModel.from_manifest(raw)})


def test_compute_limits_reject_large_policy_spaces_before_enumeration():
    m = model()
    with pytest.raises(ValueError, match="Model count"):
        FiniteAmbiguityDecisionModel({str(i): m for i in range(LIMITS["models"]+1)})
    for decisions, returns, actions, error in [(3, 9, 1, "Return count"), (9, 6, 1, "Response policy count"),
                                               (5, 7, 4, "Total policy enumeration")]:
        big = FiniteDecisionLossModel({"t": (1, 0)}, {"t": 1}, {
            str(i): ObservationTool(tuple(map(str, range(returns))), {"t": (1,)+(0,)*(returns-1)}, .01)
            for i in range(actions)}, outcome_labels=("yes", "no"),
            decision_losses={str(i): (0, 1) for i in range(decisions)}, cost_unit="expected_report_loss")
        with pytest.raises(ValueError, match=error):
            FiniteAmbiguityDecisionModel({"big": big})


def test_log_replay_retains_tiny_hypotheses_that_later_observations_recover():
    m = FiniteDecisionLossModel({"rare": (1, 0), "usual": (0, 1)}, {"rare": 1e-300, "usual": 1}, {
        "first": ObservationTool(("seen", "other"), {"rare": (1e-100, 1), "usual": (1e-300, 1)}, 0),
        "second": ObservationTool(("seen", "other"), {"rare": (1, 0), "usual": (1e-200, 1)}, 0)},
        outcome_labels=("yes", "no"), decision_losses=LOSSES, cost_unit="expected_report_loss")
    result = FiniteAmbiguityDecisionModel({"rare-but-supported": m}).evaluate(
        (("first", "seen"), ("second", "seen")), remaining_budget=0)
    state = result["model_states"]["rare-but-supported"]
    assert result["active_model_ids"] == ["rare-but-supported"]
    assert state["posterior"]["rare"] == pytest.approx(1)
    assert state["posterior"]["usual"] > 0
    assert state["history_log_likelihood"] == pytest.approx(-400*math.log(10))
    assert result["terminal_decision_if_stopping"]["decision"] == "yes"
    json.dumps(result, allow_nan=False)


def test_one_step_stop_retains_the_complementary_information_limitation():
    hypotheses, prior, first, second = {}, {}, {}, {}
    for a, b in product((0, 1), repeat=2):
        name = f"{a}{b}"
        hypotheses[name] = (int(a == b), int(a != b))
        prior[name] = .25
        first[name], second[name] = (1-a, a), (1-b, b)
    xor = FiniteDecisionLossModel(hypotheses, prior, {
        "first": ObservationTool(("0", "1"), first, .1),
        "second": ObservationTool(("0", "1"), second, .1)}, outcome_labels=("yes", "no"),
        decision_losses={"yes": (0, 1), "no": (1, 0)}, cost_unit="expected_report_loss")
    result = FiniteAmbiguityDecisionModel({"xor": xor}).evaluate((), remaining_budget=.2)
    assert result["action_id"] == "STOP"
    assert candidate(result, "first")["dominance_upper_gain"] == pytest.approx(-.1)
    # Both bits together reveal XOR exactly for .2; myopic stopping is not a
    # statement that the .3 net value of this two-query plan is absent.
    assert result["current_risk"]-.2 == pytest.approx(.3)
    assert result["bounds_scope"]["horizon"] == 1
    assert "Conditional replanning" in result["execution_contract"]
