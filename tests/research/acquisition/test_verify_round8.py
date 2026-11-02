"""Independent table reference: complementarity, budgets and coherence."""
import numpy as np
import pytest

from football_research.acquisition.verify_round8 import probability_checks, reference_plan


def actions():
    return [{"action_id": "results_view", "cost": {"money": .02, "requests": 1}},
            {"action_id": "xg_view", "cost": {"money": .03, "requests": 1}}]


def xor():
    table = np.zeros((3, 2, 2))
    for a in range(2):
        for b in range(2):
            table[a ^ b, a, b] = .25
    return table


def test_exact_complementarity_and_optional_stopping():
    table = xor()
    checks = probability_checks(table, [a["action_id"] for a in actions()])
    assert all(abs(d["evsi"]) < 1e-12 for d in checks.values())
    one = reference_plan(table, actions(), money=.05, requests=2, lambda_money=1., horizon=1)
    two = reference_plan(table, actions(), money=.05, requests=2, lambda_money=1., horizon=2)
    assert one["decision"] == "stop"
    assert two["decision"] == "action"
    assert two["expected_terminal_risk"] == pytest.approx(0)
    assert two["expected_plan_cost_vector"]["money"] == pytest.approx(.05)
    assert two["planning_net_gain"] == pytest.approx(np.log(2)-.05)


@pytest.mark.parametrize("money,requests", [(.02, 2), (.05, 1), (0., 2)])
def test_zero_cost_weight_does_not_erase_physical_budget(money, requests):
    plan = reference_plan(xor(), actions(), money=money, requests=requests, lambda_money=0., horizon=2)
    assert plan["decision"] == "stop"


def test_duplicate_correlated_information_has_no_conditional_gain():
    table = np.zeros((3, 2, 2))
    table[0, 0, 0], table[1, 1, 1] = .5, .5
    checks = probability_checks(table, [a["action_id"] for a in actions()])
    assert checks["results_view"]["evsi"] == pytest.approx(np.log(2))
    plan = reference_plan(table, actions(), money=.05, requests=2, lambda_money=1., horizon=2)
    assert plan["action_id"] == "results_view"
    assert plan["expected_plan_cost_vector"]["requests"] == pytest.approx(1)
    assert all(b["decision"] == "stop" for b in plan["branches"])


def test_irrelevant_high_entropy_response_has_zero_target_value():
    table = np.ones((3, 2, 2)) / 12
    for diagnostic in probability_checks(table, [a["action_id"] for a in actions()]).values():
        assert diagnostic["evsi"] == pytest.approx(0)
        assert diagnostic["posterior_expectation_max_error"] < 1e-12


@pytest.mark.parametrize("table", [np.zeros((3, 2, 2)), np.full((3, 2, 2), -1.),
                                  np.full((3, 2, 2), float("nan")), np.ones((2, 2))])
def test_invalid_joint_law_is_rejected(table):
    with pytest.raises(ValueError):
        probability_checks(table, ["results_view"])
