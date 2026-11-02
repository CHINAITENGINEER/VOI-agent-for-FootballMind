import math
import pytest

from football_research.acquisition.finite_tree_reference import (
    FiniteTreeReference, distorted_population, entropy, frozen_population,
)


def ref(name,model="correct"):
    rows,actions,cap=frozen_population(name)
    return FiniteTreeReference(distorted_population(rows,model),actions,cap),rows


def test_context_with_zero_immediate_information_buys_selective_continuation():
    r,_=ref("gated_stop")
    assert r.solve(horizon=1)["action_id"] is None
    tree=r.solve()
    assert tree["action_id"]=="context" and tree["children"]=={"0":"signal_a","1":None}
    h=entropy([.9,.05,.05])
    assert tree["expected_total_risk"]==pytest.approx((math.log(3)+h)/2+.12)
    assert tree["expected_queries"]==pytest.approx(1.5)
    assert r.fixed_subset()["expected_total_risk"]-tree["expected_total_risk"]==pytest.approx(.1)


def test_same_first_query_routes_to_different_second_tools():
    r,_=ref("gated_route")
    tree=r.solve()
    assert tree["action_id"]=="context" and tree["children"]=={"0":"signal_a","1":"signal_b"}
    assert tree["expected_total_risk"]==pytest.approx(entropy([.9,.05,.05])+.22)
    assert r.fixed_order_stop()["expected_total_risk"]>tree["expected_total_risk"]


def test_complementarity_helps_depth_without_creating_a_routing_advantage():
    r,_=ref("complementary")
    assert r.solve(horizon=1)["action_id"] is None
    expected=entropy([.9,.05,.05])+.24
    assert r.solve()["expected_total_risk"]==pytest.approx(expected)
    assert r.fixed_subset()["expected_total_risk"]==pytest.approx(expected)
    assert r.evaluate("adaptive-two-step")["utility_loss"]==pytest.approx(expected)


def test_redundant_observation_and_irrelevant_observations_trigger_stop():
    r,_=ref("redundant")
    assert r.solve()["action_id"]=="signal_a"
    assert all(r.solve({"signal_a":str(z)})["action_id"] is None for z in range(3))
    assert r.solve()["expected_total_risk"]==pytest.approx(entropy([.9,.05,.05])+.04)
    q,_=ref("irrelevant")
    assert q.solve()["action_id"] is None
    assert q.solve()["expected_total_risk"]==pytest.approx(entropy([.7,.2,.1]))


def test_misspecified_belief_can_plan_optimally_and_have_negative_true_gain():
    r,truth=ref("gated_route","swapped_context")
    modeled=r.evaluate("adaptive-two-step")
    actual=r.evaluate("adaptive-two-step",truth)
    assert modeled["utility_loss"]==pytest.approx(entropy([.9,.05,.05])+.22)
    assert actual["utility_loss"]==pytest.approx(-sum(math.log(p) for p in [.9,.05,.05])/3+.22)
    assert actual["utility_loss"]>math.log(3)


@pytest.mark.parametrize("name",["gated_stop","gated_route","complementary","redundant","irrelevant"])
def test_exact_population_expectation_matches_enumerated_policy_value(name):
    r,truth=ref(name)
    assert r.evaluate("adaptive-two-step")["utility_loss"]==pytest.approx(r.solve()["expected_total_risk"])
    assert r.evaluate("best-fixed-order-stop")["utility_loss"]==pytest.approx(r.fixed_order_stop()["expected_total_risk"])
    assert r.evaluate("best-fixed-subset")["utility_loss"]==pytest.approx(r.fixed_subset()["expected_total_risk"])
    independent,_=ref(name,"independent")
    assert independent.solve()["action_id"] is None
    assert independent.evaluate("adaptive-two-step",truth)["queries"]==0


def test_zero_budget_and_unsupported_return():
    rows,actions,_=frozen_population("gated_stop")
    r=FiniteTreeReference(rows,actions,0.)
    assert r.solve()["action_id"] is None
    with pytest.raises(ValueError,match="Unsupported"):
        r.posterior({"context":"missing"})
    with pytest.raises(ValueError,match="repeated"):
        r.branches({"context":"0"},"context")
    with pytest.raises(ValueError,match="Normalized"):
        FiniteTreeReference(rows[:-1],actions,.22)
