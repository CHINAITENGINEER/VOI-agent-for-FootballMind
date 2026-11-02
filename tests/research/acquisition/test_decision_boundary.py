import math

import pytest

from football_research.acquisition.decision_boundary import (
    TERMS, decompose_query, margin_bucket, sum_episode, summarize_records,
)


@pytest.mark.parametrize("value,bucket",[(-.1,0),(-.05,1),(-.02,2),(-.01,3),(0.,4),(.01,5),(.02,6),(.05,7)])
def test_fixed_margin_bins_include_equal_boundary_on_right(value,bucket):
    assert margin_bucket(value)==bucket


def test_positive_execution_margin_can_have_negative_fixed_utility():
    row=decompose_query(gain=.006,mean_estimate=.014,decision_estimate=.014,
                        cost=.03,policy_lambda=0.,utility_weight=.5)
    assert row["decision_margin"]==pytest.approx(.014)
    assert row["cost_weight_difference"]==pytest.approx(-.015)
    assert row["calibration_residual"]==pytest.approx(-.008)
    assert row["actual_utility_gain"]==pytest.approx(-.009)
    assert sum(row[k] for k in TERMS)==pytest.approx(row["actual_utility_gain"])


def test_conservative_decision_has_separate_mean_to_lower_adjustment():
    row=decompose_query(gain=.02,mean_estimate=.09,decision_estimate=.03,
                        cost=.02,policy_lambda=.5,utility_weight=.5)
    assert row["mean_minus_decision"]==pytest.approx(.06)
    assert row["cost_weight_difference"]==0.
    assert row["actual_utility_gain"]==pytest.approx(.01)


def test_multiple_queries_telescope_without_counting_them_as_independent_matches():
    a=decompose_query(gain=.1,mean_estimate=.12,decision_estimate=.12,cost=.02,policy_lambda=.5,utility_weight=.5)
    b=decompose_query(gain=-.03,mean_estimate=.04,decision_estimate=.04,cost=.03,policy_lambda=.5,utility_weight=.5)
    result=sum_episode([a,b],initial_loss=1.,final_loss=.93,cost=.05,utility_weight=.5)
    assert result["actual_utility_gain"]==pytest.approx(.045)
    assert result["queries"]==2
    summary=summarize_records([{**a,"match_id":"same"},{**b,"match_id":"same"}])
    assert summary["records"]==2 and summary["distinct_matches"]==1
    with pytest.raises(ValueError,match="telescope"):
        sum_episode([a,b],initial_loss=1.,final_loss=.92,cost=.05,utility_weight=.5)
    with pytest.raises(ValueError,match="utility"):
        sum_episode([a,b],initial_loss=1.,final_loss=.93,cost=.03,utility_weight=.5)


def test_stop_has_zero_accounting_and_zero_charge():
    assert sum_episode([],initial_loss=.9,final_loss=.9,cost=0.,utility_weight=.5)["actual_utility_gain"]==0
    with pytest.raises(ValueError,match="utility"):
        sum_episode([],initial_loss=.9,final_loss=.9,cost=.03,utility_weight=.5)


@pytest.mark.parametrize("bad",[math.nan,math.inf,True,".1"])
def test_invalid_values_are_not_silently_coerced(bad):
    with pytest.raises(ValueError):
        decompose_query(gain=bad,mean_estimate=.02,decision_estimate=.02,cost=.02,policy_lambda=.5,utility_weight=.5)


def test_negative_gain_is_valid_but_negative_cost_is_not():
    row=decompose_query(gain=-.5,mean_estimate=.02,decision_estimate=.02,cost=.02,policy_lambda=.5,utility_weight=.5)
    assert row["actual_gain"]==-.5
    with pytest.raises(ValueError,match="nonnegative"):
        decompose_query(gain=0.,mean_estimate=.02,decision_estimate=.02,cost=-.02,policy_lambda=.5,utility_weight=.5)
