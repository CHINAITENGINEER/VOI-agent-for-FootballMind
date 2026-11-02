import pytest

from football_research.acquisition.verify_round10 import audit_accounting_record


def example():
    return {"mean_estimate":.014,"decision_estimate":.014,"actual_gain":.006,
            "cost":.03,"policy_lambda":0.,"decision_margin":.014,
            "cost_weight_difference":-.015,"mean_minus_decision":0.,"calibration_residual":-.008,
            "actual_utility_gain":-.009,"reported_estimated_net_gain":-.001,"margin_bucket":5}


def test_independent_query_accounting_accepts_hand_calculated_result():
    audit_accounting_record(example(),.5)


@pytest.mark.parametrize("field",["policy_lambda","calibration_residual","cost_weight_difference","actual_utility_gain","margin_bucket"])
def test_accounting_mutations_are_detected(field):
    row=example();row[field]+=.25
    with pytest.raises(ValueError):
        audit_accounting_record(row,.5)
