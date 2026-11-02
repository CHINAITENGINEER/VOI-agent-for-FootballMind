import pytest

from football_agent.g_label_contract import (
    epsilon_greedy_propensities,
    realized_signed_gain,
    sample_action,
    validate_transition,
)


def test_unexecuted_action_has_no_label_even_when_prediction_is_negative():
    assert realized_signed_gain(
        loss_before=0.4,
        loss_after=0.6,
        selected=False,
        f_after_complete=False,
        outcome_available=False,
    ) is None


def test_executed_action_keeps_negative_realized_gain():
    assert realized_signed_gain(
        loss_before=0.4,
        loss_after=0.6,
        selected=True,
        f_after_complete=True,
        outcome_available=True,
    ) == pytest.approx(-0.2)


def test_zero_gain_is_a_valid_observed_label():
    assert realized_signed_gain(
        loss_before=0.4,
        loss_after=0.4,
        selected=True,
        f_after_complete=True,
        outcome_available=True,
    ) == pytest.approx(0.0)


def test_unselected_negative_label_is_rejected():
    errors = validate_transition(
        {"selected": False, "realized_signed_gain": -0.1}
    )
    assert "unselected_action_must_have_null_realized_gain" in errors


def test_epsilon_greedy_logs_positive_propensity_for_all_candidates():
    p = epsilon_greedy_propensities(
        ["lineup", "injury", "weather"],
        {"lineup": 0.1, "injury": -0.2, "weather": 0.0},
        epsilon=0.3,
    )
    assert sum(p.values()) == pytest.approx(1.0)
    assert all(v > 0 for v in p.values())
    assert p["lineup"] > p["injury"]
    action, propensity = sample_action(p, seed=7)
    assert action in p
    assert propensity == pytest.approx(p[action])
