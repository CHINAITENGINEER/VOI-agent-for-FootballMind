import math

import pytest

from football_research.acquisition.expected_voi import (
    COMPUTED,
    INELIGIBLE,
    ObservationBranch,
    evaluate_candidate,
    select_candidate,
)


def _uninformative_branches(prior):
    return [ObservationBranch("same", 1.0, tuple(prior))]


def test_uninformative_observation_has_zero_gross_voi():
    prior = (0.5, 0.3, 0.2)
    value = evaluate_candidate("weather", prior, _uninformative_branches(prior), cost=0.0)
    assert value.status == COMPUTED
    assert value.gross_evsi == pytest.approx(0.0)
    assert value.net_voi == pytest.approx(0.0)


def test_informative_observation_has_positive_log_loss_evsi():
    prior = (0.5, 0.3, 0.2)
    branches = [
        ObservationBranch("signal_home", 0.5, (0.8, 0.15, 0.05)),
        ObservationBranch("signal_away", 0.5, (0.2, 0.45, 0.35)),
    ]
    value = evaluate_candidate("lineup", prior, branches, cost=0.01)
    assert value.status == COMPUTED
    assert value.gross_evsi > 0.0
    assert value.net_voi < value.gross_evsi


def test_cost_delay_and_reliability_can_reverse_the_decision():
    prior = (0.5, 0.3, 0.2)
    branches = [
        ObservationBranch("home", 0.5, (0.8, 0.15, 0.05)),
        ObservationBranch("other", 0.5, (0.2, 0.45, 0.35)),
    ]
    value = evaluate_candidate(
        "lineup", prior, branches, cost=1.0, delay=10.0, delay_weight=0.1,
        reliability=0.5, reliability_penalty_weight=0.2,
    )
    assert value.net_voi < 0.0


def test_incoherent_response_model_is_ineligible_instead_of_normalized():
    prior = (0.5, 0.3, 0.2)
    branches = [ObservationBranch("bad", 1.0, (0.7, 0.2, 0.1))]
    value = evaluate_candidate("injury", prior, branches, cost=0.0)
    assert value.status == INELIGIBLE
    assert value.reason == "response_model_incoherent_with_prior"
    assert value.net_voi is None
    assert value.coherence_error == pytest.approx(0.2)


def test_branch_probabilities_and_posteriors_are_strictly_validated():
    with pytest.raises(ValueError, match="sum to one"):
        ObservationBranch("bad", 0.5, (0.5, 0.3, 0.1))
    prior = (0.5, 0.3, 0.2)
    with pytest.raises(ValueError, match="branch probabilities"):
        evaluate_candidate(
            "weather", prior,
            [ObservationBranch("a", 0.2, prior), ObservationBranch("b", 0.2, prior)],
            cost=0.0,
        )


def test_selection_prefers_positive_value_and_is_deterministic_on_ties():
    prior = (0.5, 0.3, 0.2)
    branches = [
        ObservationBranch("home", 0.5, (0.8, 0.15, 0.05)),
        ObservationBranch("other", 0.5, (0.2, 0.45, 0.35)),
    ]
    high = evaluate_candidate("lineup", prior, branches, cost=0.01)
    tie_a = evaluate_candidate("aaa", prior, branches, cost=0.0)
    tie_b = evaluate_candidate("bbb", prior, branches, cost=0.0)
    decision = select_candidate([tie_a, tie_b])
    assert decision["decision"] == "acquire"
    assert decision["candidate_id"] == "bbb"
    assert select_candidate([high])["candidate_id"] == "lineup"


def test_incomplete_candidate_values_defer_instead_of_claiming_global_stop():
    prior = (0.5, 0.3, 0.2)
    branches = [ObservationBranch("same", 1.0, prior)]
    computed = evaluate_candidate("weather", prior, branches, cost=0.0)
    unknown = evaluate_candidate(
        "lineup", prior,
        [ObservationBranch("bad", 1.0, (0.7, 0.2, 0.1))], cost=0.0,
    )
    decision = select_candidate([computed, unknown])
    assert decision["decision"] == "defer"
    assert decision["reason"] == "candidate_value_incomplete"

