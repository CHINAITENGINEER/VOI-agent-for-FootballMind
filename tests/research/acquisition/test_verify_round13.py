"""Mechanism checks for the independent content integration reference."""
from copy import deepcopy
import math

import numpy as np
import pytest

from football_research.acquisition.verify_round9 import reference_summary
from football_research.acquisition.verify_round13 import reference_marginal, select


def fixture():
    # The first return changes dependence between outcome and the next tool,
    # while the outcome marginal [1/2,1/4,1/4] is identical in both branches.
    cube = np.zeros((3, 2, 2))
    cube[:, 0, :] = .75*np.asarray([.5, .25, .25])[:, None]*np.asarray([.5, .5])
    cube[:, 1, :] = .25*np.asarray([[.5, 0], [0, .25], [0, .25]])
    records = []
    for i in range(24):
        low = i < 12
        records.append({"match_id": str(i), "kickoff": "2025-01-05T12:00:00Z" if low else "2025-01-12T12:00:00Z",
                        "model_evsi": .0 if low else .3, "delta": .2 if low else .0})
    return cube, {("xg_view", ("results_view",)): records}, {"results_view": ["common", "missing"], "xg_view": ["a", "b"]}


def test_integrate_calibrated_gain_after_each_branch_not_calibrate_average_evsi():
    cube, parents, categories = fixture()
    expected = reference_marginal(cube, "xg_view", ["results_view"], parents, categories)
    assert [r["probability"] for r in expected["branches"]] == pytest.approx([.75, .25])
    assert [r["model_evsi"] for r in expected["branches"]] == pytest.approx([0., math.log(2)])
    # Shrinkage .375; low-bin correction .04375; high-bin correction -.14375.
    assert expected["expected_gain"] == pytest.approx(.75*.04375+.25*(math.log(2)-.14375))
    naive = reference_summary(parents["xg_view", ("results_view",)], expected["marginal_model_evsi"])
    assert abs(expected["expected_gain"]-naive["expected_gain"]) > .1


def test_missing_branch_has_probability_mass_and_equal_posterior_does_not_erase_dependence():
    cube, parents, categories = fixture()
    result = reference_marginal(cube, "xg_view", ["results_view"], parents, categories)
    first, second = result["branches"]
    assert second["observation"] == "missing" and second["probability"] == .25
    assert first["posterior_probabilities"] == second["posterior_probabilities"]
    assert first["model_evsi"] != second["model_evsi"]
    # Averaging never counts the 2 branches as 48 calibration matches.
    assert result["parent_support"] == 24 and result["support"] == 12


def test_unsupported_parent_is_not_rescaled_into_a_supported_gain():
    cube, parents, categories = fixture()
    parents["xg_view", ("results_view",)] = parents["xg_view", ("results_view",)][:11]
    result = reference_marginal(cube, "xg_view", ["results_view"], parents, categories)
    assert result["expected_gain"] is None and not result["supported"]
    assert len(result["branches"]) == 2 and sum(b["probability"] for b in result["branches"]) == 1


def test_unreadable_outcome_fields_do_not_enter_content_valuation():
    cube, parents, categories = fixture()
    before = reference_marginal(cube, "xg_view", ["results_view"], parents, categories)
    for row in parents["xg_view", ("results_view",)]:
        row["current_match_outcome"] = object()
        row["unacquired_payload"] = object()
    assert reference_marginal(cube, "xg_view", ["results_view"], parents, categories) == before


@pytest.mark.parametrize("acquired", [["xg_view"], ["results_view", "xg_view"], ["unknown"]])
def test_invalid_or_already_acquired_context_rejected(acquired):
    cube, parents, categories = fixture()
    with pytest.raises(ValueError, match="context"):
        reference_marginal(cube, "xg_view", acquired, parents, categories)


def test_content_mean_control_keeps_first_decision_and_small_budget_paths():
    from football_research.acquisition.analyze_round13 import observed_policy_pairs
    original = {"fold": 2, "match_id": "m", "budget_money": .03, "kickoff": "2025-01-05T12:00:00Z",
                "strategy": "frozen-version-mean", "actions": ["xg_view"], "cost": {"money": .03},
                "final_probabilities": [.4, .3, .3], "log_loss": 1.0}
    updated = {**deepcopy(original), "strategy": "content-marginal-mean"}
    assert observed_policy_pairs([updated], [original])[0]["path_differs"] is False
    updated["actions"] = ["results_view"]
    with pytest.raises(ValueError, match="initial"):
        observed_policy_pairs([updated], [original])
    updated["actions"] = ["xg_view", "results_view"]
    with pytest.raises(ValueError, match="continuation"):
        observed_policy_pairs([updated], [original])


def test_conditional_query_stop_can_change_with_content_even_at_equal_outcome_posterior():
    cube, parents, categories = fixture()
    result = reference_marginal(cube, "xg_view", ["results_view"], parents, categories)
    action = {"action_id": "xg_view", "cost": {"money": .20, "requests": 1}}
    choices = [select([{**b, "action": action}], money=.20, requests=1) for b in result["branches"]]
    assert choices == [None, "xg_view"]
    assert select([{**result, "action": action}], money=.20, requests=1) == "xg_view"
    assert select([{**result, "action": action}], money=.19, requests=1) is None
