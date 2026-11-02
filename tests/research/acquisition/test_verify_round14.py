"""Independent decision examples for the value of contingent querying."""
import pytest

from football_research.acquisition.verify_round14 import continuation_terms, select

ACTION = {"action_id": "xg_view", "cost": {"money": .03, "requests": 1, "tokens": 0, "latency_ms": 0}}


def rows(gains=(.10, -.08), weights=(.5, .5)):
    return [{"observation": label, "probability": probability, "expected_gain": gain}
            for label, probability, gain in zip(("observed", "missing"), weights, gains)]


def test_adaptive_option_has_value_when_committing_to_second_query_has_negative_mean():
    adaptive = continuation_terms(rows(), ACTION, mode="adaptive")
    commit = continuation_terms(rows(), ACTION, mode="commit")
    assert adaptive["continuation_bonus"] == pytest.approx(.0425)
    assert commit["continuation_bonus"] == 0
    assert adaptive["surrogate_option_gap"] == pytest.approx(.0425)
    assert [r["continuation_action"] for r in adaptive["branches"]] == ["xg_view", None]
    assert [r["continuation_action"] for r in commit["branches"]] == [None, None]
    assert adaptive["expected_continuation_cost"]["money"] == pytest.approx(.015)
    assert adaptive["expected_continuation_cost"]["requests"] == .5
    # A first query with immediate net -0.01 buys a positive conditional plan.
    plan = {"action_id": "results_view", "feasible": True, "selection_score": round(-.01+adaptive["continuation_bonus"], 12)}
    assert select([plan]) == "results_view"


def test_positive_mean_commit_does_not_select_only_favorable_actual_branches():
    result = continuation_terms(rows((.10, -.02)), ACTION, mode="commit")
    assert result["continuation_bonus"] == pytest.approx(.025)
    assert result["expected_continuation_cost"]["requests"] == 1
    assert [r["continuation_action"] for r in result["branches"]] == ["xg_view", "xg_view"]
    assert result["adaptive_bonus"] >= result["commit_bonus"]


@pytest.mark.parametrize("mode", ["adaptive", "commit"])
def test_budget_or_unavailable_continuation_disables_bonus_without_free_query(mode):
    result = continuation_terms(rows(), ACTION, mode=mode, eligible=False)
    assert result["continuation_bonus"] == 0
    assert result["expected_continuation_gain"] == 0
    assert result["expected_continuation_cost"]["requests"] == 0
    assert not result["continuation_enabled"]


@pytest.mark.parametrize("mode", ["adaptive", "commit"])
def test_any_unsupported_positive_mass_branch_disables_entire_continuation(mode):
    result = continuation_terms(rows((.10, None)), ACTION, mode=mode)
    assert len(result["branches"]) == 2
    assert result["branches"][1]["observation"] == "missing"
    assert result["continuation_bonus"] == 0 and result["adaptive_bonus"] == result["commit_bonus"] == 0
    assert result["marginal_continuation_gain"] is None
    assert not result["continuation_supported"]


def test_branch_probabilities_are_not_renormalized_after_category_drop():
    with pytest.raises(ValueError, match="mass"):
        continuation_terms(rows(weights=(.5, .3)), ACTION, mode="adaptive")


def test_no_future_outcome_or_actual_category_is_used_to_value_future_choice():
    sample = rows()
    expected = continuation_terms(sample, ACTION, mode="adaptive")
    for row in sample:
        row["current_match_outcome"] = object()
    actual = continuation_terms(sample, ACTION, mode="adaptive")
    assert actual["continuation_bonus"] == expected["continuation_bonus"]
    assert actual["expected_continuation_cost"] == expected["expected_continuation_cost"]


def test_no_second_query_if_net_rounds_to_zero_and_ties_sort_by_action_id():
    result = continuation_terms(rows((.015+1e-14, .015)), ACTION, mode="adaptive")
    assert result["continuation_bonus"] == 0
    assert select([{"action_id": a, "feasible": True, "selection_score": .03}
                   for a in ("xg_view", "results_view")]) == "results_view"
    assert select([{"action_id": "results_view", "feasible": True, "selection_score": 0.}]) is None
