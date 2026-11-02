from copy import deepcopy
from datetime import timedelta

import pytest

from football_research.acquisition.state import parse_time
from football_research.evaluation.paired_acquisition import (
    STRATEGIES, evaluate_acquisition_policies, summarize_acquisition_evaluations,
)
from test_joint_returns import ACTIONS, fitted


def evaluation_rows(rows):
    result = []
    for i, source in enumerate(rows[1:4]):
        row = deepcopy(source)
        cutoff = parse_time("2025-05-01T00:00:00Z") + timedelta(days=i * 7)
        row.update(match_id=f"heldout-{i}", prediction_cutoff=cutoff.isoformat(),
                   kickoff=(cutoff + timedelta(days=1)).isoformat(),
                   result_available_at=(cutoff + timedelta(days=1, hours=3)).isoformat(),
                   features_available_at=cutoff.isoformat(), fixture_available_at=cutoff.isoformat(),
                   feature_version="synthetic-evaluation")
        result.append(row)
    return result


def evaluate(fitted, **kwargs):
    model, rows, _ = fitted
    options = dict(limits={"money": .05, "requests": 2}, lambda_money=.5, seeds=(4, 8), model_name="mechanism")
    options.update(kwargs)
    return evaluate_acquisition_policies(evaluation_rows(rows), model, ACTIONS, **options)


def test_five_policies_share_initial_budget_and_reproduce_reference_aliases(fitted, tmp_path):
    result = evaluate(fitted, output=tmp_path)
    assert {r["strategy"] for r in result["episodes"]} == set(STRATEGIES)
    assert len(result["episodes"]) == 18  # 3 matches x (4 deterministic + 2 random)
    assert all(m["unique_matches"] == 3 for m in result["summary"]["metrics"])
    assert all(e["identical"] for e in result["summary"]["policy_equivalences"])
    assert (tmp_path / "policy-trajectories.jsonl").is_file()
    for mid in {r["match_id"] for r in result["episodes"]}:
        group = [r for r in result["episodes"] if r["match_id"] == mid]
        assert len({tuple(r["initial_probabilities"]) for r in group}) == 1
        assert all(r["cost"]["requests"] <= 2 for r in group)
    assert len(result["transitions"]) == 12
    assert all(r["scope"].startswith("evaluator_only") for r in result["transitions"])


def test_single_request_budget_compares_order_without_overspending(fitted):
    result = evaluate(fitted, limits={"money": .03, "requests": 1})
    assert all(len(r["actions"]) <= 1 for r in result["episodes"])
    assert all(r["cost"]["money"] <= .03 for r in result["episodes"])
    assert len(result["transitions"]) == 6  # only initial action choices reachable
    random = next(m for m in result["summary"]["metrics"] if m["strategy"] == "uniform-random")
    assert random["episodes"] == 6 and random["unique_matches"] == 3


def test_outcomes_cannot_change_acquisitions_or_predicted_returns(fitted):
    model, rows, _ = fitted
    first = evaluation_rows(rows)
    changed = deepcopy(first)
    for row in changed:
        row.update(home_goals=9, away_goals=0)
    options = dict(limits={"money": .03, "requests": 1}, lambda_money=.5, seeds=(1,))
    a = evaluate_acquisition_policies(first, model, ACTIONS, **options)
    b = evaluate_acquisition_policies(changed, model, ACTIONS, **options)
    assert [(r["actions"], r["final_probabilities"]) for r in a["episodes"]] == [
        (r["actions"], r["final_probabilities"]) for r in b["episodes"]]
    assert [r["predicted_evsi"] for r in a["transitions"]] == [r["predicted_evsi"] for r in b["transitions"]]
    assert [r["log_loss"] for r in a["episodes"]] != [r["log_loss"] for r in b["episodes"]]


def test_paired_summary_rejects_missing_random_seed_and_changed_prior(fitted):
    records = evaluate(fitted, diagnostics=False)["episodes"]
    selected = next(i for i, r in enumerate(records) if r["strategy"] == "uniform-random")
    with pytest.raises(ValueError, match="seed"):
        summarize_acquisition_evaluations(records[:selected] + records[selected + 1:])
    changed = deepcopy(records)
    next(r for r in changed if r["strategy"] == "no-query")["initial_probabilities"] = [.6, .2, .2]
    with pytest.raises(ValueError, match="Paired policies changed"):
        summarize_acquisition_evaluations(changed)


def test_cost_conversion_used_consistently_in_utility(fitted):
    result = evaluate(fitted, lambda_money=3., diagnostics=False)
    assert all(r["utility_loss"] == pytest.approx(r["log_loss"] + 3 * r["cost"]["money"])
               for r in result["episodes"])
    assert result["summary"]["scope"]["new_holdout"] is False
    assert {r["strategy"] for r in result["summary"]["not_run"]} == {"llm-only", "oracle"}


@pytest.mark.parametrize("options", [
    {"seeds": (1, 1)}, {"limits": {"money": .03, "requests": 1, "tokens": 1}},
    {"lambda_money": float("nan")},
])
def test_unsupported_comparison_design_rejected(fitted, options):
    with pytest.raises(ValueError):
        evaluate(fitted, **options)
