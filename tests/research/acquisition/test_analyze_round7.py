import pytest

from football_research.acquisition.analyze_round7 import match_means, paired


def test_random_repeats_do_not_multiply_match_weight():
    rows = [{"strategy": "random", "budget_money": .05, "match_id": mid, "seed": seed,
             "kickoff": "2026-09-23T12:00:00Z", "log_loss": score, "brier_score": score,
             "accuracy": 0, "cost": {"money": .02, "requests": 1}}
            for mid, seed, score in [("a", 7, 1), ("a", 19, 3), ("b", 7, 10)]]
    current = match_means(rows, .5)["random", .05]
    reference = {mid: {**r, "log_loss": 0} for mid, r in current.items()}
    result = paired(current, reference, "log_loss", 200, 7)
    assert result["independent_matches"] == 2
    assert result["mean_difference"] == pytest.approx(6)
    assert current["a"]["utility_loss"] == pytest.approx(2.01)
    with pytest.raises(ValueError, match="Duplicate"):
        match_means(rows + [rows[0]], .5)


def test_paired_statistics_reject_different_games():
    with pytest.raises(ValueError, match="identical"):
        paired({"a": {}}, {"b": {}}, "log_loss", 100, 7)
