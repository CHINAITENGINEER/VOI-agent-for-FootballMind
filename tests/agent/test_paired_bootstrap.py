"""Round144 tests for episode-level paired bootstrap summaries."""
from __future__ import annotations

import pytest

from football_agent.paired_bootstrap import paired_bootstrap


def _report():
    return {
        "paired_deltas": {
            "interval_voi": [
                {"episode_id": "e1", "counterfactual_eligible": True, "final_loss_delta": -.2,
                 "net_utility_delta": .1},
                {"episode_id": "e2", "counterfactual_eligible": True, "final_loss_delta": .1,
                 "net_utility_delta": -.1},
                {"episode_id": "e3", "counterfactual_eligible": False, "final_loss_delta": None,
                 "net_utility_delta": None},
            ],
        }
    }


def test_bootstrap_is_reproducible_and_keeps_unknown_denominator():
    first = paired_bootstrap(_report(), policy="interval_voi", resamples=100, seed=7)
    second = paired_bootstrap(_report(), policy="interval_voi", resamples=100, seed=7)
    assert first == second
    assert first["estimate"] == pytest.approx(-.05)
    assert first["observed_denominator"] == 2
    assert first["unknown_denominator"] == 1
    assert first["confidence_interval"][0] <= first["estimate"] <= first["confidence_interval"][1]


def test_block_bootstrap_resamples_whole_public_strata():
    report = _report()
    report["paired_deltas"]["interval_voi"].append(
        {"episode_id": "e4", "counterfactual_eligible": True, "final_loss_delta": .3,
         "net_utility_delta": .2}
    )
    result = paired_bootstrap(report, policy="interval_voi", metric="net_utility_delta",
                              block_by={"e1": "league-a", "e2": "league-a", "e3": "league-b", "e4": "league-b"},
                              resamples=100, seed=11)
    assert result["block_count"] == 2
    assert result["observed_denominator"] == 3
    assert result["unit"] == "episode_block_mean"


def test_no_observed_pairs_returns_unknown_summary():
    report = {"paired_deltas": {"interval_voi": [
        {"episode_id": "e1", "counterfactual_eligible": False, "final_loss_delta": None}
    ]}}
    result = paired_bootstrap(report, policy="interval_voi", resamples=100)
    assert result["estimate"] is None
    assert result["confidence_interval"] is None
    assert result["resamples"] == 0
    assert result["unknown_denominator"] == 1


def test_missing_block_and_invalid_delta_are_rejected():
    with pytest.raises(ValueError, match="block_by missing"):
        paired_bootstrap(_report(), policy="interval_voi", block_by={"e1": "a"}, resamples=100)
    bad = _report()
    bad["paired_deltas"]["interval_voi"][0]["final_loss_delta"] = "bad"
    with pytest.raises(ValueError, match="must be finite"):
        paired_bootstrap(bad, policy="interval_voi", resamples=100)


def test_unknown_policy_metric_and_small_resample_count_are_rejected():
    with pytest.raises(ValueError, match="metric must be"):
        paired_bootstrap(_report(), policy="interval_voi", metric="query_count", resamples=100)
    with pytest.raises(ValueError, match="resamples"):
        paired_bootstrap(_report(), policy="interval_voi", resamples=99)


def test_duplicate_episode_rows_are_rejected():
    report = _report()
    report["paired_deltas"]["interval_voi"].append(report["paired_deltas"]["interval_voi"][0].copy())
    with pytest.raises(ValueError, match="unique"):
        paired_bootstrap(report, policy="interval_voi", resamples=100)
