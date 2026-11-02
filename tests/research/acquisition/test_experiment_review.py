"""Independent review: runtime isolation, cost parity and independent sample units."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from football_research.acquisition import experiment
from football_research.acquisition.policy import select_or_stop
from football_research.acquisition.state import GROUP_FIELDS


def action(name="elo", money=0.2):
    return {"action_id": name, "group_ids": [name], "available": True,
            "cost": {"requests": 1, "money": money, "latency_ms": 0, "tokens": 0},
            "cost_status": "simulated"}


def row(day=90):
    kickoff = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return {"match_id": f"m{day}", "competition_id": "epl", "home_team_id": "h", "away_team_id": "a",
            "kickoff": kickoff.isoformat(), "prediction_cutoff": (kickoff-timedelta(days=1)).isoformat(),
            "result_available_at": (kickoff+timedelta(hours=3)).isoformat(),
            "features_available_at": (kickoff-timedelta(days=2)).isoformat(),
            "information_mode": "historical_reconstruction", "dataset_version": "synthetic-review",
            "feature_version": "review-v1", "home_goals": day % 3, "away_goals": 1,
            **{f: float(day % 7) for fields in GROUP_FIELDS.values() for f in fields}}


class Predictor:
    def __init__(self):
        self.inputs = []

    def predict_partial(self, state):
        self.inputs.append(deepcopy(state))
        mask = [g for g, v in state["groups"].items() if v["status"] == "observed"]
        # Acquisition changes prediction; hidden values are never inspected.
        p = [0.4, 0.3, 0.3] if not mask else [0.6, 0.2, 0.2]
        return {"probabilities": p, "mask": mask, "mask_key": len(mask), "model_version": "fixture"}


class Estimator:
    version = "fixture-estimator"

    def estimate(self, state, actions, prediction=None):
        return [{"action": a, "action_id": a["action_id"], "gain": 0.2, "lower_gain": 0.1,
                 "support_matches": 50, "reason": "estimated", "estimator_batch_latency_ms": 0.01,
                 "version": self.version} for a in actions]


def run(r, strategy="fixed-order", predictor=None, money=0.2):
    return experiment.run_episode(r, predictor or Predictor(), Estimator(), [action(), action("xg")],
                                  strategy=strategy, budget_money=money, lambda_money=0,
                                  seed=7, design=experiment.DEFAULT_DESIGN)


@pytest.mark.parametrize("strategy", ["fixed-order", "random", "learned-net-gain", "conservative-net-gain", "entropy", "disagreement"])
def test_hidden_group_and_label_changes_leave_pre_acquisition_decision_identical(strategy):
    original, poisoned = row(), row()
    poisoned["home_goals"], poisoned["away_goals"] = 99, 0
    for field in GROUP_FIELDS["xg"]:
        poisoned[field] = 10000.0
    a, a_trace = run(original, strategy)
    b, b_trace = run(poisoned, strategy)
    assert a_trace[0] == b_trace[0]
    assert a["actions"] == b["actions"]
    assert a["log_loss"] != b["log_loss"]
    for event in a_trace + b_trace:
        assert "home_goals" not in event["visible_state"]
        for group in event["visible_state"]["groups"].values():
            if group["status"] != "observed":
                assert group["values"] == {}


def test_decimal_remaining_budget_is_equal_for_learned_and_fixed_policy():
    remaining = 0.3 - 0.1
    fixed, _ = run(row(), "fixed-order", money=remaining)
    learned, _ = run(row(), "learned-net-gain", money=remaining)
    assert fixed["actions"] == learned["actions"] == ["elo"]


def test_initial_prediction_is_not_recomputed_or_omitted_from_measured_cost(monkeypatch):
    # Every prediction advances the deterministic wall clock by one millisecond.
    ticks = iter(i / 1000 for i in range(100))
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(ticks))
    predictor = Predictor()
    result, _ = run(row(), "base-only", predictor=predictor)
    assert len(predictor.inputs) == 1
    assert result["predictor_latency_ms"] == pytest.approx(1.0)


def test_saved_pre_action_history_is_immutable_and_contains_no_future_actions():
    result, trace = run(row(), "learned-net-gain", money=0.4)
    assert len(result["actions"]) == 2
    for step, event in enumerate(trace):
        assert [h["action_id"] for h in event["visible_state"]["action_history"]] == result["actions"][:step]


def metrics(strategy, mid, seed, loss):
    return {"strategy": strategy, "match_id": mid, "seed": seed, "budget_money": 0.05,
            "kickoff": "2025-04-01T12:00:00Z", "log_loss": loss, "brier_score": loss,
            "accuracy": 0, "estimator_latency_ms": 0.01, "predictor_latency_ms": 0.02,
            "cost": {"money": 0.02, "requests": 1}, "stop_reason": "budget", "actions": ["elo"]}


def test_random_seeds_are_averaged_within_match_before_pairing_and_counting():
    rows = []
    for strategy in experiment.STRATEGIES:
        if strategy == "random":
            rows.extend([metrics(strategy, "a", 1, 0), metrics(strategy, "a", 2, 2), metrics(strategy, "b", 1, 3)])
        else:
            rows.extend([metrics(strategy, "a", 0, 1), metrics(strategy, "b", 0, 1)])
    design = {**experiment.DEFAULT_DESIGN, "money_budgets": [0.05]}
    summary, paired = experiment.summarize(rows, design)
    # Match a mean 1, match b mean 3: equal-match mean 2, not row mean 5/3.
    result = summary["random/money=0.05"]
    assert result["log_loss"] == 2
    assert result["independent_matches"] == 2
    assert result["episodes"] == 3
    assert paired["random/money=0.05"]["independent_matches"] == 2
    assert paired["random/money=0.05"]["mean_log_loss_difference"] == 1
    assert paired["random/money=0.05"]["confidence_interval_95"] is None


def test_forward_oof_early_predictions_are_unchanged_by_later_values_and_labels(tmp_path):
    design = {**experiment.DEFAULT_DESIGN, "oof_warmup": 15, "oof_chunk_matches": 10, "min_train_matches": 10}
    original = [row(i) for i in range(45)]
    poisoned = deepcopy(original)
    for item in poisoned[35:]:
        item["home_goals"], item["away_goals"] = 99, 0
        for fields in GROUP_FIELDS.values():
            for field in fields:
                item[field] = 10000
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    a = experiment.forward_oof(original, [action()], design, left)
    b = experiment.forward_oof(poisoned, [action()], design, right)
    before = {f"m{i}" for i in range(35)}
    a_early, b_early = [s for s in a if s["match_id"] in before], [s for s in b if s["match_id"] in before]
    assert len(a_early) == 20
    for first, second in zip(a_early, b_early):
        assert first["prediction_before"] == second["prediction_before"]
        assert first["delta"] == second["delta"]
        assert first["match_id"] not in first["oof_train_ids"]
        assert first["oof_train_cutoff"] < first["state"]["prediction_cutoff"]
