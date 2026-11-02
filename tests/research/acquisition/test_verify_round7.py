"""Evidence-mutation checks for recomputed OOF gains and fair policy coverage."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from football_research.acquisition.experiment import losses, outcome, state_after
from football_research.acquisition.experiment_round4 import DEFAULT_DESIGN, run_episode
from football_research.acquisition.state import GROUP_FIELDS, sanitize_state
from football_research.acquisition.verify_round7 import (
    verify_episode_coverage, verify_heuristic_trace, verify_oof_sample,
)


class Predictor:
    def predict_partial(self, state):
        state = sanitize_state(state)
        return {"probabilities": [.6, .2, .2] if state["mask"] else [.4, .3, .3],
                "mask": state["mask"], "mask_key": state["mask_key"], "model_version": "audit-test"}


def fixture():
    source = {"match_id": "audit7", "competition_id": "cup", "home_team_id": "a", "away_team_id": "b",
              "kickoff": "2026-09-23T12:00:00+00:00", "prediction_cutoff": "2026-09-22T12:00:00+00:00",
              "features_available_at": "2026-09-22T11:00:00+00:00", "information_mode": "historical_reconstruction",
              "result_available_at": "2026-09-23T14:00:00+00:00", "feature_version": "audit-test", "home_goals": 3, "away_goals": 0}
    source.update({f: 1.0 for fields in GROUP_FIELDS.values() for f in fields})
    actions = [{"action_id": "results", "group_ids": ["elo"], "cost": {"money": .02, "requests": 1}},
               {"action_id": "xg", "group_ids": ["xg"], "cost": {"money": .03, "requests": 1}}]
    return source, actions, Predictor()


def supervision(acquired=()):
    source, actions, model = fixture()
    state = state_after(source, actions, acquired)
    action = next(a for a in actions if a["action_id"] not in acquired)
    before = model.predict_partial(state)["probabilities"]
    after = model.predict_partial(state_after(source, actions, (*acquired, action["action_id"])))["probabilities"]
    cutoff = "2026-09-20T00:00:00+00:00"
    sample = {"match_id": source["match_id"], "kickoff": source["kickoff"],
              "result_available_at": source["result_available_at"], "state": state, "action": deepcopy(action),
              "prediction_before": before, "delta": losses(before, outcome(source))["log_loss"] - losses(after, outcome(source))["log_loss"],
              "oof_train_ids": ["prior-game"], "oof_train_cutoff": cutoff}
    return sample, source, model, actions, ["prior-game"], cutoff


@pytest.mark.parametrize("acquired", [(), ("xg",)])
def test_oof_gain_rebuilt_from_source_and_new_predictor(acquired):
    assert verify_oof_sample(*supervision(acquired))[1] == acquired


@pytest.mark.parametrize("corruption", ["gain", "probability", "outcome", "training_ids", "cutoff",
                                       "hidden_label", "free_reveal", "action_cost", "result_time"])
def test_oof_mutations_rejected(corruption):
    sample, source, model, actions, train, cutoff = supervision()
    if corruption == "gain":
        sample["delta"] += .01
    elif corruption == "probability":
        sample["prediction_before"] = [.5, .25, .25]
    elif corruption == "outcome":
        source["home_goals"], source["away_goals"] = 0, 3
    elif corruption == "training_ids":
        sample["oof_train_ids"].append(source["match_id"])
    elif corruption == "cutoff":
        sample["oof_train_cutoff"] = "2026-09-23T00:00:00+00:00"
    elif corruption == "hidden_label":
        sample["state"]["label"] = 0
    elif corruption == "free_reveal":
        sample["state"]["groups"]["elo"] = state_after(source, actions, ("results",))["groups"]["elo"]
    elif corruption == "action_cost":
        sample["action"]["cost"]["money"] = 0
    else:
        sample["result_available_at"] = "2026-09-23T15:00:00+00:00"
    with pytest.raises(ValueError):
        verify_oof_sample(sample, source, model, actions, train, cutoff)


@pytest.mark.parametrize("strategy", ["base-only", "all-available", "fixed-order", "entropy", "random"])
def test_heuristic_decisions_replayed_and_early_stop_rejected(strategy):
    source, actions, model = fixture()
    estimator = SimpleNamespace(version="audit-test")
    row, events = run_episode(source, model, estimator, estimator, actions, strategy=strategy,
                             budget_money=.05, lambda_money=0, seed=19, design=DEFAULT_DESIGN)
    verify_heuristic_trace(row, {"events": events}, source, actions, DEFAULT_DESIGN)
    events[0]["decision"]["reason"] = "invented_reason"
    with pytest.raises(ValueError, match="reason"):
        verify_heuristic_trace(row, {"events": events}, source, actions, DEFAULT_DESIGN)


def test_seed_coverage_requires_every_game_every_strategy_every_budget():
    rows = [{"fold": 0, "match_id": mid, "strategy": strategy, "budget_money": .02, "seed": seed}
            for mid in ("a", "b") for strategy in ("base-only", "random")
            for seed in ([7, 19] if strategy == "random" else [0])]
    args = ({0: {"a", "b"}}, {"base-only", "random"}, [.02], [7, 19])
    verify_episode_coverage(rows, *args)
    with pytest.raises(ValueError, match="coverage"):
        verify_episode_coverage(rows[:-1], *args)
    with pytest.raises(ValueError, match="Duplicate"):
        verify_episode_coverage(rows + [rows[-1]], *args)
