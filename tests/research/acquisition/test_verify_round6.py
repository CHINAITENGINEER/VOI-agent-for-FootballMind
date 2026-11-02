"""Mutation tests for the independent round-six evidence audit."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from football_research.acquisition.experiment_round4 import DEFAULT_DESIGN, run_episode
from football_research.acquisition.state import GROUP_FIELDS, sanitize_state
from football_research.acquisition.verify_round6 import verify_episode_trace


class Predictor:
    def predict_partial(self, state):
        state = sanitize_state(state)
        return {"probabilities": [.6, .2, .2] if state["mask"] else [.4, .3, .3],
                "mask": state["mask"], "mask_key": state["mask_key"], "model_version": "test"}


def recorded():
    source = {"match_id": "audit", "competition_id": "cup", "home_team_id": "a", "away_team_id": "b",
              "kickoff": "2026-09-23T12:00:00+00:00", "prediction_cutoff": "2026-09-22T12:00:00+00:00",
              "features_available_at": "2026-09-22T11:00:00+00:00", "information_mode": "historical_reconstruction",
              "feature_version": "test", "home_goals": 3, "away_goals": 0}
    source.update({f: 1.0 for fields in GROUP_FIELDS.values() for f in fields})
    actions = [{"action_id": "results", "group_ids": ["elo"], "cost": {"money": .02, "requests": 1}}]
    predictor = Predictor()
    estimator = SimpleNamespace(version="test")
    row, events = run_episode(source, predictor, estimator, estimator, actions,
                             strategy="all-available", budget_money=.05, lambda_money=0,
                             seed=0, design=DEFAULT_DESIGN)
    row["fold"] = 0
    trace = {k: row[k] for k in ("fold", "match_id", "strategy", "budget_money", "seed")}
    trace["events"] = events
    return row, trace, source, predictor, actions


def test_valid_replay_and_source_labels():
    args = recorded()
    assert verify_episode_trace(*args, request_budget=2) == {"events": 2, "actions": 1}


@pytest.mark.parametrize("target", ["score", "free_reveal", "hidden_label", "charge", "final_probability", "duplicate_action", "source_outcome"])
def test_rejects_corrupted_evidence(target):
    row, trace, source, predictor, actions = recorded()
    if target == "score":
        row["log_loss"] = 0
    elif target == "free_reveal":
        trace["events"][0]["visible_state"]["groups"]["elo"] = deepcopy(trace["events"][1]["visible_state"]["groups"]["elo"])
    elif target == "hidden_label":
        trace["events"][0]["visible_state"]["label"] = 0
    elif target == "charge":
        trace["events"][0]["charged_cost"]["money"] = 0
    elif target == "final_probability":
        row["final_probabilities"] = [.9, .05, .05]
    elif target == "duplicate_action":
        row["actions"].append("results")
    elif target == "source_outcome":
        source["home_goals"], source["away_goals"] = 0, 3
    with pytest.raises(ValueError):
        verify_episode_trace(row, trace, source, predictor, actions, request_budget=2)
