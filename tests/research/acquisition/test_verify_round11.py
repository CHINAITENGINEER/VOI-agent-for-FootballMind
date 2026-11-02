"""Audit regressions for temporal selection and equal-match Agent behavior."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from football_research.acquisition.analyze_round11 import behavior_summary
from football_research.acquisition.verify_round11 import LEGAL, matched_ids


def fixture():
    sources = {}
    for mid, day in (("past",1),("a",4),("b",5),("v",8),("t",10)):
        sources[mid] = {"match_id":mid,"prediction_cutoff":f"2025-01-{day:02d}T00:00:00Z",
            "result_available_at":f"2025-01-{day+1:02d}T03:00:00Z"}
    target = {"training_match_ids":["past","a","b"],"validation_match_ids":["v"],"test_match_ids":["t"],
        "split":{"train_cutoff":"2025-01-08T00:00:00Z"}}
    model = SimpleNamespace(metadata={"training_as_of":"2025-01-03T00:00:00Z"},
        base_predictor=SimpleNamespace(metadata={"training_match_ids":["past"]}))
    mixed = [{"match_id":mid,"action":{"action_id":aid},"acquired_action_ids":list(acquired),"delta":-.1}
             for mid in sources for acquired,aid in LEGAL]
    return sources,target,model,mixed


def test_selection_is_invariant_to_favorable_or_unfavorable_outcomes():
    sources,target,model,mixed = fixture()
    expected,boundary = matched_ids(sources,target,model,mixed)
    assert expected == {"a","b"}
    altered = deepcopy(mixed)
    for i,row in enumerate(altered):
        row["delta"] = -1000 if i%2 else 1000
    for row in sources.values():
        row["home_goals"] = 99
    assert matched_ids(sources,target,model,altered) == (expected,boundary)


@pytest.mark.parametrize("condition",["reference_equality","label_equality","own_training","original_purge","missing_transition"])
def test_shared_arm_selection_excludes_temporal_or_membership_leakage(condition):
    sources,target,model,mixed = fixture()
    if condition == "reference_equality":
        sources["a"]["prediction_cutoff"] = model.metadata["training_as_of"]
    elif condition == "label_equality":
        sources["a"]["result_available_at"] = sources["v"]["prediction_cutoff"]
    elif condition == "own_training":
        model.base_predictor.metadata["training_match_ids"].append("a")
    elif condition == "original_purge":
        target["training_match_ids"].remove("a")
    else:
        mixed.remove(next(r for r in mixed if r["match_id"] == "a"))
    assert matched_ids(sources,target,model,mixed)[0] == {"b"}


def test_duplicate_supervision_does_not_increase_support():
    sources,target,model,mixed = fixture()
    with pytest.raises(ValueError,match="Duplicate"):
        matched_ids(sources,target,model,mixed+[deepcopy(mixed[0])])


def test_behavior_random_repeats_are_averaged_inside_match():
    row = lambda mid,seed,actions:{"match_id":mid,"seed":seed,"actions":actions,
        "initial_probabilities":[.4,.3,.3],"final_probabilities":[.5,.3,.2] if actions else [.4,.3,.3]}
    result = behavior_summary([row("a",1,["xg_view"]),row("a",2,[]),row("b",1,[])])
    assert result["matches"] == 2 and result["raw_replays"] == 3
    assert result["first_choice_match_equivalents"] == {"xg_view":.5,"STOP":1.5}
    assert result["mean_queries"] == .25
    assert result["initial_stop_rate"] == .75
    assert result["mean_prediction_l1_change"] == pytest.approx(.05)


def test_behavior_repeated_seed_is_rejected():
    row = {"match_id":"a","seed":0,"actions":[],"initial_probabilities":[.4,.3,.3],"final_probabilities":[.4,.3,.3]}
    with pytest.raises(ValueError,match="seed"):
        behavior_summary([row,deepcopy(row)])
