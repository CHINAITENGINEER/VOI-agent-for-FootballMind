from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from football_research.acquisition.contextual_value import ContextualValueEstimator, MODES
from football_research.acquisition.state import GROUP_FIELDS


def state(day=100, match_id="future", home="h0"):
    kickoff = datetime(2025, 1, 1, tzinfo=timezone.utc)+timedelta(days=day)
    return {"match_id": match_id, "prediction_cutoff": (kickoff-timedelta(days=1)).isoformat(),
        "information_mode": "historical_reconstruction",
        "base": {"competition_id": "epl", "home_team_id": home, "away_team_id": "a", "kickoff": kickoff.isoformat()},
        "groups": {group: {"status": "not_acquired", "values": {}} for group in GROUP_FIELDS}}


ACTION = {"action_id": "fetch-xg", "group_ids": ["xg"], "available": True,
          "cost": {"requests": 1., "money": .03, "latency_ms": 0., "tokens": 0.}, "cost_status": "simulated"}


def samples(start=1, count=30):
    rows = []
    for i in range(start, start+count):
        s = state(i, f"m{i}", "h"+str(i%2))
        rows.append({"state": s, "action": ACTION, "delta": .5 if i%2 else -.2,
            "match_id": s["match_id"], "kickoff": s["base"]["kickoff"],
            "prediction_before": [.5, .3, .2],
            "result_available_at": (datetime.fromisoformat(s["base"]["kickoff"])+timedelta(hours=3)).isoformat(),
            "oof_train_cutoff": "2024-12-01T00:00:00Z", "oof_train_ids": ["prior"]})
    return rows


def fit(mode="visible-context", train=None, tail=None):
    return ContextualValueEstimator(mode=mode, min_support=5).fit(train or samples(),
        calibration_samples=tail if tail is not None else samples(40, 10),
        as_of="2025-03-01T00:00:00Z", dataset_version="synthetic-mechanism-only")


def estimate(model, s=None):
    return model.estimate_value(s or state(), [ACTION], prediction=[.5, .3, .2])[0]


def test_ablation_has_identifiable_identity_effect_only_when_enabled():
    a, b = state(home="h0"), state(home="h1")
    for mode in MODES:
        model = fit(mode)
        difference = estimate(model, b)["gain"]-estimate(model, a)["gain"]
        if mode == "visible-context":
            assert difference > .1
        else:
            assert difference == 0


@pytest.mark.parametrize("mode", MODES)
def test_unpurchased_values_unknown_context_and_labels_are_inert(mode):
    model = fit(mode)
    clean, contaminated = state(), state()
    contaminated.update(result="away", label=2, motivation="already_qualified", full_features={"home_elo": 999999})
    contaminated["base"].update(home_elo=99999, qualification_status="clinched")
    for group in contaminated["groups"].values():
        group["values"] = {"home_elo": object(), "away_rolling_xg_for": float("nan")}
    assert estimate(model, clean) == estimate(model, contaminated)


def test_unknown_identity_uses_train_only_vocabulary_and_reports_fallback():
    tail = samples(40, 10)
    for row in tail:
        row["state"]["base"]["home_team_id"] = "tail-only"
    model = fit(tail=tail)
    assert "tail-only" not in next(iter(model.cells.values()))["categorical_fit"]["home_team_id"]
    result = estimate(model, state(home="tail-only"))
    assert result["unknown_identity_fields"] == ["home_team_id"]
    assert result["fallback_reason"] == "unknown_identity_zero_effect_fallback"
    assert result["support_scope"].endswith("not exact fixture or identity interaction")


def test_observed_values_change_estimate_but_missing_group_is_never_imputed():
    train = samples()
    for i, row in enumerate(train):
        row["state"]["groups"]["elo"] = {"status": "observed", "available_at": "2024-12-01T00:00:00Z",
            "values": {"home_elo": 1000+10*i, "away_elo": 1200}, "evidence_refs": []}
        row["delta"] = i/20
    model = fit(train=train, tail=[])
    a, b = state(), state()
    for s, elo in ((a, 1050), (b, 1250)):
        s["groups"]["elo"] = {"status": "observed", "available_at": "2024-12-01T00:00:00Z",
                               "values": {"home_elo": elo, "away_elo": 1200}, "evidence_refs": []}
    assert estimate(model, b)["gain"] > estimate(model, a)["gain"]
    assert estimate(model, state())["support"] == 0
    a["groups"]["elo"]["available_at"] = "2026-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="unavailable"):
        estimate(model, a)


def test_temporal_overlap_and_self_training_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        fit(tail=samples())
    train = samples()
    train[0]["oof_train_ids"].append(train[0]["match_id"])
    with pytest.raises(ValueError, match="predictor training"):
        fit(train=train)
    with pytest.raises(ValueError, match="precedes"):
        estimate(fit(), state(day=10))


def test_duplicate_match_cell_support_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        fit(train=samples()*2)


def test_unqueried_and_source_missing_are_distinct_public_states():
    train = samples()
    missing = deepcopy(train)
    for row in missing:
        row["state"]["groups"]["elo"]["status"] = "source_missing"
    model = fit(train=train+missing)
    assert len(model.cells) == 2
    assert {cell["support"] for cell in model.cells.values()} == {30}


def test_prediction_only_ignores_acquired_numbers_with_same_probabilities():
    train = samples()
    for row in train:
        row["state"]["groups"]["elo"] = {"status": "observed", "available_at": "2024-12-01T00:00:00Z",
            "values": {"home_elo": 1200, "away_elo": 1300}, "evidence_refs": []}
    model = fit(mode="prediction-only", train=train, tail=[])
    a = deepcopy(train[0]["state"])
    a.update(match_id="future", prediction_cutoff=state()["prediction_cutoff"])
    a["base"]["kickoff"] = state()["base"]["kickoff"]
    b = deepcopy(a)
    b["groups"]["elo"]["values"]["home_elo"] = 9999
    assert estimate(model, a) == estimate(model, b)


def test_mean_uncertainty_does_not_use_future_tail_outcome_residuals():
    tail = samples(40, 10)
    large = deepcopy(tail)
    for row in large:
        row["delta"] *= 100
    a, b = fit(tail=tail), fit(tail=large)
    ea, eb = estimate(a), estimate(b)
    assert ea["gain"] == eb["gain"]
    assert ea["standard_error"] == eb["standard_error"]
    assert eb["predictive_upper_gain"] > ea["predictive_upper_gain"]


@pytest.mark.parametrize("mode", MODES)
def test_hash_roundtrip_and_tamper_detection(tmp_path, mode):
    model = fit(mode)
    path = tmp_path/"value.json"
    model.save(path)
    restored = ContextualValueEstimator.load(path)
    assert estimate(model) == estimate(restored)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["parameters"]["regularization"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        ContextualValueEstimator.load(path)
