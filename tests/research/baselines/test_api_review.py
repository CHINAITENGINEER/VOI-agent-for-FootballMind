"""Synthetic mechanism regressions from API review; not real evaluation data."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from football_research.baselines.api import PredictionService


CUTOFF = datetime(2025, 6, 1, tzinfo=timezone.utc)


def fixture():
    return {"match_id": "target", "competition_id": "synthetic-league", "home_team_id": "a", "away_team_id": "b",
            "kickoff": CUTOFF + timedelta(days=1), "fixture_available_at": CUTOFF - timedelta(days=1),
            "status": "scheduled", "score_period": "regulation", "information_mode": "operational",
            "input_manifest": ["synthetic-fixture-lineage"]}


def feature_snapshot():
    return {"match_id": "target", "prediction_cutoff": CUTOFF, "information_mode": "operational",
            "home_elo": 1600.0, "away_elo": 1500.0, "features_available_at": CUTOFF - timedelta(minutes=1),
            "elo_available_at": CUTOFF - timedelta(minutes=1), "input_manifest": {"refs": ["synthetic-features"]}}


def training_features():
    result = []
    for index, (home_goals, away_goals) in enumerate([(1, 0), (0, 0), (0, 1)] * 2):
        kickoff = CUTOFF - timedelta(days=20-index)
        result.append({"match_id": str(index), "competition_id": "synthetic-league", "home_team_id": "a", "away_team_id": "b",
                       "kickoff": kickoff, "fixture_available_at": kickoff - timedelta(days=2),
                       "result_available_at": kickoff + timedelta(hours=3), "home_goals": home_goals, "away_goals": away_goals,
                       "status": "finished", "score_period": "regulation", "information_mode": "operational",
                       "home_elo": 1800.0 - index*100, "away_elo": 1300.0 + index*100,
                       "prediction_cutoff": kickoff - timedelta(hours=1), "features_available_at": kickoff - timedelta(hours=2),
                       "elo_available_at": kickoff - timedelta(hours=2), "input_manifest": ["synthetic-result"],
                       "feature_input_manifest": {"refs": ["synthetic-prematch-features"]}})
    return result


def service(model_name):
    if model_name == "elo":
        return PredictionService.fixed_elo_mapping(as_of=CUTOFF, dataset_version="synthetic-test")
    return PredictionService.train(model_name, training_features(), as_of=CUTOFF, dataset_version="synthetic-test")


@pytest.mark.parametrize("field", ["features_available_at", "elo_available_at"])
@pytest.mark.parametrize("model_name", ["elo", "logistic", "tree"])
def test_prediction_rejects_either_future_availability_field(model_name, field):
    features = feature_snapshot()
    features[field] = CUTOFF + timedelta(seconds=1)
    with pytest.raises(ValueError, match="unavailable"):
        service(model_name).predict(fixture(), prediction_cutoff=CUTOFF, features=features)


@pytest.mark.parametrize("model_name", ["elo", "logistic", "tree"])
def test_numeric_prediction_requires_availability_evidence(model_name):
    features = feature_snapshot()
    features.pop("features_available_at")
    features.pop("elo_available_at")
    with pytest.raises(ValueError, match="availability"):
        service(model_name).predict(fixture(), prediction_cutoff=CUTOFF, features=features)


@pytest.mark.parametrize("model_name", ["elo", "logistic", "tree"])
def test_numeric_prediction_requires_explicit_snapshot(model_name):
    with pytest.raises(ValueError):
        service(model_name).predict(fixture(), prediction_cutoff=CUTOFF)


@pytest.mark.parametrize("bad_field", ["odds_line", "odds_selection_timestamps"])
def test_api_preserves_market_guard_fields(bad_field):
    model = PredictionService.train("market", [], as_of=CUTOFF, dataset_version="synthetic-test")
    features = feature_snapshot()
    features.update(odds_home=2.0, odds_draw=3.0, odds_away=4.0, odds_available_at=CUTOFF - timedelta(hours=1))
    if bad_field == "odds_line":
        features[bad_field] = 1
    else:
        features[bad_field] = [CUTOFF - timedelta(hours=1), CUTOFF - timedelta(hours=2), CUTOFF - timedelta(hours=1)]
    with pytest.raises(ValueError, match="unhandicapped|same quote"):
        model.predict(fixture(), prediction_cutoff=CUTOFF, features=features)


@pytest.mark.parametrize("model_name", ["elo", "logistic", "tree"])
def test_supervised_training_rejects_postmatch_feature_snapshot(model_name):
    rows = training_features()
    rows[0].update(prediction_cutoff=rows[0]["kickoff"] + timedelta(days=1),
                   features_available_at=rows[0]["kickoff"] + timedelta(hours=4),
                   elo_available_at=rows[0]["kickoff"] + timedelta(hours=4))
    with pytest.raises(ValueError, match="prematch prediction_cutoff"):
        PredictionService.train(model_name, rows, as_of=CUTOFF, dataset_version="synthetic-test")


@pytest.mark.parametrize("model_name", ["elo", "logistic", "tree"])
@pytest.mark.parametrize("missing", ["prediction_cutoff", "feature_input_manifest", "availability"])
def test_supervised_training_requires_historical_feature_evidence(model_name, missing):
    rows = training_features()
    if missing == "availability":
        rows[0].pop("features_available_at")
        rows[0].pop("elo_available_at")
    else:
        rows[0].pop(missing)
    with pytest.raises(ValueError):
        PredictionService.train(model_name, rows, as_of=CUTOFF, dataset_version="synthetic-test")


@pytest.mark.parametrize("model_name", ["elo", "logistic", "tree"])
@pytest.mark.parametrize("field", ["features_available_at", "elo_available_at"])
def test_supervised_training_rejects_features_after_historical_prediction(model_name, field):
    rows = training_features()
    rows[0][field] = rows[0]["prediction_cutoff"] + timedelta(seconds=1)
    with pytest.raises(ValueError, match="unavailable"):
        PredictionService.train(model_name, rows, as_of=CUTOFF, dataset_version="synthetic-test")


@pytest.mark.parametrize("model_name", ["elo", "logistic", "tree"])
def test_legal_feature_prediction_survives_json_roundtrip_and_detects_tampering(model_name, tmp_path):
    model = service(model_name)
    expected = model.predict(fixture(), prediction_cutoff=CUTOFF, features=feature_snapshot())
    path = model.save(tmp_path / "artifact.json")
    restored = PredictionService.load(path)
    assert restored.predict(fixture(), prediction_cutoff=CUTOFF, features=feature_snapshot()) == expected
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["trained_at"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        PredictionService.load(path)
