"""Constructed fixtures test mechanisms only; they are not real evaluation data."""
from datetime import datetime, timedelta, timezone
import json

import pytest
from pydantic import ValidationError

from football_research.baselines.api import Prediction, PredictionService, training_rows

UTC = timezone.utc
CUTOFF = datetime(2025, 6, 1, tzinfo=UTC)


def row(mid="old", kickoff=None, **kwargs):
    kickoff = kickoff or CUTOFF - timedelta(days=7)
    return dict(match_id=mid, competition_id="test-league", home_team_id="a", away_team_id="b",
                kickoff=kickoff, home_goals=0, away_goals=0,
                result_available_at=kickoff + timedelta(hours=3), fixture_available_at=kickoff - timedelta(days=5),
                information_mode="operational", score_period="regulation", status="finished",
                input_manifest=["synthetic-unit-fixture"], **kwargs)


def fixture():
    result = row("target", CUTOFF + timedelta(days=1))
    result.update(status="scheduled", home_goals=None, away_goals=None, result_available_at=None)
    return result


def test_cutoff_rejects_late_known_historical_result_and_deduplicates_snapshots():
    historic = row()
    late = dict(row("late"), result_available_at=CUTOFF + timedelta(seconds=1))
    exact = dict(row("exact"), result_available_at=CUTOFF)
    selected = training_rows([historic, dict(historic), late, exact], as_of=CUTOFF, information_mode="operational")
    assert [r["match_id"] for r in selected] == ["old"]


def test_prediction_roundtrip_and_no_target_label_leakage(tmp_path):
    service = PredictionService.train("frequency", [row()], as_of=CUTOFF, dataset_version="test-only")
    target = fixture()
    prediction = service.predict(target, prediction_cutoff=CUTOFF)
    target.update(home_goals=99, away_goals=0)
    assert service.predict(target, prediction_cutoff=CUTOFF) == prediction
    path = service.save(tmp_path / "model.json")
    assert PredictionService.load(path).predict(target, prediction_cutoff=CUTOFF) == prediction
    modified = json.loads(path.read_text(encoding="utf-8"))
    modified["dataset_version"] = "tampered"
    path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        PredictionService.load(path)


@pytest.mark.parametrize("change,error", [
    ({"fixture_available_at": CUTOFF + timedelta(minutes=1)}, "not known"),
    ({"kickoff": CUTOFF}, "precede kickoff"),
    ({"information_mode": "historical_reconstruction"}, "modes differ"),
    ({"status": "postponed"}, "Inactive"),
    ({"score_period": "extra_time"}, "regulation"),
])
def test_prediction_time_and_scope_guards(change, error):
    service = PredictionService.train("frequency", [row()], as_of=CUTOFF, dataset_version="test-only")
    with pytest.raises(ValueError, match=error):
        service.predict(dict(fixture(), **change), prediction_cutoff=CUTOFF)


def test_conflicting_result_versions_fail_closed():
    with pytest.raises(ValueError, match="Conflicting"):
        training_rows([row(), dict(row(), home_goals=2)], as_of=CUTOFF, information_mode="operational")


def test_prediction_contract_rejects_bad_probability_mass():
    with pytest.raises(ValidationError):
        Prediction(match_id="m", prediction_cutoff=CUTOFF, home_win_probability=.5, draw_probability=.5,
                   away_win_probability=.5, model_version="test", dataset_version="test", input_manifest={}, information_mode="operational")


def test_feature_cutoff_must_match_and_future_odds_fail():
    service = PredictionService.train("market", [], as_of=CUTOFF, dataset_version="test-only")
    features = dict(match_id="target", prediction_cutoff=CUTOFF, information_mode="operational", input_manifest={"test": True},
                    odds_home=2., odds_draw=3., odds_away=4., odds_available_at=CUTOFF + timedelta(seconds=1))
    with pytest.raises(ValueError, match="unavailable"):
        service.predict(fixture(), prediction_cutoff=CUTOFF, features=features)
    features["prediction_cutoff"] = CUTOFF - timedelta(seconds=1)
    with pytest.raises(ValueError, match="exact prediction cutoff"):
        service.predict(fixture(), prediction_cutoff=CUTOFF, features=features)
