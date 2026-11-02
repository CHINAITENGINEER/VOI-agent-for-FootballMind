"""Constructed cases test model mechanics; they are not real match evaluation."""
from datetime import datetime, timedelta, timezone
import json
import math

import numpy as np
import pytest

from football_research.baselines.models import (
    DixonColesModel, EloMapping, LeagueFrequency, LogisticBaseline, MarketBaseline,
    PoissonModel, TreeBaseline, model_from_state, score_probabilities,
)


def rows():
    start = datetime(2020, 1, 1, 12, tzinfo=timezone.utc)
    result = []
    scores = [(2, 0), (1, 1), (0, 1), (3, 0), (0, 0), (1, 2), (2, 1), (2, 2), (0, 2)]
    for i, (home, away) in enumerate(scores * 3):
        kickoff = start + timedelta(days=i)
        result.append({"match_id": str(i), "competition_id": "league", "home_team_id": str(i % 4),
                       "away_team_id": str((i + 1) % 4), "kickoff": kickoff.isoformat(),
                       "result_available_at": (kickoff + timedelta(hours=3)).isoformat(),
                       "home_goals": home, "away_goals": away, "score_period": "regulation",
                       "home_elo": 1400 + 20 * i, "away_elo": 1600 - 10 * i})
    return result


def check_probability(probabilities):
    assert len(probabilities) == 3
    assert all(math.isfinite(p) and p >= 0 for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-12)


def test_frequency_uses_training_league_and_smooths_missing_classes():
    data = rows()
    model = LeagueFrequency(alpha=1).fit(data[:2])
    assert model.predict_proba(data[2]) == pytest.approx((2/5, 2/5, 1/5))
    assert model.predict_proba({"competition_id": "unseen"}) == pytest.approx((2/5, 2/5, 1/5))
    other = {**data[2], "competition_id": "other"}
    model.fit(data[:2] + [other])
    assert model.predict_proba(other) == pytest.approx((1/4, 1/4, 2/4))


@pytest.mark.parametrize("model_class", [LeagueFrequency, PoissonModel, DixonColesModel, EloMapping, LogisticBaseline, TreeBaseline])
def test_temporal_fit_rejects_unknown_future_and_duplicate_results(model_class):
    data = rows()
    with pytest.raises(ValueError, match="unavailable"):
        model_class().fit(data, as_of=data[-1]["kickoff"])
    unknown = [{k: v for k, v in data[0].items() if k != "result_available_at"}]
    with pytest.raises(ValueError, match="result_available_at"):
        model_class().fit(unknown, as_of="2021-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="Duplicate"):
        model_class().fit([data[0], data[0]])


@pytest.mark.parametrize("model_class", [LeagueFrequency, PoissonModel, DixonColesModel, EloMapping, LogisticBaseline, TreeBaseline])
def test_models_probabilities_and_exact_json_round_trip(model_class):
    data = rows()
    model = model_class().fit(data, as_of="2021-01-01T00:00:00Z")
    expected = model.predict_proba(data[0])
    check_probability(expected)
    state = json.loads(json.dumps(model.to_state(), allow_nan=False))
    loaded = model_from_state(state)
    assert loaded.predict_proba(data[0]) == pytest.approx(expected, abs=1e-13)
    assert loaded.to_state()["audit"]["training_rows"] == len(data)


def test_score_distribution_correct_axes_adaptive_tail_and_dc_draw_effect():
    home = score_probabilities(3.5, 0.5)
    away = score_probabilities(0.5, 3.5)
    assert home[0] > home[2]
    assert home == pytest.approx(tuple(reversed(away)))
    equal = score_probabilities(1.2, 1.2)
    assert equal[0] == pytest.approx(equal[2])
    assert score_probabilities(1.2, 1.2, rho=-0.1)[1] > equal[1]
    high = score_probabilities(30, 28)
    check_probability(high)
    # A fixed 10-goal grid would grossly misrepresent these rates.
    assert high[0] > 0.5
    check_probability(score_probabilities(30, 28, rho=-0.2))
    check_probability(score_probabilities(30, 28, rho=0.2))


def test_dixon_coles_fits_a_nonzero_correction_and_positive_factors():
    data = rows()
    model = DixonColesModel().fit(data)
    state = model.leagues["league"]
    assert state["optimizer_success"]
    assert abs(state["rho"]) > 0.01
    for row in data:
        lam, mu = model.rates(row)
        rho = state["rho"]
        assert min(1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, 1 - rho) > 0


@pytest.mark.parametrize("model_class", [PoissonModel, DixonColesModel])
def test_unconverged_goal_model_is_rejected(model_class):
    with pytest.raises(ValueError, match="did not converge"):
        model_class(max_iterations=1).fit(rows())


@pytest.mark.parametrize("model_class", [PoissonModel, DixonColesModel])
def test_goal_models_learn_team_strength_and_have_auditable_optimizer(model_class):
    data = rows()
    for row in data:
        row["home_team_id"], row["away_team_id"] = "strong", "weak"
        row["home_goals"], row["away_goals"] = 3, 0
    model = model_class().fit(data)
    home, away = model.rates(data[0])
    assert home > away
    assert model.predict_proba(data[0])[0] > 0.8
    check_probability(model.predict_proba({"competition_id": "new", "home_team_id": "new1", "away_team_id": "new2"}))
    check_probability(model.predict_proba({"competition_id": "league", "home_team_id": "new1", "away_team_id": "new2"}))
    assert model.leagues["league"]["optimizer_success"]
    assert model.leagues["league"]["iterations"] > 0


def market_row():
    return {"odds_home": 2.0, "odds_draw": 3.0, "odds_away": 4.0,
            "prediction_cutoff": "2020-01-02T10:00:00Z", "kickoff": "2020-01-02T12:00:00Z",
            "odds_available_at": "2020-01-02T09:59:00Z", "odds_period": "regulation", "odds_market_type": "1x2"}


def test_market_proportional_devig_and_json():
    row = market_row()
    model = MarketBaseline().fit([])
    assert model.predict_proba(row) == pytest.approx((6/13, 4/13, 3/13))
    assert model_from_state(json.loads(json.dumps(model.to_state()))).predict_proba(row) == model.predict_proba(row)


@pytest.mark.parametrize("change,error", [
    ({"odds_available_at": "2020-01-02T10:01:00Z"}, "unavailable"),
    ({"odds_available_at": "2019-12-30T00:00:00Z"}, "maximum age"),
    ({"odds_available_at": None}, "required inputs"),
    ({"odds_source_timestamp": "2020-01-02T10:01:00Z"}, "after"),
    ({"odds_period": "extra_time"}, "regulation"),
    ({"odds_market_type": "jc_handicap_3way"}, "regulation"),
    ({"odds_line": 1}, "unhandicapped"),
    ({"odds_home": 1}, "greater than one"),
    ({"odds_home": float("inf")}, "greater than one"),
    ({"prediction_cutoff": "2020-01-02T12:00:00Z"}, "precede kickoff"),
    ({"odds_selection_timestamps": ["2020-01-02T09:00:00Z", "2020-01-02T09:01:00Z", "2020-01-02T09:00:00Z"]}, "same quote"),
])
def test_market_rejects_wrong_time_and_semantics(change, error):
    with pytest.raises(ValueError, match=error):
        MarketBaseline().predict_proba({**market_row(), **change})


def test_elo_mapping_consumes_ratings_without_updating_them():
    model = EloMapping(home_advantage=0)
    row = {"home_elo": 1500.0, "away_elo": 1500.0}
    symmetric = model.predict_proba(row)
    assert symmetric[0] == pytest.approx(symmetric[2])
    assert model.predict_proba({"home_elo": 1800.0, "away_elo": 1500.0})[0] > symmetric[0]
    assert row == {"home_elo": 1500.0, "away_elo": 1500.0}
    with pytest.raises(ValueError, match="Existing finite"):
        model.predict_proba({"home_elo": None, "away_elo": 1500.0})
    with pytest.raises(ValueError, match="unavailable"):
        model.predict_proba({**row, "elo_available_at": "2020-01-03T00:00:00Z", "prediction_cutoff": "2020-01-02T00:00:00Z"})


@pytest.mark.parametrize("model_class", [LogisticBaseline, TreeBaseline])
def test_preprocessing_is_fitted_on_training_only(model_class):
    data = rows()
    data[0]["home_elo"] = None
    model = model_class().fit(data)
    median = model.fitted["median"][:]
    assert median[0] == np.median([r["home_elo"] for r in data if r["home_elo"] is not None])
    before = json.dumps(model.to_state(), sort_keys=True)
    check_probability(model.predict_proba({"home_elo": 1e8, "away_elo": None}))
    assert json.dumps(model.to_state(), sort_keys=True) == before
    with pytest.raises(ValueError, match="excluding outcome"):
        model_class(feature_names=["home_goals"])
    with pytest.raises(ValueError, match="All H/D/A"):
        model_class().fit(data[:1])
    data[0].update(prediction_cutoff="2020-01-01T10:00:00Z", features_available_at="2020-01-01T11:00:00Z")
    with pytest.raises(ValueError, match="unavailable"):
        model_class().fit(data)


def test_rejects_nonregulation_labels_and_naive_cutoff():
    with pytest.raises(ValueError, match="regulation"):
        LeagueFrequency().fit([{**rows()[0], "score_period": "extra_time"}])
    with pytest.raises(ValueError, match="timezone"):
        LeagueFrequency().fit(rows(), as_of="2021-01-01T00:00:00")
    with pytest.raises(ValueError, match="Unknown model"):
        model_from_state({"model_version": "other"})
