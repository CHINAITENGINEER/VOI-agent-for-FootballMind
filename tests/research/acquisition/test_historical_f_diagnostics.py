"""Exact frozen-function accounting, independent of outcomes or the live run."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from football_research.acquisition.historical_context_training import HistoricalContextModel
from football_research.acquisition.historical_f_diagnostics import diagnose_transition

MODEL = Path(__file__).resolve().parents[3] / "var/submission/p9-g-timestamped-replay-20261010/model.json"


@pytest.fixture
def states():
    if not MODEL.is_file():
        pytest.skip("Separately retained frozen f artifact absent")
    model = HistoricalContextModel.load(MODEL)
    before = {"fixture": {"physical_match_id": "diagnostic-independent-fixture", "home_team_id": "home-1",
        "away_team_id": "away-1", "home_name": "Home Town", "away_name": "Away City",
        "competition_id": "league-1", "kickoff": "2026-10-12T14:00:00+00:00"},
        "decision_at": "2026-10-10T12:00:00+00:00", "prediction_cutoff": "2026-10-10T12:00:00+00:00",
        "information_mode": "operational", "prior_probabilities": None, "history": [], "documents": [],
        "observed_links": [{"url": "https://www.fulhamfc.com/news/article", "title": "Team news and training",
            "parent_url": "https://www.fulhamfc.com/news/", "available_at": "2026-10-10T11:00:00+00:00",
            "title_basis": "observed_parent_link"}]}
    after = deepcopy(before)
    after.update(decision_at="2026-10-10T13:00:00+00:00", prediction_cutoff="2026-10-10T13:00:00+00:00")
    after["documents"] = [{"url": "https://www.fulhamfc.com/news/article", "title": "Away City training",
        "text": "Away City have returned to training before the match.",
        "paragraphs": ["Away City have returned to training before the match."],
        "captured_at": "2026-10-10T12:10:00+00:00", "available_at": "2026-10-10T12:11:00+00:00",
        "source_published_at": "2026-10-10T10:00:00+00:00", "raw_sha256": "a" * 64}]
    return model, before, after


def test_decomposition_reconstructs_log_odds_and_preserves_inputs(states, monkeypatch):
    model, before, after = states
    original = deepcopy((before, after, model._payload()))
    monkeypatch.setattr(model, "estimate", lambda *a, **k: pytest.fail("Diagnostics must not call g"))
    report = diagnose_transition(model, before, after)
    assert report["maximum_accounting_error"] < 1e-10
    summed = np.sum([row["log_odds_delta"] for row in report["feature_block_deltas"]], axis=0)
    assert summed == pytest.approx(report["same_clock_log_odds_delta"])
    assert report["new_document_views"][0]["without_this_document_probabilities"] == pytest.approx(
        report["before_information_at_after_clock_probabilities"])
    assert report["expected_gain"] is None and report["realized_gain"] is None
    assert (before, after, model._payload()) == original


def test_identical_information_and_clock_has_zero_increment(states):
    model, _, after = states
    report = diagnose_transition(model, after, deepcopy(after))
    assert report["same_clock_log_odds_delta"] == [0., 0.]
    assert report["clock_only_probability_delta"] == [0., 0., 0.]
    assert report["new_document_views"] == []


def test_clock_drift_not_attributed_to_new_information(states):
    model, _, before = states
    after = deepcopy(before)
    after.update(decision_at="2026-10-10T13:30:00+00:00", prediction_cutoff="2026-10-10T13:30:00+00:00")
    report = diagnose_transition(model, before, after)
    assert report["same_clock_log_odds_delta"] == [0., 0.]
    assert np.asarray(report["observed_after_probabilities"]) - report["observed_before_probabilities"] == pytest.approx(
        report["clock_only_probability_delta"])


@pytest.mark.parametrize("change", ["fixture", "prior"])
def test_different_fixture_or_changed_prior_is_not_an_information_increment(states, change):
    model, before, after = states
    if change == "fixture": after["fixture"]["away_team_id"] = "different-team"
    else: after["prior_probabilities"] = [.4, .3, .3]
    with pytest.raises(ValueError):
        diagnose_transition(model, before, after)
