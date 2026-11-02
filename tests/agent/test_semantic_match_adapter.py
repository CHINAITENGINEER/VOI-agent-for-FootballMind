from datetime import datetime, timezone

import pytest

from football_agent.semantic_match_adapter import VERSION, attach_match_state


NOW = datetime(2026, 10, 8, 8, tzinfo=timezone.utc)


def case():
    return {"source_profile": {"teams": [
        {"side": "home", "canonical_name": "Santos"},
        {"side": "away", "canonical_name": "Flamengo"},
    ]}, "match": {"kickoff": "2026-10-10T12:00:00Z"},
        "structured_baseline": {"probabilities": [.4, .3, .3]}}


def test_adapter_binds_match_only_state_without_assigning_values():
    bound = attach_match_state(case(), {"home": "Santos", "away": "Flamengo",
                                        "competition": "Brasileirao",
                                        "kickoff": "2026-10-10T20:00:00+08:00"}, now=NOW)
    assert bound["autonomous_adapter"]["version"] == VERSION
    assert bound["autonomous_initial_state"]["contract"]["no_user_scope_question"] is True
    assert bound["autonomous_initial_state"]["baseline"]["probabilities"] is None
    assert bound["autonomous_request"]["policy"]["numerical_voi_calibrated"] is False


@pytest.mark.parametrize("match", [
    {"home": "Flamengo", "away": "Santos"},
    {"home": "Santos", "away": "Palmeiras"},
])
def test_adapter_rejects_fixture_switch_or_home_away_swap(match):
    with pytest.raises(ValueError, match="registered semantic case"):
        attach_match_state(case(), match, now=NOW)


def test_adapter_accepts_registered_input_aliases_without_changing_fixture_side():
    registered = case()
    registered["source_profile"]["teams"][0].update(canonical_name="Santos FC", input_name="Santos")
    registered["source_profile"]["teams"][1].update(canonical_name="Flamengo RJ", input_name="Flamengo")
    bound = attach_match_state(registered, "Santos vs Flamengo", now=NOW)
    assert bound["autonomous_initial_state"]["match"]["home"] == "Santos"
    assert bound["autonomous_initial_state"]["match"]["away"] == "Flamengo"


@pytest.mark.parametrize("field,value", [
    ("kickoff", "2026-10-11T12:00:00Z"),
    ("competition", "Copa do Brasil"),
    ("match_id", "other-fixture"),
])
def test_adapter_rejects_same_teams_when_explicit_fixture_context_changes(field, value):
    registered = case()
    registered["match"].update({"match_id": "fixture-1", "competition": "Brasileirao",
                                 "prediction_cutoff": "2026-10-09T12:00:00Z"})
    requested = {"home": "Santos", "away": "Flamengo", "kickoff": "2026-10-10T12:00:00Z",
                 "prediction_cutoff": "2026-10-09T12:00:00Z", "match_id": "fixture-1",
                 "competition": "Brasileirao"}
    requested[field] = value
    with pytest.raises(ValueError, match="does not match requested fixture"):
        attach_match_state(registered, requested, now=NOW)


@pytest.mark.parametrize("text", [
    "Santos Vs Flamengo", "Santos vs. Flamengo", "Santos  v  Flamengo",
])
def test_adapter_accepts_case_insensitive_latin_match_separators(text):
    bound = attach_match_state(case(), text, now=NOW)
    assert bound["autonomous_request"]["match"]["home"] == "Santos"
    assert bound["autonomous_request"]["match"]["away"] == "Flamengo"
