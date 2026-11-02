from datetime import datetime, timezone

import pytest

from football_agent.autonomous_match_contract import (
    POLICY_MODE, VERSION, STATE_VERSION,
    build_autonomous_initial_state, normalize_match_request, bind_source_capabilities,
)


NOW = datetime(2026, 10, 8, 8, tzinfo=timezone.utc)


def test_match_only_text_creates_explicit_unresolved_temporal_state():
    request = normalize_match_request("Santos vs Flamengo", now=NOW)
    assert request["schema"] == VERSION
    assert request["match"]["home"] == "Santos"
    assert request["temporal"]["status"] == "unresolved"
    assert request["temporal"]["resolution_required"] == ["kickoff"]
    assert request["policy"]["mode"] == POLICY_MODE


def test_complete_future_request_is_normalized_to_utc_and_has_stable_key():
    value = {"home": " China ", "away": "UAE", "competition": "Asian Games",
             "kickoff": "2026-10-10T20:00:00+08:00",
             "prediction_cutoff": "2026-10-10T12:00:00+08:00"}
    first = normalize_match_request(value, now=NOW)
    second = normalize_match_request(value, now=NOW)
    assert first == second
    assert first["temporal"]["status"] == "upcoming"
    assert first["match"]["kickoff"] == "2026-10-10T12:00:00Z"
    assert first["match"]["prediction_cutoff"] == "2026-10-10T04:00:00Z"


@pytest.mark.parametrize("value", [
    {"home": "A", "away": "B", "home_score": 3},
    {"home": "A", "away": "B", "result": "home"},
    {"home": "A", "away": "B", "kickoff": "2026-10-10T20:00:00"},
])
def test_outcome_leakage_and_naive_time_are_rejected(value):
    with pytest.raises(ValueError):
        normalize_match_request(value, now=NOW)


def test_prematch_past_kickoff_requires_explicit_retrospective_mode():
    with pytest.raises(ValueError, match="retrospective"):
        normalize_match_request({"home": "A", "away": "B", "kickoff": "2026-10-07T20:00:00Z"}, now=NOW)
    request = normalize_match_request({"home": "A", "away": "B", "kickoff": "2026-10-07T20:00:00Z",
                                       "mode": "retrospective"}, now=NOW)
    assert request["temporal"]["status"] == "past"


def test_initial_state_makes_agent_ownership_and_unknowns_auditable():
    state = build_autonomous_initial_state({"home": "Japan", "away": "Thailand",
                                            "competition": "Asian Games",
                                            "kickoff": "2026-10-10T12:00:00Z"})
    assert state["schema"] == STATE_VERSION
    assert state["contract"]["no_user_scope_question"] is True
    assert state["next_step"]["user_question"] is None
    ids = {item["id"] for item in state["unknown_information"]}
    assert {"qualification_constraint", "rotation_lineup", "tactical_intent"} <= ids
    catalog = {item["information_need"]: item for item in state["candidate_catalog"]}
    assert catalog["rotation_lineup"]["prerequisites"] == ["qualification_constraint"]
    assert catalog["rotation_lineup"]["conditional_value"] is None
    assert all(item["status"] == "unscored" for item in catalog.values())
    assert state["baseline"]["probabilities"] is None
    assert all(item["status"] == "unknown" for item in state["unknown_information"])


def test_capability_binding_reports_health_without_turning_it_into_value_or_order():
    state = build_autonomous_initial_state({"home": "Japan", "away": "Thailand",
                                            "kickoff": "2026-10-10T12:00:00Z"})
    bound = bind_source_capabilities(state, {
        "standings": "available", "competition_rules": "available",
        "official_lineup": "unknown", "team_news": "stale", "weather": "unavailable",
    })
    rows = {item["information_need"]: item for item in bound["candidate_catalog"]}
    assert rows["qualification_constraint"]["availability"] == "available"
    assert rows["rotation_lineup"]["availability"] == "stale_or_conflicting"
    assert rows["schedule_rest_weather"]["availability"] in {"unknown", "stale_or_conflicting"}
    assert all(item["conditional_value"] is None for item in rows.values())
    assert bound["capability_binding"]["priority_assigned"] is False
