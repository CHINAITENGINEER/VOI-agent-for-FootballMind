from copy import deepcopy
import json

import pytest

from football_agent.scenario_fixture_pack import (
    SCHEMA_VERSION,
    actor_projection,
    bundle_digest,
    generate_scenario_fixtures,
    get_scenario_fixture,
    validate_scenario_fixture,
    validate_actor_projection,
    write_fixture_bundle,
)


EXPECTED_ALIASES = {
    "qualified-but-strong-performance",
    "draw-secures-group-first",
    "key-forward-injury",
    "weather-low-value",
    "odds-news-conflict",
    "stale-source",
    "missing-source",
    "duplicate-source",
}


def test_generated_bundle_is_complete_and_valid():
    fixtures = generate_scenario_fixtures()
    assert len({item["fixture_id"] for item in fixtures}) == len(EXPECTED_ALIASES)
    assert all(item["fixture_id"].startswith("scenario-") for item in fixtures)
    assert all(item["fixture_id"] not in EXPECTED_ALIASES for item in fixtures)
    assert all(validate_scenario_fixture(item)["valid"] for item in fixtures)
    assert all(item["synthetic"] is True for item in fixtures)


def test_generation_is_deterministic_and_returns_fresh_copies():
    one = generate_scenario_fixtures()
    two = generate_scenario_fixtures()
    assert one == two
    assert bundle_digest(one) == bundle_digest(two)
    one[0]["public_snapshot"]["known_information"]["tamper"] = True
    assert one != generate_scenario_fixtures()


def test_actor_projection_hides_evaluator_policy_and_post_match_reference():
    fixture = get_scenario_fixture("qualified-but-strong-performance")
    view = actor_projection(fixture)
    assert "expected_policy" not in view
    assert "evaluation_reference" not in view
    serialized = json.dumps(view, ensure_ascii=False)
    assert "3-0" not in serialized
    assert "evaluation_value_class" not in serialized
    assert view["snapshot_digest"]
    assert view["snapshot"]["known_information"]["home_already_qualified"] is True
    assert validate_actor_projection(view)["valid"] is True


def test_japan_example_keeps_qualification_as_context_without_forcing_update():
    fixture = get_scenario_fixture("qualified-but-strong-performance")
    policy = fixture["expected_policy"]
    assert policy["high_value_candidates"] == ["rotation_check"]
    assert "weather_check" in policy["low_value_candidates"]
    assert policy["expected_first_selection"] == ["rotation_check"]
    assert "automatically" not in policy["rationale"]
    assert "3-0" in fixture["evaluation_reference"]["observed_pattern"]
    assert fixture["evaluation_reference"]["provenance"] == "user_supplied_unverified"
    assert fixture["evaluation_reference"]["not_gold"] is True
    assert fixture["expected_policy"]["gold"] is False


def test_competition_constraint_and_injury_scenarios_prioritize_decision_sensitive_questions():
    standings = get_scenario_fixture("draw-secures-group-first")
    injury = get_scenario_fixture("key-forward-injury")
    assert standings["expected_policy"]["expected_first_selection"] == ["qualification_constraint"]
    assert "qualification_constraint" in standings["expected_policy"]["high_value_candidates"]
    assert injury["expected_policy"]["expected_first_selection"] == ["key_forward_availability"]
    assert injury["expected_policy"]["high_value_candidates"] == ["key_forward_availability"]


def test_low_value_weather_has_an_explicit_stop_boundary():
    fixture = get_scenario_fixture("weather-low-value")
    expected = fixture["expected_policy"]
    assert expected["low_value_candidates"] == ["weather_detail"]
    assert expected["stop_after_candidates"] == ["lineup_check"]
    assert "停止" in expected["stop_condition"]


@pytest.mark.parametrize(
    "fixture_id, expected_state, expected_first",
    [
        ("odds-news-conflict", "conflict", "independent_lineup_verify"),
        ("stale-source", "stale", "fresh_lineup"),
        ("missing-source", "missing", "club_news_injury"),
        ("duplicate-source", "duplicate", "independent_injury_confirm"),
    ],
)
def test_source_quality_cases_are_explicit_and_route_to_independent_evidence(
    fixture_id, expected_state, expected_first
):
    fixture = get_scenario_fixture(fixture_id)
    states = {item["source_state"] for item in fixture["candidate_menu"]}
    states.update(item["status"] for item in fixture["public_snapshot"]["source_observations"])
    assert expected_state in states
    assert fixture["expected_policy"]["expected_first_selection"] == [expected_first]
    assert all(item["source_ids"] for item in fixture["candidate_menu"])


def test_public_snapshot_result_leak_is_rejected():
    fixture = get_scenario_fixture("weather-low-value")
    tampered = deepcopy(fixture)
    tampered["public_snapshot"]["known_information"]["result"] = {"winner": "home"}
    report = validate_scenario_fixture(tampered)
    assert report["valid"] is False
    assert any("public_snapshot_result_leak" in item for item in report["reasons"])
    with pytest.raises(ValueError, match="SCENARIO_FIXTURE_INVALID"):
        actor_projection(tampered)


def test_actor_projection_strict_allowlist_drops_private_candidate_fields():
    fixture = get_scenario_fixture("weather-low-value")
    tampered = deepcopy(fixture)
    tampered["candidate_menu"][0]["predicted_answer"] = "home"
    # The fixture remains structurally valid only if a private field is
    # rejected; the actor projection must never forward it.
    assert validate_scenario_fixture(tampered)["valid"] is False
    tampered["candidate_menu"][0].pop("predicted_answer")
    view = actor_projection(fixture)
    assert all("predicted_answer" not in candidate for candidate in view["candidate_menu"])


@pytest.mark.parametrize("mutation, reason", [
    (lambda f: f["public_snapshot"]["source_observations"][0].__setitem__("observed_at", "2026-10-01T00:00:00Z"), "source_after_cutoff"),
    (lambda f: f["candidate_menu"][0].__setitem__("source_ids", ["does-not-exist"]), "source_reference_missing"),
    (lambda f: f["candidate_menu"][0].__setitem__("declared_cost", {}), "cost_invalid"),
    (lambda f: f["candidate_menu"][0].__setitem__("requires", ["unknown-candidate"]), "requires_missing"),
])
def test_as_of_source_binding_and_candidate_contract_are_checked(mutation, reason):
    fixture = get_scenario_fixture("weather-low-value")
    tampered = deepcopy(fixture)
    mutation(tampered)
    report = validate_scenario_fixture(tampered)
    assert report["valid"] is False
    assert any(reason in item for item in report["reasons"])


def test_candidate_dependency_cycle_is_rejected():
    fixture = get_scenario_fixture("draw-secures-group-first")
    tampered = deepcopy(fixture)
    tampered["candidate_menu"][0]["requires"] = ["tactical_intent"]
    report = validate_scenario_fixture(tampered)
    assert report["valid"] is False
    assert any("candidate_dependency_cycle" in item for item in report["reasons"])


def test_malformed_non_string_dependency_and_source_ids_are_rejected_cleanly():
    fixture = get_scenario_fixture("weather-low-value")
    tampered = deepcopy(fixture)
    tampered["candidate_menu"][0]["requires"] = [["nested"]]
    tampered["candidate_menu"][1]["source_ids"] = [["nested"]]
    report = validate_scenario_fixture(tampered)
    assert report["valid"] is False
    assert any("requires_invalid" in item for item in report["reasons"])
    assert any("source_ids_invalid" in item for item in report["reasons"])


def test_bundle_writer_emits_versioned_deterministic_json(tmp_path):
    path = write_fixture_bundle(tmp_path / "round133-b-fixtures.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["fixture_count"] == len(EXPECTED_ALIASES)
    assert payload["bundle_digest"] == bundle_digest(payload["fixtures"])
