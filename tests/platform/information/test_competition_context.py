"""Constructed contract cases; none of these fixtures are real-match evidence."""
import json

import pytest

from football_platform.information.competition_context import project_competition_context


FIXTURE = {
    "match_id": "constructed-match", "competition_id": "constructed-group",
    "home_team_id": "team-a", "away_team_id": "team-b", "kickoff": "2026-09-20T12:00:00Z",
}
CUTOFF = "2026-09-20T11:00:00Z"


def evidence(evidence_id="qualification-a", **changes):
    result = {
        "evidence_id": evidence_id, "match_id": FIXTURE["match_id"],
        "competition_id": FIXTURE["competition_id"], "team_id": "team-a",
        "topic": "qualification", "value": "qualified",
        "available_at": "2026-09-20T09:00:00Z", "source_id": "constructed-official",
        "source_kind": "official", "epistemic_status": "verified_fact",
        "verified_by": "constructed-verifier-for-contract-test",
    }
    result.update(changes)
    return result


def project(*records, acquired=None, **changes):
    return project_competition_context(
        fixture=FIXTURE, prediction_cutoff=changes.get("cutoff", CUTOFF), observations=records,
        acquired_evidence_ids=acquired if acquired is not None else [record["evidence_id"] for record in records],
    )


def test_qualification_alone_does_not_invent_rotation_lineup_weakness_or_tactics():
    result = project(evidence())
    assert result.teams["team-a"]["qualification"].value == "qualified"
    for topic in ("rotation", "lineup", "tactical_intent", "rank_objective"):
        assert result.teams["team-a"][topic].status == "unknown"
        assert result.teams["team-a"][topic].value is None
    assert "strength" not in result.model_dump()
    assert result.teams["team-b"]["qualification"].value is None


def test_draw_sufficient_for_qualification_is_distinct_from_first_and_defense():
    record = evidence("objective", topic="rank_objective", value="draw_sufficient_for_qualification")
    result = project(record)
    assert result.teams["team-a"]["rank_objective"].value == "draw_sufficient_for_qualification"
    assert result.teams["team-a"]["tactical_intent"].status == "unknown"
    assert result.teams["team-a"]["qualification"].status == "unknown"


def test_verified_first_place_draw_condition_still_does_not_determine_tactics():
    record = evidence("objective", topic="rank_objective", value="draw_sufficient_for_first_place")
    assert project(record).teams["team-a"]["tactical_intent"].status == "unknown"


def test_other_match_dependency_retained_without_becoming_unconditional_first_place():
    premise = evidence("group-table", topic="stage", team_id=None, value="group")
    objective = evidence("objective", topic="rank_objective", value="conditional_on_other_fixture",
                         condition_fixture_ids=["other-match"], conditional_target="first_place",
                         premise_evidence_ids=["group-table"])
    result = project(premise, objective)
    assert result.teams["team-a"]["rank_objective"].value == "conditional_on_other_fixture"
    assert result.evidence["objective"].conditional_target == "first_place"
    assert result.evidence["objective"].condition_fixture_ids == ["other-match"]


def test_conditional_objective_requires_explicit_target_and_premises():
    with pytest.raises(ValueError, match="Conditional objectives"):
        project(evidence("objective", topic="rank_objective", value="conditional_on_other_fixture"))


def test_declared_tactical_hypothesis_can_reference_context_without_becoming_fact():
    premise = evidence("objective", topic="rank_objective", value="draw_sufficient_for_qualification")
    hypothesis = evidence("tactic", topic="tactical_intent", value="defense_oriented", source_kind="llm",
                          epistemic_status="hypothesis", verified_by=None, probability=0.6,
                          probability_basis="constructed_scenario_not_calibrated",
                          premise_evidence_ids=["objective"])
    result = project(premise, hypothesis)
    claim = result.teams["team-a"]["tactical_intent"]
    assert claim.status == "hypothesis" and claim.probability == 0.6
    assert result.evidence["tactic"].premise_evidence_ids == ["objective"]


@pytest.mark.parametrize("kind", ["llm", "analyst", "synthetic", "reporter"])
def test_self_declared_verified_fact_from_unverified_source_kind_is_rejected(kind):
    with pytest.raises(ValueError, match="Verified facts"):
        project(evidence(source_kind=kind))


def test_tactical_intent_is_not_an_observed_match_tactic_even_from_official_source():
    with pytest.raises(ValueError, match="remain hypotheses"):
        project(evidence(topic="tactical_intent", value="defense_oriented"))


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf")])
def test_hypothesis_probability_must_be_finite_and_in_unit_interval(probability):
    with pytest.raises(ValueError):
        project(evidence(epistemic_status="hypothesis", probability=probability,
                         probability_basis="constructed_not_calibrated"))


def test_probability_cannot_silently_mean_fact_confidence():
    with pytest.raises(ValueError, match="verification and hypothesis probability"):
        project(evidence(probability=0.9, probability_basis="wrong_kind"))


def test_unpurchased_payload_is_not_parsed_and_does_not_change_public_state():
    poison = {"evidence_id": "unbought", "topic": "lineup", "value": object(),
              "player_ids": ["SECRET_PLAYER"], "probability": float("nan")}
    blank = project(acquired=[]).model_dump(mode="json")
    actual = project(poison, acquired=[]).model_dump(mode="json")
    assert actual == blank
    assert "SECRET_PLAYER" not in json.dumps(actual)


@pytest.mark.parametrize("status", ["UNAVAILABLE", "STALE", "FAILED"])
def test_unavailable_payload_is_hidden_before_validation(status):
    poison = {"evidence_id": "unavailable", "availability_status": status, "value": object()}
    result = project(poison)
    assert not result.evidence and result.readiness == "no_context_observed"


def test_premise_not_acquired_hides_hypothesis_before_payload_validation():
    poison = {"evidence_id": "hypothesis", "premise_evidence_ids": ["unbought"], "value": object()}
    assert project(poison).evidence == {}


def test_purchased_but_missing_premise_does_not_validate_hypothesis():
    hypothesis = evidence("derived", epistemic_status="hypothesis", premise_evidence_ids=["missing"])
    assert project(hypothesis, acquired=["derived", "missing"]).evidence == {}


def test_cyclic_hypothesis_dependencies_are_not_self_supporting_evidence():
    first = evidence("a", epistemic_status="hypothesis", premise_evidence_ids=["b"])
    second = evidence("b", epistemic_status="hypothesis", premise_evidence_ids=["a"])
    assert project(first, second).evidence == {}


@pytest.mark.parametrize("field", ["available_at", "source_published_at", "observed_at"])
def test_future_acquired_lineup_is_rejected(field):
    record = evidence("xi", topic="lineup", value="starting_xi", player_ids=["player-a"])
    record[field] = "2026-09-20T11:01:00Z"
    with pytest.raises(ValueError, match="newer than prediction_cutoff"):
        project(record)


def test_available_time_cannot_precede_publication_even_before_cutoff():
    with pytest.raises(ValueError, match="before publication"):
        project(evidence(source_published_at="2026-09-20T10:00:00Z"))


def test_postmatch_report_is_rejected_even_if_timestamp_was_backdated():
    with pytest.raises(ValueError, match="Post-match"):
        project(evidence(observation_scope="post_match"))


@pytest.mark.parametrize("changes", [{"match_id": "other-match"}, {"competition_id": "other-competition"},
                                     {"team_id": "other-team"}])
def test_evidence_must_refer_to_the_same_fixture_and_a_participating_team(changes):
    with pytest.raises(ValueError, match="does not belong"):
        project(evidence(**changes))


@pytest.mark.parametrize("cutoff", [FIXTURE["kickoff"], "2026-09-20T13:00:00Z", "2026-09-20T11:00:00"])
def test_invalid_prediction_cutoff_is_rejected(cutoff):
    with pytest.raises(ValueError):
        project(cutoff=cutoff)


def test_conflicting_sources_retain_inert_provenance_and_abstain_from_single_value():
    first = evidence("provider-a")
    second = evidence("provider-b", value="not_yet_qualified", source_id="another-provider")
    result = project(first, second)
    claim = result.teams["team-a"]["qualification"]
    assert claim.status == "conflicting" and claim.value is None
    assert claim.alternatives == ["not_yet_qualified", "qualified"]
    assert claim.evidence_ids == ["provider-a", "provider-b"]
    assert result.evidence["provider-b"].source_id == "another-provider"


def test_duplicate_id_rejected_and_unknown_schema_fields_do_not_pass_through():
    with pytest.raises(ValueError, match="Duplicate"):
        project(evidence(), evidence())
    with pytest.raises(ValueError):
        project(evidence(final_score="3:0"))


def test_synthetic_examples_are_marked_separately_from_observed_verified_facts():
    result = project(evidence(source_kind="synthetic", epistemic_status="hypothesis", verified_by=None))
    assert result.readiness == "synthetic_only"
    assert result.teams["team-a"]["qualification"].status == "hypothesis"


def test_lineup_reveal_keeps_players_but_does_not_determine_strength():
    result = project(evidence("xi", topic="lineup", value="starting_xi", player_ids=["reserve-a", "reserve-b"]))
    assert result.evidence["xi"].player_ids == ["reserve-a", "reserve-b"]
    assert result.teams["team-a"]["rotation"].status == "unknown"
    assert result.teams["team-a"]["tactical_intent"].status == "unknown"


def test_stage_is_match_scoped_and_standings_are_not_guessed_from_missing_rules():
    result = project(evidence("stage", topic="stage", value="group", team_id=None))
    assert result.stage.value == "group"
    assert all(slot.status == "unknown" for team in result.teams.values() for slot in team.values())


@pytest.mark.parametrize("changes", [{"evidence_id": "../label.json"}, {"source_id": "C:labels"},
                                     {"verified_by": "fact because I say so"},
                                     {"content_hash": "https://example.com/score"},
                                     {"player_ids": ["C:\\secret"], "topic": "lineup", "value": "starting_xi"},
                                     {"premise_evidence_ids": ["../label.json"]}])
def test_public_identifiers_reject_paths_and_free_text(changes):
    record = evidence(**changes)
    with pytest.raises(ValueError, match="inert identifiers"):
        project(record, acquired=[record["evidence_id"], *record.get("premise_evidence_ids", [])])


def test_verified_conclusion_cannot_promote_an_unverified_premise():
    premise = evidence("premise", epistemic_status="hypothesis")
    conclusion = evidence("conclusion", topic="rank_objective", value="draw_sufficient_for_qualification",
                          premise_evidence_ids=["premise"])
    with pytest.raises(ValueError, match="verified supporting premises"):
        project(premise, conclusion)


def test_lineup_confirmation_and_rotation_observation_are_distinct_compatible_claims():
    lineup = evidence("xi", topic="lineup", value="starting_xi", player_ids=["reserve-a", "reserve-b"])
    rotation = evidence("rotation", topic="rotation", value="observed", premise_evidence_ids=["xi"])
    result = project(lineup, rotation)
    assert result.teams["team-a"]["lineup"].status == "verified_fact"
    assert result.teams["team-a"]["rotation"].value == "observed"


def test_two_different_verified_lineups_conflict_without_selecting_a_source():
    first = evidence("xi-a", topic="lineup", value="starting_xi", player_ids=["player-a"])
    second = evidence("xi-b", topic="lineup", value="starting_xi", player_ids=["player-b"])
    result = project(first, second)
    assert result.teams["team-a"]["lineup"].status == "conflicting"
    assert result.teams["team-a"]["lineup"].value is None


def test_same_lineup_player_set_does_not_conflict_due_to_order():
    first = evidence("xi-a", topic="lineup", value="starting_xi", player_ids=["player-a", "player-b"])
    second = evidence("xi-b", topic="lineup", value="starting_xi", player_ids=["player-b", "player-a"])
    assert project(first, second).teams["team-a"]["lineup"].status == "verified_fact"


def test_verified_conclusion_cannot_select_one_side_of_a_conflicting_premise_slot():
    first = evidence("qualified")
    second = evidence("not-qualified", value="not_yet_qualified")
    derived = evidence("objective", topic="rank_objective", value="position_locked",
                       objective_target="qualification", premise_evidence_ids=["qualified"])
    with pytest.raises(ValueError, match="unconflicted verified supporting premises"):
        project(first, second, derived)


def test_draw_ensures_qualification_while_first_place_depends_on_another_fixture():
    stage = evidence("stage", topic="stage", value="group", team_id=None)
    qualification = evidence("draw-qualification", topic="rank_objective", value="draw_sufficient_for_qualification")
    first_place = evidence("conditional-first", topic="rank_objective", value="conditional_on_other_fixture",
                           condition_fixture_ids=["other-group-fixture"], conditional_target="first_place",
                           premise_evidence_ids=["stage"])
    result = project(stage, qualification, first_place)
    objectives = result.rank_objectives["team-a"]
    assert objectives["qualification"].value == "draw_sufficient_for_qualification"
    assert objectives["first_place"].value == "conditional_on_other_fixture"
    summary = result.teams["team-a"]["rank_objective"]
    assert summary.status == "verified_fact" and summary.value is None
    assert summary.alternatives == []
    assert summary.dimension_values == {
        "qualification": "draw_sufficient_for_qualification", "first_place": "conditional_on_other_fixture"}
    assert result.teams["team-a"]["tactical_intent"].status == "unknown"


def test_opposing_conditions_for_the_same_objective_conflict():
    draw = evidence("draw", topic="rank_objective", value="draw_sufficient_for_qualification")
    win = evidence("win", topic="rank_objective", value="win_required_for_qualification")
    result = project(draw, win)
    assert result.rank_objectives["team-a"]["qualification"].status == "conflicting"
    assert result.rank_objectives["team-a"]["first_place"].status == "unknown"
    assert result.teams["team-a"]["rank_objective"].status == "conflicting"


def test_different_objective_does_not_conflict_with_verified_derivation_premise():
    qualification = evidence("draw", topic="rank_objective", value="draw_sufficient_for_qualification")
    first_place = evidence("win", topic="rank_objective", value="win_required_for_first_place")
    derived = evidence("qualified", premise_evidence_ids=["draw"])
    result = project(qualification, first_place, derived)
    assert result.teams["team-a"]["qualification"].status == "verified_fact"
    assert result.teams["team-a"]["rank_objective"].status == "verified_fact"


def test_position_locked_requires_an_explicit_objective():
    with pytest.raises(ValueError, match="explicit objective_target"):
        project(evidence("locked", topic="rank_objective", value="position_locked"))
    locked = evidence("locked", topic="rank_objective", value="position_locked", objective_target="qualification")
    result = project(locked)
    assert result.rank_objectives["team-a"]["qualification"].value == "position_locked"
    assert result.rank_objectives["team-a"]["first_place"].status == "unknown"


@pytest.mark.parametrize("value,target", [("draw_sufficient_for_qualification", "first_place"),
                                         ("win_required_for_first_place", "qualification")])
def test_explicit_target_must_match_the_rank_condition(value, target):
    with pytest.raises(ValueError, match="target disagrees"):
        project(evidence("target", topic="rank_objective", value=value, objective_target=target))


def test_conditional_and_explicit_target_must_agree():
    stage = evidence("stage", topic="stage", value="group", team_id=None)
    condition = evidence("conditional", topic="rank_objective", value="conditional_on_other_fixture",
                         objective_target="qualification", conditional_target="first_place",
                         condition_fixture_ids=["other-game"], premise_evidence_ids=["stage"])
    with pytest.raises(ValueError, match="target disagrees"):
        project(stage, condition)


@pytest.mark.parametrize("actual", ["observed", "not_observed"])
def test_rotation_plan_and_actual_rotation_are_distinct_even_when_plan_not_realized(actual):
    plan = evidence("plan", topic="rotation", value="planned")
    realized = evidence("actual", topic="rotation", value=actual)
    result = project(plan, realized)
    assert result.rotation_states["team-a"]["planned"].value == "planned"
    assert result.rotation_states["team-a"]["actual"].value == actual
    assert result.teams["team-a"]["rotation"].status == "verified_fact"
    assert result.teams["team-a"]["rotation"].dimension_values == {"planned": "planned", "actual": actual}


def test_same_phase_rotation_disagreement_remains_conflicting():
    observed = evidence("yes", topic="rotation", value="observed")
    not_observed = evidence("no", topic="rotation", value="not_observed")
    result = project(observed, not_observed)
    assert result.rotation_states["team-a"]["actual"].status == "conflicting"
    assert result.rotation_states["team-a"]["planned"].status == "unknown"
    assert result.teams["team-a"]["rotation"].status == "conflicting"
