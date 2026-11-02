import pytest

from football_agent.verify_round6 import competition_context_integration


@pytest.mark.parametrize("outcome", ["starters", "reserves"])
def test_only_paid_lineup_evidence_enters_public_context_and_model_update(outcome):
    run = competition_context_integration(lineup_outcome=outcome)
    before, response, after = run["events"]
    team = run["fixture"]["home_team_id"]
    assert before["context"]["teams"][team]["qualification"]["value"] == "qualified"
    assert before["context"]["teams"][team]["lineup"]["status"] == "unknown"
    assert "synthetic.lineup.1" not in before["context"]["evidence"]
    assert before["decision"]["action_id"] == "lineup"
    assert response["receipt"]["observation"] == outcome
    assert response["receipt"]["cost"] == .015
    assert after["context"]["teams"][team]["lineup"]["status"] == "hypothesis"
    assert after["context"]["teams"][team]["rotation"]["status"] == "unknown"
    assert after["context"]["teams"][team]["tactical_intent"]["status"] == "unknown"
    assert len(after["context"]["evidence"]["synthetic.lineup.1"]["player_ids"]) == 11
    assert after["decision"]["action_id"] == "STOP"
    assert after["decision"]["remaining_budget"] == 0
    assert run["evidence_class"] == "SYNTHETIC_ASSUMPTIONS"


def test_hidden_lineup_alternatives_leave_the_same_preacquisition_decision():
    starters = competition_context_integration(lineup_outcome="starters")
    reserves = competition_context_integration(lineup_outcome="reserves")
    assert starters["events"][0] == reserves["events"][0]
    assert starters["events"][-1]["prediction"] != reserves["events"][-1]["prediction"]


def test_unknown_constructed_response_is_rejected():
    with pytest.raises(ValueError, match="Unknown constructed lineup"):
        competition_context_integration(lineup_outcome="unverified_actual_match")
