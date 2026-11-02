from copy import deepcopy

from football_agent.value_evaluation_contract import (
    ACTOR_PROVISIONAL,
    COUNTERFACTUAL_ELIGIBLE,
    INDEPENDENT_GOLD,
    PARTIAL_OBSERVATION,
    BLOCKED,
    summarize_evaluations,
    validate_evaluation_episode,
)
from football_agent.value_evaluation_contract import _digest


CUTOFF = "2026-09-30T10:00:00Z"
SELECTED = "2026-09-30T10:01:00Z"
RETURN_1 = "2026-09-30T10:02:00Z"
RETURN_2 = "2026-09-30T10:03:00Z"
OUTCOME = "2026-09-30T10:05:00Z"


def row(candidate_id, *, status=INDEPENDENT_GOLD, evaluator="evaluator-1",
        observed=RETURN_1, before=0.50, after=0.35):
    return {
        "candidate_id": candidate_id,
        "evaluator_id": evaluator,
        "label_status": status,
        "return_digest": _digest({"candidate_id": candidate_id, "body": "frozen evaluator return"}),
        "return_observed_at": observed,
        "decision_loss_before": before,
        "decision_loss_after": after,
        "loss_unit": "log_loss",
        "declared_cost": {"http_calls": 1},
    }


def episode(*, evaluated=None, complete=True, decision="query", selected="lineup",
            outcome=True):
    evaluated = evaluated if evaluated is not None else [row("lineup"), row("weather", observed=RETURN_2, after=0.49)]
    snapshot = {
        "match_id": "M-1",
        "known": {"home_team": "甲队", "away_team": "乙队", "ranking_gap": 0.2},
        "open_questions": ["是否存在轮换证据"],
    }
    value = {
        "schema_version": "value-evaluation-contract-v1",
        "episode_id": "episode-1",
        "target_key": "match:M-1",
        "prediction_cutoff": CUTOFF,
        "pre_action": {
            "snapshot": snapshot,
            "snapshot_digest": _digest(snapshot),
            "candidate_ids": ["lineup", "weather"],
            "candidate_generation_complete": complete,
            "selection_operation_id": "selection-1",
            "decision": decision,
            "selected_candidate_id": selected if decision == "query" else None,
            "selected_at": SELECTED,
        },
        "candidate_evaluations": evaluated,
        "evaluation_outcome": ({
            "evaluator_id": "evaluator-1",
            "label_status": INDEPENDENT_GOLD,
            "outcome_digest": _digest({"target_key": "match:M-1", "result": "held by evaluator"}),
            "available_at": OUTCOME,
        } if outcome else None),
    }
    return value


def test_complete_finite_menu_is_eligible_without_claiming_numeric_voi():
    report = validate_evaluation_episode(episode())
    assert report["status"] == COUNTERFACTUAL_ELIGIBLE
    assert report["eligibility"]["candidate_coverage"] == "complete"
    assert report["eligibility"]["counterfactual_selection_eligible"] is True
    assert report["eligibility"]["numeric_voi_training_eligible"] is False
    assert report["scope"].startswith("offline evaluator metadata")


def test_selected_only_observation_is_partial_and_keeps_counterfactual_unknown():
    report = validate_evaluation_episode(episode(evaluated=[row("lineup")]))
    assert report["status"] == PARTIAL_OBSERVATION
    assert report["eligibility"]["candidate_coverage"] == "selected_only"
    assert report["eligibility"]["counterfactual_selection_eligible"] is False


def test_open_world_candidate_generation_never_becomes_complete_by_observing_listed_rows():
    report = validate_evaluation_episode(episode(complete=False))
    assert report["status"] == PARTIAL_OBSERVATION
    assert report["eligibility"]["candidate_coverage"] == "selected_only" or report["eligibility"]["candidate_coverage"] == "complete"
    assert report["eligibility"]["counterfactual_selection_eligible"] is False


def test_actor_assessment_cannot_become_an_independent_training_label():
    actor = row("lineup", status=ACTOR_PROVISIONAL, evaluator="actor-agent")
    report = validate_evaluation_episode(episode(evaluated=[actor]))
    assert report["status"] == PARTIAL_OBSERVATION
    assert report["eligibility"]["independent_gold_labels"] is False
    assert report["eligibility"]["actor_assessment_is_gold"] is False
    assert any(item["code"] == "ACTOR_LOSS_NOT_GOLD" for item in report["reasons"])


def test_actor_outcome_cannot_make_complete_candidate_menu_counterfactual_eligible():
    value = episode()
    value["evaluation_outcome"]["label_status"] = ACTOR_PROVISIONAL
    report = validate_evaluation_episode(value)
    assert report["status"] == PARTIAL_OBSERVATION
    assert report["eligibility"]["independent_gold_labels"] is True
    assert report["eligibility"]["independent_outcome_label"] is False
    assert report["eligibility"]["counterfactual_selection_eligible"] is False
    assert any(item["code"] == "OUTCOME_NOT_INDEPENDENT_GOLD" for item in report["reasons"])


def test_actor_provisional_row_without_gold_loss_remains_partial_observation():
    actor = row("lineup", status=ACTOR_PROVISIONAL, evaluator="actor-agent")
    actor.pop("decision_loss_before")
    actor.pop("decision_loss_after")
    actor.pop("loss_unit")
    report = validate_evaluation_episode(episode(evaluated=[actor, row("weather")]))
    assert report["status"] == PARTIAL_OBSERVATION
    assert report["eligibility"]["independent_gold_labels"] is False
    assert not any(item["code"] == "INVALID_DECISION_LOSS" for item in report["reasons"])


def test_pre_action_result_field_is_rejected_even_when_digest_is_recomputed():
    value = episode()
    value["pre_action"]["snapshot"]["result"] = {"winner": "甲队"}
    value["pre_action"]["snapshot_digest"] = _digest(value["pre_action"]["snapshot"])
    report = validate_evaluation_episode(value)
    assert report["status"] == BLOCKED
    assert any(item["code"] == "PRE_ACTION_RESULT_LEAK" for item in report["reasons"])


def test_selection_digest_or_candidate_binding_tamper_is_blocked():
    value = episode()
    value["pre_action"]["snapshot_digest"] = _digest({"different": True})
    report = validate_evaluation_episode(value)
    assert report["status"] == BLOCKED
    assert any(item["code"] == "PRE_ACTION_DIGEST_MISMATCH" for item in report["reasons"])

    value = episode()
    value["candidate_evaluations"][0]["candidate_id"] = "unregistered"
    report = validate_evaluation_episode(value)
    assert report["status"] == BLOCKED
    assert any(item["code"] == "EVALUATION_CANDIDATE_OUTSIDE_MENU" for item in report["reasons"])


def test_return_or_outcome_before_selection_is_blocked():
    value = episode()
    value["candidate_evaluations"][0]["return_observed_at"] = CUTOFF
    report = validate_evaluation_episode(value)
    assert report["status"] == BLOCKED
    assert any(item["code"] == "RETURN_BEFORE_SELECTION" for item in report["reasons"])

    value = episode()
    value["evaluation_outcome"]["available_at"] = RETURN_1
    report = validate_evaluation_episode(value)
    assert report["status"] == BLOCKED
    assert any(item["code"] == "OUTCOME_BEFORE_RETURN" for item in report["reasons"])


def test_terminal_stop_can_have_counterfactual_menu_but_never_names_candidate():
    report = validate_evaluation_episode(episode(decision="stop", selected=None))
    assert report["status"] == COUNTERFACTUAL_ELIGIBLE
    assert report["selected_candidate_id"] is None
    assert report["eligibility"]["candidate_coverage"] == "complete"


def test_summary_reports_eligibility_inventory_only():
    complete = episode()
    partial = episode(evaluated=[row("lineup")], outcome=False)
    blocked = deepcopy(complete)
    blocked["pre_action"]["candidate_ids"] = ["lineup", "lineup"]
    summary = summarize_evaluations([complete, partial, blocked])
    assert summary["episode_count"] == 3
    assert summary["status_counts"] == {
        COUNTERFACTUAL_ELIGIBLE: 1,
        PARTIAL_OBSERVATION: 1,
        BLOCKED: 1,
    }
    assert summary["counterfactual_eligible_episode_ids"] == ["episode-1"]
    assert summary["numeric_voi_training_eligible"] is False
    assert summary["numeric_gain"] is None
