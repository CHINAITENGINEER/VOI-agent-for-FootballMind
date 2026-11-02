import pytest

from football_agent.decision_value import synthetic_scenarios
from football_agent.policy_audit import audit_comparison, audit_episode
from football_agent.prequery_state import PreQueryStateAnnotation, PreQueryStateRegistry
from football_agent.policy_replay import POLICY_AGENT, POLICY_BASE, compare_policies, replay_policy
from football_agent.state_query_contract import KNOWN, StateHypothesis, StateQueryBoard, StateQueryCandidate


TARGET = "match-1"
CUTOFF = "2026-09-27T10:00:00Z"


def board():
    registry = PreQueryStateRegistry([
        PreQueryStateAnnotation(
            annotation_id="qualification-1", target_key=TARGET, state_variable="qualification_status",
            value="already_qualified", basis="pre_query_external_annotation",
            provenance_kind="official_pre_match_rule", source_ids=("official",),
            available_at="2026-09-27T09:00:00Z", observed_at="2026-09-27T09:01:00Z",
            prediction_cutoff=CUTOFF, scope="group_stage",
        ),
    ])
    return StateQueryBoard(
        target_key=TARGET, prediction_cutoff=CUTOFF, registry=registry,
        hypotheses=[
            StateHypothesis("qualification", TARGET, "qualification_status",
                            ("already_qualified", "must_win"), status=KNOWN,
                            known_value="already_qualified", annotation_id="qualification-1"),
            StateHypothesis("rotation", TARGET, "rotation_status", ("starters", "reserves")),
            StateHypothesis("reserve", TARGET, "reserve_strength", ("strong", "weak")),
        ],
        candidates=[
            StateQueryCandidate("qualification-refresh", TARGET, CUTOFF, "qualification_status",
                                "standings", "read", .01),
            StateQueryCandidate("lineup", TARGET, CUTOFF, "rotation_status",
                                "lineup_feed", "read", .015, action_id="lineup"),
            StateQueryCandidate("reserve-quality", TARGET, CUTOFF, "reserve_strength",
                                "squad_feed", "read", .02, action_id="reserve_quality"),
        ],
    )


def episode(policy=POLICY_AGENT):
    model = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    return replay_policy(model, policy, seed=7, budget=.08, horizon=1)


def test_agent_view_keeps_private_known_state_as_residual_unknown():
    row = audit_episode(board(), episode(), view="agent")
    assert "qualification-refresh" in row["actionable_candidate_ids"]
    assert "qualification-refresh" in row["residual_unknown_candidate_ids"]
    assert "qualification_status" in row["residual_unknown_state_variables"]
    assert row["hidden_result_included"] is False


def test_evaluator_view_excludes_private_known_state_from_residual_unknowns():
    row = audit_episode(board(), episode(), view="evaluator")
    assert "qualification-refresh" not in row["actionable_candidate_ids"]
    assert "qualification-refresh" not in row["residual_unknown_candidate_ids"]
    assert row["status_by_candidate"]["qualification-refresh"] == "state_not_actionable"


def test_queried_actions_are_mapped_back_to_independent_state_variables():
    row = audit_episode(board(), episode(), view="agent")
    assert row["queried_candidate_ids"] == ["lineup"]
    assert row["queried_action_ids"] == ["lineup"]
    assert row["queried_state_variables"] == ["rotation_status"]
    assert row["status_by_candidate"]["lineup"] == "queried"
    assert row["state_resolution_status"]["rotation_status"] == "queried_unresolved"
    assert "rotation_status" in row["unresolved_state_variables"]


def test_explicit_resolution_is_separate_from_merely_calling_a_tool():
    raw = episode()
    raw["resolved_state_variables"] = ["rotation_status"]
    row = audit_episode(board(), raw, view="agent")
    assert row["state_resolution_status"]["rotation_status"] == "resolved"
    assert "rotation_status" not in row["unresolved_state_variables"]


def test_late_and_failed_routes_remain_explicit_statuses():
    row = audit_episode(board(), episode(POLICY_BASE), view="agent",
                        candidate_availability={"lineup": "late", "reserve-quality": "failed"})
    assert row["status_by_candidate"]["lineup"] == "late"
    assert row["status_by_candidate"]["reserve-quality"] == "failed"
    assert row["residual_unknown_candidate_ids"] == ["qualification-refresh"]
    assert row["state_resolution_status"]["rotation_status"] == "unresolved_late"
    assert row["state_resolution_status"]["reserve_strength"] == "unresolved_failed"


def test_duplicate_or_unknown_action_contracts_are_rejected():
    raw = episode()
    raw["trace"][0]["selected_action"] = "missing"
    with pytest.raises(ValueError, match="absent from the state-query board"):
        audit_episode(board(), raw)


def test_comparison_audit_preserves_policy_separation_and_counts_residuals():
    model = synthetic_scenarios()["qualified_rotation_unknown"]["model"]
    comparison = compare_policies(model, seeds=range(3), budget=.08, horizon=1)
    result = audit_comparison(board(), comparison, view="agent")
    assert result["schema_version"] == "policy-semantic-audit-v1"
    assert result["hidden_result_included"] is False
    assert result["summaries"][POLICY_BASE]["mean_query_count"] == 0
    assert result["summaries"][POLICY_AGENT]["mean_residual_unknown_count"] >= 1
    assert result["summaries"][POLICY_AGENT]["availability_status_counts"]["residual_unknown"] >= 1


@pytest.mark.parametrize("bad_status", ["unknown", "unavailable", "timeout"])
def test_candidate_availability_is_a_closed_evaluator_vocabulary(bad_status):
    with pytest.raises(ValueError, match="Unsupported candidate availability"):
        audit_episode(board(), episode(POLICY_BASE), candidate_availability={"lineup": bad_status})
