from copy import deepcopy

import pytest

from football_research.acquisition.data import ControlledEpisode
from football_research.acquisition.expected_voi import select_candidate, unknown_candidate
from football_research.acquisition.joint_planner import JointRiskPlanner
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.state import GROUP_FIELDS
from football_research.evaluation.learned_voi_replay import run_learned_voi_episode, score_reconstructed_trajectory
from test_joint_returns import ACTIONS, fitted, state


BUDGET = {"limits": {"money": .1, "requests": 3}, "spent": {}}
UNKNOWN = {"action_id": "lineup_context", "group_ids": ["lineup"], "available": True,
           "cost_status": "estimated", "cost": {"money": .01, "requests": 1}}


def actual_return(before, decision, rows):
    row = {**rows[4], "match_id": before["match_id"],
           "prediction_cutoff": before["prediction_cutoff"], "kickoff": before["base"]["kickoff"],
           "features_available_at": before["prediction_cutoff"], "feature_version": "fixture"}
    env = ControlledEpisode(row, ACTIONS)
    aid = decision["selection"]["candidate_id"]
    after = env.reveal(aid)
    after["action_history"] = [{"action_id": aid}]
    return after


def test_fitted_branches_equal_existing_one_step_planner_and_are_coherent(fitted):
    model, _, _ = fitted
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": .5})
    result = policy.plan(state(), ACTIONS, BUDGET)
    native = JointRiskPlanner(model, lambda_cost={"money": .5}).plan(state(), ACTIONS, BUDGET)
    by_id = {c["action_id"]: c for c in native["candidates"]}
    for c in result["candidates"]:
        assert c["value"]["status"] == "COMPUTED"
        assert c["value"]["coherence_error"] < 1e-12
        assert c["value"]["gross_evsi"] == pytest.approx(by_id[c["action_id"]]["immediate_evsi"])
        assert c["value"]["net_voi"] == pytest.approx(by_id[c["action_id"]]["planning_net_gain"])
        assert c["value"]["response_model_version"] == model.model_version
        assert all(set(b) == {"branch_id", "probability", "posterior"} for b in c["branches"])
    assert result["selection"]["candidate_id"] == native["action_id"]


def test_brier_uses_same_learned_returns_but_correct_risk_units(fitted):
    model, _, _ = fitted
    result = LearnedJointVOIPolicy(model, lambda_cost={"money": .5}, loss="brier").plan(state(), ACTIONS, BUDGET)
    prior = result["prediction"]["probabilities"]
    for c in result["candidates"]:
        expected = sum(b["probability"] * (1-sum(p*p for p in b["posterior"])) for b in c["branches"])
        assert c["value"]["gross_evsi"] == pytest.approx(1-sum(p*p for p in prior)-expected)


def test_unknown_tool_does_not_block_a_positive_modeled_query_or_gain_a_fake_value(fitted):
    model, _, _ = fitted
    plan = LearnedJointVOIPolicy(model, lambda_cost={"money": 0}).plan(state(), ACTIONS+[UNKNOWN], BUDGET)
    assert plan["selection"]["decision"] == "acquire"
    assert plan["selection"]["candidate_id"] in {a["action_id"] for a in ACTIONS}
    row = next(c for c in plan["candidates"] if c["action_id"] == "lineup_context")
    assert row["value"]["status"] == "INELIGIBLE"
    assert row["value"]["net_voi"] is None and row["branches"] is None


@pytest.mark.parametrize("complete", [True, False])
def test_unknown_feasible_tool_prevents_stop_even_with_complete_declared_menu(fitted, complete):
    model, _, _ = fitted
    plan = LearnedJointVOIPolicy(model, lambda_cost={"money": 100}).plan(
        state(), ACTIONS+[UNKNOWN], BUDGET, candidate_generation_complete=complete)
    assert plan["selection"]["decision"] == "defer"
    assert plan["selection"]["stop_certificate"] is None


def test_stop_requires_complete_menu_and_is_only_one_step(fitted):
    model, _, _ = fitted
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 100})
    incomplete = policy.plan(state(), ACTIONS, BUDGET)
    assert incomplete["selection"]["reason"] == "candidate_generation_incomplete"
    complete = policy.plan(state(), ACTIONS, BUDGET, candidate_generation_complete=True)
    assert complete["selection"]["decision"] == "stop"
    certificate = complete["selection"]["stop_certificate"]
    assert certificate["does_not_bound_multi_step_or_undiscovered_information"] is True


def test_legacy_incomplete_option_cannot_create_false_stop_certificate():
    unknown = unknown_candidate("unknown", [.4, .3, .3], reason="no model")
    result = select_candidate([unknown], defer_when_incomplete=False)
    assert result["decision"] == "defer" and result["stop_certificate"] is None


def test_zero_loss_price_never_bypasses_physical_request_limit(fitted):
    model, _, _ = fitted
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 0, "requests": 0})
    plan = policy.plan(state(), ACTIONS, {"limits": {"money": 10, "requests": 0}, "spent": {}},
                       candidate_generation_complete=True)
    assert plan["selection"]["decision"] == "halt"
    assert plan["selection"]["reason"] == "physical_budget_exhausted"
    assert plan["selection"]["stop_certificate"] is None


def test_unknown_identity_or_insufficient_coverage_withholds_numeric_values(fitted):
    model, _, _ = fitted
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 0})
    outside = state()
    outside["base"]["competition_id"] = "unseen_national_team_competition"
    plan = policy.plan(outside, ACTIONS, BUDGET)
    assert plan["selection"]["decision"] == "defer"
    assert all(c["value"]["net_voi"] is None for c in plan["candidates"])
    strict = LearnedJointVOIPolicy(model, lambda_cost={"money": 0}, min_effective_matches=1000)
    plan = strict.plan(state(), ACTIONS, BUDGET)
    assert all(c["reason"] == "insufficient_conditional_return_support" for c in plan["candidates"])


def test_pseudocounts_are_not_empirical_support_for_missing_context(fitted):
    model, _, _ = fitted
    missing = state()
    missing["groups"]["elo"] = {"status": "source_missing"}
    missing["groups"]["recent_form"] = {"status": "source_missing"}
    support = model.observation_support(missing)
    assert support["matching_training_matches"] == 0
    assert support["kernel_effective_matches"] == 0
    assert support["smoothing_counts_as_support"] is False
    plan = LearnedJointVOIPolicy(model, lambda_cost={"money": 0}).plan(missing, ACTIONS, BUDGET)
    xg = next(c for c in plan["candidates"] if c["action_id"] == "xg_view")
    assert xg["value"]["net_voi"] is None


def test_unpaid_values_and_evaluator_fields_cannot_affect_trace(fitted):
    class Poison:
        def __deepcopy__(self, memo):
            raise AssertionError("Hidden data copied")
        def __str__(self):
            raise AssertionError("Hidden data inspected")
    model, _, _ = fitted
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": .5})
    dirty = state()
    dirty["label"] = Poison()
    for group in dirty["groups"].values():
        group["values"] = Poison()
    assert policy.plan(dirty, ACTIONS, BUDGET) == policy.plan(state(), ACTIONS, BUDGET)


def test_actual_update_matches_pre_query_branch_and_replans_without_reacquisition(fitted):
    model, rows, _ = fitted
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 0})
    before = state()
    decision = policy.plan(before, ACTIONS, BUDGET)
    after = actual_return(before, decision, rows)
    update = policy.verify_update(decision, before, after)
    assert update["prediction_after"] == model.predict_partial(after)["probabilities"]
    assert update["prediction_before"] != update["prediction_after"]
    assert update["realized_loss_reduction"] is None
    replanned = policy.plan(after, ACTIONS, BUDGET)
    acquired = next(c for c in replanned["candidates"] if c["action_id"] == update["action_id"])
    assert acquired["feasible"] is False and acquired["reason"] == "already_attempted"
    assert replanned["state_digest"] == update["after_state_digest"]


@pytest.mark.parametrize("mutation", ["late", "fixture", "unrelated", "history", "decision", "hypothesis", "failure"])
def test_invalid_or_unadmitted_returns_do_not_get_after_probabilities(fitted, mutation):
    model, rows, _ = fitted
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 0})
    before = state()
    decision = policy.plan(before, ACTIONS, BUDGET)
    after = actual_return(before, decision, rows)
    selected = next(c for c in decision["candidates"] if c["action_id"] == decision["selection"]["candidate_id"])
    if mutation == "late":
        after["groups"][selected["group_ids"][0]]["available_at"] = "2025-05-01T01:00:00Z"
    elif mutation == "fixture":
        after["match_id"] = "different-match"
    elif mutation == "unrelated":
        other = next(g for g in GROUP_FIELDS if g not in selected["group_ids"])
        after["groups"][other] = {"status": "source_missing"}
    elif mutation == "history":
        after["action_history"] = []
    elif mutation == "decision":
        decision["candidates"][0]["cost"]["money"] += 1
    elif mutation == "hypothesis":
        after = model.enumerate_action(before, selected)["branches"][0]["next_state"]
    else:
        after["action_history"][0]["status"] = "PROVIDER_ERROR"
    with pytest.raises(ValueError):
        policy.verify_update(decision, before, after)


@pytest.mark.parametrize("mutation", ["group", "duplicate", "train_match", "future_model"])
def test_model_menu_and_time_boundaries(fitted, mutation):
    model, rows, _ = fitted
    public, actions = state(), deepcopy(ACTIONS)
    if mutation == "group":
        actions[0]["group_ids"] = ["xg"]
    elif mutation == "duplicate":
        actions += [actions[0]]
    elif mutation == "train_match":
        public["match_id"] = rows[0]["match_id"]
    else:
        public["prediction_cutoff"] = "2025-03-01T00:00:00Z"
    with pytest.raises(ValueError):
        LearnedJointVOIPolicy(model, lambda_cost={"money": 0}).plan(public, actions, BUDGET)


def replay_row(rows):
    return {**rows[4], "match_id": "future", "prediction_cutoff": "2025-05-01T00:00:00Z",
            "kickoff": "2025-05-02T00:00:00Z", "features_available_at": "2025-05-01T00:00:00Z",
            "result_available_at": "2025-05-02T03:00:00Z", "feature_version": "fixture"}


def test_replay_adapts_to_actual_returns_without_accessing_target_label(fitted):
    model, rows, _ = fitted
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 0})
    row = replay_row(rows)
    changed = {**row, "home_goals": 9, "away_goals": 0}
    trace = run_learned_voi_episode(row, policy, ACTIONS, limits=BUDGET["limits"])
    assert trace == run_learned_voi_episode(changed, policy, ACTIONS, limits=BUDGET["limits"])
    assert trace["actions"] and trace["cycles"][0]["update"] is not None
    assert trace["cost"]["requests"] == len(trace["actions"])
    assert trace["terminal_selection"]["stop_certificate"] is None
    for previous, current in zip(trace["cycles"], trace["cycles"][1:]):
        assert previous["update"]["after_state_digest"] == current["decision"]["state_digest"]
        assert previous["update"]["prediction_after"] == current["decision"]["prediction"]["probabilities"]
    sidecar = score_reconstructed_trajectory(trace, row)
    altered_sidecar = score_reconstructed_trajectory(trace, changed)
    assert sidecar["outcome_index"] != altered_sidecar["outcome_index"]
    assert sidecar["after_loss"] != altered_sidecar["after_loss"]
    assert sidecar["independent_outcome_verified"] is False
    assert "outcome_index" not in str(trace)


def test_replay_rejects_live_scope_and_misbound_evaluator(fitted):
    model, rows, _ = fitted
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 0})
    row = replay_row(rows)
    with pytest.raises(ValueError, match="historical"):
        run_learned_voi_episode({**row, "information_mode": "operational"}, policy, ACTIONS, limits=BUDGET["limits"])
    trace = run_learned_voi_episode(row, policy, ACTIONS, limits=BUDGET["limits"])
    for patch in ({"match_id": "another"}, {"score_period": "extra_time"},
                  {"home_goals": None}, {"result_available_at": row["prediction_cutoff"]}):
        with pytest.raises(ValueError):
            score_reconstructed_trajectory(trace, {**row, **patch})
