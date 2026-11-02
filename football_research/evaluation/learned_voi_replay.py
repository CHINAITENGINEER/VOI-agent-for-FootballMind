"""Run frozen learned VoI decisions through the existing controlled environment.

Only the supervisor holds a complete historical row. The policy sees the
initial fixture, declared tool metadata and successively revealed evidence.
No evaluator loss is produced until the acquisition trajectory has ended.
"""
from __future__ import annotations

from copy import deepcopy

from ..acquisition.data import ControlledEpisode
from ..acquisition.experiment import losses, outcome
from ..acquisition.joint_planner import UNITS, _budget
from ..acquisition.joint_voi import LearnedJointVOIPolicy
from ..acquisition.state import parse_time


def run_learned_voi_episode(row, policy: LearnedJointVOIPolicy, actions, *, limits,
                            candidate_generation_complete=False):
    """Historical disclosure, not an assertion of operational availability.

    All tools here must be backed by the controlled environment's real frozen
    feature groups. Unsupported hypothetical tools belong in contract tests,
    not in a fabricated live acquisition trace.
    """
    if row["information_mode"] != "historical_reconstruction":
        raise ValueError("This runner is only a historical reconstruction")
    limits, spent = _budget({"limits": limits, "spent": {}})
    environment = ControlledEpisode(row, actions)
    cycles = []
    for _ in range(len(actions) + 1):
        before = environment.visible_state()
        before["action_history"] = [{"action_id": aid} for aid in environment.performed_actions]
        decision = policy.plan(before, actions, {"limits": limits, "spent": spent},
                               candidate_generation_complete=candidate_generation_complete)
        cycle = {"step": len(cycles), "state_before": before, "decision": decision,
                 "state_after": None, "update": None, "charged_cost": None}
        cycles.append(cycle)
        if decision["selection"]["decision"] != "acquire":
            break
        aid = decision["selection"]["candidate_id"]
        selected = next(c for c in decision["candidates"] if c["action_id"] == aid)
        after = environment.reveal(aid)
        after["action_history"] = [{"action_id": a} for a in environment.performed_actions]
        update = policy.verify_update(decision, before, after)
        for unit in UNITS:
            spent[unit] += selected["cost"][unit]
        _budget({"limits": limits, "spent": spent})
        cycle.update(state_after=after, update=update, charged_cost=deepcopy(selected["cost"]))
    else:
        raise RuntimeError("One-shot acquisition failed to reach a terminal decision")
    return {"schema": "learned-voi-historical-trajectory-v1", "match_id": row["match_id"],
            "prediction_cutoff": row["prediction_cutoff"], "kickoff": row["kickoff"],
            "information_mode": "historical_reconstruction", "model_version": policy.model.model_version,
            "policy_version": policy.version, "policy_manifest": deepcopy(policy.manifest),
            "initial_probabilities": cycles[0]["decision"]["prediction"]["probabilities"],
            "final_probabilities": cycles[-1]["decision"]["prediction"]["probabilities"],
            "actions": list(environment.performed_actions), "cycles": cycles,
            "terminal_selection": cycles[-1]["decision"]["selection"],
            "cost": spent, "limits": limits, "actual_external_requests": 0, "actual_llm_tokens": 0,
            "acquisition_scope": "controlled_reveal_of_archived_feature_groups",
            "cost_scope": "declared_simulation_credits_and_projection_calls_not_provider_bills",
            "numeric_voi_training_eligible": False, "independent_outcome_verified": False}


def score_reconstructed_trajectory(trajectory, row):
    """Separate evaluator sidecar, explicitly NOT an independent gold label."""
    if any(str(trajectory[key]) != str(row[key]) for key in ("match_id", "prediction_cutoff", "kickoff")):
        raise ValueError("Evaluator label identity differs from the trajectory")
    if row["information_mode"] != "historical_reconstruction" or row.get("score_period") != "regulation":
        raise ValueError("Explicit reconstructed regulation-time result required")
    if parse_time(row["result_available_at"]) <= parse_time(row["kickoff"]):
        raise ValueError("Result availability cannot precede the match")
    for key in ("home_goals", "away_goals"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("A nonnegative integer result is required")
    label = outcome(row)
    initial = losses(trajectory["initial_probabilities"], label)
    final = losses(trajectory["final_probabilities"], label)
    steps = []
    for cycle in trajectory["cycles"]:
        if cycle["update"] is None:
            continue
        update = cycle["update"]
        before = losses(update["prediction_before"], label)
        after = losses(update["prediction_after"], label)
        steps.append({"action_id": update["action_id"], "decision_digest": update["decision_digest"],
                      "before_loss": before, "after_loss": after,
                      "realized_log_loss_reduction": before["log_loss"]-after["log_loss"]})
    return {"schema": "learned-voi-historical-evaluator-v1", "match_id": row["match_id"],
            "prediction_cutoff": row["prediction_cutoff"], "result_available_at": row["result_available_at"],
            "outcome_index": label, "before_loss": initial, "after_loss": final, "steps": steps,
            "label_scope": "same_provider_reconstructed_regulation_result_not_independent_gold",
            "independent_outcome_verified": False, "numeric_voi_training_eligible": False}
