"""Mechanism controls for a frozen semantic-scenario reconstruction.

This module replays one already classified return panel against the exact same
frozen :class:`SemanticScenarioModel`.  It is deliberately a *control* and
mechanism exercise: it does not discover pages, call a classifier, read hidden
body text, use a match result, or integrate fabricated counterfactual returns.

The public entry point is :func:`compare_controls`.  ``model_manifest`` is the
manifest saved by ``SemanticScenarioModel.manifest()``; ``candidate_bindings``
is the public action menu (a mapping or a list of objects containing
``action_id``); and ``panel`` maps each action ID to its already validated
semantic observation.  A selected action without a panel return terminates
that arm with ``status='incomplete'`` rather than inventing a return.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math

from .semantic_scenario_model import SemanticScenarioModel


VERSION = "semantic-scenario-controls-v1"
POLICIES = ("adaptive_voi", "fixed_order", "no_belief_update", "no_acquisition")
OUTCOMES = ("supports", "refutes", "no_answer", "unknown")
TOLERANCE = 1e-12


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _normalise_bindings(bindings):
    """Return a detached action_id -> public binding mapping."""
    if isinstance(bindings, Mapping):
        result = {}
        for action_id, binding in bindings.items():
            if not isinstance(action_id, str) or not action_id:
                raise ValueError("candidate binding IDs must be nonempty strings")
            if not isinstance(binding, Mapping):
                raise ValueError("candidate bindings must contain objects")
            row = deepcopy(dict(binding))
            row.setdefault("action_id", action_id)
            if row["action_id"] != action_id:
                raise ValueError("candidate binding action_id disagrees with its mapping key")
            result[action_id] = row
        return result
    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence):
        raise ValueError("candidate_bindings must be a mapping or a sequence")
    result = {}
    for binding in bindings:
        if not isinstance(binding, Mapping) or not isinstance(binding.get("action_id"), str):
            raise ValueError("candidate bindings must contain action_id objects")
        action_id = binding["action_id"]
        if action_id in result:
            raise ValueError("candidate bindings contain duplicate action_id")
        result[action_id] = deepcopy(dict(binding))
    return result


def _normalise_panel(panel):
    """Normalise a return panel while retaining only already supplied returns.

    A panel may be an action mapping, a list of observation records, or a
    wrapper containing ``observations``/``returns``.  A one-item sequence under
    an action ID is accepted for callers that persist panel cells as arrays;
    multiple alternatives are rejected because choosing one would fabricate a
    branch.
    """
    if isinstance(panel, Mapping) and ("observations" in panel or "returns" in panel):
        panel = panel.get("observations", panel.get("returns"))
    rows = {}
    if isinstance(panel, Mapping):
        items = panel.items()
        for action_id, value in items:
            if isinstance(value, Mapping) and "observation" in value and isinstance(value["observation"], Mapping):
                value = value["observation"]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) == 0:
                    continue
                if len(value) != 1:
                    raise ValueError("panel action cells may contain at most one actual observation")
                value = value[0]
            if not isinstance(value, Mapping):
                raise ValueError("panel observations must be objects")
            observation = deepcopy(dict(value))
            observation.setdefault("action_id", action_id)
            if observation.get("action_id") != action_id:
                raise ValueError("panel observation action_id disagrees with its mapping key")
            if action_id in rows:
                raise ValueError("panel contains duplicate action_id")
            rows[action_id] = observation
        return rows
    if isinstance(panel, (str, bytes)) or not isinstance(panel, Sequence):
        raise ValueError("panel must be an action mapping or observation sequence")
    for value in panel:
        if not isinstance(value, Mapping) or not isinstance(value.get("action_id"), str):
            raise ValueError("panel observations must contain action_id objects")
        action_id = value["action_id"]
        if action_id in rows:
            raise ValueError("panel contains duplicate action_id")
        rows[action_id] = deepcopy(dict(value))
    return rows


def _validate_inputs(model, bindings, panel, fixed_order):
    action_ids = set(model.actions)
    unknown_bindings = set(bindings) - action_ids
    if unknown_bindings:
        raise ValueError("candidate_bindings contains unknown action IDs")
    unknown_panel = set(panel) - action_ids
    if unknown_panel:
        raise ValueError("panel contains unknown action IDs")
    for action_id, binding in bindings.items():
        if "request_cost" in binding and binding["request_cost"] != model.action_request_costs[action_id]:
            raise ValueError("candidate binding request_cost disagrees with frozen model manifest")
    if isinstance(fixed_order, (str, bytes)) or not isinstance(fixed_order, Sequence):
        raise ValueError("fixed_order must list every action exactly once")
    order = tuple(fixed_order)
    if len(order) != len(action_ids) or set(order) != action_ids:
        raise ValueError("fixed_order must contain every model action exactly once")
    if any(not isinstance(item, str) for item in order):
        raise ValueError("fixed_order IDs must be strings")
    return order


def _entropy(probabilities):
    return -math.fsum(p * math.log(p) for p in probabilities if p > 0)


def _compact_candidate(row):
    """Expose values needed for review without adding empirical gain metrics."""
    result = {key: deepcopy(row.get(key)) for key in (
        "action_id", "question", "observation_group", "request_cost", "cost", "cost_unit",
        "feasible", "reason", "gross", "net", "lower", "upper", "gross_lower", "gross_upper")}
    result["nominal_conditional_value"] = row.get("gross")
    result["nominal_conditional_net_value"] = row.get("net")
    result["robust_conditional_value"] = row.get("lower")
    result["robust_conditional_net_value"] = row.get("lower")
    return result


def _blocked_for_history(model, history):
    blocked = set()
    for observation in history:
        action_id, outcome = observation["action_id"], observation["outcome"]
        group = model.actions[action_id]["observation_group"]
        if outcome in ("supports", "refutes") or model.observation_group_policy == "first_attempt":
            blocked.add(group)
    return blocked


def _validate_observation(action_id, observation):
    if not isinstance(observation, Mapping):
        raise ValueError("panel observations must be objects")
    outcome = observation.get("outcome")
    if observation.get("action_id") != action_id:
        raise ValueError("panel observation action_id differs from selected action")
    if outcome not in OUTCOMES:
        raise ValueError("panel observation outcome is not a legal semantic return")
    return deepcopy(dict(observation))


def _state_summary(model, evaluated, policy, *, posterior_source):
    prediction = list(evaluated["prediction_nominal"])
    posterior = deepcopy(evaluated["posterior_nominal"])
    rows = [_compact_candidate(row) for row in evaluated["candidates"]]
    selected = evaluated.get("selected_action_id")
    selected_row = next((row for row in rows if row["action_id"] == selected), None)
    return {
        "prediction_before": prediction,
        "posterior_before": posterior,
        "posterior_source": posterior_source,
        "belief_update_applied": policy != "no_belief_update",
        "candidates": rows,
        "selected_action_id": selected,
        "selected_nominal_conditional_value": None if selected_row is None else selected_row["nominal_conditional_value"],
        "selected_robust_conditional_value": None if selected_row is None else selected_row["robust_conditional_value"],
        "selected_nominal_conditional_net_value": None if selected_row is None else selected_row["nominal_conditional_net_value"],
        "selected_robust_conditional_net_value": None if selected_row is None else selected_row["robust_conditional_net_value"],
    }


def _empty_result(model, policy, request_budget, *, reason, order):
    initial = model.evaluate([], request_budget)
    prediction = list(initial["prediction_nominal"])
    posterior = deepcopy(initial["posterior_nominal"])
    return {
        "policy": policy,
        "policy_description": {
            "adaptive_voi": "每步按当前后验重估一阶 robust VoI，只有正的 robust 净价值才继续",
            "fixed_order": "按公开菜单顺序读取预算内且未重复的事实，每次读取后进行 Bayes 更新",
            "no_belief_update": "ablation：冻结初始后验及动作排名；实际读取仍计费，不声称更新真实后验",
            "no_acquisition": "不读取任何事实，作为零采集对照",
        }[policy],
        "status": "completed",
        "stop_reason": reason,
        "selected_actions": [],
        "trace": [],
        "actual_read_entries": 0,
        "read_entry_count": 0,
        "requests_spent": 0,
        "remaining_requests": request_budget,
        "prediction_final": prediction,
        "posterior_final": posterior,
        "final_prediction": prediction,
        "final_posterior": posterior,
        "final_prediction_entropy_nats": _entropy(prediction),
        "final_posterior_entropy_nats": _entropy(tuple(posterior.values())),
        "terminal_entropy_is_accuracy": False,
        "terminal_entropy_interpretation": "终局预测/后验熵是模型不确定性刻画，不是赛果准确率或实测收益。",
        "used_real_match_result": False,
        "empirical_gain": None,
        "fixed_order": list(order),
    }


def _run_policy(model, bindings, panel, request_budget, order, policy):
    if policy == "no_acquisition":
        return _empty_result(model, policy, request_budget, reason="no_acquisition", order=order)
    history = []
    selected_actions = []
    trace = []
    spent = 0
    blocked = set()
    initial = model.evaluate([], request_budget)
    initial_scores = {
        row["action_id"]: row.get("lower")
        for row in initial["candidates"]
        if row.get("feasible") and row.get("lower") is not None
    }
    # A stable initial ranking is the defining ablation.  Tie-break by public
    # menu order, then action ID so JSON object order cannot alter a run.
    rank_position = {action_id: index for index, action_id in enumerate(order)}
    initial_rank = tuple(sorted(initial_scores, key=lambda a: (-initial_scores[a], rank_position[a], a)))
    status, stop_reason = "completed", None
    while spent < request_budget:
        remaining = request_budget - spent
        if policy == "no_belief_update":
            evaluated = model.evaluate([], remaining,
                                       attempted_action_ids=selected_actions,
                                       blocked_groups=blocked)
            action = next((a for a in initial_rank
                           if next((r for r in evaluated["candidates"] if r["action_id"] == a), {}).get("feasible")), None)
            if action is None:
                stop_reason = "no_affordable_unattempted_action"
            posterior_source = "frozen_initial_posterior_ablation"
        elif policy == "fixed_order":
            evaluated = model.evaluate(history, remaining)
            feasible = {r["action_id"] for r in evaluated["candidates"] if r.get("feasible")}
            action = next((a for a in order if a in feasible), None)
            if action is None:
                stop_reason = "no_affordable_unattempted_action"
            posterior_source = "bayes_updated"
        else:
            evaluated = model.evaluate(history, remaining)
            action = evaluated.get("selected_action_id")
            if action is None:
                stop_reason = evaluated.get("stop_reason") or "no_positive_declared_lower_net_value"
            posterior_source = "bayes_updated"
        before = _state_summary(model, evaluated, policy, posterior_source=posterior_source)
        event = {"step": len(trace), "policy": policy, "requests_spent_before": spent,
                 "remaining_requests_before": remaining, **before}
        if action is None:
            event.update({"selected_action_id": None, "stop_reason": stop_reason})
            trace.append(event)
            break
        binding = bindings.get(action)
        if binding is None:
            status, stop_reason = "incomplete", "missing_candidate_binding"
            event.update({"selected_action_id": action, "stop_reason": stop_reason})
            trace.append(event)
            break
        observation = panel.get(action)
        if observation is None:
            status, stop_reason = "incomplete", "missing_panel_return"
            event.update({"selected_action_id": action, "stop_reason": stop_reason,
                          "missing_panel_action_id": action,
                          "candidate_binding": deepcopy(binding)})
            trace.append(event)
            break
        observation = _validate_observation(action, observation)
        request_cost = model.action_request_costs[action]
        if request_cost > remaining:
            status, stop_reason = "completed", "insufficient_request_budget"
            event.update({"selected_action_id": None, "stop_reason": stop_reason})
            trace.append(event)
            break
        spent += request_cost
        selected_actions.append(action)
        history.append(observation)
        group = model.actions[action]["observation_group"]
        if observation["outcome"] in ("supports", "refutes") or model.observation_group_policy == "first_attempt":
            blocked.add(group)
        event.update({"selected_action_id": action, "candidate_binding": deepcopy(binding),
                      "observation": observation, "returned_observation": deepcopy(observation),
                      "request_cost": request_cost, "cost": model.request_loss_cost * request_cost,
                      "requests_spent_after": spent, "remaining_requests_after": request_budget - spent,
                      "actual_read_entry": True})
        # The no-belief-update arm intentionally keeps its displayed state at
        # the initial posterior/prediction.  Other arms replay the actual
        # return through the finite Bayes model for the next step.
        if policy == "no_belief_update":
            after = model.evaluate([], request_budget - spent,
                                   attempted_action_ids=selected_actions,
                                   blocked_groups=blocked)
            event["prediction_after"] = list(after["prediction_nominal"])
            event["posterior_after"] = deepcopy(after["posterior_nominal"])
            event["posterior_after_source"] = "frozen_initial_posterior_ablation"
            event["belief_update_applied"] = False
        else:
            after = model.evaluate(history, request_budget - spent)
            event["prediction_after"] = list(after["prediction_nominal"])
            event["posterior_after"] = deepcopy(after["posterior_nominal"])
            event["posterior_after_source"] = "bayes_updated"
            event["belief_update_applied"] = True
        trace.append(event)
        if spent >= request_budget:
            stop_reason = "request_budget_exhausted"
            break
    if stop_reason is None:
        stop_reason = "request_budget_exhausted" if spent >= request_budget else "no_affordable_unattempted_action"
    if policy == "no_belief_update":
        terminal = model.evaluate([], request_budget - spent,
                                  attempted_action_ids=selected_actions, blocked_groups=blocked)
        final_prediction = list(terminal["prediction_nominal"])
        final_posterior = deepcopy(terminal["posterior_nominal"])
    else:
        terminal = model.evaluate(history, request_budget - spent)
        final_prediction = list(terminal["prediction_nominal"])
        final_posterior = deepcopy(terminal["posterior_nominal"])
    return {
        "policy": policy,
        "policy_description": {
            "adaptive_voi": "每步按当前后验重估一阶 robust VoI，只有正的 robust 净价值才继续",
            "fixed_order": "按公开菜单顺序读取预算内且未重复的事实，每次读取后进行 Bayes 更新",
            "no_belief_update": "ablation：冻结初始后验及动作排名；实际读取仍计费，不声称更新真实后验",
            "no_acquisition": "不读取任何事实，作为零采集对照",
        }[policy],
        "status": status, "stop_reason": stop_reason,
        "selected_actions": selected_actions, "trace": trace,
        "actual_read_entries": len(selected_actions), "read_entry_count": len(selected_actions),
        "requests_spent": spent, "remaining_requests": request_budget - spent,
        "prediction_final": final_prediction, "posterior_final": final_posterior,
        "final_prediction": final_prediction, "final_posterior": final_posterior,
        "final_prediction_entropy_nats": _entropy(final_prediction),
        "final_posterior_entropy_nats": _entropy(tuple(final_posterior.values())),
        "terminal_entropy_is_accuracy": False,
        "terminal_entropy_interpretation": "终局预测/后验熵是模型不确定性刻画，不是赛果准确率或实测收益。",
        "used_real_match_result": False, "empirical_gain": None,
        "fixed_order": list(order), "initial_action_ranking": list(initial_rank),
    }


def compare_controls(model_manifest, candidate_bindings, panel, request_budget, fixed_order=None):
    """Compare one-step controls on the same actual semantic return panel.

    Returns ``results``/``strategies`` for adaptive VoI, fixed public order,
    frozen-belief ablation and no acquisition.  Values are conditional model
    values (nominal gross/net and robust lower/upper envelopes), never measured
    gains.  ``status='incomplete'`` is used when a selected action has no
    supplied panel return; no unseen or counterfactual return is generated.
    """
    if isinstance(request_budget, bool) or not isinstance(request_budget, int) or request_budget < 0:
        raise ValueError("request_budget must be a nonnegative integer")
    model = SemanticScenarioModel.from_manifest(deepcopy(model_manifest))
    bindings = _normalise_bindings(candidate_bindings)
    panel_rows = _normalise_panel(panel)
    order = tuple(model.actions) if fixed_order is None else tuple(fixed_order)
    order = _validate_inputs(model, bindings, panel_rows, order)
    results = {policy: _run_policy(model, bindings, panel_rows, request_budget, order, policy)
               for policy in POLICIES}
    choices = {policy: tuple(results[policy]["selected_actions"]) for policy in POLICIES}
    pairwise = {}
    for i, left in enumerate(POLICIES):
        for right in POLICIES[i + 1:]:
            pairwise[f"{left}__vs__{right}"] = choices[left] != choices[right]
    return {
        "version": VERSION,
        "policy_order": list(POLICIES),
        "model_id": model.model_id,
        "model_manifest_version": model_manifest.get("version"),
        "request_budget": request_budget,
        "request_cost_unit": "uniform_HTTP_request_units",
        "fixed_order": list(order),
        "candidate_binding_ids": list(bindings),
        "panel_action_ids": list(panel_rows),
        "results": results,
        "strategies": results,
        "different_action_choices": len(set(choices.values())) > 1,
        "action_choices_differ": len(set(choices.values())) > 1,
        "pairwise_action_choices_differ": pairwise,
        "future_returns_used": 0,
        "counterfactual_returns_used": 0,
        "real_match_result_used": False,
        "empirical_gain": None,
        "outcome_accuracy": None,
        "terminal_entropy_is_accuracy": False,
        "scope": "same_frozen_manifest_and_actual_panel_one_step_control_replay",
        "status": "incomplete" if any(item["status"] == "incomplete" for item in results.values()) else "completed",
        "runtime_safe": False,
    }


# Readable aliases for callers that prefer a run/comparison verb.
run_controls = compare_controls
compare_semantic_controls = compare_controls


__all__ = ["VERSION", "POLICIES", "compare_controls", "compare_semantic_controls", "run_controls"]
