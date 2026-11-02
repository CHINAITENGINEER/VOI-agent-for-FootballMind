"""Matched-policy replay for active information acquisition.

The replay harness compares information policies under the same finite model,
budget, cut-off, latent state stream, and eventual outcome stream.  It is an
evaluator artifact: the sampled latent hypothesis and match outcome stay in an
``evaluator_only`` section and never enter a policy decision or observation
trace.

The four reference policies are deliberately simple:

``agent_autonomous_finite_model``
    Re-evaluates the declared finite value model after every observation and
    may stop early.
``fixed_priority``
    Calls the first affordable action in a predeclared order, at most once.
``full_budget``
    Calls every affordable action in that same order until the budget or menu
    is exhausted.
``base_stop``
    Uses the base prediction and acquires no extra information.

These are mechanism controls, not a claim that a constructed finite model is
already a trained football system.  The comparison isolates whether value
aware selection and stopping reduce risk per unit acquisition cost.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

from .decision_value import EVIDENCE_CLASS, FiniteDecisionModel, _draw


VERSION = "matched-policy-replay-v1"
POLICY_AGENT = "agent_autonomous_finite_model"
POLICY_FIXED = "fixed_priority"
POLICY_FULL = "full_budget"
POLICY_BASE = "base_stop"
POLICY_IDS = (POLICY_AGENT, POLICY_FIXED, POLICY_FULL, POLICY_BASE)
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PolicySpec:
    """A reproducible policy control with a declared action budget."""

    policy_id: str
    description: str
    max_actions: int | None
    value_aware: bool


def default_policy_specs(tool_count: int) -> tuple[PolicySpec, ...]:
    if isinstance(tool_count, bool) or not isinstance(tool_count, int) or tool_count < 0:
        raise ValueError("tool_count must be a nonnegative integer")
    return (
        PolicySpec(POLICY_AGENT, "Recompute finite value and stop when the model selects STOP.", tool_count, True),
        PolicySpec(POLICY_FIXED, "Use the first affordable action in a fixed order, once.", 1, False),
        PolicySpec(POLICY_FULL, "Use every affordable action in the same fixed order.", tool_count, False),
        PolicySpec(POLICY_BASE, "Use the base prediction and stop immediately.", 0, False),
    )


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")


def _next_fixed_action(model: FiniteDecisionModel, state, action_order: Sequence[str], remaining: float) -> tuple[str | None, str]:
    observed = {action_id for action_id, _ in state.observations}
    observed_evidence = {
        model.tools[action_id].evidence_id
        for action_id in observed
        if model.tools[action_id].evidence_id is not None
    }
    for action_id in action_order:
        if action_id not in model.tools or action_id in observed:
            continue
        evidence_id = model.tools[action_id].evidence_id
        if evidence_id is not None and evidence_id in observed_evidence:
            continue
        if model.tools[action_id].cost > remaining + _TOLERANCE:
            return None, "budget_exhausted"
        return action_id, "fixed_priority"
    return None, "menu_exhausted"


def _next_full_action(model: FiniteDecisionModel, state, action_order: Sequence[str], remaining: float) -> tuple[str | None, str]:
    """Find the next affordable action, allowing a costly item to be skipped."""
    observed = {action_id for action_id, _ in state.observations}
    observed_evidence = {
        model.tools[action_id].evidence_id
        for action_id in observed
        if model.tools[action_id].evidence_id is not None
    }
    saw_unaffordable = False
    for action_id in action_order:
        if action_id not in model.tools or action_id in observed:
            continue
        evidence_id = model.tools[action_id].evidence_id
        if evidence_id is not None and evidence_id in observed_evidence:
            continue
        if model.tools[action_id].cost > remaining + _TOLERANCE:
            saw_unaffordable = True
            continue
        return action_id, "full_budget"
    return None, "budget_exhausted" if saw_unaffordable else "menu_exhausted"


def _sample_hidden(model: FiniteDecisionModel, seed: int) -> tuple[int, str, int, str]:
    latent_index = _draw(model.prior, random.Random(f"{seed}:latent"))
    latent = model.hypothesis_names[latent_index]
    result_index = _draw(model.outcome_probabilities[latent_index], random.Random(f"{seed}:result"))
    result = model.outcome_labels[result_index]
    return latent_index, latent, result_index, result


def _realized_loss(model: FiniteDecisionModel, prediction: Sequence[float], result_index: int) -> float:
    if model.loss == "log_loss":
        return -math.log(prediction[result_index])
    return math.fsum((probability - int(index == result_index)) ** 2
                     for index, probability in enumerate(prediction))


def replay_policy(
    model: FiniteDecisionModel,
    policy_id: str,
    *,
    seed: int,
    budget: float,
    horizon: int = 1,
    action_order: Sequence[str] | None = None,
    max_actions: int | None = None,
) -> dict:
    """Replay one policy while keeping hidden variables outside its trace."""
    if not isinstance(model, FiniteDecisionModel):
        raise ValueError("model must be a FiniteDecisionModel")
    if model.loss not in {"log_loss", "brier"}:
        raise ValueError("Prediction replay requires log_loss or brier")
    if policy_id not in POLICY_IDS:
        raise ValueError(f"Unknown policy: {policy_id}")
    _validate_seed(seed)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in (0, 1, 2):
        raise ValueError("horizon must be 0, 1 or 2")
    order = tuple(sorted(model.tools) if action_order is None else action_order)
    if len(set(order)) != len(order) or any(action_id not in model.tools for action_id in order):
        raise ValueError("action_order must contain unique registered tool IDs")
    if set(order) != set(model.tools):
        raise ValueError("action_order must cover every registered tool exactly once")
    if max_actions is None:
        max_actions = len(model.tools) if policy_id in (POLICY_AGENT, POLICY_FULL) else (1 if policy_id == POLICY_FIXED else 0)
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 0:
        raise ValueError("max_actions must be a nonnegative integer")

    latent_index, latent, result_index, result = _sample_hidden(model, seed)
    remaining = float(budget)
    state = model.initial_state()
    trace = []
    stop_reason = "max_actions_reached"
    for step in range(max_actions + 1):
        if policy_id == POLICY_BASE:
            action_id, selection_basis = None, "base_stop"
        elif policy_id == POLICY_AGENT:
            decision = model.evaluate(state, remaining_budget=remaining,
                                      horizon=min(horizon, max_actions - step))
            action_id = decision["action_id"] if decision["action_id"] != "STOP" else None
            selection_basis = "finite_model_value_gate" if action_id is not None else decision["reason"]
        else:
            if policy_id == POLICY_FIXED:
                action_id, selection_basis = _next_fixed_action(model, state, order, remaining)
                if step >= 1:
                    action_id, selection_basis = None, "fixed_priority_action_cap"
            else:
                action_id, selection_basis = _next_full_action(model, state, order, remaining)
            if action_id is not None and step >= max_actions:
                action_id, selection_basis = None, "max_actions_reached"
        event = {
            "step": step,
            "belief_before": dict(zip(model.hypothesis_names, state.belief)),
            "observed_before": [list(pair) for pair in state.observations],
            "selected_action": action_id,
            "selection_basis": selection_basis,
        }
        if action_id is None:
            stop_reason = selection_basis
            trace.append(event)
            break
        tool = model.tools[action_id]
        outcome_index = _draw(tool.likelihoods[latent], random.Random(f"{seed}:tool:{tool.evidence_id or action_id}"))
        outcome = tool.outcomes[outcome_index]
        state = model.observe(state, action_id, outcome)
        remaining = max(0.0, remaining - tool.cost)
        event.update({
            "observation": outcome,
            "belief_after": dict(zip(model.hypothesis_names, state.belief)),
            "prediction_after": list(model.prediction(state)),
            "cost": tool.cost,
            "remaining_budget_after": remaining,
        })
        trace.append(event)

    prediction = model.prediction(state)
    return {
        "schema_version": VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "policy_id": policy_id,
        "seed": seed,
        "budget": float(budget),
        "horizon": horizon,
        "max_actions": max_actions,
        "action_order": list(order),
        "trace": trace,
        "final_prediction": list(prediction),
        "spent": float(budget) - remaining,
        "query_count": sum(1 for event in trace if event["selected_action"] is not None),
        "stop_reason": stop_reason,
        "evaluator_only": {
            "sampled_hypothesis": latent,
            "sampled_result": result,
            "realized_loss": _realized_loss(model, prediction, result_index),
            "interpretation": "Hidden state and eventual outcome are shared evaluator streams; neither enters policy inputs.",
        },
    }


def compare_policies(
    model: FiniteDecisionModel,
    *,
    seeds: Iterable[int],
    budget: float,
    horizon: int = 1,
    action_order: Sequence[str] | None = None,
    policies: Sequence[str] = POLICY_IDS,
) -> dict:
    """Compare controls on matched hidden streams and return evaluator data."""
    selected = tuple(policies)
    if not selected or len(set(selected)) != len(selected) or any(policy_id not in POLICY_IDS for policy_id in selected):
        raise ValueError("policies must be nonempty unique registered policy IDs")
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("At least one seed is required")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("Seeds must be unique")
    episodes = {policy_id: [] for policy_id in selected}
    for seed in seed_values:
        row_by_policy = {}
        for policy_id in selected:
            row = replay_policy(model, policy_id, seed=seed, budget=budget, horizon=horizon, action_order=action_order)
            row_by_policy[policy_id] = row
            episodes[policy_id].append(row)
        hidden_pairs = {
            (row["evaluator_only"]["sampled_hypothesis"], row["evaluator_only"]["sampled_result"])
            for row in row_by_policy.values()
        }
        if len(hidden_pairs) != 1:
            raise AssertionError("Policies did not share the matched hidden streams")

    summaries = {}
    for policy_id, rows in episodes.items():
        summaries[policy_id] = {
            "episodes": len(rows),
            "mean_spent": math.fsum(row["spent"] for row in rows) / len(rows),
            "mean_query_count": math.fsum(row["query_count"] for row in rows) / len(rows),
            "mean_realized_loss_evaluator_only": math.fsum(row["evaluator_only"]["realized_loss"] for row in rows) / len(rows),
            "stop_reasons": dict(sorted(Counter(row["stop_reason"] for row in rows).items())),
            "action_counts": dict(sorted(Counter(
                event["selected_action"] for row in rows for event in row["trace"]
                if event["selected_action"] is not None
            ).items())),
        }
    return {
        "schema_version": VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "visibility": "evaluator_only",
        "comparison_contract": {
            "same_model_manifest": model.manifest(),
            "same_budget": float(budget),
            "same_horizon": horizon,
            "same_seed_set": list(seed_values),
            "same_action_order": list(sorted(model.tools) if action_order is None else action_order),
            "matched_hidden_streams": True,
            "eventual_match_result_is_policy_input": False,
        },
        "summaries": summaries,
        "episodes": episodes,
    }


__all__ = [
    "VERSION", "POLICY_AGENT", "POLICY_FIXED", "POLICY_FULL", "POLICY_BASE", "POLICY_IDS",
    "PolicySpec", "default_policy_specs", "replay_policy", "compare_policies",
]
