"""Contracts separating g predictions, exploration, and realized labels.

This module is deliberately independent of the formal empirical-g runtime.  It
provides the small, auditable rules used while collecting training data:

* a prediction from g is never a realized label;
* an unexecuted action has no realized gain (``None``);
* executed actions may have positive, negative, or zero realized gain; and
* epsilon-greedy exploration exposes a propensity for every candidate.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Mapping


LABEL_CONTRACT_SCHEMA = "footballmind-g-label-contract-v1"
EXPLORATION_POLICY_SCHEMA = "footballmind-g-exploration-policy-v1"


def realized_signed_gain(
    *,
    loss_before: float | None,
    loss_after: float | None,
    selected: bool,
    f_after_complete: bool,
    outcome_available: bool,
) -> float | None:
    """Return a post-outcome label, or ``None`` when it is not observable.

    The function intentionally does not inspect g's prediction.  That prevents
    predicted negative utility from being converted into a negative label.
    """

    if not selected or not f_after_complete or not outcome_available:
        return None
    if loss_before is None or loss_after is None:
        return None
    before = float(loss_before)
    after = float(loss_after)
    if not (math.isfinite(before) and math.isfinite(after)):
        raise ValueError("loss values must be finite")
    return before - after


def validate_transition(row: Mapping[str, Any]) -> list[str]:
    """Validate the non-circular label rules for one transition."""

    errors: list[str] = []
    selected = row.get("selected")
    gain = row.get("realized_signed_gain", row.get("signed_gain"))
    status = row.get("label_status")

    if selected is False and gain is not None:
        errors.append("unselected_action_must_have_null_realized_gain")
    if status == "unexecuted" and gain is not None:
        errors.append("unexecuted_action_must_have_null_realized_gain")
    if gain is not None:
        try:
            if not math.isfinite(float(gain)):
                errors.append("realized_gain_must_be_finite")
        except (TypeError, ValueError):
            errors.append("realized_gain_must_be_numeric")

    # A prospective label requires an observed execution and a propensity.
    # Historical reconstructions may retain an observed gain, but they remain
    # diagnostic when propensity is missing.
    if status == "prospective_realized" and selected is not True:
        errors.append("prospective_realized_requires_selected_true")
    if status == "prospective_realized" and row.get("propensity") is None:
        errors.append("prospective_realized_requires_propensity")
    if row.get("predicted_signed_gain") is not None and row.get("predicted_signed_gain") == gain:
        # Equality is allowed numerically; this is only a marker for callers to
        # inspect.  Do not reject it, since a prediction can happen to match a
        # label by chance.
        pass
    return errors


def epsilon_greedy_propensities(
    action_ids: Iterable[str],
    scores: Mapping[str, float],
    *,
    epsilon: float,
) -> dict[str, float]:
    """Return an epsilon-greedy distribution over an explicit candidate menu."""

    actions = list(action_ids)
    if not actions:
        raise ValueError("action menu cannot be empty")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be within [0, 1]")
    missing = [a for a in actions if a not in scores]
    if missing:
        raise ValueError(f"missing scores for actions: {missing}")
    best = max(actions, key=lambda a: float(scores[a]))
    n = len(actions)
    base = epsilon / n
    result = {a: base for a in actions}
    result[best] += 1.0 - epsilon
    return result


def sample_action(
    propensities: Mapping[str, float], *, seed: int | None = None
) -> tuple[str, float]:
    """Sample one action and return ``(action_id, propensity)``."""

    if not propensities:
        raise ValueError("propensity map cannot be empty")
    total = sum(float(v) for v in propensities.values())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
        raise ValueError("propensities must sum to one")
    if any(float(v) <= 0.0 for v in propensities.values()):
        raise ValueError("every candidate must have positive propensity")
    rng = random.Random(seed)
    draw = rng.random()
    acc = 0.0
    for action_id, probability in propensities.items():
        acc += float(probability)
        if draw <= acc:
            return action_id, float(probability)
    action_id = next(reversed(propensities))
    return action_id, float(propensities[action_id])


__all__ = [
    "LABEL_CONTRACT_SCHEMA",
    "EXPLORATION_POLICY_SCHEMA",
    "realized_signed_gain",
    "validate_transition",
    "epsilon_greedy_propensities",
    "sample_action",
]
