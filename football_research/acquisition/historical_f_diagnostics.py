"""Offline, exact log-odds accounting for a frozen historical text predictor.

This is an explanation of a fitted function, not a semantic relevance model,
an expected information gain estimator, or evidence of predictive improvement.
The controlled clock and leave-one-document views are diagnostic interventions;
they must never be substituted into an observed acquisition trajectory.
"""
from copy import deepcopy

import numpy as np

from .context_predictor import NUMERIC
from .fixture_roles import ROLES
from .historical_context_training import HistoricalContextModel

VERSION = "frozen-historical-f-logodds-diagnostic-v1"


def _parts(model, state):
    prediction = model.predict(state)
    projected, probabilities, known, base = model._probabilities(state)
    vector, _ = model.f._vector(projected)
    coefficients = np.asarray(model.f.coefficients)
    active = bool(known) and not model.metadata.get("f_selection", {}).get("base_only_won", False)
    contributions, offset = {}, 0
    width = len(model.f.encoder["vocabulary"])
    for role in ROLES:
        contributions["lexical:" + role] = (vector[offset:offset+width] @ coefficients[offset:offset+width]
                                               if active else np.zeros(3))
        offset += width
    for name in NUMERIC:
        contributions["numeric:" + name] = vector[offset] * coefficients[offset] if active else np.zeros(3)
        offset += 1
    for host in model.f.encoder["sources"]:
        contributions["source:" + host] = vector[offset] * coefficients[offset] if active else np.zeros(3)
        offset += 1
    if offset != len(vector):
        raise ValueError("The frozen encoder layout is not supported by this diagnostic")
    log_odds = np.log(np.asarray(probabilities)[1:] / probabilities[0])
    base_log_odds = np.log(np.asarray(base)[1:] / base[0])
    contribution_sum = sum((value[1:] - value[0] for value in contributions.values()), np.zeros(2))
    error = float(np.max(np.abs(log_odds - base_log_odds - contribution_sum)))
    if error > 1e-10:
        raise ValueError("Feature accounting does not reconstruct the frozen predictor")
    return {"prediction": prediction, "projected": projected, "contributions": contributions,
            "base_log_odds": base_log_odds, "log_odds": log_odds, "error": error}


def diagnose_transition(model, before, after):
    """Account for one already-observed transition without fitting or labels.

    Comparison coordinates are log(P(draw)/P(home)), log(P(away)/P(home)).
    Block deltas add exactly in these coordinates; probabilities do not add.
    Global TF-IDF normalization can change old blocks when new text arrives.
    """
    if not isinstance(model, HistoricalContextModel):
        raise TypeError("A frozen HistoricalContextModel is required")
    if before["fixture"] != after["fixture"]:
        raise ValueError("Diagnostic before/after must have the same ordered fixture")
    if before.get("prior_probabilities") != after.get("prior_probabilities"):
        raise ValueError("A changed prior cannot be attributed to an information query")
    b, a = _parts(model, before), _parts(model, after)
    at_after_clock = deepcopy(before)
    for key in ("decision_at", "prediction_cutoff"):
        at_after_clock[key] = after[key]
    c = _parts(model, at_after_clock)
    blocks = []
    for key in a["contributions"]:
        delta = a["contributions"][key] - c["contributions"][key]
        odds_delta = delta[1:] - delta[0]
        blocks.append({"block": key, "log_odds_delta": odds_delta.tolist(),
                       "max_absolute_delta": float(np.max(np.abs(odds_delta)))})
    expected_delta = a["log_odds"] - c["log_odds"]
    accounting_error = float(np.max(np.abs(expected_delta -
        sum((np.asarray(row["log_odds_delta"]) for row in blocks), np.zeros(2)) -
        (a["base_log_odds"] - c["base_log_odds"]))))
    body_views = []
    for index, document in enumerate(after.get("documents", [])):
        if any(document == old for old in before.get("documents", [])):
            continue
        view = deepcopy(after)
        del view["documents"][index]
        body_views.append({"url": document["url"], "raw_sha256": document["raw_sha256"],
            "without_this_document_probabilities": model.predict(view)["probabilities"],
            "retained": "all after-state observed link labels and history, at the same clock",
            "scope": "posthoc_input_sensitivity_not_a_possible_prequery_state"})
    return {"schema": VERSION, "model_version": model.model_version,
        "physical_match_id": after["fixture"]["physical_match_id"],
        "observed_before_probabilities": b["prediction"]["probabilities"],
        "observed_after_probabilities": a["prediction"]["probabilities"],
        "before_information_at_after_clock_probabilities": c["prediction"]["probabilities"],
        "clock_only_probability_delta": (np.asarray(c["prediction"]["probabilities"]) -
                                           b["prediction"]["probabilities"]).tolist(),
        "log_odds_coordinates": ["draw/home", "away/home"],
        "same_clock_log_odds_delta": expected_delta.tolist(),
        "feature_block_deltas": sorted(blocks, key=lambda row: (-row["max_absolute_delta"], row["block"])),
        "maximum_accounting_error": max(accounting_error, a["error"], b["error"], c["error"]),
        "new_document_views": body_views,
        "after_role_text_counts": {key: len(value) for key, value in a["projected"]["role_texts"].items()},
        "interpretation": "exact_function_accounting_not_causal_relevance_or_risk_reduction",
        "used_result_labels": False, "expected_gain": None, "realized_gain": None}
