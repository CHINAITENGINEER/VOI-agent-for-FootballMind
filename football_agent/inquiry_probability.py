"""Prospective, uncalibrated forecasts of operational query returns.

Scores assess the frozen forecast against the existing extraction protocol's
return label. They are not acquisition utility, world truth or a calibrated
likelihood conditional on a latent football state.
"""
from __future__ import annotations

from copy import deepcopy
import math

from jsonschema import Draft202012Validator

from .audit import digest
from .inquiry_choice import InformationChoiceProtocol
from .inquiry_contract import IDS, obj
from .inquiry_returns import OUTCOMES, VERSION as CLASSIFIER_VERSION


VERSION = "return-probability-inquiry-v1"
PROBABILITIES = obj({name: {"type": "number", "minimum": 0, "maximum": 1} for name in OUTCOMES})
FORECAST = obj({
    "status": {"enum": ["elicited", "abstain"]},
    "probabilities": {"anyOf": [PROBABILITIES, {"type": "null"}]},
    "basis_document_ids": IDS,
    "reason": {"type": "string", "minLength": 1, "maxLength": 600},
})
SCHEMA = deepcopy(InformationChoiceProtocol.schema)
_candidate = SCHEMA["properties"]["candidates"]["items"]
_candidate["properties"]["outcome_forecast"] = FORECAST
_candidate["required"].append("outcome_forecast")

# This extension explicitly distinguishes an elicited forecast from an
# invented claim of measured/calibrated probabilities. Legacy prompts stay put.
PROMPT = InformationChoiceProtocol.prompt + """
Operational return-probability extension (a narrow exception to the earlier prohibition on
numerical open-text probabilities): you MAY report YOUR subjective pre-query probabilities only
in outcome_forecast. They are explicitly ELICITED, UNCALIBRATED forecasts, not measured rates,
source reliability, match-result probabilities, numerical VoI or a latent-state likelihood law.
There is no trained or validated return model. Honest subjective uncertainty is allowed; if you
decline to quantify, set status=abstain, probabilities=null and explain. Do not manufacture a
historical frequency or claim calibration. Elicited probabilities must cover the seven existing
operational outcomes exactly once and sum to one; no auto-normalization or probability flooring.

Forecast the outcome that would be recorded by the next extraction/review pass for this query's
frozen intent. Use the existing OUTCOMES priority: NO_DATA -> no_answer; other failed/unusable
transport -> failure; attributable support/challenge/mixed/context -> that category; otherwise new allowed
links -> discovery; otherwise no_answer. Background WITH links can still be context_only. These
labels are mutually exclusive protocol observations, not exhaustive semantic world states.
Successful HTTP, an applicable factual input, and final risk reduction are different events.
Use only currently supplied documents and catalog/task context. basis_document_ids may be [] and
must name actually acquired documents, never guessed article URLs or unobserved alternatives.

Every candidate gets an outcome_forecast before selection/execution. The runtime freezes the
selected distribution and only checks its realized outcome later, with raw multiclass Brier and
log loss plus a fixed uniform reference. Abstention is unscored and counted separately. A selected
zero-probability outcome produces infinite log loss; it is never clipped or discarded as missing.
Unselected candidates remain unobserved. A forecast for a later revisit is a NEW forecast, not a
label for an earlier unselected candidate. Recent feedback is adaptive within this investigation,
not independent validation or automatic calibration.
The uniform-seven reference is deliberately weak: search discovery documents cannot support
source claims, so restricting predictions to reachable categories can beat it without learning
anything about source quality. The label-producing extraction pass sees previous forecasts;
self-confirmation and missed extraction remain possible, not independently resolved by scoring.

Choose information for its role in the task, not to achieve a good forecast score. A certain tool
failure can be easy to predict and still provide no useful evidence. The forecast alone supplies
no expected decision-risk reduction, and numeric query value remains unsupported. The next
extraction pass must preserve evidence even when it contradicts the forecast; never create or
suppress source claims to make the forecast look correct. Extraction semantics are unverified.
"""

SEMANTICS = {
    "origin": "llm_subjective_elicitation_not_calibrated",
    "target": "next_protocol_return_label_for_frozen_query_intent",
    "outcome_order": list(OUTCOMES),
    "raw_brier_range": [0, 2],
    "log_loss_base": "natural_log",
    "zero_probability_log_loss": "positive_infinity_with_null_numeric_value",
    "reference": "fixed_uniform_over_seven_outcomes",
    "reference_limit": "weak_reference_not_conditioned_on_tool_reachable_outcomes",
    "classifier_version": CLASSIFIER_VERSION,
    "label_revision_policy": "freeze_first_subsequent_valid_extraction_label",
    "extraction_blinded_to_forecast": False,
    "label_origin": "transport_and_llm_extraction_not_independent_semantic_truth",
    "selection_scope": "executed_generated_candidates_only",
    "acquisition_utility": None, "calibration_established": False,
}


def validate_distribution(probabilities):
    if not isinstance(probabilities, dict) or set(probabilities) != set(OUTCOMES):
        raise ValueError("A probability for each operational outcome is required")
    values = list(probabilities.values())
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Probabilities must be finite numbers in [0, 1]")
    if abs(math.fsum(values)-1) > 1e-9:
        raise ValueError("Probabilities must sum to one; no implicit renormalization")


def score_distribution(probabilities, outcome):
    validate_distribution(probabilities)
    if outcome not in OUTCOMES:
        raise ValueError("Unknown operational outcome")
    probability = probabilities[outcome]
    return {"raw_multiclass_brier": math.fsum((probabilities[key]-int(key == outcome))**2 for key in OUTCOMES),
            "log_loss_nats": -math.log(probability) if probability else None,
            "log_loss_status": "finite" if probability else "positive_infinity",
            "realized_outcome_probability": probability}


def evaluate_forecast(forecast, outcome):
    Draft202012Validator(FORECAST).validate(forecast)
    if forecast["status"] == "abstain" and forecast["probabilities"] is not None:
        raise ValueError("Abstention must leave probabilities unknown")
    baseline = score_distribution({name: 1/len(OUTCOMES) for name in OUTCOMES}, outcome)
    own = score_distribution(forecast["probabilities"], outcome) if forecast["status"] == "elicited" else None
    return {"status": "scored_uncalibrated_forecast" if own else "abstained_not_scored",
            "forecast_scores": own, "uniform_reference_scores": baseline,
            "brier_difference_vs_uniform": own["raw_multiclass_brier"]-baseline["raw_multiclass_brier"] if own else None,
            "comparison_scope": "descriptive_selected_return_prediction_not_policy_utility"}


def summarize_reviews(reviews):
    scored = [r["evaluation"]["forecast_scores"] for r in reviews if r["evaluation"]["forecast_scores"] is not None]
    infinite = sum(s["log_loss_status"] == "positive_infinity" for s in scored)
    return {"observed_reviews": len(reviews), "scored_forecasts": len(scored), "abstentions": len(reviews)-len(scored),
            "mean_raw_brier": math.fsum(s["raw_multiclass_brier"] for s in scored)/len(scored) if scored else None,
            "mean_log_loss_nats": math.fsum(s["log_loss_nats"] for s in scored)/len(scored) if scored and not infinite else None,
            "mean_log_loss_status": "positive_infinity" if infinite else "finite" if scored else "not_scored",
            "zero_probability_observations": infinite,
            "sampling_scope": "adaptive_selected_returns_not_iid_or_counterfactuals",
            "calibration_established": False, "empirical_information_value": None}


class ReturnProbabilityProtocol(InformationChoiceProtocol):
    version, schema, prompt = VERSION, SCHEMA, PROMPT
    projection_fields = (*InformationChoiceProtocol.projection_fields, "return_probability_reviews")

    def validate(self, response, public):
        Draft202012Validator(self.schema).validate(response)
        base = deepcopy(response)
        docs = {doc["document_id"] for doc in public["documents"]}
        for candidate in base["candidates"]:
            forecast = candidate.pop("outcome_forecast")
            if set(forecast["basis_document_ids"]) - docs:
                raise ValueError("Return forecast cites an unacquired document")
            if forecast["status"] == "elicited":
                validate_distribution(forecast["probabilities"])
            elif forecast["probabilities"] is not None:
                raise ValueError("Abstention must leave probabilities unknown")
        InformationChoiceProtocol().validate(base, public)
        return response

    def public(self, run):
        return {**super().public(run), "return_probability_semantics": deepcopy(SEMANTICS),
                "return_probability_reviews": deepcopy(run.get("return_probability_reviews", []))}

    def advance(self, run, response):
        assessment = super().advance(run, response)
        seen = {r["request_id"] for r in run.get("return_probability_reviews", [])}
        reviews = []
        for review in run["return_reviews"]:
            if review["request_id"] in seen:
                continue
            receipt = next(r for r in run["receipts"] if r["request_id"] == review["request_id"])
            forecast = receipt["inquiry_contract"]["outcome_forecast"]
            reviews.append({"request_id": review["request_id"], "document_id": review["document_id"],
                "forecast_hash": digest(forecast), "forecast": deepcopy(forecast), "observed_outcome": review["outcome"],
                "evaluation": evaluate_forecast(forecast, review["outcome"]),
                "classifier_version": CLASSIFIER_VERSION,
                "label_scope": "transport_and_protocol_extraction_not_independent_truth",
                "realized_decision_utility": None})
        run.setdefault("return_probability_reviews", []).extend(reviews)
        assessment["return_probability_reviews"] = reviews
        assessment["return_probability_summary"] = summarize_reviews(run["return_probability_reviews"])
        return assessment

    def contract(self, run, candidate):
        return {**super().contract(run, candidate), "outcome_forecast": deepcopy(candidate["outcome_forecast"]),
                "outcome_forecast_semantics": deepcopy(SEMANTICS)}
