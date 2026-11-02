"""Prospective return/continuation forecasts checked against actual inquiry IO.

Forecast checks concern an extracted operational state and a proposed next
action. They are not risk-reduction labels or calibrated return probabilities.
"""
from __future__ import annotations

from copy import deepcopy

from jsonschema import Draft202012Validator

from .audit import digest
from .inquiry_contract import ID, IDS, obj
from .inquiry_hypotheses import KINDS
from .inquiry_applicability import ApplicabilityProtocol, SCHEMA as BASE_SCHEMA, PROMPT as BASE_PROMPT


VERSION = "return-forecast-inquiry-v1"
OUTCOMES = ("source_support", "source_challenge", "mixed_source_signals", "context_only",
            "discovery", "no_answer", "failure")
KINDS_LIST = {"type": "array", "items": {"enum": KINDS}, "maxItems": len(KINDS), "uniqueItems": True}
NEXT_ACTION = obj({
    "kind": {"enum": ["query", "reflect", "stop", "wait", "defer", "reassess"]},
    "target_hypothesis_ids": {**IDS, "maxItems": 6}, "evidence_types": KINDS_LIST,
    "reason": {"type": "string", "minLength": 1, "maxLength": 400},
})
RETURN_MODEL = obj({"branches": {"type": "array", "minItems": len(OUTCOMES), "maxItems": len(OUTCOMES),
    "items": obj({"outcome": {"enum": list(OUTCOMES)},
                  "missing_after": {"type": "array", "minItems": 1, "maxItems": 6,
                      "items": obj({"hypothesis_id": ID, "types": KINDS_LIST})},
                  "next_action": NEXT_ACTION})}})
SCHEMA = deepcopy(BASE_SCHEMA)
candidate_schema = SCHEMA["properties"]["candidates"]["items"]
candidate_schema["properties"]["return_model"] = RETURN_MODEL
candidate_schema["required"].append("return_model")

PROMPT = BASE_PROMPT + """
Return-forecast extension:
Before every external query provide return_model with exactly these seven operational outcomes:
source_support, source_challenge, mixed_source_signals, context_only, discovery, no_answer, failure.
They describe extracted evidence, not all possible world states. Each branch predicts the remaining
requirement types for every target of the query, and the next action you would propose in that case.
This makes the anticipated use of information falsifiable. No branch probabilities or numeric gains
are configured: never invent them. Declare genuinely open next steps as reassess, not fake precision.

source_support/challenge/mixed/context_only follow the existing document contribution projection.
discovery means newly seen allowed links with no attributed source support/challenge/context;
it does NOT mean those links are relevant or true. no_answer includes NO_DATA and no usable
extracted evidence/new links. failure covers unsuccessful or unusable acquisition. These operational
categories do not prove that the actual page contained no useful information: extraction may miss it.

For each branch, missing_after lists ALL query target hypotheses and their forecast remaining
types. Input coverage is distinct from hypothesis truth. Merely supporting a proposition does not
guarantee all its inputs are available. Unknown/failed answers normally leave existing gaps open.
For a query next_action give known target hypotheses and evidence types; for every other kind use
empty target_hypothesis_ids and evidence_types. Newly emerging questions may require reassess.
Account for the remaining HTTP/model budgets AFTER the proposed current request. After the last
HTTP request a proposed next query is not executable without a new task budget.

The runtime freezes the selected plan before acquisition. On the next reasoning step it compares
that plan with the observed extracted state and your newly proposed continuation. You are free to
change plans when new evidence warrants it; explain deviations instead of forcing reality to match
the old forecast. Do not manufacture claims to make a predicted branch come true. Continue from
actual documents and current unknowns. return_reviews records these comparisons, not whether
the chosen query was optimal or useful for forecasting the match.

If all successful return branches use the same continuation, explain any remaining decision role
of the information. Such invariance is a planning diagnostic, NOT proof of zero VoI: final belief,
later actions, or unmodeled outcomes may still differ. Keep at most TWO candidates and concise
branch reasons. Prefer using inherited evidence directly; do not refresh unchanged material by habit.
"""


def action_signature(action):
    return {"kind": action["kind"], "target_hypothesis_ids": sorted(action["target_hypothesis_ids"]),
            "evidence_types": sorted(action["evidence_types"])}


def actual_continuation(response):
    if response["decision"]["kind"] != "query":
        return {"kind": response["decision"]["kind"], "target_hypothesis_ids": [], "evidence_types": []}
    candidate = next(c for c in response["candidates"] if c["candidate_id"] == response["decision"]["candidate_id"])
    return action_signature({"kind": "query", "target_hypothesis_ids": candidate["inquiry_intent"]["hypothesis_ids"],
                             "evidence_types": candidate["inquiry_intent"]["expected_evidence"]})


def gap_signature(requirements, identities):
    return sorted([{"hypothesis_id": item["hypothesis_id"], "types": sorted(item.get("types", item.get("missing_types", [])))}
                   for item in requirements if item["hypothesis_id"] in identities], key=lambda item: item["hypothesis_id"])


def summarize_model(model):
    successful = [b for b in model["branches"] if b["outcome"] not in {"no_answer", "failure"}]
    continuations = {digest(action_signature(b["next_action"])) for b in successful}
    gaps = {digest(gap_signature(b["missing_after"], {x["hypothesis_id"] for x in b["missing_after"]}))
            for b in successful}
    return {"outcomes": list(OUTCOMES), "successful_continuation_variants": len(continuations),
            "successful_gap_variants": len(gaps), "continuation_sensitive": len(continuations) > 1,
            "open_continuation_outcomes": [b["outcome"] for b in model["branches"] if b["next_action"]["kind"] == "reassess"],
            "probability_model": "UNSUPPORTED", "risk_update_model": "UNSUPPORTED",
            "expected_gain": None, "scope": "declared_plan_structure_not_information_value"}


def classify_return(receipt, query_review, document):
    if receipt["status"] == "NO_DATA":
        return "no_answer", []
    if receipt["status"] != "OK":
        return "failure", []
    if query_review["outcome"] in {"source_support", "source_challenge", "mixed_source_signals", "context_only"}:
        return query_review["outcome"], []
    known = set(receipt["inquiry_contract"]["known_urls_before"])
    new_links = sorted({link["url"] for link in document["links"]} - known)
    return ("discovery", new_links) if new_links else ("no_answer", [])


def check_forecast(receipt, query_review, document, response, requirements):
    contract = receipt["inquiry_contract"]
    outcome, new_links = classify_return(receipt, query_review, document)
    branch = next(b for b in contract["return_model"]["branches"] if b["outcome"] == outcome)
    ids = contract["intent"]["hypothesis_ids"]
    predicted_gaps = gap_signature(branch["missing_after"], ids)
    observed_gaps = gap_signature(requirements, ids)
    predicted_action = action_signature(branch["next_action"])
    proposed_action = actual_continuation(response)
    action_match = None if predicted_action["kind"] == "reassess" else predicted_action == proposed_action
    gaps_match = predicted_gaps == observed_gaps
    status = ("deviation_from_declared_plan" if not gaps_match or action_match is False else
              "open_continuation_not_scored" if action_match is None else "matched_declared_plan")
    return {"request_id": receipt["request_id"], "document_id": receipt["document_id"],
            "forecast_hash": digest(contract["return_model"]), "outcome": outcome, "new_allowed_links": new_links,
            "predicted_missing_after": predicted_gaps, "observed_missing_after": observed_gaps,
            "gap_forecast_match": gaps_match, "predicted_continuation": predicted_action,
            "proposed_continuation": proposed_action, "continuation_match": action_match, "status": status,
            "continuation_scope": "next_proposal_not_execution_or_optimality",
            "state_change_scope": "joint_extracted_state_not_single_query_causal_effect",
            "outcome_scope": "observed_extraction_not_complete_world_state",
            "expected_gain": None, "realized_decision_utility": None}


class ReturnForecastProtocol(ApplicabilityProtocol):
    version, schema, prompt = VERSION, SCHEMA, PROMPT
    projection_fields = (*ApplicabilityProtocol.projection_fields, "return_reviews")

    def validate(self, response, public):
        Draft202012Validator(self.schema).validate(response)
        base = deepcopy(response)
        for candidate in base["candidates"]:
            candidate.pop("return_model")
        ApplicabilityProtocol().validate(base, public)
        hypotheses = {h["hypothesis_id"]: h for h in public.get("hypotheses", []) + response["hypotheses"]}
        for candidate in response["candidates"]:
            branches = candidate["return_model"]["branches"]
            if {b["outcome"] for b in branches} != set(OUTCOMES):
                raise ValueError("Exactly one branch per operational return outcome required")
            targets = set(candidate["inquiry_intent"]["hypothesis_ids"])
            for branch in branches:
                ids = [x["hypothesis_id"] for x in branch["missing_after"]]
                if len(ids) != len(set(ids)) or set(ids) != targets:
                    raise ValueError("Forecast must cover precisely the query target hypotheses")
                for item in branch["missing_after"]:
                    if not set(item["types"]) <= set(hypotheses[item["hypothesis_id"]]["required_evidence"]):
                        raise ValueError("Forecast introduces an undeclared requirement")
                action = branch["next_action"]
                if set(action["target_hypothesis_ids"]) - hypotheses.keys():
                    raise ValueError("Continuation targets an undeclared hypothesis; use reassess")
                if action["kind"] == "query":
                    if not action["target_hypothesis_ids"] or not action["evidence_types"]:
                        raise ValueError("Query continuation needs targets and evidence types")
                elif action["target_hypothesis_ids"] or action["evidence_types"]:
                    raise ValueError("Non-query continuation uses empty query fields")
        return response

    def public(self, run):
        return {**super().public(run), "return_reviews": deepcopy(run.get("return_reviews", []))}

    def advance(self, run, response):
        assessment = super().advance(run, response)
        checked = {x["request_id"] for x in run.get("return_reviews", [])}
        reviews = []
        for review in run["query_reviews"]:
            if review["request_id"] in checked:
                continue
            receipt = next(r for r in run["receipts"] if r["request_id"] == review["request_id"])
            document = next(d for d in run["documents"] if d["document_id"] == review["document_id"])
            reviews.append(check_forecast(receipt, review, document, response, run["requirements"]))
        run.setdefault("return_reviews", []).extend(reviews)
        assessment["return_reviews"] = reviews
        for candidate, feedback in zip(response["candidates"], assessment["candidate_assessments"]):
            feedback["return_forecast"] = summarize_model(candidate["return_model"])
        return assessment

    def contract(self, run, candidate):
        urls = set(run["task"]["seed_urls"])
        for document in run["documents"]:
            urls.add(document["url"])
            urls.update(link["url"] for link in document["links"])
        return {**super().contract(run, candidate), "return_model": deepcopy(candidate["return_model"]),
                "known_urls_before": sorted(urls)}
