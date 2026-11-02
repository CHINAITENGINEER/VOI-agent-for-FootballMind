"""Typed inquiry contracts and reproducible, qualitative evidence accounting.

Scope/type checks constrain LLM extraction; they do not verify its semantics or
estimate forecast utility. Query reviews are observations for a future value
model, not counterfactual information-value labels.
"""
from __future__ import annotations

from copy import deepcopy

from jsonschema import Draft202012Validator

from .inquiry_contract import ID, IDS, TEXT, PROMPT, RESPONSE_SCHEMA, obj, validate_response


VERSION = "hypothesis-inquiry-v1"
KINDS = ["roster", "official_lineup", "lineup_comparison", "coach_statement",
         "standings", "competition_rules", "injury_update", "other"]
MINIMUM = {
    "squad_membership": ["roster"], "starting_lineup": ["official_lineup"],
    "actual_rotation": ["official_lineup", "lineup_comparison"],
    "rotation_intent": ["coach_statement"], "tactical_intent": ["coach_statement"],
    "qualification": ["standings", "competition_rules"],
    "availability": ["injury_update"], "other": [],
}
SCOPE = {"enum": ["match", "campaign", "general"]}
KIND_LIST = {"type": "array", "items": {"enum": KINDS}, "minItems": 1,
             "maxItems": len(KINDS), "uniqueItems": True}
SCHEMA = deepcopy(RESPONSE_SCHEMA)
SCHEMA["properties"]["hypotheses"] = {"type": "array", "maxItems": 6, "items": obj({
    "hypothesis_id": ID, "question_id": ID, "statement": TEXT,
    "hypothesis_type": {"enum": list(MINIMUM)}, "subject": ID,
    "scope": SCOPE, "scope_key": ID, "required_evidence": KIND_LIST,
})}
SCHEMA["required"].append("hypotheses")
claim = SCHEMA["properties"]["claim_updates"]["items"]
claim["properties"].update({
    "evidence": obj({"kind": {"enum": KINDS}, "subject": ID, "scope": SCOPE, "scope_key": ID}),
    "hypothesis_links": {"type": "array", "maxItems": 6, "items": obj({
        "hypothesis_id": ID, "relation": {"enum": ["supports", "challenges", "context"]}})},
})
claim["required"].extend(["evidence", "hypothesis_links"])
candidate = SCHEMA["properties"]["candidates"]["items"]
candidate["properties"]["inquiry_intent"] = obj({
    "hypothesis_ids": {**IDS, "minItems": 1, "maxItems": 6},
    "purpose": {"enum": ["resolve", "challenge", "context", "refresh"]},
    "expected_evidence": KIND_LIST, "subject": ID, "scope": SCOPE, "scope_key": ID,
    "supports_change": TEXT, "challenges_change": TEXT, "no_answer_change": TEXT,
})
candidate["required"].append("inquiry_intent")

HYPOTHESIS_PROMPT = PROMPT + """
Use the hypothesis-inquiry protocol. Autonomously propose explicit hypotheses tied to questions;
they are propositions to investigate, not facts. Subject and scope_key are stable identifiers,
for example subject England, campaign 2026-09-October, match spftll:match:2981431. A match
hypothesis names the supplied match; campaign/general text must retain that wider scope. Never
infer a match scope merely because a document was fetched during a match inquiry. If an article
contains distinct campaign and match assertions, create separately anchored claims.

Each hypothesis has required_evidence. Minimum types by hypothesis_type are:
""" + str(MINIMUM) + """
starting_lineup, actual_rotation, rotation_intent, tactical_intent and qualification use match
scope. In particular a squad roster does not establish the starting eleven or actual rotation;
actual rotation additionally needs a relevant lineup comparison. Quote-backed coaching comments
describe stated intent, not observed match tactics. Treat derived conclusions as hypothesis,
never source_claim. Preserve mismatches as context instead of stretching scope/type to fit.
Hypothesis definitions and existing claim content are immutable: to refine them create new IDs.

For each candidate state in inquiry_intent the target hypotheses, anticipated evidence scope/type,
and how supporting, challenging or absent answers would alter the next decision/question. Context
may be valuable because it changes the next acquisition even if it never resolves a match claim.
These are qualitative answer branches, not estimated gains. Do not fabricate arithmetic VoI.

hypothesis_states is computed by the runtime from anchored source_claim links and declared
scope/type. complete_source_support means only that these extracted source assertions cover the
declared evidence types; it is NOT a verified fact, independence, freshness or calibrated confidence.
Opposing claims remain mixed. Assess whether more independent/timely evidence is worthwhile.
query_reviews ties the preceding query's frozen intent to its actual document; no_usable_evidence
means no attributable usable extracted claim, not proof that the information has zero value.
After every result process that document, then use the remaining unknowns to select the next
question or stop/defer/wait. Keep the response concise and preserve unsupported uncertainties.
"""


def definitions(public, updates):
    result = {h["hypothesis_id"]: deepcopy(h) for h in public.get("hypotheses", [])}
    for hypothesis in updates:
        identity = hypothesis["hypothesis_id"]
        if identity in result and result[identity] != hypothesis:
            raise ValueError("Hypothesis definitions are immutable; use a new ID")
        result[identity] = deepcopy(hypothesis)
    return list(result.values())


def applicability(hypothesis, evidence):
    for key in ("subject", "scope", "scope_key"):
        if hypothesis[key] != evidence[key]:
            return key + "_mismatch"
    if evidence["kind"] not in hypothesis["required_evidence"]:
        return "evidence_kind_mismatch"
    return "applicable"


def project(hypotheses, claims):
    states = []
    for hypothesis in hypotheses:
        support, challenge, context = [], [], []
        for claim in claims:
            for link in claim["hypothesis_links"]:
                if link["hypothesis_id"] != hypothesis["hypothesis_id"]:
                    continue
                reason = applicability(hypothesis, claim["evidence"])
                if claim["status"] != "source_claim":
                    reason = "not_direct_source_claim"
                if link["relation"] == "context":
                    reason = "context_relation"
                record = {"claim_id": claim["claim_id"], "kind": claim["evidence"]["kind"],
                          "document_ids": sorted({c["document_id"] for c in claim["citations"]})}
                if reason != "applicable":
                    context.append({**record, "reason": reason})
                else:
                    (support if link["relation"] == "supports" else challenge).append(record)
        covered = {item["kind"] for item in support}
        missing = sorted(set(hypothesis["required_evidence"]) - covered)
        if support and challenge:
            status = "mixed_source_signals"
        elif support:
            status = "partial_source_support" if missing else "complete_source_support"
        elif challenge:
            status = "source_challenge"
        else:
            status = "context_only" if context else "unresolved"
        states.append({"hypothesis_id": hypothesis["hypothesis_id"], "status": status,
                       "support": support, "challenge": challenge, "context": context,
                       "missing_support_types": missing, "semantic_verification": "unverified",
                       "source_freshness": "unverified", "forecast_utility": None})
    return states


def assess_candidate(candidate, hypotheses):
    intent = candidate["inquiry_intent"]
    targets = []
    for h in hypotheses:
        if h["hypothesis_id"] not in intent["hypothesis_ids"]:
            continue
        applicable = [kind for kind in intent["expected_evidence"]
                      if applicability(h, {**intent, "kind": kind}) == "applicable"]
        targets.append({"hypothesis_id": h["hypothesis_id"], "applicable_expected_types": applicable,
                        "role": "potential_direct_evidence" if applicable else "context_only"})
    return {"candidate_id": candidate["candidate_id"], "targets": targets,
            "selection_score": None, "expected_gain": None}


def review_query(receipt, hypotheses, claims, current_states):
    contract = receipt["inquiry_contract"]
    ids = set(contract["intent"]["hypothesis_ids"])
    targets = [h for h in hypotheses if h["hypothesis_id"] in ids]
    attributable, ambiguous = [], []
    for claim in claims:
        if not any(link["hypothesis_id"] in ids for link in claim["hypothesis_links"]):
            continue
        cited = {c["document_id"] for c in claim["citations"]}
        if receipt["document_id"] not in cited:
            continue
        if cited == {receipt["document_id"]}:
            attributable.append(claim)
        else:
            ambiguous.append(claim["claim_id"])
    contribution = project(targets, attributable)
    has_support = any(s["support"] for s in contribution)
    has_challenge = any(s["challenge"] for s in contribution)
    if receipt["status"] != "OK":
        outcome = "acquisition_failed_or_unusable"
    elif has_support and has_challenge:
        outcome = "mixed_source_signals"
    elif has_support:
        outcome = "source_support"
    elif has_challenge:
        outcome = "source_challenge"
    elif any(s["context"] for s in contribution):
        outcome = "context_only"
    else:
        outcome = "no_usable_extracted_evidence"
    return {"request_id": receipt["request_id"], "document_id": receipt["document_id"],
            "intent": deepcopy(contract["intent"]), "state_before": deepcopy(contract["state_before"]),
            "state_after": [s for s in current_states if s["hypothesis_id"] in ids],
            "document_contribution": contribution, "outcome": outcome,
            "multi_document_attribution_unknown": ambiguous,
            "counterfactual_gain": None, "label_scope": "extracted_evidence_not_decision_utility"}


class HypothesisProtocol:
    version, schema, prompt = VERSION, SCHEMA, HYPOTHESIS_PROMPT

    def validate(self, response, public):
        Draft202012Validator(self.schema).validate(response)
        # Reuse every original evidence, timing and parameter check.
        base = deepcopy(response)
        base.pop("hypotheses")
        for item in base["claim_updates"]:
            item.pop("evidence")
            item.pop("hypothesis_links")
        for item in base["candidates"]:
            item.pop("inquiry_intent")
        validate_response(base, public)
        updates = response["hypotheses"]
        if len({h["hypothesis_id"] for h in updates}) != len(updates):
            raise ValueError("Duplicate hypothesis ID")
        hypotheses = definitions(public, updates)
        hids = {h["hypothesis_id"] for h in hypotheses}
        qids = {q["question_id"] for q in public["questions"] + response["questions"]}
        for h in hypotheses:
            if h["question_id"] not in qids or not set(MINIMUM[h["hypothesis_type"]]) <= set(h["required_evidence"]):
                raise ValueError("Hypothesis lacks a question or required evidence types")
            if h["hypothesis_type"] in {"starting_lineup", "actual_rotation", "rotation_intent", "tactical_intent", "qualification"} and h["scope"] != "match":
                raise ValueError("Match-specific hypothesis needs match scope")
        old = {c["claim_id"]: {k: v for k, v in c.items() if k not in {"updated_at", "semantic_verification"}}
               for c in public["claims"]}
        for claim in response["claim_updates"]:
            if claim["claim_id"] in old and old[claim["claim_id"]] != claim:
                raise ValueError("Claims are immutable; add a new anchored claim")
            links = [link["hypothesis_id"] for link in claim["hypothesis_links"]]
            if set(links) - hids or len(links) != len(set(links)):
                raise ValueError("Unknown or repeated hypothesis link")
        for candidate in response["candidates"]:
            if set(candidate["inquiry_intent"]["hypothesis_ids"]) - hids:
                raise ValueError("Candidate targets unknown hypotheses")
        return response

    def public(self, run):
        return {key: deepcopy(run.get(key, [])) for key in ("hypotheses", "hypothesis_states", "query_reviews")}

    def advance(self, run, response):
        run["hypotheses"] = definitions(run, response["hypotheses"])
        run["hypothesis_states"] = project(run["hypotheses"], run["claims"])
        reviewed = {r["request_id"] for r in run.get("query_reviews", [])}
        reviews = [review_query(r, run["hypotheses"], run["claims"], run["hypothesis_states"])
                   for r in run["receipts"] if r["request_id"] not in reviewed]
        run.setdefault("query_reviews", []).extend(reviews)
        return {"hypothesis_states": deepcopy(run["hypothesis_states"]), "query_reviews": reviews,
                "candidate_assessments": [assess_candidate(c, run["hypotheses"]) for c in response["candidates"]]}

    def contract(self, run, candidate):
        intent = candidate["inquiry_intent"]
        return {"intent": deepcopy(intent), "state_before": [deepcopy(s) for s in run["hypothesis_states"]
                if s["hypothesis_id"] in intent["hypothesis_ids"]]}


def get_protocol(version, *, decision_problem=None):
    if version == "decision-impact-inquiry-v1":
        from .inquiry_impact import DecisionImpactProtocol
        return DecisionImpactProtocol(decision_problem)
    if decision_problem is not None:
        raise ValueError("A terminal decision problem requires the decision-impact protocol")
    if version is None:
        return None
    if version == "rule-scope-inquiry-v1":
        from .inquiry_rule_scope import RuleScopeProtocol
        return RuleScopeProtocol()
    if version == "return-probability-inquiry-v1":
        from .inquiry_probability import ReturnProbabilityProtocol
        return ReturnProbabilityProtocol()
    if version == "information-choice-inquiry-v1":
        from .inquiry_choice import InformationChoiceProtocol
        return InformationChoiceProtocol()
    if version == "evidence-state-inquiry-v1":
        from .inquiry_evidence_state import EvidenceStateProtocol
        return EvidenceStateProtocol()
    if version == "return-forecast-inquiry-v1":
        from .inquiry_returns import ReturnForecastProtocol
        return ReturnForecastProtocol()
    if version == "applicability-inquiry-v1":
        from .inquiry_applicability import ApplicabilityProtocol
        return ApplicabilityProtocol()
    if version != VERSION:
        raise ValueError("Unknown inquiry protocol version")
    return HypothesisProtocol()
