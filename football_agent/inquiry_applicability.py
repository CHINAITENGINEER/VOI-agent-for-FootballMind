"""Evidence-bound prerequisites for autonomous inquiry across information scopes.

This projection distinguishes inputs available for reasoning from a proven
conclusion. It intentionally supplies no invented utility or confidence score.
"""
from __future__ import annotations

from copy import deepcopy

from jsonschema import Draft202012Validator

from .inquiry_contract import ID, TEXT, OPTIONAL, obj, normalized, utc
from .inquiry_hypotheses import HypothesisProtocol, SCHEMA as V1_SCHEMA, HYPOTHESIS_PROMPT, applicability


VERSION = "applicability-inquiry-v1"
CITATIONS = {"type": "array", "minItems": 1, "maxItems": 4, "items": obj({
    "document_id": ID, "quote": {"type": "string", "minLength": 8, "maxLength": 500}})}
RELATIONS = {"competition_rules": "competition_rule_applies_to_match",
             "standings": "standings_describe_match_context"}
SCHEMA = deepcopy(V1_SCHEMA)
SCHEMA["properties"]["context_bindings"] = {"type": "array", "maxItems": 6, "items": obj({
    "binding_id": ID, "hypothesis_id": ID, "claim_id": ID,
    "relation": {"enum": list(RELATIONS.values())}, "explanation": TEXT,
    "witnesses": CITATIONS, "information_as_of": OPTIONAL,
    "time_witnesses": {"type": "array", "maxItems": 4, "items": deepcopy(CITATIONS["items"])},
})}
SCHEMA["required"].append("context_bindings")
SCHEMA["properties"]["decision"]["properties"]["kind"]["enum"].append("reflect")
PROMPT = HYPOTHESIS_PROMPT + """
Applicability extension:
Evidence may be inherited from an earlier verified inquiry. Its original observation time stays
unchanged; inheritance costs zero NEW HTTP requests and is not a fresh source check. Use the
provided documents directly instead of automatically rereading them. Check whether refreshing,
independent confirmation, or a different missing input is relevant.

context_bindings links a campaign-scoped competition_rules or standings SOURCE CLAIM to a
qualification hypothesis for a particular match. Its witnesses must be exact acquired quotations
supporting that this edition/group/team context applies to this fixture. Preserve the original
claim's campaign scope. Different editions, subjects or groups need a separately justified link.
Do not link a general coaching goal, roster, or other evidence type as a match-specific intention.
Bindings are immutable and do not establish a qualification conclusion. Standings also need an
explicit information_as_of in timezone-aware ISO form with exact time_witnesses from the source;
if the effective time is absent use null and [] and retain the time gap, never substitute retrieval
time. Competition rules may have null information_as_of and no time witnesses.

requirements is a deterministic projection of prerequisite availability. It differs from
hypothesis_states, which describes direct source support for the proposition. A competition rule
can be a present prerequisite while the qualification conclusion remains unknown. All inputs
present means reasoning is possible, NOT that the proposition is true. Pay attention to
requirements.missing_types, contested types and time gaps when generating the next query.
candidate_assessments lists missing prerequisites a return might fill and inputs already present;
it is structural feedback, not estimated probability or VoI. Do not equate filling the most slots
with the greatest decision value. Refresh/challenge may remain worthwhile for an existing input.

Use decision.kind=reflect (candidate_id=null, revisit_at=null) after adding applicable bindings or
other new evidence if inspecting the computed requirement state would help choose the query.
reflect updates knowledge and returns it for the next reasoning call; it performs no HTTP request
and consumes one model call. It requires new claims, hypotheses or bindings. You can then query,
stop, wait or defer. Do not keep reflecting without new information. For known complete inputs
explain what inference is still required instead of declaring the hypothesis verified.
"""


def merge_bindings(previous, updates):
    result = {b["binding_id"]: deepcopy(b) for b in previous}
    seen = set()
    for binding in updates:
        identity = binding["binding_id"]
        if identity in seen:
            raise ValueError("Duplicate applicability binding")
        seen.add(identity)
        if identity in result and result[identity] != binding:
            raise ValueError("Applicability bindings are immutable")
        result[identity] = deepcopy(binding)
    return list(result.values())


def validate_witnesses(witnesses, documents):
    for citation in witnesses:
        doc = documents.get(citation["document_id"])
        if not doc or doc["status"] != "OK" or doc["kind"] != "page":
            raise ValueError("Applicability witness requires an acquired page")
        if normalized(citation["quote"]) not in normalized(doc["text"]):
            raise ValueError("Applicability quote absent from acquired evidence")


def project_requirements(hypotheses, claims, bindings):
    result = []
    for hypothesis in hypotheses:
        slots = []
        for kind in hypothesis["required_evidence"]:
            paths, contested, time_gaps = [], [], []
            for claim in claims:
                if claim["status"] != "source_claim" or claim["evidence"]["kind"] != kind:
                    continue
                links = [x for x in claim["hypothesis_links"] if x["hypothesis_id"] == hypothesis["hypothesis_id"]]
                if applicability(hypothesis, claim["evidence"]) == "applicable" and links:
                    if any(x["relation"] == "challenges" for x in links):
                        contested.append(claim["claim_id"])
                    elif kind == "standings":
                        # A match label alone does not establish effective time.
                        time_gaps.append({"claim_id": claim["claim_id"], "binding_id": None})
                    else:
                        paths.append({"claim_id": claim["claim_id"], "binding_id": None, "via": "same_scope"})
                for binding in bindings:
                    if binding["hypothesis_id"] != hypothesis["hypothesis_id"] or binding["claim_id"] != claim["claim_id"]:
                        continue
                    if kind == "standings" and binding["information_as_of"] is None:
                        time_gaps.append({"claim_id": claim["claim_id"], "binding_id": binding["binding_id"]})
                    elif any(x["relation"] == "challenges" for x in links):
                        contested.append(claim["claim_id"])
                    else:
                        paths.append({"claim_id": claim["claim_id"], "binding_id": binding["binding_id"], "via": "explicit_applicability"})
            status = "contested" if contested else "present" if paths else "time_unresolved" if time_gaps else "missing"
            slots.append({"kind": kind, "status": status, "evidence_paths": paths,
                          "contested_claim_ids": sorted(set(contested)), "time_gaps": time_gaps})
        missing = [slot["kind"] for slot in slots if slot["status"] != "present"]
        result.append({"hypothesis_id": hypothesis["hypothesis_id"], "slots": slots,
                       "present_types": [slot["kind"] for slot in slots if slot["status"] == "present"],
                       "missing_types": missing, "prerequisites_complete": not missing,
                       "conclusion_status": "inference_required" if not missing else "missing_prerequisites",
                       "semantic_verification": "unverified", "freshness_verification": "unverified",
                       "expected_gain": None})
    return result


def assess_requirements(candidate, hypotheses, claims, bindings, requirements):
    intent = candidate["inquiry_intent"]
    states = {s["hypothesis_id"]: s for s in requirements}
    by_claim = {c["claim_id"]: c for c in claims}
    targets = []
    for h in hypotheses:
        if h["hypothesis_id"] not in intent["hypothesis_ids"]:
            continue
        matched, unbound = set(), set()
        for kind in intent["expected_evidence"]:
            if kind not in h["required_evidence"]:
                continue
            if applicability(h, {**intent, "kind": kind}) == "applicable":
                matched.add(kind)
            elif h["hypothesis_type"] == "qualification" and kind in RELATIONS and intent["subject"] == h["subject"]:
                bound = any(b["hypothesis_id"] == h["hypothesis_id"] and
                            all(by_claim[b["claim_id"]]["evidence"][k] == intent[k] for k in ("subject", "scope", "scope_key"))
                            for b in bindings)
                (matched if bound else unbound).add(kind)
        missing = set(states[h["hypothesis_id"]]["missing_types"])
        targets.append({"hypothesis_id": h["hypothesis_id"], "missing_before": sorted(missing),
                        "potential_missing_fills": sorted(matched & missing),
                        "already_present_types": sorted(matched - missing),
                        "requires_applicability_link": sorted(unbound),
                        "could_complete_inputs": bool(missing) and missing <= matched,
                        "completion_conditions": "return is relevant, anchored, time-applicable and uncontradicted"})
    return {"candidate_id": candidate["candidate_id"], "targets": targets,
            "selection_score": None, "expected_gain": None, "value_status": "OUTCOME_AND_RISK_MODEL_REQUIRED"}


class ApplicabilityProtocol(HypothesisProtocol):
    version, schema, prompt = VERSION, SCHEMA, PROMPT
    supports_reflection = True
    supports_inheritance = True
    projection_fields = ("hypotheses", "hypothesis_states", "query_reviews", "context_bindings", "requirements")
    project_requirements = staticmethod(project_requirements)
    assess_requirements = staticmethod(assess_requirements)

    def validate(self, response, public):
        Draft202012Validator(self.schema).validate(response)
        base = deepcopy(response)
        updates = base.pop("context_bindings")
        if base["decision"]["kind"] == "reflect":
            base["decision"]["kind"] = "stop"
            new_claims = {c["claim_id"] for c in base["claim_updates"]} - {c["claim_id"] for c in public["claims"]}
            new_hypotheses = {h["hypothesis_id"] for h in base["hypotheses"]} - {h["hypothesis_id"] for h in public.get("hypotheses", [])}
            new_bindings = {b["binding_id"] for b in updates} - {b["binding_id"] for b in public.get("context_bindings", [])}
            if not (new_claims or new_hypotheses or new_bindings):
                raise ValueError("Reflection requires new evidence or hypothesis structure")
        # Call the original validator with its own v1 schema, not this subclass schema.
        HypothesisProtocol().validate(base, public)
        hypotheses = {h["hypothesis_id"]: h for h in public.get("hypotheses", []) + response["hypotheses"]}
        claims = {c["claim_id"]: c for c in public["claims"] + response["claim_updates"]}
        docs = {d["document_id"]: d for d in public["documents"]}
        for binding in merge_bindings(public.get("context_bindings", []), updates):
            h, c = hypotheses.get(binding["hypothesis_id"]), claims.get(binding["claim_id"])
            if not h or not c or h["hypothesis_type"] != "qualification" or c["status"] != "source_claim":
                raise ValueError("Applicability requires a qualification hypothesis and source claim")
            evidence = c["evidence"]
            if (h["scope"] != "match" or evidence["scope"] != "campaign" or evidence["subject"] != h["subject"]
                    or RELATIONS.get(evidence["kind"]) != binding["relation"]):
                raise ValueError("Unsupported applicability scope, subject or evidence type")
            validate_witnesses(binding["witnesses"], docs)
            validate_witnesses(binding["time_witnesses"], docs)
            if binding["information_as_of"] is not None:
                if not binding["time_witnesses"] or utc(binding["information_as_of"]) > utc(public["now"]) or utc(binding["information_as_of"]) >= utc(public["task"]["deadline"]):
                    raise ValueError("Effective time needs source witnesses and must precede decision cutoff")
            elif binding["time_witnesses"]:
                raise ValueError("Time witnesses need an explicit effective time")
        return response

    def public(self, run):
        return {**super().public(run), "context_bindings": deepcopy(run.get("context_bindings", [])),
                "requirements": deepcopy(run.get("requirements", [])),
                "inherited_evidence": [{k: deepcopy(e[k]) for k in ("parent_run_id", "parent_head_hash", "document_ids", "imported_at")}
                                       for e in run.get("evidence_imports", [])]}

    def advance(self, run, response):
        run["context_bindings"] = merge_bindings(run.get("context_bindings", []), response["context_bindings"])
        assessment = super().advance(run, response)
        run["requirements"] = self.project_requirements(run["hypotheses"], run["claims"], run["context_bindings"])
        for review in assessment["query_reviews"]:
            receipt = next(r for r in run["receipts"] if r["request_id"] == review["request_id"])
            ids = review["intent"]["hypothesis_ids"]
            review["requirements_before"] = deepcopy(receipt["inquiry_contract"]["requirements_before"])
            review["requirements_after"] = [deepcopy(s) for s in run["requirements"] if s["hypothesis_id"] in ids]
            review["requirement_change_scope"] = "joint_evidence_state_not_single_query_causal_gain"
        assessment["requirements"] = deepcopy(run["requirements"])
        assessment["candidate_assessments"] = [self.assess_requirements(c, run["hypotheses"], run["claims"], run["context_bindings"], run["requirements"])
                                                for c in response["candidates"]]
        return assessment

    def contract(self, run, candidate):
        return {**super().contract(run, candidate), "requirements_before": [deepcopy(s) for s in run["requirements"]
                if s["hypothesis_id"] in candidate["inquiry_intent"]["hypothesis_ids"]]}
