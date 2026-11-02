"""Separate acquired inputs, source direction and opposed claims for inquiry.

These are projections of anchored extraction, not verified truth or numerical
information value. Old protocols retain their original projection for replay.
"""
from __future__ import annotations

from .inquiry_applicability import assess_requirements as assess_legacy
from .inquiry_hypotheses import applicability, project
from .inquiry_returns import ReturnForecastProtocol, PROMPT as BASE_PROMPT


VERSION = "evidence-state-inquiry-v1"
SEMANTICS = {
    "input_availability": "matching_acquired_source_claim_independent_of_direction",
    "missing_types": "unavailable_or_effective_time_unresolved_not_opposed_claims",
    "proposition_direction": "extracted_source_support_challenge_or_both_not_truth",
    "conflict_status": "opposed_source_claims_are_review_signals_not_verified_contradictions",
    "selection_score": None,
}
PROMPT = BASE_PROMPT + """
Evidence-state extension (replaces the earlier conservative prerequisite semantics):
requirements.missing_types now means inputs not acquired/applicable or with unresolved effective
time. A directly applicable anchored counterstatement can supply a required input just as support
can. For example an explicit official statement that a player does NOT start answers that starting
question in the negative. Do not keep seeking a positive answer merely because the hypothesis was
challenged. Absence from a news page or failed acquisition is not an explicit negative statement.

Input availability, proposition direction and opposed source signals are separate. Both positive
and negative applicable source claims leave the input PRESENT, with conflict_status=opposed_source_claims
and contested_types as separate review signals. One negative claim alone is not a conflict.
Do not treat conflicting inputs as unavailable; investigate their time, identity or credibility if
that matters. Mixed directions are extracted signals, not an independently established contradiction.
Missing_support_types in legacy hypothesis_states is about support only, NOT missing information.

requirements.source_answer_status and query_focus explain the projected state. A one-sided source
answer is reusable with stated semantic/freshness limitations; it is not certified truth. Complete
prerequisites with context only still require inference. Campaign scope, source status and standings
effective-time constraints continue to apply in either direction. A typed context claim can supply
an input but does not answer the hypothesis by itself. Partial inputs are not a complete answer.

Use the state to decide the next action: acquire missing inputs, inspect opposed signals, move to
another relevant unanswered question, or stop/defer. Already present information can still justify
refresh or independent verification when you explain its decision role. No hard stop or zero-VoI
rule follows from coverage. Do not equate number of filled inputs with decision value.

For return_model.missing_after predict only missing INPUTS. A usable negative answer normally
fills its own type; even mixed evidence can leave inputs present while conflict remains. Seven
operational outcomes are coarse: completeness, scope and time can still differ within one category.
"""


def project_requirements(hypotheses, claims, bindings):
    directions = {s["hypothesis_id"]: s for s in project(hypotheses, claims)}
    result = []
    for hypothesis in hypotheses:
        identity = hypothesis["hypothesis_id"]
        slots = []
        for kind in hypothesis["required_evidence"]:
            paths, time_gaps = [], []
            for claim in claims:
                if claim["status"] != "source_claim" or claim["evidence"]["kind"] != kind:
                    continue
                relations = [link["relation"] for link in claim["hypothesis_links"] if link["hypothesis_id"] == identity]
                relation = relations[0] if relations else "context"
                candidates = []
                if relations and applicability(hypothesis, claim["evidence"]) == "applicable":
                    candidates.append({"claim_id": claim["claim_id"], "binding_id": None,
                                       "via": "same_scope", "relation": relation, "time_resolved": kind != "standings"})
                for binding in bindings:
                    if binding["hypothesis_id"] == identity and binding["claim_id"] == claim["claim_id"]:
                        candidates.append({"claim_id": claim["claim_id"], "binding_id": binding["binding_id"],
                            "via": "explicit_applicability", "relation": relation,
                            "time_resolved": kind != "standings" or binding["information_as_of"] is not None})
                for path in candidates:
                    resolved = path.pop("time_resolved")
                    (paths if resolved else time_gaps).append(path)
            positive = sorted({p["claim_id"] for p in paths if p["relation"] == "supports" and p["via"] == "same_scope"})
            negative = sorted({p["claim_id"] for p in paths if p["relation"] == "challenges" and p["via"] == "same_scope"})
            opposed = bool(positive and negative)
            slots.append({"kind": kind, "status": "present" if paths else "time_unresolved" if time_gaps else "missing",
                          "evidence_paths": paths, "time_gaps": time_gaps,
                          "supporting_claim_ids": positive, "challenging_claim_ids": negative,
                          "conflict_status": "opposed_source_claims" if opposed else "not_observed",
                          "contested_claim_ids": sorted(set(positive + negative)) if opposed else []})
        missing = [slot["kind"] for slot in slots if slot["status"] != "present"]
        direction = directions[identity]
        supported, challenged = bool(direction["support"]), bool(direction["challenge"])
        opposed = supported and challenged
        required = set(hypothesis["required_evidence"])
        direction_complete = ((supported and required <= {c["kind"] for c in direction["support"]}) or
                              (challenged and required <= {c["kind"] for c in direction["challenge"]}))
        # Directional claims with unresolved effective time remain visible, but
        # do not make a reusable answer before all declared inputs are available.
        answer = ("opposed_source_claims" if opposed else "inputs_incomplete" if missing else
                  "source_support" if supported and direction_complete else
                  "source_challenge" if challenged and direction_complete else "inference_required")
        focus = ("review_opposed_sources_and_acquire_inputs" if opposed and missing else
                 "review_opposed_sources" if opposed else "acquire_missing_inputs" if missing else
                 "reuse_source_answer_or_justify_verification" if direction_complete else "reason_from_available_inputs")
        result.append({"hypothesis_id": identity, "slots": slots,
            "present_types": [slot["kind"] for slot in slots if slot["status"] == "present"], "missing_types": missing,
            "contested_types": [slot["kind"] for slot in slots if slot["conflict_status"] == "opposed_source_claims"],
            "prerequisites_complete": not missing,
            "conclusion_status": "opposed_sources_require_review" if opposed else "missing_prerequisites" if missing else "inference_required",
            "proposition_direction": "mixed" if opposed else "supports" if supported else "challenges" if challenged else "undetermined",
            "conflict_status": "opposed_source_claims" if opposed else "not_observed",
            "directional_claim_ids": {"supports": [c["claim_id"] for c in direction["support"]],
                                      "challenges": [c["claim_id"] for c in direction["challenge"]]},
            "source_answer_status": answer, "query_focus": focus,
            "semantic_verification": "unverified", "freshness_verification": "unverified", "expected_gain": None})
    return result


def assess_requirements(candidate, hypotheses, claims, bindings, requirements):
    result = assess_legacy(candidate, hypotheses, claims, bindings, requirements)
    states = {item["hypothesis_id"]: item for item in requirements}
    for target in result["targets"]:
        state = states[target["hypothesis_id"]]
        target.update({key: state[key] for key in ("source_answer_status", "proposition_direction", "conflict_status", "query_focus")})
        target["contested_types"] = state["contested_types"]
        target["completion_conditions"] = "return supplies a relevant anchored time-applicable input; direction and conflict remain separate"
        target["selection_policy"] = "agent_decides_no_coverage_based_value_or_hard_stop"
    return result


class EvidenceStateProtocol(ReturnForecastProtocol):
    version, prompt = VERSION, PROMPT
    project_requirements = staticmethod(project_requirements)
    assess_requirements = staticmethod(assess_requirements)

    def public(self, run):
        return {**super().public(run), "evidence_state_semantics": dict(SEMANTICS)}
