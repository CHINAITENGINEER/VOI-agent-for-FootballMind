"""Bind open information choices to a fixed source-reporting decision.

The loss table is declared by the task owner, not learned. The distribution-free
ceiling is over all terminal actions and is not a query value or ranking score.
Quoted, typed source signals are still not independently verified world truth.
"""
from __future__ import annotations

from copy import deepcopy
import math

from jsonschema import Draft202012Validator

from .audit import digest
from .inquiry_contract import ID, TEXT, obj
from .inquiry_hypotheses import SCHEMA as HYPOTHESIS_SCHEMA, MINIMUM, project
from .inquiry_rule_scope import RuleScopeProtocol


VERSION = "decision-impact-inquiry-v1"
ACTIONS = ("report_supported", "report_challenged", "withhold")
STATES = ("positive", "negative")
SUPPORTED_TARGETS = {"rotation_intent", "tactical_intent", "starting_lineup"}
LOSS = obj({a: obj({s: {"type": "number", "minimum": 0} for s in STATES}) for a in ACTIONS})
PROBLEM = obj({
    "task_id": ID, "decision_id": ID, "target": deepcopy(HYPOTHESIS_SCHEMA["properties"]["hypotheses"]["items"]),
    "semantics": {"const": "source_position_not_world_truth"},
    "state_definitions": obj({s: TEXT for s in STATES}),
    "loss": LOSS, "loss_basis": TEXT,
})
EFFECT = obj({
    "route": {"enum": ["direct_evidence", "discovery", "indirect", "unknown"]},
    "could_change_to": {"type": "array", "items": {"enum": list(ACTIONS)}, "maxItems": 3, "uniqueItems": True},
    "condition": TEXT, "further_information_needed": TEXT,
})
SCHEMA = deepcopy(RuleScopeProtocol.schema)
SCHEMA["properties"]["candidates"]["items"]["properties"]["terminal_effect"] = EFFECT
SCHEMA["properties"]["candidates"]["items"]["required"].append("terminal_effect")
SEMANTICS = {
    "version": VERSION, "answer_scope": "source_reported_position_not_verified_truth",
    "extraction_semantics": "unverified", "loss_origin": "task_declared_not_empirically_calibrated",
    "upper_bound_scope": "all_actions_pointwise_ceiling_not_query_evsi_or_current_risk",
    "plan_scope": "illustrative_conditions_not_exhaustive_or_executed_terminal_policy",
    "query_ranking_by_upper_bound": False, "query_elimination_by_plan": False,
    "loss_state_coverage": "declared_two_states_not_exhaustive_of_absence_or_conflicting_sources",
    "actual_decision_utility": None, "global_open_world_stop_proven": False,
}
PROMPT = RuleScopeProtocol.prompt + """
Decision-impact extension:
The task has a FIXED decision_problem. Its full target hypothesis, including ID, statement,
subject, match and required evidence, is immutable. Include it exactly in hypotheses in the
first reply, with a corresponding question. Other hypotheses are legitimate intermediate
questions but do not replace this endpoint. The endpoint is a SOURCE-REPORTED position, not
verified real-world rotation, lineup or outcome. Never turn a source position into world truth.

The program computes terminal_state from direct, scope/type-matching source_claims linked to
that exact hypothesis: sufficient positive types -> report_supported, negative types ->
report_challenged, opposing signals or insufficient evidence -> withhold. Withhold is not NO.
Context, rules, qualification, publicity or a squad roster cannot silently answer a different
rotation/lineup question. Keep exact quotes and negative findings; do not force an answer.

For each candidate terminal_effect states what could change this reporting decision:
route is direct_evidence, discovery, indirect or unknown; could_change_to lists illustrative
possible terminal actions, condition gives the specific source evidence that would justify
them, and further_information_needed names remaining steps. Search is discovery, never direct
evidence; its links need reading. A rule or qualification route may enable later investigation
and has UNKNOWN total value, not zero. It need not precede coach/lineup evidence: choose according
to the fixed endpoint and current material, not a checklist order. [] is allowed for immediate
action changes but does not certify that the entire query or downstream route is worthless.

The declared loss table expresses task preferences, not a learned model. decision_ceiling is
max_y[L(current_action,y)-min_a L(a,y)], over ALL terminal actions. It is only a mathematical
ceiling assuming the declared states/losses; it is NOT EVSI, uncertainty, measured risk, a query
score, or a confidence level. Its increase after an answer change need not mean risk increased.
No prior, likelihood or calibrated return model is supplied. Do not invent those quantities,
multiply operational return forecasts by this ceiling, or rank queries by this common ceiling.
The loss bound covers ONLY the two declared source-reporting states; it does not claim an
exhaustive model of absent, contradictory, stale or misleading source collections.
terminal_effect is an illustrative reasoning plan, not an exhaustive return policy or a promise
to automatically publish an answer. New evidence is still processed through the existing checks.

Before acquiring, the program freezes the actual current terminal_state AFTER your new claims
have been merged. The next valid extraction compares the new source-reporting action with this
frozen state and distinguishes sole-new-document claims from mixed/old-document updates. A
changed answer is not automatically correct, causal risk reduction or realized VoI. Select useful
information, including direct routes to this decision even when another background input is missing.
"""


def validate_problem(problem):
    Draft202012Validator(PROBLEM).validate(problem)
    target = problem["target"]
    if target["hypothesis_type"] not in SUPPORTED_TARGETS:
        raise ValueError("Terminal impact currently supports direct stated intent or starting-lineup targets only")
    if set(MINIMUM[target["hypothesis_type"]]) != set(target["required_evidence"]):
        raise ValueError("Direct terminal targets require exactly their minimum evidence types")
    if target["hypothesis_type"] in {"starting_lineup", "actual_rotation", "rotation_intent", "tactical_intent", "qualification"} and target["scope"] != "match":
        raise ValueError("Decision target must use match scope")
    for row in problem["loss"].values():
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in row.values()):
            raise ValueError("Declared decision losses must be finite nonnegative numbers")
    return deepcopy(problem)


def decision_ceiling(loss, current_action):
    """Uniform upper bound on improvement of any policy, with no distribution."""
    if set(loss) != set(ACTIONS) or current_action not in ACTIONS:
        raise ValueError("A complete declared terminal action set is required")
    for row in loss.values():
        if set(row) != set(STATES) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in row.values()):
            raise ValueError("A complete finite nonnegative loss table is required")
    by_state = {s: loss[current_action][s]-min(loss[a][s] for a in ACTIONS) for s in STATES}
    return {"current_action": current_action, "all_action_improvement_ceiling": max(by_state.values()),
            "pointwise_improvement_ceiling": by_state, "expected_gain": None,
            "cost_in_loss_units": None, "used_for_selection": False,
            "scope": SEMANTICS["upper_bound_scope"]}


def terminal_state(problem, claims):
    target = problem["target"]
    state = project([target], claims)[0]
    required = set(target["required_evidence"])
    supporting = {c["kind"] for c in state["support"]}
    challenging = {c["kind"] for c in state["challenge"]}
    action = "withhold"
    if supporting and not challenging and required <= supporting:
        action = "report_supported"
    elif challenging and not supporting and required <= challenging:
        action = "report_challenged"
    return {"decision_id": problem["decision_id"], "hypothesis_id": target["hypothesis_id"],
            "action": action, "source_status": state["status"],
            "support_claim_ids": [c["claim_id"] for c in state["support"]],
            "challenge_claim_ids": [c["claim_id"] for c in state["challenge"]],
            "missing_positive_types": sorted(required-supporting),
            "missing_negative_types": sorted(required-challenging),
            "claim_ids_seen": [c["claim_id"] for c in claims],
            "semantics": "typed_source_position_not_verified_answer"}


def _review(problem, receipt, before, current, claims, effect):
    new = [c for c in claims if c["claim_id"] not in before["claim_ids_seen"]]
    sole = [c for c in new if {x["document_id"] for x in c["citations"]} == {receipt["document_id"]}]
    contribution = terminal_state(problem, sole)
    ids = set(contribution["support_claim_ids"]+contribution["challenge_claim_ids"])
    current_ids = set(current["support_claim_ids"]+current["challenge_claim_ids"])
    prior_ids = set(before["support_claim_ids"]+before["challenge_claim_ids"])
    other_new = sorted((current_ids-prior_ids)-ids)
    changed = current["action"] != before["action"]
    return {"request_id": receipt["request_id"], "document_id": receipt["document_id"],
            "decision_before": deepcopy(before), "decision_after": deepcopy(current),
            "terminal_effect_before": deepcopy(effect), "action_changed": changed,
            "sole_document_target_claim_ids": sorted(ids), "other_new_target_claim_ids": other_new,
            "attribution": "joint_or_other_evidence_change" if changed and other_new else
                "change_with_attributable_target_claims" if changed and ids else
                "unattributed_change" if changed else "no_action_change",
            "listed_action_observed": current["action"] in effect["could_change_to"],
            "condition_semantically_verified": False, "truth_independently_verified": False,
            "actual_decision_utility": None, "causal_risk_reduction": None}


class DecisionImpactProtocol(RuleScopeProtocol):
    version, schema, prompt = VERSION, SCHEMA, PROMPT
    projection_fields = (*RuleScopeProtocol.projection_fields, "terminal_state", "decision_ceiling", "decision_impact_reviews")

    def __init__(self, problem):
        self.decision_problem = validate_problem(problem)

    def validate(self, response, public):
        Draft202012Validator(self.schema).validate(response)
        if public["task"]["task_id"] != self.decision_problem["task_id"] or public.get("decision_problem") != self.decision_problem:
            raise ValueError("Fixed terminal decision belongs to a different task")
        base = deepcopy(response)
        for candidate in base["candidates"]:
            candidate.pop("terminal_effect")
        RuleScopeProtocol().validate(base, public)
        hypotheses = {h["hypothesis_id"]: h for h in public.get("hypotheses", [])+response["hypotheses"]}
        target = self.decision_problem["target"]
        if hypotheses.get(target["hypothesis_id"]) != target:
            raise ValueError("The exact fixed terminal hypothesis must remain present")
        for candidate in response["candidates"]:
            effect, intent = candidate["terminal_effect"], candidate["inquiry_intent"]
            if effect["route"] == "direct_evidence":
                if (candidate["tool"] != "read_url" or target["hypothesis_id"] not in intent["hypothesis_ids"]
                        or any(intent[k] != target[k] for k in ("subject", "scope", "scope_key"))
                        or not set(intent["expected_evidence"]) & set(target["required_evidence"])):
                    raise ValueError("Direct terminal evidence needs the exact target and an applicable source read")
            if candidate["tool"] == "search_web" and effect["route"] not in {"discovery", "unknown"}:
                raise ValueError("Search results are discovery, not terminal source evidence")
        return response

    def public(self, run):
        current = deepcopy(run.get("terminal_state", terminal_state(self.decision_problem, [])))
        return {**super().public(run), "decision_problem": deepcopy(self.decision_problem),
                "terminal_state": current, "decision_ceiling": decision_ceiling(self.decision_problem["loss"], current["action"]),
                "decision_impact_semantics": deepcopy(SEMANTICS),
                "decision_impact_reviews": deepcopy(run.get("decision_impact_reviews", []))}

    def advance(self, run, response):
        assessment = super().advance(run, response)
        current = terminal_state(self.decision_problem, run["claims"])
        run["terminal_state"] = current
        run["decision_ceiling"] = decision_ceiling(self.decision_problem["loss"], current["action"])
        seen = {r["request_id"] for r in run.get("decision_impact_reviews", [])}
        reviews = []
        for receipt in run["receipts"]:
            if receipt["request_id"] in seen:
                continue
            contract = receipt["inquiry_contract"]["terminal_contract"]
            if contract["problem_hash"] != digest(self.decision_problem):
                raise ValueError("Acquisition terminal contract belongs to another decision")
            reviews.append(_review(self.decision_problem, receipt, contract["state_before"], current,
                                   run["claims"], contract["terminal_effect"]))
        run.setdefault("decision_impact_reviews", []).extend(reviews)
        assessment["decision_impact"] = {"terminal_state": deepcopy(current),
            "ceiling": deepcopy(run["decision_ceiling"]),
            "candidate_effects": [{"candidate_id": c["candidate_id"], "terminal_effect": deepcopy(c["terminal_effect"]),
                "query_value": None, "scope": "illustrative_not_exhaustive_no_filter"} for c in response["candidates"]],
            "new_reviews": reviews, "semantics": deepcopy(SEMANTICS)}
        return assessment

    def contract(self, run, candidate):
        return {**super().contract(run, candidate), "terminal_contract": {
            "problem_hash": digest(self.decision_problem), "state_before": deepcopy(run["terminal_state"]),
            "terminal_effect": deepcopy(candidate["terminal_effect"]), "semantics": deepcopy(SEMANTICS)}}
