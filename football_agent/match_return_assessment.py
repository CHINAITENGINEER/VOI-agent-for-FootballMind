"""Provisional actor review of an actual return, not entailment or VoI gold."""
from copy import deepcopy
from jsonschema import validate
from .audit import stored_digest as digest

VERSION = "match-return-assessment-v1"
SELECTION_EVIDENCE_VERSION = "current-return-selection-evidence-v1"
FIELD = "return_assessment"
CAPABILITY_FIELD = "return_assessment_capability"
PENDING_FIELD = "pending_return_assessment"
HISTORY_FIELD = "question_return_assessments"
SEMANTIC_OUTCOMES = {"resolved", "partial", "irrelevant", "conflicting"}


def capability():
    return {"version": VERSION, "judgement": "ACTOR_PROVISIONAL",
            "extra_model_calls": 0, "numeric_voi_calibrated": False}


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1, "maxLength": 1200}
ASSESSMENT_SCHEMA = _object({
    "acquisition_id": TEXT, "question_id": TEXT, "context_digest": TEXT,
    "outcome": {"type": "string", "enum": ["resolved", "partial", "irrelevant",
        "conflicting", "insufficient_context", "unavailable", "discovery_only"]},
    "reason": TEXT,
    "citations": {"type": "array", "maxItems": 4, "items": _object({
        "document_id": TEXT, "quote": {"type": "string", "minLength": 1, "maxLength": 600}})},
    "remaining_unknowns": {"type": "array", "maxItems": 8, "items": TEXT},
})

PROMPT = """
ACTUAL RETURN ASSESSMENT CONTRACT:
User supplies a match; you formulate the questions and choose information.
Return return_assessment=null when pending_return_assessment is null. Otherwise
assess that ONE actual acquisition in this same planning call, before proposing
the next action. Echo its acquisition_id, question.id and context_digest exactly.
Judge the ORIGINAL question, unknown_to_resolve and useful_if in that context;
do not rewrite its test using hindsight or a revised analysis_plan. The next
plan may change, but that does not change the question asked by the prior action.
resolved means the visible source addresses that question, not verified truth.
partial/irrelevant/conflicting require exact quotations from the returned
document excerpt; conflicting requires two distinct quotations. Explain how the
return meets or fails useful_if. Never cite hidden text or a different document.
Discovery metadata can only be discovery_only or insufficient_context; HTTP
failure is unavailable or insufficient_context, never evidence of no event.
Empty or unquotable content is insufficient_context. Every outcome except
resolved retains remaining_unknowns; resolved has none. These are provisional
source judgements, not independent labels, measured risk reductions or numerical
VoI. Stopping may retain unknowns. Do not invent counterfactual returns. Prior
return assessments are unverified actor history, not new facts.
"""


def extend_schema(schema):
    result = deepcopy(schema)
    result["properties"][FIELD] = {"anyOf": [deepcopy(ASSESSMENT_SCHEMA), {"type": "null"}]}
    result["required"].append(FIELD)
    return result


def ensure_capability(config):
    if CAPABILITY_FIELD in config and config[CAPABILITY_FIELD] != capability():
        raise ValueError("RETURN_ASSESSMENT_CAPABILITY_CHANGED")


def bind_return(selected, selection_ref, acquisition_id, result):
    """Derive context only from accepted action and actual return."""
    questions = [q for q in selected["analysis_plan"]
                 if q["id"] == selected["active_question_id"]]
    if len(questions) != 1 or not selection_ref:
        raise ValueError("RETURN_REQUIRES_UNIQUE_PROSPECTIVE_QUESTION")
    document = result.get("document")
    context = {
        "version": VERSION, "acquisition_id": acquisition_id,
        "selection_operation_id": selection_ref["operation_id"],
        "selection_decision_digest": selection_ref["decision_digest"],
        "question": deepcopy(questions[0]),
        "unknown_to_resolve": selected["unknown_to_resolve"],
        "useful_if": selected["useful_if"],
        "action": selected["action"],
        "result_digest": digest({k: v for k, v in result.items() if k != "return_context"}),
        "document_id": document["document_id"] if document else None,
        "return_observation": deepcopy(document.get("return_observation")) if document else None,
    }
    context["context_digest"] = digest(context)
    return context


def validate_assessment(response, public):
    ensure_capability(public)
    context, assessment = public.get(PENDING_FIELD), response[FIELD]
    if context is None:
        if assessment is not None:
            raise ValueError("RETURN_ASSESSMENT_WITHOUT_ACTUAL_RETURN")
        return
    if assessment is None:
        raise ValueError("ACTUAL_RETURN_ASSESSMENT_REQUIRED")
    validate(assessment, ASSESSMENT_SCHEMA)
    if context["context_digest"] != digest({k: v for k, v in context.items() if k != "context_digest"}):
        raise ValueError("RETURN_CONTEXT_DIGEST_MISMATCH")
    if (assessment["acquisition_id"] != context["acquisition_id"]
            or assessment["question_id"] != context["question"]["id"]
            or assessment["context_digest"] != context["context_digest"]):
        raise ValueError("RETURN_ASSESSMENT_CONTEXT_MISMATCH")
    docs = [d for d in public["documents"] if d["document_id"] == context["document_id"]]
    doc = docs[0] if len(docs) == 1 else None
    if context["document_id"] is not None and (doc is None
            or doc.get("acquisition_id") != context["acquisition_id"]
            or doc.get("selection_operation_id") != context["selection_operation_id"]
            or doc.get("return_observation") != context["return_observation"]):
        raise ValueError("RETURN_ASSESSMENT_DOCUMENT_MISMATCH")
    readable = bool(doc and doc.get("status") == "readable" and doc.get("excerpt", "").strip())
    discovery = bool(doc and doc.get("content_kind") == "discovery")
    surface = (context.get("return_observation") or {}).get("evidence_surface")
    body_quotable = readable and surface in {"body_text", "structured_records", "local_structured_records"}
    outcome, quotes = assessment["outcome"], []
    for citation in assessment["citations"]:
        quote = citation["quote"]
        if (not body_quotable or discovery or citation["document_id"] != context["document_id"]
                or not quote.strip() or doc["excerpt"].count(quote) != 1):
            raise ValueError("RETURN_ASSESSMENT_QUOTE_NOT_VISIBLE_UNIQUE_BODY")
        if quote in quotes:
            raise ValueError("RETURN_ASSESSMENT_DUPLICATE_QUOTE")
        quotes.append(quote)
    if outcome in SEMANTIC_OUTCOMES and (not body_quotable or discovery or not quotes):
        raise ValueError("RETURN_ASSESSMENT_REQUIRES_BODY_CITATION")
    if outcome == "conflicting" and len(quotes) < 2:
        raise ValueError("RETURN_ASSESSMENT_CONFLICT_REQUIRES_TWO_SPANS")
    if outcome == "discovery_only" and (not discovery or not readable):
        raise ValueError("RETURN_ASSESSMENT_NOT_DISCOVERY")
    if outcome == "unavailable" and doc and doc.get("status") in {"readable", "empty_text", "empty_structured"}:
        raise ValueError("RETURN_ASSESSMENT_NOT_UNAVAILABLE")
    if (outcome == "resolved") != (len(assessment["remaining_unknowns"]) == 0):
        raise ValueError("RETURN_ASSESSMENT_UNKNOWNS_MISMATCH")
    if not assessment["reason"].strip() or any(not x.strip() for x in assessment["remaining_unknowns"]):
        raise ValueError("RETURN_ASSESSMENT_BLANK_REASON_OR_UNKNOWN")


def assessment_record(public, response, operation_id):
    validate_assessment(response, public)
    if response[FIELD] is None:
        return None
    return {"version": VERSION, "assessment_operation_id": operation_id,
            "decision_digest": digest(response), "context": deepcopy(public[PENDING_FIELD]),
            "assessment": deepcopy(response[FIELD]),
            "label_status": "ACTOR_PROVISIONAL_NOT_GOLD",
            "world_truth_verified": False, "independent_semantic_validation": False,
            "realized_decision_gain": None, "numeric_voi_training_eligible": False}


def current_return_evidence(public, response):
    """View the pending return's current review without committing it as history.

    Candidate validation and selection share this planning call. The previous
    public history therefore does not contain this review yet. Preserve its
    original question and return binding, and keep it explicitly provisional.
    Legacy tasks and tasks without a pending return expose no current review.
    """
    if CAPABILITY_FIELD not in public:
        return None
    if response.get("task_id") != public.get("task_id"):
        raise ValueError("CURRENT_RETURN_TASK_MISMATCH")
    if FIELD not in response:
        raise ValueError("CURRENT_RETURN_ASSESSMENT_MISSING")
    validate_assessment(response, public)
    if response[FIELD] is None:
        return None
    return {
        "version": SELECTION_EVIDENCE_VERSION,
        "context": deepcopy(public[PENDING_FIELD]),
        "assessment": deepcopy(response[FIELD]),
        "label_status": "ACTOR_PROVISIONAL_NOT_GOLD",
        "world_truth_verified": False,
        "independent_semantic_validation": False,
        "realized_decision_gain": None,
        "numeric_voi_training_eligible": False,
    }
