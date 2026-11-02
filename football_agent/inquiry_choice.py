"""Compare agent-proposed information routes without inventing numerical value.

Catalog entries are discovery hints. Costs below are tool-path lower bounds,
not expected task costs, response probabilities, source truth or information gain.
"""
from __future__ import annotations

from copy import deepcopy

from jsonschema import Draft202012Validator

from .audit import digest
from .inquiry_contract import ID, IDS, TEXT, OPTIONAL, obj
from .inquiry_continuation import MINIMUMS, remaining
from .inquiry_evidence_state import EvidenceStateProtocol
from .inquiry_tools import OpenWebTools, public_url


VERSION = "information-choice-inquiry-v1"
CATALOG_SCHEMA = {"type": "array", "maxItems": 12, "items": obj({
    "source_id": ID, "url": {"type": "string", "minLength": 8, "maxLength": 2048},
    "publisher": TEXT, "discovery_topics": {"type": "array", "items": TEXT, "minItems": 1, "maxItems": 6},
    "limitations": TEXT})}
SCHEMA = deepcopy(EvidenceStateProtocol.schema)
SCHEMA["properties"]["candidates"]["maxItems"] = 2
SCHEMA["properties"]["choice_review"] = obj({
    "alternatives": {"type": "array", "maxItems": 2, "items": obj({
        "candidate_id": ID, "information_advantage": TEXT, "source_limitations": TEXT,
        "evidence_document_ids": IDS, "not_selected_reason": OPTIONAL})},
    "tradeoff": TEXT, "missing_routes": TEXT})
SCHEMA["required"].append("choice_review")
PROMPT = EvidenceStateProtocol.prompt + """
Information-choice extension:
The source_catalog contains configured discovery entry points, NOT acquired facts, trust scores,
available answers or an exhaustive list of useful sources. Inspect original pages before claims.
Compare up to TWO distinct executable candidate routes in the SAME current evidence state.
Generate alternatives yourself when they have a real task role; zero or one is valid. Do not invent
a second route or issue two calls to satisfy a quota. Avoid minor rewordings of the same search.
Use existing inquiry_intent and return_model to state which unknown a return might resolve and
what would still be needed. A reference/news/index page normally discovers documents; loading
an index does not itself deliver a rule or coach statement. Plan its downstream reading cost.

choice_review.alternatives covers every candidate exactly once. For each, state its distinctive
information_advantage and source_limitations. evidence_document_ids must reference acquired
documents actually informing the comparison (failure records are allowed as operational context,
never as proof a football proposition is false); use [] when reasoning only from the task or catalog.
The selected query has not_selected_reason=null. All other candidates have a substantive reason
for leaving that route unqueried now, NOT a claim about its unseen return. For stop/defer/wait/reflect,
all proposed query candidates need a not_selected_reason. tradeoff explains why THIS action now
given the alternatives, remaining resources and expected evidence role. missing_routes describes
source coverage limitations; no catalog item is a proof all remaining information has zero value.

The program publishes tool-path cost lower bounds and current continuation counts. It computes
costs for every proposed candidate; never fill in money, success rates, scalar scores or numerical
VoI. Reading an index can need more calls than a final evidence page even with the same tool.
Failure may suggest changing path; unselected routes remain unobserved. Prefer an executable
route that addresses a consequential unknown, or explicitly justify stopping, waiting or deferring.
The choice remains qualitative bounded exploration, not a calibrated utility optimum.
"""


def validate_catalog(catalog, allowed_hosts):
    Draft202012Validator(CATALOG_SCHEMA).validate(catalog)
    result = deepcopy(catalog)
    ids, urls = set(), set()
    for item in result:
        item["url"] = public_url(item["url"], allowed_hosts)
        if item["source_id"] in ids or item["url"] in urls:
            raise ValueError("Source catalog entries need distinct IDs and entry URLs")
        ids.add(item["source_id"])
        urls.add(item["url"])
    return result


def _prepared(candidates, public):
    tools = OpenWebTools(None, public["tools"]["allowed_hosts"])
    prepared = {c["candidate_id"]: tools.prepare(c, public) for c in candidates}
    if len({digest(action) for action in prepared.values()}) != len(prepared):
        raise ValueError("Choice alternatives must have distinct prepared actions")
    return prepared


def _catalog(run):
    return deepcopy(run.get("configuration", {}).get("source_catalog",
                    run.get("information_choice", {}).get("source_catalog", [])))


class InformationChoiceProtocol(EvidenceStateProtocol):
    version, schema, prompt = VERSION, SCHEMA, PROMPT
    requires_continuation = True
    supports_source_catalog = True

    def validate(self, response, public):
        Draft202012Validator(self.schema).validate(response)
        base = deepcopy(response)
        review = base.pop("choice_review")
        EvidenceStateProtocol().validate(base, public)
        _prepared(response["candidates"], public)
        ids = [c["candidate_id"] for c in response["candidates"]]
        reviews = review["alternatives"]
        if len(reviews) != len(ids) or {r["candidate_id"] for r in reviews} != set(ids):
            raise ValueError("Comparison must cover each candidate exactly once")
        docs = {d["document_id"] for d in public["documents"]}
        selected = response["decision"]["candidate_id"]
        for item in reviews:
            if set(item["evidence_document_ids"]) - docs:
                raise ValueError("Comparison cites an unacquired document")
            reason = item["not_selected_reason"]
            if item["candidate_id"] == selected:
                if reason is not None:
                    raise ValueError("Selected candidate must not have an unselected reason")
            elif not isinstance(reason, str) or not reason.strip() or len(reason) > 1600:
                raise ValueError("Every unselected route needs a bounded nonempty reason")
        return response

    def public(self, run):
        return {**super().public(run), "information_choice": {
            "version": VERSION, "source_catalog": _catalog(run),
            "tool_path_lower_bounds": deepcopy(MINIMUMS),
            "cost_scope": "minimum_counts_not_expected_cost; index_pages_can_require_additional_reads",
            "catalog_semantics": "configured_discovery_hints_not_acquired_evidence_or_availability_guarantees",
            "catalog_exhaustive": False, "numerical_value": None}}

    def advance(self, run, response):
        assessment = super().advance(run, response)
        # Production has charged this proposal. Replay starts from its public
        # pre-call input, whose remaining count includes the current proposal.
        counts = (remaining(run) if "max_requests" in run["budget"] else
                  {"source_requests": run["budget"]["remaining_requests"],
                   "llm_calls": run["budget"]["remaining_llm_calls"]-1})
        tools = run.get("tools", run.get("configuration", {}).get("tools"))
        public = {"task": run["task"], "documents": run["documents"], "tools": tools}
        actions = _prepared(response["candidates"], public)
        source_tools = OpenWebTools(None, tools["allowed_hosts"])
        catalog_keys = [(item["source_id"], digest(source_tools.prepare(
            {"tool": "read_url", "url": item["url"]}, public))) for item in _catalog(run)]
        rows = []
        for candidate in response["candidates"]:
            key = digest(actions[candidate["candidate_id"]])
            need = deepcopy(MINIMUMS[candidate["tool"]])
            matches = [identity for identity, catalog_key in catalog_keys if key == catalog_key]
            history = [{"request_id": r["request_id"], "status": r["status"]}
                       for r in run["receipts"] if r["execution_key"] == key]
            rows.append({"candidate_id": candidate["candidate_id"], "execution_key": key,
                "catalog_entry_ids": matches, "minimum_counts": need,
                "within_count_budget": counts["source_requests"] >= need["source_requests_including_current"]
                    and counts["llm_calls"] >= need["llm_calls_after_current"],
                "prior_same_action_observations": history,
                "proposed_for_execution": candidate["candidate_id"] == response["decision"]["candidate_id"],
                "current_return": "UNOBSERVED_AT_CHOICE", "selection_probability": None,
                "expected_total_cost": None, "expected_gain": None})
        assessment["information_choice"] = {"review": deepcopy(response["choice_review"]),
            "candidate_routes": rows, "remaining_after_proposal": counts,
            "distinct_prepared_actions": len(rows),
            "count_feasible_candidates": sum(r["within_count_budget"] for r in rows),
            "comparison_scope": "same_state_agent_proposal_not_causal_benefit_or_semantic_verification",
            "eligibility_scope": "count_lower_bound_only; duplicate_deadline_and_execution_checks_still_apply",
            "numerical_value": None}
        return assessment
