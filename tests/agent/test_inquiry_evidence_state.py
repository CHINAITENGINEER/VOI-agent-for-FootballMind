"""Controlled evidence interventions for acquisition state, not forecast gains."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_applicability import project_requirements as legacy_requirements
from football_agent.inquiry_evidence_state import EvidenceStateProtocol, project_requirements, assess_requirements
from football_agent.inquiry_returns import OUTCOMES
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_verify import verify
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 26, 15, tzinfo=timezone.utc)
H = {"hypothesis_id": "h1", "question_id": "q1", "statement": "Player K starts in match one.",
     "hypothesis_type": "starting_lineup", "subject": "Team-A", "scope": "match", "scope_key": "match-one",
     "required_evidence": ["official_lineup"]}
H2 = {**H, "hypothesis_id": "h2", "question_id": "q2", "statement": "Player J is available for match one.",
      "hypothesis_type": "availability", "required_evidence": ["injury_update"]}
TEXT = {"supports": "Player K will start in match one.", "challenges": "Player K will not start in match one.",
        "context": "Player K is in the squad for the campaign."}


def claim(relation, *, doc="doc-one", kind=None):
    return {"claim_id": "c-"+relation, "question_id": "q1", "text": TEXT[relation], "status": "source_claim",
            "citations": [{"document_id": doc, "quote": TEXT[relation]}],
            "evidence": {"kind": kind or ("roster" if relation == "context" else "official_lineup"),
                         "subject": "Team-A", "scope": "match", "scope_key": "match-one"},
            "hypothesis_links": [{"hypothesis_id": "h1", "relation": relation}]}


def response(kind, *, target="h1", endpoint="bulletin", claims=None):
    hypothesis = H if target == "h1" else H2
    evidence = hypothesis["required_evidence"]
    candidate = {"candidate_id": "next", "question_id": hypothesis["question_id"], "tool": "read_url",
        "url": "https://example.com/"+endpoint, "query": None, "rationale": "Follow the computed information state.",
        "answer_could_change": "Address a missing input or examine opposing statements.",
        "inquiry_intent": {"hypothesis_ids": [target], "purpose": "resolve", "expected_evidence": evidence,
            "subject": "Team-A", "scope": "match", "scope_key": "match-one", "supports_change": "Reuse the answer.",
            "challenges_change": "Reuse the negative answer.", "no_answer_change": "Keep the input missing."},
        "return_model": {"branches": [{"outcome": outcome,
            "missing_after": [{"hypothesis_id": target, "types": [] if outcome in OUTCOMES[:3] else evidence}],
            "next_action": {"kind": "reassess", "target_hypothesis_ids": [], "evidence_types": [],
                            "reason": "Inspect any remaining question."}} for outcome in OUTCOMES]}}
    return {"questions": [{"question_id": h["question_id"], "question": h["statement"],
        "decision_relevance": "Personnel evidence for the same match.", "answer_could_change": "Choose the next query.",
        "based_on_documents": []} for h in (H, H2)], "hypotheses": [deepcopy(H), deepcopy(H2)],
        "claim_updates": claims or [], "context_bindings": [], "candidates": [candidate] if kind == "query" else [],
        "decision": {"kind": kind, "candidate_id": "next" if kind == "query" else None,
                     "reason": "Use the available state rather than seeking a particular answer.", "revisit_at": None}}


@pytest.mark.parametrize("relation,direction", [("supports", "supports"), ("challenges", "challenges")])
def test_either_direction_supplies_an_input_without_certifying_truth(relation, direction):
    item = project_requirements([H], [claim(relation)], [])[0]
    assert item["missing_types"] == [] and item["present_types"] == ["official_lineup"]
    assert item["proposition_direction"] == direction and item["conflict_status"] == "not_observed"
    assert item["source_answer_status"] == ("source_support" if relation == "supports" else "source_challenge")
    assert item["semantic_verification"] == "unverified" and item["expected_gain"] is None
    if relation == "challenges":
        assert legacy_requirements([H], [claim(relation)], [])[0]["missing_types"] == ["official_lineup"]


def test_opposed_claims_leave_the_input_present_and_request_conflict_review():
    item = project_requirements([H], [claim("supports"), claim("challenges", doc="doc-two")], [])[0]
    assert item["missing_types"] == [] and item["prerequisites_complete"]
    assert item["contested_types"] == ["official_lineup"] and item["conflict_status"] == "opposed_source_claims"
    assert item["query_focus"] == "review_opposed_sources"
    assert item["slots"][0]["status"] == "present" and len(item["slots"][0]["contested_claim_ids"]) == 2


@pytest.mark.parametrize("mismatch", ["subject", "scope", "scope_key", "kind", "inference"])
def test_negative_evidence_does_not_bypass_type_scope_or_source_checks(mismatch):
    item = claim("challenges")
    if mismatch == "inference": item["status"] = "hypothesis"
    else: item["evidence"][mismatch] = "other" if mismatch == "kind" else "wrong"
    state = project_requirements([H], [item], [])[0]
    assert state["missing_types"] == ["official_lineup"] and state["proposition_direction"] == "undetermined"


def test_context_can_supply_an_input_without_answering_the_proposition():
    state = project_requirements([H], [claim("context", kind="official_lineup")], [])[0]
    assert not state["missing_types"] and state["source_answer_status"] == "inference_required"
    assert state["query_focus"] == "reason_from_available_inputs"


def test_partial_negative_evidence_is_not_a_complete_multi_input_answer():
    h = {**H, "hypothesis_type": "actual_rotation", "required_evidence": ["official_lineup", "lineup_comparison"]}
    state = project_requirements([h], [claim("challenges")], [])[0]
    assert state["missing_types"] == ["lineup_comparison"] and state["source_answer_status"] == "inputs_incomplete"
    context = claim("context", kind="lineup_comparison")
    state = project_requirements([h], [claim("challenges"), context], [])[0]
    assert not state["missing_types"] and state["source_answer_status"] == "inference_required"


def test_bound_standings_still_need_effective_time_in_either_direction():
    h = {**H, "hypothesis_type": "qualification", "required_evidence": ["standings"]}
    item = claim("challenges", kind="standings")
    item["evidence"].update(scope="campaign", scope_key="campaign-one")
    link = {"binding_id": "b1", "hypothesis_id": "h1", "claim_id": item["claim_id"], "information_as_of": None}
    assert project_requirements([h], [item], [link])[0]["missing_types"] == ["standings"]
    link["information_as_of"] = NOW.isoformat()
    state = project_requirements([h], [item], [link])[0]
    assert not state["missing_types"] and state["proposition_direction"] == "undetermined"
    assert state["source_answer_status"] == "inference_required"
    item["hypothesis_links"] = []
    assert project_requirements([h], [item], [link])[0]["present_types"] == ["standings"]


def test_requery_is_explained_as_existing_input_not_blocked_or_given_zero_value():
    claims = [claim("challenges")]
    states = project_requirements([H], claims, [])
    candidate = response("query")["candidates"][0]
    candidate["inquiry_intent"]["purpose"] = "refresh"
    assessment = assess_requirements(candidate, [H], claims, [], states)
    target = assessment["targets"][0]
    assert target["potential_missing_fills"] == [] and target["already_present_types"] == ["official_lineup"]
    assert target["source_answer_status"] == "source_challenge" and assessment["expected_gain"] is None


@pytest.mark.parametrize("relations,expected", [(["supports"], "injury"), (["challenges"], "injury"),
    (["supports", "challenges"], "confirmation"), (["context"], "lineup")])
def test_full_workflow_reads_computed_state_before_selecting_next_tool(tmp_path, relations, expected):
    calls, seen = [], []
    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            return SimpleNamespace(content=("<main>"+" ".join(TEXT[r] for r in relations)+"</main>").encode(),
                                   status_code=200, headers={"content-type": "text/html"}, close=lambda: None)
        def close(self): pass
    class Provider:
        def configuration(self): return {"provider": "constructed_state_policy", "configured": True}
        def decide(self, state):
            seen.append(deepcopy(state))
            if len(seen) == 1: answer = response("query")
            elif len(seen) == 2:
                answer = response("reflect", claims=[claim(r, doc=state["documents"][0]["document_id"]) for r in relations])
            elif len(seen) == 3:
                first = state["requirements"][0]
                endpoint = ("confirmation" if first["conflict_status"] == "opposed_source_claims" else
                            "lineup" if first["missing_types"] else "injury")
                answer = response("query", target="h2" if endpoint == "injury" else "h1", endpoint=endpoint)
            else: answer = response("defer")
            return {"response": answer, "receipt": {"real_provider_call": False}}
    tools = OpenWebTools(RawArchive(tmp_path/"evidence"), ["example.com"], clock=lambda: NOW, session_factory=Session)
    workflow = InquiryWorkflow(tmp_path/"runs", Provider(), tools, clock=lambda: NOW, protocol=EvidenceStateProtocol())
    task = {"task_id": "constructed-state-intervention", "objective": "Resolve personnel unknowns using acquired evidence.",
        "deadline": (NOW+timedelta(hours=1)).isoformat(), "initial_facts": [],
        "seed_urls": ["https://example.com/"+name for name in ["bulletin", "lineup", "injury", "confirmation"]]}
    run = workflow.run(workflow.start(task, max_requests=2, max_llm_calls=4)["run_id"])
    assert len(calls) == 2 and calls[-1].endswith("/"+expected)
    assert run["state"] == "DEFERRED" and not run["pending"]
    reflected = next(e for e in run["events"] if e["type"] == "inquiry_reflection_completed")
    assert seen[2]["requirements"] == reflected["snapshot"]["requirements"]
    proof = verify(tmp_path, run["run_id"])
    assert proof["raw_observations_verified"] == 2 and proof["new_network_requests"] == 0
    assert len(calls) == 2 and proof["real_completed_llm_turns"] == 0
