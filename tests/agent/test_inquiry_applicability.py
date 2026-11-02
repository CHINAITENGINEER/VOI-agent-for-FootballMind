"""Constructed scope and acquisition interventions, not football efficacy data."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_applicability import ApplicabilityProtocol, assess_requirements, project_requirements
from football_agent.inquiry_hypotheses import project
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_verify import verify
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 26, 15, tzinfo=timezone.utc)
TASK = {"task_id": "constructed-qualification", "objective": "Determine which qualification input is still needed.",
        "deadline": (NOW + timedelta(hours=1)).isoformat(), "initial_facts": [],
        "seed_urls": ["https://example.com/rules", "https://example.com/standings"]}
H = {"hypothesis_id": "h", "question_id": "q", "statement": "The result can change qualification.",
     "hypothesis_type": "qualification", "subject": "Team-A", "scope": "match", "scope_key": "match-one",
     "required_evidence": ["competition_rules", "standings"]}
MEMBERSHIP = "Team A's match-one belongs to campaign-one."
RULES = "Two teams advance from the group."
STANDINGS = "The standings are tied at zero points."
TIME = "Standings effective at 2026-09-26T14:00:00Z."


def source_claim(kind, document_id="doc-one"):
    return {"claim_id": kind, "question_id": "q", "text": "An input to qualification reasoning, not the conclusion.",
            "status": "source_claim", "citations": [{"document_id": document_id, "quote": RULES if kind == "competition_rules" else STANDINGS}],
            "evidence": {"kind": kind, "subject": "Team-A", "scope": "campaign", "scope_key": "campaign-one"},
            "hypothesis_links": [{"hypothesis_id": "h", "relation": "context"}]}


def binding(kind, document_id="doc-one"):
    return {"binding_id": "b-"+kind, "hypothesis_id": "h", "claim_id": kind,
            "relation": "competition_rule_applies_to_match" if kind == "competition_rules" else "standings_describe_match_context",
            "explanation": "The source explicitly identifies the match's campaign.",
            "witnesses": [{"document_id": document_id, "quote": MEMBERSHIP}],
            "information_as_of": None if kind == "competition_rules" else "2026-09-26T14:00:00Z",
            "time_witnesses": [] if kind == "competition_rules" else [{"document_id": document_id, "quote": TIME}]}


def answer(kind="query", evidence_kind="competition_rules", claims=None, bindings=None):
    endpoint = "rules" if evidence_kind == "competition_rules" else "standings"
    return {"questions": [{"question_id": "q", "question": "What remains needed for qualification reasoning?",
                "decision_relevance": "Identify the unfilled input.", "answer_could_change": "Select rules or standings next.",
                "based_on_documents": []}], "hypotheses": [deepcopy(H)], "claim_updates": claims or [],
            "context_bindings": bindings or [],
            "candidates": [{"candidate_id": "c", "question_id": "q", "tool": "read_url",
                "url": "https://example.com/"+endpoint, "query": None, "rationale": "Fill the remaining input.",
                "answer_could_change": "Permit further reasoning or preserve an explicit gap.",
                "inquiry_intent": {"hypothesis_ids": ["h"], "purpose": "resolve", "expected_evidence": [evidence_kind],
                    "subject": "Team-A", "scope": "campaign", "scope_key": "campaign-one",
                    "supports_change": "Reassess input coverage.", "challenges_change": "Keep opposing evidence.",
                    "no_answer_change": "Leave the gap open."}}] if kind == "query" else [],
            "decision": {"kind": kind, "candidate_id": "c" if kind == "query" else None,
                         "reason": "Choose the next action from the available evidence.", "revisit_at": None}}


def public():
    return {"task": TASK, "now": NOW.isoformat(), "questions": [], "claims": [], "hypotheses": [],
            "context_bindings": [], "documents": [{"document_id": "doc-one", "kind": "page", "status": "OK",
                "text": ' '.join((RULES, STANDINGS, MEMBERSHIP, TIME))}]}


def test_scoped_rule_fills_a_prerequisite_without_asserting_qualification():
    claims = [source_claim("competition_rules")]
    assert project_requirements([H], claims, [])[0]["missing_types"] == ["competition_rules", "standings"]
    result = project_requirements([H], claims, [binding("competition_rules")])[0]
    assert result["present_types"] == ["competition_rules"] and result["missing_types"] == ["standings"]
    assert result["conclusion_status"] == "missing_prerequisites"
    assert project([H], claims)[0]["status"] == "context_only"
    candidate = answer(evidence_kind="standings")["candidates"][0]
    feedback = assess_requirements(candidate, [H], claims, [binding("competition_rules")], [result])
    assert feedback["targets"][0]["potential_missing_fills"] == ["standings"]
    assert feedback["targets"][0]["could_complete_inputs"] is True
    candidate = answer()["candidates"][0]
    feedback = assess_requirements(candidate, [H], claims, [binding("competition_rules")], [result])
    assert feedback["targets"][0]["already_present_types"] == ["competition_rules"]
    assert feedback["expected_gain"] is None


def test_all_inputs_present_still_requires_inference_and_standings_time_is_not_retrieval_time():
    claims = [source_claim(k) for k in ("competition_rules", "standings")]
    bindings = [binding(k) for k in ("competition_rules", "standings")]
    result = project_requirements([H], claims, bindings)[0]
    assert result["prerequisites_complete"] and result["conclusion_status"] == "inference_required"
    assert result["semantic_verification"] == "unverified" and result["expected_gain"] is None
    bindings[1].update(information_as_of=None, time_witnesses=[])
    result = project_requirements([H], claims, bindings)[0]
    assert result["missing_types"] == ["standings"] and result["slots"][1]["status"] == "time_unresolved"


@pytest.mark.parametrize("mistake", ["unacquired", "invented_quote", "subject", "coaching", "future_time", "immutable"])
def test_invalid_links_and_future_effective_time_are_rejected(mistake):
    kind = "standings" if mistake == "future_time" else "competition_rules"
    item, link, state = source_claim(kind), binding(kind), public()
    if mistake == "unacquired": link["witnesses"][0]["document_id"] = "future-doc"
    if mistake == "invented_quote": link["witnesses"][0]["quote"] = "A fabricated membership witness"
    if mistake == "subject": item["evidence"]["subject"] = "Team-B"
    if mistake == "coaching": item["evidence"]["kind"] = "coach_statement"
    if mistake == "future_time": link["information_as_of"] = "2026-09-27T00:00:00Z"
    if mistake == "immutable":
        state["context_bindings"] = [deepcopy(link)]
        link["explanation"] = "Changed interpretation."
    with pytest.raises(ValueError):
        ApplicabilityProtocol().validate(answer("reflect", claims=[item], bindings=[link]), state)


class Provider:
    def __init__(self, responses):
        self.responses, self.seen = responses, []
    def configuration(self):
        return {"provider": "constructed", "configured": True}
    def decide(self, state):
        self.seen.append(deepcopy(state))
        response = self.responses[len(self.seen)-1]
        return {"response": response(state) if callable(response) else deepcopy(response),
                "receipt": {"real_provider_call": False}}


def build(root, responses, *, stamp=NOW):
    calls = []
    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            text = (RULES if url.endswith('/rules') else STANDINGS + ' ' + TIME) + ' ' + MEMBERSHIP
            return SimpleNamespace(content=('<main>'+text+'</main>').encode(), status_code=200,
                                   headers={"content-type":"text/html"}, close=lambda:None)
        def close(self):
            pass
    provider = Provider(responses)
    tools = OpenWebTools(RawArchive(root/'evidence'), ['example.com'], clock=lambda:stamp, session_factory=Session)
    return InquiryWorkflow(root/'runs', provider, tools, clock=lambda:stamp, protocol=ApplicabilityProtocol()), provider, calls


@pytest.mark.parametrize("inherited_kind", ["competition_rules", "standings"])
def test_reused_evidence_changes_next_query_after_computed_reflection(tmp_path, inherited_kind):
    parent_root, child_root = tmp_path/'parent', tmp_path/'child'
    parent, _, parent_calls = build(parent_root, [answer(evidence_kind=inherited_kind), answer("defer")])
    prior = parent.run(parent.start(TASK)["run_id"])
    needed = "standings" if inherited_kind == "competition_rules" else "competition_rules"
    def extract_existing(state):
        assert len(state['documents']) == 1 and state['budget']['remaining_requests'] == 1
        doc_id = state['documents'][0]['document_id']
        return answer('reflect', claims=[source_claim(inherited_kind,doc_id)], bindings=[binding(inherited_kind,doc_id)])
    def choose_from_projected_state(state):
        missing = state['requirements'][0]['missing_types']
        assert missing == [needed]
        assert state['inherited_evidence'][0]['parent_run_id'] == prior['run_id']
        return answer(evidence_kind=missing[0])
    def after_new_return(state):
        doc_id = state['documents'][-1]['document_id']
        return answer('defer', claims=[source_claim(needed,doc_id)], bindings=[binding(needed,doc_id)])
    child, provider, child_calls = build(child_root, [extract_existing, choose_from_projected_state, after_new_return],
                                         stamp=NOW+timedelta(minutes=10))
    run = child.start(TASK,max_requests=1,max_llm_calls=3)
    run = child.inherit_evidence(run['run_id'],parent_root=parent_root,parent_run_id=prior['run_id'],
                                parent_head_hash=prior['events'][-1]['event_hash'],document_ids=[prior['documents'][0]['document_id']])
    assert run['documents'][0]['available_at'] == prior['documents'][0]['available_at']
    reflected = child.step(run['run_id'])
    assert reflected['state']=='READY' and reflected['budget']['http_attempts']==0 and not child_calls
    final = child.run(run['run_id'])
    assert final['requirements'][0]['prerequisites_complete']
    assert final['hypothesis_states'][0]['status']=='context_only'
    assert len(parent_calls)==len(child_calls)==1
    assert child_calls[0].endswith('/standings' if needed=='standings' else '/rules')
    review=final['query_reviews'][0]
    assert review['requirements_before'][0]['missing_types']==[needed]
    assert review['requirements_after'][0]['missing_types']==[] and review['counterfactual_gain'] is None
    proof=verify(child_root,final['run_id'])
    assert proof['inherited_documents_verified']==1 and proof['raw_observations_verified']==1
    assert proof['new_network_requests']==0 and len(child_calls)==1
    assert provider.seen[0]['documents'][0]['available_at']==prior['documents'][0]['available_at']
    with pytest.raises(ValueError,match='once before'):
        child.inherit_evidence(final['run_id'],parent_root=parent_root,parent_run_id=prior['run_id'],
                              parent_head_hash=prior['events'][-1]['event_hash'],document_ids=[prior['documents'][0]['document_id']])


def test_reflection_cannot_consume_unbounded_calls_without_new_state():
    state=public()
    state['hypotheses']=[deepcopy(H)]
    with pytest.raises(ValueError,match='Reflection requires new'):
        ApplicabilityProtocol().validate(answer('reflect'),state)


def test_inheritance_rejects_unpinned_parent_and_future_evidence(tmp_path):
    from football_agent.inquiry_memory import load_inherited
    parent,_,calls=build(tmp_path/'parent',[answer(),answer('defer')])
    prior=parent.run(parent.start(TASK)['run_id'])
    options={'root':tmp_path/'parent','run_id':prior['run_id'],'expected_head':prior['events'][-1]['event_hash'],
             'document_ids':[prior['documents'][0]['document_id']],'now':NOW,'deadline':TASK['deadline'],'allowed_hosts':['example.com']}
    with pytest.raises(ValueError,match='pinned'):
        load_inherited(**{**options,'expected_head':'0'*64})
    with pytest.raises(ValueError,match='unavailable'):
        load_inherited(**{**options,'now':NOW-timedelta(minutes=1)})
    assert len(calls)==1


def test_older_document_does_not_allow_importing_a_future_parent_snapshot(tmp_path):
    from football_agent.inquiry_memory import load_inherited
    parent,_,calls=build(tmp_path/'parent',[answer(),answer('defer')])
    run=parent.step(parent.start(TASK)['run_id'])
    parent.clock=lambda: NOW+timedelta(minutes=10)
    prior=parent.run(run['run_id'])
    assert prior['documents'][0]['available_at']==NOW.isoformat()
    with pytest.raises(ValueError,match='snapshot was unavailable'):
        load_inherited(tmp_path/'parent',prior['run_id'],prior['events'][-1]['event_hash'],
                       [prior['documents'][0]['document_id']],now=NOW+timedelta(minutes=5),
                       deadline=TASK['deadline'],allowed_hosts=['example.com'])
    assert len(calls)==1
