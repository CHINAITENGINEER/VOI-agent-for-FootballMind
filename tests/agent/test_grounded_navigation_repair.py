from copy import deepcopy
from datetime import timedelta
import pytest
from football_agent.grounded_navigation_repair import RepairableGroundedMemoryWorkflow
from football_agent.grounded_navigation_repair_verify import verify_repaired_grounded_episode
from football_agent.inquiry_contract import utc
from test_grounded_navigation_episode import setup
from test_candidate_grounded_navigation import convert
from test_target_navigation import choice
from test_information_navigation import rechain_constructed


def invalid(public):
    response=convert(choice(public));response['task_contract_digest']='mistyped-digest'
    return response


def prepare(flow,rid):
    repaired=RepairableGroundedMemoryWorkflow(flow.store.root.parent,flow.provider,flow.tools,clock=flow.clock)
    run=flow.get(rid)
    reg={'run_id':rid,'parent_head_hash':run['events'][-1]['event_hash'],'configuration':repaired.configuration()}
    return repaired,reg


def test_repair_retains_grounded_memory_cost_and_deadline(tmp_path,monkeypatch):
    flow,rid,seen,handoff,parent,parent_reg=setup(tmp_path,monkeypatch,[invalid,lambda p:convert(choice(p))])
    old=flow.step(rid);repaired,reg=prepare(flow,rid)
    admitted=repaired.reopen(rid,reg)
    assert old['reason']=='INVALID_NAVIGATION_RESPONSE' and admitted['budget']==old['budget']
    assert admitted['task']==old['task'] and admitted['target_contract']==old['target_contract']
    assert admitted['response_validation_feedback']['errors'][0]['detail']=='Response changed task contract'
    run=repaired.step(rid)
    assert run['reason']=='AGENT_DEFER' and run['budget']['llm_calls']==2 and run['budget']['http_attempts']==0
    assert seen[1]['prior_information_tests']==handoff['actor_return_assessments']
    assert seen[1]['response_validation_feedback']['budget_refunded'] is False
    proof=verify_repaired_grounded_episode(tmp_path/'new',rid,parent_reg,reg)
    assert proof['response_repairs']==1 and proof['completed_model_returns']==2
    assert repaired.step(rid)==run


def test_second_repair_is_not_admitted(tmp_path,monkeypatch):
    flow,rid,seen,handoff,parent,parent_reg=setup(tmp_path,monkeypatch,[invalid,invalid])
    flow.step(rid);repaired,reg=prepare(flow,rid);repaired.reopen(rid,reg);run=repaired.step(rid)
    assert run['reason']=='INVALID_NAVIGATION_RESPONSE'
    with pytest.raises(ValueError,match='One explicit repair'):
        repaired.reopen(rid,reg)
    assert repaired.get(rid)==run


def test_repair_keeps_original_deadline(tmp_path,monkeypatch):
    flow,rid,seen,handoff,parent,parent_reg=setup(tmp_path,monkeypatch,[invalid])
    old=flow.step(rid);repaired,reg=prepare(flow,rid)
    repaired.clock=lambda:utc(old['task']['deadline'])+timedelta(seconds=1)
    with pytest.raises(ValueError,match='Original deadline expired'):
        repaired.reopen(rid,reg)
    assert repaired.get(rid)==old


def test_replay_rejects_invented_repair_feedback(tmp_path,monkeypatch):
    flow,rid,seen,handoff,parent,parent_reg=setup(tmp_path,monkeypatch,[invalid])
    flow.step(rid);repaired,reg=prepare(flow,rid);run=repaired.reopen(rid,reg)
    forged=deepcopy(run);event=forged['events'][-1]
    event['data']['feedback']['errors'][0]['detail']='query the reviewer-selected page'
    rechain_constructed(repaired,forged)
    with pytest.raises(ValueError,match='feedback differs'):
        verify_repaired_grounded_episode(tmp_path/'new',rid,parent_reg,reg)
