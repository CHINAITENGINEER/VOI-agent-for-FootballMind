from copy import deepcopy

import pytest

from football_agent.navigation_response_repair import RepairableTargetNavigationWorkflow
from test_target_navigation import make,query,read_new,finish,choice


def invalid_read(public):
    result=read_new(public)
    result['target_state'].update(status='candidate_only',candidate_label='Constructed fixture',support_claim_indices=[0])
    result['return_assessment']=None
    return result


def repair(flow,rid):
    repaired=RepairableTargetNavigationWorkflow(flow.store.root.parent,flow.provider,flow.tools,clock=flow.clock)
    old=flow.get(rid)
    registration={'run_id':rid,'parent_head_hash':old['events'][-1]['event_hash'],'configuration':repaired.configuration()}
    return repaired,registration


def test_feedback_then_agent_read_and_reconsider_preserves_all_costs(tmp_path):
    flow,rid,seen,requests,*_=make(tmp_path,[query,invalid_read,read_new,finish],max_llm_calls=4)
    flow.step(rid);old=flow.step(rid)
    assert old['reason']=='INVALID_NAVIGATION_RESPONSE'
    repaired,registration=repair(flow,rid)
    opened=deepcopy(old['opened'])
    reopened=repaired.reopen(rid,registration)
    assert reopened['budget']==old['budget'] and reopened['opened']==opened
    assert len(reopened['decisions'])==1
    repaired.step(rid);run=repaired.step(rid)
    assert run['reason']=='AGENT_DEFER' and run['budget']['llm_calls']==4
    assert run['budget']['http_attempts']==1 and run['budget']['window_reads']==1 and len(requests)==1
    assert len(seen[2]['response_validation_feedback']['errors'])==2
    assert seen[3]['response_validation_feedback'] is None
    assert repaired.step(rid)==run


def test_agent_may_defer_after_feedback_without_executing_rejected_read(tmp_path):
    flow,rid,seen,requests,*_=make(tmp_path,[query,invalid_read,choice],max_llm_calls=4)
    flow.step(rid);flow.step(rid)
    repaired,registration=repair(flow,rid)
    repaired.reopen(rid,registration)
    run=repaired.step(rid)
    assert run['reason']=='AGENT_DEFER' and run['budget']['window_reads']==0 and len(requests)==1


@pytest.mark.parametrize('case',['unknown','defer','no_capacity','wrong_head'])
def test_repair_admission_preserves_unknowns_and_capacity(tmp_path,case):
    if case in {'unknown','defer'}:
        flow,rid,*_=make(tmp_path,[TimeoutError() if case=='unknown' else choice])
        flow.step(rid)
    else:
        flow,rid,*_=make(tmp_path,[query,invalid_read],max_llm_calls=3 if case=='no_capacity' else 4,max_window_reads=2)
        flow.step(rid);flow.step(rid)
    repaired,registration=repair(flow,rid)
    if case=='wrong_head':registration['parent_head_hash']='changed'
    with pytest.raises(ValueError):repaired.reopen(rid,registration)


@pytest.mark.parametrize('tamper',[False,True])
def test_repair_replay_reconstructs_validation_feedback(tmp_path,monkeypatch,tamper):
    import football_agent.navigation_response_repair_verify as verifier
    from test_information_navigation import rechain_constructed
    flow,rid,_,_,handoff,_=make(tmp_path,[query,invalid_read,read_new,finish],max_llm_calls=4)
    flow.step(rid);flow.step(rid)
    repaired,registration=repair(flow,rid)
    repaired.reopen(rid,registration)
    repaired.step(rid);run=repaired.step(rid)
    monkeypatch.setattr(verifier,'navigation_handoff',lambda *args:deepcopy(handoff))
    monkeypatch.setattr(verifier,'verify_acquisition',lambda root,rid,config,result,*args:deepcopy(result['document']))
    if tamper:
        event=next(e for e in run['events'] if e['type']=='navigation_response_repair_admitted')
        event['data']['feedback']['errors'][0]['detail']='invented feedback'
        rechain_constructed(flow,run)
        with pytest.raises(ValueError,match='feedback differs'):
            verifier.verify_repaired_navigation(tmp_path,rid,registration)
    else:
        proof=verifier.verify_repaired_navigation(tmp_path,rid,registration)
        assert proof['response_repairs']==1 and proof['model_attempts']==4
        assert proof['new_window_reads']==1 and proof['decisions_after_new_body_read']==1
