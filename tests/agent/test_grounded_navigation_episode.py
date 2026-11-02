
from copy import deepcopy
from datetime import timedelta

import pytest

from football_agent.audit import digest
from football_agent.fixture_bound_review import task_contract
from football_agent.grounded_navigation_episode import GroundedMemoryWorkflow,repaired_actor_handoff
from football_agent.grounded_navigation_verify import verify_grounded_episode
from test_target_navigation import make,query,read_new,finish,choice
from test_navigation_response_repair import invalid_read,repair
from test_candidate_grounded_navigation import convert
from test_information_navigation import NOW,rechain_constructed


def setup(tmp_path,monkeypatch,replies):
    import football_agent.navigation_response_repair_verify as parent_verifier
    old,rid,_,_,original_handoff,_=make(tmp_path/'parent',[query,invalid_read,read_new,finish],max_llm_calls=4)
    old.step(rid);old.step(rid)
    repaired,registration=repair(old,rid);repaired.reopen(rid,registration)
    repaired.step(rid);parent=repaired.step(rid)
    monkeypatch.setattr(parent_verifier,'navigation_handoff',lambda *args:deepcopy(original_handoff))
    monkeypatch.setattr(parent_verifier,'verify_acquisition',lambda root,rid,config,result,*args:deepcopy(result['document']))
    handoff=repaired_actor_handoff(tmp_path/'parent',rid,parent['events'][-1]['event_hash'],registration)
    seen=[]
    class Provider:
        def configuration(self):return {'provider':'constructed-no-live-call'}
        def decide(self,public):
            seen.append(deepcopy(public))
            return {'response':replies[len(seen)-1](public),'receipt':{'real_provider_call':False}}
    clock=lambda:NOW+timedelta(minutes=1)
    flow=GroundedMemoryWorkflow(tmp_path/'new',Provider(),old.tools,clock=clock)
    task={**parent['task'],'task_id':'new-grounded-episode'}
    contract=task_contract(task_id=task['task_id'],question=task['objective'],subject_team='Team A',team_aliases=[],reference_time=clock())
    run=flow.start(task,handoff,contract,max_llm_calls=3,max_http_attempts=1,max_window_reads=2)
    return flow,run['run_id'],seen,handoff,parent,registration


def read_archive(public):
    doc=public['documents'][0]
    return convert(choice(public,'read_window',document_id=doc['document_id'],window_id=doc['catalog'][1]['window_id']))


def test_real_executor_inherits_scoped_feedback_without_refunding_parent(tmp_path,monkeypatch):
    flow,rid,seen,handoff,parent,registration=setup(tmp_path,monkeypatch,[read_archive,lambda p:convert(finish(p))])
    flow.step(rid);run=flow.step(rid)
    assert run['reason']=='AGENT_DEFER' and run['budget']['llm_calls']==2 and run['budget']['window_reads']==1
    assert seen[0]['prior_information_tests']==handoff['actor_return_assessments']
    assert len(seen[0]['prior_information_tests'])==2
    assert seen[0]['episode_boundary']['parent_budget_final']['llm_calls']==4
    assert seen[0]['budget']['remaining_llm_calls_including_this']==3
    assert seen[0]['prior_information_tests'][-1]['prospective_information_test']['requirement']=='all_fixture_order'
    assert seen[0]['task_contract']['reference_time']!=parent['target_contract']['reference_time']
    proof=verify_grounded_episode(tmp_path/'new',rid,registration)
    assert proof['inherited_actor_assessments']==2 and proof['model_attempts']==2 and proof['new_window_reads']==1
    assert flow.step(rid)==run


def test_same_parent_task_id_is_not_silently_reopened(tmp_path,monkeypatch):
    flow,rid,seen,handoff,parent,registration=setup(tmp_path,monkeypatch,[])
    with pytest.raises(ValueError,match='distinct task'):
        flow.start(parent['task'],handoff,parent['target_contract'])
    assert not seen


def test_parent_head_mismatch_rejects_handoff(tmp_path,monkeypatch):
    flow,rid,seen,handoff,parent,registration=setup(tmp_path,monkeypatch,[])
    with pytest.raises(ValueError,match='head changed'):
        repaired_actor_handoff(tmp_path/'parent',parent['run_id'],'altered',registration)


def test_replay_rejects_hash_consistent_invented_source_feedback(tmp_path,monkeypatch):
    flow,rid,seen,handoff,parent,registration=setup(tmp_path,monkeypatch,[lambda p:convert(choice(p))])
    run=flow.step(rid)
    event=next(e for e in run['events'] if e['type']=='navigation_reasoning_reserved')
    old_ref=event['data']['input_ref']
    public=flow.store.get_blob(rid,old_ref)
    public['prior_information_tests'][-1]['assessment']['reason']='injected evaluator opinion'
    new_ref=flow.store.put_blob(rid,public)
    def replace(value):
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        if isinstance(value,list):return [replace(v) for v in value]
        return new_ref if value==old_ref else value
    rechain_constructed(flow,replace(run))
    with pytest.raises(ValueError,match='previously visible evidence'):
        verify_grounded_episode(tmp_path/'new',rid,registration)
