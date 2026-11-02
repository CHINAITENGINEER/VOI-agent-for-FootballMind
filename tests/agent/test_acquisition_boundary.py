from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import runpy

import pytest

from football_agent.acquisition_boundary import (CandidateBatch, PolicyChoice,
    ProposedChoiceBaseline, SelectionContext, materialize_choice, policy_identity)
from football_agent.acquisition_replay import verify_run
from football_agent.experience import ExperienceService, public_state

helpers=runpy.run_path(str(Path(__file__).with_name('test_match_experience.py')))
match_decision, MatchTools=helpers['match_decision'],helpers['MatchTools']


class FixtureProposer:
    def __init__(self):
        self.seen=[]

    def propose(self, public):
        self.seen.append(deepcopy(public))
        if public['documents']:
            candidates=[match_decision(public,'defer')]
        else:
            a=match_decision(public)
            b=deepcopy(a)
            a['query']='fixture source A'
            b['query']='fixture source B'
            candidates=[a,b]
        return {'candidates':candidates,'recommended_index':0,'test_double':True}


class FixturePolicy:
    version='fixture-independent-selection-v1'
    configuration={'scope':'CONSTRUCTED_INTERFACE_TEST_NOT_VOI'}

    def __init__(self):
        self.seen=[]

    def select(self,context,batch):
        self.seen.append(context.state)
        if context.state['documents']:
            return PolicyChoice(context.task_id,context.snapshot_digest,terminal_action='stop',
                                reason='构造接口检查：收到返回后独立停止。')
        return PolicyChoice(context.task_id,context.snapshot_digest,
                            candidate_id=batch.candidates[-1].candidate_id,
                            reason='构造接口检查：选择第二个候选，与模型建议不同。')


@pytest.fixture
def runtime(tmp_path):
    proposer,policy=FixtureProposer(),FixturePolicy()
    service=ExperienceService(tmp_path,proposer_factory=lambda f,e:proposer,selection_policy=policy,
                              tools_factory=lambda f:MatchTools(),mode='test_fixture')
    run=service.create({'match':'构造甲队 vs 构造乙队'})['run_id']
    yield service,run,proposer,policy
    service.close()


def test_policy_overrides_model_recommendation_then_replans_from_return(runtime):
    service,run,proposer,policy=runtime
    service._drive(run,True)
    state=service.get(run)
    assert state['turns'][0]['query']=='fixture source B'
    assert [t['action'] for t in state['turns']]==['search_web','stop']
    assert state['status']=='deferred' and state['spent']=={'http_calls':1,'llm_calls':2}
    assert len(policy.seen[0]['documents'])==0 and len(policy.seen[1]['documents'])==1
    assert policy.seen[1]['documents'][0]['selection_operation_id']
    assert state['numeric_voi'] is None
    assert state['selection_trace'][0]['candidate_count']==2
    op=state['selection_trace'][0]['operation_id']
    original=json.loads((service.folder(run)/'operations'/op/'result.json').read_text(encoding='utf-8'))
    assert original['recommended_index']==0 and original['candidates'][0]['query']=='fixture source A'
    report=verify_run(service.folder(run))
    assert report['accepted_decisions_verified']==2 and report['linked_http_receipts']==1
    assert report['real_calls_made_by_replay']==0


def test_policy_can_stop_without_source_or_extra_model_call(runtime):
    service,run,proposer,policy=runtime
    policy.select=lambda c,b:PolicyChoice(c.task_id,c.snapshot_digest,terminal_action='stop',reason='测试停止')
    service._drive(run,True)
    s=service.get(run)
    assert s['spent']=={'http_calls':0,'llm_calls':1}
    assert s['status']=='deferred' and s['assessment']['status']=='unresolved'
    assert len(proposer.seen)==1 and not s['documents']


@pytest.mark.parametrize('case',['unproposed','stale','other_task','both_actions','nan'])
def test_invalid_choice_is_rejected_before_any_request(runtime,case):
    service,run,proposer,policy=runtime
    def invalid(c,b):
        values={'task_id':c.task_id,'snapshot_digest':c.snapshot_digest,
                'candidate_id':b.candidates[0].candidate_id,'reason':'invalid fixture'}
        if case=='unproposed':values['candidate_id']='missing'
        if case=='stale':values['snapshot_digest']='old'
        if case=='other_task':values['task_id']='other'
        if case=='both_actions':values['terminal_action']='stop'
        if case=='nan':values['diagnostics_json']='{'+chr(34)+'gain'+chr(34)+':NaN}'
        return PolicyChoice(**values)
    policy.select=invalid
    service._drive(run,True)
    s=service.get(run)
    assert s['status']=='paused' and s['spent']=={'http_calls':0,'llm_calls':1}
    assert s['turns']==[] and s['queued_action'] is None
    assert list(service.folder(run).glob('operations/*/result.json'))


def test_context_and_candidate_access_do_not_mutate_archived_values(runtime):
    service,run,_,_=runtime
    public=public_state(service.get(run))
    r=match_decision(public)
    batch=CandidateBatch.build(public,[r],recommended_index=0,origin='fixture')
    before=batch.record()
    public['documents'].append({'future':'not visible'})
    view=batch.context.state
    view['documents'].append({'attempt':'mutation'})
    candidate=batch.candidates[0].response
    candidate['query']='changed'
    assert batch.record()==before and batch.context.state['documents']==[]
    with pytest.raises(FrozenInstanceError):batch.context.task_id='other'


def test_different_candidate_assessments_are_not_silently_mixed(runtime):
    service,run,_,_=runtime
    public=public_state(service.get(run))
    a=match_decision(public); b=deepcopy(a)
    b['query']='other'; b['assessment']['answer']='different belief'
    with pytest.raises(ValueError,match='SHARED_ANALYSIS'):
        CandidateBatch.build(public,[a,b],origin='fixture')


def test_duplicate_candidates_and_missing_recommendation_fail_explicitly(runtime):
    service,run,_,_=runtime
    public=public_state(service.get(run)); a=match_decision(public)
    with pytest.raises(ValueError,match='DUPLICATE'):
        CandidateBatch.build(public,[a,deepcopy(a)],origin='fixture')
    batch=CandidateBatch.build(public,[a],origin='fixture')
    with pytest.raises(ValueError,match='EXPLICIT_RECOMMENDATION'):
        ProposedChoiceBaseline().select(batch.context,batch)


def test_changed_policy_configuration_pauses_before_spending_more(runtime):
    service,run,_,policy=runtime
    policy.configuration={'scope':'different version of fixture'}
    service._drive(run,True)
    s=service.get(run)
    assert s['status']=='paused' and 'POLICY_OR_PROPOSER_CHANGED' in s['last_error']
    assert s['spent']=={'http_calls':0,'llm_calls':0}


def test_restart_without_original_policy_does_not_silently_fall_back(runtime):
    service,run,_,_=runtime
    restarted=ExperienceService(service.root,mode='test_fixture')
    try:
        restarted._drive(run,True)
        s=restarted.get(run)
        assert s['status']=='paused' and s['spent']['llm_calls']==0
    finally:restarted.close()


def test_policy_mutating_identity_during_selection_is_rejected(runtime):
    service,run,_,policy=runtime
    choose=policy.select
    def change(c,b):
        result=choose(c,b); policy.configuration={'changed':True}; return result
    policy.select=change
    service._drive(run,True)
    assert service.get(run)['last_validation_error']=='POLICY_CHANGED_DURING_SELECTION'
    assert service.get(run)['spent']['http_calls']==0


def test_cancellation_during_selection_prevents_execution(runtime):
    service,run,_,policy=runtime
    choose=policy.select
    def cancel(c,b):service.cancel(run);return choose(c,b)
    policy.select=cancel
    service._drive(run,True)
    s=service.get(run)
    assert s['status']=='cancelled' and s['spent']['http_calls']==0 and not s['turns']


def test_tampered_queued_action_never_reaches_tool(runtime):
    service,run,_,_=runtime
    service.step(run)
    service.mutate(run,lambda s:s['queued_action'].update(query='tampered'))
    service._drive(run,True)
    assert service.get(run)['status']=='paused' and service.get(run)['spent']['http_calls']==0


def test_final_model_attempt_can_stop_while_candidates_remain(runtime):
    service,run,proposer,policy=runtime
    service.mutate(run,lambda s:s['spent'].update(llm_calls=s['config']['limits']['llm_calls']-1))
    policy.select=lambda c,b:PolicyChoice(c.task_id,c.snapshot_digest,terminal_action='defer',reason='final decision')
    service._drive(run,True)
    assert service.get(run)['status']=='deferred' and service.get(run)['spent']['http_calls']==0


@pytest.mark.parametrize('target',['proposal','input','selection','decision','raw','receipt'])
def test_offline_replay_detects_tampering(runtime,target):
    service,run,_,_=runtime
    service._drive(run,True)
    folder=service.folder(run); op=service.get(run)['selection_trace'][0]['operation_id']
    mapping={'proposal':'proposal.json','input':'input.json','selection':'selection.json',
             'decision':'decision.json','raw':'result.json'}
    path=folder/'operations'/op/mapping[target] if target!='receipt' else next(folder.glob('acquisitions/*/result.json'))
    value=json.loads(path.read_text(encoding='utf-8'))
    if target=='proposal':value['candidates'][0]['response']['query']='changed'
    elif target=='input':value['match_input']='other match'
    elif target=='selection':value['choice']['snapshot_digest']='wrong snapshot'
    elif target=='decision':value['reason']='changed'
    elif target=='raw':value['recommended_index']=1
    else:value['requests'][0]['http_status']=403
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
    with pytest.raises(ValueError):verify_run(folder)


def test_unknown_costs_remain_unknown_not_zero(runtime):
    service,run,_,_=runtime
    public=public_state(service.get(run))
    batch=CandidateBatch.build(public,[match_decision(public)],origin='fixture')
    cost=batch.record()['candidates'][0]['resource_cost']
    assert cost['http_request_lower_bound']==1
    assert cost['money_estimate'] is None and cost['decision_loss_equivalent'] is None


@pytest.mark.parametrize('target',['link','result','trace','document'])
def test_missing_or_changed_execution_evidence_is_not_silently_ignored(runtime,target):
    service,run,_,_=runtime
    service._drive(run,True)
    folder=service.folder(run)
    if target=='link':next(folder.glob('acquisitions/*/selection-link.json')).unlink()
    elif target=='result':next(folder.glob('acquisitions/*/result.json')).unlink()
    elif target=='trace':service.mutate(run,lambda s:s.update(selection_trace=[]))
    else:service.mutate(run,lambda s:s['documents'][0].update(text='altered source'))
    with pytest.raises(ValueError):verify_run(folder)


def test_sensitive_config_is_not_silently_redacted_into_same_identity():
    policy=FixturePolicy()
    policy.configuration={'secret':'test-sensitive-value'}
    with pytest.raises(ValueError,match='NOT_ARCHIVE_STABLE'):policy_identity(policy)
