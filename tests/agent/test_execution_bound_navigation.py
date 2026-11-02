from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import pytest
from football_agent.audit import digest
from football_agent.execution_bound_navigation import (ExecutionBoundWorkflow,ExecutionBoundCodexProvider,
    request_record,verify_response_binding,SCHEMA)
from test_grounded_navigation_episode import setup
from test_candidate_grounded_navigation import convert
from test_target_navigation import choice,query,finish


def read_new(public):
    doc=public['documents'][-1]
    return choice(public,'read_window',document_id=doc['document_id'],window_id=doc['catalog'][0]['window_id'])


def raw_choice(builder,public):
    legacy={**deepcopy(public),'task_contract_digest':digest(public['task_contract'])}
    raw=convert(builder(legacy));raw.pop('task_contract_digest')
    if builder is query:raw['query']['url']='https://example.com/new-schedule' 
    return raw


def bound_reply(public,raw,request_id=None):
    receipt={'real_provider_call':False,'scope':'constructed executor test'}
    request=request_record(public,request_id or str(uuid4()))
    return {'response':deepcopy(raw),'receipt':receipt,'execution_binding':{
        **request,'response_digest':digest(raw),'provider_receipt_digest':digest(receipt)}}


def make_bound(tmp_path,monkeypatch,replies):
    old,old_id,_,handoff,parent,_=setup(tmp_path,monkeypatch,[])
    seen=[]
    class Provider:
        def configuration(self):return {'provider':'constructed-executor-binding'}
        def decide(self,public):
            reply=replies[len(seen)](public);seen.append(deepcopy(reply));return reply
    original_execute=old.tools.execute
    def fresh_source(prepared,**kwargs):
        result=original_execute(prepared,**kwargs)
        result['document'].update(document_id='bound-doc',url='https://example.com/new-schedule')
        return result
    old.tools.execute=fresh_source
    flow=ExecutionBoundWorkflow(tmp_path/'bound',Provider(),old.tools,clock=old.clock)
    original=old.get(old_id)
    original['task']['seed_urls'].append('https://example.com/new-schedule')
    run=flow.start(original['task'],handoff,original['target_contract'],max_llm_calls=3,max_http_attempts=1,max_window_reads=2)
    return flow,run['run_id'],seen


def test_actual_executor_queries_reads_reconsiders_without_model_hash(tmp_path,monkeypatch):
    replies=[lambda p:bound_reply(p,raw_choice(query,p)),lambda p:bound_reply(p,raw_choice(read_new,p)),lambda p:bound_reply(p,raw_choice(finish,p))]
    flow,rid,seen=make_bound(tmp_path,monkeypatch,replies)
    flow.step(rid);flow.step(rid);run=flow.step(rid)
    assert run['reason']=='AGENT_DEFER'
    assert (run['budget']['llm_calls'],run['budget']['http_attempts'],run['budget']['window_reads'])==(3,1,1)
    assert all('task_contract_digest' not in r['response'] for r in seen)
    assert [d['choice'] for d in run['decisions']]==[r['response'] for r in seen]
    assert seen[-1]['response']['return_assessment']['outcome']=='partial'
    assert flow.step(rid)==run


@pytest.mark.parametrize('field',['task_id','public_input_digest','response_digest','provider_receipt_digest','execution_context'])
def test_corrupted_binding_blocks_action_before_source_call(tmp_path,monkeypatch,field):
    def response(public):
        reply=bound_reply(public,raw_choice(query,public));reply['execution_binding'][field]='wrong'
        return reply
    flow,rid,_=make_bound(tmp_path,monkeypatch,[response]);run=flow.step(rid)
    assert run['reason']=='INVALID_NAVIGATION_RESPONSE'
    assert run['budget']['llm_calls']==1 and run['budget']['http_attempts']==0 and not run['decisions']


def test_model_injected_identity_field_is_rejected(tmp_path,monkeypatch):
    def response(public):
        raw=raw_choice(query,public);raw['task_contract_digest']=digest(public['task_contract'])
        return bound_reply(public,raw)
    flow,rid,_=make_bound(tmp_path,monkeypatch,[response]);run=flow.step(rid)
    assert run['reason']=='INVALID_NAVIGATION_RESPONSE' and run['budget']['http_attempts']==0


def test_request_identity_reuse_is_rejected(tmp_path,monkeypatch):
    flow,rid,_=make_bound(tmp_path,monkeypatch,[lambda p:bound_reply(p,raw_choice(query,p),'same-request'),lambda p:bound_reply(p,raw_choice(read_new,p),'same-request')])
    flow.step(rid);run=flow.step(rid)
    assert run['reason']=='INVALID_NAVIGATION_RESPONSE' and run['budget']['window_reads']==0
    assert run['budget']['llm_calls']==2 and run['budget']['http_attempts']==1


def test_provider_keeps_raw_output_and_writes_binding_outside_choice(tmp_path,monkeypatch):
    flow,rid,_=make_bound(tmp_path,monkeypatch,[]);public=flow.public(flow.get(rid))
    expected=raw_choice(choice,public)
    def runner(command,**kwargs):
        output=Path(command[command.index('--output-last-message')+1])
        output.write_text(json.dumps(expected,ensure_ascii=False),encoding='utf-8')
        return SimpleNamespace(returncode=0,stdout=json.dumps({'type':'turn.completed','usage':{}}),stderr='')
    provider=ExecutionBoundCodexProvider(tmp_path/'provider',executable='constructed-command',runner=runner)
    reply=provider.decide(public)
    assert reply['response']==expected and 'task_contract_digest' not in reply['response']
    requests=list((tmp_path/'provider/bindings').glob('*/request.json'));assert len(requests)==1
    request=json.loads(requests[0].read_text(encoding='utf-8'))
    assert verify_response_binding(reply,public,request)
    original=list((tmp_path/'provider').glob('*/response.json'));assert len(original)==1
    assert json.loads(original[0].read_text(encoding='utf-8'))==reply['response']
    changed=deepcopy(public);changed['task_contract']['subject_team']='a different team'
    with pytest.raises(ValueError):verify_response_binding(reply,changed,request)
    assert 'task_contract_digest' not in SCHEMA['properties']


def test_executor_rejects_changed_task_context(tmp_path,monkeypatch):
    flow,rid,_=make_bound(tmp_path,monkeypatch,[]);run=flow.get(rid);public=flow.public(run)
    raw=raw_choice(choice,public);public['task']['objective']='changed'
    with pytest.raises(ValueError,match='context changed'):
        flow.validate_choice(raw,run,public)
