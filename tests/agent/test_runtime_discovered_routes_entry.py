"""New search and newly observed URL share durable exploration, without routes."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
from types import FunctionType

import pytest

from football_agent import runtime_exploration as exploration
from football_agent import runtime_discovered_routes as discovery
from football_agent.acquisition_replay import verify_run
from football_agent.acquisition_boundary import policy_identity
from football_agent.audit import atomic_write
from football_agent.experience import public_state
from football_agent.state_bound_runtime_policy import StateBoundVoISelectionPolicy

P = runpy.run_path(str(Path(__file__).with_name('test_runtime_exploration_entry.py')))


def fixture_runtime(root, *, costs=None, options=None, interruption=None, fail_body=False):
    # Reuse the exact transport/proposer scenario, replacing only the frozen
    # registry and policy at construction (before create persists identity).
    def package():
        config=P['model_package']()
        config['routes']=[r for r in config['routes'] if r['route_id'] in {'b','c'}]
        config['history']=[h for h in config['history'] if h['link']['candidate_id'] in {'b','c'}]
        config['family_size']=len(config['routes'])
        return config
    def policy(model, **kwargs):
        return StateBoundVoISelectionPolicy(model, **kwargs,
            discovered_routes=costs or {'search_web':{'bing':.1},'read_url':.1})
    original=P['fixture_runtime']
    namespace=dict(original.__globals__, model_package=package, StateBoundVoISelectionPolicy=policy)
    factory=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    factory.__kwdefaults__=original.__kwdefaults__
    return factory(root,options=options,interruption=interruption,fail_body=fail_body)


def run_scenario(root, **kwargs):
    service,run,calls,seen=fixture_runtime(root,**kwargs)
    service._drive(run,True)
    return service,run,calls,seen


def test_match_only_new_search_new_url_source_state_value_selection_and_stop(tmp_path):
    service,run,calls,seen=run_scenario(tmp_path)
    try:
        state=service.get(run)
        assert state['last_error'] is None
        assert [r['action'] for r in state['turns']]==['read_url','search_web','read_url','search_web','stop']
        assert state['turns'][3]['query'].endswith(' source b')
        assert state['spent']=={'http_calls':4,'llm_calls':5}
        assert state['user_context']==[] and state['question_to_user'] is None
        registered=state['config']['selection_policy']['configuration']['model_descriptor']['routes']
        assert {r['route_id'] for r in registered}=={'b','c'}
        report=verify_run(service.folder(run),include_action_value_ledger=True)
        reserved=report['bounded_exploration']['reservations']
        assert len(reserved)==2
        assert [r['discovered_route']['request']['action'] for r in reserved]==['search_web','read_url']
        assert reserved[1]['discovered_route']['origin']=={'kind':'observed_link',
            'document_id':seen[2]['documents'][-1]['document_id'],'link_index':0,'url':P['URL']}
        assert all(r['value_status']=='VALUE_UNMEASURED' and r['realized_information_gain'] is None for r in reserved)
        assert all(not r['discovered_route']['empirical_support'] for r in reserved)
        assert report['source_assertions_verified']==1
        selected=[e for e in report['action_value_ledger']['entries'] if e['selection_basis']==exploration.BASIS]
        assert len(selected[0]['candidate_value_assessments'])==2
        assert all(e['expected_information_value'] is None for e in selected)
        assert len(calls)==4
    finally:service.close()


@pytest.mark.parametrize('kwargs',[{'options':{'max_actions':1,'max_declared_cost':.2}},
                                  {'options':{'max_actions':2,'max_declared_cost':.1}},
                                  {'costs':{'search_web':{'bing':.1},'read_url':None}}])
def test_new_url_stops_at_declared_allowance_or_missing_tool_cost(tmp_path,kwargs):
    service,run,calls,_=run_scenario(tmp_path,**kwargs)
    try:
        assert service.get(run)['last_error'] is None
        assert service.get(run)['turns'][-1]['action']=='defer'
        assert len(calls)==2 and P['URL'] not in calls
        assert verify_run(service.folder(run))['bounded_exploration']['reserved_actions']==1
    finally:service.close()


@pytest.mark.parametrize('phase',['plan','fetch'])
def test_cost_profile_changed_pauses_before_another_call(tmp_path,phase):
    service,run,calls,seen=fixture_runtime(tmp_path)
    try:
        for _ in range(3 if phase=='fetch' else 4):service.step(run)
        before=(len(calls),len(seen))
        old=service.selection_policy
        service.selection_policy=StateBoundVoISelectionPolicy(old.model,allow_synthetic=True,
            exploration={'max_actions':2,'max_declared_cost':.2},
            discovered_routes={'search_web':{'bing':.05},'read_url':.1})
        service.step(run)
        assert service.get(run)['status']=='paused'
        assert (len(calls),len(seen))==before
        assert verify_run(service.folder(run))['bounded_exploration']['reserved_actions']==1
    finally:service.close()


@pytest.mark.parametrize('interruption',['cancel','deadline','invalid'])
def test_unaccepted_new_action_never_reserves(tmp_path,interruption):
    service,run,_,_=run_scenario(tmp_path,interruption=interruption)
    try:
        assert verify_run(service.folder(run))['bounded_exploration']['reserved_actions']==0
    finally:service.close()


def test_failed_new_body_keeps_budget_and_payload_identity(tmp_path):
    service,run,_,_=fixture_runtime(tmp_path,fail_body=True)
    try:
        for _ in range(6):service.step(run)
        state=service.get(run)
        before=public_state(state)[exploration.STATE_FIELD]
        assert before['reserved_actions']==2
        assert discovery.action_key(state['turns'][-1]) in before['used_discovered_action_keys']
        assert verify_run(service.folder(run))['bounded_exploration']==before
    finally:service.close()


@pytest.mark.parametrize('field',['cost','route_id','origin','question','request'])
def test_tampered_dynamic_reservation_is_reconstructed_from_frozen_capability(tmp_path,field):
    service,run,_,_=run_scenario(tmp_path)
    try:
        def mutate(state):
            op=next(o for o in state['operations'] if exploration.RESERVATION in o
                    and o[exploration.RESERVATION]['discovered_route']['request']['action']=='read_url')
            route=op[exploration.RESERVATION]['discovered_route']
            if field=='cost':route['cost']=0
            elif field=='route_id':route['route_id']='discovered:fake'
            elif field=='origin':route['origin']['document_id']='fake'
            elif field=='question':route['question']['id']='fake'
            else:route['request']['url']='https://example.org/unobserved'
        service.mutate(run,mutate)
        with pytest.raises(ValueError,match='RESERVATION_MISMATCH'):
            verify_run(service.folder(run))
    finally:service.close()
