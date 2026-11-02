from copy import deepcopy
import json
import math

import pytest

from football_research.acquisition.finite_information import build_scenarios,FiniteJointModel,HYPOTHESIS_KEY
from football_research.acquisition.experiment_round15 import select_fixed_policies,run_returns,fixed_plan,empty_state,budget_for
from football_research.acquisition.joint_planner import JointRiskPlanner,entropy


@pytest.fixture
def scenarios():return build_scenarios()


def test_frozen_population_sizes_masses_and_full_true_support(scenarios):
    assert {s:len(v['worlds']) for s,v in scenarios.items()}=={'gated_stop':18,'gated_route':54,'complementary':27,'redundant':9,'irrelevant':12}
    assert {s:len({tuple(sorted(w['observations'].items())) for w in v['worlds']}) for s,v in scenarios.items()}=={'gated_stop':6,'gated_route':18,'complementary':9,'redundant':3,'irrelevant':4}
    assert sum(len(s['models']) for s in scenarios.values())==12
    for scenario in scenarios.values():
        assert math.fsum(w['mass'] for w in scenario['worlds'])==pytest.approx(1)
        for spec in scenario['models'].values():
            model=FiniteJointModel(spec)
            for world in scenario['worlds']:
                p=model.predict_partial({'observations':world['observations'],'action_history':[]})['probabilities']
                assert sum(p)==pytest.approx(1) and p[world['outcome']]>0


def test_independent_variants_preserve_marginals_remove_all_information(scenarios):
    for scenario in scenarios.values():
        model=FiniteJointModel(scenario['models']['independent'])
        before=model.predict_partial(empty_state())['probabilities']
        for world in scenario['worlds']:
            assert model.predict_partial({'observations':world['observations']})['probabilities']==pytest.approx(before)
        true=scenario['models']['correct']['rows']
        for aid in model.action_ids:
            for z in model.categories[aid]:
                original=math.fsum(w['mass'] for w in true if w['observations'][aid]==z)
                distorted=math.fsum(w['mass'] for w in model.spec['rows'] if w['observations'][aid]==z)
                assert distorted==pytest.approx(original)


def test_martingale_for_every_one_observation_prefix(scenarios):
    for scenario in scenarios.values():
        for spec in scenario['models'].values():
            model=FiniteJointModel(spec)
            states=[empty_state()]+[{'observations':{aid:z}} for aid in model.action_ids for z in model.categories[aid]]
            for state in states:
                before=model.predict_partial(state)['probabilities']
                for action in scenario['actions']:
                    if action['action_id'] in state['observations']:continue
                    enumeration=model.enumerate_action(state,action)
                    assert sum(b['probability'] for b in enumeration['branches'])==pytest.approx(1)
                    mean=[math.fsum(b['probability']*b['posterior_probabilities'][y] for b in enumeration['branches']) for y in range(3)]
                    assert mean==pytest.approx(before)
                    for b in enumeration['branches']:
                        assert model.predict_hypothetical(b['next_state'])['probabilities']==pytest.approx(b['posterior_probabilities'])
                        with pytest.raises(ValueError):model.predict_partial(b['next_state'])


def test_complementary_two_step_queries_despite_zero_immediate_information(scenarios):
    s=scenarios['complementary'];m=FiniteJointModel(s['models']['correct'])
    one=JointRiskPlanner(m,horizon=1,lambda_cost={'money':1}).plan(empty_state(),s['actions'],budget_for(s))
    two=JointRiskPlanner(m,horizon=2,lambda_cost={'money':1}).plan(empty_state(),s['actions'],budget_for(s))
    assert one['decision']=='stop' and two['action_id']=='signal_a'
    assert all(c['immediate_evsi']==pytest.approx(0,abs=1e-12) for c in two['candidates'])
    assert two['expected_total_risk']==pytest.approx(entropy([.9,.05,.05])+.24)
    fixed=select_fixed_policies(m,s)
    assert fixed['best-fixed-subset']['selected_order']==['signal_a','signal_b']
    assert fixed['best-fixed-subset']['selected_expected_total_risk']==pytest.approx(two['expected_total_risk'])


def test_gated_context_changes_next_tool_and_stop_and_histories_are_public(scenarios):
    for sid in ('gated_route','gated_stop'):
        s=scenarios[sid];m=FiniteJointModel(s['models']['correct']);fixed=select_fixed_policies(m,s)
        paths=[]
        for g in ('0','1'):
            returns={aid:'0' for aid in m.action_ids};returns['context']=g
            result,events=run_returns(s,m,'adaptive-two-step',returns,fixed_selections=fixed);paths.append(result['actions'])
            assert all(set(e['visible_state'])=={'observations','action_history'} for e in events)
            assert result['cost']['money']<=s['budget_money']+1e-12 and len(result['actions'])<=2
        assert paths==([['context','signal_a'],['context','signal_b']] if sid=='gated_route' else [['context','signal_a'],['context']])


def test_fixed_order_stop_is_strong_and_cannot_reroute(scenarios):
    s=scenarios['gated_route'];m=FiniteJointModel(s['models']['correct']);selected=select_fixed_policies(m,s)
    strong=selected['best-fixed-order-stop'];weak=selected['best-fixed-subset']
    assert strong['selected_expected_total_risk']<weak['selected_expected_total_risk']
    adaptive=JointRiskPlanner(m,horizon=2,lambda_cost={'money':1}).plan(empty_state(),s['actions'],budget_for(s))
    assert adaptive['expected_total_risk']<strong['selected_expected_total_risk']
    order=strong['selected_order']
    for world in s['worlds']:
        result,_=run_returns(s,m,'best-fixed-order-stop',world['observations'],fixed_selections=selected)
        assert result['actions']==order[:len(result['actions'])]


def test_redundancy_and_irrelevance_stop_where_information_is_zero(scenarios):
    s=scenarios['redundant'];m=FiniteJointModel(s['models']['correct']);fixed=select_fixed_policies(m,s)
    r,_=run_returns(s,m,'adaptive-two-step',s['worlds'][0]['observations'],fixed_selections=fixed)
    assert r['actions']==['signal_a'] and r['cost']['money']==.04
    assert m.enumerate_action({'observations':{'signal_a':'0'}},s['actions'][1])['evsi']==pytest.approx(0,abs=1e-12)
    s=scenarios['irrelevant'];m=FiniteJointModel(s['models']['correct']);fixed=select_fixed_policies(m,s)
    for policy in ('myopic','adaptive-two-step','best-fixed-subset','best-fixed-order-stop'):
        r,_=run_returns(s,m,policy,s['worlds'][0]['observations'],fixed_selections=fixed)
        assert r['actions']==[] and r['cost']['money']==0


def test_public_state_rejects_hidden_fields_failed_unknown_and_duplicate_inputs(scenarios):
    m=FiniteJointModel(scenarios['redundant']['models']['correct'])
    for key in ('outcome','world_id','returns','label',HYPOTHESIS_KEY):
        with pytest.raises(ValueError):m.predict_partial({**empty_state(),key:0})
    with pytest.raises(ValueError):m.predict_partial({'observations':{'signal_a':'unknown'}})
    with pytest.raises(ValueError):m.predict_partial({'observations':{'signal_a':'0','signal_b':'1'}})
    with pytest.raises(ValueError):m.predict_partial({'observations':{'signal_a':'0'},'action_history':[{'action_id':'signal_a','status':'TIMEOUT'}]})
    with pytest.raises(ValueError):m.enumerate_action({'observations':{'signal_a':'0'}},{'action_id':'signal_a'})
    with pytest.raises(ValueError):m.predict_partial({'observations':{},'action_history':[{'action_id':'signal_a','status':'observed'}]})


def test_malformed_distribution_and_saved_artifact_rejected(scenarios,tmp_path):
    original=scenarios['gated_stop']['models']['correct']
    changes=[lambda s:s['rows'][0].update(mass=-1),lambda s:s['rows'][0].update(outcome=True),
        lambda s:s['rows'].append(deepcopy(s['rows'][0])),lambda s:s['rows'][0]['observations'].update(secret='0'),lambda s:s['rows'][0].update(mass=.9)]
    for change in changes:
        spec=deepcopy(original);change(spec)
        with pytest.raises(ValueError):FiniteJointModel(spec)
    model=FiniteJointModel(original);p=tmp_path/'model.json';model.save(p)
    assert FiniteJointModel.load(p).predict_partial(empty_state())==model.predict_partial(empty_state())
    raw=json.loads(p.read_text(encoding='utf-8'));raw['model_version']='tampered';p.write_text(json.dumps(raw),encoding='utf-8')
    with pytest.raises(ValueError):FiniteJointModel.load(p)


def test_fixed_order_checks_prefix_and_physical_request_limit(scenarios):
    s=scenarios['gated_route'];m=FiniteJointModel(s['models']['correct'])
    with pytest.raises(ValueError):fixed_plan(m,s,['signal_a','signal_b'],{'observations':{'signal_b':'0'}},budget_for(s),allow_stop=True)
    plan=JointRiskPlanner(m,horizon=2,lambda_cost={'money':1}).plan(empty_state(),s['actions'],{'limits':{'money':.42,'requests':1},'spent':{}})
    assert plan['expected_plan_cost_vector']['requests']<=1
