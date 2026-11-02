from copy import deepcopy
import numpy as np
import pytest
from football_research.acquisition import historical_gain_head as old
from football_research.acquisition import state_candidate_gain as g1
from football_research.acquisition import learned_continuation_g as g2
from football_research.acquisition.historical_policy_comparison import rollout


def query(title='injury update', count=2):
    return g1.build_query({'schema':old.QUERY_SCHEMA,'probabilities':[.4,.3,.3],
        'acquired_documents':0,'visible_candidates':count,'article_age_hours':0.,'title':title},
        {'physical_match_id':'test','used_input_ids':[],'claims':[]})


def setup(gain=-.01, future=.03, count=2):
    base=g1.fit([{'physical_match_id':mid,'query':query(count=count),'signed_gain':gain} for mid in ('a','b')])
    menu=[{'action_id':str(i),'query':query(count=count)} for i in range(count)]
    packet={'candidate':menu[0],'menu':menu,'cost_loss_units':0.}
    # Explicit synthetic unit fixture, not a trained football artifact.
    model={'schema':g2.VERSION,'immediate_model':base,
           'coefficients':np.zeros(len(g2.vector(base,packet))).tolist(),'intercept':future}
    return model,packet


def test_negative_first_step_can_be_selected_for_future_value():
    model,p=setup()
    assert g2.choose(model,p['menu'],1,0)['action_id'] is None
    two=g2.choose(model,p['menu'],2,0)
    assert two['action_id']=='0'
    assert two['candidates'][0]['immediate_gain']==pytest.approx(-.01)
    assert two['candidates'][0]['two_step_net_value']==pytest.approx(.02)


def test_one_step_or_no_remaining_action_cannot_hallucinate_continuation():
    model,p=setup(count=1)
    assert g2.estimate(model,p,remaining_reads=3)['continuation_net_value']==0
    model,p=setup()
    assert g2.estimate(model,p,remaining_reads=1)['continuation_net_value']==0


def test_future_net_cost_not_subtracted_twice():
    model,p=setup(gain=.01,future=.02);p['cost_loss_units']=.005
    assert g2.estimate(model,p,remaining_reads=2)['two_step_net_value']==pytest.approx(.025)


@pytest.mark.parametrize('key',['next_state','body','outcome','signed_gain','prediction_after'])
def test_postquery_fields_rejected(key):
    model,p=setup();p[key]='hidden'
    with pytest.raises(ValueError,match='contract'):g2.estimate(model,p,remaining_reads=2)


def test_crossfit_guard_and_roundtrip(tmp_path):
    model,p=setup();base=model['immediate_model']
    rows=[{'physical_match_id':mid,'packet':deepcopy(p),'teacher_fit_match_ids':[other],
           'continuation_target':.02} for mid,other in [('a','b'),('b','a')]]
    trained=g2.fit(rows,base);path=tmp_path/'g.json';g2.save(trained,path)
    assert g2.choose(trained,p['menu'],2,0)==g2.choose(g2.load(path),p['menu'],2,0)
    rows[0]['teacher_fit_match_ids']=['a','b']
    with pytest.raises(ValueError,match='whole target match'):g2.fit(rows,base)


def test_current_menu_binding_and_no_mutation():
    model,p=setup();saved=deepcopy(p);g2.estimate(model,p,remaining_reads=2)
    assert p==saved
    p['candidate']=deepcopy(p['candidate']);p['candidate']['query']['public']['title']='different'
    with pytest.raises(ValueError,match='differs'):g2.estimate(model,p,remaining_reads=2)


def test_evaluator_exposes_only_public_current_menu_and_obeys_budget():
    model,p=setup()
    graph={'s':[{'action_id':a['action_id'],'query':a['query'],'signed_gain':.5,'next_state_id':'end'} for a in p['menu']]}
    seen=[]
    def selector(menu,left,cost):
        assert all(set(a)=={'action_id','query'} for a in menu)
        seen.append(left)
        return g2.choose(model,menu,left,cost)
    first=rollout(graph,'s',policy='menu',budget=2,cost=0,menu_selector=selector)
    graph['s'][0]['signed_gain']=-100
    second=rollout(graph,'s',policy='menu',budget=2,cost=0,menu_selector=selector)
    assert first['paths'][0]['steps'][0]['action_id']==second['paths'][0]['steps'][0]['action_id']
    assert first['expected_reads']==1 and seen==[2,2]
