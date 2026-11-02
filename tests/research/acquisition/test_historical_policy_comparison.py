from copy import deepcopy
import pytest
from football_research.acquisition.historical_policy_comparison import rollout


def graph():
    return {'s': [{'action_id':'a','query':{'score': .1},'next_state_id':'a','signed_gain':.3},
                  {'action_id':'b','query':{'score': .2},'next_state_id':'b','signed_gain':-.1}],
            'a': [{'action_id':'b','query':{'score':-.1},'next_state_id':'ab','signed_gain':-.1}],
            'b': [{'action_id':'a','query':{'score':.1},'next_state_id':'ab','signed_gain':.3}]}


def scorer(public):
    assert set(public) == {'action_id','query'}
    return public['query']['score']


def test_random_exact_budget_cost_and_telescoping():
    a = rollout(graph(),'s',policy='random',budget=1,cost=.01)
    assert a['expected_gain'] == pytest.approx(.1)
    assert a['expected_net'] == pytest.approx(.09)
    assert len(a['paths']) == 2 and a['logged_propensity'] is None
    b = rollout(graph(),'s',policy='random',budget=2,cost=.01)
    assert b['expected_gain'] == pytest.approx(.2)
    assert b['expected_reads'] == 2


def test_g_decision_does_not_see_outcomes_or_future_gain():
    original = graph()
    a = rollout(original,'s',policy='g',budget=1,cost=0,scorer=scorer)
    changed = deepcopy(original)
    changed['s'][0]['signed_gain'] = 100
    changed['s'][1]['signed_gain'] = -100
    b = rollout(changed,'s',policy='g',budget=1,cost=0,scorer=scorer)
    assert a['paths'][0]['steps'][0]['action_id'] == b['paths'][0]['steps'][0]['action_id'] == 'b'


def test_stop_threshold_forced_ranking_and_zero_budget():
    a = rollout(graph(),'s',policy='g',budget=2,cost=.2,scorer=scorer)
    assert a['expected_reads'] == 0
    b = rollout(graph(),'s',policy='g_forced',budget=1,cost=.2,scorer=scorer)
    assert b['expected_reads'] == 1
    assert rollout(graph(),'s',policy='fixed',budget=0,cost=0)['expected_reads'] == 0


def test_ties_fixed_determinism_and_nan_rejection():
    a = rollout(graph(),'s',policy='g',budget=1,cost=0,scorer=lambda _: .1)
    assert a['paths'][0]['steps'][0]['action_id'] == 'a'
    with pytest.raises(ValueError,match='Finite'):
        rollout(graph(),'s',policy='g',budget=1,cost=0,scorer=lambda _: float('nan'))
