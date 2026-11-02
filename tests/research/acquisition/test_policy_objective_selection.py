from copy import deepcopy
import pytest
from football_research.acquisition import policy_objective_selection as s


def records():
    return [{'representation':mode,'physical_match_id':mid,'budget':b,'cost':c,
             'expected_net':{'baseline':.01,'structured':.02,'combined':-.01}[mode],
             'fit_match_ids':['other']}
            for mode in s.MODES for mid in ('a','b') for b in s.BUDGETS for c in s.COSTS]


def test_maximize_return_with_equal_match_weight_not_mse():
    result=s.select(records(),['a','b'])
    assert result['selected']=='structured'
    assert result['mean_net']['structured']==pytest.approx(.02)
    assert result['independent_sample_count']==2


def test_deterministic_tie_and_negative_returns_retained():
    rows=records()
    for r in rows:r['expected_net']=-.01
    result=s.select(rows,['a','b'])
    assert result['selected']=='baseline' and result['mean_net']['baseline']==pytest.approx(-.01)


def test_missing_duplicate_and_unequal_panel_rejected():
    rows=records()
    with pytest.raises(ValueError,match='Complete'):s.select(rows[:-1],['a','b'])
    with pytest.raises(ValueError,match='Duplicate'):s.select(rows+[rows[0]],['a','b'])
    with pytest.raises(ValueError):s.select(rows,['a'])


def test_in_sample_prediction_cannot_select():
    rows=records();rows[0]['fit_match_ids']=['a','other']
    with pytest.raises(ValueError,match='excluded'):s.select(rows,['a','b'])


@pytest.mark.parametrize('key,value',[('expected_net',float('nan')),('expected_net',True),('budget',True),('cost',True),('cost',.002)])
def test_nonfinite_and_undeclared_scenarios_rejected(key,value):
    rows=records();rows[0][key]=value
    with pytest.raises(ValueError):s.select(rows,['a','b'])


def test_forbidden_extra_data_rejected_and_no_mutation():
    rows=records();original=deepcopy(rows);s.select(rows,['b','a']);assert rows==original
    rows[0]['heldout_test_gain']=100
    with pytest.raises(ValueError):s.select(rows,['a','b'])
