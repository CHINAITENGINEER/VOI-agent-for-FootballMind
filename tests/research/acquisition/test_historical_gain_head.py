from copy import deepcopy
import json
import numpy as np
import pytest
from football_research.acquisition import historical_gain_head as g


def query(title='team news', p=None):
    return {'schema': g.QUERY_SCHEMA, 'probabilities': p or [.4,.3,.3],
            'acquired_documents':0,'visible_candidates':2,'article_age_hours':1.,'title':title}


def rows():
    return [{'physical_match_id':'a','query':query('confirmed lineup'),'signed_gain':.1},
            {'physical_match_id':'b','query':query('ticket sales'),'signed_gain':-.1},
            {'physical_match_id':'c','query':query('team news'),'signed_gain':0.}]


def test_gain_matches_log_loss_and_keeps_negative_and_zero():
    p, q = [.4,.3,.3], [.5,.25,.25]
    assert np.isclose(g.signed_gain(p,q,0), -np.log(.4)+np.log(.5))
    assert g.signed_gain(p,q,1) < 0
    assert g.signed_gain(p,p,2) == 0


@pytest.mark.parametrize('hidden',['body','prediction_after','outcome','signed_gain','effect_class'])
def test_postquery_fields_cannot_enter_feature_contract(hidden):
    q = query(); q[hidden] = 'forbidden'
    with pytest.raises(ValueError,match='contract'): g.numeric(q)


def test_title_vocabulary_train_only_and_prediction_ignores_row_labels():
    r = rows(); model = g.fit(r); before = deepcopy(model)
    g.estimate(model,query('neverobservedtoken'))
    assert model == before and 'neverobservedtoken' not in model['encoder']['terms']
    r[0]['outcome'] = 2; r[0]['body'] = 'unread'
    assert g.estimate(model,r[0]['query']) == g.estimate(model,rows()[0]['query'])


def test_equal_match_weight_invariant_to_replicating_states():
    r=rows(); a=g.fit(r); b=g.fit([r[0]]*9+r[1:])
    assert a['encoder']==b['encoder']
    assert np.allclose(a['coefficients'],b['coefficients'],atol=1e-12)
    assert np.isclose(a['constant_baseline'],b['constant_baseline'])


def test_all_zero_training_is_valid_and_negative_predictions_allowed():
    r=rows()
    for x in r: x['signed_gain']=0.
    model=g.fit(r)
    assert abs(g.estimate(model,query())) < 1e-12
    for x in r: x['signed_gain']=-.2
    assert g.estimate(g.fit(r),query()) < 0


def test_invalid_probability_counts_age_rejected():
    for field,val in [('probabilities',[4,3,3]),('acquired_documents',True),('visible_candidates',-1),('article_age_hours',float('nan'))]:
        q=query();q[field]=val
        with pytest.raises(ValueError):g.numeric(q)


def test_roundtrip_and_tamper(tmp_path):
    model=g.fit(rows());path=tmp_path/'g.json';g.save(model,path)
    assert g.estimate(model,query())==g.estimate(g.load(path),query())
    raw=json.loads(path.read_text(encoding='utf-8'));raw['intercept']+=1
    path.write_text(json.dumps(raw),encoding='utf-8')
    with pytest.raises(ValueError,match='binding'):g.load(path)
