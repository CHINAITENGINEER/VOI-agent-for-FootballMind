import json
import numpy as np
import pytest
from football_research.acquisition.postread_g_use import build_features, neural_objective, fit_selector, score, select


def arguments():
    rng = np.random.default_rng(7)
    vectors = {a: rng.normal(size=384) for a in ['a', 'b']}
    return dict(mode='body_state', probabilities=[.5,.3,.2], candidate_id='b',
        g_seen_ids=['a','b'], f_adopted_ids=[], menu_ids=['a','b'],
        embedding_by_id=vectors, title_by_id={a:v.copy() for a,v in vectors.items()},
        age_days=1., source_side=0, host_index=0, host_count=2)


def test_seen_is_not_adopted_and_state_matters():
    args = arguments()
    before = build_features(**args)
    args['f_adopted_ids'] = ['a']
    assert not np.array_equal(before, build_features(**args))
    args['g_seen_ids'] = ['b']
    with pytest.raises(ValueError):
        build_features(**args)


def test_body_changes_only_fullbody_inputs():
    args = arguments()
    local = build_features(**{**args, 'mode':'title_local'})
    body = build_features(**args)
    args['embedding_by_id']['b'] *= -1
    assert np.array_equal(local, build_features(**{**args, 'mode':'title_local'}))
    assert not np.array_equal(body, build_features(**args))


def test_analytic_gradient():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(5,3)); y = rng.normal(size=5); w = np.full(5,.2)
    theta = rng.normal(size=3*2+2+2+1)
    _, gradient = neural_objective(theta,z,y,w,2,.1)
    numerical = []
    for j in range(len(theta)):
        delta = np.zeros_like(theta); delta[j] = 1e-6
        numerical.append((neural_objective(theta+delta,z,y,w,2,.1)[0]-
            neural_objective(theta-delta,z,y,w,2,.1)[0])/2e-6)
    np.testing.assert_allclose(gradient,numerical,atol=1e-8,rtol=1e-6)


@pytest.mark.parametrize('kind',['ridge','mlp'])
def test_negative_samples_and_saved_model(kind):
    x = np.array([[-1.,0.],[0.,1.],[1.,0.],[2.,1.]])
    model = fit_selector(x,[-.2,0.,.1,.3],['m1','m1','m2','m3'],
        kind=kind,alpha=.1,mode='body_state',hidden=2)
    assert model['fit_match_ids'] == ['m1','m2','m3']
    np.testing.assert_array_equal(score(model,x),score(json.loads(json.dumps(model)),x))
    assert not model['outcome_after_gain_input_allowed']


def test_stop_is_inference_not_training_filter():
    x = np.ones((3,2))
    model = fit_selector(x,[-.1,-.1,-.1],['a','b','c'],kind='ridge',alpha=1.,mode='body_only')
    selected, values = select(model,['x'],x[:1])
    assert selected is None and values[0] < 0
    assert len(model['fit_match_ids']) == 3
