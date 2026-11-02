import numpy as np
import pytest
from football_research.acquisition.postread_state_weights import state_weights
from football_research.acquisition.postread_g_use import fit_selector, score
from football_research.acquisition.corpus_rolling_g import match_weights


def test_match_and_depth_mass_and_reachable_horizon():
    mids = ['a']*8+['b']; depths=[0,0,1,1,1,1,2,3,0]
    w=state_weights(mids,depths,mode='match_uniform_depths')
    assert w[:8].sum()==.5 and w[-1]==.5
    for d in range(4):assert sum(v for m,k,v in zip(mids,depths,w) if m=='a' and k==d)==.125
    r=state_weights(mids,depths,mode='match_reachable_depths')
    assert r[7]==0
    np.testing.assert_allclose(r[:8].sum(),.5)


def test_default_model_and_explicit_original_weights_match():
    x=np.array([[0.,2.],[1.,-1.],[2.,3.],[3.,1.]])
    y=[-.2,0.,.3,-.1]; mids=['a','a','b','c']
    kwargs=dict(kind='ridge',alpha=10.,mode='body_state')
    default=fit_selector(x,y,mids,**kwargs)
    weighted=fit_selector(x,y,mids,sample_weights=match_weights(mids),**kwargs)
    np.testing.assert_allclose(score(default,x),score(weighted,x),atol=1e-14,rtol=0)


def test_duplicate_edges_do_not_change_balanced_fit():
    x=np.array([[0.,2.],[1.,-1.],[2.,3.]])
    y=np.array([-.2,0.,.3]); mids=['a','a','b']; depths=[0,1,0]
    kwargs=dict(kind='ridge',alpha=1.,mode='body_state')
    w=state_weights(mids,depths,mode='match_uniform_depths')
    model=fit_selector(x,y,mids,sample_weights=w,**kwargs)
    idx=[0,0,1,2]; mm=[mids[i] for i in idx]; dd=[depths[i] for i in idx]
    duplicate=fit_selector(x[idx],y[idx],mm,sample_weights=state_weights(mm,dd,mode='match_uniform_depths'),**kwargs)
    np.testing.assert_allclose(score(model,x),score(duplicate,x),atol=1e-12,rtol=0)


@pytest.mark.parametrize('weights',[[0,0],[-1,2],[float('nan'),1],[1]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        fit_selector([[0],[1]],[0,.1],['a','b'],kind='ridge',alpha=1,mode='body_state',sample_weights=weights)


def test_unreachable_match_rejected():
    with pytest.raises(ValueError):state_weights(['a'],[3],mode='match_reachable_depths')
