import json
import numpy as np
import pytest
from football_research.acquisition.convex_text_f import objective,fit,predict


def sample():
    rng=np.random.default_rng(21)
    x=rng.normal(size=(18,23)); y=np.arange(18)%3
    x[:,19]=y-1; x[:,20]=(y==1)
    return x,np.tile([.4,.3,.3],(18,1)),y,np.full(18,1/18)


def test_gradient():
    rng=np.random.default_rng(3); z=rng.normal(size=(6,4)); theta=rng.normal(size=12)
    base=np.tile([.4,.3,.3],(6,1)); y=np.arange(6)%3; w=np.full(6,1/6)
    _,grad=objective(theta,z,base,y,w,.2)
    num=[]
    for i in range(12):
        delta=np.zeros(12); delta[i]=1e-6
        num.append((objective(theta+delta,z,base,y,w,.2)[0]-objective(theta-delta,z,base,y,w,.2)[0])/2e-6)
    np.testing.assert_allclose(grad,num,atol=1e-8,rtol=1e-6)


@pytest.mark.parametrize('branch',['body','metadata','bias'])
def test_zero_evidence_identity_and_serialization(branch):
    x,b,y,w=sample(); m=fit(x,b,y,w,alpha=1,branch=branch)
    np.testing.assert_array_equal(predict(m,x,b,np.zeros(18)),b)
    np.testing.assert_array_equal(predict(m,x,b,np.ones(18)),predict(json.loads(json.dumps(m)),x,b,np.ones(18)))


def test_body_content_can_change_prediction_and_controls_do_not_use_it():
    x,b,y,w=sample(); other=x.copy(); other[:,19:]*=-1
    for branch in ['metadata','bias']:
        m=fit(x,b,y,w,alpha=.1,branch=branch)
        np.testing.assert_array_equal(predict(m,x,b,np.ones(18)),predict(m,other,b,np.ones(18)))
    m=fit(x,b,y,w,alpha=.1,branch='body')
    assert np.max(abs(predict(m,x,b,np.ones(18))-predict(m,other,b,np.ones(18)))) > .01
