"""Convex text residual control: eliminates hidden-layer collapse as a confound."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp

SCHEMA = 'footballmind-convex-text-residual-f-v1'


def objective(theta, z, base, y, weights, alpha):
    coef = theta.reshape(z.shape[1], 3)
    logits = np.log(base)+z@coef
    loss = weights@(logsumexp(logits, axis=1)-logits[np.arange(len(y)),y])
    grad = softmax(logits,axis=1)
    grad[np.arange(len(y)),y] -= 1
    return float(loss+.5*alpha*(theta@theta)), (z.T@(grad*weights[:,None])).ravel()+alpha*theta


def fit(x, base, y, weights, *, alpha, branch='body', metadata_dimension=19):
    x,base,y,w = np.asarray(x,float),np.asarray(base,float),np.asarray(y,int),np.asarray(weights,float)
    if (x.ndim != 2 or base.shape != (len(x),3) or y.shape != (len(x),)
        or w.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(base).all()
        or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1)
        or np.any(base<=0) or not np.allclose(base.sum(1),1) or np.any((y<0)|(y>2))
        or not np.isfinite(alpha) or alpha<=0 or metadata_dimension > x.shape[1]):
        raise ValueError('Invalid convex f training data')
    if branch == 'body':
        columns = np.arange(11,x.shape[1])
    elif branch == 'metadata':
        columns = np.arange(11,metadata_dimension)
    elif branch == 'bias':
        columns = np.array([],dtype=int)
    else:
        raise ValueError('Unknown evidence branch')
    selected = x[:,columns]; mean = w@selected
    scale = np.maximum(np.sqrt(w@((selected-mean)**2)),.05)
    z = np.column_stack(((selected-mean)/scale,np.ones(len(x))))
    res = minimize(objective,np.zeros(z.shape[1]*3),args=(z,base,y,w,alpha),jac=True,
        method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-11,'gtol':1e-7})
    if not res.success or not np.isfinite(res.x).all():
        raise RuntimeError(str(res.message))
    return dict(schema=SCHEMA,input_dimension=x.shape[1],branch=branch,columns=columns.tolist(),
        mean=mean.tolist(),scale=scale.tolist(),coef=res.x.reshape(-1,3).tolist(),alpha=alpha,
        parameter_count=len(res.x),production_ready=False,historical_version_unknown=True,
        optimizer=dict(success=True,iterations=int(res.nit),objective=float(res.fun)))


def predict(model,x,base,read_counts):
    x,base,counts = np.asarray(x,float),np.asarray(base,float),np.asarray(read_counts)
    if (model.get('schema')!=SCHEMA or x.ndim!=2 or x.shape[1]!=model['input_dimension']
        or base.shape!=(len(x),3) or counts.shape!=(len(x),) or np.any(counts<0)
        or not np.isfinite(x).all() or not np.isfinite(base).all() or np.any(base<=0)
        or not np.allclose(base.sum(1),1)):
        raise ValueError('Invalid convex f prediction input')
    z = np.column_stack(((x[:,model['columns']]-model['mean'])/model['scale'],np.ones(len(x))))
    result = softmax(np.log(base)+(counts>0)[:,None]*(z@np.array(model['coef'])),axis=1)
    result[counts==0] = base[counts==0]
    return result
