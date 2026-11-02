"""Local post-reading evidence selector. g-visible bodies and f-adopted evidence differ."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from football_research.acquisition.corpus_rolling_g import match_weights, prequery_features

SCHEMA = 'footballmind-postread-g-use-v1'


def build_features(*, mode, probabilities, candidate_id, g_seen_ids, f_adopted_ids,
                   menu_ids, embedding_by_id, title_by_id, age_days, source_side,
                   host_index, host_count):
    menu, seen, adopted = list(menu_ids), set(g_seen_ids), list(f_adopted_ids)
    if (len(menu) != len(set(menu)) or len(adopted) != len(set(adopted))
            or not set(adopted) <= seen or not seen <= set(menu)
            or candidate_id not in seen or candidate_id in adopted):
        raise ValueError('Candidate must be g-visible, not yet f-adopted')
    titles = [title_by_id[a] for a in adopted]
    public = prequery_features(probabilities=probabilities, read_count=len(adopted),
        menu_count=len(menu), age_days=age_days, source_side=source_side,
        host_index=host_index, host_count=host_count,
        title_vector=title_by_id[candidate_id], read_title_vectors=titles)
    if mode == 'title_local':
        return public
    if mode not in ('title_state', 'body_state', 'body_only'):
        raise ValueError('Unknown postread representation')
    vectors = title_by_id if mode == 'title_state' else embedding_by_id
    d = np.asarray(vectors[candidate_id], dtype=float)
    current = [np.asarray(vectors[a], dtype=float) for a in adopted]
    if d.shape != (384,) or not np.isfinite(d).all() or any(
            v.shape != (384,) or not np.isfinite(v).all() for v in current):
        raise ValueError('Invalid visible embedding')
    if mode == 'body_only':
        return d.copy()
    state = np.mean(current, axis=0) if current else np.zeros(384)
    sim = max((float(d@v/max(np.linalg.norm(d)*np.linalg.norm(v), 1e-12)) for v in current), default=0.)
    # Public numeric state, candidate/state representations and explicit interactions.
    return np.r_[public[:-384], sim, d, state, d-state, d*state]


def _unpack(theta, d, h):
    n = d*h
    return theta[:n].reshape(d, h), theta[n:n+h], theta[n+h:n+2*h], theta[-1]


def neural_objective(theta, z, y, w, hidden, alpha):
    a, b, v, c = _unpack(theta, z.shape[1], hidden)
    t = np.tanh(z@a+b); pred = t@v+c; err = pred-y
    value = .5*float(w@(err**2))+.5*alpha*float(theta@theta)
    grad = w*err; dt = grad[:, None]*v*(1-t*t)
    return value, np.r_[(z.T@dt).ravel(), dt.sum(0), t.T@grad, grad.sum()]+alpha*theta


def fit_selector(x, gains, match_ids, *, kind, alpha, mode, seed=20261020, hidden=16,
                 sample_weights=None):
    x, y = np.asarray(x, dtype=float), np.asarray(gains, dtype=float)
    if (x.ndim != 2 or y.shape != (len(x),) or len(match_ids) != len(y)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0):
        raise ValueError('Invalid training data')
    w = match_weights(match_ids) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w < 0) or not w.sum() > 0:
        raise ValueError('Invalid selector training weights')
    if sample_weights is not None:
        w = w/w.sum()
    mean = w@x
    scale = np.maximum(np.sqrt(w@((x-mean)**2)), .05); z = (x-mean)/scale
    center = float(w@y); target_scale = max(float(np.sqrt(w@((y-center)**2))), 1e-8)
    yn = (y-center)/target_scale
    model = {'schema': SCHEMA, 'mode': mode, 'kind': kind, 'alpha': alpha,
        'input_dimension': x.shape[1], 'mean': mean.tolist(), 'scale': scale.tolist(),
        'target_center': center, 'target_scale': target_scale,
        'fit_match_ids': sorted(set(match_ids)), 'production_ready': False,
        'decision': 'adopt_into_fixed_f_or_stop', 'candidate_body_input_allowed': mode.startswith('body'),
        'outcome_after_gain_input_allowed': False, 'cost_penalty': 0.}
    if kind == 'ridge':
        # Dual form keeps fitting cheap with many embedding dimensions and few matches.
        zw = z*np.sqrt(w[:, None]); yw = yn*np.sqrt(w)
        coef = zw.T @ np.linalg.solve(zw@zw.T+alpha*np.eye(len(y)), yw)
        model.update(coef=coef.tolist(), parameter_count=len(coef)+1)
    elif kind == 'mlp':
        rng = np.random.default_rng(seed)
        initial = np.r_[rng.normal(0, .2/np.sqrt(z.shape[1]), (z.shape[1], hidden)).ravel(),
                        np.zeros(hidden), rng.normal(0, .05, hidden), 0.]
        res = minimize(neural_objective, initial, args=(z, yn, w, hidden, alpha), jac=True,
            method='L-BFGS-B', options={'maxiter': 6000, 'maxfun': 18000, 'ftol': 1e-10, 'gtol': 1e-6})
        if not res.success or not np.isfinite(res.x).all():
            raise RuntimeError('g optimization failed: '+str(res.message))
        model.update(parameters=res.x.tolist(), hidden=hidden, parameter_count=len(res.x),
                     optimizer={'iterations': int(res.nit), 'success': True})
    else:
        raise ValueError('Unknown local learner')
    return model


def score(model, features):
    x = np.asarray(features, dtype=float)
    if (model.get('schema') != SCHEMA or x.ndim != 2
            or x.shape[1] != model['input_dimension'] or not np.isfinite(x).all()):
        raise ValueError('Invalid selector input')
    z = (x-model['mean'])/model['scale']
    if model['kind'] == 'ridge':
        p = z@np.array(model['coef'])
    elif model['kind'] == 'mlp':
        a, b, v, c = _unpack(np.array(model['parameters']), x.shape[1], model['hidden'])
        p = np.tanh(z@a+b)@v+c
    else:
        raise ValueError('Unknown model kind')
    return p*model['target_scale']+model['target_center']


def select(model, ids, features):
    if not ids:
        return None, []
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate candidate')
    values = score(model, features)
    if len(values) != len(ids):
        raise ValueError('Menu does not match feature rows')
    best = int(np.argmax(values))
    return (ids[best] if values[best] > 0 else None), values.tolist()
