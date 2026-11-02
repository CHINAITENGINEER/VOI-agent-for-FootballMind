"""Research-only pre-query gain regression; no body or outcome input at inference."""
from __future__ import annotations

import itertools
from collections import Counter
import numpy as np

SCHEMA = 'footballmind-corpus-rolling-g-v1'


def subsets(ids):
    ids = tuple(ids)
    if len(ids) != len(set(ids)) or len(ids) > 12:
        raise ValueError('Expected unique bounded article menu')
    return [s for k in range(len(ids)+1) for s in itertools.combinations(ids, k)]


def match_weights(match_ids):
    counts = Counter(match_ids)
    if not counts:
        raise ValueError('Empty training panel')
    return np.array([1/(len(counts)*counts[mid]) for mid in match_ids])


def prequery_features(*, probabilities, read_count, menu_count, age_days,
                      source_side, host_index, host_count, title_vector,
                      read_title_vectors):
    """Explicit query-only arguments. Caller must not substitute unread body vectors."""
    p = np.asarray(probabilities, dtype=float)
    title = np.asarray(title_vector, dtype=float)
    read_titles = np.asarray(read_title_vectors, dtype=float)
    if (p.shape != (3,) or not np.isfinite(p).all() or np.any(p <= 0)
            or not np.isclose(p.sum(), 1) or title.shape != (384,)
            or not np.isfinite(title).all() or read_count < 0
            or read_count >= menu_count or age_days < 0
            or source_side not in (0, 1) or not 0 <= host_index < host_count):
        raise ValueError('Invalid public query state')
    if read_count:
        if read_titles.shape != (read_count, 384) or not np.isfinite(read_titles).all():
            raise ValueError('Read-title state does not match read count')
        norms = np.maximum(np.linalg.norm(read_titles, axis=1)*np.linalg.norm(title), 1e-12)
        similarity = float(np.max(read_titles @ title / norms))
    else:
        if read_titles.size:
            raise ValueError('Unexpected read titles')
        similarity = 0.
    return np.r_[p, -np.sum(p*np.log(p)), p.max(), np.sort(p)[-1]-np.sort(p)[-2],
                 read_count, menu_count-read_count, age_days, source_side, similarity,
                 np.eye(host_count)[host_index], title]


def fit_gain(x, y, match_ids, *, kind='ridge', alpha=1.):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or len(match_ids) != len(y):
        raise ValueError('Training dimensions differ')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0:
        raise ValueError('Nonfinite training data or invalid penalty')
    w = match_weights(match_ids)
    center = float(w @ y)
    mean = w @ x
    scale = np.maximum(np.sqrt(w @ ((x-mean)**2)), .05)
    z = (x-mean)/scale
    if kind == 'ridge':
        coef = np.linalg.solve(z.T @ (w[:, None]*z)+alpha*np.eye(x.shape[1]),
                               z.T @ (w*(y-center)))
    elif kind in ('zero', 'constant'):
        coef = np.zeros(x.shape[1])
        if kind == 'zero':
            center = 0.
    else:
        raise ValueError('Unknown gain model')
    return {'schema': SCHEMA, 'kind': kind, 'alpha': alpha, 'mean': mean.tolist(),
            'scale': scale.tolist(), 'coef': coef.tolist(), 'intercept': center,
            'input_dimension': x.shape[1], 'fit_match_ids': sorted(set(match_ids)),
            'target': 'same_f_log_loss_before_minus_after',
            'feature_layout': 'p3_entropy_max_margin_read_remaining_age_side_title_similarity_host_onehot_title384',
            'unread_body_input': False, 'production_ready': False,
            'provenance': 'historical_reconstruction_rolling_f_out_of_fit'}


def predict_gain(model, x):
    x = np.asarray(x, dtype=float)
    if (model.get('schema') != SCHEMA or x.ndim != 2
            or x.shape[1] != model['input_dimension'] or not np.isfinite(x).all()):
        raise ValueError('Invalid gain inference input')
    return ((x-np.array(model['mean']))/np.array(model['scale'])) @ np.array(model['coef'])+model['intercept']


def choose_action(model, action_ids, query_vectors, cost):
    if cost < 0 or not np.isfinite(cost) or len(set(action_ids)) != len(action_ids):
        raise ValueError('Invalid action menu or cost')
    if not action_ids:
        return None, []
    values = predict_gain(model, query_vectors)-cost
    if len(values) != len(action_ids):
        raise ValueError('Action/vector dimensions differ')
    i = int(np.argmax(values))
    return (action_ids[i] if values[i] > 0 else None), values.tolist()
