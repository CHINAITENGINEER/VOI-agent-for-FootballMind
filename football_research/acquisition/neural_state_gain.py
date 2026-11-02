"""Small CPU MLP for fixed-f signed information gain; research only.

Keeps the existing pre-query representation. No outcomes/after/body are read by
inference. Output normalization is fitted on training matches only. This is a
neural regressor, not an LLM and not an automatically activated online policy.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import state_candidate_gain as linear
from .context_predictor import _digest

VERSION = 'historical-neural-state-gain-v1'


def unpack(theta, dimension):
    sizes = [(dimension, 16), (16,), (16, 8), (8,), (8, 1), (1,)]
    arrays, start = [], 0
    for shape in sizes:
        stop = start + int(np.prod(shape))
        arrays.append(np.asarray(theta[start:stop]).reshape(shape))
        start = stop
    if start != len(theta):
        raise ValueError('Neural parameter dimension differs')
    return arrays


def forward(arrays, x):
    w1, b1, w2, b2, w3, b3 = arrays
    first = np.tanh(x@w1+b1)
    second = np.tanh(first@w2+b2)
    return (second@w3+b3).ravel(), first, second


def objective(theta, x, target, weights, alpha):
    arrays = unpack(theta, x.shape[1])
    w1, b1, w2, b2, w3, b3 = arrays
    prediction, first, second = forward(arrays, x)
    error = prediction-target
    output_grad = (weights*error)[:, None]
    second_grad = (output_grad@w3.T)*(1-second**2)
    first_grad = (second_grad@w2.T)*(1-first**2)
    grads = [x.T@first_grad+alpha*w1, first_grad.sum(axis=0),
             first.T@second_grad+alpha*w2, second_grad.sum(axis=0),
             second.T@output_grad+alpha*w3, output_grad.sum(axis=0)]
    loss = .5*np.dot(weights, error**2)+.5*alpha*sum(np.sum(w*w) for w in (w1, w2, w3))
    return float(loss), np.concatenate([v.ravel() for v in grads])


def validate(model):
    if model.get('schema') != VERSION or model.get('hidden_layers') != [16, 8]:
        raise ValueError('Neural schema/architecture differs')
    if model.get('activation') != 'tanh' or model.get('query_schema') != linear.QUERY_SCHEMA:
        raise ValueError('Neural input/activation differs')
    if model.get('relation_features') != list(linear.FEATURES):
        raise ValueError('Relation contract differs')
    dim = 7+len(model['encoder']['terms'])+len(linear.FEATURES)
    if model.get('input_dimension') != dim:
        raise ValueError('Input dimension differs')
    theta = np.asarray(model['parameters'], dtype=float)
    if theta.ndim != 1 or not np.all(np.isfinite(theta)):
        raise ValueError('Invalid neural parameters')
    arrays = unpack(theta, dim)
    if not np.isfinite(model['target_mean']) or not np.isfinite(model['target_scale']) or model['target_scale'] <= 0:
        raise ValueError('Invalid output normalization')
    return arrays


def fit(rows, representation, *, seed=20261020, alpha=.01):
    if not rows or isinstance(alpha, bool) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Training rows and positive regularization required')
    ids = sorted({r['physical_match_id'] for r in rows})
    if ids != sorted(representation['fit_match_ids']):
        raise ValueError('Encoder must be fitted on exactly the same training matches')
    y = np.asarray([r['signed_gain'] for r in rows], dtype=float)
    if not np.all(np.isfinite(y)) or any(isinstance(r['signed_gain'], bool) for r in rows):
        raise ValueError('Finite signed gains required')
    x = np.asarray([linear.vector(representation, r['query']) for r in rows])
    counts = Counter(r['physical_match_id'] for r in rows)
    weights = np.asarray([1/(len(counts)*counts[r['physical_match_id']]) for r in rows])
    mean = float(weights@y)
    scale = max(float(np.sqrt(weights@((y-mean)**2))), 1e-8)
    rng = np.random.default_rng(seed)
    initial = []
    for n, m in ((x.shape[1], 16), (16, 8), (8, 1)):
        initial += [rng.normal(0, np.sqrt(2/(n+m)), (n, m)).ravel(), np.zeros(m)]
    result = minimize(objective, np.concatenate(initial), args=(x, (y-mean)/scale, weights, alpha),
                      jac=True, method='L-BFGS-B', options={'maxiter': 10000, 'maxfun': 30000, 'ftol': 1e-10, 'gtol': 1e-6})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError('Neural optimization failed: '+str(result.message))
    model = {'schema': VERSION, 'encoder': deepcopy(representation['encoder']),
             'query_schema': linear.QUERY_SCHEMA, 'relation_features': list(linear.FEATURES),
             'input_dimension': x.shape[1], 'hidden_layers': [16, 8], 'activation': 'tanh',
             'parameters': result.x.tolist(), 'parameter_count': len(result.x),
             'target_mean': mean, 'target_scale': scale, 'fit_match_ids': ids,
             'seed': seed, 'alpha': alpha, 'weighting': 'equal_match_then_equal_transition',
             'objective': 'normalized_signed_gain_weighted_MSE_plus_kernel_L2',
             'optimizer': {'success': bool(result.success), 'iterations': int(result.nit),
                           'objective': float(result.fun), 'termination': str(result.message),
                           'gradient_max_abs': float(np.max(np.abs(result.jac)))},
             'provenance': 'historical_reconstructed_fixed_f_research',
             'production_ready': False, 'online_selection_allowed': False,
             'calibration_status': 'not_established'}
    if representation.get('f_labeler_sha256'):
        model['f_labeler_sha256'] = representation['f_labeler_sha256']
    validate(model)
    return model


def estimate(model, query):
    arrays = validate(model)
    x = linear.vector(model, query)
    value = forward(arrays, x[None, :])[0][0]*model['target_scale']+model['target_mean']
    if not np.isfinite(value):
        raise ValueError('Nonfinite predicted gain')
    return float(value)


def save(model, path):
    validate(model)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump({**model, 'model_sha256': _digest(model)}, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    model = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = model.pop('model_sha256')
    if digest != _digest(model):
        raise ValueError('Neural model digest differs')
    validate(model)
    return model
