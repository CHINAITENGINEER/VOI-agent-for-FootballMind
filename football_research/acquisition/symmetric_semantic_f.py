"""Research residual f sharing semantic effects across home/away roles.

Evidence routing and temporal resolution reuse the frozen reviewed contracts.
Team swap is an inductive bias on residuals, not a claim of equal home advantage:
the supplied prior retains that information. No lexical/player-name coefficients.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp

from . import entity_evidence_updates as temporal
from . import scoped_reviewed_f as scoped
from .context_predictor import _digest

VERSION = 'symmetric-semantic-residual-f-v1'
FEATURES = tuple(sorted(
    ['direct:' + p for p in scoped.PREDICATES]
    + ['selection:' + c for c in scoped.SELECTION]
    + ['context:' + c + ':' + s for c in scoped.CONTEXT
       for s in ('this_match', 'team_context')]))


def public_features(row):
    p = np.asarray(row['prior'], dtype=float)
    if p.shape != (3,) or not np.all(np.isfinite(p)) or np.any(p <= 0) or abs(p.sum()-1) > 1e-10:
        raise ValueError('Positive normalized prior required')
    used = row['used_input_ids']
    if any(e['input_id'] not in used for e in row['evidence_events']):
        raise ValueError('Evidence must have been observed in this state')
    routed, _ = scoped.route(temporal.resolve(row)[0])
    present = {'home': set(), 'away': set()}
    for e in routed['evidence']:
        lane = scoped.field(e['atoms'], 'lane:')
        if lane == 'direct':
            key = 'direct:' + scoped.field(e['atoms'], 'predicate:')
        elif lane == 'selection':
            key = 'selection:' + scoped.field(e['atoms'], 'concept:')
        elif lane == 'context':
            key = 'context:' + scoped.field(e['atoms'], 'concept:') + ':' + scoped.field(e['atoms'], 'scope:')
        else:
            raise ValueError('Unknown routed lane')
        present[e['team']].add(key)
    home = np.asarray([float(k in present['home']) for k in FEATURES])
    away = np.asarray([float(k in present['away']) for k in FEATURES])
    return p, home-away, home+away


def validate_model(model):
    if model.get('schema') != VERSION or model.get('feature_names') != list(FEATURES):
        raise ValueError('Symmetric f schema/feature contract differs')
    if model.get('routing_contract') != scoped.rule_manifest():
        raise ValueError('Evidence routing contract differs')
    w = np.asarray(model['coefficients'], dtype=float)
    if w.shape != (2, len(FEATURES)) or not np.all(np.isfinite(w)):
        raise ValueError('Invalid symmetric f parameters')
    return w


def predict(model, row):
    w = validate_model(model)
    p, difference, total = public_features(row)
    if not np.any(difference) and not np.any(total):
        return p.copy()
    direction, draw_shift = difference @ w[0], total @ w[1]
    return softmax(np.log(p) + [direction+draw_shift, -2*draw_shift, -direction+draw_shift])


def fit(rows, regularization=10.):
    if not rows or isinstance(regularization, bool) or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('Rows and positive finite regularization required')
    labels = {}
    for r in rows:
        mid, y = r['physical_match_id'], r['outcome']
        if type(y) is not int or y not in (0, 1, 2) or (mid in labels and labels[mid] != y):
            raise ValueError('Consistent per-match outcome required')
        labels[mid] = y
    values = [public_features(r) for r in rows]
    p, d, s = (np.asarray([v[i] for v in values]) for i in range(3))
    y = np.asarray([r['outcome'] for r in rows])
    counts = Counter(r['physical_match_id'] for r in rows)
    weights = np.asarray([1/(len(counts)*counts[r['physical_match_id']]) for r in rows])
    prior_logits = np.log(p)

    def objective(theta):
        w = theta.reshape(2, len(FEATURES))
        a, b = d @ w[0], s @ w[1]
        logits = prior_logits + np.column_stack((a+b, -2*b, -a+b))
        loss = np.dot(weights, logsumexp(logits, axis=1)-logits[np.arange(len(y)), y])
        residual = softmax(logits, axis=1)-np.eye(3)[y]
        residual *= weights[:, None]
        grad = np.stack((d.T @ (residual[:, 0]-residual[:, 2]),
                         s.T @ (residual[:, 0]-2*residual[:, 1]+residual[:, 2])))
        return float(loss + .5*regularization*np.sum(w*w)), (grad+regularization*w).ravel()

    result = minimize(objective, np.zeros(2*len(FEATURES)), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-9})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError('Symmetric f optimization failed: ' + str(result.message))
    return {'schema': VERSION, 'feature_names': list(FEATURES),
            'routing_contract': scoped.rule_manifest(),
            'coefficients': result.x.reshape(2, len(FEATURES)).tolist(),
            'regularization': regularization, 'fit_match_ids': sorted(counts),
            'optimizer': {'success': bool(result.success), 'iterations': int(result.nit),
                          'objective': float(result.fun)},
            'parameter_count': 2*len(FEATURES),
            'representation': 'typed semantic presence, shared home-away residual effects',
            'provenance': 'historical_reconstructed_research',
            'production_ready': False, 'calibration_status': 'not_established'}


def save(model, path):
    validate_model(model)
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump({**deepcopy(model), 'model_sha256': _digest(model)}, f, ensure_ascii=False,
                  indent=2, allow_nan=False)


def load(path):
    model = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = model.pop('model_sha256')
    if digest != _digest(model):
        raise ValueError('Symmetric f digest differs')
    validate_model(model)
    return model
