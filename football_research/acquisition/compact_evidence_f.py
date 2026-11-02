"""Four-parameter research ablation; not a runtime predictor or replacement f.

Uses only availability evidence already accepted by the frozen semantic bank.
It deliberately tests reduced capacity, not broader semantic understanding.
All statuses share home/away antisymmetric coefficients; no effect sign is
hand assigned. A mention is not a verified player count or a severity score.
"""
from collections import Counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .entity_semantic_f import vector as validate_projection
from .semantic_f import FEATURES

VERSION = 'compact-availability-antisymmetric-f-research-v1'
STATUSES = ('unavailable', 'available', 'doubtful', 'recovering')
REGULARIZATIONS = (None, 10., 1., .1)


def features(state):
    validate_projection(state)
    values = {}
    for team in ('home', 'away'):
        values[team] = []
        for status in STATUSES:
            i = FEATURES.index(f'{team}:availability:{status}')
            # Union of title and acquired-body evidence; duplicate mentions do
            # not alter this ablation's binary support indicator.
            values[team].append(float(state['title'][i] > 0 or state['novel_body'][i] > 0))
    return np.asarray(values['home']) - np.asarray(values['away'])


def predict(beta, state):
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (4,) or not np.all(np.isfinite(beta)):
        raise ValueError('Four finite coefficients required')
    z = float(features(state) @ beta)
    return softmax(np.log(state['prior']) + np.array([z, 0., -z]))


def fit(rows, regularization):
    if not rows:
        raise ValueError('Training rows required')
    counts, labels = Counter(), {}
    for row in rows:
        mid, outcome = row['physical_match_id'], row['outcome']
        if type(outcome) is not int or outcome not in (0, 1, 2):
            raise ValueError('Outcome must be an integer class')
        if row['state']['physical_match_id'] != mid or (mid in labels and labels[mid] != outcome):
            raise ValueError('Consistent match identity and label required')
        labels[mid] = outcome
        counts[mid] += 1
    x = np.asarray([features(r['state']) for r in rows])
    logprior = np.log([r['state']['prior'] for r in rows])
    y = np.asarray([r['outcome'] for r in rows])
    weights = np.asarray([1. / (len(counts) * counts[r['physical_match_id']]) for r in rows])
    direction = np.array([1., 0., -1.])
    if regularization is not None and (not np.isfinite(regularization) or regularization <= 0):
        raise ValueError('Positive regularization or explicit base-only None required')

    def objective(beta):
        logits = logprior + (x @ beta)[:, None] * direction
        loss = logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]
        residual = softmax(logits, axis=1) @ direction - direction[y]
        lam = regularization or 0.
        return (float(weights @ loss + .5 * lam * (beta @ beta)),
                x.T @ (weights * residual) + lam * beta)

    if regularization is None:
        beta = np.zeros(4)
        return beta.tolist(), {'success': True, 'method': 'base_only', 'objective': objective(beta)[0]}
    result = minimize(objective, np.zeros(4), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'gtol': 1e-9, 'ftol': 1e-12})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f'Compact f optimization failed: {result.message}')
    return result.x.tolist(), {'success': True, 'method': 'L-BFGS-B',
                               'objective': float(result.fun), 'iterations': int(result.nit)}
