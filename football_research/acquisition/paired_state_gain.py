"""Research g: supervise both fixed-f gain and within-match gain changes.

No extra inference inputs or parameters. Pair labels use training transitions;
inference still accepts only the strict pre-query state/candidate features.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from . import state_candidate_gain as base
from .context_predictor import _digest

VERSION = 'historical-paired-state-gain-v1'


def build_pairs(rows):
    graph = defaultdict(dict)
    for i, r in enumerate(rows):
        if r['action_id'] in graph[r['state_id']]:
            raise ValueError('Duplicate state/action')
        graph[r['state_id']][r['action_id']] = i
    pairs = []
    for acquired in rows:
        before = graph[acquired['state_id']]
        following = graph.get(acquired['next_state_id'], {})
        for action, after_index in sorted(following.items()):
            if action == acquired['action_id'] or action not in before:
                continue
            before_index = before[action]
            a, b = rows[before_index], rows[after_index]
            if a['physical_match_id'] != acquired['physical_match_id'] or b['physical_match_id'] != a['physical_match_id']:
                raise ValueError('Pair crosses physical matches')
            if a['body_input_id'] != b['body_input_id']:
                raise ValueError('Same action must retain target body identity')
            delta = float(b['signed_gain']-a['signed_gain'])
            if not np.isfinite(delta):
                raise ValueError('Finite pair target required')
            pairs.append({'physical_match_id': a['physical_match_id'], 'before_index': before_index,
                          'after_index': after_index, 'before_state_id': a['state_id'],
                          'after_state_id': b['state_id'], 'action_id': action,
                          'acquired_action_id': acquired['action_id'], 'gain_delta': delta})
    return pairs


def fit(rows, *, pair_weight=1., alpha=.1):
    if isinstance(pair_weight, bool) or not np.isfinite(pair_weight) or pair_weight < 0:
        raise ValueError('Nonnegative pair weight required')
    model = base.fit(rows, alpha=alpha)
    pairs = build_pairs(rows)
    if pair_weight and pairs:
        x = np.asarray([base.vector(model, r['query']) for r in rows])
        y = np.asarray([r['signed_gain'] for r in rows])
        counts = Counter(r['physical_match_id'] for r in rows)
        weights = np.array([1/(len(counts)*counts[r['physical_match_id']]) for r in rows])
        pair_counts = Counter(p['physical_match_id'] for p in pairs)
        # Each match has the same total weight in each objective; pairs are
        # correlated comparisons, never claimed to be independent examples.
        pair_weights = np.array([pair_weight/(len(pair_counts)*pair_counts[p['physical_match_id']]) for p in pairs])
        d = x.shape[1]
        absolute = np.c_[x, np.ones(len(x))]
        differences = np.array([np.r_[x[p['after_index']]-x[p['before_index']], 0.] for p in pairs])
        targets = np.array([p['gain_delta'] for p in pairs])
        penalty = np.c_[np.sqrt(alpha)*np.eye(d), np.zeros(d)]
        design = np.vstack([absolute*np.sqrt(weights[:, None]), differences*np.sqrt(pair_weights[:, None]), penalty])
        response = np.r_[y*np.sqrt(weights), targets*np.sqrt(pair_weights), np.zeros(d)]
        coefficient = np.linalg.lstsq(design, response, rcond=None)[0]
        model['coefficients'] = coefficient[:-1].tolist(); model['intercept'] = float(coefficient[-1])
    model.update(schema=VERSION, pair_weight=float(pair_weight), training_pairs=len(pairs),
                 target='absolute_fixed_f_gain_and_same_action_state_difference',
                 production_ready=False, empirical_selection_allowed=False)
    return model


def estimate(model, query):
    if model.get('schema') != VERSION:
        raise ValueError('Paired state g schema differs')
    return base.estimate({**model, 'schema': base.VERSION}, query)


def save(model, path):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump({**model, 'model_sha256': _digest(model)}, f, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    model = json.loads(Path(path).read_text(encoding='utf-8')); digest = model.pop('model_sha256')
    if model.get('schema') != VERSION or _digest(model) != digest:
        raise ValueError('Paired state g binding differs')
    return model
