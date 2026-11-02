"""Research two-step g with cross-fitted continuation targets.

An auxiliary head predicts E[max(0, max_b g1(s', b)-cost) | s, a, menu].
Its targets may use successor states from TRAINING matches only. At inference
the head sees the current public menu; it never opens an actual successor.
This is fitted approximate policy continuation, not a calibrated EVSI model.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from . import state_candidate_gain as immediate
from .context_predictor import _digest

VERSION = 'historical-crossfit-continuation-g-v1'
PACKET_FIELDS = {'candidate', 'menu', 'cost_loss_units'}
ACTION_FIELDS = {'action_id', 'query'}


def validate_packet(base_model, packet):
    if set(packet) != PACKET_FIELDS:
        raise ValueError('Exact current-menu continuation contract required')
    cost = packet['cost_loss_units']
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not np.isfinite(cost) or cost < 0:
        raise ValueError('Nonnegative finite cost required')
    menu = packet['menu']; candidate = packet['candidate']
    if not isinstance(menu, list) or not menu or not isinstance(candidate, dict) or set(candidate) != ACTION_FIELDS:
        raise ValueError('Current candidate and nonempty visible menu required')
    for action in menu:
        if (not isinstance(action, dict) or set(action) != ACTION_FIELDS
                or not isinstance(action['action_id'], str) or not action['action_id']):
            raise ValueError('Exact pre-query action contract required')
        immediate.vector(base_model, action['query'])
    ids = [a['action_id'] for a in menu]
    if len(ids) != len(set(ids)) or candidate['action_id'] not in ids:
        raise ValueError('Unique visible action binding required')
    if candidate != menu[ids.index(candidate['action_id'])]:
        raise ValueError('Candidate differs from frozen visible menu')
    # All candidates belong to the same currently visible state.
    public = candidate['query']['public']
    for action in menu:
        p = action['query']['public']
        if any(p[k] != public[k] for k in ('probabilities', 'acquired_documents', 'visible_candidates')):
            raise ValueError('Menu candidates must share the current state')
    if public['visible_candidates'] != len(menu):
        raise ValueError('Visible menu count differs')


def vector(base_model, packet):
    validate_packet(base_model, packet)
    candidate = packet['candidate']; menu = packet['menu']
    own = immediate.vector(base_model, candidate['query'])
    other = [immediate.vector(base_model, a['query']) for a in menu if a['action_id'] != candidate['action_id']]
    pooled = np.mean(other, axis=0) if other else np.zeros_like(own)
    return np.r_[own, pooled, (len(menu)-1)/3, packet['cost_loss_units']/.01]


def fit(rows, base_model, *, alpha=.1):
    if not rows or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Training rows and positive regularization required')
    ids = set(base_model['fit_match_ids'])
    if {r['physical_match_id'] for r in rows} != ids:
        raise ValueError('Continuation and immediate training cohorts must match')
    for row in rows:
        mid = row['physical_match_id']; teacher_ids = set(row['teacher_fit_match_ids'])
        if mid in teacher_ids or teacher_ids != ids-{mid}:
            raise ValueError('Continuation target teacher must exclude the whole target match')
        if not np.isfinite(row['continuation_target']) or row['continuation_target'] < 0:
            raise ValueError('Nonnegative finite continuation target required')
        vector(base_model, row['packet'])
    counts = Counter(r['physical_match_id'] for r in rows)
    weights = [1/(len(counts)*counts[r['physical_match_id']]) for r in rows]
    ridge = Ridge(alpha=alpha, solver='svd').fit(
        np.asarray([vector(base_model, r['packet']) for r in rows]),
        np.asarray([r['continuation_target'] for r in rows]), sample_weight=weights)
    return {'schema': VERSION, 'immediate_model': deepcopy(base_model),
            'fit_match_ids': sorted(ids), 'alpha': alpha,
            'coefficients': ridge.coef_.tolist(), 'intercept': float(ridge.intercept_),
            'target': 'leave_match_out_predicted_one_step_successor_net_value_with_stop',
            'horizon': 2, 'provenance': 'historical_reconstructed_research',
            'production_ready': False, 'online_selection_allowed': False,
            'calibration_status': 'not_established'}


def estimate(model, packet, *, remaining_reads):
    if model.get('schema') != VERSION:
        raise ValueError('Continuation model contract differs')
    if type(remaining_reads) is not int or remaining_reads < 1:
        raise ValueError('Positive remaining read budget required')
    base = model['immediate_model']; x = vector(base, packet)
    w = np.asarray(model['coefficients'], float)
    if x.shape != w.shape or not np.all(np.isfinite(w)) or not np.isfinite(model['intercept']):
        raise ValueError('Invalid continuation parameters')
    gain = immediate.estimate(base, packet['candidate']['query'])
    raw = float(x@w+model['intercept'])
    future = max(0., raw) if remaining_reads >= 2 and len(packet['menu']) >= 2 else 0.
    return {'immediate_gain': gain, 'current_cost': packet['cost_loss_units'],
            'raw_continuation_estimate': raw, 'continuation_net_value': future,
            'two_step_net_value': gain-packet['cost_loss_units']+future,
            'horizon': min(2, remaining_reads)}


def choose(model, menu, remaining_reads, cost):
    if type(remaining_reads) is not int or remaining_reads < 0:
        raise ValueError('Nonnegative remaining budget required')
    if not menu or remaining_reads == 0:
        return {'action_id': None, 'reason': 'no_feasible_candidate_or_budget', 'candidates': []}
    scored = [{'action_id': a['action_id'], **estimate(model,
        {'candidate': a, 'menu': menu, 'cost_loss_units': cost}, remaining_reads=remaining_reads)}
        for a in sorted(menu, key=lambda a: a['action_id'])]
    best = max(scored, key=lambda s: s['two_step_net_value'])
    take = best['two_step_net_value'] > 0
    return {'action_id': best['action_id'] if take else None,
            'reason': 'positive_learned_two_step_net_value' if take else 'nonpositive_learned_two_step_net_value',
            'candidates': scored}


def save(model, path):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump({**model, 'model_sha256': _digest(model)}, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    model = json.loads(Path(path).read_text(encoding='utf-8')); digest = model.pop('model_sha256')
    if model.get('schema') != VERSION or _digest(model) != digest:
        raise ValueError('Continuation model binding differs')
    return model
