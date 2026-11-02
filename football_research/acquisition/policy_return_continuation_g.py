"""Research continuation fitted to a cross-fit policy's signed observed return.

The successor action is selected using pre-query scores, never observed gains.
Targets use training successors only; inference reuses the frozen public-menu
vector. Clipping a fitted policy return is a heuristic, not optimal EVSI.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from . import learned_continuation_g as previous
from . import state_candidate_gain as immediate
from .context_predictor import _digest

VERSION = 'historical-crossfit-policy-return-continuation-g-v1'


def successor_target(teacher, successors, cost):
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not np.isfinite(cost) or cost < 0:
        raise ValueError('Nonnegative finite cost required')
    ordered = sorted(successors, key=lambda e: e['action_id'])
    if len({e['action_id'] for e in ordered}) != len(ordered):
        raise ValueError('Duplicate successor action')
    scores = [{'action_id': e['action_id'], 'predicted_gain': immediate.estimate(teacher, e['query'])}
              for e in ordered]
    selected = max(scores, key=lambda s: s['predicted_gain']) if scores else None
    if selected is None or selected['predicted_gain'] <= cost:
        return {'selected_action_id': None, 'continuation_target': 0., 'scores': scores,
                'reason': 'teacher_stops', 'selected_realized_gain': None}
    edge = next(e for e in ordered if e['action_id'] == selected['action_id'])
    gain = edge['signed_gain']
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not np.isfinite(gain):
        raise ValueError('Finite signed selected-action return required')
    return {'selected_action_id': selected['action_id'], 'continuation_target': float(gain-cost),
            'scores': scores, 'reason': 'teacher_selects_before_return_access',
            'selected_realized_gain': gain}


def fit(rows, base_model, *, alpha=.1):
    if not rows or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Training rows and positive regularization required')
    ids = set(base_model['fit_match_ids'])
    if {r['physical_match_id'] for r in rows} != ids:
        raise ValueError('Training cohort mismatch')
    for r in rows:
        if set(r['teacher_fit_match_ids']) != ids-{r['physical_match_id']}:
            raise ValueError('Teacher must exclude whole target match')
        if not np.isfinite(r['continuation_target']):
            raise ValueError('Finite signed continuation target required')
    counts = Counter(r['physical_match_id'] for r in rows)
    weights = [1/(len(counts)*counts[r['physical_match_id']]) for r in rows]
    x = np.asarray([previous.vector(base_model, r['packet']) for r in rows])
    ridge = Ridge(alpha=alpha, solver='svd').fit(x, [r['continuation_target'] for r in rows], sample_weight=weights)
    return {'schema': VERSION, 'immediate_model': deepcopy(base_model), 'alpha': alpha,
            'coefficients': ridge.coef_.tolist(), 'intercept': float(ridge.intercept_),
            'fit_match_ids': sorted(ids), 'horizon': 2,
            'target': 'signed_observed_return_of_leave_whole_match_out_one_step_policy',
            'inference_clipping': 'nonnegative_fitted_policy_return_heuristic_not_optimal_EVSI',
            'provenance': 'historical_reconstructed_research',
            'production_ready': False, 'online_selection_allowed': False,
            'calibration_status': 'not_established'}


def _inference_view(model):
    if model.get('schema') != VERSION:
        raise ValueError('Policy-return continuation schema differs')
    # Share arithmetic only. Serialized model and target provenance stay distinct.
    return {**model, 'schema': previous.VERSION}


def estimate(model, packet, *, remaining_reads):
    return previous.estimate(_inference_view(model), packet, remaining_reads=remaining_reads)


def choose(model, menu, remaining_reads, cost):
    return previous.choose(_inference_view(model), menu, remaining_reads, cost)


def save(model, path):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump({**model, 'model_sha256': _digest(model)}, f, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    model = json.loads(Path(path).read_text(encoding='utf-8')); digest = model.pop('model_sha256')
    if model.get('schema') != VERSION or _digest(model) != digest:
        raise ValueError('Policy-return model binding differs')
    return model
