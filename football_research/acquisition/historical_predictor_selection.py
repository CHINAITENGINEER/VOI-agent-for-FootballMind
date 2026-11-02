"""A-only physical-match cross-validation of the existing role-text predictor.

This module selects shrinkage, not a new representation. Every fold learns its
own physical-game TF-IDF and smoothed frequency offset. B/C outcomes and live
returns are deliberately absent from the API. The returned predictor is a
kernel for the historical combined artifact, not a standalone deployment model.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.feature_extraction.text import TfidfVectorizer

from .context_predictor import ENCODER_VERSION, NUMERIC, PARAMETERS, _digest, _grams
from .fixture_roles import ROLES
from .role_context_predictor import RoleContextPredictor


VERSION = "A-physical-match-loo-role-offset-selection-v1"
REGULARIZATIONS = (20., 10., 1., .1, .01, .001, .0001)
TIE_TOLERANCE = 1e-12


def _base(rows, pseudocount):
    labels = {row['physical_match_id']: row['outcome'] for row in rows}
    counts = Counter(labels.values())
    return [(counts[y]+pseudocount)/(len(labels)+3*pseudocount) for y in range(3)]


def _prepare(rows):
    """Validate supplied states without creating/removing training examples."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('A-only predictor rows must be a nonempty sequence')
    facts, prepared = {}, []
    for original in rows:
        row = deepcopy(original)
        mid, y, state = row.get('physical_match_id'), row.get('outcome'), row.get('state')
        if not isinstance(mid, str) or not mid or type(y) is not int or y not in (0, 1, 2):
            raise ValueError('Independent physical match and three-class outcome required')
        if not isinstance(state, dict) or state.get('prior') is not None:
            raise ValueError('A-only selection requires missing prior; offsets are fold-trained')
        if mid in facts and facts[mid] != y:
            raise ValueError('One physical match cannot have conflicting outcomes')
        facts[mid] = y
        if (set(state.get('role_texts', {})) != set(ROLES)
                or not isinstance(state.get('texts'), list)
                or not isinstance(state.get('sources'), list)
                or len(state.get('numeric', [])) != len(NUMERIC)
                or not np.all(np.isfinite(state['numeric']))):
            raise ValueError('Prepared role-text projection is required')
        for texts in [state['texts'], *state['role_texts'].values()]:
            if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
                raise ValueError('Projected text channels must contain actual strings')
        prepared.append(row)
    return prepared


def _design(rows):
    """Exactly the original physical-game encoder and objective matrices."""
    model = RoleContextPredictor()
    per_match = defaultdict(set)
    for row in rows:
        per_match[row['physical_match_id']].update(row['state']['texts'])
    corpus = [sorted(per_match[mid]) for mid in sorted(per_match)]
    if any(any(_grams(texts)) for texts in corpus):
        vectorizer = TfidfVectorizer(analyzer=_grams, max_features=PARAMETERS['max_features'], sublinear_tf=True)
        vectorizer.fit(corpus)
        vocabulary, idf = vectorizer.get_feature_names_out().tolist(), vectorizer.idf_.tolist()
    else:
        vocabulary, idf = [], []
    model.encoder = {'version': ENCODER_VERSION, 'vocabulary': vocabulary, 'idf': idf,
                     'numeric_names': list(NUMERIC),
                     'sources': sorted({s for row in rows for s in row['state']['sources']})}
    x = np.asarray([model._vector(row['state'])[0] for row in rows])
    labels = np.asarray([row['outcome'] for row in rows])
    offsets = np.log(np.asarray([row['state']['prior'] for row in rows]))
    sizes = Counter(row['physical_match_id'] for row in rows)
    weights = np.asarray([1./(len(sizes)*sizes[row['physical_match_id']]) for row in rows])
    return model, x, labels, offsets, weights


def _optimize(design, regularization):
    model, x, labels, offsets, weights = design
    fitted = RoleContextPredictor()
    fitted.encoder = deepcopy(model.encoder)
    if regularization is None:
        coefficients = np.zeros((x.shape[1], 3))
        objective = float(weights @ (logsumexp(offsets, axis=1)-offsets[np.arange(len(labels)), labels]))
        info = {'method': 'base_only_no_optimizer', 'success': True, 'iterations': 0,
                'objective': objective, 'regularization': None}
    else:
        lam = float(regularization)
        if not math.isfinite(lam) or lam <= 0:
            raise ValueError('Positive finite regularization required')

        def objective(flat):
            coef = flat.reshape(x.shape[1], 3)
            logits = offsets+x@coef
            loss = weights@(logsumexp(logits, axis=1)-logits[np.arange(len(labels)), labels])
            difference = softmax(logits, axis=1)
            difference[np.arange(len(labels)), labels] -= 1.
            return float(loss+lam/2*np.sum(coef*coef)), (x.T@(difference*weights[:, None])+lam*coef).ravel()

        result = minimize(objective, np.zeros(x.shape[1]*3), jac=True, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9})
        if not result.success or not np.all(np.isfinite(result.x)):
            raise ValueError('Role-text predictor optimizer did not converge: '+str(result.message))
        coefficients = result.x.reshape(x.shape[1], 3)
        info = {'method': 'scipy-L-BFGS-B', 'success': True, 'iterations': int(result.nit),
                'objective': float(result.fun), 'regularization': lam}
    fitted.coefficients = coefficients.tolist()
    return fitted, info


def _fit_regularized_rows(rows, *, regularization):
    """Explicit-parameter equivalent of _fit_rows, with already-set offsets."""
    return _optimize(_design(rows), regularization)


def _with_base(rows, base):
    result = deepcopy(rows)
    for row in result:
        row['state']['prior'] = list(base)
    return result


def _score(model, rows, base):
    losses = []
    for row in rows:
        vector, _ = model._vector(row['state'])
        logits = np.log(base)+vector@np.asarray(model.coefficients)
        losses.append(float(logsumexp(logits)-logits[row['outcome']]))
    return float(np.mean(losses))


def fit_A_predictor(rows, *, pseudocount=.5):
    """Return (kernel, selection report, A-only base); kernel may be None.

    LOO is an A-only tuning procedure, not a held-out performance estimate.
    An insufficient class fold causes an explicit incomplete result; no weaker
    split, externally supplied prior or B/C fallback is silently substituted.
    """
    if isinstance(pseudocount, bool) or not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError('Positive finite class pseudocount required')
    prepared = _prepare(rows)
    mids = sorted({row['physical_match_id'] for row in prepared})
    facts = {row['physical_match_id']: row['outcome'] for row in prepared}
    base = _base(prepared, pseudocount)
    candidates = [{'name': 'base_only', 'regularization': None}]+[
        {'name': 'ridge_'+format(lam, '.10g'), 'regularization': lam} for lam in REGULARIZATIONS]
    report = {'schema': VERSION, 'status': 'selecting', 'physical_matches': len(mids),
        'input_rows': len(rows), 'training_states': len(prepared),
        'empty_text_states': sum(not row['state']['texts'] for row in prepared),
        'training_physical_match_ids': mids, 'states_added_or_removed': False,
        'startup_scope': 'supplied_states_only_empty_lexical_input_returns_fold_base',
        'selection_scope': 'A_only_whole_match_leave_one_out_not_independent_test',
        'scoring': 'equal_physical_match_then_equal_supplied_state_log_loss',
        'vocabulary_scope': 'fit_each_fold_on_training_physical_match_text_unions_only',
        'empirical_base_scope': 'one_outcome_per_training_physical_match_fold_local',
        'pseudocount_per_class': pseudocount, 'regularization_grid': list(REGULARIZATIONS),
        'legacy_regularization': PARAMETERS['regularization'],
        'tie_rule': 'prefer_base_then_larger_regularization_within_absolute_1e-12',
        'calibration_status': 'not_established', 'folds': [], 'candidates': candidates,
        'input_projection_sha256': _digest(prepared), 'empirical_base': base}
    insufficient = []
    for heldout in mids:
        train_ids = [mid for mid in mids if mid != heldout]
        if len({facts[mid] for mid in train_ids}) < 2:
            insufficient.append({'heldout_match_id': heldout, 'training_match_ids': train_ids,
                                 'reason': 'fewer_than_two_training_outcome_classes'})
    if insufficient or len(mids) < 3:
        report.update(status='insufficient_group_folds', insufficient_folds=insufficient,
                      selected_candidate=None, selected_regularization=None, base_only_won=None)
        return None, report, base
    scores = {candidate['name']: [] for candidate in candidates}
    failures = {candidate['name']: [] for candidate in candidates}
    for heldout in mids:
        train = [row for row in prepared if row['physical_match_id'] != heldout]
        test = [row for row in prepared if row['physical_match_id'] == heldout]
        fold_base = _base(train, pseudocount)
        training = _with_base(train, fold_base)
        design = _design(training)
        encoder = design[0].encoder
        fold = {'heldout_match_id': heldout, 'training_match_ids': sorted({r['physical_match_id'] for r in train}),
                'train_states': len(train), 'heldout_states': len(test), 'empirical_base': fold_base,
                'training_projection_sha256': _digest(training),
                'encoder_sha256': _digest(encoder), 'vocabulary_size': len(encoder['vocabulary']),
                'candidates': []}
        for candidate in candidates:
            name = candidate['name']
            try:
                model, optimizer = _optimize(design, candidate['regularization'])
            except ValueError as error:
                failure = {'heldout_match_id': heldout, 'reason': str(error)}
                failures[name].append(failure)
                fold['candidates'].append({'name': name, 'status': 'optimizer_failed', **failure})
                continue
            score = _score(model, test, fold_base)
            scores[name].append(score)
            fold['candidates'].append({'name': name, 'log_loss': score, 'optimizer': optimizer,
                                       'coefficients_sha256': _digest(model.coefficients)})
        report['folds'].append(fold)
    selected, best = None, math.inf
    for candidate in candidates:
        name = candidate['name']
        eligible = len(scores[name]) == len(mids) and not failures[name]
        candidate.update(status='eligible' if eligible else 'incomplete_folds',
                         completed_folds=len(scores[name]), failures=failures[name],
                         mean_match_log_loss=float(np.mean(scores[name])) if eligible else None)
        if eligible and candidate['mean_match_log_loss'] < best-TIE_TOLERANCE:
            best, selected = candidate['mean_match_log_loss'], candidate
    if selected is None:
        report.update(status='no_complete_candidate', selected_candidate=None, selected_regularization=None, base_only_won=None)
        return None, report, base
    model, optimizer = _fit_regularized_rows(_with_base(prepared, base), regularization=selected['regularization'])
    report.update(status='fitted', selected_candidate=selected['name'], selected_regularization=selected['regularization'],
                  base_only_won=selected['regularization'] is None,
                  selected_mean_match_log_loss=best, final_fit=optimizer,
                  final_encoder_sha256=_digest(model.encoder), final_coefficients_sha256=_digest(model.coefficients))
    model.metadata = {'predictor_selection': deepcopy(report), 'empirical_base': list(base),
                      'training_physical_match_ids': mids, 'kernel_only': True}
    return model, report, base
