"""Constructed selection invariants, never reported as football evidence."""
from copy import deepcopy

import numpy as np
import pytest

from football_research.acquisition.context_predictor import NUMERIC, PARAMETERS
from football_research.acquisition.fixture_roles import ROLES
from football_research.acquisition.historical_predictor_selection import (
    REGULARIZATIONS, _design, _fit_regularized_rows, fit_A_predictor,
)
from football_research.acquisition.role_context_predictor import RoleContextPredictor


def examples(*, text=True):
    rows = []
    for i in range(6):
        for empty in (True, False):
            words = [] if empty or not text else ['Home Club '+['attack strong', 'balanced draw', 'defence weak'][i % 3]]
            rows.append({'physical_match_id': 'constructed:'+str(i), 'outcome': i % 3,
                'state': {'prior': None, 'texts': words, 'sources': ['example.test'] if words else [],
                          'numeric': [float(bool(words))]+[0.]*(len(NUMERIC)-1),
                          'role_texts': {role: words if role == ROLES[0] else [] for role in ROLES}}})
    return rows


def test_explicit_regularization_reproduces_original_kernel_numerically():
    rows = examples()
    for row in rows:
        row['state']['prior'] = [.4, .3, .3]
    old = RoleContextPredictor()
    _, result = old._fit_rows(rows)
    selected, optimizer = _fit_regularized_rows(rows, regularization=PARAMETERS['regularization'])
    assert selected.encoder == old.encoder
    np.testing.assert_allclose(selected.coefficients, old.coefficients, rtol=1e-12, atol=1e-14)
    assert optimizer['objective'] == pytest.approx(result.fun, abs=1e-14)
    assert optimizer['iterations'] == result.nit


def test_heldout_outcome_never_enters_its_training_base_encoder_or_coefficients():
    rows = examples()
    original = deepcopy(rows)
    _, before, _ = fit_A_predictor(rows)
    changed = deepcopy(rows)
    for row in changed:
        if row['physical_match_id'] == 'constructed:0':
            row['outcome'] = 1
    _, after, _ = fit_A_predictor(changed)
    left = next(f for f in before['folds'] if f['heldout_match_id'] == 'constructed:0')
    right = next(f for f in after['folds'] if f['heldout_match_id'] == 'constructed:0')
    for key in ('training_match_ids', 'empirical_base', 'encoder_sha256', 'training_projection_sha256'):
        assert left[key] == right[key]
    for a, b in zip(left['candidates'], right['candidates']):
        assert a['coefficients_sha256'] == b['coefficients_sha256']
        assert a['optimizer'] == b['optimizer']
    assert left['empirical_base'] == pytest.approx([1.5/6.5, 2.5/6.5, 2.5/6.5])
    assert rows == original
    assert before['regularization_grid'] == list(REGULARIZATIONS)


def test_encoder_uses_only_training_game_unions_and_game_equal_objective_weights():
    rows = examples()
    train = [row for row in rows if row['physical_match_id'] != 'constructed:0']
    for row in train:
        row['state']['prior'] = [1/3]*3
    # One physical game's multiple prefixes cannot acquire more total weight.
    train.extend(deepcopy(train[:2]))
    model, _, _, _, weights = _design(train)
    for mid in {row['physical_match_id'] for row in train}:
        total = sum(w for row, w in zip(train, weights) if row['physical_match_id'] == mid)
        assert total == pytest.approx(.2)
    outside = deepcopy(rows[1])
    outside['state']['texts'] = ['zzzzzz unique heldout token']
    assert 'zzzzz' not in model.encoder['vocabulary']
    assert model.encoder['sources'] == ['example.test']


def test_empty_input_ties_retain_base_only_and_preserve_all_supplied_states():
    rows = examples(text=False)
    rows.append(deepcopy(rows[0]))  # The selector must not silently alter a supplied panel.
    model, report, base = fit_A_predictor(rows)
    assert report['status'] == 'fitted'
    assert report['selected_candidate'] == 'base_only'
    assert report['base_only_won'] is True
    assert report['selected_regularization'] is None
    assert report['training_states'] == len(rows)
    assert report['states_added_or_removed'] is False
    assert base == pytest.approx([1/3]*3)
    assert np.all(np.asarray(model.coefficients) == 0)
    assert model.encoder['vocabulary'] == []
    assert report['final_fit']['method'] == 'base_only_no_optimizer'


def test_insufficient_training_classes_are_explicit_incomplete_result():
    rows = examples()
    for row in rows:
        row['outcome'] = int(row['physical_match_id'] == 'constructed:0')
    model, report, base = fit_A_predictor(rows)
    assert model is None
    assert report['status'] == 'insufficient_group_folds'
    assert report['folds'] == []
    assert report['insufficient_folds'][0]['heldout_match_id'] == 'constructed:0'
    assert base == pytest.approx([5.5/7.5, 1.5/7.5, .5/7.5])


def test_rejects_external_offset_and_conflicting_game_labels():
    rows = examples()
    rows[0]['state']['prior'] = [.5, .3, .2]
    with pytest.raises(ValueError, match='missing prior'):
        fit_A_predictor(rows)
    rows = examples()
    rows[0]['outcome'] = 2
    with pytest.raises(ValueError, match='conflicting'):
        fit_A_predictor(rows)
