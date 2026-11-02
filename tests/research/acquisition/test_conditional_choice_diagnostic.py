"""Constructed contracts only; these rows are never a real-match result."""
from copy import deepcopy

import numpy as np
import pytest

from football_research.acquisition.conditional_choice_diagnostic import (
    centered_design, comparable_rows, compare_state_rankings, evaluate_relative, fit_relative_choice,
)
from football_research.acquisition.historical_context_training import _gain_design
from football_research.acquisition.fixture_roles import ROLES


def rows():
    result = []
    for match in range(4):
        for action, label in enumerate(('actual preview', 'fitness update')):
            roles = {role: [] for role in ROLES}
            roles['home'] = ['Home Club known text']
            candidates = {role: [] for role in ROLES}
            candidates['home'] = ['Home Club '+label]
            result.append({'physical_match_id': f'constructed:{match}',
                'discovered_indices': [0, 1], 'acquired_indices': [], 'action_index': action,
                'gain': (1 if action else -1)*.01 + match*.3,
                'url': f'https://example.test/news/{match}/{action}',
                'context': {'texts': roles['home'], 'labels': candidates['home'],
                    'role_texts': roles, 'candidate_role_texts': candidates,
                    'numeric': [0.]*28, 'channel': 'read_url|example.test',
                    'sources': ['example.test'], 'mask': [True]*13, 'date': f'2024-01-{match+1:02d}'}})
    return result


def test_menu_centering_removes_match_and_state_intercepts_without_positive_utility_claim():
    data = rows()
    encoder, _ = _gain_design(data)
    x, y, weights = centered_design(encoder, data)
    changed = deepcopy(data)
    for index, row in enumerate(changed):
        row['gain'] += 10*(index//2+1)
        row['unread_body'] = 'This unavailable text must never enter the model vector.'
    xx, yy, ww = centered_design(encoder, changed)
    assert np.array_equal(x, xx)
    assert np.allclose(y, yy, atol=1e-13)
    assert np.array_equal(weights, ww)
    assert weights.sum() == pytest.approx(1.)
    for start in range(0, len(data), 2):
        assert np.allclose(x[start:start+2, 1:].mean(axis=0), 0.)
        assert y[start:start+2].mean() == pytest.approx(0., abs=1e-13)
        assert weights[start:start+2].sum() == pytest.approx(.25)


def test_comparisons_require_exactly_same_menu_and_read_state():
    data = rows()[:2]
    assert len(comparable_rows(data)) == 2
    changed = deepcopy(data)
    changed[1]['acquired_indices'] = [2]
    assert comparable_rows(changed) == []
    duplicate = data+[deepcopy(data[0])]
    with pytest.raises(ValueError, match='unique'):
        comparable_rows(duplicate)


def test_zero_baseline_ties_are_uniform_not_file_order():
    data = rows()[:2]
    data[0]['gain'], data[1]['gain'] = -.2, .2
    scored = evaluate_relative(data, [0., 0.])
    assert scored['metrics']['oracle_relative_regret'] == pytest.approx(.2)
    assert scored['metrics']['top_choice_accuracy_with_uniform_ties'] == .5
    assert scored['metrics']['pairwise_accuracy'] == .5
    assert evaluate_relative(list(reversed(data)), [0., 0.])['metrics'] == scored['metrics']
    correct = evaluate_relative(data, [-.1, .1])
    assert correct['metrics']['oracle_relative_regret'] == 0.
    assert correct['metrics']['pairwise_accuracy'] == 1.


def test_zero_winner_is_preserved_and_whole_matches_leave_every_fold():
    data = rows()
    for index, row in enumerate(data):
        row['gain'] = float(index//2)
    model, report = fit_relative_choice(data)
    assert report['zero_baseline_won'] is True
    assert report['selected_regularization'] is None
    assert np.all(model['coefficients'] == 0.)
    assert report['B_physical_matches'] == 4
    assert report['B_same_state_menus'] == 4
    assert report['B_distinct_fixture_article_pairs'] == 4
    assert report['stopping_head_replaced'] is False
    for fold in report['folds']:
        assert fold['held_out_physical_match_id'] not in fold['training_physical_match_ids']
        assert len(fold['training_physical_match_ids']) == 3


def test_one_independent_game_cannot_claim_group_validated_choice():
    model, report = fit_relative_choice(rows()[:2])
    assert model is None
    assert report['status'] == 'insufficient_independent_comparable_B_matches'


def test_numerical_sensitivity_does_not_count_as_correct_state_dependent_ranking():
    data = rows()[:2]
    for row in data:
        row['discovered_indices'] = [0, 1, 2]
    changed = deepcopy(data)
    for row in changed:
        row['acquired_indices'] = [2]
        row['gain'] = -row['gain']
    evaluated = evaluate_relative(data+changed, [-.1, .1, -.05, .05])
    checked = compare_state_rankings(evaluated)
    assert checked['comparable_state_pairs'] == 1
    assert checked['actual_reversals'] == 1
    assert checked['prediction_reversals'] == 0
    assert checked['actual_reversals_captured'] == 0
