"""Bounded same-state article ranking diagnostic, never an acquisition policy.

The main frozen predictor and absolute stopping model remain unchanged. A
relative score may be positive when every article has negative absolute value.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import argparse
import json
from pathlib import Path

import numpy as np

from .historical_context_training import (
    G_SELECTION_GRID, HistoricalContextModel, _action, _digest, _discovery_panel,
    _gain_design, _state_for, _weighted_gain_ridge, prepare_historical_cases,
)

VERSION = 'same-state-relative-choice-diagnostic-v1'
TOLERANCE = 1e-12


def _menu_key(row):
    return (row['physical_match_id'], tuple(row['discovered_indices']), tuple(row['acquired_indices']))


def comparable_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[_menu_key(row)].append(row)
    result = []
    for key in sorted(groups):
        menu = sorted(groups[key], key=lambda row: row['action_index'])
        actions = [row['action_index'] for row in menu]
        if len(set(actions)) != len(actions):
            raise ValueError('Same-state candidate actions must be unique')
        if len(menu) >= 2:
            result.extend(menu)
    return result


def _groups(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[_menu_key(row)].append(index)
    return groups


def centered_design(encoder, rows):
    """Only observed state/candidate features enter x; realized gains enter y."""
    x = np.asarray([encoder._vector(row['context'])[0] for row in rows])
    y = np.asarray([row['gain'] for row in rows])
    menus = _groups(rows)
    menu_counts = defaultdict(int)
    for key in menus:
        menu_counts[key[0]] += 1
    weights = np.zeros(len(rows))
    for key, indices in menus.items():
        if len(indices) < 2:
            raise ValueError('Relative choice requires at least two candidates in exactly the same state')
        x[indices] -= x[indices].mean(axis=0)
        y[indices] -= y[indices].mean()
        weights[indices] = 1./(len(menu_counts)*menu_counts[key[0]]*len(indices))
    # Shared ridge helper expects an unpenalized intercept. Equal menu weighting
    # makes its true optimum zero; it is removed from the persisted rank model.
    x[:, 0] = 1.
    return x, y, weights


def _fit(encoder, rows, strength):
    x, y, weights = centered_design(encoder, rows)
    coefficients, diagnostics = _weighted_gain_ridge(x, y, weights, strength)
    coefficients[0] = 0.
    return coefficients, diagnostics


def evaluate_relative(rows, predictions):
    """Uniform tie handling avoids turning file order into a ranking baseline."""
    predictions = np.asarray(predictions)
    if len(predictions) != len(rows) or not np.all(np.isfinite(predictions)):
        raise ValueError('One finite relative score is required per candidate')
    fixtures = defaultdict(list)
    detail = []
    for key, indices in _groups(rows).items():
        actual = np.asarray([rows[index]['gain'] for index in indices])
        predicted = predictions[indices]
        predicted = predicted-predicted.mean()
        centered = actual-actual.mean()
        chosen = np.flatnonzero(predicted >= predicted.max()-TOLERANCE)
        best = set(np.flatnonzero(actual >= actual.max()-TOLERANCE).tolist())
        pair_accuracies, pair_errors = [], []
        for a, b in combinations(range(len(indices)), 2):
            truth, guess = actual[a]-actual[b], predicted[a]-predicted[b]
            pair_errors.append(float((guess-truth)**2))
            if abs(truth) > TOLERANCE:
                pair_accuracies.append(.5 if abs(guess) <= TOLERANCE else float(truth*guess > 0))
        metrics = {
            'centered_mse': float(np.mean((predicted-centered)**2)),
            'pair_difference_mse': float(np.mean(pair_errors)),
            'pairwise_accuracy': float(np.mean(pair_accuracies)) if pair_accuracies else None,
            'top_choice_accuracy_with_uniform_ties': sum(i in best for i in chosen)/len(chosen),
            'oracle_relative_regret': float(actual.max()-actual[chosen].mean()),
        }
        fixtures[key[0]].append(metrics)
        detail.append({'physical_match_id': key[0], 'discovered_indices': list(key[1]),
            'acquired_indices': list(key[2]), 'actions': [rows[i]['action_index'] for i in indices],
            'urls': [rows[i]['url'] for i in indices], 'actual_absolute_gains': actual.tolist(),
            'actual_centered_gains': centered.tolist(), 'relative_scores': predicted.tolist(),
            'uniform_tie_action_indices': [rows[indices[i]]['action_index'] for i in chosen],
            'metrics': metrics})
    metrics = list(next(iter(fixtures.values()))[0]) if fixtures else []
    by_fixture = {}
    for mid, menus in fixtures.items():
        by_fixture[mid] = {metric: float(np.mean([menu[metric] for menu in menus if menu[metric] is not None]))
            if any(menu[metric] is not None for menu in menus) else None for metric in metrics}
    means = {metric: float(np.mean([scores[metric] for scores in by_fixture.values() if scores[metric] is not None]))
        if any(scores[metric] is not None for scores in by_fixture.values()) else None for metric in metrics}
    return {'weighting': 'equal_physical_fixture_then_equal_same_state_menu',
            'metrics': means, 'per_fixture': by_fixture, 'menus': detail}


def compare_state_rankings(evaluation):
    """Separate numerical sensitivity from changed ordering of common actions."""
    comparisons = []
    for before, after in combinations(evaluation['menus'], 2):
        if (before['physical_match_id'] != after['physical_match_id']
                or before['discovered_indices'] != after['discovered_indices']):
            continue
        first = dict(zip(before['actions'], before['relative_scores']))
        second = dict(zip(after['actions'], after['relative_scores']))
        first_truth = dict(zip(before['actions'], before['actual_absolute_gains']))
        second_truth = dict(zip(after['actions'], after['actual_absolute_gains']))
        for a, b in combinations(sorted(set(first)&set(second)), 2):
            margin_a, margin_b = first[a]-first[b], second[a]-second[b]
            truth_a, truth_b = first_truth[a]-first_truth[b], second_truth[a]-second_truth[b]
            comparisons.append({'physical_match_id': before['physical_match_id'], 'actions': [a, b],
                'discovered_indices': before['discovered_indices'],
                'first_acquired_indices': before['acquired_indices'], 'second_acquired_indices': after['acquired_indices'],
                'first_predicted_margin': margin_a, 'second_predicted_margin': margin_b,
                'prediction_reversal': bool(margin_a*margin_b < 0 and min(abs(margin_a), abs(margin_b)) > TOLERANCE),
                'actual_reversal': bool(truth_a*truth_b < 0 and min(abs(truth_a), abs(truth_b)) > TOLERANCE)})
    return {'comparable_state_pairs': len(comparisons),
        'prediction_reversals': sum(item['prediction_reversal'] for item in comparisons),
        'actual_reversals': sum(item['actual_reversal'] for item in comparisons),
        'actual_reversals_captured': sum(item['prediction_reversal'] and item['actual_reversal'] for item in comparisons),
        'actual_gain_semantics': 'realized_fixed_predictor_gain_using_the_heldout_outcome_not_expected_value',
        'observed_reversal_requires_policy_reversal': False,
        'scope': 'diagnostic_shared_candidate_ordering_not_independent_matches_or_proof_of_utility',
        'comparisons': comparisons}


def fit_relative_choice(rows):
    rows = comparable_rows(rows)
    ids = sorted({row['physical_match_id'] for row in rows})
    if len(ids) < 2:
        return None, {'status': 'insufficient_independent_comparable_B_matches', 'physical_matches': len(ids)}
    errors = [[] for _ in G_SELECTION_GRID]
    folds = []
    for held_out in ids:
        train = [row for row in rows if row['physical_match_id'] != held_out]
        valid = [row for row in rows if row['physical_match_id'] == held_out]
        encoder, _ = _gain_design(train)
        xv, _, _ = centered_design(encoder, valid)
        fold = {'held_out_physical_match_id': held_out,
            'training_physical_match_ids': [mid for mid in ids if mid != held_out],
            'training_encoder_sha256': _digest(encoder.encoder), 'hypotheses': []}
        for index, strength in enumerate(G_SELECTION_GRID):
            coefficients, _ = _fit(encoder, train, strength)
            scored = evaluate_relative(valid, xv@coefficients)
            errors[index].append(scored['metrics']['centered_mse'])
            fold['hypotheses'].append({'regularization': strength, 'metrics': scored['metrics']})
        folds.append(fold)
    means = [float(np.mean(values)) for values in errors]
    winner = 0
    for index, score in enumerate(means[1:], 1):
        if score < means[winner]-(1e-15+1e-12*abs(means[winner])):
            winner = index
    encoder, _ = _gain_design(rows)
    coefficients, fit = _fit(encoder, rows, G_SELECTION_GRID[winner])
    article_pairs = {(key[0], *sorted((rows[i]['url'], rows[j]['url'])))
                     for key, indices in _groups(rows).items() for i, j in combinations(indices, 2)}
    report = {'status': 'fitted_diagnostic_only', 'candidate_regularizations': list(G_SELECTION_GRID),
        'criterion': 'B_leave_one_physical_match_out_equal_menu_centered_signed_gain_MSE',
        'fold_encoder_scope': 'training_matches_only_no_heldout_vocabulary_idf_or_categories',
        'fold_direction': 'group_leave_one_out_not_prospective_rolling_origin',
        'selected_index': winner, 'selected_regularization': G_SELECTION_GRID[winner],
        'zero_baseline_won': winner == 0, 'mean_validation_mse': means,
        'folds': folds, 'final_fit': fit, 'B_physical_matches': len(ids),
        'B_same_state_menus': len(_groups(rows)), 'B_candidate_rows': len(rows),
        'B_distinct_fixture_article_pairs': len(article_pairs),
        'B_pair_comparisons': sum(len(indices)*(len(indices)-1)//2 for indices in _groups(rows).values()),
        'B_selection_rows_sha256': _digest(rows),
        'tie_rule': 'zero_then_greater_regularization_with_1e-15_plus_1e-12_relative_tolerance',
        'no_absolute_gain_interpretation': True, 'stopping_head_replaced': False}
    return {'encoder': encoder, 'coefficients': coefficients}, report


def _check_rows(model, cases):
    """Build C under the same frozen f, same menu visibility and same projection."""
    result = []
    for case in cases:
        mid = case['fixture']['physical_match_id']
        if mid not in model.metadata['C_match_ids']:
            continue
        for _, _, discovered, acquired, before in _discovery_panel(case):
            actions = [i for i in discovered if i not in acquired]
            if len(actions) < 2:
                continue
            projected, p, _, _ = model._probabilities(before)
            model._heldout(projected, value=True)
            for index in actions:
                after = _state_for(case, acquired_indices=acquired+[index], discovered_indices=discovered)
                q = model._probabilities(after)[1]
                doc = case['documents'][index]
                result.append({'physical_match_id': mid, 'gain': float(np.log(q[case['outcome']])-np.log(p[case['outcome']])),
                    'discovered_indices': discovered, 'acquired_indices': acquired, 'action_index': index,
                    'url': doc['url'], 'context': model._context(before, _action(doc, index))})
    return result


def run_diagnostic(model_path, cases_path, output_dir):
    model_path, cases_path, output = map(Path, (model_path, cases_path, output_dir))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty diagnostic destination; existing evidence is preserved')
    model = HistoricalContextModel.load(model_path)
    b_path = model_path.parent/'B-targets.json'
    b = json.loads(b_path.read_text(encoding='utf-8'))
    if _digest(b) != model.metadata['B_targets_sha256']:
        raise ValueError('B gain targets must bind the unchanged frozen predictor')
    cases, _ = prepare_historical_cases(json.loads(cases_path.read_text(encoding='utf-8')))
    if _digest(cases) != model.metadata['corpus_sha256']:
        raise ValueError('C cases must be the exact original frozen corpus')
    ranker, selection = fit_relative_choice(b)
    report = {'schema': VERSION, 'selection': selection,
        'bindings': {'frozen_model_version': model.model_version,
            'frozen_model_sha256': sha256(model_path.read_bytes()).hexdigest(),
            'B_targets_sha256': sha256(b_path.read_bytes()).hexdigest(),
            'cases_sha256': sha256(cases_path.read_bytes()).hexdigest()},
        'original_absolute_stopping_model_unchanged': True, 'network_calls': 0,
        'limits': ['only_comparable_physical_matches_are_independent_not_enumerated_choice_rows',
            'C_previously_inspected_diagnostic_not_fresh_confirmation',
            'reconstructed_menu_reveals_not_historical_agent_decisions',
            'relative_score_never_authorizes_acquisition_or_stopping',
            'role_text_similarity_is_not_semantic_reliability_or_calibrated_support']}
    output.mkdir(parents=True, exist_ok=True)
    if ranker is not None:
        c = _check_rows(model, cases)
        xc, _, _ = centered_design(ranker['encoder'], c)
        report['C_selected'] = evaluate_relative(c, xc@ranker['coefficients'])
        report['C_zero_baseline'] = evaluate_relative(c, np.zeros(len(c)))
        state_comparison = compare_state_rankings(report['C_selected'])
        with (output/'state-rank-comparison.json').open('x', encoding='utf-8') as stream:
            json.dump(state_comparison, stream, ensure_ascii=False, indent=2, allow_nan=False)
        report['C_physical_matches'] = len({row['physical_match_id'] for row in c})
        report['C_same_state_menus'] = len(_groups(c))
        artifact = {'schema': VERSION, 'scope': 'menu_centered_relative_choice_only_not_absolute_value',
            'encoder': ranker['encoder'].encoder, 'coefficients': ranker['coefficients'].tolist(),
            'selected_regularization': selection['selected_regularization'], 'bindings': report['bindings'],
            'selection': selection, 'stopping_head_replaced': False}
        with (output/'relative-model.json').open('x', encoding='utf-8') as stream:
            json.dump(artifact, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (output/'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run_diagnostic(args.model, args.cases, args.output)
    print(json.dumps({'status': report['selection']['status'],
        'B_physical_matches': report['selection'].get('B_physical_matches'),
        'zero_baseline_won': report['selection'].get('zero_baseline_won'),
        'selected_regularization': report['selection'].get('selected_regularization'),
        'C_selected': report.get('C_selected', {}).get('metrics'),
        'C_zero_baseline': report.get('C_zero_baseline', {}).get('metrics')}, indent=2))


if __name__ == '__main__':
    main()
