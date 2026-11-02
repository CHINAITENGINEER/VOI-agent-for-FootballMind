"""Constructed archived panels test the adapter; they are not football results."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json

import pytest

from football_research.acquisition.historical_context_training import (
    HistoricalContextModel, VERSION, _action, _digest, _discovery_panel, _state_for,
    fit_historical_context, prepare_historical_cases,
)


def examples(root):
    rows, cases = [], []
    for i in range(12):
        kickoff = datetime(2024, 1, 5, 15, tzinfo=timezone.utc)+timedelta(days=i*7)
        label = i % 3
        fixture = {'physical_match_id': 'constructed:'+str(i), 'match_id': str(i),
                   'home_team_id': 'home', 'away_team_id': 'away', 'home_name': 'Home Club',
                   'away_name': 'Away Club', 'competition_id': 'constructed', 'kickoff': kickoff.isoformat()}
        docs = []
        for j in range(2):
            text = ('Home Club ' + ('strong attack' if label == 0 else 'balanced game' if label == 1 else 'weak attack')
                    + f' and Away Club. Archived constructed article {i} {j}.')
            path = root/f'article-{i}-{j}.txt'
            path.write_text(text, encoding='utf-8')
            docs.append({'url': f'https://example.test/article/{i}/{j}',
                'title': 'Home Club and Away Club '+('preview' if j == 0 else 'fitness update'), 'text': text,
                'source_published_at': (kickoff-timedelta(days=2-j)).isoformat(),
                'captured_at': '2026-01-01T00:00:00+00:00', 'raw_path': str(path),
                'raw_sha256': sha256(path.read_bytes()).hexdigest()})
        result_at = (kickoff+timedelta(hours=3)).isoformat()
        rows.append({'source': 'constructed', 'source_id': str(i), 'score_period': 'regulation',
                     'home_goals': [2, 1, 0][label], 'away_goals': 1,
                     'home_team_id': 'home', 'away_team_id': 'away', 'competition_id': 'constructed',
                     'kickoff': kickoff.isoformat(), 'result_available_at': result_at})
        cases.append({'fixture': fixture, 'prior_probabilities': None, 'outcome': label,
                      'result_available_at': result_at, 'documents': docs})
    path = root/'labels.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    for i, case in enumerate(cases):
        case['label_provenance'] = {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest(),
                                    'source': 'constructed', 'source_id': str(i)}
    return cases


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp('historical-role-adapter')
    cases = examples(root)
    model, report = fit_historical_context(cases, root/'model', dataset_version='constructed-contract-only',
                                            a_fraction=.5, b_fraction=.25)
    assert model is not None, report
    return cases, model, report, root


def test_small_sample_fits_the_same_kernels_and_b_targets_recompute(fitted):
    cases, model, report, root = fitted
    assert report['status'] == 'fitted_research_prototype'
    assert len(report['splits']['A']) == 6 and len(report['splits']['B']) == 3
    assert report['A_prefixes'] == 6*9
    assert report['B_transitions'] == 3*6
    targets = json.loads((root/'model'/'B-targets.json').read_text())
    import math
    assert any(abs(row['gain']) > 1e-10 for row in targets)
    for row in targets:
        y = row['outcome']
        assert row['gain'] == pytest.approx(math.log(row['prediction_after'][y])-math.log(row['prediction_before'][y]))
    assert len({(row['physical_match_id'], tuple(row['discovered_indices']),
                 tuple(row['acquired_indices']), row['action_index']) for row in targets}) == len(targets)
    assert model.metadata['capture_times_rewritten'] is False
    assert model.metadata['kernel_reuse'][0].startswith('RoleContextPredictor')


def test_unread_body_and_outcome_cannot_change_prequery_estimate(fitted):
    cases, model, _, _ = fitted
    case = deepcopy(cases[-1])
    before = _state_for(case, 0)
    action = _action(case['documents'][0], 0)
    forecast, value = model.predict(before), model.estimate(before, action)
    assert value['support_status'] == 'prototype_supported'
    changed = deepcopy(case)
    changed['outcome'] = (case['outcome']+1) % 3
    changed['documents'][0]['text'] = 'Completely different unread article including future result.'
    assert model.predict(_state_for(changed, 0)) == forecast
    assert model.estimate(_state_for(changed, 0), {**action, 'hidden_body': changed['documents'][0]['text']}) == value


def test_undiscovered_titles_and_bodies_cannot_enter_current_state(fitted):
    cases, model, _, _ = fitted
    case = deepcopy(cases[-1])
    before = _state_for(case, discovered_indices=[0])
    action = _action(case['documents'][0], 0)
    forecast, value = model.predict(before), model.estimate(before, action)
    changed = deepcopy(case)
    changed['documents'][1]['title'] = 'Future result not yet discovered'
    changed['documents'][1]['text'] = 'Hidden body must not alter f or g'
    hidden = _state_for(changed, discovered_indices=[0])
    assert hidden == before
    assert model.predict(hidden) == forecast
    assert model.estimate(hidden, action) == value
    with pytest.raises(ValueError, match='previously discovered'):
        _state_for(case, acquired_indices=[1], discovered_indices=[0])
    with pytest.raises(ValueError, match='actually visible'):
        model.estimate(before, _action(case['documents'][1], 1))


def test_actual_background_titles_are_visible_without_leaking_target_titles(fitted, tmp_path):
    cases, _, _, _ = fitted
    case = deepcopy(cases[-1])
    path = tmp_path/'constructed-listing.html'
    path.write_text('Actual constructed parent menu title', encoding='utf-8')
    link = {'url': 'https://example.test/background', 'title': 'Actual constructed parent menu title',
            'parent_url': 'https://example.test/news', 'raw_path': str(path),
            'raw_sha256': sha256(path.read_bytes()).hexdigest(),
            'source_published_at': case['documents'][0]['source_published_at'],
            'available_at': case['documents'][0]['source_published_at'],
            'captured_at': '2026-01-01T00:00:00+00:00'}
    case['background_links'] = [link, {**link, 'url': case['documents'][0]['url'],
                                     'title': case['documents'][0]['title']}]
    prepared, audit = prepare_historical_cases([case])
    before = _state_for(prepared[0], discovered_indices=[])
    assert [row['title'] for row in before['observed_links']] == [link['title']]
    assert before['documents'] == []
    assert audit['target_titles_removed_from_background'] == 1
    changed = deepcopy(case)
    changed['background_links'][0]['raw_sha256'] = '0'*64
    with pytest.raises(ValueError, match='background source bytes'):
        prepare_historical_cases([changed])
    changed = deepcopy(case)
    changed['background_links'][0]['available_at'] = case['fixture']['kickoff']
    prepared, audit = prepare_historical_cases([changed])
    assert prepared[0]['background_links'] == []
    assert any(row['reason'] == 'background_title_not_prematch' for row in audit['excluded'])


def test_three_real_panel_entries_have_27_unique_states_and_actions(fitted):
    cases, _, _, _ = fitted
    case = deepcopy(cases[-1])
    # State enumeration only: constructed test fixture, never a real corpus row.
    third = deepcopy(case['documents'][0])
    third['url'] = 'https://example.test/third'
    third['title'] = 'An independently discovered third constructed title'
    case['documents'].append(third)
    panels = list(_discovery_panel(case))
    assert len(panels) == 27
    triples = [(tuple(discovered), tuple(acquired), i)
               for _, _, discovered, acquired, _ in panels for i in discovered if i not in acquired]
    assert len(triples) == len(set(triples)) == 27


def _unseen_state_pattern(case):
    state = _state_for(case, 0)
    state['observed_links'].append({'url': 'https://example.test/observed-home-title',
        'title': 'Home Club update', 'parent_url': 'https://example.test/news',
        'title_basis': 'historical_headline_panel', 'available_at': state['decision_at']})
    return state


def test_state_pattern_generalization_is_explicit_uncalibrated_and_body_blind(fitted):
    cases, model, _, _ = fitted
    case = cases[-1]
    state = _unseen_state_pattern(case)
    action = _action(case['documents'][0], 0)
    strict = model.estimate(state, action)
    assert strict['support_status'] == 'UNKNOWN'
    assert strict == model.estimate(state, action, state_generalization=False)
    assert 'state_pattern_extrapolation' not in strict['support']
    value = model.estimate(state, action, state_generalization=True)
    assert value['support_status'] == 'prototype_state_generalization'
    assert value['reason'] == 'uncalibrated_state_pattern_generalization'
    assert value['support']['exact_state_matches'] == 0
    assert value['support']['state_pattern_extrapolation'] is True
    assert value['support']['physical_matches'] == 3
    assert isinstance(value['expected_gain'], float)
    assert value['expected_gain'] == value['expected_signed_gain']
    assert value['cost_deducted'] is False
    assert value['calibration_status'] == 'not_established' and value['stop_certificate'] is None
    changed = deepcopy(case)
    changed['outcome'] = (case['outcome']+1) % 3
    changed['documents'][0]['text'] = 'A different unread body including future results'
    assert model.estimate(_unseen_state_pattern(changed), action, state_generalization=True) == value


def test_state_generalization_keeps_source_candidate_role_and_independence_gates(fitted):
    cases, model, _, _ = fitted
    state = _unseen_state_pattern(cases[-1])
    action = _action(cases[-1]['documents'][0], 0)
    unseen_source = deepcopy(state)
    unseen_source['observed_links'][0]['url'] = 'https://unseen.test/article'
    estimate = model.estimate(unseen_source, {**action, 'url': 'https://unseen.test/article'}, state_generalization=True)
    assert estimate['support_status'] == 'UNKNOWN' and estimate['expected_gain'] is None
    unseen_role = deepcopy(state)
    unseen_role['observed_links'][0]['title'] = 'Home Club'
    estimate = model.estimate(unseen_role, {**action, 'title': 'Home Club'}, state_generalization=True)
    assert estimate['support_status'] == 'UNKNOWN' and estimate['expected_gain'] is None
    # A separately constructed test model has many neighbor rows but only one
    # physical game; actual artifacts and learned coefficients stay untouched.
    single_game = deepcopy(model)
    for row in single_game.g.support_records:
        row['physical_match_id'] = 'one-constructed-training-fixture'
    single_game.model_version = VERSION+':'+_digest(single_game._payload())[:24]
    estimate = single_game.estimate(state, action, state_generalization=True)
    assert estimate['support_status'] == 'UNKNOWN' and estimate['expected_gain'] is None
    assert estimate['support']['physical_matches'] == 1
    assert estimate['support']['state_pattern_extrapolation'] is False


def test_headlines_are_already_part_of_before_forecast_and_roles_are_used(fitted):
    cases, model, _, _ = fitted
    before = _state_for(cases[-1], 0)
    p = model.predict(before)
    assert p['known_character_features'] > 0
    after = _state_for(cases[-1], 1)
    q = model.predict(after)
    assert p['probabilities'] != q['probabilities']
    assert model.estimate(after, _action(cases[-1]['documents'][0], 0))['reason'] == 'already_read_document'
    after['documents'][0]['text'] = 'Home Club has a weak attack.'
    swapped = deepcopy(after)
    for key in ('team_id', 'name'):
        swapped['fixture']['home_'+key], swapped['fixture']['away_'+key] = swapped['fixture']['away_'+key], swapped['fixture']['home_'+key]
    assert model.f._vector(model._probabilities(after)[0])[0].tolist() != model.f._vector(model._probabilities(swapped)[0])[0].tolist()


def test_unknown_source_is_unknown_and_not_a_zero_gain(fitted):
    cases, model, _, _ = fitted
    state = _state_for(cases[-1], 0)
    action = _action(cases[-1]['documents'][0], 0)
    state['observed_links'][0]['url'] = action['url'] = 'https://new-source.test/article'
    estimate = model.estimate(state, action)
    assert estimate['support_status'] == 'UNKNOWN'
    assert estimate['expected_gain'] is None


def test_same_candidate_is_conditioned_on_already_read_state(fitted):
    cases, model, _, _ = fitted
    case = cases[-1]
    action = _action(case['documents'][1], 1)
    before, after = _state_for(case, 0), _state_for(case, 1)
    first, second = model.estimate(before, action), model.estimate(after, action)
    assert first['support_status'] == second['support_status'] == 'prototype_supported'
    assert abs(first['expected_gain']-second['expected_gain']) > 1e-12


def test_frozen_pair_roundtrip_and_training_identity_rejected(fitted):
    cases, model, _, root = fitted
    loaded = HistoricalContextModel.load(root/'model')
    state = _state_for(cases[-1], 0)
    action = _action(cases[-1]['documents'][0], 0)
    assert loaded.predict(state) == model.predict(state)
    assert loaded.estimate(state, action) == model.estimate(state, action)
    with pytest.raises(ValueError, match='training fixture'):
        loaded.predict(_state_for(cases[0], 0))
    loaded.g.coefficients[0] += .01
    with pytest.raises(ValueError, match='artifact changed'):
        loaded.estimate(state, action)


def test_source_bytes_and_label_tampering_are_rejected(fitted):
    cases, _, _, _ = fitted
    changed = deepcopy(cases)
    changed[0]['outcome'] = (changed[0]['outcome']+1) % 3
    with pytest.raises(ValueError, match='archived regulation result'):
        prepare_historical_cases(changed)
    changed = deepcopy(cases)
    changed[0]['documents'][0]['raw_sha256'] = '0'*64
    with pytest.raises(ValueError, match='source bytes'):
        prepare_historical_cases(changed)


def test_small_corpus_reports_not_ready_without_padding_or_fabrication(fitted):
    cases, _, _, _ = fitted
    model, report = fit_historical_context(cases[:3], dataset_version='constructed-too-small')
    assert model is None and report['status'] == 'not_ready'
    assert report['audit']['physical_matches'] == 3


def test_A_selection_binds_frozen_f_and_recomputes_B_gains_without_changing_legacy(fitted, tmp_path):
    import math
    cases, legacy, _, _ = fitted
    assert 'f_selection' not in legacy.metadata
    model, report = fit_historical_context(cases, tmp_path/'selected',
        dataset_version='constructed-A-only-selection-contract', a_fraction=.5, b_fraction=.25,
        f_fit_scheme='a_group_loo_v1', g_fit_scheme='b_group_loo_v1')
    assert report['f_selection']['status'] == 'fitted'
    assert report['f_selection']['training_physical_match_ids'] == sorted(report['splits']['A'])
    assert report['f_selection']['training_states'] == report['A_prefixes']
    assert model._payload()['f_parameters']['fit_scheme'] == 'a_group_loo_v1'
    assert model._payload()['f_parameters']['regularization'] == report['f_selection']['selected_regularization']
    loaded = HistoricalContextModel.load(tmp_path/'selected')
    assert loaded.predict(_state_for(cases[-1], 1)) == model.predict(_state_for(cases[-1], 1))
    targets = json.loads((tmp_path/'selected'/'B-targets.json').read_text(encoding='utf-8'))
    by_id = {c['fixture']['physical_match_id']: c for c in cases}
    for row in targets:
        case = by_id[row['physical_match_id']]
        before = _state_for(case, discovered_indices=row['discovered_indices'], acquired_indices=row['acquired_indices'])
        after = _state_for(case, discovered_indices=row['discovered_indices'],
                           acquired_indices=row['acquired_indices']+[row['action_index']])
        p, q = loaded._probabilities(before)[1], loaded._probabilities(after)[1]
        assert row['gain'] == pytest.approx(math.log(q[row['outcome']])-math.log(p[row['outcome']]))
    if report['f_selection']['base_only_won']:
        assert all(row['gain'] == 0 for row in targets)
        assert loaded.predict(_state_for(cases[-1], 1))['context_signal_enabled'] is False


def test_group_ridge_dual_matches_weighted_primal_and_constant_is_fixture_weighted():
    import numpy as np
    from football_research.acquisition.historical_context_training import _group_weights, _weighted_gain_ridge
    rows = [{'physical_match_id': mid} for mid in ('a', 'a', 'a', 'b', 'c')]
    weights = _group_weights(rows)
    x = np.array([[1., 0., 1.], [1., 1., 0.], [1., 1., 1.], [1., 3., 2.], [1., -2., 1.]])
    y = np.array([.1, .2, -.1, .8, -.4])
    beta, report = _weighted_gain_ridge(x, y, weights, .01)
    penalty = np.diag([0., .01, .01])
    primal = np.linalg.solve(x.T@(x*weights[:, None])+penalty, x.T@(y*weights))
    assert np.allclose(beta, primal, atol=1e-12)
    assert weights[:3].sum() == pytest.approx(1/3)
    assert report['objective'] == pytest.approx(weights@((y-x@primal)**2)+primal@penalty@primal)
    constant, report = _weighted_gain_ridge(x, y, weights, None)
    assert constant[0] == pytest.approx((np.mean(y[:3])+y[3]+y[4])/3)
    assert list(constant[1:]) == [0., 0.]
    assert report['effective_nonintercept_degrees_of_freedom'] == 0


def test_b_selection_holds_out_whole_fixtures_and_keeps_constant_winner(fitted):
    from football_research.acquisition.historical_context_training import _fit_b_group_loo, G_SELECTION_GRID
    _, _, _, root = fitted
    targets = json.loads((root/'model'/'B-targets.json').read_text())
    # Constructed regression contract: constant targets must select the simpler
    # model rather than demanding a context effect or positive outcome.
    for row in targets:
        row['gain'] = -.25
    model, selection = _fit_b_group_loo(targets)
    assert selection['constant_baseline_won'] is True
    assert selection['selected_regularization'] is None
    assert selection['candidate_regularizations'] == list(G_SELECTION_GRID)
    assert model.coefficients[0] == pytest.approx(-.25)
    assert set(model.coefficients[1:]) == {0.}
    assert len(selection['folds']) == 3
    for fold in selection['folds']:
        assert fold['held_out_physical_match_id'] not in fold['training_physical_match_ids']
        assert len(fold['training_physical_match_ids']) == 2
        assert fold['validation_transitions'] == 6
        assert fold['training_transitions'] == 12
        assert len(fold['hypotheses']) == len(G_SELECTION_GRID)


def test_gain_encoder_never_fits_held_out_text(fitted):
    from football_research.acquisition.historical_context_training import _gain_design, _grams
    _, _, _, root = fitted
    targets = json.loads((root/'model'/'B-targets.json').read_text())
    held_out = targets[-1]['physical_match_id']
    train = [row for row in targets if row['physical_match_id'] != held_out]
    encoder, _ = _gain_design(train)
    context = deepcopy(targets[-1]['context'])
    context['texts'].append('zzzzzzholdoutonlyzzzzzz')
    before = deepcopy(encoder.encoder)
    encoder._vector(context)
    assert encoder.encoder == before
    assert 'zzz' not in encoder.encoder['vocabulary']
    assert all(record['physical_match_id'] != held_out for record in encoder.support_records)


def test_explicit_b_selection_artifact_roundtrips_and_old_scheme_stays_unchanged(fitted, tmp_path):
    cases, old, _, _ = fitted
    assert 'g_selection' not in old.metadata
    new, report = fit_historical_context(cases, tmp_path/'selected', dataset_version='constructed-selection-contract',
        a_fraction=.5, b_fraction=.25, g_fit_scheme='b_group_loo_v1')
    assert new is not None, report
    selected = report['g_selection']
    assert set(selected['selection_physical_match_ids']) == set(report['splits']['B'])
    assert not set(selected['selection_physical_match_ids']) & set(report['splits']['A']+report['splits']['C'])
    payload = json.loads((tmp_path/'selected'/'model.json').read_text())
    assert payload['g_parameters']['regularization'] == selected['selected_regularization']
    assert payload['g_parameters']['constant_model'] == selected['constant_baseline_won']
    loaded = HistoricalContextModel.load(tmp_path/'selected')
    state = _state_for(cases[-1], 0)
    action = _action(cases[-1]['documents'][0], 0)
    assert loaded.predict(state) == new.predict(state)
    assert loaded.estimate(state, action) == new.estimate(state, action)
    assert 'g_selection' not in old.metadata
    changed = deepcopy(cases)
    changed[-1]['documents'][0]['text'] += ' Changed held-out C text cannot select B regularization.'
    changed_raw = tmp_path/'changed-held-out-C.txt'
    changed_raw.write_text(changed[-1]['documents'][0]['text'], encoding='utf-8')
    changed[-1]['documents'][0]['raw_path'] = str(changed_raw)
    changed[-1]['documents'][0]['raw_sha256'] = sha256(changed_raw.read_bytes()).hexdigest()
    refit, changed_report = fit_historical_context(changed, dataset_version='constructed-selection-contract',
        a_fraction=.5, b_fraction=.25, g_fit_scheme='b_group_loo_v1')
    assert changed_report['g_selection'] == selected
    assert refit.f.coefficients == new.f.coefficients
    assert refit.g.coefficients == new.g.coefficients
    with pytest.raises(ValueError, match='gain fitting scheme'):
        fit_historical_context(cases, dataset_version='invalid', g_fit_scheme='choose_positive')
