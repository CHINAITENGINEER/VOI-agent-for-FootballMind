"""Temporal/feature invariants on constructed data; not football results."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.fixture_roles import ROLES
from football_research.acquisition.match_conditioned_f import (
    MatchConditionedF, ReconstructedMatchPrior, _encoder, _fit, _predict, _vector,
    bind_prior, fit_forward_f, project,
)


NOW = datetime(2025, 9, 1, tzinfo=timezone.utc)


def history():
    rows = []
    for i in range(13):
        kick = NOW+timedelta(days=i)
        home, away = ('North', 'South') if i % 2 == 0 else ('South', 'North')
        rows.append({'source': 'constructed', 'source_id': str(i), 'match_id': str(i),
            'home_team_id': home, 'away_team_id': away,
            'home_team_name': home, 'away_team_name': away,
            'competition_id': 'league', 'competition_name': 'League',
            'kickoff': kick.isoformat(), 'result_available_at': (kick+timedelta(hours=3)).isoformat(),
            'home_goals': 2 if home == 'North' else 0, 'away_goals': 0 if home == 'North' else 2,
            'score_period': 'regulation', 'status': 'finished', 'information_mode': 'historical_reconstruction'})
    return rows


def state_and_snapshot():
    fixture = {'physical_match_id': 'constructed:10', 'match_id': '10',
        'home_team_id': 'North', 'away_team_id': 'South', 'home_name': 'North', 'away_name': 'South',
        'competition': 'League', 'competition_id': 'league',
        'kickoff': (NOW+timedelta(days=10)).isoformat()}
    decision = NOW+timedelta(days=9, hours=23)
    state = {'fixture': fixture, 'decision_at': decision.isoformat(), 'prediction_cutoff': decision.isoformat(),
        'information_mode': 'historical_reconstruction', 'documents': [], 'observed_links': [], 'history': []}
    prior = ReconstructedMatchPrior(history(), source_binding={'constructed': True})
    return state, prior.snapshot(fixture, decision)


def doc(text, *, title='North team update', url='https://example.test/news/one'):
    return {'url': url, 'title': title, 'text': text,
        'source_published_at': (NOW+timedelta(days=9)).isoformat(),
        'captured_at': (NOW+timedelta(days=15)).isoformat()}


def training_rows(*, text=True):
    rows = []
    for i in range(9):
        start = NOW+timedelta(days=i*7)
        words = ['North fitness update '+['strong attack', 'balanced draw', 'weak defence'][i % 3]] if text else []
        for read in (False, True):
            block = {r: words if r == ROLES[0] else [] for r in ROLES}
            rows.append({'physical_match_id': 'example:'+str(i), 'outcome': i % 3,
                'result_available_at': (start+timedelta(hours=6)).isoformat(),
                'state': {'physical_match_id': 'example:'+str(i), 'decision_at': start.isoformat(),
                    'prior': [.5, .25, .25],
                    'channels': {'title': [], 'body': [block] if read and text else []}}})
    return rows


def test_current_and_future_outcome_changes_cannot_enter_match_prior():
    state, reference = state_and_snapshot()
    changed = history()
    for row in changed[10:]:
        row['home_goals'], row['away_goals'] = 0, 9
    prior = ReconstructedMatchPrior(changed, source_binding={'constructed': True})
    assert prior.snapshot(state['fixture'], state['decision_at']) == reference
    assert 'constructed:10' not in reference['training_match_ids']
    assert len(reference['training_match_ids']) == 10
    assert reference['probabilities'][0] > reference['probabilities'][2]
    assert reference['operational_eligible'] is False


def test_result_at_cutoff_is_excluded_and_target_identity_is_strict():
    state, _ = state_and_snapshot()
    prior = ReconstructedMatchPrior(history(), source_binding={})
    snapshot = prior.snapshot(state['fixture'], history()[9]['result_available_at'])
    assert 'constructed:9' not in snapshot['training_match_ids']
    bad = deepcopy(state['fixture']); bad['home_name'] = 'South'
    with pytest.raises(ValueError, match='Ordered team'):
        prior.snapshot(bad, state['decision_at'])
    with pytest.raises(ValueError, match='precede kickoff'):
        prior.snapshot(state['fixture'], state['fixture']['kickoff'])


def test_baseline_binding_rejects_tampering_future_and_operational_use():
    state, snapshot = state_and_snapshot()
    assert bind_prior(state, snapshot)['prior_probabilities'] == pytest.approx(snapshot['probabilities'])
    corrupt = deepcopy(snapshot); corrupt['probabilities'] = [.2, .3, .5]
    with pytest.raises(ValueError):
        bind_prior(state, corrupt)
    corrupt['snapshot_sha256'] = _digest({k: v for k, v in corrupt.items() if k != 'snapshot_sha256'})
    with pytest.raises(ValueError, match='probabilities differ'):
        bind_prior(state, corrupt)
    with pytest.raises(ValueError):
        bind_prior({**state, 'decision_at': NOW.isoformat()}, snapshot)
    with pytest.raises(ValueError):
        bind_prior({**state, 'information_mode': 'operational'}, snapshot)


def test_body_reveal_does_not_renormalize_known_title_and_clock_has_no_effect():
    state, snapshot = state_and_snapshot()
    article = doc('North defence training South attack')
    state['observed_links'] = [{'url': article['url'], 'title': article['title'],
        'available_at': article['source_published_at'], 'title_basis': 'historical_headline_panel'}]
    before = project(bind_prior(state, snapshot))
    after_state = {**state, 'documents': [article]}
    after = project(bind_prior(after_state, snapshot))
    assert before['channels']['title'] == after['channels']['title']
    assert before['channels']['body'] == []
    assert len(after['channels']['body']) == 1
    rows = [{'physical_match_id': 'one', 'state': after}]
    encoder = _encoder(rows)
    vector = _vector(encoder, after)
    later = deepcopy(after_state)
    later['decision_at'] = later['prediction_cutoff'] = (NOW+timedelta(days=9, hours=23, minutes=1)).isoformat()
    later['history'] = [{'status': 'FAILED'}]
    np.testing.assert_array_equal(vector, _vector(encoder, project(bind_prior(later, snapshot))))
    np.testing.assert_array_equal(_vector(encoder, before)[:len(ROLES)*len(encoder['vocabulary'])],
                                  vector[:len(ROLES)*len(encoder['vocabulary'])])


def test_oov_new_body_does_not_dilute_old_body_and_identical_body_is_deduplicated():
    state, snapshot = state_and_snapshot()
    first = doc('North defence training South attack')
    state['documents'] = [first]
    projected = project(bind_prior(state, snapshot))
    encoder = _encoder([{'physical_match_id': 'one', 'state': projected}])
    initial = _vector(encoder, projected)
    # The title is already known; only truly OOV body text is appended.
    state['documents'].append(doc('棋谱沙漠银河图谱', title=first['title'], url='https://example.test/news/two'))
    np.testing.assert_array_equal(initial, _vector(encoder, project(bind_prior(state, snapshot))))
    state['documents'] = [first, {**first, 'url': 'https://example.test/news/duplicate'}]
    np.testing.assert_array_equal(initial, _vector(encoder, project(bind_prior(state, snapshot))))


def test_unread_body_and_later_title_are_not_visible():
    state, snapshot = state_and_snapshot()
    baseline = project(bind_prior(state, snapshot))
    state['unread_body'] = 'North loses heavily'
    assert project(bind_prior(state, snapshot)) == baseline
    state['observed_links'] = [{'url': 'https://example.test/news/later', 'title': 'Future news',
        'title_basis': 'historical_headline_panel', 'available_at': (NOW+timedelta(days=11)).isoformat()}]
    with pytest.raises(ValueError, match='not visible'):
        project(bind_prior(state, snapshot))


def test_forward_fold_coefficients_do_not_depend_on_heldout_labels():
    rows = training_rows()
    before = fit_forward_f(rows, dataset_binding={'constructed': True})
    changed = deepcopy(rows)
    for row in changed:
        if row['physical_match_id'] == 'example:6':
            row['outcome'] = 2
    after = fit_forward_f(changed, dataset_binding={'constructed': True})
    a, b = before.metadata['selection']['folds'][0], after.metadata['selection']['folds'][0]
    assert a['latest_training_result'] < a['first_validation_decision']
    assert a['training_ids'] == b['training_ids']
    for left, right in zip(a['hypotheses'], b['hypotheses']):
        assert left['encoder_sha256'] == right['encoder_sha256']
        assert left['coefficients_sha256'] == right['coefficients_sha256']
    assert set(a['training_ids']).isdisjoint(a['validation_ids'])


def test_no_text_selects_prior_only_and_save_load_is_exact(tmp_path):
    model = fit_forward_f(training_rows(text=False), dataset_binding={'constructed': True})
    assert model.metadata['selection']['base_only_selected'] is True
    state, snapshot = state_and_snapshot()
    expected = model.predict(state, prior_snapshot=snapshot)
    assert expected['probabilities'] == pytest.approx(snapshot['probabilities'])
    path = tmp_path/'model.json'; model.save(path)
    assert MatchConditionedF.load(path).predict(state, prior_snapshot=snapshot) == expected
    with pytest.raises(FileExistsError):
        model.save(path)
    raw = json.loads(path.read_text(encoding='utf-8')); raw['metadata']['training_states'] += 1
    path.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='digest mismatch'):
        MatchConditionedF.load(path)


def test_duplicate_prefixes_do_not_reweight_a_match():
    rows = training_rows()
    enc, coef, _ = _fit(rows, 1.)
    duplicated = rows+deepcopy([r for r in rows if r['physical_match_id'] == 'example:0'])
    enc2, coef2, _ = _fit(duplicated, 1.)
    assert enc == enc2
    np.testing.assert_allclose(coef, coef2, atol=1e-10, rtol=1e-9)
    state = deepcopy(rows[1]['state'])
    p = _predict(enc, coef, state)
    state['prior'] = [.2, .3, .5]
    assert not np.allclose(p, _predict(enc, coef, state))
