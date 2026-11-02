"""Verified constructed fixtures exercise role interactions, not real utility."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket

import numpy as np
import pytest

from football_research.acquisition.context_predictor import PARAMETERS, _digest, _state
from football_research.acquisition.fixture_roles import ROLES, bind_fixture_roles, role_state
from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor
from football_research.acquisition.role_context_predictor import RoleContextPredictor, VERSION, training_readiness
from test_fixture_roles import identity_inputs, bound_visible, visible_for

AS_OF = '2026-09-15T00:00:00+00:00'
TEXT = 'HJK striker is unavailable for selection.'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*a, **k):
        raise AssertionError('Role model checks must remain offline')
    monkeypatch.setattr(socket, 'getaddrinfo', denied)
    monkeypatch.setattr(socket.socket, 'connect', denied)


def inputs(root, native, *, day=20, reverse=False):
    return identity_inputs(root, home='VPS' if reverse else 'HJK', away='HJK' if reverse else 'VPS',
        native=str(native), date=f'202609{day:02}',
        at=datetime(2026, 9, day, 7, tzinfo=timezone.utc))


@pytest.fixture(scope='module')
def sample(tmp_path_factory):
    root = tmp_path_factory.mktemp('role-f')
    rows = []
    for i in range(12):
        y = i % 3
        case = inputs(root/str(i), 4900000+i, day=i+1, reverse=y == 0)
        text = 'HJK and VPS squads remain unchanged.' if y == 1 else TEXT
        for prefix in (0, 1):
            view = bound_visible(case, *([text] if prefix else []),
                                 decision_at=case['at']+timedelta(minutes=2+prefix))
            rows.append({'physical_match_id': case['binding']['match_id'], 'visible_state': view,
                'purpose': 'f_train', 'outcome': y, 'result_available_at': f'2026-09-{i+1:02}T19:00:00+00:00',
                'run_id': 'constructed-role-run-'+str(i), 'prefix_index': prefix})
    future = [inputs(root/'future-home', 4901000), inputs(root/'future-away', 4901001, reverse=True)]
    return rows, future


def fit(rows):
    return RoleContextPredictor().fit(rows, as_of=AS_OF, dataset_version='constructed-role-channel-contract-v1',
                                      evidence_origin='constructed_mechanism')


@pytest.fixture(scope='module')
def model(sample):
    return fit(sample[0])


def test_identical_text_moves_between_independent_role_channels_and_fitted_forecasts(sample, model):
    first, second = [bound_visible(case, TEXT) for case in sample[1]]
    a, b = role_state(first), role_state(second)
    assert a['texts'] == b['texts'] == [TEXT]
    assert a['role_texts']['home'] == b['role_texts']['away'] == [TEXT]
    assert a['numeric'] == b['numeric'] and a['sources'] == b['sources']
    x, _ = model._vector(a)
    z, _ = model._vector(b)
    n = len(model.encoder['vocabulary'])
    assert not np.array_equal(x, z)
    np.testing.assert_allclose(x[:n], z[n:2*n], atol=1e-15, rtol=0)
    np.testing.assert_allclose(x[n:2*n], z[:n], atol=1e-15, rtol=0)
    assert np.linalg.norm(x[:len(ROLES)*n]) == pytest.approx(1.)
    p = model.predict(first, physical_match_id=a['physical_match_id'])
    q = model.predict(second, physical_match_id=b['physical_match_id'])
    assert p['probabilities'][2] > p['probabilities'][0]
    assert q['probabilities'][0] > q['probabilities'][2]
    assert p['numeric_information_value'] is q['numeric_information_value'] is None
    assert p['calibration_status'] == 'not_established'
    # Exact same old representation remains role-invariant. It is not modified.
    legacy = OptionalPriorContextPredictor().fit(sample[0], as_of=AS_OF, dataset_version='old-control',
                                               evidence_origin='constructed_mechanism')
    assert legacy.predict(first, physical_match_id=a['physical_match_id'])['probabilities'] == legacy.predict(
        second, physical_match_id=b['physical_match_id'])['probabilities']


def test_independent_training_weight_gates_and_role_coverage_remain_explicit(sample, model):
    rows = sample[0]
    ready = training_readiness(rows[:12]*20, as_of=AS_OF, evidence_origin='constructed_mechanism')
    assert ready['physical_matches'] == 6 and ready['distinct_prefixes'] == 12 and not ready['ready_to_fit']
    with pytest.raises(ValueError, match='not_ready'):
        fit(rows[:12]*20)
    coverage = model.metadata['role_coverage']['missing_prior']
    assert {r: coverage[r]['physical_matches'] for r in ROLES} == {
        'home': 4, 'away': 4, 'both': 4, 'unattributed': 0, 'ambiguous': 0}
    assert model.metadata['readiness']['physical_matches'] == 12
    assert model.metadata['empirical_base']['class_counts'] == {'0': 4, '1': 4, '2': 4}
    assert PARAMETERS['min_physical_matches'] == 12 and PARAMETERS['regularization'] == 10.
    repeated = fit(rows+[deepcopy(row) for row in rows for _ in range(3)])
    assert repeated.encoder == model.encoder
    np.testing.assert_array_equal(repeated.coefficients, model.coefficients)


def test_unattributed_text_is_retained_and_unknown_role_disclosed_not_scored_useless(sample, model):
    view = bound_visible(sample[1][0], 'A striker is unavailable for selection.')
    state = role_state(view)
    assert state['role_texts']['unattributed'] == ['A striker is unavailable for selection.']
    vector, known = model._vector(state)
    n = len(model.encoder['vocabulary'])
    assert known > 0 and np.linalg.norm(vector[3*n:4*n]) > 0
    result = model.predict(view, physical_match_id=state['physical_match_id'])
    assert result['role_support']['unseen_training_roles'] == ['unattributed']
    assert result['role_support_status'] == 'unseen_role_extrapolation'
    assert result['numeric_information_value'] is None and not result['production_ready']


def test_original_market_prior_domain_does_not_borrow_missing_domain_support(sample, model):
    view = bound_visible(sample[1][0], TEXT, prior=[.6, .25, .15])
    result = model.predict(view, physical_match_id=sample[1][0]['binding']['match_id'])
    assert result['probabilities'] == [.6, .25, .15]
    assert result['fallback_reason'] == 'provided_prior_context_unsupported'
    assert result['context_effect_status'] == 'UNKNOWN'
    assert result['prior_domain_context_supported'] is False


def test_empty_and_out_of_vocabulary_keep_only_training_learned_base(sample, model):
    for texts in ((), ('龘靐齉爩龗',)):
        view = bound_visible(sample[1][0], *texts)
        result = model.predict(view, physical_match_id=sample[1][0]['binding']['match_id'])
        assert result['probabilities'] == model.metadata['empirical_base']['probabilities']
        assert result['logit_residual'] == [0., 0., 0.]


def test_missing_tampered_identity_and_renamed_training_fixture_are_rejected(sample, model):
    raw = visible_for(sample[1][0], TEXT)
    with pytest.raises(ValueError, match='sidecar'):
        model.predict(raw, physical_match_id=sample[1][0]['binding']['match_id'])
    view = bound_visible(sample[1][0], TEXT)
    changed = deepcopy(view)
    changed['before']['fixture']['home_team_id'], changed['before']['fixture']['away_team_id'] = (
        changed['before']['fixture']['away_team_id'], changed['before']['fixture']['home_team_id'])
    with pytest.raises(ValueError, match='scope|roles'):
        model.predict(changed, physical_match_id=sample[1][0]['binding']['match_id'])
    with pytest.raises(ValueError, match='physical identity'):
        model.predict(view, physical_match_id='renamed-physical-event')
    with pytest.raises(ValueError, match='Training physical'):
        model.predict(sample[0][1]['visible_state'], physical_match_id=sample[0][1]['physical_match_id'])


def test_training_requires_verified_physical_id_labels_and_only_A(sample):
    for field, replacement, reason in [('physical_match_id', 'fake-alias', 'identity'),
                                      ('purpose', 'calibration', 'f_train'),
                                      ('result_available_at', '2026-09-16T00:00:00Z', 'Settlement')]:
        rows = deepcopy(sample[0])
        rows[0][field] = replacement
        with pytest.raises(ValueError, match=reason):
            fit(rows)


def test_role_forecast_roundtrip_and_new_schema_cannot_load_as_old(sample, model, tmp_path):
    path = tmp_path/'role.json'
    model.save(path)
    loaded = RoleContextPredictor.load(path)
    view = bound_visible(sample[1][0], TEXT)
    assert loaded.predict(view, physical_match_id=sample[1][0]['binding']['match_id']) == model.predict(
        view, physical_match_id=sample[1][0]['binding']['match_id'])
    with pytest.raises(ValueError, match='schema'):
        OptionalPriorContextPredictor.load(path)
    old = OptionalPriorContextPredictor().fit(sample[0], as_of=AS_OF, dataset_version='old-artifact',
                                            evidence_origin='constructed_mechanism')
    old_path = tmp_path/'old.json'
    old.save(old_path)
    with pytest.raises(ValueError, match='schema'):
        RoleContextPredictor.load(old_path)
    with pytest.raises(FileExistsError):
        model.save(path)


@pytest.mark.parametrize('target', ['encoder', 'parameters', 'coverage'])
def test_resigned_role_contract_tampering_cannot_change_layout_or_support(model, tmp_path, target):
    path = tmp_path/'role.json'
    model.save(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if target == 'encoder':
        payload['encoder']['role_order'].reverse()
    elif target == 'parameters':
        payload['parameters']['min_physical_matches'] = 6
    else:
        payload['metadata']['role_coverage']['missing_prior']['home']['physical_matches'] = 100
        payload['metadata']['readiness']['role_coverage'] = deepcopy(payload['metadata']['role_coverage'])
    payload['model_version'] = VERSION+':'+_digest({k: v for k, v in payload.items() if k != 'model_version'})[:24]
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='schema|encoder|coverage'):
        RoleContextPredictor.load(path)


def development_rows(sample):
    rows = deepcopy(sample[0])
    for case, text in zip(sample[1], ('A striker is unavailable for selection.', TEXT)):
        for index in (0, 1):
            rows.append({'physical_match_id': case['binding']['match_id'], 'run_id': case['binding']['match_id']+'-run',
                'prefix_index': index, 'outcome': 0, 'result_available_at': '2026-09-20T19:00:00+00:00',
                'visible_state': bound_visible(case, *([text] if index else []),
                    decision_at=case['at']+timedelta(minutes=2+index))})
    return rows


def test_role_development_reuses_split_and_retains_unknown_role_overall_losses(sample):
    from football_research.evaluation.role_context_development import evaluate_role_context_development
    result = evaluate_role_context_development(development_rows(sample), training_cutoff=AS_OF,
        as_of='2026-09-25T00:00:00+00:00', evidence_origin='constructed_mechanism', dataset_version='role-development-unit')
    assert result['status'] == 'evaluated' and result['schema'] == 'role-context-retrospective-development-v1'
    assert result['readiness']['physical_matches'] == 12
    terminal = result['summary']['terminal']
    assert terminal['all_forecasts_including_explicit_fallbacks']['physical_matches'] == 2
    assert terminal['supported_context_comparisons_only']['physical_matches'] == 1
    assert terminal['role_extrapolation_physical_matches'] == 1
    assert result['per_match'][0]['terminal']['context_gain'] is None
    assert not result['independent_confirmation'] and not result['information_value_established']


def test_role_development_validates_pending_sidecars_before_excluding_unlabelled_matches(sample):
    from football_research.evaluation.role_context_development import evaluate_role_context_development
    rows = development_rows(sample)
    for row in rows[-2:]:
        row['outcome'] = row['result_available_at'] = None
    rows[-1]['visible_state'].pop('fixture_roles')
    with pytest.raises(ValueError, match='sidecar'):
        evaluate_role_context_development(rows, training_cutoff=AS_OF, as_of='2026-09-25T00:00:00+00:00',
            evidence_origin='constructed_mechanism', dataset_version='pending-role-contract')


def test_shared_development_hooks_preserve_original_complete_report(sample, monkeypatch):
    import football_research.acquisition.optional_prior_predictor as optional
    import football_research.evaluation.optional_context_development as current
    backup = Path('var/parallel/agent/round203/source-before/football_research/evaluation/optional_context_development.py')
    old = {'__name__': 'football_research.evaluation._frozen_round202',
           '__package__': 'football_research.evaluation'}
    exec(compile(backup.read_text(encoding='utf-8'), str(backup), 'exec'), old)
    class Clock:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 10, 7, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(current, 'datetime', Clock)
    monkeypatch.setattr(optional, 'datetime', Clock)
    old['datetime'] = Clock
    args = {'training_cutoff': AS_OF, 'as_of': '2026-09-25T00:00:00+00:00',
            'evidence_origin': 'constructed_mechanism', 'dataset_version': 'old-default-compatibility'}
    rows = development_rows(sample)
    assert current.evaluate_context_development(rows, **args) == old['evaluate_context_development'](rows, **args)
