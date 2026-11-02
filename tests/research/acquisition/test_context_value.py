"""Fixed-f constructed gain tests, not an evaluation on real football labels."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
import pytest

from football_research.acquisition.context_gain import VERSION as GAIN_VERSION, _digest, _query_input, _signed_records
from football_research.acquisition.context_predictor import ContextPredictor
from football_research.acquisition.context_value import ContextValueModel, training_readiness, PARAMETERS, _public
from test_context_gain import transition, save
from test_context_predictor import visible, records, WORDS, AS_OF


G_AS_OF = '2027-02-01T00:00:00+00:00'
CONTEXTS = {'positive': WORDS[1], 'negative': WORDS[2]}


class ConstructedFitClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2027, 2, 2, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Context value fitting and prediction must use zero network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    monkeypatch.setattr('football_research.acquisition.context_value.datetime', ConstructedFitClock)


def _fixture_part(frozen, index, *, context='positive', physical=None, failure=False):
    """Actual known-prefix -> actual observed-prefix, with independent target."""
    physical = physical or f'B:{index}'
    parts = transition(frozen, physical=physical, outcome=0 if context == 'positive' else 1,
                       text=None if failure else WORDS[0], status='FAILED' if failure else 'OK')
    prefixes, pairs, plans = parts
    before = json.loads(json.dumps(visible('provider:'+physical, text=CONTEXTS[context])).replace('2025-', '2027-'))
    before['candidates'] = prefixes[0]['visible_state']['candidates']
    for doc in before['before']['documents']:
        doc['document_id'] = 'known:'+physical
    after = prefixes[1]['visible_state']
    after['before']['documents'] = deepcopy(before['before']['documents'])+after['before']['documents']
    after['before']['contents'] = deepcopy(before['before']['contents'])+after['before']['contents']
    for n, doc in enumerate(after['before']['documents']):
        doc['document_index'] = n
    prefixes[0]['visible_state'] = before
    # One fixed schedule established before fitting, not selected by g's result.
    day = 20+index % 3
    prefixes, pairs, plans = json.loads(json.dumps([prefixes, pairs, plans]).replace('2027-01-20', f'2027-01-{day:02}'))
    return prefixes, pairs, plans


def gain_bundle(frozen, folder, *, count=24, failure=True, all_negative=False):
    folder.mkdir(parents=True)
    prefixes, pairs, plans = [], [], {}
    for i in range(count):
        part = _fixture_part(frozen, i, context='positive' if not all_negative and i < count//2 else 'negative',
                             failure=failure and i == count//2-1)
        prefixes += part[0]
        pairs += part[1]
        plans.update(part[2])
    features, targets, excluded = _signed_records(prefixes, pairs, plans, model=frozen[0],
        model_sha=frozen[2], as_of=G_AS_OF, origin='constructed_mechanism')
    assert not excluded
    source = {'schema': 'constructed-fixed-f-prefix-pairs-v1', 'input_bindings': {},
              'prefixes': prefixes, 'transitions': pairs, 'scope': 'constructed_mechanism_only'}
    save(folder/'features.json', features)
    save(folder/'targets.json', targets)
    save(folder/'source-inputs.json', source)
    save(folder/'calibration-registrations.json', list(plans.values()))
    report = {'schema': GAIN_VERSION, 'as_of': G_AS_OF, 'evidence_origin': 'constructed_mechanism',
        'predictor_sha256': frozen[2], 'predictor_model_version': frozen[0].model_version,
        'features_sha256': _digest(features), 'targets_sha256': _digest(targets),
        'source_inputs_file_sha256': sha256((folder/'source-inputs.json').read_bytes()).hexdigest(),
        'registrations_file_sha256': sha256((folder/'calibration-registrations.json').read_bytes()).hexdigest()}
    save(folder/'report.json', report)
    return folder


def fitted(tmp_path):
    """Public helper for policy tests: constructed f, frozen path, fitted g."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    f = ContextPredictor().fit(records(), as_of=AS_OF, dataset_version='constructed-g-value-f',
                               evidence_origin='constructed_mechanism')
    fpath = tmp_path/'f.json'
    f.save(fpath)
    frozen = (f, fpath, sha256(fpath.read_bytes()).hexdigest())
    bundle = gain_bundle(frozen, tmp_path/'gains')
    with patch('football_research.acquisition.context_value.datetime', ConstructedFitClock):
        g = ContextValueModel().fit(bundle, predictor_path=fpath, as_of=G_AS_OF,
                                   dataset_version='constructed-g-value-B')
    return f, fpath, g


def query_inputs(f, *, context='positive', day=10):
    """Same visible request, different already-read context; February 2027."""
    dummy = (f, Path('not-read'), '0'*64)
    prefixes, pairs, _ = _fixture_part(dummy, 0, context=context, physical='heldout:future')
    state = json.loads(json.dumps(prefixes[0]['visible_state']).replace('2027-01-20', f'2027-02-{day:02}'))
    forecast = f.predict(state, physical_match_id='heldout:future')['probabilities']
    return _query_input(state, pairs[0]['action'], forecast)


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp('context-value')
    return (*fitted(root), root)


def predict(models, context='positive', **kwargs):
    f, path, g, _ = models
    return g.predict(query_inputs(f, context=context), physical_match_id=kwargs.get('physical', 'heldout:future'),
                     predictor_sha256=sha256(path.read_bytes()).hexdigest())


def test_same_request_different_known_context_learns_distinct_signed_estimates(models):
    a, b = predict(models), predict(models, 'negative')
    assert a['support_status'] == b['support_status'] == 'supported'
    assert abs(a['expected_signed_gain']-b['expected_signed_gain']) > 1e-10
    assert a['support']['physical_matches'] >= 12 and b['support']['kickoff_dates'] >= 2
    assert a['reason'] and a['value_kind'] == 'empirical_signed_fixed_predictor_gain_not_Bayes_EVSI'
    assert a['uncertainty'] is a['stop_certificate'] is None and not a['cost_deducted']
    assert models[2].metadata['negative_targets'] > 0
    rows = json.loads((models[3]/'gains/targets.json').read_text())
    assert any(row['response_status'] == 'FAILED' for row in rows)
    assert any(row['signed_gain'] < 0 for row in rows) and any(row['signed_gain'] > 0 for row in rows)


def test_unread_response_outcome_gain_and_fixture_ids_never_enter_features(models):
    f, path, g, _ = models
    inputs = query_inputs(f)
    original = g.predict(inputs, physical_match_id='C:1', predictor_sha256=g.metadata['predictor_sha256'])
    changed = deepcopy(inputs)
    changed.update(outcome=2, signed_gain=999, future_response=WORDS[2], after={'texts': ['SECRET']})
    changed['candidate'].update(actual_response='SECRET', hidden_label=0, expected_value=99)
    changed['candidate']['request']['url'] = 'https://www.uefa.com/completely/new/nonce'
    changed['state']['match_id'] = 'winner-secret'
    assert _public(inputs) == _public(changed)
    assert g.predict(changed, physical_match_id='C:2', predictor_sha256=g.metadata['predictor_sha256']) == original


@pytest.mark.parametrize('mutation,reason', [
    ('source', 'unseen_tool_or_request_source'), ('visible_source', 'unseen_visible_source'),
    ('oov', 'visible_text_out_of_vocabulary'), ('empty', 'insufficient_independent_local_matches')])
def test_unsupported_inputs_are_none_not_zero(models, mutation, reason):
    f, _, g, _ = models
    inputs = query_inputs(f)
    if mutation == 'source':
        inputs['candidate']['request']['url'] = 'https://new-source.example/article'
    elif mutation == 'visible_source':
        inputs['state']['sources'] = ['new-source.example']
    elif mutation == 'oov':
        inputs['state']['texts'] = ['龍龘麤靐龖龗']
    else:
        inputs['state']['texts'] = []
    result = g.predict(inputs, physical_match_id='C:unknown', predictor_sha256=g.metadata['predictor_sha256'])
    assert result['support_status'] == 'UNKNOWN' and result['reason'] == reason
    assert result['expected_signed_gain'] is result['expected_gain'] is None


@pytest.mark.parametrize('mutation', ['f_identity', 'g_identity', 'wrong_f', 'time', 'future_parent'])
def test_frozen_f_and_whole_fixture_holdout_clocks(models, mutation):
    f, _, g, _ = models
    inputs, physical, digest = query_inputs(f), 'C:future', g.metadata['predictor_sha256']
    if mutation == 'f_identity':
        physical = f.metadata['training_physical_match_ids'][0]
    elif mutation == 'g_identity':
        physical = g.metadata['training_physical_match_ids'][0]
    elif mutation == 'wrong_f':
        digest = '0'*64
    elif mutation == 'time':
        inputs['state']['decision_at'] = G_AS_OF
    else:
        inputs['candidate']['observed_parents'] = [{'parent': {'document_id': 'future', 'raw_sha256': 'f'*64,
            'observed_at': '2027-02-11T00:00:00Z', 'available_at': '2027-02-11T00:00:00Z'},
            'labels': [{'text': 'unread future label'}]}]
    with pytest.raises(ValueError):
        g.predict(inputs, physical_match_id=physical, predictor_sha256=digest)


def test_save_load_binds_exact_f_and_keeps_signed_predictions(models, tmp_path):
    f, fpath, g, _ = models
    path = tmp_path/'g.json'
    g.save(path)
    restored = ContextValueModel.load(path, predictor_path=fpath)
    assert restored.predict(query_inputs(f), physical_match_id='C:future', predictor_sha256=g.metadata['predictor_sha256']) == predict(models)
    with pytest.raises(FileExistsError):
        g.save(path)
    payload = json.loads(path.read_text())
    payload['coefficients'][0] += 1
    save(tmp_path/'tampered.json', payload)
    with pytest.raises(ValueError, match='version mismatch'):
        ContextValueModel.load(tmp_path/'tampered.json', predictor_path=fpath)


def test_duplicate_bundle_does_not_manufacture_matches_or_reweight(models):
    _, fpath, g, root = models
    report = training_readiness([root/'gains', root/'gains'], predictor_path=fpath, as_of=G_AS_OF)
    assert report['physical_matches'] == 24 and report['transitions'] == 24
    duplicated = ContextValueModel().fit([root/'gains', root/'gains'], predictor_path=fpath,
                                        as_of=G_AS_OF, dataset_version='duplicate-input-test')
    assert duplicated.encoder == g.encoder
    np.testing.assert_allclose(duplicated.coefficients, g.coefficients, rtol=0, atol=1e-14)
    assert PARAMETERS['regularization'] == 20 and PARAMETERS['min_physical_matches'] == 12


def test_supported_negative_average_is_not_clamped_to_zero(models, tmp_path):
    f, fpath, _, _ = models
    frozen = f, fpath, sha256(fpath.read_bytes()).hexdigest()
    folder = gain_bundle(frozen, tmp_path/'negative', all_negative=True, failure=False)
    g = ContextValueModel().fit(folder, predictor_path=fpath, as_of=G_AS_OF, dataset_version='negative-mechanism')
    result = g.predict(query_inputs(f, context='negative'), physical_match_id='C:negative',
                       predictor_sha256=frozen[2])
    assert result['support_status'] == 'supported' and result['expected_signed_gain'] < 0


def test_insufficient_independent_matches_cannot_be_fixed_by_repeat_copies(models, tmp_path):
    f, fpath, _, _ = models
    folder = gain_bundle((f, fpath, sha256(fpath.read_bytes()).hexdigest()), tmp_path/'eleven', count=11)
    report = training_readiness([folder]*3, predictor_path=fpath, as_of=G_AS_OF)
    assert not report['ready_to_fit'] and report['physical_matches'] == 11
    with pytest.raises(ValueError, match='not_ready'):
        ContextValueModel().fit(folder, predictor_path=fpath, as_of=G_AS_OF, dataset_version='insufficient')


def test_changed_targets_rejected_even_when_derived_file_hash_is_resigned(models, tmp_path):
    _, fpath, _, root = models
    folder = tmp_path/'copy'
    folder.mkdir()
    for path in (root/'gains').glob('*.json'):
        (folder/path.name).write_bytes(path.read_bytes())
    targets = json.loads((folder/'targets.json').read_text())
    targets[0]['signed_gain'] = 100
    save(folder/'targets.json', targets)
    report = json.loads((folder/'report.json').read_text())
    report['targets_sha256'] = _digest(targets)
    save(folder/'report.json', report)
    with pytest.raises(ValueError, match='Signed gain'):
        training_readiness(folder, predictor_path=fpath, as_of=G_AS_OF)


def test_current_real_A_has_no_B_targets_and_cannot_publish_g(tmp_path):
    real = Path('var/parallel/agent/round195/model/real-A')
    report = training_readiness(real, as_of=G_AS_OF)
    assert not report['ready_to_fit'] and report['physical_matches'] == report['transitions'] == 0
    assert report['evidence_origin'] == 'real_observational'
    assert 'frozen_context_predictor_missing' in report['reasons']
    unfitted = ContextValueModel()
    with pytest.raises(ValueError, match='not_ready'):
        unfitted.fit(real, predictor_path=None, as_of=G_AS_OF, dataset_version='real-A-not-B')
    with pytest.raises(ValueError):
        unfitted.save(tmp_path/'forbidden-model.json')
    assert not (tmp_path/'forbidden-model.json').exists()
