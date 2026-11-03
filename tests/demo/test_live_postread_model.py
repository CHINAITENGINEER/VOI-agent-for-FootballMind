"""Frozen local model, causal prefix metadata and no-label input boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from demo.live_postread_model import LivePostreadModel, RELEASE, MODEL_PATH, F_SHA, G_SHA, DEFAULT_HEAD


def test_optional_body_budget_rejects_before_encoder_inference():
    from threading import RLock
    from types import SimpleNamespace
    from demo.live_postread_model import BodyEncodingBudgetExceeded
    model = object.__new__(LivePostreadModel)
    model.lock = RLock()
    contract = SimpleNamespace(fingerprint=lambda: 'fixed')
    model.contract = model.long_contract = contract
    model.caches = {'fixed': SimpleNamespace(get=lambda text: None)}
    model.encoder = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text, truncation: list(range(4097))),
        encode_one=lambda text: pytest.fail('Over-budget full body must not be encoded'))
    original = {'document_id': 'long-body', 'text': 'Preserve the complete body.', 'existing_claims': None}
    before = deepcopy(original)
    with pytest.raises(BodyEncodingBudgetExceeded) as caught:
        model.encode_document(original, max_input_tokens=4096)
    assert caught.value.token_count == 4097
    assert caught.value.max_input_tokens == 4096
    assert caught.value.reason == 'body_encoding_token_budget_exceeded'
    assert original == before


def test_optional_body_budget_reuses_full_cached_vector():
    from threading import RLock
    from types import SimpleNamespace
    model = object.__new__(LivePostreadModel)
    model.lock = RLock()
    contract = SimpleNamespace(fingerprint=lambda: 'fixed')
    model.contract = model.long_contract = contract
    vector = np.zeros(1024, dtype=np.float32); vector[0] = 1
    metadata = {'token_count': 9000, 'truncated_tokens': 0}
    model.caches = {'fixed': SimpleNamespace(root=Path('cache'), get=lambda text: (vector, metadata))}
    model.encoder = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: pytest.fail('Cache hit must not tokenize')),
        encode_one=lambda text: pytest.fail('Cache hit must not encode'))
    result = model.encode_document({'document_id': 'cached-long', 'text': 'Full saved text.'}, max_input_tokens=4096)
    assert result['encoding']['cache_hit'] is True
    assert result['encoding']['token_count'] == 9000
    assert result['encoding']['truncated_tokens'] == 0
    assert np.array_equal(result['vector'], vector)


def test_optional_body_budget_rejects_before_loading_encoder_weights():
    from threading import RLock
    from types import SimpleNamespace
    from demo.live_postread_model import BodyEncodingBudgetExceeded
    model = object.__new__(LivePostreadModel)
    model.lock = RLock()
    contract = SimpleNamespace(fingerprint=lambda: 'fixed')
    model.contract = model.long_contract = contract
    model.caches = {'fixed': SimpleNamespace(get=lambda text: None)}
    model.encoder = None
    model._budget_tokenizer = SimpleNamespace(encode=lambda text, truncation: list(range(5000)))
    model.encoder_class = lambda *a, **k: pytest.fail('Over-budget body must not load encoder weights')
    with pytest.raises(BodyEncodingBudgetExceeded):
        model.encode_document({'document_id': 'too-long', 'text': 'Complete body.'}, max_input_tokens=4096)
    assert model.encoder is None


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    if not LivePostreadModel.readiness()['ready']:
        pytest.skip('Frozen local release/snapshot not installed; no download attempted')
    # This suite exercises the full-text head; workspace default may be state-only.
    return LivePostreadModel(cache_root=tmp_path_factory.mktemp('lpm'), head=DEFAULT_HEAD)


@pytest.fixture
def panel(model):
    case = next(iter(model.backend.cases.values()))
    fixture = {'physical_match_id': case['physical_match_id'], 'kickoff': '2025-01-01T17:30:00Z',
               'home_team': 'Brentford', 'away_team': 'Arsenal',
               'prediction_cutoff': '2025-01-01T16:30:00Z'}
    docs = []
    for item in case['documents']:
        doc = deepcopy(item)
        doc['text'] = (model.release_dir / doc['body_file']).read_bytes().decode('utf-8')
        docs.append(doc)
    vectors = {d['text_sha256']: model.backend.vectors[d['text_sha256']].copy() for d in docs}
    return fixture, case['prior'], docs, vectors


def test_readiness_has_no_weight_load_or_download(monkeypatch):
    monkeypatch.setattr('demo.live_postread_model._load_release', lambda *a: pytest.fail('Must not load weights'))
    result = LivePostreadModel.readiness()
    assert result['weights_loaded'] is False and result['network_requests'] == 0
    assert result['model_path'].endswith('97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')


def test_missing_snapshot_has_no_legacy_fallback(tmp_path):
    result = LivePostreadModel.readiness(model_path=tmp_path)
    assert not result['ready'] and result['missing_files']
    with pytest.raises(FileNotFoundError):
        LivePostreadModel(model_path=tmp_path, cache_root=tmp_path / 'cache')


def test_frozen_binding_and_initial_prediction(model, panel):
    fixture, prior, _, _ = panel
    result = model.initial_prediction(fixture, prior)
    assert result['f_model_sha256'] == F_SHA
    assert result['used_input_ids'] == [] and result['f_executed']
    assert sum(result['probabilities']) == pytest.approx(1., abs=1e-6)
    assert model.bindings['g_model_sha256'] == G_SHA
    assert model.bindings['encoder_native_dimension'] == 1024
    assert model.bindings['model_prefix_dimension'] == 256
    assert model.bindings['g_feature_dimension'] == 1140


def test_g_reads_pool_f_only_adopted_and_scoring_has_no_after(model, panel):
    fixture, prior, docs, vectors = panel
    result = model.evaluate(fixture, prior, docs, vectors, [], budget=3)
    assert result['g_readable_ids'] == [d['document_id'] for d in docs]
    assert result['f_used_ids'] == []
    assert len(result['candidate_scores']) == len(docs)
    assert result['new_after_prediction'] is None
    assert result['threshold'] == .01 and result['candidate_body_embedding_used']
    if result['selected_index'] is not None:
        selected = result['selected_index']
        updated = model.predict(fixture, prior, docs, vectors, [selected])
        assert updated['used_input_ids'] == [docs[selected]['document_id']]


def test_unadopted_future_pool_does_not_change_current_f(model, panel):
    fixture, prior, docs, vectors = panel
    before = model.predict(fixture, prior, docs, vectors, [0])
    reduced = model.predict(fixture, prior, docs[:1], vectors, [0])
    assert before['probabilities'] == reduced['probabilities']
    assert before['hidden_state'] == reduced['hidden_state']
    assert before['used_input_ids'] == reduced['used_input_ids']


def test_metadata_recomputes_actual_order_not_frozen_pool_order(model, panel):
    fixture, prior, docs, vectors = panel
    docs = deepcopy(docs[:2])
    claim = {'team_side': 'home', 'entity': 'Player A', 'kind': 'availability', 'value': 'unavailable', 'scope': 'this_match'}
    docs[0]['existing_claims'] = [claim]
    docs[1]['existing_claims'] = [claim]
    ev, mid, _, _ = model._evaluator(fixture, prior, docs, vectors, [1])
    meta = ev.panel.sequences[mid]['meta']
    repeat = model.metadata_names.index('repeated_claim_count')
    assert meta[1, repeat] == 0 and meta[0, repeat] == 1
    assert (meta[:, model.metadata_names.index('historical_reconstruction')] == 0).all()
    assert 'document_records' not in ev.panel.sequences[mid]
    from fm_defense_core.sequential_postread_g import causal_metadata
    assert np.array_equal(causal_metadata(ev.panel.sequences[mid], [1], 0), meta[0])
    ev2, mid2, _, _ = model._evaluator(fixture, prior, docs, vectors, [0, 1])
    meta2 = ev2.panel.sequences[mid2]['meta']
    assert meta2[0, repeat] == 0 and meta2[1, repeat] == 1


def test_empty_pool_and_budget_stop_do_not_create_candidate_after(model, panel):
    fixture, prior, docs, vectors = panel
    empty = model.evaluate(fixture, prior, [], [], [], budget=3)
    assert empty['stop_reason'] == 'candidate_pool_exhausted'
    assert empty['candidate_scores'] == [] and empty['selected_index'] is None
    exhausted = model.evaluate(fixture, prior, docs, vectors, [0], budget=1)
    assert exhausted['stop_reason'] == 'budget_exhausted'
    assert exhausted['new_after_prediction'] is None


@pytest.mark.parametrize('field', ['outcome_class', 'realized_gain', 'p_after', 'prediction_after', 'loss_after'])
def test_label_and_candidate_after_inputs_rejected(model, panel, field):
    fixture, prior, docs, vectors = panel
    polluted = deepcopy(docs)
    polluted[0][field] = 0
    with pytest.raises(ValueError, match='forbidden'):
        model.evaluate(fixture, prior, polluted, vectors, [], budget=3)


def test_body_sha_vector_shape_and_repeated_adoption_rejected(model, panel):
    fixture, prior, docs, vectors = panel
    bad = deepcopy(docs);bad[0]['text'] += 'Changed bytes'
    with pytest.raises(ValueError, match='Body SHA'):
        model.predict(fixture, prior, bad, vectors, [])
    badvectors = dict(vectors);badvectors[docs[0]['text_sha256']] = np.ones(1024, dtype=np.float32)
    with pytest.raises(ValueError, match='unit norm'):
        model.predict(fixture, prior, docs, badvectors, [])
    badvectors = dict(vectors);badvectors[docs[0]['text_sha256']] = vectors[docs[1]['text_sha256']]
    with pytest.raises(ValueError, match='frozen exact-body vector'):
        model.predict(fixture, prior, docs, badvectors, [])
    with pytest.raises(ValueError, match='adopted indices'):
        model.predict(fixture, prior, docs, vectors, [0, 0])


def test_cost_only_changes_net_score_not_gain_or_f(model, panel):
    fixture, prior, docs, vectors = panel
    a = model.evaluate(fixture, prior, docs, vectors, [], budget=20)
    b = model.evaluate(fixture, prior, docs, vectors, [], budget=20, cost_loss_units=10.)
    assert [r['score'] for r in a['candidate_scores']] == [r['score'] for r in b['candidate_scores']]
    assert a['prediction'] == b['prediction']
    assert b['selected_index'] is None
    assert all(r['net_gain'] == pytest.approx(r['score'] - 10.) for r in b['candidate_scores'])
    assert b['budget_outside_training_support']


def test_failed_f_raises_without_fabricated_prediction(model, panel, monkeypatch):
    fixture, prior, docs, vectors = panel
    def fail(*a, **kw):
        raise ValueError('injected genuine f failure')
    monkeypatch.setattr(model.f, 'predict_prefixes', fail)
    with pytest.raises(ValueError, match='genuine f failure'):
        model.predict(fixture, prior, docs, vectors, [0])


def test_literal_team_binding_keeps_uncertainty(model):
    fixture = {'home_team': 'Arsenal', 'away_team': 'Chelsea'}
    doc = {'text': 'ArsenalW is different. Arsenal trained. Chelsea trained.', 'source_owner_team': 'Arsenal'}
    b = model._binding(doc, fixture)
    assert b['exact_home_team_mentions'] == 1
    assert b['exact_away_team_mentions'] == 1
    assert b['target_aliases_partial'] and b['source_target_home']
    assert not b['source_target_unknown']


def test_team_mentions_match_frozen_training_overlap_and_whitespace_rule(model):
    fixture = {'home_team': 'Tottenham Hotspur', 'away_team': 'Manchester City',
               'home_aliases': ['Tottenham', 'Tottenham Hotspur'],
               'away_aliases': ['Manchester City', 'Manchester']}
    doc = {'text': 'Tottenham Hotspur faces Manchester\n  City. Tottenham trained.',
           'source_owner_team': 'A known third club'}
    b = model._binding(doc, fixture)
    assert b['exact_home_team_mentions'] == 2
    assert b['exact_away_team_mentions'] == 1
    assert not b['source_target_home'] and not b['source_target_away']
    assert not b['source_target_unknown']
    fixture['home_aliases'] = []
    assert model._binding(doc, fixture)['target_aliases_unknown']


def test_equivalent_timezones_preserve_metadata_prediction_and_scores(model, panel):
    fixture, prior, docs, vectors = panel
    utc_fixture = deepcopy(fixture)
    utc_fixture['kickoff'] = '2025-01-01T17:00:00Z'
    offset_fixture = deepcopy(utc_fixture)
    offset_fixture['kickoff'] = '2025-01-02T01:00:00+08:00'
    utc_docs = deepcopy(docs[:2]);offset_docs = deepcopy(utc_docs)
    for doc in utc_docs:
        doc['published_at'] = '2024-12-31T16:30:00Z'
    for doc in offset_docs:
        doc['published_at'] = '2025-01-01T00:30:00+08:00'
    ev1, mid1, _, _ = model._evaluator(utc_fixture, prior, utc_docs, vectors, [0])
    ev2, mid2, _, _ = model._evaluator(offset_fixture, prior, offset_docs, vectors, [0])
    a = ev1.panel.sequences[mid1]['meta'];b = ev2.panel.sequences[mid2]['meta']
    assert np.array_equal(a, b)
    assert a[0, model.metadata_names.index('log1p_days_before_match')] == pytest.approx(np.log(2))
    p1 = model.predict(utc_fixture, prior, utc_docs, vectors, [0])
    p2 = model.predict(offset_fixture, prior, offset_docs, vectors, [0])
    assert p1 == p2
    g1 = model.evaluate(utc_fixture, prior, utc_docs, vectors, [0], budget=2)
    g2 = model.evaluate(offset_fixture, prior, offset_docs, vectors, [0], budget=2)
    assert g1['candidate_scores'] == g2['candidate_scores']
    assert offset_docs[0]['published_at'] == '2025-01-01T00:30:00+08:00'
    assert offset_fixture['kickoff'] == '2025-01-02T01:00:00+08:00'


def test_date_only_publication_keeps_day_precision_without_inventing_hour(model):
    doc = {'published_at': '2025-01-01', 'body_characters': 10, 'existing_claims': None}
    meta = model._live_meta(doc, [], '2025-01-02')
    assert meta[model.metadata_names.index('log1p_days_before_match')] == pytest.approx(np.log(2))
    assert meta[model.metadata_names.index('publication_time_unknown')] == 0
    assert doc['published_at'] == '2025-01-01'


def test_real_local_qwen_body_encoding_and_frozen_inference(model, panel, monkeypatch):
    import socket
    def offline(*a, **kw):
        pytest.fail('No network allowed in local encoding smoke')
    monkeypatch.setattr(socket, 'create_connection', offline)
    monkeypatch.setattr(socket.socket, 'connect', offline)
    monkeypatch.setattr(socket, 'getaddrinfo', offline)
    fixture, prior, docs, _ = panel
    doc = min(docs, key=lambda d: len(d['text']))
    encoded = model.encode_document(doc)
    again = model.encode_document(doc)
    assert again['encoding']['cache_hit']
    assert np.array_equal(encoded['vector'], again['vector'])
    assert encoded['encoding']['truncated_tokens'] == 0
    assert encoded['encoding']['text_sha256'] == hashlib.sha256(doc['text'].encode('utf-8')).hexdigest()
    frozen_vector = model.backend.vectors[doc['text_sha256']]
    assert float(encoded['vector'] @ frozen_vector) > .99999
    assert float(np.max(np.abs(encoded['vector'] - frozen_vector))) < .001
    result = model.evaluate(fixture, prior, [encoded['document']], [encoded['vector']], [], budget=1)
    assert len(result['candidate_scores']) == 1
    after = model.predict(fixture, prior, [encoded['document']], [encoded['vector']], [0])
    assert after['used_input_ids'] == [doc['document_id']]
    assert after['f_model_sha256'] == F_SHA
