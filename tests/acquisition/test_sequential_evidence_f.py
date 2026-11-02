import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from football_research.acquisition.sequential_evidence_f import (
    FrozenF, SequenceEvidenceNetwork, SequenceFConfig, checked_prior,
    match_balanced_prefix_loss, save_frozen,
)


def make_model(architecture):
    torch.manual_seed(41)
    config = SequenceFConfig(architecture=architecture, embedding_dim=8, metadata_names=('days', 'unknown'), hidden_dim=12)
    network = SequenceEvidenceNetwork(config)
    # Nonzero research weights make prefix invariance a substantive causal test.
    with torch.no_grad():
        head = network.residual[-1] if architecture == 'set_residual_mlp' else network.residual
        head.weight.normal_(std=0.2)
        head.bias.normal_(std=0.1)
    return network


def data():
    rng = np.random.default_rng(19)
    return [.5, .3, .2], rng.normal(size=(4, 16)), rng.normal(size=(4, 2))


@pytest.mark.parametrize('architecture', ['set_residual_mlp', 'gru'])
def test_unread_future_vectors_cannot_change_prefix(architecture):
    f = FrozenF(make_model(architecture))
    prior, vectors, meta = data()
    all_prefixes = f.predict_prefixes(prior, vectors, meta, ['a', 'b', 'c', 'd'])
    isolated = f.predict_prefixes(prior, vectors[:2], meta[:2], ['a', 'b'])
    for i in range(3):
        np.testing.assert_allclose(all_prefixes[i]['probabilities'], isolated[i]['probabilities'], atol=1e-7, rtol=0)
    vectors[2:] *= -100
    meta[2:] += 500
    changed_future = f.predict_prefixes(prior, vectors, meta)
    np.testing.assert_allclose(changed_future[2]['probabilities'], isolated[2]['probabilities'], atol=1e-7, rtol=0)
    assert not np.allclose(all_prefixes[-1]['probabilities'], all_prefixes[0]['probabilities'])


@pytest.mark.parametrize('architecture', ['set_residual_mlp', 'gru'])
def test_predictions_never_update_parameters_or_buffers(architecture):
    network = make_model(architecture)
    before = {key: value.clone() for key, value in network.state_dict().items()}
    f = FrozenF(network)
    prior, vectors, meta = data()
    first = f.predict(prior, vectors, meta)
    second = f.predict(prior, vectors, meta)
    np.testing.assert_array_equal(first, second)
    for key, value in network.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not any(p.requires_grad for p in network.parameters())
    assert not network.training


def test_set_model_is_order_invariant_but_gru_tracks_order():
    prior, vectors, meta = data()
    set_f = FrozenF(make_model('set_residual_mlp'))
    np.testing.assert_allclose(set_f.predict(prior, vectors, meta), set_f.predict(prior, vectors[::-1].copy(), meta[::-1].copy()), atol=1e-7, rtol=0)
    gru_f = FrozenF(make_model('gru'))
    assert not np.allclose(gru_f.predict(prior, vectors, meta), gru_f.predict(prior, vectors[::-1].copy(), meta[::-1].copy()), atol=1e-5)


@pytest.mark.parametrize('architecture', ['set_residual_mlp', 'gru'])
def test_empty_prefix_is_same_calibrated_prior(architecture):
    f = FrozenF(make_model(architecture))
    p = f.predict([.5, .3, .2], [], [], [])
    np.testing.assert_allclose(p, [.5, .3, .2], atol=1e-7)
    trace = f.predict_prefixes(*data(), order=['a', 'b', 'c', 'd'])
    assert trace[0]['used_input_ids'] == []
    assert trace[2]['used_input_ids'] == ['a', 'b']
    assert trace[-1]['parameter_updated'] is False
    np.testing.assert_array_equal(f.hidden_state([.5, .3, .2], [], [], []), trace[0]['hidden_state'])


def test_each_match_has_equal_total_prefix_weight():
    logits = torch.tensor([[[1., 0., 0.], [3., 0., 0.], [-9., 9., 0.]], [[0., 2., 0.], [0., 3., 0.], [0., 4., 0.]]])
    labels = torch.tensor([0, 1])
    lengths = torch.tensor([1, 2])
    expected = (torch.nn.functional.cross_entropy(logits[0, :2], torch.tensor([0, 0])) + torch.nn.functional.cross_entropy(logits[1], torch.tensor([1, 1, 1]))) / 2
    torch.testing.assert_close(match_balanced_prefix_loss(logits, labels, lengths), expected)


@pytest.mark.parametrize('architecture', ['set_residual_mlp', 'gru'])
def test_checkpoint_roundtrip_and_integrity(tmp_path, architecture):
    network = make_model(architecture)
    manifest = save_frozen(network, tmp_path / 'f')
    f = FrozenF.load(tmp_path / 'f')
    prior, vectors, meta = data()
    np.testing.assert_array_equal(f.predict(prior, vectors, meta), FrozenF(network).predict(prior, vectors, meta))
    assert f.f_model_sha256 == manifest['f_model_sha256']
    assert len(f.hidden_state(prior, vectors[:1], meta[:1])) == 12
    checkpoint = tmp_path / 'f/model.pt'
    checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='binding'):
        FrozenF.load(tmp_path / 'f')


@pytest.mark.parametrize('prior', [[50, 30, 20], [.5, .3, .3], [.5, float('nan'), .5], [-.1, .5, .6]])
def test_reject_invalid_priors(prior):
    with pytest.raises(ValueError):
        checked_prior(prior)


def test_missing_metadata_duplicate_ids_and_zero_vectors_are_rejected():
    f = FrozenF(make_model('gru'))
    prior, vectors, meta = data()
    with pytest.raises(ValueError, match='metadata required'):
        f.predict(prior, vectors)
    with pytest.raises(ValueError, match='unique'):
        f.predict(prior, vectors, meta, ['a', 'a', 'b', 'c'])
    with pytest.raises(ValueError, match='zero'):
        f.predict(prior, np.zeros_like(vectors), meta)


def test_predict_many_matches_independent_prefix_calls():
    f = FrozenF(make_model('gru'))
    prior, vectors, meta = data()
    result = f.predict_many(prior, [vectors[:1], vectors[:3]], [meta[:1], meta[:3]], [['a'], ['a', 'b', 'c']])
    np.testing.assert_array_equal(result[0], f.predict(prior, vectors[:1], meta[:1]))
    np.testing.assert_array_equal(result[1], f.predict(prior, vectors[:3], meta[:3]))


def test_set_invariance_when_prefix_relations_are_recomputed_for_each_order():
    from football_research.acquisition.full_history_sequence_metadata import METADATA_NAMES, sequence_metadata
    documents = [
        {'text_sha256': 'aaa', 'published_at': '2022-01-01', 'body_characters': 300,
         'existing_claims': [{'team': 'home', 'entity': 'Player A', 'kind': 'availability', 'value': 'doubtful', 'scope': 'this_match'}]},
        {'text_sha256': 'bbb', 'published_at': '2022-01-02', 'body_characters': 500,
         'existing_claims': [{'team': 'home', 'entity': 'Player A', 'kind': 'availability', 'value': 'unavailable', 'scope': 'this_match'}]},
        {'text_sha256': 'ccc', 'published_at': '2022-01-03', 'body_characters': 800, 'existing_claims': None},
    ]
    original = np.asarray(sequence_metadata(documents, '2022-01-05'))
    reversed_meta = np.asarray(sequence_metadata(documents[::-1], '2022-01-05'))
    assert not np.array_equal(original[::-1], reversed_meta)
    torch.manual_seed(5)
    network = SequenceEvidenceNetwork(SequenceFConfig(architecture='set_residual_mlp', embedding_dim=8, metadata_names=tuple(METADATA_NAMES), hidden_dim=12))
    with torch.no_grad():
        network.residual[-1].weight.normal_(std=.3)
    frozen = FrozenF(network)
    vectors = np.random.default_rng(7).normal(size=(3, 16))
    np.testing.assert_allclose(frozen.predict([.4, .3, .3], vectors, original),
        frozen.predict([.4, .3, .3], vectors[::-1].copy(), reversed_meta), atol=1e-7, rtol=0)


def test_old_f_subset_requires_exact_body_and_match_binding(tmp_path):
    from football_research.acquisition.entity_semantic_f import EntitySemanticF, CHANNEL_FEATURES
    from football_research.acquisition.semantic_f import public_input, EXTRACTION_VERSION
    from football_research.acquisition.sequential_evidence_f import LegacyValidatedFSubset, file_sha256
    old = EntitySemanticF(np.zeros((4 * len(CHANNEL_FEATURES), 3)).tolist(), {'test_fixture_only': True})
    model_path = tmp_path / 'legacy.json'
    old.save(model_path)
    body = 'Player A is unavailable for this match.'
    public = public_input({'physical_match_id': 'original:1', 'home_name': 'Home', 'away_name': 'Away', 'kickoff': None},
                         {'url': 'https://example.test/news/one', 'title': 'Team news', 'text': body}, 'body')
    claims = [{'team': 'home', 'entity': 'Player A', 'kind': 'availability', 'value': 'unavailable',
               'scope': 'this_match', 'statement': 'Player A is unavailable for this match.', 'quotes': [body]}]
    f = LegacyValidatedFSubset(model_path, [{'input': public, 'response': {'schema': EXTRACTION_VERSION, 'input_id': public['input_id'], 'claims': claims}}], file_sha256(model_path))
    document = {'document_id': 'doc:one', 'text_sha256': hashlib.sha256(body.encode()).hexdigest()}
    known = f.predict_prefixes([.4, .3, .3], [document], 'canonical:1', aliases=['original:1'])
    assert known[-1]['status'] == 'evaluable'
    assert known[-1]['candidate_claim_count'] == 1
    other_match = f.predict_prefixes([.4, .3, .3], [document], 'different:2')
    assert other_match[-1]['probabilities'] is None
    missing = f.predict_prefixes([.4, .3, .3], [dict(document, document_id='missing', text_sha256='not_in_bank'), document], 'original:1')
    assert missing[1]['probabilities'] is None
    assert missing[2]['probabilities'] is None
    assert missing[2]['reason'] == 'earlier_prefix_extraction_missing_or_ambiguous'


def test_training_cli_uses_only_f_labels_and_defers_holdout_evaluation(tmp_path, monkeypatch):
    """Tiny fabricated loop fixture; no project model or scientific data is fitted."""
    path = Path(__file__).resolve().parents[2] / 'tools/train_full_history_sequence_f.py'
    spec = importlib.util.spec_from_file_location('sequence_f_test_cli', path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    directory = tmp_path / 'data'
    directory.mkdir()
    matches, outcomes, documents, sequences = [], [], [], []
    vectors = {}
    roles = ['f_train'] * 4 + ['f_dev'] * 2 + ['g_fit', 'test']
    rng = np.random.default_rng(3)
    for i, role in enumerate(roles):
        mid = f'test:fixture:{i}'
        date = '2020-01-01' if role == 'f_train' else '2022-02-01' if role == 'f_dev' else '2023-01-01' if role == 'g_fit' else '2026-01-01'
        split = 'f_fit' if role in ('f_train', 'f_dev') else role
        matches.append({'physical_match_id': mid, 'match_date': date, 'split_role': split, 'fit_role': role, 'prior': [.4, .3, .3]})
        # Invalid held-out labels prove training must not consume those values.
        outcomes.append({'physical_match_id': mid, 'outcome_class': i % 3 if role in ('f_train', 'f_dev') else 999})
        ids = []
        for j in range(2):
            digest = hashlib.sha256(f'{i}:{j}'.encode()).hexdigest()
            did = 'sha256:' + digest
            documents.append({'document_id': did, 'text_sha256': digest})
            vectors[digest] = rng.normal(size=16)
            ids.append(did)
        sequences.append({'physical_match_id': mid, 'document_ids': ids, 'document_meta': [[1., 0.], [2., 0.]], 'eligible': True})
    for name, rr in [('matches', matches), ('outcomes', outcomes), ('documents', documents), ('sequences', sequences)]:
        (directory / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rr), encoding='utf-8')
    (directory / 'schema.json').write_text(json.dumps({'document_meta_names': ['days', 'unknown']}), encoding='utf-8')
    (directory / 'dataset-summary.json').write_text(json.dumps({'f_dev_start_date': '2022-01-01'}), encoding='utf-8')
    vector_path = tmp_path / 'vectors.npz'
    np.savez(vector_path, **vectors)
    output = tmp_path / 'models'
    # Avoid global socket patch persistence after this local test.
    import socket
    monkeypatch.setattr(socket.socket, 'connect', socket.socket.connect)
    monkeypatch.setattr(socket, 'create_connection', socket.create_connection)
    monkeypatch.setattr(socket, 'getaddrinfo', socket.getaddrinfo)
    common = ['train_f', '--data', str(directory), '--output', str(output), '--threads', '1', '--prefix-dim', '8', '--hidden-dim', '8']
    monkeypatch.setattr('sys.argv', common + ['--baseline-only'])
    trainer.main()
    monkeypatch.setattr('sys.argv', common + ['--embeddings', str(vector_path), '--seeds', '11', '--architectures', 'gru', '--epochs', '1', '--patience', '1'])
    trainer.main()
    assert (output / 'selection-freeze.json').exists()
    assert not (output / 'evaluation').exists()
    comparison = json.loads((output / 'model-comparison.json').read_text())
    assert set(comparison[0]['metrics']) == {'f_train', 'f_dev'}
    saved = [json.loads(x) for x in (output / 'selected/prefix-predictions.jsonl').read_text().splitlines()]
    assert {r['split_role'] for r in saved} == {'f_train', 'f_dev'}
    monkeypatch.setattr('sys.argv', common + ['--embeddings', str(vector_path), '--evaluate-frozen'])
    with pytest.raises(ValueError, match='Invalid outcome class'):
        trainer.main()
    assert not (output / 'evaluation').exists()
