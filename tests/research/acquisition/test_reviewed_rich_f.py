from copy import deepcopy
import json
import numpy as np
import pytest
from football_research.acquisition import reviewed_rich_f as f


def row(mid='a', outcome=2):
    return {'physical_match_id': mid, 'outcome': outcome, 'prior': [.4, .3, .3],
            'evidence': [{'channel': 'body', 'team': 'home',
                          'atoms': ['predicate:availability:unavailable', 'scope:this_match'],
                          'text': 'player will not play due to injury'}]}


def test_empty_and_wholly_unknown_evidence_preserve_prior():
    r = row(); model = f.fit([r], 1.)
    empty = {**r, 'evidence': []}
    unknown = {**r, 'evidence': [{'channel': 'body', 'team': 'home',
                                'atoms': ['unknown:atom'], 'text': 'zzyyxx'}]}
    assert np.allclose(f.predict(model, empty), r['prior'])
    assert np.allclose(f.predict(model, unknown), r['prior'])
    assert f.predict(model, r)[2] > r['prior'][2]


def test_duplicate_evidence_does_not_amplify_prediction():
    r = row(); model = f.fit([r], 1.)
    more = deepcopy(r); more['evidence'] *= 8
    assert np.array_equal(f.features(model['encoder'], r), f.features(model['encoder'], more))


def test_vocabulary_is_fit_only_on_training_matches():
    r = row(); encoder = f.encoder_for([r])
    heldout = row('unseen'); heldout['evidence'][0]['text'] = 'uniquefuturetoken'
    before = deepcopy(encoder)
    f.features(encoder, heldout)
    assert encoder == before
    assert encoder['fit_match_ids'] == ['a']
    assert 'uniquefuturetoken' not in encoder['terms']


def test_reviewed_negation_scope_and_unmodeled_content_remain_explicit():
    claim = {'team': 'home', 'entity': 'Smith', 'kind': 'other', 'value': 'unmodeled',
             'scope': 'team_context', 'quotes': ['Smith trains but will not play for Club A.']}
    e = f.evidence_tokens(claim, 'body', {'home_name': 'Club A', 'away_name': 'Club B'}, 'training')
    assert 'smith' not in e['text'] and 'club a' not in e['text']
    assert 'will not play' in e['text']
    assert e['atoms'] == ['predicate:other:unmodeled', 'scope:team_context', 'concept:training']


def test_equal_match_weight_survives_whole_match_replication():
    a, b = row(), row('b', 0)
    b['evidence'][0]['team'] = 'away'
    model = f.fit([a, b], 1.)
    repeated = f.fit([a] * 8 + [b], 1.)
    assert model['encoder'] == repeated['encoder']
    assert np.allclose(model['coefficients'], repeated['coefficients'], atol=1e-8)


def test_saved_model_reload_and_tamper_rejection(tmp_path):
    r = row(); model = f.fit([r], 1.); path = tmp_path / 'model.json'
    f.save(model, path)
    assert np.array_equal(f.predict(model, r), f.predict(f.load(path), r))
    payload = json.loads(path.read_text()); payload['coefficients'][0][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='binding'):
        f.load(path)


def test_invalid_prior_and_conflicting_labels_rejected():
    a = row()
    with pytest.raises(ValueError, match='Consistent'):
        f.fit([a, row('a', 0)], 1.)
    for prior in ([4, 3, 3], [0., .5, .5], [float('nan'), .3, .3]):
        with pytest.raises(ValueError, match='prior'):
            f.fit([{**a, 'prior': prior}], 1.)


def test_null_regularization_is_explicit_zero_head():
    r = row(); model = f.fit([r], None)
    assert not np.asarray(model['coefficients']).any()
    assert np.allclose(f.predict(model, r), r['prior'])
