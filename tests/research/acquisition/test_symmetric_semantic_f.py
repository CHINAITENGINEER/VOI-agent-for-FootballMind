from copy import deepcopy
import json

import numpy as np
import pytest

from football_research.acquisition import symmetric_semantic_f as f


def row(mid='m', outcome=0, predicate='availability:unavailable', concept=None):
    atoms = ['predicate:'+predicate, 'scope:this_match']
    if concept:
        atoms.append('concept:'+concept)
    event = {'event_id': 'e', 'input_id': 'i', 'physical_match_id': mid,
             'entity': 'Player', 'source_host': 'club.example', 'document_url': 'https://club.example/a',
             'source_published_at': '2025-08-14T12:00:00Z',
             'evidence': {'channel': 'body', 'team': 'home', 'text': 'Source quotation', 'atoms': atoms}}
    return {'physical_match_id': mid, 'outcome': outcome, 'prior': [.4, .3, .3],
            'decision_at': '2025-08-15T12:00:00Z', 'used_input_ids': ['i'], 'evidence_events': [event]}


def test_fit_predict_swap_empty_and_serialization(tmp_path):
    r = row()
    model = f.fit([r, row('n', 2, 'availability:available')])
    p = f.predict(model, r)
    assert np.all(p > 0) and np.isclose(p.sum(), 1)
    assert not np.allclose(p, r['prior'])
    swapped = deepcopy(r)
    swapped['prior'] = r['prior'][::-1]
    swapped['evidence_events'][0]['evidence']['team'] = 'away'
    assert np.allclose(f.predict(model, swapped), p[::-1], atol=1e-14)
    empty = {**r, 'used_input_ids': [], 'evidence_events': []}
    assert np.array_equal(f.predict(model, empty), r['prior'])
    file = tmp_path/'model.json'
    f.save(model, file)
    assert np.array_equal(p, f.predict(f.load(file), r))
    payload = json.loads(file.read_text())
    payload['coefficients'][0][0] += 1
    file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='digest'):
        f.load(file)


def test_same_fact_duplicate_and_labels_do_not_change_features():
    r = row()
    duplicate = deepcopy(r['evidence_events'][0])
    duplicate.update(event_id='e2', input_id='i2', document_url='https://club.example/b')
    duplicated = deepcopy(r)
    duplicated['evidence_events'].append(duplicate)
    duplicated['used_input_ids'].append('i2')
    duplicated.update(outcome=2, prediction_after=[1, 0, 0], signed_gain=100)
    assert all(np.array_equal(x, y) for x, y in zip(f.public_features(r), f.public_features(duplicated)))


def test_training_participation_is_not_medical_clearance():
    r = row(predicate='other:unmodeled', concept='training_participation')
    _, d, _ = f.public_features(r)
    assert d[f.FEATURES.index('direct:availability:available')] == 0
    assert d[f.FEATURES.index('context:training_participation:this_match')] == 1


def test_unsupported_evidence_keeps_prior():
    model = f.fit([row()])
    r = row(predicate='other:unmodeled', concept='ticket_sales')
    assert np.array_equal(f.predict(model, r), r['prior'])


@pytest.mark.parametrize('change', ['future', 'unseen', 'bad_prior', 'wrong_match'])
def test_input_guards(change):
    r = row()
    if change == 'future':
        r['evidence_events'][0]['source_published_at'] = '2025-09-01T00:00:00Z'
    elif change == 'unseen':
        r['used_input_ids'] = []
    elif change == 'bad_prior':
        r['prior'] = [40, 30, 30]
    else:
        r['evidence_events'][0]['physical_match_id'] = 'other'
    with pytest.raises(ValueError):
        f.public_features(r)


def test_equal_match_weighting_and_inconsistent_labels():
    a, b = row(), row('n', 2, 'availability:available')
    x, y = f.fit([a, b]), f.fit([a]*7+[b])
    assert np.allclose(x['coefficients'], y['coefficients'], atol=1e-12)
    with pytest.raises(ValueError, match='outcome'):
        f.fit([a, {**a, 'outcome': 2}])
