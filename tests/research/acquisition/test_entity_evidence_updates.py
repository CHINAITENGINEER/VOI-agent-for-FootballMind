from copy import deepcopy
import numpy as np
import pytest
from football_research.acquisition import entity_evidence_updates as f


def event(name, concept='possible_bench', predicate='other:unmodeled', scope='team_context',
          published='2025-08-14T12:00:00Z', entity='Joe Gomez', host='www.liverpoolfc.com'):
    return {'event_id': name, 'input_id': name, 'physical_match_id': 'match', 'entity': entity,
            'source_host': host, 'document_url': 'https://' + host + '/' + name,
            'source_published_at': published, 'evidence': {
                'channel': 'body', 'team': 'home', 'text': name,
                'atoms': [f'predicate:{predicate}', f'scope:{scope}'] + ([f'concept:{concept}'] if concept else [])}}


def row(*events):
    return {'physical_match_id': 'match', 'decision_at': '2025-08-15T18:59:00Z',
            'outcome': 0, 'prior': [.4, .3, .3], 'evidence': [e['evidence'] for e in events],
            'evidence_events': list(events)}


def bench(**kwargs):
    return event('later', concept='confirmed_bench', predicate='availability:available', scope='this_match',
                 published='2025-08-15T17:00:00Z', **kwargs)


def test_realistic_pending_to_confirmed_selection_keeps_full_audit():
    a, b = event('early'), bench(); r = row(a, b); original = deepcopy(r)
    resolved, audit = f.resolve(r)
    assert resolved['evidence'] == [b['evidence']]
    assert audit['events'][0]['superseded_by'] == ['later']
    assert r == original
    assert f.resolve(row(b, a))[0]['evidence'] == resolved['evidence']


def test_training_and_unavailable_are_independent_not_conflict():
    training = event('training', 'training_participation')
    absent = event('absent', 'confirmed_absence', 'availability:unavailable', 'this_match')
    resolved, audit = f.resolve(row(training, absent))
    assert len(resolved['evidence']) == 2 and not audit['conflicts']


@pytest.mark.parametrize('change', ['missing_time', 'same_time', 'different_host', 'same_document', 'different_entity'])
def test_ambiguous_or_unrelated_update_does_not_supersede(change):
    a, b = event('early'), bench()
    if change == 'missing_time': a['source_published_at'] = None
    if change == 'same_time': b['source_published_at'] = a['source_published_at']
    if change == 'different_host': b['source_host'] = 'other.example'
    if change == 'same_document': b['document_url'] = a['document_url']
    if change == 'different_entity': b['entity'] = 'Another Player'
    resolved, audit = f.resolve(row(a, b))
    assert len(resolved['evidence']) == 2
    assert not any(e['superseded_by'] for e in audit['events'])


def test_contradictory_confirmations_not_arbitrarily_resolved():
    a, b = event('early'), bench()
    c = event('third', 'confirmed_starter', 'availability:available', 'this_match', '2025-08-15T18:00:00Z')
    resolved, audit = f.resolve(row(a, b, c))
    assert len(resolved['evidence']) == 3 and len(audit['conflicts']) == 1


def test_doubtful_resolved_but_training_not_medical_clearance():
    a = event('early', None, 'availability:doubtful', 'this_match')
    b = event('later', None, 'availability:unavailable', 'this_match', '2025-08-15T17:00:00Z')
    assert len(f.resolve(row(a, b))[0]['evidence']) == 1
    b['evidence']['atoms'] = ['predicate:other:unmodeled', 'scope:team_context', 'concept:training_participation']
    assert len(f.resolve(row(a, b))[0]['evidence']) == 2


def test_binding_future_and_naive_time_rejected():
    e = event('e'); e['physical_match_id'] = 'other'
    with pytest.raises(ValueError, match='binding'): f.resolve(row(e))
    e = event('e', published='2025-08-16T00:00:00Z')
    with pytest.raises(ValueError, match='Future'): f.resolve(row(e))
    e = event('e', published='2025-08-14T00:00:00')
    with pytest.raises(ValueError, match='Timezone'): f.resolve(row(e))


def test_labels_never_affect_resolution_and_historical_scope_not_updated():
    r = row(event('early'), bench()); other = {**r, 'outcome': 2}
    assert f.resolve(r)[0]['evidence'] == f.resolve(other)[0]['evidence']
    r['evidence_events'][0]['evidence']['atoms'][1] = 'scope:historical'
    assert len(f.resolve(r)[0]['evidence']) == 2


def test_prediction_roundtrip_and_empty_input(tmp_path):
    r = row(event('early'), bench()); model = f.fit([r], 1.)
    path = tmp_path/'model.json'; f.save(model, path)
    assert np.array_equal(f.predict(model, r), f.predict(f.load(path), r))
    assert np.allclose(f.predict(model, row()), [.4, .3, .3])
