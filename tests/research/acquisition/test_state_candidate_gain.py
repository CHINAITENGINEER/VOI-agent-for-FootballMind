from copy import deepcopy
import numpy as np
import pytest
from football_research.acquisition import state_candidate_gain as g
from football_research.acquisition import historical_gain_head as old
from tools.train_state_candidate_g import before_snapshot


def public(title='Joe Gomez injury update'):
    return {'schema': old.QUERY_SCHEMA, 'probabilities': [.4,.3,.3], 'acquired_documents': 1,
            'visible_candidates': 2, 'article_age_hours': 1., 'title': title}


def snapshot():
    return {'physical_match_id': 'm', 'used_input_ids': ['observed'], 'claims': [
        {'input_id': 'observed', 'channel': 'body', 'entity': 'Joe Gomez', 'team': 'home',
         'facet': 'availability', 'value': 'doubtful', 'pending': True, 'text': 'Joe Gomez is doubtful.'}]}


def test_pending_entity_candidate_relation_without_unread_body():
    s = snapshot(); q = g.build_query(public(), s)
    assert q['relations']['pending_entity_title_overlap'] == 1
    assert q['relations']['pending_x_injury'] > 0
    assert g.build_query(public('Joe Gomezon tickets'), s)['relations']['entity_title_overlap'] == 0
    assert g.build_query(public('ticket sales'), s)['relations']['pending_x_injury'] == 0


@pytest.mark.parametrize('field', ['body', 'prediction_after', 'outcome', 'signed_gain'])
def test_hidden_fields_rejected_in_public_or_snapshot(field):
    q = public(); q[field] = 'hidden'
    with pytest.raises(ValueError): g.build_query(q, snapshot())
    s = snapshot(); s[field] = 'hidden'
    with pytest.raises(ValueError): g.build_query(public(), s)


def test_claim_from_unobserved_input_rejected_and_no_mutation():
    s = snapshot(); original = deepcopy(s); g.build_query(public(), s)
    assert s == original
    s['claims'][0]['input_id'] = 'unread'
    with pytest.raises(ValueError, match='already observed'): g.build_query(public(), s)


def test_training_participation_does_not_clear_medical_pending():
    s = snapshot(); c = deepcopy(s['claims'][0])
    c.update(facet='context:training_participation', value='training_participation', pending=False)
    s['claims'].append(c)
    r = g.build_query(public(), s)['relations']
    assert r['pending_entities'] > 0 and r['conflicting_entity_facets'] == 0


def test_fit_equal_match_weight_roundtrip_negative_zero(tmp_path):
    rows = [{'physical_match_id': str(i), 'query': g.build_query(public(title), snapshot()), 'signed_gain': gain}
            for i, (title, gain) in enumerate([('injury update', .03), ('ticket sales', -.02), ('team news', 0.)])]
    a = g.fit(rows); b = g.fit([rows[0]]*9+rows[1:])
    assert np.allclose(a['coefficients'], b['coefficients'])
    path = tmp_path/'model.json'; g.save(a, path)
    assert g.estimate(a, rows[0]['query']) == g.estimate(g.load(path), rows[0]['query'])
    for r in rows: r['signed_gain'] = 0.
    assert abs(g.estimate(g.fit(rows), rows[0]['query'])) < 1e-12


def test_snapshot_projection_ignores_labels_and_after():
    row = {'physical_match_id': 'm', 'decision_at': '2026-01-01T00:00:00Z',
           'used_input_ids': [], 'evidence_events': [], 'outcome': 0}
    before = before_snapshot(row)
    row.update(outcome=2, signed_gain=100, prediction_after=[1,0,0], body='unread')
    assert before_snapshot(row) == before
