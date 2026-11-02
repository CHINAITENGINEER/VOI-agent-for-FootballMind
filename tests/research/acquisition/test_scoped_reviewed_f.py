from copy import deepcopy
import json
import numpy as np
import pytest
from football_research.acquisition import scoped_reviewed_f as f
from football_research.acquisition import reviewed_rich_f as rich


def evidence(predicate='availability:unavailable', scope='this_match', concept=None, text='will not play'):
    return {'channel': 'body', 'team': 'home', 'text': text,
            'atoms': [f'predicate:{predicate}', f'scope:{scope}'] + ([f'concept:{concept}'] if concept else [])}


def row(*es):
    return {'physical_match_id': 'm', 'outcome': 2, 'prior': [.4, .3, .3], 'evidence': list(es)}


def test_irrelevant_text_cannot_move_or_dilute_supported_prediction():
    direct = evidence(); r = row(direct); fitted = f.fit([r], 1.)
    ticket = evidence('other:unmodeled', 'team_context', 'ticket_sales', 'will not play ticket sales')
    assert np.array_equal(f.predict(fitted, r), f.predict(fitted, row(ticket, direct)))
    assert np.allclose(f.predict(fitted, row(ticket)), r['prior'])
    _, audit = f.route(row(ticket))
    assert audit[0]['lane'] == 'unsupported' and audit[0]['information_value'] is None


def test_training_context_stays_separate_from_confirmed_absence():
    training = evidence('other:unmodeled', 'team_context', 'training_participation', 'training but cannot play')
    routed, audit = f.route(row(training, evidence()))
    assert {a['lane'] for a in audit} == {'context', 'direct'}
    context = next(e for e in routed['evidence'] if 'lane:context' in e['atoms'])
    assert context['text'] == ''
    assert all(not a.startswith('predicate:availability:') for a in context['atoms'])
    assert len(f.route(row(training), 'direct_only')[0]['evidence']) == 0


def test_selection_is_not_medical_clearance_and_roles_remain_distinct():
    rows = [f.route(row(evidence('availability:available', concept=c)))[0]
            for c in ('confirmed_starter', 'confirmed_bench', 'confirmed_squad')]
    encoder = rich.encoder_for(rows)
    vectors = [rich.features(encoder, r) for r in rows]
    assert all('predicate:availability:available' not in r['evidence'][0]['atoms'] for r in rows)
    assert all(not np.array_equal(vectors[i], vectors[j]) for i in range(3) for j in range(i))


def test_historical_unresolved_unknown_team_not_transferred_to_match():
    es = [evidence(scope='historical'), evidence(scope='unresolved'), evidence()]
    es[-1]['team'] = 'unknown'
    routed, audit = f.route(row(*es))
    assert not routed['evidence']
    assert all(a['information_value'] is None for a in audit)


def test_exact_title_body_repeat_counts_once():
    body = evidence(); title = deepcopy(body); title['channel'] = 'title'
    a = f.route(row(body))[0]; b = f.route(row(body, title))[0]
    assert a['evidence'] == b['evidence']


def test_route_does_not_use_label_or_change_original():
    original = row(evidence()); before = deepcopy(original)
    changed = {**original, 'outcome': 0}
    assert f.route(original)[0]['evidence'] == f.route(changed)[0]['evidence']
    assert original == before


def test_ambiguous_annotations_and_unknown_mode_rejected():
    e = evidence(); e['atoms'].append('scope:historical')
    with pytest.raises(ValueError, match='Ambiguous'):
        f.route(row(e))
    with pytest.raises(ValueError, match='mode'):
        f.route(row(), 'unregistered')


def test_roundtrip_and_rule_tampering_rejected(tmp_path):
    r = row(evidence()); fitted = f.fit([r], 1.); path = tmp_path / 'model.json'
    f.save(fitted, path)
    assert np.array_equal(f.predict(f.load(path), r), f.predict(fitted, r))
    changed = deepcopy(fitted); changed['rules']['lexical_context_enabled'] = True
    with pytest.raises(ValueError, match='contract'):
        f.predict(changed, r)
    payload = json.loads(path.read_text(encoding='utf-8')); payload['mode'] = 'direct_only'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='binding'):
        f.load(path)
