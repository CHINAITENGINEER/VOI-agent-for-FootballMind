"""Entity-aware semantic representation tests; no accuracy claim."""
from copy import deepcopy
import numpy as np

from football_research.acquisition.entity_semantic_f import (
    EntityEvidenceBank, EntitySemanticF, PARAMETERS, REPRESENTATION_SHA256,
    vector,
)
from football_research.acquisition.semantic_f import FEATURES, public_input
from tests.research.acquisition.test_semantic_f import state, article, record


def _bank_and_state():
    s = state()
    a = article('North forward Alex is ruled out of the South match.', title='Alex update')
    b = article('North defender Ben is ruled out of the South match.', title='Ben update')
    b['url'] = 'https://example.test/news/ben'
    rows = []
    for d, entity in ((a, 'Alex'), (b, 'Ben')):
        rows.append(record(s, d, 'title', empty=True))
        r = record(s, d, 'body'); r['response']['claims'][0]['entity'] = entity
        rows.append(r)
    s['observed_links'] = [{'url': a['url'], 'title': a['title'], 'title_basis': 'historical_headline_panel',
                            'available_at': a['source_published_at']},
                           {'url': b['url'], 'title': b['title'], 'title_basis': 'historical_headline_panel',
                            'available_at': b['source_published_at']}]
    return s, a, b, rows


def test_distinct_entities_are_counted_and_repeated_entity_is_deduplicated():
    s, a, b, rows = _bank_and_state()
    bank = EntityEvidenceBank(rows)
    s['documents'] = [a]
    one = bank.project(s)
    s['documents'] = [a, b]
    two = bank.project(s)
    idx = FEATURES.index('home:availability:unavailable')
    assert one['novel_body_entity_counts'][idx] == 1
    assert two['novel_body_entity_counts'][idx] == 2
    assert two['novel_body'][idx] == 1
    assert vector(two).tolist() != vector(one).tolist()
    s['documents'] = [a, b, deepcopy(a)]
    # _project de-duplicates the same source document before extraction.
    assert bank.project(s)['novel_body_entity_counts'][idx] == 2


def test_same_entity_status_conflict_is_retained_without_last_write_wins():
    s, a, b, rows = _bank_and_state()
    c = article('North forward Alex is doubtful for the South match.', title='Alex late test')
    c['url'] = 'https://example.test/news/alex-late'
    conflict = record(s, c, 'body', value='doubtful')
    conflict['response']['claims'][0]['entity'] = 'Alex'
    rows.append(record(s, c, 'title', empty=True)); rows.append(conflict)
    s['observed_links'].append({'url': c['url'], 'title': c['title'],
                                'title_basis': 'historical_headline_panel', 'available_at': c['source_published_at']})
    s['documents'] = [a, c]
    projected = EntityEvidenceBank(rows).project(s)
    statuses = next(x for x in projected['multiple_status_entities'] if x['entity'] == 'alex')
    assert statuses['statuses'] == ['doubtful', 'unavailable']


def test_model_roundtrip_binds_representation(tmp_path):
    s, a, b, rows = _bank_and_state()
    bank = EntityEvidenceBank(rows)
    w = np.zeros((4*48, 3))
    model = EntitySemanticF(w.tolist(), {'representation_sha256': REPRESENTATION_SHA256})
    path = tmp_path/'model.json'; model.save(path)
    restored = EntitySemanticF.load(path)
    assert restored.model_version == model.model_version
    assert restored.payload() == model.payload()
