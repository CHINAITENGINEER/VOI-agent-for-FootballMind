"""Pre-read entity f query binding tests."""
import numpy as np

from football_research.acquisition.entity_semantic_f import EntityEvidenceBank, EntitySemanticF, REPRESENTATION_SHA256
from football_research.acquisition.entity_semantic_query import query_input, response_audit, validate_query
from football_research.acquisition.semantic_f import FEATURES, public_input
from football_research.acquisition.match_conditioned_f import bind_prior
from football_research.acquisition.historical_context_training import _state_for
from tests.research.acquisition.test_match_conditioned_f import state_and_snapshot
from tests.research.acquisition.test_semantic_f import article, record


def _fixture_setup():
    s, prior = state_and_snapshot()
    a = article('North forward Alex is ruled out of the South match.', title='Alex update')
    b = article('North defender Ben is ruled out of the South match.', title='Ben update')
    b['url'] = 'https://example.test/news/ben'
    # The imported article helper has an unrelated clock. Bind its publication
    # and capture to the constructed fixture's actual pre-kickoff decision.
    for d in (a, b):
        d['source_published_at'] = s['decision_at']
        d['captured_at'] = s['decision_at']
    records = []
    for d, entity in ((a, 'Alex'), (b, 'Ben')):
        records.append(record(s, d, 'title', empty=True))
        r = record(s, d, 'body'); r['response']['claims'][0]['entity'] = entity
        records.append(r)
    s['observed_links'] = [{'url': d['url'], 'title': d['title'],
                            'available_at': d['source_published_at'],
                            'title_basis': 'historical_headline_panel'} for d in (a, b)]
    bank = EntityEvidenceBank(records)
    # Vector order is title coarse/count, then novel body coarse/count.
    w = np.zeros((192, 3))
    body_count_index = 48 + 24 + FEATURES.index('home:availability:unavailable')
    w[body_count_index] = [-.2, .1, .1]
    model = EntitySemanticF(w.tolist(), {'representation_sha256': REPRESENTATION_SHA256})
    return s, prior, a, b, bank, model


def test_entity_query_exposes_counts_without_unread_body():
    s, prior, a, b, bank, model = _fixture_setup()
    action = {'action_id': 'read-b', 'tool': 'read_url', 'url': b['url'],
              'cost': {'money': 0., 'requests': 1}, 'cost_status': 'simulated'}
    query = query_input(s, action, model=model, prior_snapshot=prior, evidence_bank=bank)
    validate_query(query)
    assert query['current']['novel_body_entity_counts'][0] == 0
    assert b['text'] not in str(query)


def test_entity_query_audit_marks_vector_change():
    s, prior, a, b, bank, model = _fixture_setup()
    # Alex body is already read, so Ben becomes a second distinct entity.
    s['documents'] = [a]
    action = {'action_id': 'read-b', 'tool': 'read_url', 'url': b['url'],
              'cost': {'money': 0., 'requests': 1}, 'cost_status': 'simulated'}
    query = query_input(s, action, model=model, prior_snapshot=prior, evidence_bank=bank)
    after = dict(s); after['documents'] = [a, b]
    audit = response_audit(query, after, model=model, prior_snapshot=prior, evidence_bank=bank)
    assert audit['feature_vector_changed'] is True
    assert audit['max_probability_change'] > 0
    assert audit['interpretation'] == 'entity_f_forecast_changed_not_yet_a_gain'
