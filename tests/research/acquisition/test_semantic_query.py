"""Constructed protocol tests; no accuracy or empirical VoI claims."""
from copy import deepcopy
from datetime import timedelta
import json

import numpy as np
import pytest

from football_agent.query_value_selection import select_query_candidates
from football_agent.semantic_query_policy import SemanticQueryPolicy, TARGET
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.semantic_f import FEATURES, EvidenceBank, SemanticResidualF
from football_research.acquisition.semantic_query import (
    VERSION, query_input, response_audit, realized_target, predictor_support,
)
from tests.research.acquisition.test_match_conditioned_f import state_and_snapshot, doc, NOW
from tests.research.acquisition.test_semantic_f import record


@pytest.fixture
def setup():
    s, prior = state_and_snapshot()
    a = doc('North forward Alex is unavailable against South.')
    b = doc('North defender Ben is unavailable against South.',
            title='North defensive update', url='https://example.test/news/two')
    records = []
    for d, entity in ((a, 'Alex'), (b, 'Ben')):
        records.append(record(s, d, 'title', empty=True))
        r = record(s, d, 'body'); r['response']['claims'][0]['entity'] = entity
        records.append(r)
    s['observed_links'] = [{'url': d['url'], 'title': d['title'],
        'available_at': d['source_published_at'], 'title_basis': 'historical_headline_panel'} for d in (a, b)]
    w = np.zeros((4*len(FEATURES), 3))
    w[len(FEATURES)+FEATURES.index('home:availability:unavailable')] = [-.2, .1, .1]
    model = SemanticResidualF(w.tolist(), {'evidence_origin': 'constructed_mechanism'})
    return s, prior, [a, b], records, model


def action(d, aid='read-a'):
    return {'action_id': aid, 'tool': 'read_url', 'url': d['url'],
            'cost': {'money': 0., 'requests': 1}, 'cost_status': 'simulated'}


def prepare(setup, *, index=0):
    s, prior, docs, records, model = setup
    bank = EvidenceBank(records)
    q = query_input(s, action(docs[index]), model=model, prior_snapshot=prior, evidence_bank=bank)
    return q, bank


def test_prequery_inputs_do_not_depend_on_unread_body_or_labels(setup):
    s, prior, docs, records, model = setup
    q, _ = prepare(setup)
    # Remove every as-yet-unread body from the private bank; this must not even
    # change a public hash. Changing arbitrary hidden labels also cannot matter.
    title_bank = EvidenceBank([r for r in records if r['input']['channel'] == 'title'])
    noisy = {**s, 'outcome': 2, 'future_returns': [docs[0]], 'realized_gain': 999}
    noisy_action = {**action(docs[0]), 'body': docs[0]['text'], 'after': [.9, .05, .05]}
    assert query_input(noisy, noisy_action, model=model, prior_snapshot=prior, evidence_bank=title_bank) == q
    assert docs[0]['text'] not in json.dumps(q)
    assert 'semantic_bank_sha256' not in q
    assert q['candidate']['unread_body_novelty'] == 'UNKNOWN'


def test_different_player_same_feature_is_not_reported_as_duplicate_information(setup):
    s, prior, docs, records, model = setup
    s['documents'] = [docs[0]]
    q, bank = prepare(setup, index=1)
    audit = response_audit(q, {**s, 'documents': docs}, model=model, prior_snapshot=prior, evidence_bank=bank)
    assert audit['max_probability_change'] == 0
    assert audit['interpretation'] == 'new_predicates_collapsed_by_binary_features'
    assert audit['feature_collisions'][0]['claim']['entity'] == 'Ben'
    assert audit['information_value'] is None


def test_repeat_predicate_does_not_prove_article_is_redundant(setup):
    s, prior, docs, records, model = setup
    records[3]['response']['claims'][0]['entity'] = 'Alex'
    s['documents'] = [docs[0]]
    q, bank = prepare(setup, index=1)
    audit = response_audit(q, {**s, 'documents': docs}, model=model, prior_snapshot=prior, evidence_bank=bank)
    assert audit['max_probability_change'] == 0
    assert audit['interpretation'] == 'same_extracted_predicates_not_proof_of_duplicate_information'


def test_unmodeled_or_missing_claim_is_not_zero_information_value(setup):
    s, prior, docs, records, model = setup
    records[1]['response']['claims'][0]['scope'] = 'team_context'
    q, bank = prepare(setup)
    a = response_audit(q, {**s, 'documents': [docs[0]]}, model=model, prior_snapshot=prior, evidence_bank=bank)
    assert a['interpretation'] == 'assertions_outside_frozen_feature_support'
    assert a['prediction_before'] == a['prediction_after']
    records[1]['response']['claims'] = []
    q, bank = prepare(setup)
    a = response_audit(q, {**s, 'documents': [docs[0]]}, model=model, prior_snapshot=prior, evidence_bank=bank)
    assert a['interpretation'] == 'no_assertions_extracted_not_proof_of_irrelevance'


def test_actual_hindsight_target_is_separate_includes_negative_and_zero(setup):
    s, prior, docs, records, model = setup
    q, bank = prepare(setup)
    audit = response_audit(q, {**s, 'documents': [docs[0]]}, model=model, prior_snapshot=prior, evidence_bank=bank)
    stamp = (NOW+timedelta(days=10, hours=3)).isoformat()
    labels = [realized_target(q, audit, outcome=y, result_available_at=stamp,
                             label_provenance={'source': 'constructed'}) for y in (0, 2)]
    assert labels[0]['realized_gain_nats'] < 0 < labels[1]['realized_gain_nats']
    assert labels[0]['realized_prediction_harm'] is True
    assert all(r['empirical_expected_gain'] is None for r in labels)
    assert 'outcome' not in q and 'outcome' not in audit
    empty = response_audit(q, s, model=model, prior_snapshot=prior, evidence_bank=bank)
    zero = realized_target(q, empty, outcome=0, result_available_at=stamp, label_provenance={'source': 'constructed'})
    assert zero['realized_gain_nats'] == 0 and zero['zero_targets_retained']
    with pytest.raises(ValueError, match='after-match'):
        realized_target(q, audit, outcome=0, result_available_at=s['decision_at'], label_provenance={'source': 'constructed'})


def test_changed_model_request_or_evidence_is_rejected(setup):
    s, prior, docs, records, model = setup
    q, bank = prepare(setup)
    with pytest.raises(ValueError, match='different request'):
        response_audit(q, {**s, 'documents': [docs[1]]}, model=model, prior_snapshot=prior, evidence_bank=bank)
    different = SemanticResidualF(np.zeros((4*len(FEATURES), 3)).tolist(), model.metadata)
    with pytest.raises(ValueError, match='same frozen f'):
        response_audit(q, s, model=different, prior_snapshot=prior, evidence_bank=bank)
    q['current']['probabilities'] = [.1, .1, .8]
    with pytest.raises(ValueError, match='modified'):
        response_audit(q, s, model=model, prior_snapshot=prior, evidence_bank=bank)


def test_unobserved_candidate_title_not_inferred_from_future_bank(setup):
    s, prior, docs, records, model = setup
    s['observed_links'] = []
    with pytest.raises(ValueError, match='previously observed'):
        prepare(setup)


def test_known_extraction_change_cannot_be_charged_to_the_new_article(setup):
    s, prior, docs, records, model = setup
    s['documents'] = [docs[0]]
    q, _ = prepare(setup, index=1)
    altered = deepcopy(records)
    altered[1]['response']['claims'][0]['value'] = 'available'
    with pytest.raises(ValueError, match='Previously visible semantic'):
        response_audit(q, {**s, 'documents': docs}, model=model, prior_snapshot=prior,
                       evidence_bank=EvidenceBank(altered))


def test_even_rehashed_nested_hindsight_field_is_rejected(setup):
    q, _ = prepare(setup)
    q['candidate']['future_body_gain'] = .9
    q['query_sha256'] = _digest({k: v for k, v in q.items() if k != 'query_sha256'})
    p = SemanticQueryPolicy(predictor_payload_sha256=q['predictor_payload_sha256'])
    with pytest.raises(ValueError, match='nested query fields'):
        p.plan([q], {'limits': {'money': 0, 'requests': 1}, 'spent': {}})


class FakeG:
    """Constructed selection fixture, not a trained or deployed model."""
    model_version = 'constructed-query-model'
    def __init__(self, digest, gain=-.1):
        self.metadata = {'input_schema': VERSION, 'predictor_payload_sha256': digest,
            'target': TARGET, 'training_physical_match_ids': ['B'],
            'predictor_training_physical_match_ids': ['A'], 'evidence_origin': 'constructed_mechanism'}
        self.calls, self.gain = [], gain

    def predict(self, query):
        self.calls.append(deepcopy(query))
        return {'support_status': 'supported', 'expected_signed_gain': self.gain, 'reason': 'constructed_test'}


def test_inert_f_never_becomes_a_numeric_stop_even_when_g_would_say_zero(setup):
    s, prior, docs, records, _ = setup
    # Nonzero but class-common coefficients also cannot change softmax.
    model = SemanticResidualF(np.ones((4*len(FEATURES), 3)).tolist(), {'constructed': True})
    q = query_input(s, action(docs[0]), model=model, prior_snapshot=prior, evidence_bank=EvidenceBank(records))
    g = FakeG(_digest(model.payload()), gain=0.)
    policy = SemanticQueryPolicy(predictor_payload_sha256=_digest(model.payload()), value_model=g)
    plan = policy.plan([q], {'limits': {'money': 0, 'requests': 3}, 'spent': {}}, candidate_generation_complete=True)
    assert not g.calls
    assert plan['selection']['decision'] == 'defer'
    assert plan['candidates'][0]['value']['expected_signed_gain'] is None
    assert plan['candidates'][0]['reason'] == 'predictor_cannot_resolve_query_value'


def test_title_sensitive_f_is_not_blocked_by_body_blindness(setup):
    s, prior, docs, records, _ = setup
    w = np.zeros((4*len(FEATURES), 3))
    w[FEATURES.index('home:availability:unavailable')] = [-.2, .1, .1]
    model = SemanticResidualF(w.tolist(), {'constructed': True})
    q = query_input(s, action(docs[0]), model=model, prior_snapshot=prior, evidence_bank=EvidenceBank(records))
    assert q['current']['predictor_support']['body_response_status'] == 'insensitive'
    assert q['current']['predictor_support']['text_response_status'] == 'responsive'
    g = FakeG(q['predictor_payload_sha256'], gain=.1)
    policy = SemanticQueryPolicy(predictor_payload_sha256=q['predictor_payload_sha256'], value_model=g)
    plan = policy.plan([q], {'limits': {'money': 0, 'requests': 1}, 'spent': {}})
    assert plan['selection']['decision'] == 'acquire' and g.calls == [q]


def test_compatible_g_can_select_positive_or_stop_at_supported_menu(setup):
    q, _ = prepare(setup)
    g = FakeG(q['predictor_payload_sha256'], gain=.1)
    p = SemanticQueryPolicy(predictor_payload_sha256=q['predictor_payload_sha256'],
                            value_model=g, lambda_cost={'requests': .01})
    budget = {'limits': {'money': 0, 'requests': 3}, 'spent': {}}
    plan = p.plan([q], budget)
    assert plan['selection']['decision'] == 'acquire'
    assert plan['candidates'][0]['value']['net_voi'] == pytest.approx(.09)
    assert g.calls == [q]
    g.gain = -.1
    assert p.plan([q], budget)['selection']['decision'] == 'defer'
    assert p.plan([q], budget, candidate_generation_complete=True)['selection']['decision'] == 'stop'


def test_unknown_candidate_prevents_nonpositive_stop_and_budget_still_halts(setup):
    q, _ = prepare(setup)
    p = SemanticQueryPolicy(predictor_payload_sha256=q['predictor_payload_sha256'])
    assert p.plan([q], {'limits': {'money': 0, 'requests': 0}, 'spent': {}})['selection']['reason'] == 'physical_budget_exhausted'
    rows = [{'action_id': 'known', 'feasible': True, 'value': {'status': 'COMPUTED', 'net_voi': -.1}},
            {'action_id': 'unknown', 'feasible': True, 'value': {'status': 'INELIGIBLE', 'net_voi': None}}]
    result = select_query_candidates(rows, candidate_generation_complete=True)
    assert result['decision'] == 'defer' and result['stop_certificate'] is None


def test_incompatible_g_and_mixed_state_menu_rejected(setup):
    q, _ = prepare(setup)
    with pytest.raises(ValueError, match='exact f'):
        SemanticQueryPolicy(predictor_payload_sha256=q['predictor_payload_sha256'], value_model=FakeG('another-f'))
    another = deepcopy(q)
    another['candidate']['action_id'] = 'second'
    another['current']['history'].append({'status': 'FAILED'})
    another['query_sha256'] = _digest({k: v for k, v in another.items() if k != 'query_sha256'})
    p = SemanticQueryPolicy(predictor_payload_sha256=q['predictor_payload_sha256'])
    with pytest.raises(ValueError, match='same actual before'):
        p.plan([q, another], {'limits': {'money': 0, 'requests': 3}, 'spent': {}})
