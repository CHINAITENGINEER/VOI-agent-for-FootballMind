"""Mechanism tests on constructed inputs, never claims about match accuracy."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest
from jsonschema import ValidationError

from football_research.acquisition.semantic_f import (
    FEATURES, EXTRACTION_VERSION, EvidenceBank, SemanticResidualF, SourceAssertionProtocol,
    public_input, validate_response, fit_semantic_f, fit_diagnostic_candidate, vector, _fit, _predict,
)

TIME = datetime(2025, 10, 1, tzinfo=timezone.utc)


def state():
    return {'fixture': {'physical_match_id': 'test:match', 'match_id': 'test:match',
        'home_team_id': 'North', 'away_team_id': 'South', 'home_name': 'North', 'away_name': 'South',
        'competition_id': 'test:league', 'competition': 'Test League',
        'kickoff': (TIME+timedelta(days=1)).isoformat()},
        'decision_at': TIME.isoformat(), 'prediction_cutoff': TIME.isoformat(),
        'information_mode': 'historical_reconstruction', 'prior_probabilities': [.4, .3, .3],
        'prior_basis': 'constructed', 'documents': [], 'observed_links': [], 'history': []}


def article(text='North forward Alex is ruled out of the South match.', title='North squad update'):
    return {'url': 'https://example.test/news/update', 'title': title, 'text': text,
        'source_published_at': (TIME-timedelta(hours=2)).isoformat(),
        'captured_at': (TIME+timedelta(days=3)).isoformat()}


def record(s, doc, channel, *, team='home', value='unavailable', scope='this_match', empty=False):
    public = public_input(s['fixture'], doc, channel)
    return {'input': public, 'response': {'schema': EXTRACTION_VERSION, 'input_id': public['input_id'],
        'claims': [] if empty else [{'team': team, 'entity': 'Alex', 'kind': 'availability', 'value': value,
            'scope': scope, 'statement': 'The source states the player status.', 'quotes': [public['text']]}]},
        'provenance': {'kind': 'constructed_test'}}


def test_outcomes_and_predictions_never_enter_extraction_input():
    s, doc = state(), article()
    polluted_fixture = {**s['fixture'], 'outcome': 2, 'probabilities': [.01, .01, .98]}
    polluted_doc = {**doc, 'gain': 999, 'after_prediction': [1, 0, 0]}
    assert public_input(polluted_fixture, polluted_doc, 'body') == public_input(s['fixture'], doc, 'body')
    title = public_input(s['fixture'], doc, 'title')
    assert title['text'] == doc['title']
    assert doc['text'] not in json.dumps(title)


def test_exact_quotes_do_not_allow_unread_body_or_response_probabilities():
    s, doc = state(), article()
    r = record(s, doc, 'title')
    r['response']['claims'][0]['quotes'] = [doc['text']]
    with pytest.raises(ValueError, match='exact acquired text'):
        validate_response(r['input'], r['response'])
    r = record(s, doc, 'body')
    r['response']['predicted_gain'] = .1
    with pytest.raises(ValidationError):
        validate_response(r['input'], r['response'])
    r = record(s, doc, 'body'); r['input']['probabilities'] = [.3, .3, .4]
    with pytest.raises(ValueError, match='undeclared'):
        validate_response(r['input'], r['response'])


def test_unrevealed_body_cannot_change_forecast_and_missing_extraction_is_not_zero():
    s, doc = state(), article()
    bank = EvidenceBank([record(s, doc, 'title', empty=True), record(s, doc, 'body')])
    before = bank.project(s)
    s['observed_links'] = [{'url': doc['url'], 'title': doc['title'],
                            'title_basis': 'historical_headline_panel',
                            'available_at': (TIME-timedelta(hours=1)).isoformat()}]
    discovered = bank.project(s)
    assert vector(discovered).tolist() == vector(before).tolist()
    assert len(discovered['used_input_ids']) == 1
    s['documents'] = [doc]
    after = bank.project(s)
    assert sum(after['novel_body']) == 1
    assert after['novel_body'][FEATURES.index('home:availability:unavailable')] == 1
    with pytest.raises(ValueError, match='SEMANTIC_EXTRACTION_MISSING'):
        EvidenceBank([record(s, doc, 'title', empty=True)]).project(s)


def test_known_title_fact_is_not_charged_again_as_new_body_information():
    s = state(); doc = article(title='North Alex is ruled out against South')
    bank = EvidenceBank([record(s, doc, 'title'), record(s, doc, 'body')])
    s['observed_links'] = [{'url': doc['url'], 'title': doc['title'],
                            'title_basis': 'historical_headline_panel', 'available_at': TIME.isoformat()}]
    before = bank.project(s)
    s['documents'] = [doc]
    after = bank.project(s)
    assert sum(before['title']) == 1
    assert sum(after['novel_body']) == 0
    assert vector(before).tolist() == vector(after).tolist()


def test_duplicate_or_paraphrased_assertions_do_not_amplify_features():
    s, doc = state(), article()
    duplicate = {**doc, 'url': 'https://example.test/news/another',
                 'text': 'North have confirmed Alex cannot play against South.'}
    bank = EvidenceBank([record(s, d, ch, empty=(ch == 'title'))
                         for d in (doc, duplicate) for ch in ('title', 'body')])
    s['documents'] = [doc]; before = bank.project(s)
    s['documents'].append(duplicate)
    assert vector(bank.project(s)).tolist() == vector(before).tolist()


@pytest.mark.parametrize('team,scope', [('unknown', 'this_match'), ('home', 'historical'), ('home', 'unresolved')])
def test_unresolved_or_historical_claims_are_retained_outside_predictor_support(team, scope):
    s, doc = state(), article()
    bank = EvidenceBank([record(s, doc, 'title', empty=True), record(s, doc, 'body', team=team, scope=scope)])
    s['documents'] = [doc]; projected = bank.project(s)
    assert not any(vector(projected))
    assert len(projected['unmodeled_claims']) == 1


def test_failed_acquisition_metadata_and_clock_dont_create_football_features():
    s = state(); bank = EvidenceBank([])
    baseline = bank.project(s)
    s['history'] = [{'status': 'FAILED', 'latency': 50, 'cost': .1}]
    s['decision_at'] = (TIME+timedelta(hours=1)).isoformat()
    s['prediction_cutoff'] = s['decision_at']
    assert vector(bank.project(s)).tolist() == vector(baseline).tolist()


def test_frozen_semantic_output_and_future_document_are_rejected():
    s, doc = state(), article()
    r = record(s, doc, 'body')
    bank = EvidenceBank([record(s, doc, 'title', empty=True), r])
    s['documents'] = [doc]
    bank.records[r['input']['input_id']]['response']['claims'][0]['value'] = 'available'
    with pytest.raises(ValueError, match='modified'):
        bank.project(s)
    future = deepcopy(s)
    future['documents'][0]['source_published_at'] = (TIME+timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError, match='not available'):
        EvidenceBank([record(s, doc, 'title', empty=True), r]).project(future)


def training_rows():
    rows = []
    for i in range(12):
        t = TIME+timedelta(days=i*7)
        y = i % 3
        for acquired in (False, True):
            body = [0.]*len(FEATURES)
            if acquired:
                body[FEATURES.index(['away:availability:unavailable', 'home:availability:doubtful',
                                     'home:availability:unavailable'][y])] = 1.
            rows.append({'physical_match_id': str(i), 'outcome': y,
                'result_available_at': (t+timedelta(hours=5)).isoformat(),
                'state': {'physical_match_id': str(i), 'decision_at': t.isoformat(),
                    'prior': [.4, .3, .3], 'title': [0.]*len(FEATURES), 'novel_body': body}})
    return rows


def test_learned_effects_are_not_hand_assigned_and_fit_replays(tmp_path):
    rows = training_rows()
    coefficients, _ = _fit(rows, .1)
    before, after = rows[0]['state'], rows[1]['state']
    assert _predict(coefficients, before).tolist() == pytest.approx(before['prior'])
    assert _predict(coefficients, after)[0] > _predict(coefficients, before)[0]
    model = fit_semantic_f(rows, dataset_binding={'kind': 'constructed_test'})
    path = tmp_path/'model.json'; model.save(path)
    restored = SemanticResidualF.load(path)
    assert model.model_version == restored.model_version
    assert model.coefficients == restored.coefficients
    assert model.metadata['selection']['independent_validation_matches'] == 6
    for fold in model.metadata['selection']['folds']:
        assert set(fold['training_ids']).isdisjoint(fold['validation_ids'])
    corrupt = json.loads(path.read_text(encoding='utf-8')); corrupt['coefficients'][0][0] += .01
    path.write_text(json.dumps(corrupt), encoding='utf-8')
    with pytest.raises(ValueError, match='digest'):
        SemanticResidualF.load(path)


def test_existing_provider_uses_the_new_source_only_protocol_without_new_transport():
    from football_agent.deepseek_provider import DeepSeekInquiryProvider
    s, doc = state(), article()
    provider = DeepSeekInquiryProvider('.', protocol=SourceAssertionProtocol, api_key='constructed-test-key')
    payload = provider._payload(public_input(s['fixture'], doc, 'body'))
    assert 'probabilities' not in payload['messages'][1]['content']
    assert 'tools' not in payload
    assert provider.configuration()['automatic_retries'] == 0
    assert provider.configuration()['inquiry_protocol'] == EXTRACTION_VERSION


def test_diagnostic_candidate_cannot_use_changed_labels_or_replace_selection():
    rows = training_rows()
    selected = fit_semantic_f(rows, dataset_binding={'kind': 'constructed_test'})
    original = selected.payload()
    candidate = fit_diagnostic_candidate(rows, selected_model=selected)
    assert selected.payload() == original
    assert candidate.metadata['selection'] == selected.metadata['selection']
    best = min((c for c in selected.metadata['selection']['candidates'] if c['regularization'] is not None),
               key=lambda c: c['mean_match_log_loss'])
    assert candidate.metadata['applied_regularization'] == best['regularization']
    assert candidate.metadata['production_ready'] is False
    contaminated = deepcopy(rows)
    contaminated[0]['outcome'] = (contaminated[0]['outcome']+1) % 3
    with pytest.raises(ValueError, match='exact original A'):
        fit_diagnostic_candidate(contaminated, selected_model=selected)
