"""Offline body/claim/projection controls, not provider or live-runtime tests."""
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from football_research.acquisition.entity_semantic_f import EntityEvidenceBank, EntitySemanticF, _predict
from football_research.acquisition.semantic_f import EXTRACTION_VERSION, FEATURES, public_input
from football_research.acquisition.historical_context_training import _project
from football_research.acquisition.news_link_projection import VERSION as NEWS_PROJECTION


def control_inputs(kind):
    state = {
        'fixture': {'physical_match_id': 'constructed:body-control',
                    'match_id': 'constructed:body-control', 'home_team_id': 'North',
                    'away_team_id': 'South', 'home_name': 'North', 'away_name': 'South',
                    'competition_id': 'constructed:league', 'competition': 'Constructed League',
                    'kickoff': '2025-10-02T12:00:00Z'},
        'decision_at': '2025-10-01T12:00:00Z', 'prediction_cutoff': '2025-10-01T12:00:00Z',
        'information_mode': 'historical_reconstruction',
        'prior_probabilities': [.4, .3, .3], 'prior_basis': 'constructed_contract_test_not_market',
        'documents': [], 'observed_links': [], 'history': [],
    }
    text = 'North forward Alex is ruled out of the South match.'
    predicate, value, scope = 'availability', 'unavailable', 'this_match'
    if kind == 'unmodeled_body':
        text = 'North forward Alex attended a community event last week.'
        predicate, value, scope = 'other', 'unmodeled', 'team_context'
    elif kind == 'zero_coefficient_feature':
        text = 'North coach confirms planned rotation against South.'
        predicate, value = 'selection', 'rotation_planned'
    doc = {'url': 'https://example.test/control', 'title': 'North update', 'text': text,
           'source_published_at': '2025-10-01T10:00:00Z',
           'captured_at': '2025-10-03T10:00:00Z'}
    if kind == 'already_known_in_title':
        doc['title'] = text
    records = []
    # Use the predictor's canonical fixture before computing semantic input IDs.
    # Equivalent ISO timestamps are not byte-identical hash inputs.
    canonical_fixture = _project(state, text_projection=NEWS_PROJECTION)['fixture']
    for channel in ('title', 'body'):
        public = public_input(canonical_fixture, doc, channel)
        claims = []
        if channel == 'body' or kind == 'already_known_in_title':
            claims = [{'team': 'home', 'entity': 'Alex', 'kind': predicate, 'value': value,
                       'scope': scope, 'statement': text, 'quotes': [public['text']]}]
        records.append({'input': public,
                        'response': {'schema': EXTRACTION_VERSION,
                                     'input_id': public['input_id'], 'claims': claims},
                        'provenance': {'kind': 'constructed_contract_test',
                                       'extraction': 'hand_authored_expected_response_no_provider'}})
    state['observed_links'] = [{'url': doc['url'], 'title': doc['title'],
                               'title_basis': 'historical_headline_panel',
                               'available_at': doc['source_published_at']}]
    return state, doc, records


def run_controls(model):
    rows = []
    for kind in ('modeled_body', 'unmodeled_body', 'already_known_in_title', 'zero_coefficient_feature'):
        state, doc, records = control_inputs(kind)
        bank = EntityEvidenceBank(records)
        before = bank.project(state)
        state['documents'] = [doc]
        after = bank.project(state)
        p, q = _predict(model.coefficients, before), _predict(model.coefficients, after)
        shift = float(np.sum(np.abs(q - p)))
        body_id = records[1]['input']['input_id']
        responded = shift > 1e-12
        expected = kind == 'modeled_body'
        rows.append({'case': kind, 'inputs': {'body': doc, 'semantic_records': records},
                     'body_consumed': body_id in after['used_input_ids'],
                     'before': p.tolist(), 'after': q.tolist(), 'probability_l1_shift': shift,
                     'changed_body_features': [f for i, f in enumerate(FEATURES)
                         if before['novel_body_entity_counts'][i] != after['novel_body_entity_counts'][i]],
                     'used_input_ids': after['used_input_ids'],
                     'unmodeled_claims': after['unmodeled_claims'],
                     'probabilities_valid': bool(np.all(np.isfinite(q)) and np.all(q >= 0)
                                                 and np.isclose(q.sum(), 1)),
                     'expected_response_for_this_frozen_model': expected,
                     'check_passed': responded == expected,
                     'gain': None, 'g_training_eligible': False})
    errors = []
    expected_errors = {
        'missing_body_extraction': 'SEMANTIC_EXTRACTION_MISSING:',
        'bad_input_id': 'Source assertion input binding differs',
        'bad_quote': 'Claim must cite an exact acquired text span',
        'bad_predicate': 'Source predicate/value mismatch',
        'future_body': 'The document is not available at this decision',
    }
    for kind in ('missing_body_extraction', 'bad_input_id', 'bad_quote', 'bad_predicate', 'future_body'):
        state, doc, records = control_inputs('modeled_body')
        state['documents'] = [doc]
        if kind == 'missing_body_extraction':
            records = records[:1]
        elif kind == 'bad_input_id':
            records[1]['response']['input_id'] = 'not-the-input-id'
        elif kind == 'bad_quote':
            records[1]['response']['claims'][0]['quotes'] = ['A quote absent from the acquired body.']
        elif kind == 'bad_predicate':
            records[1]['response']['claims'][0]['value'] = 'unmodeled'
        else:
            doc['source_published_at'] = '2025-10-01T13:00:00Z'
        error = None
        try:
            EntityEvidenceBank(records).project(state)
        except ValueError as exc:
            error = str(exc)
        errors.append({'case': kind, 'rejected': error is not None, 'error': error,
                       'expected_error': expected_errors[kind],
                       'correct_rejection': error is not None and expected_errors[kind] in error,
                       'prediction_after': None, 'gain': None})
    return {'schema': 'entity-f-body-contract-controls-v1', 'model_version': model.model_version,
            'prior_origin': 'constructed_contract_test_not_market',
            'article_origin': 'constructed_contract_test_not_real_articles',
            'extraction_origin': 'hand_authored_expected_responses_not_model_extraction',
            'path': 'public_input -> validate_response -> EntityEvidenceBank.project -> frozen _predict',
            'runtime_policy_executed': False, 'provider_executed': False,
            'network_requests': 0, 'training': False, 'g_labels_emitted': 0,
            'positive_and_negative_controls': rows, 'rejection_controls': errors,
            'passed': all(r['check_passed'] and r['body_consumed'] and r['probabilities_valid'] for r in rows)
                      and all(r['correct_rejection'] for r in errors),
            'limitations': ['Does not validate provider extraction or live runtime wiring.',
                            'Response does not establish accuracy or information value.',
                            'Zero response is not a training exclusion rule.',
                            'Historical adapter clocks here are constructed test inputs, not measured observations.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.model.read_bytes()
    report = run_controls(EntitySemanticF.load(args.model))
    if args.model.read_bytes() != raw:
        raise ValueError('Model changed during controls')
    report.update(model_raw_sha256=sha256(raw).hexdigest(), executed_at=datetime.now(timezone.utc).isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'passed': report['passed'], 'cases': len(report['positive_and_negative_controls']),
                      'rejections': len(report['rejection_controls'])}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
