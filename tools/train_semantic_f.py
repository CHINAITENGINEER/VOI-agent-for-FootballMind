"""Prepare outcome-free extraction inputs, then fit an offline semantic f.

Uses existing source archives, Poisson snapshots and optimizer. The extract
command reuses DeepSeekInquiryProvider, with no new HTTP/retry implementation.
An assistant-reviewed pilot can import responses using the same strict schema;
its provenance must never be presented as a DeepSeek call or prospective data.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.historical_context_training import (
    HistoricalContextModel, _discovery_panel, _state_for, prepare_historical_cases,
)
from football_research.acquisition.match_conditioned_f import bind_prior
from football_research.acquisition.semantic_f import (
    PARAMETERS, EXTRACTION_VERSION, EvidenceBank, SemanticResidualF,
    SourceAssertionProtocol, public_input, fit_semantic_f, fit_diagnostic_candidate, validate_response,
)

CASES = ROOT/'var/submission/p8-g-stream-v3/cases.json'
OLD = ROOT/'var/submission/p9-g-timestamped-replay-20261010/model.json'
PRIORS = ROOT/'var/f-match-conditioned-20261010-v1/structured-priors.json'


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def binding(path):
    return {'path': str(path), 'sha256': sha256(Path(path).read_bytes()).hexdigest()}


def no_network(*args, **kwargs):
    raise AssertionError('Offline training and verification do not call network')


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    old = HistoricalContextModel.load(OLD)
    protocol = {'schema': 'semantic-residual-experiment-v1',
        'frozen_at': datetime.now(timezone.utc).isoformat(), 'parameters': PARAMETERS,
        'inputs': {name: binding(path) for name, path in [('cases', CASES), ('old_f', OLD), ('priors', PRIORS)]},
        'splits': {group: old.metadata[group+'_match_ids'] for group in ('A', 'B', 'C')},
        'extraction': 'source_assertions_blinded_input_no_probabilities_labels_or_gain',
        'source_semantics': 'LLM_classification_exact_quotes_not_world_truth',
        'fit': 'A_only_purged_forward_equal_match_weights',
        'evaluation': 'B_C_previously_reviewed_development_corpus_not_blind_test',
        'extraction_invariance': 'missing_output_is_error_duplicate_concept_does_not_add_weight',
        'g_training': False, 'live_promotion': False,
        'assistant_pilot_allowed': True,
        'assistant_pilot_limit': 'current_session_has_prior_project_context_not_independently_blinded_annotator',
        'provider_budget': {'max_calls': 124, 'automatic_retries': 0},
        'support_limit': 'unmodeled_claims_remain_unknown_not_proof_of_zero_information_value'}
    write(output/'protocol.json', protocol)
    cases, audit = prepare_historical_cases(read(CASES))
    unique, refs = {}, []
    for ci, case in enumerate(cases):
        for di, document in enumerate(case['documents']):
            for channel in ('title', 'body'):
                item = public_input(case['fixture'], document, channel)
                unique[item['input_id']] = item
                refs.append({'ref': f'c{ci}d{di}:{channel}', 'input_id': item['input_id'],
                    'source_document_id': document.get('document_id'),
                    'source_raw_sha256': document.get('raw_sha256')})
    write(output/'extraction-inputs.json', list(unique.values()))
    write(output/'extraction-index.json', refs)
    write(output/'source-audit.json', audit)
    write(output/'extraction.schema.json', SourceAssertionProtocol.schema)
    (output/'extraction-prompt.txt').write_text(SourceAssertionProtocol.prompt, encoding='utf-8')
    print(json.dumps({'stage': 'prepared', 'source_articles': sum(len(c['documents']) for c in cases),
                      'extraction_inputs': len(unique), 'output': str(output)}, ensure_ascii=True), flush=True)


def extract(output):
    from football_agent.deepseek_provider import DeepSeekInquiryProvider
    provider = DeepSeekInquiryProvider(output/'provider-calls', protocol=SourceAssertionProtocol,
                                      max_tokens=4096, thinking='disabled', timeout_seconds=90)
    if not provider.configuration()['configured']:
        raise ValueError('DEEPSEEK_API_KEY is not configured; no provider request made')
    inputs, protocol = read(output/'extraction-inputs.json'), read(output/'protocol.json')
    if len(inputs) > protocol['provider_budget']['max_calls']:
        raise ValueError('Extraction exceeds frozen provider-call budget')
    destination = output/'provider-records'
    destination.mkdir(exist_ok=False)
    for index, public in enumerate(inputs):
        result = provider.decide(public)
        validate_response(public, result['response'])
        record = {'input': public, 'response': result['response'], 'provenance': {
            'kind': 'deepseek_api', 'provider_attempt_folder': provider.last_attempt_folder,
            'receipt': deepcopy(result['receipt']), 'extracted_at': datetime.now(timezone.utc).isoformat()}}
        write(destination/(public['input_id']+'.json'), record)
        print(json.dumps({'stage': 'extracted', 'completed': index+1, 'total': len(inputs)}), flush=True)


def train(output, records_path):
    socket.socket.connect, socket.getaddrinfo = no_network, no_network
    protocol = read(output/'protocol.json')
    if protocol['parameters'] != PARAMETERS:
        raise ValueError('Frozen extraction/training protocol changed')
    for item in protocol['inputs'].values():
        if binding(item['path'])['sha256'] != item['sha256']:
            raise ValueError('Frozen source input changed')
    original = {p: sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'var/f-text-live-integration-20261010-v1').rglob('*') if p.is_file()}
    records = ([read(p) for p in sorted(records_path.glob('*.json'))]
               if records_path.is_dir() else read(records_path))
    bank = EvidenceBank(records)
    if set(bank.records) != {x['input_id'] for x in read(output/'extraction-inputs.json')}:
        raise ValueError('Complete frozen extraction panel required; no silent omission')
    cases, _ = prepare_historical_cases(read(CASES))
    splits, snapshots = protocol['splits'], read(PRIORS)
    if set(sum(splits.values(), [])) != {c['fixture']['physical_match_id'] for c in cases}:
        raise ValueError('Original physical-match split changed')
    prepared, original_states = {}, {}
    for case in cases:
        mid = case['fixture']['physical_match_id']
        prepared[mid], original_states[mid] = [], []
        for discovered, acquired, _, _, state in _discovery_panel(case):
            projected = bank.project(bind_prior(state, snapshots[mid]))
            prepared[mid].append({'physical_match_id': mid, 'outcome': case['outcome'],
                'result_available_at': case['result_available_at'], 'state': projected})
            original_states[mid].append((discovered, acquired, state))
    training = [r for mid in splits['A'] for r in prepared[mid]]
    model = fit_semantic_f(training, dataset_binding={'cases': protocol['inputs']['cases'],
        'A_match_ids': splits['A'], 'semantic_bank_sha256': bank.sha256,
        'extractor_provenance': sorted({r['provenance']['kind'] for r in records})})
    model.save(output/'model.json')
    restored = SemanticResidualF.load(output/'model.json')
    write(output/'selection.json', model.metadata['selection'])
    print(json.dumps({'stage': 'f_frozen', 'version': model.model_version,
        'regularization': model.metadata['selection']['selected_regularization']}), flush=True)
    evaluation, targets = {}, []
    for group in ('B', 'C'):
        matches = []
        for case in cases:
            mid = case['fixture']['physical_match_id']
            if mid not in splits[group]:
                continue
            predictions = []
            for discovered, acquired, state in original_states[mid]:
                value = restored.predict(state, prior_snapshot=snapshots[mid], evidence_bank=bank)
                if value != model.predict(state, prior_snapshot=snapshots[mid], evidence_bank=bank):
                    raise AssertionError('Frozen model replay differs')
                predictions.append({'discovered_mask': discovered, 'acquired_mask': acquired,
                    'probabilities': value['probabilities'], 'prior': value['initial_probabilities'],
                    'semantic_inputs': value['used_semantic_input_ids']})
            losses = {name: float(np.mean([-np.log(p[key][case['outcome']]) for p in predictions]))
                for name, key in [('structured_prior', 'prior'), ('semantic_f', 'probabilities')]}
            matches.append({'physical_match_id': mid, 'outcome': case['outcome'],
                            'losses': losses, 'predictions': predictions})
            if group == 'B':
                for dm, am, discovered, acquired, before in _discovery_panel(case):
                    p = model.predict(before, prior_snapshot=snapshots[mid], evidence_bank=bank)['probabilities']
                    for index in discovered:
                        if index in acquired:
                            continue
                        after = _state_for(case, acquired_indices=acquired+[index], discovered_indices=discovered)
                        q = model.predict(after, prior_snapshot=snapshots[mid], evidence_bank=bank)['probabilities']
                        targets.append({'sample_id': f'{mid}:{dm}:{am}:{index}', 'physical_match_id': mid,
                            'model_version': model.model_version, 'prediction_before': p, 'prediction_after': q,
                            'gain': float(np.log(q[case['outcome']]/p[case['outcome']])),
                            'gain_units': 'log_loss_nats', 'cost': None, 'net_gain': None,
                            'strict_g_training_eligible': False, 'evidence_origin': 'historical_reconstruction'})
        evaluation[group] = {'matches': matches, 'mean_match_log_loss': {
            name: float(np.mean([m['losses'][name] for m in matches])) for name in ('structured_prior', 'semantic_f')}}
    write(output/'evaluation.json', evaluation)
    write(output/'B-diagnostic-targets.json', targets)
    if any(sha256(p.read_bytes()).hexdigest() != h for p, h in original.items()):
        raise AssertionError('Prior live artifacts changed')
    claims = [claim for r in records for claim in r['response']['claims']]
    summary = {'model_version': model.model_version,
        'semantic_inputs': len(records), 'claims': len(claims),
        'modeled_match_claims': sum(c['scope'] == 'this_match' and c['team'] in ('home', 'away') and c['kind'] != 'other' for c in claims),
        'extractor_provenance': sorted({r['provenance']['kind'] for r in records}),
        'A_matches': len(splits['A']), 'A_states': len(training),
        'selection': {k: v for k, v in model.metadata['selection'].items() if k != 'folds'},
        'evaluation': {g: value['mean_match_log_loss'] for g, value in evaluation.items()},
        'old_live_files_unchanged': len(original), 'g_training_calls': 0, 'live_promoted': False,
        'B_targets': len(targets), 'B_nonzero_targets': sum(abs(t['gain']) > 1e-12 for t in targets),
        'strict_g_training_eligible_rows': 0,
        'limitations': ['Prior-reviewed Leeds-centered historical development corpus.',
            'Exact quotes check provenance, not semantic correctness or causality.',
            'Narrow fixed semantic vocabulary; unsupported information remains unknown.',
            'Assistant pilot is not an independently blinded extraction or real provider batch.',
            'No historical snapshot proof; no live promotion or g replacement.']}
    write(output/'report.json', summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


def diagnose(output, records_path):
    """Inspect learned response/harm without using outcomes to choose queries."""
    socket.socket.connect, socket.getaddrinfo = no_network, no_network
    protocol = read(output/'protocol.json')
    if protocol['parameters'] != PARAMETERS:
        raise ValueError('Frozen extraction/training protocol changed')
    for item in protocol['inputs'].values():
        if binding(item['path'])['sha256'] != item['sha256']:
            raise ValueError('Frozen source input changed')
    records = ([read(p) for p in sorted(records_path.glob('*.json'))]
               if records_path.is_dir() else read(records_path))
    bank = EvidenceBank(records)
    selected = SemanticResidualF.load(output/'model.json')
    if bank.sha256 != selected.metadata['dataset_binding']['semantic_bank_sha256']:
        raise ValueError('Extraction bank differs from the selected model experiment')
    cases, _ = prepare_historical_cases(read(CASES))
    snapshots = read(PRIORS)
    training = [{'physical_match_id': case['fixture']['physical_match_id'], 'outcome': case['outcome'],
        'result_available_at': case['result_available_at'],
        'state': bank.project(bind_prior(state, snapshots[case['fixture']['physical_match_id']]))}
        for mid in protocol['splits']['A'] for case in cases if case['fixture']['physical_match_id'] == mid
        for _, _, _, _, state in _discovery_panel(case)]
    destination = output/'diagnostics-v1'
    destination.mkdir(exist_ok=False)
    write(destination/'protocol.json', {'frozen_at': datetime.now(timezone.utc).isoformat(),
        'selection': 'lowest_recorded_A_forward_loss_among_nonzero_candidates_only',
        'source_protocol': binding(output/'protocol.json'), 'selected_model': binding(output/'model.json'),
        'semantic_records': {'sha256': bank.sha256},
        'purpose': 'post_selection_diagnostic_not_confirmation_or_live_promotion',
        'query_policy': None, 'g_fit': False, 'cost_or_net_gain': None,
        'constraints': ['Original one-SE choice and its artifacts unchanged.',
            'No B/C outcomes in fitting or hyperparameter selection.',
            'No retrospective realized gain used as an expected gain or query ranking.',
            'No claim of new independently blinded test evidence.']})
    candidate = fit_diagnostic_candidate(training, selected_model=selected)
    candidate.save(destination/'candidate-model.json')
    candidate = SemanticResidualF.load(destination/'candidate-model.json')
    evaluation, effects = {}, []
    for group in ('B', 'C'):
        matches = []
        for case in cases:
            mid = case['fixture']['physical_match_id']
            if mid not in protocol['splits'][group]:
                continue
            loss_prior, loss_candidate, states = [], [], {}
            for dm, am, discovered, acquired, state in _discovery_panel(case):
                prediction = candidate.predict(state, prior_snapshot=snapshots[mid], evidence_bank=bank)
                states[(dm, am)] = prediction
                loss_prior.append(-np.log(prediction['initial_probabilities'][case['outcome']]))
                loss_candidate.append(-np.log(prediction['probabilities'][case['outcome']]))
            matches.append({'physical_match_id': mid, 'fixture': case['fixture'],
                'outcome': case['outcome'], 'states': len(states),
                'mean_state_log_loss': {'structured_prior': float(np.mean(loss_prior)),
                                       'candidate': float(np.mean(loss_candidate))}})
            for dm, am, discovered, acquired, before in _discovery_panel(case):
                p = states[(dm, am)]
                for i in discovered:
                    if i in acquired:
                        continue
                    q = states[(dm, am | (1 << i))]
                    document = case['documents'][i]
                    extraction = bank.records[public_input(case['fixture'], document, 'body')['input_id']]
                    claims = extraction['response']['claims']
                    modeled = [c for c in claims if c['scope'] == 'this_match'
                               and c['team'] in ('home', 'away') and c['kind'] != 'other']
                    changed_concepts = (p['projection']['novel_body'] != q['projection']['novel_body'])
                    representation = ('new_modeled_concepts' if changed_concepts else
                        'modeled_concepts_already_present' if modeled else
                        'outside_frozen_feature_support' if claims else 'no_assertions_extracted')
                    delta = np.asarray(q['probabilities'])-p['probabilities']
                    effects.append({'sample_id': f'{mid}:{dm}:{am}:{i}', 'group': group,
                        'physical_match_id': mid, 'fixture': case['fixture'], 'outcome': case['outcome'],
                        'discovered_mask': dm, 'acquired_mask': am, 'article_index': i,
                        'document_url': document['url'], 'document_title': document['title'],
                        'source_published_at': document['source_published_at'],
                        'claims': claims, 'representation_status': representation,
                        'model_version': candidate.model_version,
                        'prediction_before': p['probabilities'], 'prediction_after': q['probabilities'],
                        'max_probability_change': float(np.max(np.abs(delta))),
                        'realized_log_loss_change_nats': float(np.log(q['probabilities'][case['outcome']]
                                                                     / p['probabilities'][case['outcome']])),
                        'expected_gain_before_read': None, 'risk_of_harm_before_read': None,
                        'cost': None, 'net_gain': None,
                        'strict_g_training_eligible': False,
                        'ineligible_reasons': ['historical_version_unverified', 'diagnostic_model_not_selected',
                                               'not_independently_blinded_extraction'],
                        'interpretation': 'realized_diagnostic_response_not_query_value_or_causal_effect'})
            # Check a restored f against its saved outputs using actual source states.
            restored = SemanticResidualF.load(destination/'candidate-model.json')
            for dm, am, _, _, state in _discovery_panel(case):
                if restored.predict(state, prior_snapshot=snapshots[mid], evidence_bank=bank) != states[(dm, am)]:
                    raise AssertionError('Diagnostic replay differs')
        evaluation[group] = {'matches': matches, 'mean_match_log_loss': {
            key: float(np.mean([m['mean_state_log_loss'][key] for m in matches]))
            for key in ('structured_prior', 'candidate')}}
    write(destination/'candidate-evaluation.json', evaluation)
    write(destination/'candidate-action-effects.json', effects)
    summary = {'candidate_model_version': candidate.model_version,
        'applied_regularization': candidate.metadata['applied_regularization'],
        'original_selection_unchanged': selected.model_version,
        'evaluation': {group: value['mean_match_log_loss'] for group, value in evaluation.items()},
        'independent_development_matches': sum(len(v['matches']) for v in evaluation.values()),
        'counterfactual_panel_action_rows': len(effects),
        'action_counts_are_not_independent_trials': True,
        'positive_realized_change': sum(e['realized_log_loss_change_nats'] > 1e-12 for e in effects),
        'negative_realized_change': sum(e['realized_log_loss_change_nats'] < -1e-12 for e in effects),
        'unchanged': sum(abs(e['realized_log_loss_change_nats']) <= 1e-12 for e in effects),
        'representation_counts': {key: sum(e['representation_status'] == key for e in effects)
            for key in sorted({e['representation_status'] for e in effects})},
        'strict_g_training_eligible_rows': 0, 'live_promoted': False,
        'provider_http_calls': 0, 'independent_blind_test': False,
        'scope': 'finite archived article panel; no evidence of open-world query selection success'}
    write(destination/'report.json', summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'extract', 'train', 'diagnose'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--records')
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    if args.stage == 'prepare':
        prepare(folder)
    elif args.stage == 'extract':
        extract(folder)
    else:
        if not args.records:
            parser.error('train/diagnose needs --records from a completed extraction')
        action = train if args.stage == 'train' else diagnose
        action(folder, Path(args.records).resolve())
