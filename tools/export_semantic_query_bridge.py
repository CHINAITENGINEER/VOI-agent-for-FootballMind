"""Export outcome-free g inputs, separate hindsight targets and offline episodes.

Reuses the archived source panel, frozen f, selector and exploration sampler.
No data collection, g fitting, live promotion or outcome-aware action ranking.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_semantic_f import CASES, PRIORS, read, write, binding, no_network
from football_agent.inquiry_exploration import UniformInquiryExploration
from football_agent.semantic_query_policy import SemanticQueryPolicy
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.historical_context_training import (
    _discovery_panel, _state_for, prepare_historical_cases,
)
from football_research.acquisition.semantic_f import EvidenceBank, SemanticResidualF
from football_research.acquisition.semantic_query import query_input, response_audit, realized_target


def request(case, index):
    return {'action_id': 'article-'+str(index), 'tool': 'read_url',
        'url': case['documents'][index]['url'], 'cost': {'money': 0., 'requests': 1},
        'cost_status': 'simulated'}


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def episode(case, *, model, bank, prior, seed, request_limit, query_protocol=None):
    """A finite archived environment; the policy never receives the case object."""
    make_query = query_protocol.query_input if query_protocol else query_input
    audit_response = query_protocol.response_audit if query_protocol else response_audit
    schema_args = {'input_schema': query_protocol.VERSION} if query_protocol else {}
    policy = SemanticQueryPolicy(predictor_payload_sha256=_digest(model.payload()),
                                 lambda_cost={'requests': .001}, **schema_args)
    sampler = UniformInquiryExploration(seed)
    mid = case['fixture']['physical_match_id']
    run = {'run_id': 'semantic-panel-'+mid.replace(':', '-'),
        'configuration': {'inquiry_exploration': sampler.configuration(), 'policy': policy.manifest,
            'request_limit': request_limit, 'evidence_origin': 'historical_reconstruction',
            'request_cost_scope': 'simulated_panel_reveal_credits_not_HTTP_or_dollars',
            'candidate_generation_complete': False}, 'decisions': []}
    acquired, history = [], []
    for step in range(request_limit+1):
        state = _state_for(case, acquired_indices=acquired)
        state['history'] = deepcopy(history)
        queries = [make_query(state, request(case, i), model=model, prior_snapshot=prior, evidence_bank=bank)
                   for i in range(len(case['documents'])) if i not in acquired]
        budget = {'limits': {'money': 0., 'requests': request_limit},
                  'spent': {'money': 0., 'requests': len(acquired)}}
        plan = policy.plan(queries, budget, candidate_generation_complete=False)
        row = {'step': step, 'queries': queries, 'plan': plan, 'selected_action': None,
               'information_sufficiency_established': False}
        selected = plan['selection']['candidate_id']
        if selected is None:
            eligible = [c['action_id'] for c in plan['candidates'] if c['feasible']
                        and c['value']['status'] != 'COMPUTED']
            if eligible and len(acquired) < request_limit:
                draw = sampler.select(run, {'decision': {'kind': 'query', 'candidate_id': eligible[0],
                    'revisit_at': None}, 'candidates': [{'candidate_id': q['candidate']['action_id']} for q in queries]}, eligible)
                selected = draw['candidate_id']
                row.update(selection_basis='bounded_unknown_information_exploration',
                           exploration_record=draw['exploration_record'])
            else:
                terminal = ('physical_budget_exhausted' if len(acquired) >= request_limit else
                            'archived_menu_exhausted_open_environment_unresolved')
                row['terminal_reason'] = terminal
                run['decisions'].append(row)
                run['terminal_reason'] = terminal
                break
        else:
            row['selection_basis'] = 'positive_supported_fixed_f_query_gain'
        chosen = next(q for q in queries if q['candidate']['action_id'] == selected)
        index = next(i for i in range(len(case['documents'])) if request(case, i)['action_id'] == selected)
        acquired.append(index)
        history.append({'action_id': selected, 'status': 'OK'})
        after = _state_for(case, acquired_indices=acquired)
        after['history'] = deepcopy(history)
        row.update(selected_action=selected, query_sha256=chosen['query_sha256'],
            response=audit_response(chosen, after, model=model, prior_snapshot=prior, evidence_bank=bank))
        run['decisions'].append(row)
    run.update(physical_match_id=mid, model_version=model.model_version,
               panel_reads=len(acquired), actual_HTTP_requests=0,
               expected_gains_estimated=0, open_world_policy_success_established=False)
    return run


def run(source, output):
    socket.socket.connect = socket.socket.connect_ex = socket.getaddrinfo = socket.create_connection = no_network
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = {'cases': CASES, 'priors': PRIORS, 'protocol': source/'protocol.json',
        'records': source/'assistant-records.json', 'selected_f': source/'model.json',
        'candidate_f': source/'diagnostics-v1/candidate-model.json',
        'old_diagnostics': source/'diagnostics-v1/candidate-action-effects.json'}
    bindings = {k: binding(p) for k, p in source_files.items()}
    # Freeze scope before reading any targets for this run. This is an already
    # reviewed development panel, never a new independent/blinded experiment.
    config = {'schema': 'semantic-query-bridge-experiment-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'input_bindings': bindings,
        'g_fit': False, 'network_requests': 0, 'live_promotion': False,
        'row_scope': 'all_B_C_one_read_transitions_from_original_frozen_panel',
        'zero_target_policy': 'keep_all_valid_fixed_f_zeros_do_not_select_on_improvement',
        'learner_inputs': 'pre-read-only.jsonl', 'targets': 'hindsight-targets.jsonl',
        'post_read_audits': 'post-read-audits.jsonl',
        'episodes': {'physical_match_ids': ['understat:28986', 'understat:29015', 'understat:29038'],
                     'seed': 20261010, 'request_limit': 2,
                     'scope': 'historical_finite_panel_mechanism_check_not_original_agent_execution'},
        'g_integration': 'existing_g_models_bind_different_f_or_synthetic_teacher_not_substituted',
        'limits': ['historical_article_version_assumed', 'prior_reconstructed_after_the_fact',
                   'previously_reviewed_nonblind_corpus', 'diagnostic_candidate_not_selected_for_live_use',
                   'assistant_extraction_not_independently_blinded', 'small_dependent_state_panel']}
    write(output/'protocol.json', config)
    cases, _ = prepare_historical_cases(read(CASES))
    protocol, priors = read(source_files['protocol']), read(PRIORS)
    records = read(source_files['records'])
    bank = EvidenceBank(records)
    model = SemanticResidualF.load(source_files['candidate_f'])
    selected = SemanticResidualF.load(source_files['selected_f'])
    if bank.sha256 != model.metadata['dataset_binding']['semantic_bank_sha256']:
        raise ValueError('Wrong semantic source panel')
    old = {r['sample_id']: r for r in read(source_files['old_diagnostics'])}
    before_rows, audit_rows, target_rows, groups = [], [], [], []
    for group in ('B', 'C'):
        for case in cases:
            mid = case['fixture']['physical_match_id']
            if mid not in protocol['splits'][group]:
                continue
            for dm, am, discovered, acquired, before in _discovery_panel(case):
                for index in discovered:
                    if index in acquired:
                        continue
                    key = f'{mid}:{dm}:{am}:{index}'
                    q = query_input(before, request(case, index), model=model,
                                    prior_snapshot=priors[mid], evidence_bank=bank)
                    after = _state_for(case, acquired_indices=acquired+[index], discovered_indices=discovered)
                    audit = response_audit(q, after, model=model, prior_snapshot=priors[mid], evidence_bank=bank)
                    target = realized_target(q, audit, outcome=case['outcome'],
                        result_available_at=case['result_available_at'], label_provenance=case['label_provenance'])
                    original = old[key]
                    if (audit['prediction_before'] != original['prediction_before']
                            or audit['prediction_after'] != original['prediction_after']
                            or abs(target['realized_gain_nats']-original['realized_log_loss_change_nats']) > 1e-14):
                        raise AssertionError('The bridge changed an already frozen f response')
                    before_rows.append({'sample_id': key, 'query': q})
                    audit_rows.append({'sample_id': key, 'audit': audit})
                    target_rows.append({'sample_id': key, 'target': target})
                    groups.append({'sample_id': key, 'physical_match_id': mid, 'group': group,
                        'query_sha256': q['query_sha256'], 'weight_unit': 'physical_match_not_prefix'})
    if set(r['sample_id'] for r in before_rows) != set(old):
        raise AssertionError('A counterfactual transition was dropped or added')
    write_jsonl(output/'pre-read-only.jsonl', before_rows)
    write_jsonl(output/'post-read-audits.jsonl', audit_rows)
    write_jsonl(output/'hindsight-targets.jsonl', target_rows)
    write_jsonl(output/'physical-match-splits.jsonl', groups)
    traces, replay_count = [], 0
    for variant, f in [('selected_prior_only', selected), ('diagnostic_semantic_f', model)]:
        for mid in config['episodes']['physical_match_ids']:
            case = next(c for c in cases if c['fixture']['physical_match_id'] == mid)
            result = episode(case, model=f, bank=bank, prior=priors[mid],
                             seed=config['episodes']['seed'], request_limit=config['episodes']['request_limit'])
            replay = episode(case, model=SemanticResidualF.load(source_files['selected_f' if
                variant == 'selected_prior_only' else 'candidate_f']), bank=EvidenceBank(read(source_files['records'])),
                prior=read(PRIORS)[mid], seed=config['episodes']['seed'], request_limit=config['episodes']['request_limit'])
            if result != replay:
                raise AssertionError('Full offline episode replay differs')
            replay_count += 1
            traces.append({'variant': variant, 'fixture': case['fixture'], 'run': result})
    write(output/'offline-decision-trajectories.json', traces)
    contract = {'schema': 'semantic-f-g-consumer-v2', 'model_version': model.model_version,
        'predictor_payload_sha256': _digest(model.payload()), 'f_fit_physical_match_ids': protocol['splits']['A'],
        'gain_fit_recommended_groups': ['B'], 'development_evaluation_groups': ['C'],
        'independent_blind_test': False, 'actor_input_only': 'pre-read-only.jsonl',
        'supervisor_only': ['hindsight-targets.jsonl', 'post-read-audits.jsonl'],
        'do_not_encode_as_predictors': ['physical_match_id', 'query_sha256', 'sample_id', 'source_URL_identity'],
        'target': 'conditional_frozen_predictor_log_loss_change',
        'gain_units': 'natural_log_loss_nats_before_cost',
        'zero_fixed_f_labels_valid_but_information_irrelevance_not_established': True,
        'realized_harm_target': 'one_outcome_log_loss_increase_not_negative_expected_net_probability',
        'information_scope': 'finite_reconstructed_menu_not_open_world_coverage',
        'risk_probability_calibrated': False, 'prospective_eligible': False,
        'retraining_required_if_f_or_semantic_representation_changes': True}
    write(output/'g-consumer-contract.json', contract)
    if any(binding(item['path'])['sha256'] != item['sha256'] for item in bindings.values()):
        raise AssertionError('A frozen input artifact changed')
    audits = [r['audit'] for r in audit_rows]
    summary = {'pre_read_queries': len(before_rows), 'post_read_audits': len(audit_rows),
        'separate_realized_targets': len(target_rows), 'physical_matches': len({r['physical_match_id'] for r in groups}),
        'all_previous_predictions_and_labels_preserved': True,
        'interpretations': dict(Counter(a['interpretation'] for a in audits)),
        'rows_with_collapsed_new_predicates': sum(bool(a['feature_collisions']) for a in audits),
        'zero_gain_targets_retained': sum(abs(r['target']['realized_gain_nats']) <= 1e-12 for r in target_rows),
        'full_episodes_replayed_identically': replay_count,
        'episode_panel_reads': sum(t['run']['panel_reads'] for t in traces),
        'episode_terminal_reasons': dict(Counter(t['run']['terminal_reason'] for t in traces)),
        'numeric_g_estimates': 0, 'g_models_trained': 0, 'HTTP_requests': 0,
        'prospective_training_eligible_rows': 0, 'live_models_changed': False,
        'evaluation_scope': 'mechanism_and_f_g_interface_not_accuracy_or_autonomous_VoI_validation'}
    write(output/'report.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.source, args.output)
