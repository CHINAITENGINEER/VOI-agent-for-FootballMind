"""Freeze, fit, replay and audit the entity-aware semantic f.

This is an offline research artifact. It never fits g, calls the network, or
promotes the diagnostic nonzero candidate. The selected f and diagnostic f are
kept separate so the g thread can bind the correct model deliberately.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_semantic_f import CASES, PRIORS, read, write, binding
from tools.export_semantic_query_bridge import episode, request
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.historical_context_training import (
    _discovery_panel, _state_for, prepare_historical_cases,
)
from football_research.acquisition.match_conditioned_f import bind_prior
from football_research.acquisition.semantic_f import EvidenceBank as CoarseEvidenceBank
from football_research.acquisition.semantic_f import SemanticResidualF
from football_research.acquisition.entity_semantic_f import (
    EntityEvidenceBank, EntitySemanticF, PARAMETERS, REPRESENTATION_SHA256, REPRESENTATION_VERSION,
    fit_diagnostic_candidate, fit_entity_semantic_f,
)
from football_research.acquisition import entity_semantic_query as entity_query

SOURCE = ROOT/'var/f-semantic-residual-20261010-v1'
RECORDS = SOURCE/'assistant-records.json'
OLD_MODEL = SOURCE/'model.json'
OLD_BRIDGE = ROOT/'var/f-g-semantic-bridge-20261010-v2/post-read-audits.jsonl'
OUTPUT_DEFAULT = ROOT/'var/f-entity-semantic-20261011-v1'


def jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def _guard_network(*args, **kwargs):
    raise AssertionError('Entity f experiment is offline; network access is forbidden')


def _prepare_rows(cases, snapshots, bank, splits):
    prepared = {}
    for case in cases:
        mid = case['fixture']['physical_match_id']
        prepared[mid] = []
        for dm, am, discovered, acquired, state in _discovery_panel(case):
            prepared[mid].append({'physical_match_id': mid, 'outcome': case['outcome'],
                'result_available_at': case['result_available_at'],
                'state': bank.project(bind_prior(state, snapshots[mid]))})
    return prepared, [r for mid in splits['A'] for r in prepared[mid]]


def _evaluate(cases, snapshots, bank, model, splits):
    result = {}
    for group in ('A', 'B', 'C'):
        matches = []
        for case in cases:
            mid = case['fixture']['physical_match_id']
            if mid not in splits[group]:
                continue
            losses, predictions = [], []
            for dm, am, discovered, acquired, state in _discovery_panel(case):
                value = model.predict(state, prior_snapshot=snapshots[mid], evidence_bank=bank)
                predictions.append({'discovered_mask': dm, 'acquired_mask': am,
                    'probabilities': value['probabilities'], 'prior': value['initial_probabilities'],
                    'entity_count_projection': {
                        'title': value['projection']['title_entity_counts'],
                        'novel_body': value['projection']['novel_body_entity_counts']}})
                losses.append(-math.log(value['probabilities'][case['outcome']]))
            prior_losses = [-math.log(p['prior'][case['outcome']]) for p in predictions]
            matches.append({'physical_match_id': mid, 'outcome': case['outcome'],
                            'states': len(predictions),
                            'mean_state_log_loss': {'structured_prior': sum(prior_losses)/len(prior_losses),
                                'entity_f': sum(losses)/len(losses)}, 'predictions': predictions})
        result[group] = {'matches': matches,
            'mean_match_log_loss': {
                name: (sum(m['mean_state_log_loss'][name] for m in matches)/len(matches) if matches else None)
                for name in ('structured_prior', 'entity_f')}}
    return result


def _transition_rows(cases, snapshots, bank, model, splits, old_audits):
    before_rows, audit_rows, targets, split_rows = [], [], [], []
    for group in ('B', 'C'):
        for case in cases:
            mid = case['fixture']['physical_match_id']
            if mid not in splits[group]:
                continue
            for dm, am, discovered, acquired, before in _discovery_panel(case):
                for index in discovered:
                    if index in acquired:
                        continue
                    sid = f'{mid}:{dm}:{am}:{index}'
                    q = entity_query.query_input(before, request(case, index), model=model,
                        prior_snapshot=snapshots[mid], evidence_bank=bank)
                    after = _state_for(case, acquired_indices=acquired+[index], discovered_indices=discovered)
                    audit = entity_query.response_audit(q, after, model=model,
                        prior_snapshot=snapshots[mid], evidence_bank=bank)
                    target = entity_query.realized_target(q, audit, outcome=case['outcome'],
                        result_available_at=case['result_available_at'], label_provenance=case['label_provenance'])
                    before_rows.append({'sample_id': sid, 'group': group, 'query': q})
                    audit_rows.append({'sample_id': sid, 'group': group, 'audit': audit})
                    targets.append({'sample_id': sid, 'group': group, 'target': target})
                    split_rows.append({'sample_id': sid, 'physical_match_id': mid, 'group': group,
                        'query_sha256': q['query_sha256'], 'weight_unit': 'physical_match_not_prefix',
                        'strict_g_training_eligible': False})
    if len(before_rows) != len(old_audits):
        raise AssertionError(f'Frozen transition count changed: {len(before_rows)} != {len(old_audits)}')
    return before_rows, audit_rows, targets, split_rows


def _collision_audit(cases, snapshots, bank, selected, diagnostic, old_audits):
    result = []
    for case in cases:
        mid = case['fixture']['physical_match_id']
        for dm, am, discovered, acquired, before in _discovery_panel(case):
            for index in discovered:
                if index in acquired:
                    continue
                sid = f'{mid}:{dm}:{am}:{index}'
                old = old_audits.get(sid)
                if not old or not old.get('feature_collisions'):
                    continue
                after = _state_for(case, acquired_indices=acquired+[index], discovered_indices=discovered)
                row = {'sample_id': sid, 'physical_match_id': mid,
                    'old_binary_collision_claims': old['feature_collisions']}
                for label, model in (('selected_f', selected), ('diagnostic_f', diagnostic)):
                    q = entity_query.query_input(before, request(case, index), model=model,
                        prior_snapshot=snapshots[mid], evidence_bank=bank)
                    audit = entity_query.response_audit(q, after, model=model,
                        prior_snapshot=snapshots[mid], evidence_bank=bank)
                    row[label] = {'feature_vector_changed': audit['feature_vector_changed'],
                        'entity_count_delta': audit['entity_count_delta'],
                        'max_probability_change': audit['max_probability_change'],
                        'interpretation': audit['interpretation'],
                        'new_predicates': audit['new_predicates']}
                result.append(row)
    return result


def _write_receipt(output):
    files = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'delivery-receipt.json':
            files.append({'path': path.relative_to(output).as_posix(),
                          'bytes': path.stat().st_size,
                          'sha256': sha256(path.read_bytes()).hexdigest()})
    receipt = {'schema': 'footballmind-entity-f-delivery-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'root': str(output),
        'files': files, 'file_count': len(files), 'manifest_sha256': _digest(files)}
    write(output/'delivery-receipt.json', receipt)
    return receipt


def run(output=OUTPUT_DEFAULT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # The experiment imports no transport; this guard makes accidental calls a
    # hard failure instead of an unlogged provider request.
    original = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo, socket.create_connection)
    socket.socket.connect = socket.socket.connect_ex = socket.getaddrinfo = socket.create_connection = _guard_network
    try:
        protocol_source = SOURCE/'protocol.json'
        source_paths = {'cases': CASES, 'priors': PRIORS, 'records': RECORDS,
                        'old_f': OLD_MODEL, 'old_bridge_audits': OLD_BRIDGE, 'old_protocol': protocol_source}
        bindings = {key: binding(path) for key, path in source_paths.items()}
        source_protocol = read(protocol_source)
        if source_protocol['parameters'] != read(SOURCE/'protocol.json')['parameters']:
            raise AssertionError('Source protocol unexpectedly changed')
        records = read(RECORDS)
        bank = EntityEvidenceBank(records)
        coarse_bank = CoarseEvidenceBank(records)
        cases, _ = prepare_historical_cases(read(CASES))
        snapshots = read(PRIORS)
        splits = source_protocol['splits']
        prepared, training = _prepare_rows(cases, snapshots, bank, splits)
        frozen_protocol = {'schema': 'entity-semantic-f-experiment-v1',
            'frozen_at': datetime.now(timezone.utc).isoformat(), 'source_bindings': bindings,
            'parameters': PARAMETERS, 'representation_sha256': REPRESENTATION_SHA256,
            'training': 'A_only_purged_forward_equal_physical_match_weights',
            'splits': splits, 'entity_representation': 'distinct source entity counts per coarse predicate',
            'entity_identity': 'casefolded whitespace normalized exact source entity; no aliases',
            'body_novelty': 'set difference at full entity-predicate level, not coarse status only',
            'contradictions': 'retain every source status; no last-write-wins truth imputation',
            'g_fit': False, 'live_promotion': False,
            'diagnostic_candidate': 'A forward best nonzero only; not selected for operation',
            'unknown_entity': 'revealed source entities count immediately; no name importance learned for unseen future names',
            'scope': 'historical_reconstruction_and_constructed_query_interface_not_prospective_evidence'}
        write(output/'protocol.json', frozen_protocol)
        write(output/'entity-representation.json', {'schema': REPRESENTATION_VERSION,
            'parameters': PARAMETERS, 'representation_sha256': REPRESENTATION_SHA256,
            'source_record_count': len(records), 'modeled_entity_predicates': sum(
                1 for r in records for c in r['response']['claims']
                if c['scope']=='this_match' and c['team'] in ('home','away') and c['kind']!='other'),
            'note': 'Entity names are source identity keys and count units, not player quality or importance.'})
        selected = fit_entity_semantic_f(training, dataset_binding={
            'cases': bindings['cases'], 'A_match_ids': splits['A'],
            'semantic_bank_sha256': bank.sha256, 'source_protocol': bindings['old_protocol']})
        selected.save(output/'model.json')
        selected = EntitySemanticF.load(output/'model.json')
        diagnostic = fit_diagnostic_candidate(training, selected_model=selected)
        (output/'diagnostics').mkdir()
        diagnostic.save(output/'diagnostics/candidate-model.json')
        diagnostic = EntitySemanticF.load(output/'diagnostics/candidate-model.json')
        evaluation = {'selected_f': _evaluate(cases, snapshots, bank, selected, splits),
                      'diagnostic_f': _evaluate(cases, snapshots, bank, diagnostic, splits)}
        write(output/'evaluation.json', evaluation)
        old_audits = {json.loads(line)['sample_id']: json.loads(line)['audit']
                      for line in OLD_BRIDGE.read_text(encoding='utf-8').splitlines()}
        old_collisions = _collision_audit(cases, snapshots, bank, selected, diagnostic, old_audits)
        jsonl(output/'entity-collision-audit.jsonl', old_collisions)
        selected_before, selected_audits, selected_targets, split_rows = _transition_rows(
            cases, snapshots, bank, selected, splits, old_audits)
        jsonl(output/'selected-f-pre-read.jsonl', selected_before)
        jsonl(output/'selected-f-post-read-audits.jsonl', selected_audits)
        jsonl(output/'selected-f-hindsight-targets.jsonl', selected_targets)
        jsonl(output/'physical-match-splits.jsonl', split_rows)
        diagnostic_before, diagnostic_audits, diagnostic_targets, _ = _transition_rows(
            cases, snapshots, bank, diagnostic, splits, old_audits)
        jsonl(output/'diagnostics/diagnostic-f-pre-read.jsonl', diagnostic_before)
        jsonl(output/'diagnostics/diagnostic-f-post-read-audits.jsonl', diagnostic_audits)
        jsonl(output/'diagnostics/diagnostic-f-hindsight-targets.jsonl', diagnostic_targets)
        # Both trajectories use bounded unknown exploration; neither claims a
        # numeric g estimate. The selected f is the only model bound in the g
        # consumer contract.
        traces = []
        for label, model in (('selected_f', selected), ('diagnostic_f', diagnostic)):
            for mid in ('understat:28986', 'understat:29015', 'understat:29038'):
                case = next(c for c in cases if c['fixture']['physical_match_id'] == mid)
                trace = episode(case, model=model, bank=bank, prior=snapshots[mid], seed=20261011,
                    request_limit=2, query_protocol=entity_query)
                replay = episode(case, model=EntitySemanticF.load(output/'model.json' if label=='selected_f'
                    else output/'diagnostics/candidate-model.json'), bank=EntityEvidenceBank(read(RECORDS)),
                    prior=read(PRIORS)[mid], seed=20261011, request_limit=2, query_protocol=entity_query)
                if trace != replay:
                    raise AssertionError('Entity offline trajectory replay differs')
                traces.append({'model': label, 'physical_match_id': mid, 'trace': trace})
        write(output/'offline-decision-trajectories.json', traces)
        contract = {'schema': 'entity-semantic-f-g-consumer-v1',
            'input_schema': entity_query.VERSION, 'model_version': selected.model_version,
            'predictor_payload_sha256': _digest(selected.payload()),
            'representation_sha256': REPRESENTATION_SHA256,
            'f_fit_physical_match_ids': splits['A'], 'gain_fit_recommended_groups': ['B'],
            'development_evaluation_groups': ['C'], 'independent_blind_test': False,
            'actor_input_only': 'selected-f-pre-read.jsonl',
            'supervisor_only': ['selected-f-hindsight-targets.jsonl', 'selected-f-post-read-audits.jsonl'],
            'diagnostic_candidate_excluded': True, 'target': 'conditional_frozen_predictor_log_loss_change',
            'numeric_g_fit': False, 'prospective_eligible': False,
            'retraining_required_if_f_or_representation_changes': True}
        write(output/'g-consumer-contract.json', contract)
        selected_interpretations = Counter(x['audit']['interpretation'] for x in selected_audits)
        diagnostic_interpretations = Counter(x['audit']['interpretation'] for x in diagnostic_audits)
        summary = {'schema': 'entity-semantic-f-report-v1',
            'selected_model_version': selected.model_version,
            'diagnostic_model_version': diagnostic.model_version,
            'selected_regularization': selected.metadata['selection']['selected_regularization'],
            'diagnostic_regularization': diagnostic.metadata['applied_regularization'],
            'semantic_records': len(records), 'A_training_states': len(training),
            'entity_representation_sha256': REPRESENTATION_SHA256,
            'old_binary_collision_transitions': len(old_collisions),
            'selected_entity_vector_distinguished': sum(x['selected_f']['feature_vector_changed'] for x in old_collisions),
            'diagnostic_entity_vector_distinguished': sum(x['diagnostic_f']['feature_vector_changed'] for x in old_collisions),
            'selected_forecast_changed_on_old_collisions': sum(x['selected_f']['max_probability_change'] > 1e-12 for x in old_collisions),
            'diagnostic_forecast_changed_on_old_collisions': sum(x['diagnostic_f']['max_probability_change'] > 1e-12 for x in old_collisions),
            'selected_transitions': len(selected_before), 'selected_zero_gain_targets': sum(
                abs(x['target']['realized_gain_nats']) <= 1e-12 for x in selected_targets),
            'diagnostic_nonzero_gain_targets': sum(abs(x['target']['realized_gain_nats']) > 1e-12 for x in diagnostic_targets),
            'selected_interpretations': dict(selected_interpretations),
            'diagnostic_interpretations': dict(diagnostic_interpretations),
            'offline_HTTP_requests': 0, 'g_models_trained': 0, 'prospective_training_eligible_rows': 0,
            'old_f_and_old_bridge_unchanged': True,
            'scope': 'representation_and_f_g_contract_research_not_accuracy_or_live_VoI_validation',
            'limitations': ['A-only sample remains small and Leeds-centered.',
                'Source quotes establish provenance, not truth or causal player importance.',
                'Selected A-only rule retained prior-only because nonzero entity candidate was not selected.',
                'Diagnostic nonzero candidate is excluded from g consumer and live use.',
                'All labels are historical reconstruction; empirical expected gain and harm risk remain unknown.']}
        write(output/'report.json', summary)
        (output/'review-report.txt').write_text(
            'FootballMind 实体级 f 实验审阅报告\n\n'
            f"正式选择：{selected.model_version}\n"
            f"A-only 选择的正则：{selected.metadata['selection']['selected_regularization']}（prior-only 时保持不变）\n"
            f"诊断候选：{diagnostic.model_version}，正则 {diagnostic.metadata['applied_regularization']}；不进入 g 或实时路径。\n\n"
            f"旧二值碰撞转移：{len(old_collisions)} 条；实体计数输入被区分：{summary['selected_entity_vector_distinguished']} 条；"
            f"诊断 f 的概率实际变化：{summary['diagnostic_forecast_changed_on_old_collisions']} 条。\n"
            f"正式 f 的转移标签：{len(selected_targets)} 条，其中零标签 {summary['selected_zero_gain_targets']} 条；"
            f"诊断 f 非零标签 {summary['diagnostic_nonzero_gain_targets']} 条。\n\n"
            '解释：实体只作为来源谓词的去重键和计数单位，未赋予球员身价、位置或重要性。'
            '同一实体的 contradictory 状态保留，不进行最后写入覆盖。未知未来实体也能计入计数，但没有个人身份系数。\n'
            '正式模型仍按 A-only forward one-standard-error 规则选择；验证没有支持将非零候选提升为正式 f。'
            '因此 g consumer 绑定正式 f，诊断轨迹仅用于证明表示层修复。\n', encoding='utf-8')
        receipt = _write_receipt(output)
        print(json.dumps({'output': str(output), 'receipt': receipt, 'summary': summary}, ensure_ascii=False, indent=2))
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo, socket.create_connection = original


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(OUTPUT_DEFAULT))
    run(parser.parse_args().output)
