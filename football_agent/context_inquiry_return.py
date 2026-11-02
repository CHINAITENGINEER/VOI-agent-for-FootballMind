'''Return audited inquiry observations to planning without asserting football facts.

This adapter exposes what was acquired and extracted by a cutoff. Semantic
correctness, freshness, truth, prediction and information value remain separate.
'''
from copy import deepcopy
import json
from pathlib import Path
from uuid import UUID

from .audit import RunStore, digest, stored_digest
from .autonomous_context_loop import LoopResult
from .contextual_adaptive_voi import normalize_context
from .inquiry_contract import utc
from .inquiry_evidence_state import EvidenceStateProtocol, project_requirements
from .return_dataset import export_run
from football_research.evaluation.policy_experiment_protocol import audit_parameter_freeze

VERSION = 'context-inquiry-return-v1'
NAMESPACE = 'inquiry.'


def project_handoff_context(*, handoff_root, handoff_id, parent, reference, as_of, freeze_manifest):
    '''Project an audited causal prefix, bound to its original zero-query parent.

Only the reserved inquiry.* namespace changes. Hypothesis definitions travel
with their observations; they are not mapped to domain truths such as qualified
or will_rotate. Existing numerical policies must explicitly opt in to the new
keys and obtain suitable calibration before claiming decision value.
'''
    if not isinstance(parent, LoopResult):
        raise ValueError('An original LoopResult parent is required')
    identity = str(UUID(handoff_id))
    if identity != handoff_id:
        raise ValueError('Canonical handoff identity required')
    path = Path(handoff_root).resolve() / (identity + '.json')
    envelope = json.loads(path.read_text(encoding='utf-8'))
    receipt = {k: v for k, v in envelope.items() if k != 'record_hash'}
    if stored_digest(receipt) != envelope.get('record_hash'):
        raise ValueError('Handoff receipt hash mismatch')
    binding = receipt['binding']
    if (receipt['handoff_id'] != identity or binding['parent_result_hash'] != digest(parent.as_dict())
            or binding['parent_target_key'] != parent.target_key
            or binding['parent_episode_id'] != parent.episode_id):
        raise ValueError('Handoff does not bind the supplied parent')
    if receipt.get('run_id') != reference.get('run_id'):
        raise ValueError('Reference is not the bound child inquiry')
    root = Path(reference['root']).resolve()
    if (root / 'runs').resolve() != Path(binding['inquiry_store']).resolve():
        raise ValueError('Reference storage differs from the handoff')
    cutoff = utc(parent.decisions[-1]['plan']['board']['prediction_cutoff'])
    freeze_audit = audit_parameter_freeze(freeze_manifest, earliest_test_prediction_cutoff=cutoff.isoformat())
    config = freeze_manifest['parameters']['context_inquiry_handoff']
    if (freeze_audit['status'] != 'PASS' or freeze_manifest['parameter_hash'] != binding['parameter_hash']
            or digest(config) != binding['configuration_hash']):
        raise ValueError('Original frozen handoff configuration required')
    selected_time = utc(as_of)
    if not utc(binding['context_available_at']) <= selected_time <= cutoff:
        raise ValueError('Projection time must fall within the original context and prediction cutoff')
    initial = normalize_context(parent.final_context)
    if any(key.startswith(NAMESPACE) for key in initial):
        raise ValueError('Initial parent context occupies the reserved inquiry namespace')
    # Reuse the full existing quote, source, protocol, selection and raw archive
    # replay checks. Export also validates evidence-origin and sampling claims.
    rows, source = export_run(reference, as_of=as_of)
    run = RunStore(root / 'runs').load(reference['run_id'])
    if not run['events'] or run['events'][-1]['event_hash'] != reference['head_hash']:
        raise ValueError('Inquiry advanced after verification; pin its current head and retry')
    if digest(run['task']) != binding['task_hash'] or run['task']['task_id'] != identity:
        raise ValueError('Child task differs from the original handoff')
    if digest(run['configuration']) != config['workflow_configuration_hash']:
        raise ValueError('Child configuration differs from its bound workflow')
    if any(run['budget'][key] != config[key] for key in ('max_requests', 'max_llm_calls')):
        raise ValueError('Child budget differs from the frozen handoff')
    if utc(freeze_manifest['frozen_at']) > utc(run['events'][0]['at']):
        raise ValueError('Configuration was not frozen before the child inquiry')
    if utc(run['task']['deadline']) != cutoff:
        raise ValueError('Child deadline differs from the parent cutoff')
    if run['configuration'].get('inquiry_protocol') != EvidenceStateProtocol.version:
        raise ValueError('Evidence-state protocol is required for this projection')
    eligible = []
    previous = None
    for event in run['events']:
        stamp = utc(event['at'])
        if previous is not None and stamp < previous:
            raise ValueError('Event clock moved backwards')
        previous = stamp
        if stamp <= selected_time:
            eligible.append(event)
    if not eligible:
        raise ValueError('No inquiry event was available at the requested time')
    event = eligible[-1]
    snapshot = event['snapshot']
    for document in snapshot['documents']:
        if utc(document['available_at']) > selected_time:
            raise ValueError('Snapshot contains a future document')
    hypotheses = snapshot.get('hypotheses', [])
    claims = snapshot['claims']
    requirements = project_requirements(hypotheses, claims, snapshot.get('context_bindings', []))
    if requirements != snapshot.get('requirements', []):
        raise ValueError('Evidence state differs from the replayed source projection')
    updates = {
        NAMESPACE + 'run_id': run['run_id'], NAMESPACE + 'state': snapshot['state'],
        NAMESPACE + 'reason': snapshot['reason'],
        NAMESPACE + 'http_attempts': snapshot['budget']['http_attempts'],
        NAMESPACE + 'llm_calls': snapshot['budget']['llm_calls'],
        NAMESPACE + 'document_count': len(snapshot['documents']),
        NAMESPACE + 'claim_count': len(claims),
    }
    states = []
    by_id = {item['hypothesis_id']: item for item in hypotheses}
    for requirement in requirements:
        hypothesis = by_id[requirement['hypothesis_id']]
        prefix = NAMESPACE + 'hypothesis.' + digest(hypothesis) + '.'
        observed = {
            'source_answer': requirement['source_answer_status'],
            'source_direction': requirement['proposition_direction'],
            'conflict': requirement['conflict_status'],
            'missing_types': ','.join(sorted(requirement['missing_types'])),
            'contested_types': ','.join(sorted(requirement['contested_types'])),
            'query_focus': requirement['query_focus'],
            'semantic_verification': 'unverified', 'freshness_verification': 'unverified',
        }
        updates.update({prefix + key: value for key, value in observed.items()})
        relevant = [c for c in claims if any(link['hypothesis_id'] == hypothesis['hypothesis_id']
                                           for link in c['hypothesis_links'])]
        states.append({'context_key_prefix': prefix, 'hypothesis': deepcopy(hypothesis),
            'requirements': deepcopy(requirement), 'claims': deepcopy(relevant),
            'source_truth_label': None, 'expected_information_value': None})
    following = normalize_context({**initial, **updates})
    return {
        'schema_version': VERSION, 'parent_episode_id': parent.episode_id,
        'target_key': parent.target_key, 'handoff_id': identity, 'reference': deepcopy(reference),
        'as_of': selected_time.isoformat(), 'evidence_available_at': event['at'],
        'prefix_event_hash': event['event_hash'], 'prefix_event_sequence': event['seq'],
        'previous_context': initial, 'next_context': following,
        'changed_keys': sorted(updates), 'evidence_states': states,
        'documents': [{key: d.get(key) for key in ('document_id', 'url', 'raw_sha256',
            'observed_at', 'available_at', 'source_published_at', 'publication_status', 'status')}
            for d in snapshot['documents']],
        'query_rows': rows, 'source': source,
        'structural_verification': 'existing_inquiry_quote_scope_time_and_archive_replay',
        'semantic_verification': 'unverified', 'prediction_updated': False,
        'domain_truth_updated': False, 'expected_information_value': None,
        'realized_decision_utility': None, 'new_network_requests': 0,
    }
