'''Offline selection/evidence linkage inspection. No provider or network calls.'''
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from filelock import FileLock

from .acquisition_boundary import VERSION, replay_boundary, CandidateBatch
from .audit import stored_digest as digest
from .local_evidence import ACTION as LOCAL_QUERY, action_payload, project_query, capability as local_capability
from .return_observation import project_return_observation
from . import match_return_assessment as return_review
from . import runtime_source_claims as source_claims
from . import runtime_source_support as source_support
from . import runtime_exploration as exploration
from . import acquisition_admission as admission
from .locks import run_lock


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _allowed_policy_fingerprints(state):
    """Return the current identity plus identities explicitly migrated from."""
    current = state.get('config', {}).get('selection_policy')
    allowed = {current.get('fingerprint')} if isinstance(current, dict) else set()
    if not isinstance(current, dict):
        return allowed
    for event in state.get('events', []):
        if event.get('type') != 'selection_policy_migrated':
            continue
        data = event.get('data', {})
        if (isinstance(data, dict) and data.get('to_fingerprint') == current.get('fingerprint')
                and data.get('to_version') == current.get('version')
                and isinstance(data.get('from_fingerprint'), str)):
            allowed.add(data['from_fingerprint'])
    return allowed


def verify_run(folder, *, include_action_value_ledger=False):
    folder = Path(folder)
    # Replay observes one stable state and its archived operations. It never
    # waits for or interferes with an active source/provider call.
    with run_lock(folder/'execution.lock', folder.name, timeout=0):
        with FileLock(str(folder/'state.lock'), timeout=10):
            return _verify_run(folder, include_action_value_ledger=include_action_value_ledger)


def _verify_run(folder, *, include_action_value_ledger=False):
    from .experience import validate_decision, public_state
    folder=Path(folder)
    state=read(folder/'state.json')
    operations={o['operation_id']:o for o in state['operations']}
    traces={t['operation_id']:t for t in state.get('selection_trace',[])}
    decisions={}
    snapshots={}
    model_inputs={}
    return_contexts={}
    return_request_positions={}
    operation_positions={op_id: i for i, op_id in enumerate(operations)}
    allowed_policy_fingerprints = _allowed_policy_fingerprints(state)
    return_review.ensure_capability(state['config'])
    review_enabled=return_review.CAPABILITY_FIELD in state['config']
    claims_enabled=source_claims.CAPABILITY_FIELD in state['config']
    claim_records=[]
    exploration_ops=[]
    exploration_enabled = exploration.descriptor(state['config'].get('selection_policy')) is not None
    if claims_enabled:
        source_claims.ensure_capability(state['config'])
    support_contract = source_support.policy_contract(state['config'].get('selection_policy') or {})
    if state['config'].get(source_support.CAPABILITY_FIELD) != support_contract:
        raise ValueError('REPLAY_SOURCE_SUPPORT_POLICY_MISMATCH')
    support_snapshots = 0
    legacy, failed, pending=0,0,0
    fixture_identity_verified = 0
    for op_id, op in operations.items():
        if op['kind']!='model':
            continue
        if op.get('selection_boundary_version')!=VERSION:
            legacy+=1
            continue
        directory=folder/'operations'/op_id
        public=read(directory/'input.json')
        model_inputs[op_id]=public
        if public.get(return_review.CAPABILITY_FIELD) != state['config'].get(return_review.CAPABILITY_FIELD):
            raise ValueError('REPLAY_RETURN_CAPABILITY_MISMATCH')
        if public['task_id']!=state['run_id'] or digest(public)!=op['input_digest']:
            raise ValueError('REPLAY_INPUT_MISMATCH')
        if exploration_enabled:
            expected_exploration = exploration.project_state(state['config']['selection_policy'],
                exploration_ops, [item[0] for item in decisions.values()], state['config']['match_input'])
            if public.get(exploration.STATE_FIELD) != expected_exploration:
                raise ValueError('REPLAY_EXPLORATION_STATE_MISMATCH')
        elif exploration.STATE_FIELD in public:
            raise ValueError('REPLAY_UNEXPECTED_EXPLORATION_STATE')
        if public.get(source_claims.CAPABILITY_FIELD) != state['config'].get(source_claims.CAPABILITY_FIELD):
            raise ValueError('REPLAY_SOURCE_CLAIM_CAPABILITY_MISMATCH')
        if public.get(source_support.CAPABILITY_FIELD) != support_contract:
            raise ValueError('REPLAY_SOURCE_SUPPORT_CAPABILITY_MISMATCH')
        if support_contract is None and source_support.STATE_FIELD in public:
            raise ValueError('REPLAY_UNEXPECTED_SOURCE_SUPPORT_STATE')
        if claims_enabled:
            if public.get(source_claims.HISTORY_FIELD) != claim_records:
                raise ValueError('REPLAY_SOURCE_CLAIM_HISTORY_MISMATCH')
            if public.get(source_claims.STATE_FIELD) != source_claims.project_context(public):
                raise ValueError('REPLAY_SOURCE_CLAIM_STATE_MISMATCH')
            visible = [d for d in state['documents'] if d.get('selection_operation_id') in decisions]
            if public['documents'] != public_state(dict(state, documents=visible))['documents']:
                raise ValueError('REPLAY_SOURCE_CLAIM_VISIBLE_DOCUMENT_MISMATCH')
        identity_enabled = any((policy or {}).get('configuration', {}).get('fixture_bridge_version')
                               for policy in (public.get('selection_policy'), op.get('selection_policy'),
                                              state['config'].get('selection_policy')))
        if identity_enabled or 'fixture_identity' in public:
            from football_spider.storage.raw_archive import RawArchive
            from .runtime_fixture_bridge import (VERSION as BRIDGE_VERSION,
                                                  prepare_fixture_identity,
                                                  validate_fixture_identity)
            from .runtime_identity_discovery import identity_source_attempts
            if public.get('identity_source_attempts') != identity_source_attempts(
                    [item[0] for item in decisions.values()]):
                raise ValueError('REPLAY_IDENTITY_ATTEMPT_COUNT_MISMATCH')
            if public.get('match_input') != state['config'].get('match_input'):
                raise ValueError('REPLAY_IDENTITY_USER_INPUT_MISMATCH')
            try:
                clocks = [datetime.fromisoformat(value.replace('Z', '+00:00')) for value in
                          (state['created_at'], public['snapshot_as_of'], op['started_at'])]
                valid_clocks = (all(value.tzinfo is not None and value.utcoffset() is not None for value in clocks)
                                and clocks[0] <= clocks[1] <= clocks[2]
                                and public.get('reference_utc') == state['created_at'])
            except (KeyError, ValueError, TypeError, AttributeError):
                valid_clocks = False
            if not valid_clocks:
                raise ValueError('REPLAY_IDENTITY_CLOCK_MISMATCH')
            ids = [doc.get('document_id') for doc in public['documents']]
            originals = [doc for doc in state['documents']
                         if doc.get('selection_operation_id') in decisions]
            if ids != [doc.get('document_id') for doc in originals]:
                raise ValueError('REPLAY_IDENTITY_DOCUMENT_SET_MISMATCH')
            for doc in originals:
                if doc.get('selection_operation_id') not in decisions:
                    raise ValueError('REPLAY_IDENTITY_SOURCE_NOT_YET_SELECTED')
                source_ops = [item for item in operations.values()
                              if item.get('group_id') == doc.get('acquisition_id')]
                if not source_ops or any(operation_positions[item['operation_id']] >= operation_positions[op_id]
                                         for item in source_ops):
                    raise ValueError('REPLAY_IDENTITY_SOURCE_NOT_YET_AVAILABLE')
            validate_fixture_identity(public)
            expected_identity = prepare_fixture_identity(
                RawArchive(folder/'archive'), public, originals,
                include_local=public.get('fixture_identity', {}).get('version') == BRIDGE_VERSION)
            if public.get('fixture_identity') != expected_identity:
                raise ValueError('REPLAY_FIXTURE_IDENTITY_MISMATCH')
            fixture_identity_verified += 1
        if support_contract is not None:
            from football_spider.storage.raw_archive import RawArchive
            expected_support = source_support.prepare_support(RawArchive(folder/'archive'), public, originals)
            if public.get(source_support.STATE_FIELD) != expected_support:
                raise ValueError('REPLAY_SOURCE_SUPPORT_STATE_MISMATCH')
            support_snapshots += 1
        if 'result_digest' in op and digest(read(directory/'result.json'))!=op['result_digest']:
            raise ValueError('REPLAY_RAW_PROPOSAL_MISMATCH')
        if op['status']=='pending':
            pending+=1
            continue
        if op['status']=='failed':
            failed+=1
            continue
        # Responses discarded by cancellation/deadline do not become executed decisions.
        if 'decision_digest' not in op:
            continue
        proposal=read(directory/'proposal.json')
        selection=read(directory/'selection.json')
        if read(directory/'policy-output.json')!=selection['choice']:
            raise ValueError('REPLAY_RAW_POLICY_OUTPUT_MISMATCH')
        response=replay_boundary(public,proposal,selection)
        if digest(response)!=op['decision_digest'] or response!=read(directory/'decision.json'):
            raise ValueError('REPLAY_ACCEPTED_DECISION_MISMATCH')
        if selection['policy']!=op['selection_policy']:
            raise ValueError('REPLAY_SELECTED_POLICY_MISMATCH')
        if (state['config'].get('selection_policy') is not None
                and selection['policy'].get('fingerprint') not in allowed_policy_fingerprints):
            raise ValueError('REPLAY_RUN_POLICY_MISMATCH')
        trace=traces.get(op_id)
        if (not trace or trace['selection_digest']!=digest(selection)
                or trace['snapshot_digest']!=digest(public)
                or trace['policy']!=selection['policy']
                or trace['candidate_count']!=len(proposal['candidates'])):
            raise ValueError('REPLAY_SELECTION_TRACE_MISMATCH')
        validate_decision(response,public)
        exploration_batch = CandidateBatch.build(public,[c['response'] for c in proposal['candidates']],
            recommended_index=next((i for i,c in enumerate(proposal['candidates'])
                if c['candidate_id']==proposal['recommended_id']),None),origin=proposal['origin'])
        reservation = exploration.reservation(public, response, selection['choice'], op_id, batch=exploration_batch)
        if op.get(exploration.RESERVATION) != reservation:
            raise ValueError('REPLAY_EXPLORATION_RESERVATION_MISMATCH')
        exploration_ops.append(op)
        decisions[op_id]=(response,selection)
        snapshots[op_id]=public
        if claims_enabled:
            from football_spider.storage.raw_archive import RawArchive
            source_claims.verify_archived_claims(RawArchive(folder/'archive'),
                public, response[source_claims.FIELD], state['documents'])
            record=source_claims.claim_record(public,response,op_id)
            record_path=directory/'source-claims.json'
            if record is not None:
                if op.get('source_claim_digest')!=digest(record) or not record_path.exists() or read(record_path)!=record:
                    raise ValueError('REPLAY_SOURCE_CLAIM_RECORD_MISMATCH')
                claim_records.append(record)
            elif op.get('source_claim_digest') is not None or record_path.exists():
                raise ValueError('REPLAY_UNEXPECTED_SOURCE_CLAIM_RECORD')
    for op_id, op in operations.items():
        if exploration.RESERVATION in op and op_id not in decisions:
            raise ValueError('REPLAY_UNACCEPTED_EXPLORATION_RESERVATION')
    if claims_enabled:
        if state.get(source_claims.HISTORY_FIELD)!=claim_records:
            raise ValueError('REPLAY_SOURCE_CLAIM_FINAL_HISTORY_MISMATCH')
        for op_id, op in operations.items():
            if op['kind']=='model' and op_id not in decisions and (op.get('source_claim_digest')
                    or (folder/'operations'/op_id/'source-claims.json').exists()):
                raise ValueError('REPLAY_UNACCEPTED_SOURCE_CLAIM_RECORD')
    acquired=0
    http_bound=0
    platform_verified=0
    local_verified=0
    local_attempts=0
    local_operations=[o for o in operations.values() if o['kind']=='local_read']
    legacy_local = state['config'].get('local_query_capability') == local_capability(legacy=True)
    replay_local_capability = local_capability(legacy=legacy_local)
    if local_operations or 'local_query_capability' in state['config']:
        if (state['config'].get('local_query_capability') != replay_local_capability
                or state.get('local_usage',{}).get('reads') != len(local_operations)
                or len(local_operations)>local_capability()['max_reads']):
            raise ValueError('REPLAY_LOCAL_BUDGET_MISMATCH')
    for path in (folder/'acquisitions').glob('*/input.json'):
        linked=any(o.get('group_id')==path.parent.name and o.get('selection_operation_id') for o in operations.values())
        if (linked or state['config'].get('selection_boundary_version')==VERSION) and not (path.parent/'selection-link.json').exists():
            raise ValueError('REPLAY_MISSING_ACQUISITION_LINK')
    configured_admission_version = state['config'].get(admission.VERSION_FIELD)
    if configured_admission_version == admission.VERSION:
        state_admission_ids = {record.get('acquisition_id') for record in state.get(admission.STATE_FIELD, [])}
        sidecar_ids = {path.parent.name for path in (folder/'acquisitions').glob('*/admission.json')}
        link_ids = {path.parent.name for path in (folder/'acquisitions').glob('*/selection-link.json')}
        pending_link_ids = {operation.get('group_id') for operation in operations.values()
                            if operation.get('status') == 'pending' and operation.get('group_id')}
        if state_admission_ids != sidecar_ids or sidecar_ids != (link_ids - pending_link_ids):
            raise ValueError('REPLAY_RETURN_ADMISSION_ORPHAN')
    elif state.get(admission.STATE_FIELD):
        raise ValueError('REPLAY_UNEXPECTED_RETURN_ADMISSION_STATE')
    for path in (folder/'acquisitions').glob('*/selection-link.json'):
        link=read(path)
        configured_admission_version = state['config'].get(admission.VERSION_FIELD)
        link_admission_version = link.get(admission.VERSION_FIELD)
        if configured_admission_version == admission.VERSION and link_admission_version != admission.VERSION:
            raise ValueError('REPLAY_MISSING_RETURN_ADMISSION_MARKER')
        if link_admission_version is not None and link_admission_version != configured_admission_version:
            raise ValueError('REPLAY_RETURN_ADMISSION_VERSION_MISMATCH')
        item=decisions.get(link['operation_id'])
        if item is None:
            raise ValueError('REPLAY_ACQUISITION_WITHOUT_DECISION')
        response,selection=item
        action=read(path.parent/'input.json')
        expected=action_payload(response)
        if (action!=expected or digest(action)!=link['action_digest']
                or digest(response)!=link['decision_digest']
                or selection['choice']['candidate_id']!=link['candidate_id']):
            raise ValueError('REPLAY_ACQUISITION_LINK_MISMATCH')
        acquired+=1
        result_path=path.parent/'result.json'
        if result_path.exists():
            result=read(result_path)
            admission_record = None
            admission_path = path.parent / 'admission.json'
            selection_link_version = link.get(admission.VERSION_FIELD)
            if selection_link_version == admission.VERSION:
                if not admission_path.exists():
                    raise ValueError('REPLAY_MISSING_RETURN_ADMISSION')
                admission_record = read(admission_path)
                group_operations = [o for o in operations.values()
                                    if o.get('group_id') == path.parent.name]
                admitted = admission.verify_record(admission_record, state=state,
                    acquisition_id=path.parent.name, selection=link,
                    action=action, result=result, operations=group_operations)
            elif admission_path.exists():
                raise ValueError('REPLAY_UNEXPECTED_RETURN_ADMISSION')
            if review_enabled:
                context=return_review.bind_return(response, link, path.parent.name, result)
                if result.get('return_context') != context:
                    raise ValueError('REPLAY_RETURN_CONTEXT_MISMATCH')
                if link['operation_id'] in return_contexts:
                    raise ValueError('REPLAY_DUPLICATE_RETURN_FOR_SELECTION')
                if admission_record is None or admitted:
                    return_contexts[link['operation_id']]=context
                return_request_positions[link['operation_id']]=[operation_positions[o['operation_id']]
                    for o in operations.values() if o.get('group_id')==path.parent.name]
            elif 'return_context' in result:
                raise ValueError('REPLAY_RETURN_CONTEXT_WITHOUT_CAPABILITY')
            returned_ids={r['request_id'] for r in result['requests']}
            recorded_ids={o['operation_id'] for o in operations.values()
                          if o.get('group_id')==path.parent.name and o['kind']=='http'}
            if returned_ids!=recorded_ids or len(returned_ids)!=len(result['requests']):
                raise ValueError('REPLAY_HTTP_REQUEST_SET_MISMATCH')
            for request in result['requests']:
                op=operations.get(request['request_id'])
                if (not op or op['kind']!='http' or op.get('group_id')!=path.parent.name
                        or op.get('selection_operation_id')!=link['operation_id']
                        or op.get('receipt')!=request):
                    raise ValueError('REPLAY_HTTP_RECEIPT_MISMATCH')
                http_bound+=1
            document=result['document']
            if (admission_record is not None and not admitted
                    and response['action'] != LOCAL_QUERY
                    and admission_record.get('result_digest') != digest(result)):
                raise ValueError('REPLAY_RETURN_ADMISSION_RESULT_MISMATCH')
            local_ops=[o for o in local_operations if o.get('group_id')==path.parent.name]
            if response['action'] == LOCAL_QUERY:
                receipt=result.get('local_receipt')
                if (len(local_ops)!=1 or not receipt or local_ops[0].get('receipt')!=receipt
                        or receipt.get('operation_id')!=local_ops[0]['operation_id']
                        or local_ops[0].get('selection_operation_id')!=link['operation_id']
                        or local_ops[0].get('action_digest')!=digest(action)
                        or receipt.get('action_digest')!=digest(action)
                        or receipt.get('version')!=replay_local_capability['version']
                        or receipt.get('http_calls')!=0 or result['requests']
                        or receipt.get('status') not in {'returned','failed'}
                        or local_ops[0]['status']!=receipt['status']
                        or type(receipt.get('elapsed_seconds')) not in (float,int)
                        or not math.isfinite(receipt['elapsed_seconds']) or receipt['elapsed_seconds']<0
                        or (receipt['status']=='returned') != bool(document)
                        or receipt.get('document_digest')!=(digest(document) if document else None)):
                    raise ValueError('REPLAY_LOCAL_RECEIPT_MISMATCH')
                local_attempts+=1
                if document:
                    from football_spider.storage.raw_archive import RawArchive
                    parent_id=response['local_query']['document_id']
                    before=snapshots[link['operation_id']]['documents']
                    if parent_id not in {d['document_id'] for d in before}:
                        raise ValueError('REPLAY_LOCAL_PARENT_NOT_PREVIOUSLY_ACQUIRED')
                    parents=[d for d in state['documents'] if d['document_id']==parent_id]
                    reconstructed=project_query(RawArchive(folder/'archive'),parents,
                                                 response['local_query'],receipt['operation_id'],
                                                 legacy=legacy_local)
                    # return_observation is a runtime projection attached after
                    # the immutable local query is reconstructed. Verify both
                    # layers: a changed source projection and a changed
                    # technical observation must fail offline replay.
                    expected_reconstruction={k:v for k,v in document.items()
                                             if k != 'return_observation'}
                    if reconstructed!=expected_reconstruction:
                        raise ValueError('REPLAY_LOCAL_PROJECTION_MISMATCH')
                    expected_observation=project_return_observation(
                        reconstructed, action=response, acquisition_id=path.parent.name,
                        selection_operation_id=link['operation_id'])
                    if document.get('return_observation')!=expected_observation:
                        raise ValueError('REPLAY_RETURN_OBSERVATION_MISMATCH')
                    local_verified+=1
            elif local_ops or result.get('local_receipt'):
                raise ValueError('REPLAY_UNEXPECTED_LOCAL_RECEIPT')
            if document:
                if document.get('platform'):
                    from football_spider.storage.raw_archive import RawArchive
                    from .structured_sources import verify_platform_document
                    if document.get('redirect_chain') != result['requests']:
                        raise ValueError('REPLAY_PLATFORM_REQUEST_CHAIN_MISMATCH')
                    verify_platform_document(RawArchive(folder/'archive'),document)
                    platform_verified+=1
                expected_observation=project_return_observation(
                    document, action=response, acquisition_id=path.parent.name,
                    selection_operation_id=link['operation_id'])
                if document.get('return_observation')!=expected_observation:
                    raise ValueError('REPLAY_RETURN_OBSERVATION_MISMATCH')
                if admission_record is None or admitted:
                    expected_document={**document,'acquisition_id':path.parent.name,
                                       'selection_operation_id':link['operation_id']}
                    visible=[d for d in state['documents'] if d.get('acquisition_id')==path.parent.name]
                    if visible!=[expected_document]:
                        raise ValueError('REPLAY_VISIBLE_DOCUMENT_MISMATCH')
                else:
                    # Raw late returns stay archived, while actor documents and
                    # pending return state must remain untouched.
                    visible=[d for d in state['documents'] if d.get('acquisition_id')==path.parent.name]
                    if visible:
                        raise ValueError('REPLAY_REJECTED_RETURN_VISIBLE')
        elif any(o.get('group_id')==path.parent.name and o['status']=='returned' for o in operations.values()):
            raise ValueError('REPLAY_MISSING_ACQUISITION_RESULT')
    records=[]
    expected_pending=None
    if review_enabled:
        for op_id, public in model_inputs.items():
            if public.get(return_review.PENDING_FIELD) != expected_pending:
                raise ValueError('REPLAY_PENDING_RETURN_SNAPSHOT_MISMATCH')
            if public.get(return_review.HISTORY_FIELD) != records:
                raise ValueError('REPLAY_RETURN_HISTORY_SNAPSHOT_MISMATCH')
            if expected_pending:
                prior_id=expected_pending['selection_operation_id']
                if any(pos>=operation_positions[op_id] for pos in return_request_positions[prior_id]):
                    raise ValueError('REPLAY_RETURN_NOT_YET_AVAILABLE')
            if op_id not in decisions:
                if operations[op_id].get('return_assessment_digest'):
                    raise ValueError('REPLAY_UNACCEPTED_RETURN_ASSESSMENT')
                continue
            response=decisions[op_id][0]
            record=return_review.assessment_record(public,response,op_id)
            recorded_digest=operations[op_id].get('return_assessment_digest')
            record_path=folder/'operations'/op_id/'return-assessment.json'
            if record is not None:
                if recorded_digest!=digest(record) or not record_path.exists() or read(record_path)!=record:
                    raise ValueError('REPLAY_RETURN_ASSESSMENT_RECORD_MISMATCH')
                records.append(record)
                expected_pending=None
            elif recorded_digest is not None or record_path.exists():
                raise ValueError('REPLAY_UNEXPECTED_RETURN_ASSESSMENT_RECORD')
            if op_id in return_contexts:
                if expected_pending is not None:
                    raise ValueError('REPLAY_RETURN_OVERWRITTEN_BEFORE_ASSESSMENT')
                expected_pending=return_contexts[op_id]
        if state.get(return_review.HISTORY_FIELD)!=records:
            raise ValueError('REPLAY_RETURN_HISTORY_MISMATCH')
        if state.get(return_review.PENDING_FIELD)!=expected_pending:
            raise ValueError('REPLAY_PENDING_RETURN_STATE_MISMATCH')
        if state['turns']!=[item[0] for item in decisions.values()]:
            raise ValueError('REPLAY_RETURN_ACCEPTED_TURNS_MISMATCH')
    report={'run_id':state['run_id'],'accepted_decisions_verified':len(decisions),
            'linked_acquisitions':acquired,'linked_http_receipts':http_bound,
            'legacy_model_operations':legacy,'failed_model_operations':failed,
            'pending_model_operations':pending,'status':state['status'],
            'scope':'selection reconstruction and request linkage; not semantic truth or policy benefit',
            'real_calls_made_by_replay':0,'numeric_voi_calibrated':False}
    if fixture_identity_verified:
        report['fixture_identity_snapshots_verified'] = fixture_identity_verified
    if exploration_enabled:
        report['bounded_exploration'] = exploration.project_state(state['config']['selection_policy'],
            exploration_ops, [item[0] for item in decisions.values()], state['config']['match_input'])
    if claims_enabled:
        report.update(source_claim_records_verified=len(claim_records),
            source_assertions_verified=sum(len(record['assertions']) for record in claim_records),
            source_assertions_are_independent_gold=False)
    if support_contract is not None:
        report.update(source_support_snapshots_verified=support_snapshots,
                      source_support_semantics=source_support.SEMANTICS,
                      source_support_is_world_truth=False)
    if review_enabled:
        report.update(question_return_contexts_verified=len(return_contexts),
            question_return_assessments_verified=len(records),
            pending_question_return_assessment=expected_pending is not None,
            actor_assessments_are_independent_gold=False)
    if platform_verified:
        report['platform_documents_reconstructed']=platform_verified
    if local_operations:
        report['linked_local_reads']=local_attempts
        report['local_documents_reconstructed']=local_verified
        report['pending_local_reads']=sum(o['status']=='pending' for o in local_operations)
        report['failed_local_reads']=sum(o['status']=='failed' for o in local_operations)
    from .action_value_ledger import _project_verified_ledger
    from .value_evaluation_contract import ledger_to_evaluation_episodes, _digest as evaluation_digest
    ledger = _project_verified_ledger(folder, state)
    report['action_value_ledger_entries'] = ledger['summary']['entry_count']
    report['action_value_ledger_digest'] = ledger['ledger_digest']
    report['action_value_ledger_numeric_voi_calibrated'] = False
    episodes = ledger_to_evaluation_episodes(
        ledger, run_id=state.get('run_id'), prediction_cutoff=state.get('created_at'))
    report['value_evaluation_episode_count'] = len(episodes)
    report['value_evaluation_episode_digest'] = evaluation_digest(episodes)
    report['value_evaluation_numeric_voi_training_eligible'] = False
    if include_action_value_ledger:
        report['action_value_ledger'] = ledger
        report['value_evaluation_episodes'] = episodes
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify_run(args.run_directory),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
