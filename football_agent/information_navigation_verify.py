'''Offline reconstruction of information navigation; never calls model or network.'''
from copy import deepcopy
import json
from pathlib import Path

from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from .audit import RunStore, digest
from .evidence_windows import EvidenceWindows
from .information_navigation import InformationNavigationWorkflow, NavigationProtocol, SCHEMA, PROMPT, reading_handoff
from .inquiry_contract import utc
from .inquiry_tools import OpenWebTools
from .inquiry_transport import RoutedOpenWebTools, request_header_profile
from .inquiry_verify import canonical_query_url


def verify_acquisition(root, run_id, config, result, action, intent_at, returned_at):
    archive = RawArchive(Path(root) / 'evidence')
    observation = Observation.model_validate(result['observation'])
    archive.read_body(observation)
    payload = archive.root / observation.payload_path
    metadata = payload.with_name(payload.name[:-len('.body.gz')] + '.observation.json')
    if Observation.model_validate_json(metadata.read_bytes()).model_dump() != observation.model_dump():
        raise ValueError('Raw observation metadata changed')
    if observation.context.get('inquiry_run_id') != run_id or observation.context.get('inquiry_request_id') != result['request_id']:
        raise ValueError('Observation belongs to a different request')
    if any(getattr(observation,k) != action[k] for k in ('endpoint','params','source')):
        raise ValueError('Observed source differs from selected action')
    policy = config['tools'].get('transport_policy')
    if policy:
        profile = request_header_profile(action['endpoint'])
        expected = {'version':policy['version'],'route':policy['route'],'policy_sha256':digest(policy),
            'tls_verification':True,'authentication':'none_no_netrc',
            'request_header_preset':profile['preset_id'],'request_headers':profile['headers']}
        if any(observation.context.get('transport',{}).get(k) != v for k,v in expected.items()):
            raise ValueError('Transport differs from registered route')
    if not utc(intent_at) <= utc(observation.context['requested_at']) <= utc(observation.observed_at) <= utc(result['document']['available_at']) <= utc(returned_at):
        raise ValueError('Source receipt violates acquisition time order')
    tools = OpenWebTools(archive, config['tools']['allowed_hosts'],
        clock=lambda: utc(result['document']['available_at']),
        extraction_version=1 if config['tools']['version'] == 'open-web-read-v1' else 2)
    extracted, recorded = tools.project(observation,action), deepcopy(result['document'])
    extracted['url'], recorded['url'] = canonical_query_url(extracted['url']), canonical_query_url(recorded['url'])
    if extracted != recorded: raise ValueError('Document differs from original raw response extraction')
    return deepcopy(result['document'])


def verify_navigation(root, run_id):
    root = Path(root)
    store = RunStore(root / 'runs')
    audit, run = store.verify(run_id), store.load(run_id)
    initial = run['events'][0]['snapshot']
    config = initial['configuration']
    if config['version'] != InformationNavigationWorkflow.version or config['protocol'] != NavigationProtocol.version or config['protocol_digest'] != digest({'schema':SCHEMA,'prompt':PROMPT}):
        raise ValueError('Unknown navigation configuration')
    handoff = store.get_blob(run_id, initial['handoff_ref'])
    if handoff != reading_handoff(handoff['parent_root'], handoff['parent_run_id'], handoff['parent_head_hash']):
        raise ValueError('Inherited information need differs from verified parent')
    doc = handoff['source']['document']
    if initial['document_order'] != [doc['document_id']] or set(initial['documents']) != {doc['document_id']} or store.get_blob(run_id,initial['documents'][doc['document_id']]) != doc:
        raise ValueError('Initial document differs from inherited archive')
    if initial['opened'] != {doc['document_id']:handoff['opened']} or initial['claims'] != handoff['claims']:
        raise ValueError('Initial reading coverage or claims changed')
    if utc(handoff['parent_completed_at']) > utc(run['events'][0]['at']) or utc(run['events'][0]['at']) >= utc(initial['task']['deadline']):
        raise ValueError('Invalid handoff/start time')
    if initial['state'] != 'READY' or initial['reason'] != 'STARTED' or initial['decisions'] or initial['receipts'] or initial['pending']:
        raise ValueError('Nonempty initial action state')
    budget = initial['budget']
    if any(budget[k] != 0 for k in ('llm_calls','http_attempts','window_reads','delivered_body_chars')) or budget['llm_usage'] or budget['actual_money'] is not None:
        raise ValueError('Nonzero initial new-task costs')
    if not 2 <= budget['max_llm_calls'] <= 5 or not 0 <= budget['max_http_attempts'] <= 2 or not 1 <= budget['max_window_reads'] < budget['max_llm_calls']:
        raise ValueError('Invalid task limits')
    archive = RawArchive(root / 'evidence')
    policy = config['tools'].get('transport_policy')
    tools = (RoutedOpenWebTools(archive,config['tools']['allowed_hosts'],route=policy['route'])
        if policy else OpenWebTools(archive,config['tools']['allowed_hosts']))
    if tools.configuration() != config['tools']: raise ValueError('Tools configuration changed')
    flow = InformationNavigationWorkflow(root,None,tools)
    expected, phase, previous = deepcopy(initial), 'initial', None
    call_ids, request_ids, new_ids = set(), set(), set()
    returned, body_reads, input_sizes, returns_reconsidered, body_reconsidered = 0, 0, [], 0, 0
    reply = public = prepared = choice = intent_at = None
    declared_only = []
    for event in run['events']:
        kind, data = event['type'], event['data']
        if phase == 'finished': raise ValueError('Event follows terminal state')
        if previous and utc(event['at']) < utc(previous['at']): raise ValueError('Event time reversed')
        b = expected['budget']
        if kind == 'navigation_started':
            if event['seq'] != 1 or data != {'handoff_ref':initial['handoff_ref'],'new_source_observations':0,'parent_head_hash':handoff['parent_head_hash']}:
                raise ValueError('Start provenance changed')
            phase = 'ready'
        elif kind == 'navigation_reasoning_reserved':
            if phase != 'ready' or b['llm_calls'] >= b['max_llm_calls'] or utc(event['at']) >= utc(initial['task']['deadline']):
                raise ValueError('Invalid model reservation')
            public = store.get_blob(run_id,data['input_ref'])
            if public != flow.public(expected,now=public['now']):
                raise ValueError('Model input differs from previously visible evidence')
            if not utc(previous['at']) <= utc(public['now']) <= utc(event['at']): raise ValueError('Model input time changed')
            pending = event['snapshot']['pending']
            if set(pending) != {'kind','call_id','input_ref'} or pending['kind'] != 'llm' or pending['input_ref'] != data['input_ref'] or pending['call_id'] in call_ids:
                raise ValueError('Invalid pending model identity')
            call_ids.add(pending['call_id'])
            expected['pending'] = deepcopy(pending)
            b['llm_calls'] += 1
            input_sizes.append(len(json.dumps(public,ensure_ascii=False,separators=(',',':')).encode('utf-8')))
            phase = 'model_reserved'
        elif kind == 'navigation_reasoning_failed':
            if phase != 'model_reserved' or data['outcome'] != 'unknown': raise ValueError('Unexpected model failure')
            phase = 'model_failed'
        elif kind == 'navigation_reasoning_returned':
            if phase != 'model_reserved' or data['input_ref'] != expected['pending']['input_ref']: raise ValueError('Unbound model return')
            reply = store.get_blob(run_id,data['output_ref'])
            b['llm_usage'].append(reply.get('receipt',{}))
            expected['pending'], phase = None, 'returned'
            returned += 1
        elif kind == 'navigation_response_rejected':
            if phase != 'returned': raise ValueError('Unexpected response rejection')
            try: flow.validate(reply['response'],expected,public)
            except Exception: pass
            else: raise ValueError('Valid response labelled invalid')
            phase = 'rejected'
        elif kind == 'navigation_decision':
            if phase != 'returned' or store.get_blob(run_id,data['input_ref']) != public or store.get_blob(run_id,data['output_ref']) != reply:
                raise ValueError('Decision not bound to real input/return')
            choice = reply['response']
            claims, prepared = flow.validate(choice,expected,public)
            decision = {'choice':deepcopy(choice),'opened_before':deepcopy(expected['opened']),
                'claims':claims,'expected_gain':None,'selection_basis':'qualitative_bounded_navigation'}
            if data['decision'] != decision: raise ValueError('Decision changed after return')
            for claim in claims:
                if claim not in expected['claims']: expected['claims'].append(claim)
            expected['decisions'].append(decision)
            returns_reconsidered += bool(new_ids)
            body_reconsidered += any(expected['opened'].get(k) for k in new_ids)
            phase = 'decided'
        elif kind == 'navigation_window_opened':
            if phase != 'decided' or choice['action'] != 'read_window' or b['window_reads'] >= b['max_window_reads'] or b['llm_calls'] >= b['max_llm_calls']:
                raise ValueError('Read not admitted by chosen action/budget')
            source = flow.documents(expected)[choice['document_id']]
            window = EvidenceWindows(source).read(choice['window_id'])
            if data['document_id'] != source['document_id'] or store.get_blob(run_id,data['output_ref']) != window or data['new_http_attempts'] != 0:
                raise ValueError('Selected body or its cost changed')
            expected['opened'].setdefault(source['document_id'],[]).append(window['window_id'])
            b['window_reads'] += 1
            b['delivered_body_chars'] += len(window['text'])
            body_reads += 1
            phase = 'ready'
        elif kind == 'navigation_acquisition_reserved':
            if phase != 'decided' or choice['action'] != 'query_source' or data['action'] != prepared or data['request_id'] in request_ids or b['http_attempts'] >= b['max_http_attempts'] or b['llm_calls'] >= b['max_llm_calls']:
                raise ValueError('Query not admitted by chosen action/budget')
            if utc(event['at']) >= utc(initial['task']['deadline']): raise ValueError('Source dispatched after deadline')
            request_ids.add(data['request_id'])
            b['http_attempts'] += 1
            expected['pending'] = {'kind':'http','request_id':data['request_id'],'action':prepared,'execution_key':digest(prepared)}
            intent_at, phase = event['at'], 'source_reserved'
        elif kind == 'navigation_acquisition_failed':
            if phase != 'source_reserved' or data['outcome'] != 'unknown': raise ValueError('Unexpected source failure')
            phase = 'source_failed'
        elif kind == 'navigation_acquisition_returned':
            result = store.get_blob(run_id,data['output_ref'])
            if phase != 'source_reserved' or result['request_id'] != expected['pending']['request_id'] or result['http_attempts'] != 1:
                raise ValueError('Unbound source return')
            doc = verify_acquisition(root,run_id,config,result,prepared,intent_at,event['at'])
            if utc(event['at']) >= utc(initial['task']['deadline']): doc.update(status='LATE_RETURN',text='',links=[])
            identity = doc['document_id']
            if identity in expected['documents'] or data['document_id'] != identity: raise ValueError('Repeated source identity')
            doc_ref = event['snapshot']['documents'][identity]
            if store.get_blob(run_id,doc_ref) != doc: raise ValueError('Available document projection changed')
            expected['documents'][identity] = doc_ref
            expected['document_order'].append(identity)
            expected['opened'][identity] = []
            expected['receipts'].append({'action':deepcopy(choice['query']),'document_id':identity,'status':doc['status'],
                'result_ref':data['output_ref'],'execution_key':digest(prepared)})
            expected['pending'] = None
            new_ids.add(identity)
            phase = 'ready'
        elif kind == 'reading_finished':
            reason, state, valid = data['reason'], data['state'], False
            expected_state = 'DEFERRED'
            if reason in {'AGENT_STOP','AGENT_DEFER'}:
                valid = phase == 'decided' and choice['action'].upper() == reason[6:]
                expected_state = 'STOPPED' if reason == 'AGENT_STOP' else 'DEFERRED'
            elif reason == 'PROVIDER_FAILED_OR_RESPONSE_UNKNOWN': valid = phase == 'model_failed'
            elif reason == 'ACQUISITION_REQUIRES_RECONCILIATION': valid = phase == 'source_failed'
            elif reason == 'INVALID_NAVIGATION_RESPONSE': valid = phase == 'rejected'
            elif reason == 'NO_RECONSIDERATION_CAPACITY': valid = phase == 'decided' and b['llm_calls'] >= b['max_llm_calls']
            elif reason == 'HTTP_BUDGET_EXHAUSTED': valid = phase == 'decided' and choice['action'] == 'query_source' and b['http_attempts'] >= b['max_http_attempts']
            elif reason == 'READ_BUDGET_EXHAUSTED': valid = phase == 'decided' and choice['action'] == 'read_window' and b['window_reads'] >= b['max_window_reads']
            elif reason == 'MODEL_BUDGET_EXHAUSTED': valid = phase == 'ready' and b['llm_calls'] >= b['max_llm_calls']
            elif reason == 'UNKNOWN_CALL_REQUIRES_RECONCILIATION': valid = phase in {'model_reserved','source_reserved'} and expected['pending'] is not None
            elif reason in {'DEADLINE_REACHED','LATE_RESPONSE','LATE_ACQUISITION_RETURN'}:
                valid = utc(event['at']) >= utc(initial['task']['deadline']) and phase in {'ready','returned','decided'}
                expected_state = 'STOPPED'
            elif reason in {'CONFIGURATION_CHANGED','INVALID_ACQUISITION_RECEIPT','DUPLICATE_DOCUMENT_ID'}:
                valid = phase == ('ready' if reason == 'CONFIGURATION_CHANGED' else 'source_reserved')
                declared_only.append(reason)
            if not valid or state != expected_state or data['value_stop_proven'] is not False:
                raise ValueError('Unsupported terminal transition')
            expected.update(state=state,reason=reason)
            phase = 'finished'
        else: raise ValueError('Unknown navigation event')
        if event['snapshot'] != expected: raise ValueError('Snapshot differs from reconstructed navigation prefix')
        previous = event
    return {'status':'VERIFIED_NAVIGATION_NOT_INFORMATION_VALUE','audit':audit,
        'model_attempts':expected['budget']['llm_calls'],'completed_model_returns':returned,
        'source_attempt_reservations':expected['budget']['http_attempts'],
        'raw_source_returns_verified':len(new_ids),'new_window_reads':body_reads,
        'decisions_verified':len(expected['decisions']),'decisions_after_new_return':returns_reconsidered,
        'decisions_after_new_body_read':body_reconsidered,'input_json_utf8_bytes':input_sizes,
        'pending':expected['pending'],'declared_only_stop_conditions':declared_only,
        'numerical_VoI_verified':False,'independent_policy_benefit':None,'source_semantics_verified':False}
