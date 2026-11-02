'''Task-bound autonomous acquisition with explicit, prospective information tests.

The tests are qualitative hypotheses, not calibrated VoI estimates. Execution,
source admission, budgets and journaling reuse the existing navigation workflow.
'''
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from jsonschema import Draft202012Validator

from .audit import RunStore, digest
from .evidence_windows import EvidenceWindows
from .fixture_bound_review import task_contract
from .information_navigation import (InformationNavigationWorkflow, SCHEMA as BASE_SCHEMA,
    PROMPT as BASE_PROMPT, TEXT, OPTIONAL)
from .information_navigation_verify import verify_navigation
from .inquiry_contract import obj, utc
from .inquiry_tools import public_url

REQUIREMENTS = ['fixture_identity', 'reference_time_applicability', 'all_fixture_order',
                'kickoff_date_or_time', 'match_status', 'evidence_conflict']
INDICES = {'type': 'array', 'maxItems': 3, 'uniqueItems': True,
           'items': {'type': 'integer', 'minimum': 0, 'maximum': 2}}
EXTENSIONS = obj({
    'task_contract_digest': TEXT,
    'target_state': obj({'status': {'enum': ['unresolved', 'candidate_only', 'proposed_target']},
        'candidate_label': OPTIONAL, 'support_claim_indices': INDICES,
        'open_requirements': {'type': 'array', 'maxItems': len(REQUIREMENTS), 'uniqueItems': True,
                              'items': {'enum': REQUIREMENTS}}}),
    'information_test': {'anyOf': [{'type': 'null'}, obj({
        'requirement': {'enum': REQUIREMENTS}, 'useful_if': TEXT, 'irrelevant_if': TEXT})]},
    'return_assessment': {'anyOf': [{'type': 'null'}, obj({
        'document_id': TEXT, 'requirement': {'enum': REQUIREMENTS},
        'outcome': {'enum': ['unread', 'partial', 'resolved', 'irrelevant', 'conflicting', 'failed']},
        'reason': TEXT, 'evidence_claim_indices': INDICES})]}})['properties']
SCHEMA = obj({**deepcopy(BASE_SCHEMA['properties']), **EXTENSIONS})
PROMPT = BASE_PROMPT + '''
Keep task_contract and its reference_time FIXED across this episode. question may focus an
intermediate unknown but may not substitute a convenient candidate for the original target.
Echo task_contract_digest. target_state is your provisional judgment, NOT verified world truth.
unresolved has candidate_label=null. candidate_only/proposed_target need a label and at least
one supporting claim index (zero-based in THIS response's claims). proposed_target requires
no open_requirements and evidence of the current/immediate-next fixture across all competitions,
home AND away, applicable at reference_time. Use explicit local date/time context or fixture IDs
where supported; do not collect arbitrary fields or insist on an unsupported UTC offset.
Open requirements are task dimensions, not a required tool sequence. Choose which uncertainty
matters now. A candidate-only state still has at least one open requirement.

Before read_window/query_source, information_test states ONE current open requirement,
useful_if (what answer could change the target decision) and irrelevant_if (what would NOT).
These are qualitative conditional hypotheses. Never fabricate expected error reduction or VoI.
For stop/defer information_test=null. stop requires proposed_target; defer may retain uncertainty.

When last_action_result exists, return_assessment must address its SAME document and prior
information_test.requirement. After a fetch, metadata/previews are unread (or failed); they do
not resolve the hypothesis until a selected body is read. After a body read, cite exact opened
body claims for partial/resolved/irrelevant/conflicting. resolved means this ONE requirement,
not necessarily the full task. A failed return is missing information, not evidence of absence.
Use null return_assessment on the first turn. Keep explanations concise (about 80 Chinese
characters each). Defer is legitimate; no quota or prescribed page order.
'''


class TargetNavigationProtocol:
    version, schema, prompt = 'target-information-navigation-v1', SCHEMA, PROMPT


def navigation_handoff(root, run_id, expected_head):
    proof = verify_navigation(root, run_id)
    store = RunStore(Path(root) / 'runs')
    run = store.load(run_id)
    if proof['audit']['head_hash'] != expected_head:
        raise ValueError('Navigation parent head changed')
    if run['state'] not in {'STOPPED', 'DEFERRED'} or run['pending'] or not run['decisions']:
        raise ValueError('Require resolved terminal navigation parent')
    documents = [store.get_blob(run_id, run['documents'][key]) for key in run['document_order']]
    return {'version': 'target-navigation-handoff-v1', 'parent_root': str(Path(root).resolve()),
        'parent_run_id': run_id, 'parent_head_hash': expected_head,
        'parent_completed_at': run['events'][-1]['at'], 'documents': documents,
        'document_digests': {d['document_id']: digest(d) for d in documents},
        'opened': deepcopy(run['opened']), 'claims': deepcopy(run['claims']),
        'last_choice': deepcopy(run['decisions'][-1]['choice']),
        'parent_budget': deepcopy(run['budget']),
        'scope': 'Prior actor proposal and available evidence; no retrospective reviewer labels.'}


def validate_contract(contract, task):
    expected = task_contract(**{key: contract[key] for key in
        ('task_id', 'question', 'subject_team', 'team_aliases', 'reference_time')})
    if expected != contract or contract['task_id'] != task['task_id'] or contract['question'] != task['objective']:
        raise ValueError('Task contract identity/objective changed')


class TargetNavigationWorkflow(InformationNavigationWorkflow):
    version = 'target-information-navigation-workflow-v1'

    def configuration(self):
        result = super().configuration()
        result.update(protocol=TargetNavigationProtocol.version,
                      protocol_digest=digest({'schema': SCHEMA, 'prompt': PROMPT}))
        return result

    def start(self, task, handoff, contract, *, max_llm_calls=5, max_http_attempts=2, max_window_reads=3):
        if set(task) != {'task_id', 'objective', 'deadline', 'seed_urls'}:
            raise ValueError('Explicit task identity, objective, deadline and seed URLs required')
        validate_contract(contract, task)
        now, reference = utc(self.clock()), utc(contract['reference_time'])
        if not utc(handoff['parent_completed_at']) <= reference <= now < utc(task['deadline']):
            raise ValueError('New episode reference/source timing invalid')
        if type(max_llm_calls) is not int or not 2 <= max_llm_calls <= 5:
            raise ValueError('Model budget must be 2..5')
        if type(max_http_attempts) is not int or not 0 <= max_http_attempts <= 2:
            raise ValueError('Network budget must be 0..2')
        if type(max_window_reads) is not int or not 1 <= max_window_reads < max_llm_calls:
            raise ValueError('Reserve reconsideration after reading')
        docs = handoff['documents']
        ids = [d['document_id'] for d in docs]
        if not docs or len(ids) != len(set(ids)) or set(ids) != set(handoff['opened']) or set(ids) != set(handoff['document_digests']):
            raise ValueError('Every unique inherited document needs bound reading coverage')
        for doc in docs:
            if digest(doc) != handoff['document_digests'][doc['document_id']] or utc(doc['available_at']) > reference:
                raise ValueError('Invalid inherited document binding/time')
            if doc['kind'] == 'page' and doc['status'] == 'OK' and doc.get('text'):
                EvidenceWindows(doc, window_chars=self.window_chars, preview_chars=self.preview_chars).public(handoff['opened'][doc['document_id']])
            elif handoff['opened'][doc['document_id']]:
                raise ValueError('Non-readable document has body coverage')
        task = deepcopy(task)
        task['seed_urls'] = [public_url(u, self.tools.allowed_hosts) for u in task['seed_urls']]
        identity = str(uuid4())
        run = {'run_id': identity, 'task': task, 'configuration': self.configuration(),
            'target_contract': deepcopy(contract), 'target_contract_digest': digest(contract),
            'handoff_ref': self.store.put_blob(identity, handoff),
            'documents': {d['document_id']: self.store.put_blob(identity, d) for d in docs},
            'document_order': ids, 'opened': deepcopy(handoff['opened']),
            'claims': deepcopy(handoff['claims']), 'decisions': [], 'receipts': [],
            'state': 'READY', 'reason': 'STARTED', 'pending': None,
            'budget': {'max_llm_calls': max_llm_calls, 'max_http_attempts': max_http_attempts,
                'max_window_reads': max_window_reads, 'llm_calls': 0, 'http_attempts': 0,
                'window_reads': 0, 'delivered_body_chars': 0, 'llm_usage': [], 'actual_money': None}, 'events': []}
        with self.store.lock(identity):
            self._event(run, 'navigation_started', {'handoff_ref': run['handoff_ref'],
                'new_source_observations': 0, 'parent_head_hash': handoff['parent_head_hash']})
        return deepcopy(run)

    def public(self, run, *, now=None):
        validate_contract(run['target_contract'], run['task'])
        if digest(run['target_contract']) != run['target_contract_digest']:
            raise ValueError('Task contract digest changed')
        result = super().public(run, now=now)
        result.update(task_contract=deepcopy(run['target_contract']),
                      task_contract_digest=run['target_contract_digest'], last_action_result=None)
        if run['decisions']:
            last = run['decisions'][-1]['choice']
            if last['action'] in {'read_window', 'query_source'}:
                key = last['document_id'] if last['action'] == 'read_window' else run['receipts'][-1]['document_id']
                result['last_action_result'] = {'document_id': key,
                    'exposure': 'body' if last['action'] == 'read_window' else 'metadata',
                    'window_id': last['window_id'] if last['action'] == 'read_window' else None,
                    'status': self.documents(run)[key]['status'],
                    'information_test': deepcopy(last['information_test'])}
        return result

    def validate(self, choice, run, public):
        Draft202012Validator(SCHEMA).validate(choice)
        if choice['task_contract_digest'] != run['target_contract_digest']:
            raise ValueError('Response changed task contract')
        base = {key: choice[key] for key in BASE_SCHEMA['properties']}
        claims, prepared = super().validate(base, run, public)
        state = choice['target_state']
        indices = state['support_claim_indices']
        if any(index >= len(claims) for index in indices):
            raise ValueError('Target support index outside this response claims')
        if state['status'] == 'unresolved':
            if state['candidate_label'] is not None or not state['open_requirements']:
                raise ValueError('Unresolved target must retain explicit unknowns')
        elif not state['candidate_label'] or not indices:
            raise ValueError('Candidate needs evidence support')
        if state['status'] == 'candidate_only' and not state['open_requirements']:
            raise ValueError('Candidate-only target still has unknown requirements')
        if state['status'] == 'proposed_target' and state['open_requirements']:
            raise ValueError('Proposed target still has open requirements')
        probe = choice['information_test']
        if choice['action'] in {'read_window', 'query_source'}:
            if probe is None or probe['requirement'] not in state['open_requirements']:
                raise ValueError('Action must test a current open requirement')
        elif probe is not None:
            raise ValueError('Terminal action has no acquisition hypothesis')
        if choice['action'] == 'stop' and state['status'] != 'proposed_target':
            raise ValueError('Stop requires provisional target; defer preserves unknowns')
        prior, assessment = public['last_action_result'], choice['return_assessment']
        if prior is None:
            if assessment is not None: raise ValueError('No returned action to assess')
        else:
            if assessment is None or assessment['document_id'] != prior['document_id'] or assessment['requirement'] != prior['information_test']['requirement']:
                raise ValueError('Assess the actual return against its prospective question')
            refs, outcome = assessment['evidence_claim_indices'], assessment['outcome']
            if any(index >= len(claims) or claims[index]['anchor']['document_id'] != prior['document_id']
                   or (prior['exposure'] == 'body' and claims[index]['anchor']['window_id'] != prior['window_id'])
                   for index in refs):
                raise ValueError('Return support must cite the returned document')
            if prior['status'] != 'OK':
                if outcome != 'failed' or refs: raise ValueError('Failed return remains missing information')
            elif prior['exposure'] == 'metadata':
                if outcome != 'unread' or refs: raise ValueError('Preview is not body evidence')
            elif outcome not in {'partial', 'resolved', 'irrelevant', 'conflicting'} or not refs:
                raise ValueError('Body assessment requires anchored support')
            if outcome == 'resolved' and assessment['requirement'] in state['open_requirements']:
                raise ValueError('Resolved requirement remains open')
        return claims, prepared
