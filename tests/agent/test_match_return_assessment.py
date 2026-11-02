from copy import deepcopy
from pathlib import Path
import json
import runpy

import pytest
from jsonschema import validate

from football_agent import match_return_assessment as review
from football_agent.acquisition_boundary import CandidateBatch, PolicyChoice, materialize_choice
from football_agent.acquisition_replay import verify_run
from football_agent.audit import atomic_write
from football_agent.experience import ExperienceService, public_state, validate_decision
from football_agent.match_proposals import MatchCandidateProvider, MatchLocalReturnProposalProtocol

h = runpy.run_path(str(Path(__file__).with_name('test_match_experience.py')))
match_decision, Provider, MatchTools = (h[k] for k in ('match_decision', 'Provider', 'MatchTools'))


def actual_review(public, outcome=None):
    context = public[review.PENDING_FIELD]
    if context is None:
        return None
    docs = [d for d in public['documents'] if d['document_id'] == context['document_id']]
    doc = docs[0] if docs else None
    if outcome is None:
        outcome = ('unavailable' if not doc or doc['status'] == 'unavailable' else
                   'discovery_only' if doc['content_kind'] == 'discovery' else
                   'partial' if doc['status'] == 'readable' else 'insufficient_context')
    return {'acquisition_id': context['acquisition_id'], 'question_id': context['question']['id'],
            'context_digest': context['context_digest'], 'outcome': outcome,
            'reason': '对照获取前判据：来源未完整回答本场正式阵容问题。',
            'citations': ([{'document_id': doc['document_id'], 'quote': doc['excerpt']}]
                          if outcome in review.SEMANTIC_OUTCOMES else []),
            'remaining_unknowns': [] if outcome == 'resolved' else ['本场正式首发尚待核实']}


def decision(public, action='stop'):
    result = match_decision(public, action)
    result[review.FIELD] = actual_review(public)
    return result


def fixture_runtime(root):
    seen = []
    def choose(public):
        seen.append(deepcopy(public))
        response = decision(public, ['search_web', 'read_url', 'stop'][min(len(public['documents']), 2)])
        if public['documents']:
            # Reusing an ID for a revised next question must not rewrite the
            # question or useful-return test of the preceding acquisition.
            response['analysis_plan'][0]['question'] = '新计划：本场是否存在轮换证据'
            response['useful_if'] = '正文列出本场轮换球员'
        return response
    service = ExperienceService(root, provider_factory=lambda f,e: Provider(choose),
        tools_factory=lambda f: MatchTools(), mode='test_fixture', enable_return_assessment=True)
    run = service.create({'match': '测试甲队 vs 测试乙队'})['run_id']
    return service, run, seen


def test_match_only_search_body_replan_stop_records_actual_returns(tmp_path):
    service, run, seen = fixture_runtime(tmp_path)
    try:
        service._drive(run, True)
        state = service.get(run)
        assert state['status'] == 'deferred'
        assert [t['action'] for t in state['turns']] == ['search_web', 'read_url', 'stop']
        assert state['spent'] == {'llm_calls': 3, 'http_calls': 2}
        assert not state['user_context'] and state['numeric_voi'] is None
        records = state[review.HISTORY_FIELD]
        assert [r['assessment']['outcome'] for r in records] == ['discovery_only', 'partial']
        assert records[0]['context']['question']['question'] == '确认具体比赛与阵容对应关系'
        assert records[0]['context']['useful_if'] == '具体比赛与正式首发可以对应'
        assert records[1]['context']['question']['question'] == '新计划：本场是否存在轮换证据'
        assert records[1]['context']['useful_if'] == '正文列出本场轮换球员'
        assert seen[0][review.PENDING_FIELD] is None
        assert seen[1][review.HISTORY_FIELD] == []
        assert seen[2][review.HISTORY_FIELD] == records[:1]
        assert state[review.PENDING_FIELD] is None
        assert all(r['world_truth_verified'] is False and r['realized_decision_gain'] is None
                   and r['numeric_voi_training_eligible'] is False for r in records)
        assert all(d['return_observation']['semantic_relevance'] == 'unassessed' for d in state['documents'])
        audit = verify_run(service.folder(run))
        assert audit['question_return_contexts_verified'] == audit['question_return_assessments_verified'] == 2
        assert not audit['pending_question_return_assessment']
        assert not audit['actor_assessments_are_independent_gold']
    finally:
        service.close()


@pytest.fixture
def body_public(tmp_path):
    service, run, seen = fixture_runtime(tmp_path)
    try:
        for _ in range(4):
            service.step(run)
        state = service.get(run)
        assert len(state['documents']) == 2 and state[review.PENDING_FIELD]
        yield service, run, public_state(state)
    finally:
        service.close()


@pytest.mark.parametrize('field,value,error', [
    ('acquisition_id', 'another-return', 'CONTEXT_MISMATCH'),
    ('question_id', 'another-question', 'CONTEXT_MISMATCH'),
    ('context_digest', 'another-test', 'CONTEXT_MISMATCH'),
    ('citations', [], 'REQUIRES_BODY_CITATION'),
    ('citations', [{'document_id': 'search-fixture', 'quote': 'Two fixtures'}], 'QUOTE_NOT_VISIBLE'),
    ('citations', [{'document_id': 'match-page', 'quote': 'Hidden formal lineup'}], 'QUOTE_NOT_VISIBLE'),
    ('outcome', 'resolved', 'UNKNOWNS_MISMATCH'),
    ('outcome', 'unavailable', 'NOT_UNAVAILABLE'),
    ('outcome', 'discovery_only', 'NOT_DISCOVERY'),
    ('outcome', 'conflicting', 'CONFLICT_REQUIRES_TWO_SPANS'),
    ('reason', '  ', 'BLANK_REASON'),
])
def test_reject_unbound_or_unsupported_return_judgements(body_public, field, value, error):
    _, _, public = body_public
    response = decision(public)
    response[review.FIELD][field] = value
    with pytest.raises(ValueError, match=error):
        validate_decision(response, public)


def test_discovery_cannot_supply_semantic_negative_or_positive(tmp_path):
    service, run, seen = fixture_runtime(tmp_path)
    try:
        service.step(run); service.step(run)
        public = public_state(service.get(run))
        for outcome in ['resolved', 'partial', 'irrelevant', 'conflicting']:
            response = decision(public)
            response[review.FIELD] = actual_review(public, outcome)
            with pytest.raises(ValueError, match='QUOTE_NOT_VISIBLE|REQUIRES_BODY'):
                validate_decision(response, public)
    finally:
        service.close()


def test_invalid_review_charged_once_preserves_pending_until_retry(body_public):
    service, run, public = body_public
    def invalid(p):
        response = decision(p)
        response[review.FIELD]['acquisition_id'] = 'wrong'
        return response
    service.provider_factory = lambda f,e: Provider(invalid)
    failed = service.step(run)
    assert failed['status'] == 'paused' and failed['spent']['llm_calls'] == 3
    assert failed[review.PENDING_FIELD] == public[review.PENDING_FIELD]
    assert len(failed[review.HISTORY_FIELD]) == 1
    assert verify_run(service.folder(run))['pending_question_return_assessment']
    service.provider_factory = lambda f,e: Provider(lambda p: decision(p))
    completed = service.step(run)
    assert completed['spent']['llm_calls'] == 4 and len(completed[review.HISTORY_FIELD]) == 2
    assert not verify_run(service.folder(run))['pending_question_return_assessment']


@pytest.mark.parametrize('failure', ['provider_error', 'cancel', 'deadline'])
def test_unaccepted_review_never_commits_semantic_history(body_public, failure):
    service, run, public = body_public
    def choose(p):
        if failure == 'provider_error':
            raise RuntimeError('fixture provider failed')
        if failure == 'cancel':
            service.cancel(run)
        else:
            service.mutate(run, lambda s: s.update(deadline='2020-01-01T00:00:00+00:00'))
        return decision(p)
    service.provider_factory = lambda f,e: Provider(choose)
    after = service.step(run)
    assert len(after[review.HISTORY_FIELD]) == 1
    assert after[review.PENDING_FIELD] == public[review.PENDING_FIELD]
    assert verify_run(service.folder(run))['question_return_assessments_verified'] == 1


def test_candidates_must_share_one_actual_return_judgement(body_public):
    _, _, public = body_public
    first, second = decision(public, 'stop'), decision(public, 'defer')
    second[review.FIELD]['reason'] = 'different judgement'
    with pytest.raises(ValueError, match='SHARED_ANALYSIS'):
        CandidateBatch.build(public, [first, second], origin='test', recommended_index=0)


@pytest.mark.parametrize('target', ['context', 'record', 'state_history', 'pending_state'])
def test_replay_rejects_modified_prospective_test_or_review(tmp_path, target):
    service, run, seen = fixture_runtime(tmp_path)
    try:
        service._drive(run, True)
        state = service.get(run)
        folder = service.folder(run)
        record = state[review.HISTORY_FIELD][-1]
        if target == 'context':
            path = folder/'acquisitions'/record['context']['acquisition_id']/'result.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            data['return_context']['useful_if'] = 'changed after acquisition'
        elif target == 'record':
            path = folder/'operations'/record['assessment_operation_id']/'return-assessment.json'
            data = deepcopy(record); data['assessment']['outcome'] = 'resolved'
        else:
            path = folder/'state.json'; data = deepcopy(state)
            if target == 'state_history': data[review.HISTORY_FIELD] = []
            else: data[review.PENDING_FIELD] = record['context']
        atomic_write(path, data)
        with pytest.raises(ValueError, match='REPLAY_.*RETURN'):
            verify_run(folder)
    finally:
        service.close()


def test_proposer_has_shared_review_and_separate_action_schema(body_public):
    _, _, public = body_public
    response = decision(public)
    response['local_query'] = None
    schema = MatchLocalReturnProposalProtocol.schema
    shared = {k: deepcopy(v) for k,v in response.items() if k in schema['properties']}
    item = {k: deepcopy(v) for k,v in response.items()
            if k in schema['properties']['candidates']['items']['properties']}
    payload = {**shared, 'candidates': [item], 'recommended_index': 0}
    validate(payload, schema)
    assert review.FIELD not in item and payload[review.FIELD] == response[review.FIELD]
    provider = MatchCandidateProvider(Path('.'), provider=Provider(lambda p: payload))
    result = provider.propose(public)
    assert result['candidates'] == [response]
    assert result['proposal_protocol'] == MatchLocalReturnProposalProtocol.version


def test_cli_new_task_enables_contract_without_calls(tmp_path, capsys):
    from football_agent.runtime_cli import main
    main(['--root', str(tmp_path), 'create', '--match', '测试甲队 vs 测试乙队'])
    state = json.loads(capsys.readouterr().out)
    assert state['config'][review.CAPABILITY_FIELD] == review.capability()
    assert state['spent'] == {'http_calls': 0, 'llm_calls': 0}
    assert verify_run(tmp_path/'runs'/state['run_id'])['question_return_assessments_verified'] == 0


def test_legacy_library_contract_remains_usable(tmp_path):
    service = ExperienceService(tmp_path, mode='test_fixture',
        provider_factory=lambda f,e: Provider(lambda p: match_decision(p, 'stop')))
    try:
        run = service.create({'match': '旧版构造甲队 vs 构造乙队'})['run_id']
        state = service.step(run)
        assert state['status'] == 'deferred'
        assert review.CAPABILITY_FIELD not in public_state(state)
        assert 'question_return_assessments_verified' not in verify_run(service.folder(run))
    finally:
        service.close()


@pytest.mark.parametrize('kind', ['no_document', 'blocked', 'empty'])
def test_failed_or_empty_return_preserves_unknowns_and_can_stop(tmp_path, kind):
    class Tools:
        def execute(self, action, *, run_id, reserve_request, cancelled):
            request_id = reserve_request(action['url'])
            doc = None
            status = 200 if kind == 'empty' else 403
            if kind != 'no_document':
                doc = h['project_body'](b'', action['url'], status)
                doc.update(document_id='empty-or-failed', observed_at=h['now']().isoformat(), source_published_at=None)
            return {'document': doc, 'requests': [{'request_id': request_id, 'http_status': status, 'error': None}]}
    def choose(p):
        response = decision(p, 'stop' if p[review.PENDING_FIELD] else 'read_url')
        if response['action'] == 'read_url': response['url'] = p['source_catalog']['entries'][0]['url']
        return response
    service = ExperienceService(tmp_path, provider_factory=lambda f,e: Provider(choose),
        tools_factory=lambda f: Tools(), mode='test_fixture', enable_return_assessment=True)
    try:
        run = service.create({'match': '构造甲队 vs 构造乙队'})['run_id']
        service._drive(run, True)
        state = service.get(run)
        assert state['status'] == 'deferred'
        record = state[review.HISTORY_FIELD][0]
        assert record['assessment']['outcome'] == ('insufficient_context' if kind == 'empty' else 'unavailable')
        assert record['assessment']['citations'] == [] and record['assessment']['remaining_unknowns']
        assert record['realized_decision_gain'] is None
        assert state['spent'] == {'http_calls': 1, 'llm_calls': 2}
        assert verify_run(service.folder(run))['question_return_assessments_verified'] == 1
    finally:
        service.close()


def test_local_query_assesses_original_question_without_extra_network(tmp_path):
    local = runpy.run_path(str(Path(__file__).with_name('test_local_evidence.py')))
    calls = []
    rows = [local['fixture_row'](matchId=1000+i, home='构造甲队'+str(i)) for i in range(80)]
    rows[-1]['home'] = '窗口外目标甲队'
    class Proposer:
        def propose(self, p):
            if not p['documents']:
                response = decision(p, 'read_url')
                response['url'] = p['source_catalog']['entries'][0]['url']
            elif len(p['documents']) == 1:
                response = local['local_decision'](p, team_name='窗口外目标甲队')
                response[review.FIELD] = actual_review(p, 'insufficient_context')
            else:
                response = decision(p)
                # A full structured line may exceed the quotation bound. Cite
                # a visible unique identity span, preserving unresolved lineup.
                response[review.FIELD]['citations'][0]['quote'] = '窗口外目标甲队'
            return {'candidates': [response], 'recommended_index': 0, 'test_double': True}
    service = ExperienceService(tmp_path, proposer_factory=lambda f,e: Proposer(),
        tools_factory=lambda f: local['h']['make_tools'](f, calls, {'recent': local['recent_payload'](rows)}),
        mode='test_fixture', enable_return_assessment=True)
    try:
        run = service.create({'match': '窗口外目标甲队 vs 构造乙队', 'limits': {'http_calls': 1}})['run_id']
        service._drive(run, True)
        state = service.get(run)
        assert state['status'] == 'deferred'
        assert [t['action'] for t in state['turns']] == ['read_url', 'query_acquired', 'stop']
        assert len(calls) == 1 and state['local_usage']['reads'] == 1
        assert state['spent'] == {'http_calls': 1, 'llm_calls': 3}
        assert state[review.HISTORY_FIELD][-1]['context']['action'] == 'query_acquired'
        audit = verify_run(service.folder(run))
        assert audit['local_documents_reconstructed'] == 1
        assert audit['question_return_assessments_verified'] == 2
    finally:
        service.close()


def test_independent_policy_stop_preserves_actual_return_review(body_public):
    _, _, public = body_public
    response = decision(public, 'search_web')
    response['query'] = 'different followup lineup inquiry'
    batch = CandidateBatch.build(public, [response], origin='test', recommended_index=0)
    choice = PolicyChoice(task_id=public['task_id'], snapshot_digest=batch.context.snapshot_digest,
                          terminal_action='stop', reason='fixture policy stop with unresolved evidence')
    selected = materialize_choice(batch.context, batch, choice)
    validate_decision(selected, public)
    assert selected['action'] == 'stop' and selected[review.FIELD] == response[review.FIELD]


def test_contract_mismatch_fails_before_charging_budget(tmp_path):
    service, run, seen = fixture_runtime(tmp_path)
    try:
        service.mutate(run, lambda s: s['config'][review.CAPABILITY_FIELD].update(version='future-unknown'))
        state = service.step(run)
        assert state['status'] == 'paused' and state['spent']['llm_calls'] == 0
        assert not seen and state['last_error'] == 'RETURN_ASSESSMENT_CAPABILITY_CHANGED'
    finally:
        service.close()


def test_no_assessment_can_be_invented_before_acquisition(tmp_path):
    service, run, seen = fixture_runtime(tmp_path)
    try:
        public = public_state(service.get(run))
        response = decision(public)
        response[review.FIELD] = {'acquisition_id': 'invented', 'question_id': 'fixture',
            'context_digest': 'invented', 'outcome': 'unavailable', 'reason': 'no actual acquisition',
            'citations': [], 'remaining_unknowns': ['unknown']}
        with pytest.raises(ValueError, match='WITHOUT_ACTUAL_RETURN'):
            validate_decision(response, public)
    finally:
        service.close()


def test_semantic_outcomes_remain_provisional_even_with_valid_citations(body_public):
    _, _, public = body_public
    for outcome in ['resolved', 'partial', 'irrelevant', 'conflicting']:
        response = decision(public)
        response[review.FIELD] = actual_review(public, outcome)
        if outcome == 'conflicting':
            # This only tests quotation binding: these two spans do NOT prove
            # contradiction. The runtime must not claim it verified entailment.
            response[review.FIELD]['citations'] = [
                {'document_id': 'match-page', 'quote': 'Squad list only'},
                {'document_id': 'match-page', 'quote': 'onerror=alert(1)'}]
        validate_decision(response, public)
        record = review.assessment_record(public, response, 'fixture-assessment')
        assert record['independent_semantic_validation'] is False
        assert record['world_truth_verified'] is False and record['numeric_voi_training_eligible'] is False


def test_context_binds_redacted_archive_and_replay_does_not_hide_later_mutation(tmp_path):
    service, run, seen = fixture_runtime(tmp_path)
    class MetadataTools(MatchTools):
        def execute(self, *args, **kwargs):
            result = super().execute(*args, **kwargs)
            result['provider_metadata'] = {'api_token': 'fixture-only-not-a-real-key'}
            return result
    service.tools_factory = lambda f: MetadataTools()
    try:
        service._drive(run, True)
        state = service.get(run)
        assert state['status'] == 'deferred'
        assert verify_run(service.folder(run))['question_return_assessments_verified'] == 2
        context = state[review.HISTORY_FIELD][-1]['context']
        path = service.folder(run)/'acquisitions'/context['acquisition_id']/'result.json'
        result = json.loads(path.read_text(encoding='utf-8'))
        assert result['provider_metadata']['api_token'] == '[REDACTED]'
        # Deliberately bypass the redacting writer to model archive tampering.
        result['provider_metadata']['api_token'] = 'post-storage-change'
        path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
        with pytest.raises(ValueError, match='REPLAY_RETURN_CONTEXT_MISMATCH'):
            verify_run(service.folder(run))
    finally:
        service.close()
