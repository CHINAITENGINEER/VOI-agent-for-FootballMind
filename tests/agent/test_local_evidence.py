from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import runpy

import pytest
from jsonschema import ValidationError, validate

from football_agent.acquisition_boundary import CandidateBatch, PolicyChoice, materialize_choice, SHARED_FIELDS
from football_agent.acquisition_replay import verify_run
from football_agent.audit import atomic_write, digest
from football_agent.experience import (ExperienceService, public_state, validate_decision, now, action_key,
                                      MatchAnalysisProtocol, MatchLocalAnalysisProtocol)
from football_agent.local_evidence import ACTION, capability, project_query
from football_agent.match_proposals import MatchCandidateProvider, MatchLocalProposalProtocol, MatchProposalProtocol

h = runpy.run_path(str(Path(__file__).with_name('test_structured_sources.py')))
fixture_row, recent_payload, fetch_once = (h[k] for k in ('fixture_row', 'recent_payload', 'fetch_once'))
match_decision = h['match_decision']


def query(document, **changes):
    return {'document_id': document['document_id'], 'record_kind': 'fixture',
            'source_match_id': None, 'team_name': None, 'team_match': 'exact',
            'offset': 0, 'limit': 20, **changes}


def local_decision(public, **changes):
    response = match_decision(public, ACTION)
    response.update(local_query=query(public['documents'][0], **changes), active_question_id='fixture')
    return response


def fixture_runtime(root, calls=None, seen=None):
    calls = calls if calls is not None else []
    seen = seen if seen is not None else []
    rows = [fixture_row(matchId=1000+i, home='构造甲队'+str(i)) for i in range(80)]
    rows[-1]['home'] = '窗口外目标甲队'
    class Proposer:
        def propose(self, public):
            seen.append(deepcopy(public))
            if not public['documents']:
                response = match_decision(public, 'read_url')
                response['url'] = public['source_catalog']['entries'][0]['url']
            elif len(public['documents']) == 1:
                response = local_decision(public, team_name='窗口外目标甲队')
            else:
                response = match_decision(public, 'stop')
                response['assessment'] = {
                    'status': 'partial', 'answer': '已定位源记录，正式阵容和战术仍待核实。',
                    'claims': [{'statement': '来源中含目标球队的比赛记录',
                                'document_id': public['documents'][-1]['document_id'],
                                'quote': public['documents'][-1]['excerpt']}],
                    'remaining_unknowns': ['正式阵容与战术']}
            return {'candidates': [response], 'recommended_index': 0, 'fixture': True}
    service = ExperienceService(root, mode='test_fixture',
        proposer_factory=lambda f,e: Proposer(),
        tools_factory=lambda f: h['make_tools'](f, calls, {'recent': recent_payload(rows)}))
    run = service.create({'match': '窗口外目标甲队 vs 构造乙队',
                          'limits': {'http_calls': 1, 'llm_calls': 5}})['run_id']
    return service, run, calls, seen


def test_complete_runtime_finds_unseen_target_after_network_budget_exhausted(tmp_path):
    service, run, calls, seen = fixture_runtime(tmp_path)
    try:
        service._drive(run, True)
        state = service.get(run)
        assert state['status'] == 'deferred'
        assert [t['action'] for t in state['turns']] == ['read_url', ACTION, 'stop']
        assert len(calls) == 1 and state['spent'] == {'http_calls': 1, 'llm_calls': 3}
        assert state['local_usage'] == {'reads': 1} and state['numeric_voi'] is None
        assert seen[1]['remaining']['http_calls'] == 0 and not seen[1]['final_decision_only']
        assert '窗口外目标甲队' not in seen[1]['documents'][0]['excerpt']
        assert '窗口外目标甲队' not in state['documents'][0]['text']
        page = seen[-1]['documents'][-1]
        assert page['structured_records'][0]['source_id'] == '1079'
        assert page['local_evidence']['records_matched'] == 1
        assert page['observed_at'] == seen[1]['documents'][0]['observed_at']
        assert page['source_published_at'] is None and page['request_id'] is None
        assert page['raw_sha256'] == state['documents'][0]['raw_sha256']
        assert '1079' in page['links'][0]['url'] and not state['user_context']
        audit = verify_run(service.folder(run))
        assert audit['linked_http_receipts'] == audit['linked_local_reads'] == 1
        assert audit['platform_documents_reconstructed'] == audit['local_documents_reconstructed'] == 1
        assert audit['real_calls_made_by_replay'] == 0
    finally:
        service.close()


def test_pagination_reveals_every_row_once_with_actual_next_offset(tmp_path):
    result, calls, archive = fetch_once(tmp_path, recent_payload([fixture_row(matchId=1000+i) for i in range(80)]))
    parent, offset, ids = result['document'], 0, []
    while offset is not None:
        page = project_query(archive, [parent], query(parent, offset=offset), str(offset))
        metadata = page['local_evidence']
        rows = [json.loads(line) for line in page['text'].splitlines()]
        assert 0 < len(rows) < 20 and len(page['text']) <= 4500
        ids.extend(row['source_id'] for row in rows)
        assert metadata['records_returned'] == len(rows)
        assert metadata['records_matched'] == 80
        offset = metadata['next_offset']
        if offset is not None: assert offset == len(ids)
    assert ids == [str(1000+i) for i in range(80)] and len(calls) == 1


def test_literal_filters_preserve_scope_unknowns_and_no_match_semantics(tmp_path):
    payload = recent_payload([fixture_row(home='Alpha'), fixture_row(matchId=1002, home='Alpha B')])
    result, calls, archive = fetch_once(tmp_path, payload)
    parent = result['document']
    exact = project_query(archive, [parent], query(parent, team_name='Alpha'), 'exact')
    contains = project_query(archive, [parent], query(parent, team_name='Alpha', team_match='contains'), 'contains')
    absent = project_query(archive, [parent], query(parent, team_name='alpha'), 'absent')
    by_id = project_query(archive, [parent], query(parent, source_match_id='1002'), 'id')
    conjunction = project_query(archive, [parent], query(parent, source_match_id='1002', team_name='Alpha'), 'both')
    assert exact['local_evidence']['records_matched'] == 1
    assert contains['local_evidence']['records_matched'] == 2
    assert absent['status'] == conjunction['status'] == 'empty_structured'
    assert not absent['text'] and not absent['links']
    assert 'no_result_is_not_evidence_of_match_absence' in absent['signals']
    assert json.loads(by_id['text'])['source_id'] == '1002'
    assert json.loads(by_id['text'])['score']['home'] is None
    assert len(calls) == 1


@pytest.mark.parametrize('changes', [
    {'offset': -1}, {'offset': True}, {'limit': 0}, {'limit': 21}, {'limit': 1.5},
    {'team_name': ' '}, {'record_kind': 'odds', 'team_name': '甲'},
    {'source_match_id': '../archive'}, {'document_id': 'doc-00000000-0000-0000-0000-000000000000'},
    {'path': '../other-task'},
])
def test_query_rejects_bad_arguments_and_unacquired_parent(tmp_path, changes):
    result, _, archive = fetch_once(tmp_path, recent_payload())
    parent = result['document']
    with pytest.raises((ValueError, ValidationError)):
        project_query(archive, [parent], query(parent, **changes), 'reject')


def test_oversized_record_has_explicit_gap_and_forward_pagination(tmp_path):
    rows = [fixture_row(league='L'*6000), fixture_row(matchId=1002)]
    result, _, archive = fetch_once(tmp_path, recent_payload(rows))
    parent = result['document']
    first = project_query(archive, [parent], query(parent), 'oversized')
    assert first['status'] == 'unsupported_content' and not first['text']
    assert first['local_evidence']['omitted_oversized_offsets'] == [0]
    assert first['local_evidence']['next_offset'] == 1
    second = project_query(archive, [parent], query(parent, offset=1), 'next')
    assert json.loads(second['text'])['source_id'] == '1002'
    assert second['local_evidence']['next_offset'] is None


def test_no_unseen_quotes_and_local_arguments_are_part_of_action_identity(tmp_path):
    service, run, _, _ = fixture_runtime(tmp_path)
    try:
        service.step(run); state = service.step(run)
        p = public_state(state)
        a = local_decision(p, offset=0); b = local_decision(p, offset=5)
        validate_decision(a, p)
        assert action_key(a) != action_key(b)
        batch = CandidateBatch.build(p, [a,b], recommended_index=0, origin='constructed')
        assert batch.candidates[0].candidate_id != batch.candidates[1].candidate_id
        cost = batch.record()['candidates'][0]['resource_cost']
        assert cost['http_request_lower_bound'] == cost['http_request_upper_bound'] == 0
        assert cost['local_read_attempts'] == 1 and cost['decision_loss_equivalent'] is None
        stopped = materialize_choice(batch.context, batch, PolicyChoice(p['task_id'],
            batch.context.snapshot_digest, terminal_action='stop', reason='已达到当前预算边界'))
        assert stopped['local_query'] is None
        validate_decision(stopped, p)
        a['assessment']['claims'] = [{'statement':'未读取的比赛', 'quote':'窗口外目标甲队',
                                      'document_id':p['documents'][0]['document_id']}]
        with pytest.raises(ValueError, match='UNSUPPORTED_QUOTATION'): validate_decision(a, p)
        a['assessment']['claims'] = []
        p['do_not_repeat'] = [action_key(a)]
        with pytest.raises(ValueError, match='REPEATED_ACTION'): validate_decision(a, p)
        network = match_decision(p)
        with pytest.raises(ValueError, match='HTTP_BUDGET_EXHAUSTED'): validate_decision(network, p)
    finally:
        service.close()


@pytest.mark.parametrize('kind', ['cancel', 'deadline', 'capability', 'budget'])
def test_queued_local_query_obeys_execution_time_guards(tmp_path, kind):
    service, run, calls, _ = fixture_runtime(tmp_path)
    try:
        for _ in range(3): service.step(run)
        state = service.get(run)
        assert state['queued_action']['action'] == ACTION
        if kind == 'cancel': service.cancel(run)
        elif kind == 'deadline': service.mutate(run, lambda s:s.update(deadline=(now()-timedelta(seconds=1)).isoformat()))
        elif kind == 'capability': service.mutate(run, lambda s:s['config']['local_query_capability'].update(version='changed'))
        else: service.mutate(run, lambda s:s['local_usage'].update(reads=capability()['max_reads']))
        service._drive(run, True)
        state = service.get(run)
        assert len(calls) == 1 and len(state['documents']) == 1
        assert state['local_usage']['reads'] == (capability()['max_reads'] if kind == 'budget' else 0)
        assert state['status'] in {'cancelled','deadline_expired','paused'}
    finally:
        service.close()


def test_query_error_is_charged_once_without_hidden_network_fallback(tmp_path):
    service, run, calls, _ = fixture_runtime(tmp_path)
    try:
        for _ in range(3): service.step(run)
        body = next((service.folder(run)/'archive').rglob('*.body.gz'))
        original = body.read_bytes(); body.write_bytes(b'bad archive')
        state = service.step(run)
        assert state['status'] == 'paused' and state['local_usage']['reads'] == 1
        assert len(calls) == 1 and len(state['documents']) == 1
        assert state['operations'][-1]['status'] == 'failed'
        public=public_state(state)
        assert public['last_local_read']['status'] == 'failed' and public['last_local_read']['error']
        assert public['previous_choices'][-1]['local_query']['team_name'] == '窗口外目标甲队'
        body.write_bytes(original)
        audit = verify_run(service.folder(run))
        assert audit['linked_local_reads'] == 1 and audit['local_documents_reconstructed'] == 0
    finally:
        service.close()


def test_interruption_preserves_reserved_local_budget_and_requires_reconciliation(tmp_path, monkeypatch):
    service, run, calls, _ = fixture_runtime(tmp_path)
    try:
        for _ in range(3): service.step(run)
        def interrupted(*args): raise KeyboardInterrupt()
        monkeypatch.setattr('football_agent.experience.project_query', interrupted)
        with pytest.raises(KeyboardInterrupt): service.step(run)
        assert service.get(run)['local_usage']['reads'] == 1
        service.recover()
        state = service.step(run)
        assert state['status'] == 'needs_reconciliation'
        assert state['pending']['kind'] == 'local_read' and len(calls) == 1
        assert verify_run(service.folder(run))['pending_local_reads'] == 1
    finally:
        service.close()


@pytest.mark.parametrize('part', ['projection', 'receipt', 'budget'])
def test_replay_detects_local_tampering(tmp_path, part):
    service, run, _, _ = fixture_runtime(tmp_path)
    try:
        service._drive(run, True)
        state = service.get(run)
        page = state['documents'][-1]
        path = service.folder(run)/'acquisitions'/page['acquisition_id']/'result.json'
        result = json.loads(path.read_text(encoding='utf-8'))
        if part == 'budget':
            state['local_usage']['reads'] = 0
        elif part == 'receipt':
            result['local_receipt']['http_calls'] = 1
        else:
            result['document']['text'] = result['document']['text'].replace('窗口外目标甲队','篡改甲队')
            result['local_receipt']['document_digest'] = digest(result['document'])
            state['operations'][-1 if state['operations'][-1]['kind']=='local_read' else -2]['receipt'] = result['local_receipt']
            page['text'] = result['document']['text']
        atomic_write(path, result)
        atomic_write(service.folder(run)/'state.json', state)
        with pytest.raises(ValueError, match='REPLAY_LOCAL_'): verify_run(service.folder(run))
    finally:
        service.close()


def test_new_proposal_protocol_exposes_typed_local_query_without_changing_legacy(tmp_path):
    service, run, _, _ = fixture_runtime(tmp_path/'runtime')
    try:
        service.step(run); state = service.step(run)
        p = public_state(state)
        response = local_decision(p, team_name='窗口外目标甲队')
        raw = {**{k:deepcopy(response[k]) for k in SHARED_FIELDS},
               'candidates':[{k:v for k,v in response.items() if k not in SHARED_FIELDS and k!='alternatives'}],
               'recommended_index':0}
        validate(raw, MatchLocalProposalProtocol.schema)
        class Provider:
            def decide(self, public): return {'response':raw, 'fixture':True}
        result = MatchCandidateProvider(tmp_path/'provider', provider=Provider()).propose(p)
        assert result['proposal_protocol'] == MatchLocalProposalProtocol.version
        validate_decision(result['candidates'][0], p)
        old = deepcopy(p); old.pop('local_query_capability')
        with pytest.raises(ValidationError): validate_decision(response, old)
        validate_decision(match_decision(old,'stop'), old)
    finally:
        service.close()


def test_local_odds_query_preserves_market_and_settlement_semantics(tmp_path):
    payload=json.loads((h['FIXTURES']/'odds.json').read_text(encoding='utf-8'))
    # A constructed response for fetch_once's fixture request; original file is unchanged.
    payload['data']['footballMoreSpInfo']['matchId']=1001
    result, calls, archive=fetch_once(tmp_path,payload,'odds')
    parent=result['document']
    page=project_query(archive,[parent],query(parent,record_kind='odds',source_match_id='1001'),'odds')
    rows=[json.loads(line) for line in page['text'].splitlines()]
    original=[json.loads(line) for line in parent['text'].splitlines()]
    assert rows and rows == original[:len(rows)]
    assert all(row['source_match_id']=='1001' and row['period']=='regulation' for row in rows)
    assert not page['links'] and len(calls)==1


def test_local_reads_require_budget_for_followup_reasoning(tmp_path):
    service, run, _, _ = fixture_runtime(tmp_path)
    try:
        service.step(run); state=service.step(run)
        state['spent']['llm_calls']=state['config']['limits']['llm_calls']-1
        public=public_state(state)
        assert public['final_decision_only']
        with pytest.raises(ValueError,match='FINAL_DECISION_REQUIRED'):
            validate_decision(local_decision(public),public)
        state['spent']['llm_calls']=1
        state['local_usage']['reads']=capability()['max_reads']
        public=public_state(state)
        assert public['final_decision_only'] and public['remaining']['local_reads']==0
    finally:
        service.close()


def test_legacy_wire_protocol_stays_frozen_and_local_protocol_is_strict():
    assert ACTION not in MatchAnalysisProtocol.schema['properties']['action']['enum']
    assert 'local_query' not in MatchAnalysisProtocol.schema['properties']
    assert 'LOCAL EVIDENCE QUERY CONTRACT' not in MatchAnalysisProtocol.prompt
    old=MatchProposalProtocol.schema['properties']['candidates']['items']
    assert 'local_query' not in old['properties'] and ACTION not in old['properties']['action']['enum']
    for schema in [MatchLocalAnalysisProtocol.schema,
                   MatchLocalProposalProtocol.schema['properties']['candidates']['items']]:
        assert set(schema['required']) == set(schema['properties'])
        assert 'local_query' in schema['required']
        assert ACTION in schema['properties']['action']['enum']
