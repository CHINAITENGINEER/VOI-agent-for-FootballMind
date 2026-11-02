"""Synthetic transport, real durable acquisition/claim history and replay."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from football_agent import runtime_source_claims as claims
from football_agent.acquisition_boundary import CandidateBatch
from football_agent.acquisition_replay import verify_run
from football_agent.audit import atomic_write
from football_agent.experience import ExperienceService, now
from football_agent.experience_sources import ExperienceWebTools
from football_agent.structured_sources import registered_request
from football_spider.storage.raw_archive import RawArchive

H = runpy.run_path(str(Path(__file__).with_name('test_structured_sources.py')))
R = runpy.run_path(str(Path(__file__).with_name('test_match_return_assessment.py')))
URL1, URL2 = 'https://example.org/report-one', 'https://example.org/report-two'


def fixture_runtime(root, *, interruption=None):
    calls, seen, chosen = [], [], []
    start, end = (now() - timedelta(days=1)).isoformat(), (now() + timedelta(days=1)).isoformat()
    window = f'Source applicability: {start} until {end}.'
    quote1, quote2 = '构造甲队 reports qualified.', '构造甲队 reports pending qualification.'
    payload = H['recent_payload']([H['fixture_row'](matchTime='2026-10-03T18:00:00')])

    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            calls.append(url)
            if registered_request(url):
                return SimpleNamespace(status_code=200, content=json.dumps(payload, ensure_ascii=False).encode(),
                                       headers={'Content-Type': 'application/json'})
            if url in (URL1, URL2):
                text = quote1 if url == URL1 else quote2
                body = f'<html><article>{text} {window}</article></html>'
            else:
                body = f'<html><a href="{URL1}">First source</a><a href="{URL2}">Second source</a></html>'
            return SimpleNamespace(status_code=200, content=body.encode(), headers={'Content-Type': 'text/html'})

    class Proposer:
        def propose(self, public):
            seen.append(deepcopy(public))
            count = len(public['documents'])
            action = ('read_url', 'search_web', 'read_url', 'read_url', 'stop')[min(count, 4)]
            response = H['match_decision'](public, action)
            response['return_assessment'] = R['actual_review'](public)
            if response['return_assessment']:
                for citation in response['return_assessment']['citations']:
                    citation['quote'] = citation['quote'][:400]
            response[claims.FIELD] = []
            if action == 'read_url':
                response['url'] = public['source_catalog']['entries'][0]['url'] if count == 0 else URL1 if count == 2 else URL2
            if count >= 3:
                doc = public['documents'][-1]
                response[claims.FIELD] = [{'target_key': 'spftll:1001', 'subject': 'home',
                    'subject_name': '构造甲队', 'predicate': 'qualification_state',
                    'value': 'qualified' if count == 3 else 'pending',
                    'document_id': doc['document_id'], 'quote': quote1 if count == 3 else quote2,
                    'validity': {'from': start, 'until': end, 'quote': window}}]
                if count == 3:
                    if interruption == 'invalid': response[claims.FIELD][0]['target_key'] = 'spftll:9999'
                    elif interruption == 'cancel': service.cancel(public['task_id'])
                    elif interruption == 'deadline':
                        service.mutate(public['task_id'], lambda s: s.update(deadline=(now()-timedelta(seconds=1)).isoformat()))
            chosen.append(deepcopy(response))
            return {'candidates': [response], 'recommended_index': 0, 'test_double': True}

    service = ExperienceService(root, proposer_factory=lambda f,e: Proposer(), mode='test_fixture',
        tools_factory=lambda f: ExperienceWebTools(RawArchive(f), dns_check=lambda url: None,
                                                   session_factory=lambda route: Session()),
        enable_return_assessment=True, enable_source_claims=True)
    run = service.create({'match': '2026-10-03 构造甲队 vs 构造乙队',
                          'limits': {'llm_calls': 6, 'http_calls': 5}})['run_id']
    return service, run, calls, seen, chosen


def run_scenario(root):
    service, run, calls, seen, chosen = fixture_runtime(root)
    service._drive(run, True)
    return service, run, calls, seen, chosen


def test_match_only_acquisition_commits_source_state_and_conflict_replays(tmp_path):
    service, run, calls, seen, chosen = run_scenario(tmp_path)
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert [t['action'] for t in state['turns']] == ['read_url', 'search_web', 'read_url', 'read_url', 'stop']
        assert state['spent'] == {'http_calls': 4, 'llm_calls': 5}
        assert not state['user_context'] and state['question_to_user'] is None
        assert len(state[claims.HISTORY_FIELD]) == 2
        assert seen[3][claims.HISTORY_FIELD] == []
        assert seen[4][claims.STATE_FIELD]['context']['home_reported_qualification_state'] == 'qualified'
        final = claims.project_context(seen[4], chosen[4])
        assert final['context']['home_reported_qualification_state'] == 'unknown'
        assert final['state']['home']['conflicting_fields'] == ['qualification_state']
        assert final['state']['home']['context']['rotation_risk'] == 'unknown'
        assert not final['world_truth_verified'] and not final['numeric_voi_training_eligible']
        replay = verify_run(service.folder(run))
        assert replay['source_claim_records_verified'] == replay['source_assertions_verified'] == 2
        assert replay['accepted_decisions_verified'] == 5 and replay['real_calls_made_by_replay'] == 0
        assert len(calls) == 4
    finally:
        service.close()


@pytest.mark.parametrize('interruption', ['invalid', 'cancel', 'deadline'])
def test_failed_or_discarded_planning_does_not_commit_claims(tmp_path, interruption):
    service, run, _, _, _ = fixture_runtime(tmp_path, interruption=interruption)
    try:
        service._drive(run, True)
        state = service.get(run)
        assert state[claims.HISTORY_FIELD] == []
        assert len(state['turns']) == 3
        assert not list(service.folder(run).glob('operations/*/source-claims.json'))
        assert verify_run(service.folder(run))['source_assertions_verified'] == 0
    finally:
        service.close()


@pytest.mark.parametrize('mutation', ['record', 'history', 'raw_body', 'input_projection', 'discarded_record'])
def test_replay_rejects_claim_or_evidence_mutation(tmp_path, mutation):
    service, run, _, _, _ = run_scenario(tmp_path)
    try:
        state = service.get(run)
        folder = service.folder(run)
        record = state[claims.HISTORY_FIELD][0]
        path = folder/'operations'/record['operation_id']/'source-claims.json'
        if mutation == 'record':
            value = json.loads(path.read_text(encoding='utf-8'))
            value['assertions'][0]['claim']['value'] = 'invented'
            atomic_write(path, value)
        elif mutation == 'history':
            service.mutate(run, lambda s: s[claims.HISTORY_FIELD].clear())
        elif mutation == 'raw_body':
            oid = record['assertions'][0]['claim']['document_id'].removeprefix('doc-')
            body = next((folder/'archive/raw/experience_web').glob('*/*_' + oid + '.body.gz'))
            body.write_bytes(b'corrupt raw body')
        elif mutation == 'input_projection':
            path = path.with_name('input.json')
            value = json.loads(path.read_text(encoding='utf-8'))
            value[claims.STATE_FIELD]['context']['home_reported_qualification_state'] = 'invented'
            atomic_write(path, value)
        else:
            first = state['operations'][0]['operation_id']
            atomic_write(folder/'operations'/first/'source-claims.json', record)
        with pytest.raises((ValueError, OSError, EOFError)):
            verify_run(folder)
    finally:
        service.close()


def test_candidates_cannot_supply_different_current_source_states(tmp_path):
    service, run, _, seen, chosen = run_scenario(tmp_path)
    try:
        one, two = deepcopy(chosen[3]), deepcopy(chosen[3])
        two.update(action='search_web', url=None, query='another route', engine='bing')
        two[claims.FIELD][0]['value'] = 'pending'
        with pytest.raises(ValueError, match='CANDIDATES_REQUIRE_SHARED_ANALYSIS'):
            CandidateBatch.build(seen[3], [one, two], recommended_index=0, origin='test')
    finally:
        service.close()
