"""Cold-start exploration through the actual durable executor (synthetic HTTP)."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from football_agent import runtime_exploration as exploration
from football_agent import runtime_source_claims as claims
from football_agent.acquisition_replay import verify_run
from football_agent.audit import atomic_write, digest
from football_agent.experience import ExperienceService, now, public_state
from football_agent.experience_sources import ExperienceWebTools, VERSION as TRANSPORT_VERSION
from football_agent.runtime_voi_model import RuntimeVoIModel, SOURCE_CLAIM_VERSION, SOURCE_CLAIM_SEMANTICS
from football_agent.state_bound_runtime_policy import StateBoundVoISelectionPolicy
from football_agent.structured_sources import registered_request
from football_spider.storage.raw_archive import RawArchive

H = runpy.run_path(str(Path(__file__).with_name('test_structured_sources.py')))
R = runpy.run_path(str(Path(__file__).with_name('test_match_return_assessment.py')))
V = runpy.run_path(str(Path(__file__).with_name('test_state_bound_runtime_policy.py')))
URL = 'https://example.org/context'


def model_package():
    config = V['package']((('a', 9, .9), ('b', 3, .9), ('c', 3, -.5), ('c', 4, -.5)),
                         keys=['operation_document_count', 'home_reported_qualification_state'])
    config.update(version=SOURCE_CLAIM_VERSION, context_semantics=SOURCE_CLAIM_SEMANTICS)
    config['routes'].append({'route_id': 'body', 'match': {'action': 'read_url', 'url': URL},
        'source_id': 'web:example.org', 'source_version': TRANSPORT_VERSION,
        'action_id': 'read-context', 'cost': .1, 'unit': 'decision_loss'})
    other = deepcopy(config['routes'][0])
    other.update(route_id='d', action_id='discover-d', cost=.15)
    other['match']['query_template'] = '{match_input} source d'
    config['routes'].append(other)
    config['family_size'] = len(config['routes'])
    return config


def fixture_runtime(root, *, fail_body=False, interruption=None, options=None):
    seen, calls = [], []
    start, end = (now()-timedelta(days=1)).isoformat(), (now()+timedelta(days=1)).isoformat()
    window, quote = f'Applies {start} until {end}.', '构造甲队 reports qualified.'
    payload = H['recent_payload']([H['fixture_row'](matchTime='2026-10-03T18:00:00')])

    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            calls.append(url)
            if registered_request(url):
                return SimpleNamespace(status_code=200, content=json.dumps(payload, ensure_ascii=False).encode(),
                                       headers={'Content-Type': 'application/json'})
            body = (f'<html><article>{quote} {window}</article></html>' if url == URL
                    else f'<html><a href="{URL}">Team source statement</a></html>')
            return SimpleNamespace(status_code=503 if fail_body and url == URL else 200,
                                   content=body.encode(), headers={'Content-Type': 'text/html'})

    class Proposer:
        def propose(self, public):
            seen.append(deepcopy(public))
            count = len(public['documents'])
            action = 'read_url' if count in (0, 2) else 'search_web'
            response = H['match_decision'](public, action)
            response['return_assessment'] = R['actual_review'](public)
            if response['return_assessment']:
                for cite in response['return_assessment']['citations']:
                    cite['quote'] = cite['quote'][:400]
            response[claims.FIELD] = []
            if action == 'read_url':
                response['url'] = public['source_catalog']['entries'][0]['url'] if count == 0 else URL
            else:
                response['query'] = public['match_input'] + ' source ' + ('a' if count == 1 else 'c')
                response['engine'] = 'bing'
            if count == 3:
                response[claims.FIELD] = [{'target_key': 'spftll:1001', 'subject': 'home',
                    'subject_name': '构造甲队', 'predicate': 'qualification_state', 'value': 'qualified',
                    'document_id': public['documents'][-1]['document_id'], 'quote': quote,
                    'validity': {'from': start, 'until': end, 'quote': window}}]
            if count == 1:
                if interruption == 'cancel': service.cancel(public['task_id'])
                elif interruption == 'deadline':
                    service.mutate(public['task_id'], lambda s: s.update(deadline=(now()-timedelta(seconds=1)).isoformat()))
                elif interruption == 'invalid': response['task_id'] = 'wrong-task'
            responses = [response]
            if count == 1:
                other = deepcopy(response)
                other['query'] = public['match_input'] + ' source d'
                # The proposer prefers the expensive unknown; the allowance
                # rule keeps the full menu but selects the cheaper unknown.
                responses.insert(0, other)
            if count == 3:
                positive = deepcopy(response)
                positive['query'] = public['match_input'] + ' source b'
                responses.append(positive)
            return {'candidates': responses, 'recommended_index': 0, 'test_double': True}

    policy = StateBoundVoISelectionPolicy(RuntimeVoIModel(model_package()), allow_synthetic=True,
        exploration=options or {'max_actions': 2, 'max_declared_cost': .2})
    service = ExperienceService(root, proposer_factory=lambda f,e: Proposer(), mode='test_fixture',
        selection_policy=policy, enable_return_assessment=True,
        tools_factory=lambda f: ExperienceWebTools(RawArchive(f), dns_check=lambda url: None,
                                                   session_factory=lambda route: Session()))
    run = service.create({'match': '2026-10-03 构造甲队 vs 构造乙队',
                          'limits': {'llm_calls': 7, 'http_calls': 6}})['run_id']
    return service, run, calls, seen


def run_scenario(root, **kwargs):
    service, run, calls, seen = fixture_runtime(root, **kwargs)
    service._drive(run, True)
    return service, run, calls, seen


def test_cold_start_explores_then_evidence_drives_value_selection_and_stop(tmp_path):
    service, run, calls, seen = run_scenario(tmp_path)
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert [r['action'] for r in state['turns']] == ['read_url','search_web','read_url','search_web','stop']
        assert state['turns'][3]['query'].endswith(' source b')
        assert state['spent'] == {'http_calls': 4, 'llm_calls': 5}
        assert not state['user_context'] and state['question_to_user'] is None
        assert seen[1][claims.STATE_FIELD]['context']['home_reported_qualification_state'] == 'unknown'
        assert seen[-1][claims.STATE_FIELD]['context']['home_reported_qualification_state'] == 'qualified'
        report = verify_run(service.folder(run), include_action_value_ledger=True)
        budget = report['bounded_exploration']
        assert budget['reserved_actions'] == 2 and budget['reserved_declared_cost'] == .2
        assert budget['remaining_actions'] == 0
        assert report['accepted_decisions_verified'] == 5
        assert report['source_assertions_verified'] == 1
        ledger = report['action_value_ledger']['entries']
        explored = next(e for e in ledger if e['selection_basis'] == exploration.BASIS)
        assert len(explored['candidate_value_assessments']) == 2
        assert explored['expected_information_value'] is None
        from football_agent.value_evaluation_contract import _adapter_candidate_ids
        assert len(_adapter_candidate_ids(explored)) == 2
        assert all(not r['numeric_voi_training_eligible'] for r in budget['reservations'])
        assert len(calls) == 4
    finally: service.close()


@pytest.mark.parametrize('options', [{'max_actions':1,'max_declared_cost':.2},
                                   {'max_actions':2,'max_declared_cost':.1}])
def test_either_exploration_limit_stops_unknown_before_body_fetch(tmp_path, options):
    service, run, calls, seen = run_scenario(tmp_path, options=options)
    try:
        state = service.get(run)
        assert state['last_error'] is None and state['turns'][-1]['action'] == 'defer'
        assert len(calls) == 2 and URL not in calls
        assert verify_run(service.folder(run))['bounded_exploration']['reserved_actions'] == 1
    finally: service.close()


@pytest.mark.parametrize('interruption', ['cancel', 'deadline', 'invalid'])
def test_discarded_or_invalid_planning_never_reserves_exploration(tmp_path, interruption):
    service, run, _, _ = run_scenario(tmp_path, interruption=interruption)
    try:
        assert not any(exploration.RESERVATION in o for o in service.get(run)['operations'])
        assert verify_run(service.folder(run))['bounded_exploration']['reserved_actions'] == 0
    finally: service.close()


def test_failed_http_still_spends_and_restart_keeps_accepted_reservations(tmp_path):
    service, run, _, _ = fixture_runtime(tmp_path, fail_body=True)
    try:
        for _ in range(6): service.step(run)
        before = public_state(service.get(run))[exploration.STATE_FIELD]
        assert before['reserved_actions'] == 2 and before['remaining_actions'] == 0
        assert service.get(run)['spent']['http_calls'] == 3
        service.close()
        resumed = ExperienceService(tmp_path, mode='test_fixture', selection_policy=service.selection_policy)
        try:
            assert public_state(resumed.get(run))[exploration.STATE_FIELD] == before
            assert verify_run(resumed.folder(run))['bounded_exploration'] == before
        finally: resumed.close()
    finally: service.close()


@pytest.mark.parametrize('phase', ['plan', 'fetch'])
def test_changed_allowance_pauses_before_new_provider_or_http_call(tmp_path, phase):
    service, run, calls, seen = fixture_runtime(tmp_path)
    try:
        for _ in range(3 if phase == 'fetch' else 4): service.step(run)
        before = (len(calls), len(seen))
        service.selection_policy = StateBoundVoISelectionPolicy(RuntimeVoIModel(model_package()),
            allow_synthetic=True, exploration={'max_actions':3,'max_declared_cost':.3})
        service.step(run)
        assert service.get(run)['status'] == 'paused'
        assert (len(calls), len(seen)) == before
        assert verify_run(service.folder(run))['bounded_exploration']['reserved_actions'] == 1
    finally: service.close()


def test_clarification_answer_preserves_accepted_exploration_allowance(tmp_path):
    service, run, _, _ = fixture_runtime(tmp_path)
    try:
        for _ in range(4): service.step(run)
        before = public_state(service.get(run))[exploration.STATE_FIELD]
        # Exercise the existing answer path; this fixture does not ask users to
        # formulate research questions in normal execution.
        service.mutate(run, lambda s: s.update(status='needs_input', question_to_user='确认比赛身份'))
        service.answer(run, '同一场比赛')
        assert public_state(service.get(run))[exploration.STATE_FIELD] == before
    finally: service.close()


@pytest.mark.parametrize('mutation', ['cost', 'group', 'delete', 'snapshot', 'unaccepted'])
def test_replay_rejects_exploration_reservation_or_snapshot_tampering(tmp_path, mutation):
    service, run, _, _ = run_scenario(tmp_path)
    try:
        def edit(state):
            op = next(o for o in state['operations'] if exploration.RESERVATION in o)
            if mutation == 'cost': op[exploration.RESERVATION]['declared_cost'] = 0
            elif mutation == 'group': op[exploration.RESERVATION]['route_group'] = 'fake'
            elif mutation == 'delete': del op[exploration.RESERVATION]
            elif mutation == 'unaccepted':
                state['operations'].append({'operation_id':'fake','kind':'http', exploration.RESERVATION:op[exploration.RESERVATION]})
            else:
                path = service.folder(run)/'operations'/op['operation_id']/'input.json'
                public = json.loads(path.read_text(encoding='utf-8'))
                public[exploration.STATE_FIELD]['remaining_actions'] = 99
                atomic_write(path, public)
                op['input_digest'] = digest(public)
        service.mutate(run, edit)
        with pytest.raises(ValueError, match='EXPLORATION'):
            verify_run(service.folder(run))
    finally: service.close()
