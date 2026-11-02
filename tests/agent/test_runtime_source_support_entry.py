"""Durable fixture-scoped support drives acquisition; all transports are fakes.

These tests exercise independently parsed source structure, not entailment,
source truth, real information gain, or a learned new-query policy.
"""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from football_agent import runtime_exploration as exploration
from football_agent import runtime_source_claims as claims
from football_agent import runtime_source_support as support
from football_agent.acquisition_replay import verify_run
from football_agent.audit import atomic_write, digest
from football_agent.experience import ExperienceService, now
from football_agent.experience_sources import ExperienceWebTools
from football_agent.runtime_fixture_bridge import prepare_fixture_identity
from football_agent.runtime_voi_model import RuntimeVoIModel
from football_agent.state_bound_runtime_policy import StateBoundVoISelectionPolicy
from football_agent.structured_sources import registered_request
from football_spider.storage.raw_archive import RawArchive

P = runpy.run_path(str(Path(__file__).with_name('test_runtime_exploration_entry.py')))
H, R, URL = P['H'], P['R'], P['URL']
HOME_STATE = 'home_reported_qualification_state'


def model_package():
    config = P['model_package']()
    config.update(version='runtime-voi-model-v3', context_semantics=support.SEMANTICS,
                  source_context_contract={'version': support.VERSION,
                      'sources': [{'source_id': 'example', 'url': URL}]})
    config['routes'] = [r for r in config['routes'] if r['route_id'] in {'b', 'c'}]
    config['history'] = [h for h in config['history'] if h['link']['candidate_id'] in {'b', 'c'}]
    config['family_size'] = len(config['routes'])
    return config


def make_policy(config=None):
    return StateBoundVoISelectionPolicy(RuntimeVoIModel(config or model_package()),
        allow_synthetic=True, exploration={'max_actions': 2, 'max_declared_cost': .2},
        discovered_routes={'search_web': {'bing': .1}, 'read_url': .1})


def fixture_runtime(root, *, scenario='current', actor_claim=None, interruption=None):
    """Identical menu/history/budget across sources; scenario changes source only."""
    if scenario not in {'current', 'wrong_kickoff', 'other_opponent', 'other_competition',
                        'prose', 'expired', 'not_started', 'duplicate_json'}:
        raise ValueError('unsupported fixture scenario')
    calls, seen = [], []
    instant = now()
    start, end = (instant-timedelta(days=1)).isoformat(), (instant+timedelta(days=1)).isoformat()
    payload = H['recent_payload']([H['fixture_row'](matchTime='2026-10-03T18:00:00')])
    source = {'schema': 'football-fixture-reports-v1',
        'fixture': {'home_team': '构造甲队', 'away_team': '构造乙队',
                    'kickoff': '2026-10-03T10:00:00Z', 'competition': '构造杯'},
        'reports': [{'subject': 'home', 'predicate': 'qualification_state', 'value': 'qualified',
                     'valid_from': start, 'valid_until': end}]}
    if scenario == 'wrong_kickoff': source['fixture']['kickoff'] = '2026-10-02T10:00:00Z'
    elif scenario == 'other_opponent': source['fixture']['away_team'] = '构造丙队'
    elif scenario == 'other_competition': source['fixture']['competition'] = '另一个杯赛'
    elif scenario == 'expired':
        source['reports'][0].update(valid_from=(instant-timedelta(days=2)).isoformat(),
                                    valid_until=(instant-timedelta(days=1)).isoformat())
    elif scenario == 'not_started':
        source['reports'][0].update(valid_from=(instant+timedelta(days=1)).isoformat(),
                                    valid_until=(instant+timedelta(days=2)).isoformat())

    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            calls.append(url)
            if registered_request(url):
                body, mime = json.dumps(payload, ensure_ascii=False), 'application/json'
            elif url == URL and scenario != 'prose':
                body, mime = json.dumps(source, ensure_ascii=False), 'application/json'
                if scenario == 'duplicate_json':
                    # The normal transport exposes only the last value; raw
                    # source validation must retain the lost ambiguity.
                    body = body.replace('"value": "qualified"',
                                        '"value": "pending", "value": "qualified"')
            elif url == URL:
                body = ('<html><article>构造甲队 reports qualified for 构造杯 against 构造乙队 '
                        f'on 2026-10-03. Applies {start} until {end}.</article></html>')
                mime = 'text/html'
            else:
                body, mime = f'<html><a href="{URL}">Source report</a></html>', 'text/html'
            return SimpleNamespace(status_code=200, content=body.encode('utf-8'),
                                   headers={'Content-Type': mime})

    class Proposer:
        def propose(self, public):
            seen.append(deepcopy(public))
            count = len(public['documents'])
            action = 'read_url' if count in (0, 2) else 'search_web'
            response = H['match_decision'](public, action)
            response['return_assessment'] = R['actual_review'](public)
            if response['return_assessment']:
                for citation in response['return_assessment']['citations']:
                    citation['quote'] = citation['quote'][:400]
            response[claims.FIELD] = []
            if action == 'read_url':
                response['url'] = public['source_catalog']['entries'][0]['url'] if count == 0 else URL
            else:
                response['query'] = public['match_input'] + ' source ' + ('a' if count == 1 else 'c')
                response['engine'] = 'bing'
            if count == 3 and actor_claim:
                # Citation provenance is valid, while the actor's predicate/value
                # is deliberately unsupported by the actual structured report.
                doc = public['documents'][-1]
                response[claims.FIELD] = [{'target_key': 'spftll:1001', 'subject': 'home',
                    'subject_name': '构造甲队', 'predicate': 'tactical_layout', 'value': actor_claim,
                    'document_id': doc['document_id'], 'quote': doc['excerpt'],
                    'validity': {'from': start, 'until': end, 'quote': doc['excerpt']}}]
            if count == 3 and interruption:
                if interruption == 'cancel': service.cancel(public['task_id'])
                elif interruption == 'deadline':
                    service.mutate(public['task_id'], lambda s: s.update(
                        deadline=(now()-timedelta(seconds=1)).isoformat()))
                else: raise ValueError('unsupported interruption')
            responses = [response]
            if count == 3:
                positive = deepcopy(response)
                positive['query'] = public['match_input'] + ' source b'
                responses.append(positive)
            return {'candidates': responses, 'recommended_index': 0, 'test_double': True}

    service = ExperienceService(root, proposer_factory=lambda f,e: Proposer(), mode='test_fixture',
        selection_policy=make_policy(), enable_return_assessment=True,
        tools_factory=lambda f: ExperienceWebTools(RawArchive(f), dns_check=lambda url: None,
                                                   session_factory=lambda route: Session()))
    run = service.create({'match': '2026-10-03 构造甲队 vs 构造乙队',
                          'limits': {'llm_calls': 7, 'http_calls': 6}})['run_id']
    return service, run, calls, seen


def run_scenario(root, scenario='current', **kwargs):
    service, run, calls, seen = fixture_runtime(root, scenario=scenario, **kwargs)
    service._drive(run, True)
    return service, run, calls, seen


def test_match_only_source_record_changes_selection_without_actor_claims(tmp_path):
    service, run, calls, seen = run_scenario(tmp_path)
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert [t['action'] for t in state['turns']] == ['read_url', 'search_web', 'read_url', 'search_web', 'stop']
        assert state['turns'][3]['query'].endswith(' source b')
        assert state['spent'] == {'http_calls': 4, 'llm_calls': 5}
        assert len(calls) == 4 and len(seen) == 5
        assert state['user_context'] == [] and state['question_to_user'] is None
        assert seen[2][support.STATE_FIELD]['context'][HOME_STATE] == 'unknown'
        actual = seen[3][support.STATE_FIELD]
        assert actual['context'][HOME_STATE] == 'qualified'
        assert actual['context']['home_reported_tactical_layout'] == 'unknown'
        assert actual['state']['home']['context']['rotation_risk'] == 'unknown'
        assert not actual['actor_assertions_used'] and not actual['world_truth_verified']
        assert all(t[claims.FIELD] == [] for t in state['turns'])
        report = verify_run(service.folder(run), include_action_value_ledger=True)
        assert report['source_assertions_verified'] == 0
        assert report['source_support_snapshots_verified'] == 5
        assert report['source_support_semantics'] == support.SEMANTICS
        assert not report['source_support_is_world_truth']
        assert report['real_calls_made_by_replay'] == 0
        budget = report['bounded_exploration']
        assert budget['reserved_actions'] == 2 and budget['reserved_declared_cost'] == .2
        assert [r['discovered_route']['request']['action'] for r in budget['reservations']] == ['search_web', 'read_url']
        assert all(r['value_status'] == 'VALUE_UNMEASURED' for r in budget['reservations'])
    finally: service.close()


@pytest.mark.parametrize('scenario,reason', [
    ('wrong_kickoff', 'source_fixture_mismatch'),
    ('other_opponent', 'source_fixture_mismatch'),
    ('other_competition', 'source_fixture_mismatch'),
    ('prose', 'source_format_not_supported_prose_unassessed'),
    ('expired', 'source_interval_expired'),
    ('not_started', 'source_interval_not_started'),
    ('duplicate_json', 'source_raw_json_ambiguous_or_invalid'),
])
def test_same_menu_history_and_costs_defer_when_source_does_not_support_fixture(tmp_path, scenario, reason):
    service, run, calls, seen = run_scenario(tmp_path, scenario)
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert [t['action'] for t in state['turns']] == ['read_url', 'search_web', 'read_url', 'defer']
        assert state['spent'] == {'http_calls': 3, 'llm_calls': 4} and len(calls) == 3
        actual = seen[-1][support.STATE_FIELD]
        assert actual['context'][HOME_STATE] == 'unknown'
        assert any(row['reason'] == reason for row in actual['excluded_documents'])
        if scenario == 'duplicate_json':
            assert actual['invalid_raw_json_document_ids'] == [seen[-1]['documents'][-1]['document_id']]
        assert state['user_context'] == [] and state['question_to_user'] is None
        report = verify_run(service.folder(run))
        assert report['bounded_exploration']['remaining_actions'] == 0
        assert report['source_support_snapshots_verified'] == 4
    finally: service.close()


@pytest.mark.parametrize('scenario', ['current', 'duplicate_json'])
def test_replay_rebuilds_raw_quarantine_even_after_all_derived_states_are_recomputed(tmp_path, scenario):
    service, run, _, _ = run_scenario(tmp_path, scenario)
    try:
        folder = service.folder(run)
        source_doc = next(d for d in service.get(run)['documents'] if d['url'] == URL)

        def forge(state):
            for op in state['operations']:
                path = folder/'operations'/op['operation_id']/'input.json'
                if not path.exists(): continue
                public = json.loads(path.read_text(encoding='utf-8'))
                if not any(d['document_id'] == source_doc['document_id'] for d in public.get('documents', [])):
                    continue
                # Inject a quarantine for good bytes, or erase it for ambiguous
                # bytes, then forge consistent projection/claim/input hashes.
                invalid_ids = [source_doc['document_id']] if scenario == 'current' else []
                public[support.STATE_FIELD] = support._project_context(public, invalid_raw_json_ids=invalid_ids)
                public[claims.STATE_FIELD] = claims.project_context(public)
                atomic_write(path, public)
                op['input_digest'] = digest(public)
                return
            raise AssertionError('missing source-bearing input')

        service.mutate(run, forge)
        with pytest.raises(ValueError, match='REPLAY_SOURCE_SUPPORT_STATE_MISMATCH'):
            verify_run(folder)
    finally: service.close()


@pytest.mark.parametrize('scenario', ['current', 'wrong_kickoff', 'other_opponent', 'expired', 'prose'])
def test_actor_supplied_tactics_never_promotes_an_unsupported_source(tmp_path, scenario):
    service, run, calls, seen = run_scenario(tmp_path, scenario, actor_claim='defensive')
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert seen[3][support.STATE_FIELD]['context']['home_reported_tactical_layout'] == 'unknown'
        assert state['turns'][3]['action'] == ('search_web' if scenario == 'current' else 'defer')
        if scenario == 'current': assert state['turns'][3]['query'].endswith(' source b')
        report = verify_run(service.folder(run))
        assert report['source_assertions_verified'] == 1
        assert not report['source_support_is_world_truth']
    finally: service.close()


@pytest.mark.parametrize('phase', ['plan', 'fetch'])
def test_changed_source_contract_pauses_before_provider_or_http_call(tmp_path, phase):
    service, run, calls, seen = fixture_runtime(tmp_path)
    try:
        for _ in range(5 if phase == 'fetch' else 6): service.step(run)
        before = len(calls), len(seen)
        config = model_package()
        config['source_context_contract']['sources'][0]['url'] = 'https://example.org/changed'
        service.selection_policy = make_policy(config)
        service.step(run)
        assert service.get(run)['status'] == 'paused'
        assert (len(calls), len(seen)) == before
        verify_run(service.folder(run))
    finally: service.close()


@pytest.mark.parametrize('interruption', ['cancel', 'deadline'])
def test_discarded_source_supported_plan_does_not_execute_or_reserve_again(tmp_path, interruption):
    service, run, calls, seen = run_scenario(tmp_path, interruption=interruption)
    try:
        state = service.get(run)
        assert len(state['turns']) == 3 and len(calls) == 3
        assert state['spent'] == {'http_calls': 3, 'llm_calls': 4}
        assert verify_run(service.folder(run))['bounded_exploration']['reserved_actions'] == 2
    finally: service.close()


@pytest.mark.parametrize('mutation', ['raw', 'support_state', 'source_excerpt', 'source_and_public_derived'])
def test_replay_rebuilds_source_support_beyond_recomputed_input_digest(tmp_path, mutation):
    service, run, _, seen = run_scenario(tmp_path)
    try:
        folder = service.folder(run)
        body_doc = next(d for d in service.get(run)['documents'] if d['url'] == URL)
        if mutation == 'raw':
            oid = body_doc['document_id'].removeprefix('doc-')
            raw = next((folder/'archive/raw/experience_web').glob('*/*_' + oid + '.body.gz'))
            raw.write_bytes(b'corrupt raw source bytes')
        else:
            def edit(state):
                for op in state['operations']:
                    path = folder/'operations'/op['operation_id']/'input.json'
                    if not path.exists(): continue
                    public = json.loads(path.read_text(encoding='utf-8'))
                    if not any(d['document_id'] == body_doc['document_id'] for d in public.get('documents', [])):
                        continue
                    if mutation == 'support_state':
                        public[support.STATE_FIELD]['context'][HOME_STATE] = 'must_win'
                    else:
                        doc = next(d for d in public['documents'] if d['document_id'] == body_doc['document_id'])
                        doc['excerpt'] = doc['excerpt'].replace('qualified', 'must_win')
                        if mutation == 'source_and_public_derived':
                            original = next(d for d in state['documents'] if d['document_id'] == body_doc['document_id'])
                            original['text'] = original['text'].replace('qualified', 'must_win')
                        visible_ids = {d['document_id'] for d in public['documents']}
                        originals = [d for d in state['documents'] if d['document_id'] in visible_ids]
                        public['fixture_identity'] = prepare_fixture_identity(
                            RawArchive(folder/'archive'), public, originals, include_local=True)
                        # Even a self-consistent re-projection is checked against
                        # actual archived bytes, not this actor-visible snapshot.
                        public[support.STATE_FIELD] = support.project_context(public)
                        public[claims.STATE_FIELD] = claims.project_context(public)
                    atomic_write(path, public)
                    op['input_digest'] = digest(public)
                    return
                raise AssertionError('missing source-bearing input')
            service.mutate(run, edit)
        reason = {'raw': None, 'support_state': 'SOURCE_SUPPORT_STATE_CHANGED',
                  'source_excerpt': 'REPLAY_SOURCE_CLAIM_VISIBLE_DOCUMENT_MISMATCH',
                  'source_and_public_derived': 'SOURCE_CLAIM_ARCHIVE_BODY_CHANGED'}[mutation]
        with pytest.raises((ValueError, OSError, EOFError), match=reason):
            verify_run(folder)
    finally: service.close()
