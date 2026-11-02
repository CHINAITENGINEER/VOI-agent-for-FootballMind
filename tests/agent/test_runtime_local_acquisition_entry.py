"""Match-only acquisition of hidden local odds through the durable executor.

Every HTTP/model response and gain-history row here is a synthetic fixture.
Declared-cost exploration is not estimated VoI, source truth, or a real-feed
evaluation. The same menu can select local data or web data at different costs.
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
from football_agent.acquisition_replay import verify_run
from football_agent.experience import ExperienceService, now
from football_agent.experience_sources import ExperienceWebTools
from football_agent.local_evidence import capability as local_capability
from football_agent.runtime_voi_model import RuntimeVoIModel
from football_agent.state_bound_runtime_policy import StateBoundVoISelectionPolicy
from football_agent.structured_sources import registered_request
from football_spider.storage.raw_archive import RawArchive

P = runpy.run_path(str(Path(__file__).with_name('test_runtime_source_support_entry.py')))
H, R = P['H'], P['R']
MATCH = '2027-10-01 窗口外目标甲队 vs 构造乙队'
COSTS = {'search_web': {'bing': .1}, 'read_url': .1, 'query_acquired': .05}
OPTIONS = {'max_actions': 1, 'max_declared_cost': .2}


def make_policy(*, costs=None, options=None):
    return StateBoundVoISelectionPolicy(RuntimeVoIModel(P['model_package']()),
        allow_synthetic=True, exploration=options or OPTIONS,
        discovered_routes=costs or COSTS)


def _response(public, action):
    response = H['match_decision'](public, action)
    response['return_assessment'] = R['actual_review'](public)
    if response['return_assessment']:
        for citation in response['return_assessment']['citations']:
            citation['quote'] = citation['quote'][:400]
    response[claims.FIELD] = []
    response['local_query'] = None
    return response


def fixture_runtime(root, *, costs=None, options=None, http_calls=3, page_limit=20,
                    interruption=None):
    """Future fixture, original archive, hidden odds, and a multi-action menu."""
    calls, seen = [], []
    rows = [H['fixture_row'](matchId=1000+i, home='构造甲队'+str(i),
                matchTime='2027-10-01T18:00:00',
                spfWinFoot=2.1, spfEqualFoot=3.2, spfLoseFoot=3.4,
                goalFoot=-1, rspfWinFoot=3.7, rspfEqualFoot=3.8, rspfLoseFoot=1.8)
            for i in range(80)]
    rows[-1]['home'] = '窗口外目标甲队'
    payload = {'code': 0, 'data': [{'list': rows}]}

    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            calls.append(url)
            if registered_request(url):
                body, mime = json.dumps(payload, ensure_ascii=False), 'application/json'
            else:
                body, mime = '<html><article>No new fixture evidence.</article></html>', 'text/html'
            return SimpleNamespace(status_code=200, content=body.encode('utf-8'),
                                   headers={'Content-Type': mime})

    class Proposer:
        def propose(self, public):
            seen.append(deepcopy(public))
            count = len(public['documents'])
            if count == 0:
                response = _response(public, 'read_url')
                response['url'] = public['source_catalog']['entries'][1]['url']
                candidates = [response]
            elif count == 1:
                response = _response(public, 'query_acquired')
                response['active_question_id'] = 'fixture'
                response['local_query'] = {
                    'document_id': public['documents'][0]['document_id'],
                    'record_kind': 'fixture', 'source_match_id': None,
                    'team_name': '窗口外目标甲队', 'team_match': 'normalized_exact',
                    'offset': 0, 'limit': 20}
                candidates = [response]
            else:
                web = _response(public, 'search_web')
                web.update(query=public['match_input']+' 当前报价覆盖'+(' 补充来源' if count > 2 else ''), engine='bing',
                    unknown_to_resolve='本场已采集的赔率记录是否覆盖目标盘口',
                    useful_if='返回绑定本场源比赛 ID 的当前报价；报价本身不证明阵容或战术',
                    why_worth_cost='比较尚未读取的本地证据与追加网络请求的声明成本',
                    analysis_plan=[{'id': 'quote_coverage',
                        'question': '本场尚未读取的报价覆盖哪些盘口',
                        'decision_relevance': '明确已有市场信息的覆盖与缺失', 'status': 'open'}],
                    active_question_id='quote_coverage')
                local = deepcopy(web)
                local.update(action='query_acquired', query=None, engine=None)
                pages = [d for d in public['documents']
                         if d.get('local_evidence', {}).get('query', {}).get('record_kind') == 'odds']
                offset = pages[-1]['local_evidence']['next_offset'] if pages else 0
                local['local_query'] = {
                    'document_id': public['documents'][0]['document_id'],
                    'record_kind': 'odds', 'source_match_id': '1079',
                    'team_name': None, 'team_match': 'exact',
                    'offset': offset if offset is not None else 0, 'limit': page_limit}
                # The actor recommends web; eligibility, declared cost, and
                # allowance decide. No imperative to fetch every local page.
                candidates = [web, local] if offset is not None else [web]
                if count == 2 and interruption:
                    if interruption == 'cancel': service.cancel(public['task_id'])
                    elif interruption == 'deadline':
                        service.mutate(public['task_id'], lambda state: state.update(
                            deadline=(now()-timedelta(seconds=1)).isoformat()))
                    elif interruption == 'invalid':
                        for candidate in candidates: candidate['task_id'] = 'wrong-task'
                    else: raise ValueError('unsupported interruption')
            return {'candidates': candidates, 'recommended_index': 0, 'test_double': True}

    service = ExperienceService(root, mode='test_fixture',
        proposer_factory=lambda f,e: Proposer(),
        selection_policy=make_policy(costs=costs, options=options), enable_return_assessment=True,
        tools_factory=lambda f: ExperienceWebTools(RawArchive(f), dns_check=lambda url: None,
                                                   session_factory=lambda route: Session()))
    run = service.create({'match': MATCH, 'limits': {'llm_calls': 8, 'http_calls': http_calls}})['run_id']
    return service, run, calls, seen


def run_scenario(root, **kwargs):
    service, run, calls, seen = fixture_runtime(root, **kwargs)
    service._drive(run, True)
    return service, run, calls, seen


def _reserved(report):
    return report['bounded_exploration']['reservations']


def test_match_only_resolves_hidden_fixture_then_selects_hidden_odds(tmp_path):
    service, run, calls, seen = run_scenario(tmp_path)
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert [t['action'] for t in state['turns']] == ['read_url', 'query_acquired', 'query_acquired', 'defer']
        assert '窗口外目标甲队' not in seen[1]['documents'][0]['excerpt']
        assert not any(row['kind'] == 'odds' for row in seen[2]['documents'][0]['structured_records'])
        assert seen[2]['fixture_identity']['resolution']['status'] == 'resolved'
        assert seen[2]['fixture_identity']['resolution']['target_key'] == 'spftll:1079'
        assert state['local_usage']['reads'] == 2
        assert state['spent'] == {'http_calls': 1, 'llm_calls': 4} and len(calls) == 1
        assert state['user_context'] == [] and state['question_to_user'] is None
        page = seen[-1]['documents'][-1]
        assert page['structured_records'] and all(row['kind'] == 'odds' and row['source_match_id'] == '1079'
                                                 for row in page['structured_records'])
        assert page['observed_at'] == seen[1]['documents'][0]['observed_at']
        assert page['source_published_at'] is None and page['request_id'] is None
        assert page['raw_sha256'] == seen[1]['documents'][0]['raw_sha256']
        assert page['local_evidence']['query']['team_name'] is None
        report = verify_run(service.folder(run), include_action_value_ledger=True)
        assert report['local_documents_reconstructed'] == report['linked_local_reads'] == 2
        assert report['real_calls_made_by_replay'] == 0 and not report['numeric_voi_calibrated']
        reservations = _reserved(report)
        assert len(reservations) == 1 and reservations[0]['declared_cost'] == .05
        assert reservations[0]['discovered_route']['request']['action'] == 'query_acquired'
        assert reservations[0]['value_status'] == 'VALUE_UNMEASURED'
        assert reservations[0]['realized_information_gain'] is None
        assert not reservations[0]['numeric_voi_training_eligible']
        selected = [e for e in report['action_value_ledger']['entries'] if e['selection_basis'] == exploration.BASIS]
        assert len(selected) == 1 and len(selected[0]['candidate_value_assessments']) == 2
        assert selected[0]['expected_information_value'] is None
    finally: service.close()


def test_same_menu_can_choose_web_when_local_declared_cost_is_higher(tmp_path):
    costs = {**COSTS, 'query_acquired': .15}
    service, run, calls, seen = run_scenario(tmp_path, costs=costs)
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert [t['action'] for t in state['turns']] == ['read_url', 'query_acquired', 'search_web', 'defer']
        assert state['local_usage']['reads'] == 1 and len(calls) == 2
        reservation = _reserved(verify_run(service.folder(run)))[0]
        assert reservation['declared_cost'] == .1
        assert reservation['discovered_route']['request']['action'] == 'search_web'
        assert state['user_context'] == [] and seen[2]['fixture_identity']['resolution']['status'] == 'resolved'
    finally: service.close()


def test_exhausted_http_budget_still_permits_local_information_acquisition(tmp_path):
    service, run, calls, seen = run_scenario(tmp_path, http_calls=1)
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert seen[2]['remaining']['http_calls'] == 0 and not seen[2]['final_decision_only']
        assert state['turns'][2]['action'] == 'query_acquired'
        assert len(calls) == 1 and state['local_usage']['reads'] == 2
        assert len(_reserved(verify_run(service.folder(run)))) == 1
    finally: service.close()


def test_next_page_uses_verified_next_offset_and_shared_exploration_allowance(tmp_path):
    service, run, calls, seen = run_scenario(tmp_path, http_calls=1, page_limit=2,
        options={'max_actions': 2, 'max_declared_cost': .1})
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert [t['action'] for t in state['turns']] == ['read_url', 'query_acquired', 'query_acquired', 'query_acquired', 'defer']
        pages = [d for d in state['documents'] if d.get('local_evidence', {}).get('query', {}).get('record_kind') == 'odds']
        assert len(pages) == 2
        assert pages[0]['local_evidence']['next_offset'] == pages[1]['local_evidence']['offset'] == 2
        assert pages[1]['local_evidence']['next_offset'] == 4
        assert state['local_usage']['reads'] == 3 and len(calls) == 1
        report = verify_run(service.folder(run))
        assert report['bounded_exploration']['reserved_declared_cost'] == .1
        assert len(_reserved(report)) == 2 and report['bounded_exploration']['remaining_actions'] == 0
    finally: service.close()


@pytest.mark.parametrize('allowance,expected_odds_pages', [(.04, 0), (.05, 1)])
def test_local_reads_are_not_free_and_declared_cost_allowance_stops_pages(tmp_path, allowance, expected_odds_pages):
    service, run, calls, _ = run_scenario(tmp_path, http_calls=1, page_limit=2,
        options={'max_actions': 2, 'max_declared_cost': allowance})
    try:
        state = service.get(run)
        assert state['last_error'] is None and state['turns'][-1]['action'] == 'defer'
        assert state['local_usage']['reads'] == 1+expected_odds_pages and len(calls) == 1
        assert len(_reserved(verify_run(service.folder(run)))) == expected_odds_pages
    finally: service.close()


@pytest.mark.parametrize('phase', ['plan', 'fetch'])
def test_changed_local_cost_profile_pauses_before_another_provider_or_read(tmp_path, phase):
    service, run, calls, seen = fixture_runtime(tmp_path)
    try:
        for _ in range(4 if phase == 'plan' else 5): service.step(run)
        before = (len(calls), len(seen), service.get(run)['local_usage']['reads'])
        service.selection_policy = make_policy(costs={**COSTS, 'query_acquired': .06})
        service.step(run)
        assert service.get(run)['status'] == 'paused'
        assert (len(calls), len(seen), service.get(run)['local_usage']['reads']) == before
        assert len(_reserved(verify_run(service.folder(run)))) == int(phase == 'fetch')
    finally: service.close()


@pytest.mark.parametrize('interruption', ['cancel', 'deadline', 'invalid'])
def test_unaccepted_local_information_action_never_reserves(tmp_path, interruption):
    service, run, calls, _ = run_scenario(tmp_path, interruption=interruption)
    try:
        state = service.get(run)
        assert state['local_usage']['reads'] == 1 and len(calls) == 1
        assert not any(exploration.RESERVATION in o for o in state['operations'])
        assert len(_reserved(verify_run(service.folder(run)))) == 0
    finally: service.close()


def test_exhausted_execution_local_budget_blocks_queued_query_without_network_fallback(tmp_path):
    service, run, calls, seen = fixture_runtime(tmp_path)
    try:
        for _ in range(5): service.step(run)
        assert service.get(run)['queued_action']['local_query']['record_kind'] == 'odds'
        before = (len(calls), len(seen), len(service.get(run)['documents']))
        # Exercise the executor guard, not a synthetic claim of a replayable
        # history with eight real local reads.
        service.mutate(run, lambda state: state['local_usage'].update(reads=local_capability()['max_reads']))
        service.step(run)
        assert service.get(run)['status'] == 'paused'
        assert (len(calls), len(seen), len(service.get(run)['documents'])) == before
    finally: service.close()


@pytest.mark.parametrize('mutation', ['cost', 'request'])
def test_replay_rejects_tampered_local_exploration_reservation(tmp_path, mutation):
    service, run, _, _ = run_scenario(tmp_path)
    try:
        def forge(state):
            op = next(o for o in state['operations'] if exploration.RESERVATION in o)
            route = op[exploration.RESERVATION]['discovered_route']
            if mutation == 'cost': route['cost'] = 0
            else: route['request']['local_query']['offset'] = 2
        service.mutate(run, forge)
        with pytest.raises(ValueError, match='RESERVATION_MISMATCH'):
            verify_run(service.folder(run))
    finally: service.close()


def test_replay_reconstructs_local_pages_from_original_archive(tmp_path):
    service, run, _, _ = run_scenario(tmp_path)
    try:
        assert verify_run(service.folder(run))['local_documents_reconstructed'] == 2
        raw = next((service.folder(run)/'archive').rglob('*.body.gz'))
        raw.write_bytes(b'corrupted synthetic archive')
        with pytest.raises((ValueError, OSError), match='REPLAY|FIXTURE|PLATFORM|gzip'):
            verify_run(service.folder(run))
    finally: service.close()
