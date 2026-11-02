from copy import deepcopy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from football_agent.acquisition_replay import verify_run
from football_agent.experience import ExperienceService, public_state, validate_decision
from football_agent.experience_sources import ExperienceWebTools, project_body
from football_agent.structured_sources import (source_catalog, registered_request,
    project_platform, detail_links, verify_platform_document)
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive

helpers = runpy.run_path(str(Path(__file__).with_name('test_match_experience.py')))
match_decision, Provider = helpers['match_decision'], helpers['Provider']
FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures/collectors/spftll'


def fixture_row(**changes):
    return {'matchId': 1001, 'home': '构造甲队', 'away': '构造乙队', 'league': '构造杯',
            'matchTime': '2026-10-01T18:00:00', 'finishedTitle': '未开',
            'homeScore': None, 'awayScore': None, **changes}


def recent_payload(rows=None):
    return {'code': 0, 'data': [{'matchList': rows if rows is not None else [fixture_row()]}]}


class FakeSession:
    def __init__(self, route, calls, payloads):
        self.calls, self.payloads = calls, payloads

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def get(self, url, **kwargs):
        self.calls.append({'url': url, **kwargs})
        spec = registered_request(url)
        payload = self.payloads[spec['kind']]
        if isinstance(payload, Exception): raise payload
        status, data = payload if isinstance(payload, tuple) else (200, payload)
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        return SimpleNamespace(status_code=status, content=body, headers={'Content-Type': 'application/json'})


def make_tools(root, calls, payloads):
    return ExperienceWebTools(RawArchive(root), dns_check=lambda url: None,
        session_factory=lambda route: FakeSession(route, calls, payloads))


def fetch_once(tmp_path, payload, kind='recent'):
    calls, reserved = [], []
    tools = make_tools(tmp_path, calls, {kind: payload})
    url = source_catalog()['entries'][0]['url'] if kind == 'recent' else detail_links('1001')[0 if kind == 'detail' else 1]['url']
    def reserve(url):
        reserved.append(url)
        return 'fixture-request-' + str(len(reserved))
    result = tools.execute({'tool': 'read_url', 'url': url}, run_id='fixture-run', reserve_request=reserve)
    return result, calls, tools.archive


def test_catalog_exposes_capabilities_without_prefetch_or_user_source_design(tmp_path):
    service = ExperienceService(tmp_path, mode='test_fixture')
    try:
        state = service.create({'match': '构造甲队 vs 构造乙队'})
        public = public_state(state)
        assert state['spent'] == {'llm_calls': 0, 'http_calls': 0}
        assert not state['documents'] and not state['config']['seed_urls']
        assert len(public['source_catalog']['entries']) == 2
        action = match_decision(public, 'read_url')
        action['url'] = public['source_catalog']['entries'][0]['url']
        validate_decision(action, public)
        action['url'] = detail_links('1001')[0]['url']
        with pytest.raises(ValueError, match='URL_NOT_OBSERVED'):
            validate_decision(action, public)
    finally:
        service.close()


@pytest.mark.parametrize('url', [
    'https://example.org/api/match/GetList',
    'http://justpost.haoyun999.cn/api/match/GetList',
    'https://justpost.haoyun999.cn/api/match/GetList?date=2020-01-01',
    'https://justpost.haoyun999.cn/api/match/GetInfo?gameKind=0&matchId=1001&matchId=1002',
    'https://justpost.haoyun999.cn/api/match/GetInfo?gameKind=0&matchId=guess',
    'https://user:pass@justpost.haoyun999.cn/api/match/GetList',
])
def test_unregistered_requests_do_not_gain_platform_capability(url):
    assert registered_request(url) is None


def test_single_acquisition_reuses_normalizer_and_preserves_raw_binding(tmp_path):
    result, calls, archive = fetch_once(tmp_path, recent_payload())
    document = result['document']
    assert len(calls) == len(result['requests']) == 1
    assert calls[0]['headers']['Referer'] == 'https://xx.spftll.cn/'
    assert 'token' not in calls[0]['headers'] and calls[0]['allow_redirects'] is False
    row = json.loads(document['text'])
    assert row['score'] == {'home': None, 'away': None}
    assert row['score_period'] == 'unknown' and row['home']['id_quality'] == 'match_scoped'
    assert document['source_published_at'] is None and document['semantic_relevance'] == 'unassessed'
    assert document['platform']['quality'] == 'normalized'
    assert document['links'] == detail_links('1001')
    assert verify_platform_document(archive, document)['raw_sha256'] == document['raw_sha256']


@pytest.mark.parametrize('payload,expected', [
    ({'code': 0, 'data': []}, 'empty_structured'),
    ({'code': 0, 'data': None}, 'empty_structured'),
    ({'code': 1, 'data': []}, 'invalid_content'),
    ({'code': 0}, 'invalid_content'),
    ({'code': 0, 'data': {}}, 'invalid_content'),
    ({'code': 0, 'data': [{'unexpected': []}]}, 'invalid_content'),
    (b'<html>challenge</html>', 'invalid_content'),
    ((403, b'Forbidden'), 'unavailable'),
])
def test_empty_invalid_and_access_failure_never_become_fixture_facts(tmp_path, payload, expected):
    result, calls, archive = fetch_once(tmp_path, payload)
    document = result['document']
    assert document['status'] == expected and not document['text'] and not document['links']
    assert len(calls) == 1
    verify_platform_document(archive, document)


def test_partial_invalid_rows_keep_unknowns_and_validated_rows_separate(tmp_path):
    rows = [fixture_row(), fixture_row(matchId=1002, homeScore=-1), fixture_row(matchId=1003, home='')]
    result, _, archive = fetch_once(tmp_path, recent_payload(rows))
    doc = result['document']
    assert doc['platform']['quality'] == 'partial' and doc['platform']['issues'] == 2
    assert doc['platform']['matches'] == 1 and len(doc['text'].splitlines()) == 1
    assert doc['links'] == detail_links('1001')
    verify_platform_document(archive, doc)


@pytest.mark.parametrize('kind,payload', [
    ('detail', {'code': 0, 'data': fixture_row(matchId=1002)}),
    ('odds', {'code': 0, 'data': {'footballMoreSpInfo': {'matchId': 1002, 'spf_win': 2.0}}}),
])
def test_detail_and_odds_id_must_match_observed_request(tmp_path, kind, payload):
    result, _, archive = fetch_once(tmp_path, payload, kind)
    assert result['document']['status'] == 'invalid_content'
    assert not result['document']['links']
    verify_platform_document(archive, result['document'])


@pytest.mark.parametrize('name', ['recent', 'list', 'detail', 'odds'])
def test_existing_redacted_real_captures_parse_offline_without_ingestion(name):
    meta = json.loads((FIXTURES / (name + '.metadata.json')).read_text(encoding='utf-8'))
    assert meta['sample_kind'] == 'real_capture_redacted_copy' and not meta['production_ingestion_allowed']
    obs = Observation.model_validate(meta['observation'])
    url = obs.endpoint + ('?' + urlencode(obs.params) if obs.params else '')
    body = (FIXTURES / (name + '.json')).read_bytes()
    doc = project_platform(obs, body, project_body(body, url, 200, 'application/json'))
    assert doc['status'] == 'readable' and doc['platform']['quality'] in {'normalized', 'partial'}
    assert doc['platform']['matches'] + doc['platform']['quotes'] > 0
    if name == 'odds':
        quotes = [json.loads(line) for line in doc['text'].splitlines()]
        assert {q['market_type'] for q in quotes} == {'1x2', 'jc_handicap_3way', 'exact_total_goals'}
        assert all(q['period'] == 'regulation' for q in quotes)


def test_projection_bound_preserves_complete_rows_and_admits_incomplete_coverage(tmp_path):
    result, _, archive = fetch_once(tmp_path, recent_payload([fixture_row(matchId=1001+i) for i in range(80)]))
    doc = result['document']
    assert doc['text_truncated'] and len(doc['text']) <= 12000
    assert doc['platform']['records_displayed'] < doc['platform']['matches'] == 80
    assert all(json.loads(line)['kind'] == 'fixture' for line in doc['text'].splitlines())
    verify_platform_document(archive, doc)


def test_no_request_without_reservation(tmp_path):
    calls = []
    tools = make_tools(tmp_path, calls, {'recent': recent_payload()})
    action = {'tool': 'read_url', 'url': source_catalog()['entries'][0]['url']}
    result = tools.execute(action, run_id='fixture', reserve_request=lambda url: None)
    assert result['document'] is None and not calls and not list(tmp_path.rglob('*.observation.json'))


def test_raw_projection_tampering_is_detected(tmp_path):
    result, _, archive = fetch_once(tmp_path, recent_payload())
    changed = deepcopy(result['document'])
    changed['text'] = changed['text'].replace('构造甲队', '假冒球队')
    with pytest.raises(ValueError, match='PLATFORM_PROJECTION_REPLAY_MISMATCH'):
        verify_platform_document(archive, changed)
    changed = deepcopy(result['document'])
    changed['source_published_at'] = changed['observed_at']
    with pytest.raises(ValueError, match='PLATFORM_OBSERVATION_BINDING_MISMATCH'):
        verify_platform_document(archive, changed)
    next((archive.root/'raw').rglob('*.body.gz')).write_bytes(b'corrupted')
    with pytest.raises((ValueError, OSError, EOFError)):
        verify_platform_document(archive, result['document'])


def test_runtime_selects_catalog_then_observed_detail_with_full_offline_replay(tmp_path):
    calls, seen = [], []
    def choose(public):
        seen.append(deepcopy(public))
        count = len(public['documents'])
        response = match_decision(public, 'read_url' if count < 2 else 'stop')
        if count == 0:
            response['url'] = public['source_catalog']['entries'][0]['url']
        elif count == 1:
            response['url'] = public['documents'][0]['links'][0]['url']
        return response
    service = ExperienceService(tmp_path, mode='test_fixture',
        provider_factory=lambda f,e: Provider(choose),
        tools_factory=lambda f: make_tools(f, calls, {'recent': recent_payload(), 'detail': {'code':0, 'data':fixture_row()}}))
    try:
        run = service.create({'match': '构造甲队 vs 构造乙队'})['run_id']
        assert not calls
        service._drive(run, True)
        state = service.get(run)
        assert state['status'] == 'deferred'
        assert state['spent'] == {'llm_calls': 3, 'http_calls': 2}
        assert [t['action'] for t in state['turns']] == ['read_url', 'read_url', 'stop']
        assert len(calls) == 2 and not state['user_context']
        assert not seen[0]['documents'] and seen[1]['documents'][0]['platform']['source'] == 'spftll'
        visible = seen[1]['documents'][0]
        assert visible['structured_records'] == [json.loads(line) for line in visible['excerpt'].splitlines()]
        assert visible['platform']['records_visible'] == 1
        audit = verify_run(service.folder(run))
        assert audit['platform_documents_reconstructed'] == audit['linked_http_receipts'] == 2
        assert audit['real_calls_made_by_replay'] == 0
    finally:
        service.close()


def test_public_structured_records_use_only_complete_visible_lines(tmp_path):
    result, _, _ = fetch_once(tmp_path/'archive',recent_payload([fixture_row(matchId=1001+i) for i in range(50)]))
    service=ExperienceService(tmp_path/'runtime',mode='test_fixture')
    try:
        state=service.create({'match':'构造甲队 vs 构造乙队'})
        state['documents']=[result['document']]
        public=public_state(state)
        visible=public['documents'][0]
        assert visible['excerpt_truncated'] and len(visible['excerpt'])<=4500
        assert 0<len(visible['structured_records'])<result['document']['platform']['records_displayed']
        assert visible['structured_records']==[json.loads(line) for line in visible['excerpt'].splitlines()]
        assert 'records_visible' not in result['document']['platform']
    finally:
        service.close()


@pytest.mark.parametrize('after_selection', [False, True])
def test_catalog_change_pauses_before_spending_or_fetching(tmp_path, monkeypatch, after_selection):
    calls=[]
    def choose(public):
        response=match_decision(public, 'read_url')
        response['url']=public['source_catalog']['entries'][0]['url']
        return response
    service=ExperienceService(tmp_path, mode='test_fixture', provider_factory=lambda f,e: Provider(choose),
        tools_factory=lambda f: make_tools(f,calls,{'recent':recent_payload()}))
    try:
        run=service.create({'match':'构造甲队 vs 构造乙队'})['run_id']
        if after_selection: service.step(run)
        changed=source_catalog(); changed['version']='different-fixture-version'
        monkeypatch.setattr('football_agent.experience.source_catalog',lambda: changed)
        service._drive(run,True)
        state=service.get(run)
        assert state['status']=='paused' and not calls
        assert state['spent']=={'http_calls':0,'llm_calls':int(after_selection)}
    finally:
        service.close()
