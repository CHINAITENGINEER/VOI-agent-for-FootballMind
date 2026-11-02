"""Constructed transport fixtures exercise the complete registered-source bridge.

The event IDs, clocks and final scores here are test data, never real labels.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import socket

import pytest

from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.match_evidence import build_match_evidence
from football_agent.match_source_profiles import resolve_verified_profile
from football_research.data_adapters.finland_stage_fixture import (
    REGISTERED_TEAMS, RESULTS_URL, RESULTS_PARAMS,
)
from football_research.data_adapters.live_fixture_preparation import (
    LEGACY_VERSION, VERSION, _registry, prepare_live_fixture,
)
from football_research.data_adapters.verified_fixtures import (
    verify_identity_binding, verify_independent_settlement,
)
from football_research.data_adapters.finland_result_capture import (
    capture_finland_result, detail_url, verify_capture_receipts,
)
from football_spider.storage.raw_archive import RawArchive

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 7, 10, tzinfo=timezone.utc)
KICKOFF = datetime(2026, 10, 8, 15, tzinfo=timezone.utc)
FINAL = KICKOFF + timedelta(hours=3)
ARTICLE = 'https://www.veikkausliiga.com/uutiset/actual-observed-preview/'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail('Only constructed archived transport is allowed')
    monkeypatch.setattr(socket, 'getaddrinfo', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    monkeypatch.setattr(socket.socket, 'connect', deny)


class Source:
    def __init__(self, home='383', away='2226', native='9876543', provider='3987654'):
        self.home_id, self.away_id, self.native, self.provider = home, away, native, provider
        self.home, self.away = [REGISTERED_TEAMS['spftll:team:'+ident] for ident in (home, away)]
        self.calls = []
        self.fail_stage = None
        self.wrong_group = False
        self.alias = None

    def schedule(self, score='-'):
        options = ''.join(f'<option value="{team["native_team_id"]}">{team["name"]}</option>'
                          for team in REGISTERED_TEAMS.values())
        route = '/tilastot/2026/veikkausliiga/ottelut/'+self.native+'/'
        return f'''<html><title>Otteluohjelma | Veikkausliiga</title>
        <a href="/">League</a><a href="/uutiset/">Uutiset</a>
        <select id="season"><option value="2026" selected>2026</option></select>
        <select id="category"><option value="veikkausliiga" selected>Veikkausliiga</option></select>
        <select name="group"><option value="{'3' if self.wrong_group else '2'}" selected>Mestaruussarja</option></select>
        <select name="team">{options}</select><table><tr data-time="20261008">
        <td>1</td><td>To 8.10.2026</td><td>18:00</td><td>8.10.2026</td>
        <td><a href="{route}">{self.home['name']} - {self.away['name']}</a></td>
        <td><a href="{route}seuranta/">Seuranta</a></td><td>{score}</td><td>1000</td>
        </tr></table></html>'''.encode()

    def terminal(self):
        return f'''<html><meta property="og:url" content="{RESULTS_URL+self.native+'/seuranta/'}">
        <main><section class="game-header"><div class="head finished">
        <div class="home"><div class="team-name"><a>{self.home['name']}</a></div></div>
        <div class="away"><div class="team-name"><a>{self.away['name']}</a></div></div>
        <span id="home-goals">2</span><span id="away-goals">1</span>
        <div class="game-state">Pelattu 08.10.2026</div></div></section>
        <div class="game-events"><p class="info center">Ottelu päättyi</p>
        <p class="info center">Puoliaika päättyi</p><p class="info center">Puoliaika alkoi</p>
        <p class="info center">Puoliaika päättyi</p><p class="info center">Puoliaika alkoi</p>
        <p class="info center">Ottelu alkoi</p></div></main></html>'''.encode()

    def __call__(self, spec, *, archive, clock):
        self.calls.append(deepcopy(spec))
        stage = spec['stage']
        assert (archive.root.parent/'requests'/(stage+'.intent.json')).exists() or (
            archive.root/'requests'/(stage+'.intent.json')).exists()
        fixture = {'matchId': int(self.provider), 'matchTime': '2026-10-08T23:00:00',
            'homeId': int(self.home_id), 'home': self.alias or self.home['provider_names'][0],
            'awayId': int(self.away_id), 'away': self.away['provider_names'][0],
            'leagueId': 13, 'league': '芬超', 'footMatchState': 0,
            'homeScore': None, 'awayScore': None}
        if stage == 'fixture':
            body = json.dumps({'code': 0, 'data': fixture}).encode()
        elif stage == 'quote':
            body = json.dumps({'code': 0, 'data': {'footballMoreSpInfo': {
                'matchId': int(self.provider), 'spf_win': 2.0, 'spf_draw': 3.2, 'spf_lost': 3.0}}}).encode()
        elif stage == 'schedule' and spec['source'] == 'spftll':
            body = json.dumps({'code': 0, 'data': [{'matchList': [fixture]}]}).encode()
        elif stage == 'finland_championship_schedule':
            body = self.schedule()
        elif stage == 'schedule':
            body = self.schedule('2 — 1')
        elif stage == 'detail':
            body = self.terminal()
        else:
            raise AssertionError(stage)
        return archive.save(spec['source'], spec['url'], spec['params'], body,
            status=503 if stage == self.fail_stage else 200, observed_at=clock(),
            context={'requested_at': spec['requested_at'],
                     'preparation_request_digest': spec['request_digest']})


def prepare(tmp_path, source=None, **kwargs):
    source = source or Source()
    result = prepare_live_fixture(source.provider, output=tmp_path/'prepared', clock=lambda: NOW,
                                  transport=source, **kwargs)
    return source, result


def binding(result):
    return json.loads(Path(result['identity_bundle_path']).read_text(encoding='utf-8'))['bindings'][result['match_id']]


@pytest.mark.parametrize('home,away', [('383', '2226'), ('1986', '2228'), ('2228', '383')])
def test_new_pairs_use_observed_native_id_not_two_fixture_hardcodes(tmp_path, home, away):
    source, result = prepare(tmp_path, Source(home, away))
    assert result['status'] == 'ready', result
    assert result['schema'] == VERSION and result['archived_http_responses'] == 3
    assert [x['stage'] for x in source.calls] == ['fixture', 'quote', 'finland_championship_schedule']
    saved = binding(result)
    assert saved['canonical']['source_event_id'] == 'veikkausliiga:2026:match:9876543'
    assert verify_identity_binding(saved, archive_root=result['identity_archive'], as_of=NOW) == saved
    assert detail_url(saved) == RESULTS_URL+'9876543/seuranta/'
    assert result['collection']['source_profile_version'] == 4
    assert result['collection']['candidate_discovery_version'] == 10
    assert result['collection']['news_projection_version'] == 5
    assert result['collection']['evidence_version'] == 'match-evidence-extraction-v5'
    assert ARTICLE not in result['collection']['seed_urls'] and result['labels'] == []
    assert all('/news/' not in url for url in result['collection']['seed_urls'])
    again = prepare_live_fixture(source.provider, output=tmp_path/'prepared', resume=True,
                                 clock=lambda: NOW, transport=source)
    assert again == result and len(source.calls) == 3


def test_english_named_query_uses_registered_teams_and_beijing_date(tmp_path):
    source = Source()
    result = prepare_live_fixture({'home': 'HJK', 'away': 'VPS', 'date': '2026-10-08'},
        output=tmp_path/'prepared', clock=lambda: NOW, transport=source)
    assert result['status'] == 'ready'
    assert [x['stage'] for x in source.calls] == ['schedule', 'quote', 'finland_championship_schedule']


@pytest.mark.parametrize('problem,expected,count', [('alias','unsupported',1), ('group','unavailable',3),
                                                   ('budget','budget_exhausted',2), ('failure','unavailable',3)])
def test_unsupported_or_incomplete_identity_never_becomes_ready(tmp_path, problem, expected, count):
    source = Source()
    if problem == 'alias': source.alias = 'HJK Women'
    if problem == 'group': source.wrong_group = True
    if problem == 'failure': source.fail_stage = 'finland_championship_schedule'
    _, result = prepare(tmp_path, source, max_http_requests=2 if problem == 'budget' else 6)
    assert result['status'] == expected and len(source.calls) == count
    assert not Path(result['identity_bundle_path']).exists()
    if problem == 'failure':
        again = prepare_live_fixture(source.provider, output=tmp_path/'prepared', resume=True,
                                     clock=lambda: NOW, transport=source)
        assert again == result and len(source.calls) == count


def test_result_capture_reuses_two_half_gateway_and_keeps_not_due_pending(tmp_path, monkeypatch):
    import football_research.data_adapters.settlement as settlement
    class SimulatedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return FINAL.astimezone(tz) if tz is not None else FINAL.replace(tzinfo=None)
    monkeypatch.setattr(settlement, 'datetime', SimulatedClock)
    source, result = prepare(tmp_path)
    saved = binding(result)
    pending = capture_finland_result(saved, identity_archive_root=result['identity_archive'],
        output=tmp_path/'pending', clock=lambda: NOW, transport=source)
    assert pending['status'] == 'PENDING' and pending['settlement_rows'] == [] and len(source.calls) == 3
    assert verify_capture_receipts(saved, tmp_path/'pending', as_of=NOW)['http_requests_actual'] == 0
    final = capture_finland_result(saved, identity_archive_root=result['identity_archive'],
        output=tmp_path/'finished', clock=lambda: FINAL, transport=source)
    assert final['status'] == 'VERIFIED' and len(source.calls) == 5
    assert final['settlement_rows'][0]['score_period'] == 'regulation'
    assert final['settlement_rows'][0]['available_at'] == FINAL.isoformat()
    report = verify_capture_receipts(saved, tmp_path/'finished', as_of=FINAL)
    assert report['http_requests_actual'] == 2
    raw = RawArchive(tmp_path/'finished')
    assert verify_independent_settlement(saved, identity_archive_root=result['identity_archive'],
        result_archive_root=raw.root, observation_ids=[x.id for x in raw.observations()], as_of=FINAL) == final


def generator(saved, version=9, profile=None):
    canonical = saved['canonical']
    profile = profile or resolve_verified_profile(saved, version=4)
    match = {'match_id': saved['match_id'], 'home_team': canonical['home_name'],
        'away_team': canonical['away_name'], 'kickoff_at': canonical['kickoff'],
        'prediction_cutoff': (KICKOFF-timedelta(minutes=1)).isoformat()}
    instance = EvidenceDocumentCandidates(match_terms=[match['home_team'], match['away_team']],
        seed_urls=profile['seed_urls'], allowed_hosts=profile['allowed_hosts'], numeric_actions=[],
        web_cost={'requests':1, 'money':.01}, match=match, profile=profile, discovery_version=version)
    return instance, match


def public(match, *, status='OK', url='https://www.veikkausliiga.com/uutiset/', links=None):
    documents = [{'document_id':'constructed-index', 'status':status, 'url':url, 'kind':'page',
        'text':'', 'structured_news':{'role':'index'}, 'raw_sha256':'a'*64,
        'observed_at':NOW.isoformat(), 'available_at':NOW.isoformat(), 'source_published_at':None,
        'links':links if links is not None else [{'url':ARTICLE,'title':'Observed headline'}]}]
    return {'state':{'base':{'kickoff':match['kickoff_at']}}, 'documents':documents,
            'evidence':build_match_evidence(match, documents, version='match-evidence-extraction-v5')}


def test_v9_discovers_only_actual_same_host_links_and_preserves_v8_menu(tmp_path):
    _, result = prepare(tmp_path)
    saved = binding(result)
    current, match = generator(saved)
    prior, _ = generator(saved, version=8)
    payload = public(match)
    assert ARTICLE not in {a.get('url') for a in prior(payload)}
    assert ARTICLE in {a.get('url') for a in current(payload)}
    assert ARTICLE not in {a.get('url') for a in current(public(match, links=[]))}
    assert ARTICLE not in {a.get('url') for a in current(public(match, status='FAILED'))}
    assert ARTICLE not in {a.get('url') for a in current(public(match, url='https://attacker.invalid/'))}
    action = next(a for a in current(payload) if a.get('url') == ARTICLE)
    assert 'expected_gain' not in action and action['cost']['requests'] == 1
    config = current.configuration()
    assert EvidenceDocumentCandidates.from_configuration(config).configuration() == config
    assert EvidenceDocumentCandidates.from_configuration(config)(payload) == current(payload)


def test_news_route_requires_explicit_new_profile_binding_and_ignores_navigation(tmp_path):
    _, result = prepare(tmp_path)
    saved = binding(result)
    profile = resolve_verified_profile(saved, version=4)
    profile.pop('identity_schema')
    unsafe, match = generator(saved, profile=profile)
    assert ARTICLE not in {a.get('url') for a in unsafe(public(match))}
    current, match = generator(saved)
    links = [{'url':url,'title':'Navigation'} for url in (
        'https://www.veikkausliiga.com/uutiset/', 'https://www.veikkausliiga.com/uutiset/page/',
        'https://www.veikkausliiga.com/uutiset/tag/example/',
        'https://attacker.invalid/uutiset/actual-observed-preview/')]
    expected = {a.get('url') for a in current(public(match, links=[]))}
    assert {a.get('url') for a in current(public(match, links=links))} == expected
    with pytest.raises(ValueError, match='version 4'):
        resolve_verified_profile(saved, version=3)


def test_legacy_capability_inventory_matches_original_frozen_preparation():
    path = ROOT/'var/parallel/agent/round194/live-match/preparation/prepare-plan.json'
    plan = json.loads(path.read_text(encoding='utf-8'))
    from football_research.acquisition.value import _canonical
    assert plan['schema'] == LEGACY_VERSION
    assert sha256(_canonical(_registry(LEGACY_VERSION)).encode()).hexdigest() == plan['registry_sha256']
    assert 'finland_stage_teams' not in _registry(LEGACY_VERSION)


def test_old_gnistan_profile_bytes_are_unchanged_under_current_dispatch():
    path = ROOT/'var/parallel/agent/round194/live-match/preparation/identity-bundle.json'
    saved = next(iter(json.loads(path.read_text(encoding='utf-8'))['bindings'].values()))
    source = ROOT/'var/parallel/agent/round196/source-before/football_agent/match_source_profiles.py'
    spec = importlib.util.spec_from_file_location('football_agent._r196_old_profiles', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    assert resolve_verified_profile(saved, version=3) == old.resolve_verified_profile(saved, version=3)
    assert detail_url(saved).endswith('/4353542/seuranta/')
