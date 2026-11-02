"""Offline Finnish article capability regression, including archived real links."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import socket

import pytest

from football_agent.evidence_candidates import EvidenceDocumentCandidates


ROOT = Path(__file__).resolve().parents[2]
HOST = 'www.veikkausliiga.com'
ENTRY = 'https://'+HOST+'/'
ARTICLE = ENTRY+'uutiset/2026/10/06/observed-player-update'
MATCH = {'match_id': 'constructed:unregistered-pair', 'home_team': 'HJK', 'away_team': 'VPS',
         'kickoff_at': '2026-10-08T15:00:00+00:00', 'prediction_cutoff': '2026-10-08T14:00:00+00:00'}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail('Candidate tests use only constructed or existing archived bytes')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)
    monkeypatch.setattr(socket, 'getaddrinfo', deny)


def make(version=10, *, cls=EvidenceDocumentCandidates, profile_mutation=None, **kwargs):
    profile = {'version': 'verified-match-source-profiles-v4',
        'identity_schema': 'finland-stage-fixture-identity-v1',
        'teams': [{'side': side, 'canonical_name': team, 'registration_status': 'raw_verified_fixture'}
                  for side, team in [('home', 'HJK'), ('away', 'VPS')]],
        'entries': [{'kind': 'news_index', 'url': ENTRY, 'club': 'HJK', 'side': 'home',
                     'source_scope': 'verified_competition_shared_entry'}]}
    if profile_mutation:
        profile_mutation(profile)
    return cls(match_terms=['HJK', 'VPS'], seed_urls=[ENTRY], allowed_hosts=[HOST], numeric_actions=[],
        web_cost={'money': .01, 'requests': 1}, match=MATCH, profile=profile, discovery_version=version, **kwargs)


def public(links=None):
    return {'state': {'match_id': MATCH['match_id'], 'base': {'kickoff': MATCH['kickoff_at']},
                     'prediction_cutoff': MATCH['prediction_cutoff']},
        'decision_at': '2026-10-07T08:00:00+00:00',
        'documents': [{'document_id': 'observed-home', 'status': 'OK', 'url': ENTRY,
            'raw_sha256': 'a'*64, 'kind': 'index', 'text': '',
            'observed_at': '2026-10-07T07:00:00+00:00', 'available_at': '2026-10-07T07:01:00+00:00',
            'links': links if links is not None else [{'url': ARTICLE, 'title': 'Observed player update'}]}]}


def discovered(generator, value):
    return [a['url'] for a in generator(value) if a.get('tool') == 'read_url' and a['url'] not in generator.seed_urls]


def test_real_archived_home_links_become_reachable_without_topics_or_match_ids():
    receipt = json.loads((ROOT/'var/parallel/agent/round196/transport-check/receipt.json').read_text(encoding='utf-8'))
    value = public()
    value['documents'] = [receipt['document']]
    observed = [link['url'] for link in receipt['document']['links'] if '/uutiset/2026/' in link['url']]
    assert len(observed) == 16
    assert discovered(make(9), value) == []
    current = make()
    assert discovered(current, value) == observed[:12]
    changed = deepcopy(value)
    changed['state']['match_id'] = 'a-completely-different-provider-id'
    assert current(changed) == current(value)
    action = next(a for a in current(value) if a.get('url') == observed[0])
    assert action['cost']['requests'] == 1 and 'expected_gain' not in action
    # Reading the home consumes 1 of 4 requests. Its article is a distinct
    # feasible request costing 1, so discovery now leaves room to read it.
    assert 1+action['cost']['requests'] <= 4


def test_new_route_requires_the_actual_visible_parent_and_registered_source():
    assert discovered(make(), public()) == [ARTICLE]
    for mutation in ['failed', 'absent', 'future', 'before_observed', 'foreign_parent', 'cutoff', 'no_clock']:
        value = public()
        if mutation == 'failed': value['documents'][0]['status'] = 'FAILED'
        elif mutation == 'absent': value['documents'][0]['links'] = []
        elif mutation == 'future': value['documents'][0]['available_at'] = '2026-10-07T09:00:00+00:00'
        elif mutation == 'before_observed': value['documents'][0]['observed_at'] = '2026-10-07T07:02:00+00:00'
        elif mutation == 'foreign_parent': value['documents'][0]['url'] = 'https://attacker.invalid/'
        elif mutation == 'cutoff': value['state']['prediction_cutoff'] = '2026-10-07T06:00:00+00:00'
        else: value.pop('decision_at')
        assert discovered(make(), value) == [], mutation
    for field in ['identity_schema', 'source_scope', 'registration_status']:
        def remove(profile):
            if field == 'identity_schema': profile.pop(field)
            elif field == 'source_scope': profile['entries'][0].pop(field)
            else: profile['teams'][0][field] = 'unverified'
        assert discovered(make(profile_mutation=remove), public()) == []


@pytest.mark.parametrize('path', [
    'uutiset/2026/02/30/nonexistent-calendar-day', 'uutiset/2026/13/01/invalid-month',
    'uutiset/2026/10/article-without-day', 'uutiset/2026/10/06/',
    'uutiset/2026/10/06/123456', 'uutiset/2026/10/06/page',
    'uutiset/2026/10/06/shop-opening', 'uutiset/2026/10/06/actual-story?next=other',
    'uutiset/2026/10/06/actual-story/nested/',
])
def test_invalid_calendar_or_navigation_paths_do_not_gain_capability(path):
    assert discovered(make(), public([{'url': ENTRY+path, 'title': 'Observed'}])) == []


def test_calendar_valid_leap_day_and_source_order_roundtrip():
    links = [{'url': ENTRY+'uutiset/2024/02/29/observed-leap-day', 'title': 'Observed'},
             {'url': ARTICLE, 'title': 'Observed'}]
    generator = make(max_discovered_links=1)
    assert discovered(generator, public(links)) == [links[0]['url']]
    restored = EvidenceDocumentCandidates.from_configuration(generator.configuration())
    assert restored.configuration() == generator.configuration()
    assert restored(public(links)) == generator(public(links))


def test_observed_candidate_passes_strict_tls_prepare_without_execution():
    from football_agent.news_discovery import NewsDiscoveryTools
    generator = make()
    value = public()
    value['task'] = {'seed_urls': generator.seed_urls}
    action = next(a for a in generator(value) if a.get('url') == ARTICLE)
    tools = NewsDiscoveryTools(None, [HOST], default_route='direct',
        route_overrides={HOST: 'system_tls'}, index_urls=[ENTRY], projection_version=5)
    prepared = tools.prepare(action, value)
    assert prepared == {'tool': 'read_url', 'endpoint': ARTICLE, 'params': {}, 'source': 'open_page'}
    assert ARTICLE in tools._system_tls_urls
    with pytest.raises(ValueError, match='seeds or acquired links'):
        tools.prepare(action, {**value, 'documents': []})


def test_v1_to_v9_keep_exact_original_configurations_and_menus():
    path = ROOT/'var/parallel/agent/round197/source-before/football_agent/evidence_candidates.py'
    spec = importlib.util.spec_from_file_location('football_agent._before_r197_candidates', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = public([{'url': ARTICLE, 'title': 'Observed'},
                    {'url': ENTRY+'uutiset/legacy-flat-story', 'title': 'Observed'},
                    {'url': ENTRY+'2026/10/06/legacy-root-date', 'title': 'Observed'}])
    for version in range(1, 10):
        current, previous = make(version), make(version, cls=module.EvidenceDocumentCandidates)
        assert current.configuration() == previous.configuration(), version
        assert current(value) == previous(value), version
