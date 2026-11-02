"""Observed date permalinks are capabilities, never article/value labels."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from football_agent.evidence_candidates import EvidenceDocumentCandidates, DATED_ARTICLE_VERSION
from test_evidence_candidates import make, document, public


HOST = 'https://www.leedsunited.com'


def discovered(generator, payload):
    return [a['url'] for a in generator(payload)
            if a.get('tool') == 'read_url' and a['url'] not in generator.seed_urls]


@pytest.mark.parametrize('path', ['/2026/10/ordinary-update/', '/2024/02/29/player-update/',
                                  '/2026/10/uutisia-öljy/'])
def test_actual_official_dated_link_adds_capability_without_gain(path):
    url = HOST + path
    payload = public([document(links=[{'url': url, 'title': 'Observed announcement'}])])
    old, new = make(discovery_version=7), make(discovery_version=8)
    assert discovered(old, payload) == []
    assert discovered(new, payload) == [url]
    action = next(a for a in new(payload) if a.get('url') == url)
    assert 'expected_gain' not in action and action['group_ids'] == []
    assert new.configuration()['version'] == DATED_ARTICLE_VERSION
    assert EvidenceDocumentCandidates.from_configuration(new.configuration())(payload) == new(payload)


@pytest.mark.parametrize('path', ['/2026/00/story', '/2026/13/story', '/2026/02/29/story',
    '/2026/04/31/story', '/0000/01/story', '/2026/10/', '/2026/10/06/',
    '/2026/10/category/', '/2026/10/feed/', '/2026/10/fixtures/', '/2026/10/file.pdf',
    '/2026/10/picture.jpg', '/2026/10/tickets/', '/2026/10/foo/bar'])
def test_invalid_calendar_navigation_or_static_resources_are_not_new_capabilities(path):
    payload = public([document(links=[{'url': HOST + path, 'title': 'Update'}])])
    assert discovered(make(discovery_version=8), payload) == []


def test_absent_failed_foreign_and_unregistered_parent_cannot_supply_date_links():
    generator = make(discovery_version=8)
    url = HOST + '/2026/10/ordinary-update/'
    assert discovered(generator, public()) == []
    assert discovered(generator, public([document(status='FAILED', links=[{'url': url}])])) == []
    assert discovered(generator, public([document(links=[{'url': 'https://www.premierleague.com/2026/10/leeds-update'}])])) == []
    assert discovered(generator, public([document(url='https://www.premierleague.com/en/news', links=[{'url': url}])])) == []


def test_observed_order_duplicate_cap_and_sidecar_remain():
    first, second = HOST+'/2026/10/zulu/', HOST+'/2026/10/alpha/'
    payload = public([document(links=[{'url': first, 'title': 'First'}, {'url': first+'#body', 'title': 'Duplicate'},
                                      {'url': second, 'title': 'Second'}])])
    payload['decision_at'] = '2026-10-07T09:00:00Z'
    payload['state']['prediction_cutoff'] = '2026-10-09T11:30:00Z'
    generator = make(discovery_version=8, max_discovered_links=1)
    actions = generator(payload)
    assert discovered(generator, payload) == [first]
    assert generator.candidate_contexts(payload, actions) is not None


def test_old_versions_match_exact_prechange_implementation():
    saved = Path('var/parallel/agent/round194/source-before/football_agent/evidence_candidates.py')
    spec = importlib.util.spec_from_file_location('football_agent._round194_before_dated', saved)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = public([document(links=[{'url': HOST+'/2026/10/example', 'title': 'Leeds'},
                                      {'url': HOST+'/en/news/real-article', 'title': 'Leeds'}])])
    for version in range(1, 8):
        current = make(discovery_version=version)
        old = module.EvidenceDocumentCandidates.from_configuration(current.configuration())
        assert old.configuration() == current.configuration()
        assert old(payload) == current(payload)


def test_actual_gnistan_archive_derivation_keeps_original_run_unchanged():
    folder = Path('var/parallel/agent/round194/live-match/analysis/match-3085210')
    original = (folder/'cycle-001.json').read_bytes()
    cycle = json.loads(original)
    config = json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
    cfg = config['supervisor_metadata']['generator']
    old = EvidenceDocumentCandidates.from_configuration(cfg)
    fields = ('match_terms', 'seed_urls', 'allowed_hosts', 'numeric_actions', 'web_cost', 'match',
              'profile', 'max_discovered_links', 'max_followup_queries', 'max_candidates')
    new = EvidenceDocumentCandidates(**{k: deepcopy(cfg[k]) for k in fields}, discovery_version=8)
    before, after = old(cycle['public_before']), new(cycle['public_before'])
    additions = [a for a in after if a not in before]
    actual = {link['url'] for doc in cycle['public_before']['documents'] for link in doc['links']}
    assert len(additions) == 4
    assert all(a.get('url') in actual and 'expected_gain' not in a for a in additions)
    assert any('fc-inter' in a['url'] for a in additions)
    assert (folder/'cycle-001.json').read_bytes() == original
