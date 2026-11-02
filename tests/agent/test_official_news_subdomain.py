"""Observed official news links remain actor discoveries, with exact replay."""
from copy import deepcopy

import pytest

from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.match_source_profiles import resolve_verified_profile


ROOT = 'https://canadasoccer.com/'
ARTICLE = 'https://news.canadasoccer.com/5-things-to-watch-during-the-october-fifa-international-window'
MATCH = {'match_id': 'constructed-canada-case', 'home_team': 'United States', 'away_team': 'Canada',
         'kickoff_at': '2026-10-07T00:00:00+00:00', 'prediction_cutoff': '2026-10-06T23:59:00+00:00'}


def binding():
    return {'schema': 'canada-senior-fixture-identity-v1', 'status': 'CONFIRMED',
            'canonical': {'source_url': ROOT + 'match/6630/', 'home_name': 'United States', 'away_name': 'Canada'},
            'binding_sha256': 'a' * 64}


def generator(version=4):
    profile = resolve_verified_profile(binding(), version=3)
    return EvidenceDocumentCandidates(match_terms=['United States', 'Canada'],
        seed_urls=profile['seed_urls'], allowed_hosts=profile['allowed_hosts'],
        numeric_actions=[], web_cost={'money': .01, 'requests': 1}, match=MATCH,
        profile=profile, discovery_version=version)


def public(*, status='OK', url=ROOT, links=()):
    return {'state': {'base': {'kickoff': MATCH['kickoff_at']}},
            'documents': [{'status': status, 'url': url, 'links': list(links)}]}


def test_observed_news_subdomain_enters_only_after_actual_home_response():
    actor = generator()
    before = actor({'state': {'base': {'kickoff': MATCH['kickoff_at']}}, 'documents': []})
    assert ARTICLE not in [a.get('url') for a in before]
    observed = public(links=[{'url': ARTICLE, 'title': 'Read story'}])
    after = actor(observed)
    assert ARTICLE in [a.get('url') for a in after]
    assert ARTICLE not in [a.get('url') for a in generator(3)(observed)]
    restored = EvidenceDocumentCandidates.from_configuration(actor.configuration())
    assert restored.configuration() == actor.configuration()
    assert restored(observed) == after
    assert all('expected_gain' not in a for a in after)


@pytest.mark.parametrize('options', [
    {'status': 'FAILED'}, {'url': ROOT+'match/6630/'},
    {'url': 'https://news.canadasoccer.com/unrelated-article'},
])
def test_only_the_registered_successful_entry_can_discover_nameless_news(options):
    assert ARTICLE not in [a.get('url') for a in generator()(public(
        links=[{'url': ARTICLE, 'title': 'Read story'}], **options))]


def test_navigation_commerce_and_unregistered_domains_stay_out():
    urls = ['https://news.canadasoccer.com', 'https://news.canadasoccer.com/tags',
            'https://news.canadasoccer.com/shop-official-shirt',
            'https://news.canadasoccer.com.evil.invalid/october-window',
            'https://unregistered.example/news/canada']
    actions = generator()(public(links=[{'url': url, 'title': 'Read story'} for url in urls]))
    assert not set(urls) & {a.get('url') for a in actions}


@pytest.mark.parametrize('version', [1, 2])
def test_new_fixture_requires_explicit_new_source_profile(version):
    with pytest.raises(ValueError, match='version 3'):
        resolve_verified_profile(binding(), version=version)


def test_subdomain_is_a_registered_capability_without_preselected_articles():
    profile = resolve_verified_profile(binding(), version=3)
    assert profile['allowed_hosts'] == ['canadasoccer.com', 'news.canadasoccer.com']
    assert profile['seed_urls'] == [ROOT, ROOT+'match/6630/']
    assert profile['selection_basis'] == 'verified_identity_to_capabilities_not_information_value'
