from copy import deepcopy

import pytest

from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.match_source_profiles import resolve_verified_profile

BINDING = {'status': 'CONFIRMED', 'binding_sha256': 'a' * 64, 'canonical': {
    'home_name': 'Estonia', 'away_name': 'Slovakia',
    'source_url': 'https://www.uefa.com/uefanationsleague/match/2046001/'}}
MATCH = {'match_id': 'synthetic-national-fixture', 'home_team': 'Estonia', 'away_team': 'Slovakia',
         'kickoff_at': '2026-10-10T18:45:00Z', 'prediction_cutoff': '2026-10-10T18:44:00Z'}
INDEXES = ['https://jalgpall.ee/koondis/1/uudised', 'https://futbalsfz.sk/?menu=39']
ARTICLES = ['https://jalgpall.ee/koondis/1/uudised/koondis-valmistub',
            'https://futbalsfz.sk/muzi-a-priprava-muzstva/']  # Synthetic parser fixtures, not captured URLs.


def generator(version=3, profile=None):
    profile = profile or resolve_verified_profile(BINDING, version=2)
    return EvidenceDocumentCandidates(match_terms=['Estonia', 'Slovakia'], match=MATCH,
        seed_urls=profile['seed_urls'], allowed_hosts=profile['allowed_hosts'], profile=profile,
        numeric_actions=[], web_cost={'requests': 1.}, discovery_version=version)


def public(source=None, links=(), status='OK'):
    return {'state': {'base': {'kickoff': MATCH['kickoff_at']}}, 'documents': [] if source is None else [
        {'status': status, 'url': source, 'links': [{'url': url, 'title': 'Local text'} for url in links]}]}


def test_registered_entries_require_confirmed_canonical_identity_and_opt_in():
    old = resolve_verified_profile(BINDING)
    assert old == resolve_verified_profile(BINDING, version=1)
    assert not set(INDEXES) & set(old['seed_urls'])
    assert set(INDEXES) <= set(resolve_verified_profile(BINDING, version=2)['seed_urls'])
    for status in ['UNCONFIRMED', 'DISPUTED']:
        with pytest.raises(ValueError, match='confirmed'):
            resolve_verified_profile({**BINDING, 'status': status}, version=2)
    other = deepcopy(BINDING)
    other['canonical'].update(home_name='Scotland', away_name='Estonia Women')
    assert resolve_verified_profile(other, version=2) == {
        **resolve_verified_profile(other), 'version': 'verified-match-source-profiles-v2'}


@pytest.mark.parametrize('index,article', list(zip(INDEXES, ARTICLES)))
def test_local_articles_are_observed_same_host_only_and_old_menus_replay(index, article):
    payload = public(index, [article])
    for version in [1, 2, 3]:
        original = generator(version)
        rebuilt = EvidenceDocumentCandidates.from_configuration(original.configuration())
        assert rebuilt.configuration() == original.configuration()
        assert rebuilt(payload) == original(payload)
        assert (article in [a.get('url') for a in original(payload)]) == (version == 3)
    new = generator()
    for blocked in [public(), public(index, [article], 'FAILED'),
                    public(ARTICLES[0] if index == INDEXES[0] else ARTICLES[1], [article]),
                    public(INDEXES[1] if index == INDEXES[0] else INDEXES[0], [article])]:
        assert article not in [a.get('url') for a in new(blocked)]


@pytest.mark.parametrize('index,bad', [
    (INDEXES[0], ['https://jalgpall.ee/koondis/1/uudised/', 'https://jalgpall.ee/koondis/1/uudised/category']),
    (INDEXES[1], ['https://futbalsfz.sk/', 'https://futbalsfz.sk/muzi-a-/',
                  'https://futbalsfz.sk/muzi-a-category/', 'https://futbalsfz.sk/category/muzi-a/'])])
def test_local_indexes_and_category_routes_are_not_articles(index, bad):
    proposed = [a.get('url') for a in generator()(public(index, bad))]
    assert not set(bad) & set(proposed)
