"""Navigation can expose links without being promoted to query evidence."""
from copy import deepcopy

import pytest

from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.match_source_profiles import resolve_verified_profile

ROOT = 'https://canadasoccer.com/'
ARTICLE = 'https://news.canadasoccer.com/observed-october-training-update'
MATCH = {'match_id': 'constructed-canada-case', 'home_team': 'United States', 'away_team': 'Canada',
         'kickoff_at': '2026-10-07T00:00:00+00:00', 'prediction_cutoff': '2026-10-06T23:59:00+00:00'}


def generator(version=5):
    binding = {'schema': 'canada-senior-fixture-identity-v1', 'status': 'CONFIRMED',
        'canonical': {'source_url': ROOT+'match/6630/', 'home_name': 'United States', 'away_name': 'Canada'},
        'binding_sha256': 'a'*64}
    profile = resolve_verified_profile(binding, version=3)
    return EvidenceDocumentCandidates(match_terms=['United States', 'Canada'],
        seed_urls=profile['seed_urls'], allowed_hosts=profile['allowed_hosts'], numeric_actions=[],
        web_cost={'money': .01, 'requests': 1}, match=MATCH, profile=profile, discovery_version=version)


def public(url=ROOT+'match/6630/', status='OK', links=None):
    return {'state': {'base': {'kickoff': MATCH['kickoff_at']}}, 'documents': [
        {'document_id': 'doc-1', 'url': url, 'status': status, 'kind': 'page',
         'structured_news': {'role': 'unknown'}, 'links': links or []}]}


@pytest.mark.parametrize('source', [ROOT, ROOT+'match/6630/',
    'https://news.canadasoccer.com/already-read-article'])
def test_registered_source_links_are_observed_capabilities_not_seeded_articles(source):
    actor = generator()
    initial = public(source)
    assert ARTICLE not in {a.get('url') for a in actor(initial)}
    observed = public(source, links=[{'url': ARTICLE, 'title': 'Read story'}])
    actions = actor(observed)
    assert ARTICLE in {a.get('url') for a in actions}
    assert all('expected_gain' not in a for a in actions)
    restored = EvidenceDocumentCandidates.from_configuration(actor.configuration())
    assert restored(observed) == actions
    if source != ROOT:
        assert ARTICLE not in {a.get('url') for a in generator(4)(observed)}


@pytest.mark.parametrize('status', ['FAILED', 'NO_DATA'])
def test_unsuccessful_official_response_cannot_expand_menu(status):
    observed = public(status=status, links=[{'url': ARTICLE, 'title': 'Read story'}])
    assert ARTICLE not in {a.get('url') for a in generator()(observed)}


def test_navigation_commerce_and_lookalike_domain_not_article_candidates():
    urls = ['https://news.canadasoccer.com/', 'https://news.canadasoccer.com/tags',
        'https://news.canadasoccer.com/shop-official-shirt',
        'https://news.canadasoccer.com.evil.invalid/observed-training',
        'https://canadasoccer.com/directory', 'https://canadasoccer.com/coaching']
    actions = generator()(public(links=[{'url': u, 'title': 'Read story'} for u in urls]))
    assert not set(urls) & {a.get('url') for a in actions}


def test_v5_does_not_consume_legacy_navigation_evidence_followups():
    observed = public()
    query = 'Canada Coach Education TELUS Directory'
    observed['evidence'] = {'schema_version': 'match-evidence-extraction-v3',
        'match_id': MATCH['match_id'], 'evidence': [{'evidence_id': 'e-1', 'anchor': {'document_id': 'doc-1'}}],
        'state': {'followup_candidates': [{'tool': 'search_web', 'query': query,
            'expected_gain': None, 'proposal_status': 'grounded_query_not_value_ranked',
            'based_on_evidence_ids': ['e-1']}]}}
    old = deepcopy(observed)
    assert query in {a.get('query') for a in generator(4)(observed)}
    assert query not in {a.get('query') for a in generator()(observed)}
    assert observed == old


def test_actual_source_order_preserved_without_topic_weights():
    urls = ['https://news.canadasoccer.com/z-training-update',
            'https://news.canadasoccer.com/a-team-update']
    actions = generator()(public(links=[{'url': u, 'title': 'Read story'} for u in urls]))
    assert [a['url'] for a in actions if a.get('url') in urls] == urls


def test_new_html_candidate_version_keeps_its_own_evidence_contract():
    actor = generator(6)
    assert actor.configuration()['followup_evidence_version'] == 'match-evidence-extraction-v5'
    assert generator(5).configuration()['followup_evidence_version'] == 'match-evidence-extraction-v4'
    restored = EvidenceDocumentCandidates.from_configuration(actor.configuration())
    assert restored(public()) == actor(public())
    changed = actor.configuration()
    changed['followup_evidence_version'] = 'match-evidence-extraction-v4'
    with pytest.raises(ValueError, match='reconstruct'):
        EvidenceDocumentCandidates.from_configuration(changed)
