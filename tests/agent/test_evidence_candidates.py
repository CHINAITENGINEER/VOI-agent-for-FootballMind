from copy import deepcopy

import pytest

from football_agent.evidence_candidates import EvidenceDocumentCandidates, LEGACY_VERSION, VERSION
from football_agent.learned_voi_runtime import _actions
from football_agent.learned_voi_web import _web_action
from football_agent.match_evidence import build_match_evidence
from football_agent.match_source_profiles import resolve_profile


MATCH = {'match_id': 'fixture-one', 'home_team': 'Arsenal', 'away_team': 'Leeds',
         'kickoff_at': '2026-10-10T11:30:00Z', 'prediction_cutoff': '2026-10-09T11:30:00Z'}
COST = {'money': .01, 'requests': 1.}
NUMERIC = [{'action_id': 'results_view', 'group_ids': ['elo', 'recent_form'],
            'available': True, 'cost_status': 'simulated', 'cost': COST}]
HOME = 'https://www.leedsunited.com/en?article=26856'
ARTICLE = 'https://www.leedsunited.com/en/news/tanaka-helps-blue-samurai-to-kirin-cup-success'


def make(**kwargs):
    profile = resolve_profile('Arsenal', 'Leeds')
    options = dict(match_terms=['Arsenal', 'Leeds'], seed_urls=profile['seed_urls'],
                   allowed_hosts=profile['allowed_hosts'], numeric_actions=NUMERIC,
                   web_cost=COST, match=MATCH, profile=profile)
    options.update(kwargs)
    return EvidenceDocumentCandidates(**options)


def document(text='', *, url=HOME, links=None, status='OK', docid='doc-one'):
    return {'document_id': docid, 'status': status, 'kind': 'page', 'url': url,
            'raw_sha256': 'a'*64, 'text': text, 'links': links or [],
            'observed_at': '2026-10-06T09:00:00Z', 'available_at': '2026-10-06T09:00:01Z',
            'source_published_at': None}


def public(documents=()):
    return {'state': {'base': {'kickoff': MATCH['kickoff_at']}}, 'documents': list(documents),
            'evidence': build_match_evidence(MATCH, list(documents))}


def test_observed_uncertainty_creates_specific_query_after_return_only():
    generator = make()
    before = generator(public())
    news = document('Arsenal striker Kai Havertz may miss Leeds on 10 October 2026 with a muscle injury.',
                    url='https://www.premierleague.com/en/news/4727187')
    after = generator(public([news]))
    assert len([a for a in before if a.get('tool') == 'search_web']) == 1
    followups = [a for a in after if a.get('tool') == 'search_web' and 'Havertz' in a['query']]
    assert followups
    assert all(a['group_ids'] == [] and 'expected_gain' not in a for a in followups)
    assert all(a == _web_action('search_web', a['query'], cost=COST) for a in followups)
    assert _actions(after) == after


def test_official_home_discovers_player_article_without_complete_team_name():
    home = document('Leeds United Football Club', links=[{'url': ARTICLE, 'title': 'Tanaka helps Blue Samurai'}])
    actions = make()(public([home]))
    assert any(a.get('url') == ARTICLE for a in actions)
    foreign = document('Premier League news', url='https://www.premierleague.com/en/news',
                       links=[{'url': ARTICLE, 'title': 'Tanaka helps Blue Samurai'}])
    assert not any(a.get('url') == ARTICLE for a in make()(public([foreign])))


def test_navigation_commerce_and_cross_host_links_do_not_expand_menu():
    links = [{'url': 'https://www.leedsunited.com/en/news/category/mens', 'title': 'Mens'},
             {'url': 'https://www.leedsunited.com/en/news/ticket-information-arsenal-away', 'title': 'Arsenal tickets'},
             {'url': 'https://www.leedsunited.com/en/news/official-shirt-in-store', 'title': 'Leeds shop'},
             {'url': 'https://www.leedsunited.com/en/privacy-policy', 'title': 'Leeds privacy'},
             {'url': 'https://attacker.invalid/news/arsenal', 'title': 'Arsenal'},
             {'url': 'https://www.premierleague.com/en/news/unrelated', 'title': 'Unrelated player update'},
             {'url': 'https://www.leedsunited.com/en/news/arsenal-tactical-preview', 'title': 'Arsenal tactical preview'}]
    actions = make()(public([document('Leeds news', links=links)]))
    discovered = [a['url'] for a in actions if a.get('url') not in make().seed_urls and a.get('tool') == 'read_url']
    assert discovered == ['https://www.leedsunited.com/en/news/arsenal-tactical-preview']


def test_failed_page_cannot_supply_links_or_grounded_queries():
    failed = document('Arsenal may rotate against Leeds.', links=[{'url': ARTICLE, 'title': 'Read'}], status='FAILED')
    actions = make()(public([failed]))
    assert len([a for a in actions if a.get('tool') == 'search_web']) == 1
    assert not any(a.get('url') == ARTICLE for a in actions)


def test_only_followups_backed_by_admitted_evidence_enter_menu():
    payload = public()
    payload['evidence']['state']['followup_candidates'] = [
        {'tool': 'search_web', 'query': 'Arsenal made-up expensive query', 'based_on_evidence_ids': ['absent'],
         'expected_gain': None, 'proposal_status': 'grounded_query_not_value_ranked'}]
    assert len([a for a in make()(payload) if a.get('tool') == 'search_web']) == 1
    payload['evidence']['match_id'] = 'different-match'
    with pytest.raises(ValueError, match='another match'):
        make()(payload)


def test_exact_configuration_rebuild_and_mutation_detection():
    generator = make()
    payload = public([document('Leeds United', links=[{'url': ARTICLE, 'title': 'Read'}])])
    configuration = generator.configuration()
    rebuilt = EvidenceDocumentCandidates.from_configuration(configuration)
    assert rebuilt.configuration() == configuration
    assert rebuilt(payload) == generator(payload)
    configuration['candidate_generation_complete'] = True
    with pytest.raises(ValueError, match='reconstruct exactly'):
        EvidenceDocumentCandidates.from_configuration(configuration)


def test_large_menu_is_capped_and_keeps_numeric_actions():
    links = [{'url': 'https://www.leedsunited.com/en/news/player-' + str(i), 'title': 'Player update'} for i in range(100)]
    generator = make(max_candidates=9, max_discovered_links=32)
    actions = generator(public([document('Leeds United', links=links)]))
    assert len(actions) == 9
    assert actions[0]['action_id'] == 'results_view'
    assert len({a['information_key'] for a in actions}) == len(actions)
    assert generator.configuration()['candidate_generation_complete'] is False


def test_duplicate_observed_urls_collapse_and_page_order_is_stable():
    links = [{'url': ARTICLE, 'title': 'Read'}, {'url': ARTICLE + '#comments', 'title': 'Comments'}]
    actions = make()(public([document('Leeds United', links=links)]))
    assert sum(a.get('url') == ARTICLE for a in actions) == 1


@pytest.mark.parametrize('kwargs', [{'max_candidates': 65}, {'max_candidates': 0},
                                  {'max_followup_queries': 33}, {'max_discovered_links': 33}])
def test_invalid_bounds_rejected(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


def test_wrong_fixture_date_cannot_be_silently_queried():
    payload = public()
    payload['state']['base']['kickoff'] = '2026-10-11T11:30:00Z'
    with pytest.raises(ValueError, match='fixture date'):
        make()(payload)


def test_frozen_v1_configuration_and_lexical_menus_survive_source_order_v2():
    links = [{'url': 'https://www.leedsunited.com/en/news/zulu-current-story', 'title': 'Latest'},
             {'url': 'https://www.leedsunited.com/en/news/alpha-older-story', 'title': 'Older'}]
    payload = public([document('Leeds United', links=links)])
    old = make(discovery_version=1, max_discovered_links=1)
    current = make(discovery_version=2, max_discovered_links=1)
    old_config, new_config = old.configuration(), current.configuration()
    assert old_config['version'] == LEGACY_VERSION
    assert 'discovered_link_order' not in old_config
    assert new_config['version'] == VERSION
    assert new_config['discovered_link_order'] == 'document_then_observed_link_order_not_value_ranked'
    old_urls = [action.get('url') for action in old(payload)]
    new_urls = [action.get('url') for action in current(payload)]
    assert links[1]['url'] in old_urls and links[0]['url'] not in old_urls
    assert links[0]['url'] in new_urls and links[1]['url'] not in new_urls
    for config, generator in [(old_config, old), (new_config, current)]:
        restored = EvidenceDocumentCandidates.from_configuration(config)
        assert restored.configuration() == config
        assert restored(payload) == generator(payload)


def test_legacy_configuration_cannot_silently_claim_source_order():
    config = make(discovery_version=1).configuration()
    config['discovered_link_order'] = 'document_then_observed_link_order_not_value_ranked'
    with pytest.raises(ValueError, match='reconstruct exactly'):
        EvidenceDocumentCandidates.from_configuration(config)
