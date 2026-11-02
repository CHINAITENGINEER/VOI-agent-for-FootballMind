"""Versioned metadata order must neither fetch bodies nor hide source caps."""
from copy import deepcopy

import pytest

from football_agent.evidence_candidates import EvidenceDocumentCandidates, VISIBLE_ORDER_VERSION
from football_agent.learned_voi_runtime import _actions
from football_agent.learned_voi_web import _web_action


COST = {'money': .01, 'requests': 1.}
HOME = 'https://www.arsenal.com/news'
AWAY = 'https://www.leedsunited.com/en/news'
FIXTURES = 'https://www.arsenal.com/fixtures'
MATCH = {'match_id': 'fixture-test', 'home_team': '阿森纳', 'away_team': '利兹联',
         'home_aliases': ['Arsenal'], 'away_aliases': ['Leeds', 'Leeds United'],
         'kickoff_at': '2026-10-10T11:30:00Z', 'prediction_cutoff': '2026-10-10T11:00:00Z'}
PROFILE = {'version': 'test-frozen-source-profile-v1',
           'teams': [{'side': 'home', 'canonical_name': 'Arsenal'},
                     {'side': 'away', 'canonical_name': 'Leeds United'}],
           'entries': [{'kind': 'news_index', 'url': HOME, 'side': 'home', 'club': 'Arsenal'},
                       {'kind': 'news_index', 'url': AWAY, 'side': 'away', 'club': 'Leeds United'},
                       {'kind': 'fixture_index', 'url': FIXTURES}]}


def generator(version=12, **kwargs):
    options = dict(match_terms=['Arsenal', 'Leeds United'], seed_urls=[HOME, AWAY, FIXTURES],
                   allowed_hosts=['www.arsenal.com', 'www.leedsunited.com'], numeric_actions=[],
                   web_cost=COST, match=MATCH, profile=PROFILE, discovery_version=version)
    options.update(kwargs)
    return EvidenceDocumentCandidates(**options)


def article(slug, title='', **kwargs):
    return {'url': f'https://www.leedsunited.com/en/news/{slug}', 'title': title, **kwargs}


def document(links, **kwargs):
    return {'document_id': 'directory', 'status': 'OK', 'kind': 'index', 'url': AWAY,
            'observed_at': '2026-10-09T08:00:00Z', 'available_at': '2026-10-09T08:00:01Z',
            'raw_sha256': 'a' * 64, 'text': 'Visible source directory', 'links': links, **kwargs}


def public(documents=(), attempts=()):
    return {'state': {'base': {'kickoff': MATCH['kickoff_at']},
                      'prediction_cutoff': MATCH['prediction_cutoff']},
            'decision_at': '2026-10-09T09:00:00Z', 'documents': list(documents),
            'attempts': list(attempts)}


def test_v12_starts_with_away_then_home_news_and_keeps_all_previous_actions():
    new, old = generator(), generator(11)
    payload = public()
    actions = new(payload)
    assert [row.get('url') for row in actions] == [AWAY, HOME, FIXTURES, None]
    assert {row['action_id'] for row in actions} == {row['action_id'] for row in old(payload)}
    assert actions == _actions(actions)
    assert old(payload)[0]['tool'] == 'search_web'
    assert 'visible_metadata_order' not in old.configuration()


def test_observed_articles_precede_remaining_directory_with_auditable_cues_and_rank():
    links = [article('first', 'Club community update'),
             article('second', 'Training and injury update'),
             article('third', 'Arsenal preview for 10 October 2026')]
    payload = public([document(links)])
    new = generator()
    actions = new(payload)
    assert [row['url'] for row in actions[:4]] == [links[2]['url'], links[1]['url'], links[0]['url'], HOME]
    ordering = new.candidate_contexts(payload, actions)['ordering']
    third = ordering['by_action_id'][actions[0]['action_id']]
    assert third['source_rank'] == 3
    assert third['source_observation']['link_index'] == 2
    assert third['source_observation']['cues']['opponent_and_full_fixture_date'] is True
    assert ordering['caps']['article_cap_shortage'] is False
    assert all(row == _web_action(row['tool'], row.get('url', row.get('query')), cost=COST)
               for row in actions)


def test_ties_keep_source_order_and_body_changes_cannot_change_candidates():
    links = [article('z-last-alphabet', 'Team news'), article('a-first-alphabet', 'Team news')]
    payload = public([document(links)])
    new = generator()
    actions = new(payload)
    assert [row['url'] for row in actions[:2]] == [row['url'] for row in links]
    changed = deepcopy(payload)
    changed['documents'][0]['text'] = 'Arsenal injury confirmed on 10 October 2026. UNSEEN BODY.'
    changed['documents'][0]['structured_news'] = {'segments': [{'text': 'Hidden target body'}]}
    assert new(changed) == actions
    assert new.candidate_contexts(changed, actions)['ordering'] == new.candidate_contexts(payload, actions)['ordering']


def test_known_news_channel_is_kept_but_cannot_be_promoted_to_body_article():
    channel = article('first-team', 'Arsenal injury news 2026-10-10')
    body = article('player-update', 'Club update')
    payload = public([document([channel, body])])
    new = generator()
    actions = new(payload)
    assert actions[0]['url'] == body['url']
    assert channel['url'] in [action.get('url') for action in actions]
    channel_action = next(action for action in actions if action.get('url') == channel['url'])
    row = new.candidate_contexts(payload, actions)['ordering']['by_action_id'][channel_action['action_id']]
    assert row['tier'] == 'other_observed_link'
    assert row['source_observation']['directory_route_role'] == 'news_index'


def test_snippet_and_filtered_anchor_labels_are_only_visible_order_inputs():
    links = [article('hidden-title', 'Arsenal preview 10 October 2026', observed_anchors=[]),
             article('snippet', 'Update', snippet='Arsenal injury news'),
             article('label', '', observed_anchors=[{'anchor_index': 7,
                 'text': 'Arsenal preview 2026-10-10', 'text_truncated': False}])]
    payload = public([document(links)])
    new = generator()
    actions = new(payload)
    assert [row['url'] for row in actions[:3]] == [links[2]['url'], links[1]['url'], links[0]['url']]
    hidden = new.candidate_contexts(payload, actions)['ordering']['by_action_id'][actions[2]['action_id']]
    assert not any(hidden['source_observation']['cues'].values())


@pytest.mark.parametrize('updates', [
    {'status': 'FAILED'}, {'available_at': '2026-10-09T09:00:01Z'},
    {'observed_at': '2026-10-09T09:00:02Z', 'available_at': '2026-10-09T08:00:01Z'},
    {'available_at': None},
])
def test_failed_future_or_unproven_parent_cannot_grant_an_article(updates):
    link = article('future', 'Arsenal injury 2026-10-10')
    actions = generator()(public([document([link], **updates)]))
    assert link['url'] not in [row.get('url') for row in actions]
    assert actions[0]['url'] == AWAY


def test_link_cap_is_applied_before_sort_and_reports_a_missing_later_priority_article():
    links = [article(f'entry-{index}', 'Club update') for index in range(13)]
    links[-1]['title'] = 'Arsenal injury preview 10 October 2026'
    payload = public([document(links)])
    new = generator()
    actions = new(payload)
    assert links[-1]['url'] not in [row.get('url') for row in actions]
    assert {row['action_id'] for row in actions} == {row['action_id'] for row in generator(11)(payload)}
    caps = new.candidate_contexts(payload, actions)['ordering']['caps']
    assert caps == {'max_discovered_links': 12, 'eligible_observed_article_links': 13,
                    'retained_by_link_cap': 12, 'omitted_by_link_cap': 1,
                    'omitted_article_links_by_menu_cap': 0, 'max_candidates': 64,
                    'global_candidate_generation_complete': False, 'article_cap_shortage': True}


def test_menu_cap_retains_legacy_set_and_reports_articles_excluded_by_cap():
    payload = public([document([article('one'), article('two'), article('three')])])
    new, old = generator(max_candidates=5), generator(11, max_candidates=5)
    actions = new(payload)
    assert {row['action_id'] for row in actions} == {row['action_id'] for row in old(payload)}
    caps = new.candidate_contexts(payload, actions)['ordering']['caps']
    assert caps['omitted_article_links_by_menu_cap'] == 2
    assert caps['article_cap_shortage'] is True


def test_attempted_failed_directory_is_not_treated_as_unvisited_but_is_retained():
    new = generator()
    away = next(row for row in new(public()) if row.get('url') == AWAY)
    actions = new(public(attempts=[{'information_key': away['information_key'], 'status': 'FAILED'}]))
    assert actions[0]['url'] == HOME
    assert away in actions


def test_first_nonofficial_exposure_does_not_hide_later_official_source_proof():
    link = article('arsenal-preview', 'Arsenal preview 2026-10-10')
    search = document([link], document_id='search', url='https://www.arsenal.com/search', kind='search')
    payload = public([search, document([link])])
    new = generator()
    actions = new(payload)
    first = new.candidate_contexts(payload, actions)['ordering']['by_action_id'][actions[0]['action_id']]
    assert actions[0]['url'] == link['url']
    assert first['source_rank'] == 1
    assert first['source_observation']['parent_document_id'] == 'directory'


def test_v12_exact_rebuild_freezes_order_contract_without_changing_legacy_versions():
    payload = public([document([article('one', 'Training')])])
    for version in range(1, 13):
        original = generator(version)
        config = original.configuration()
        restored = EvidenceDocumentCandidates.from_configuration(config)
        assert restored.configuration() == config
        assert restored(payload) == original(payload)
        assert ('visible_metadata_order' in config) == (version == 12)
    config = generator().configuration()
    assert config['version'] == VISIBLE_ORDER_VERSION
    config['visible_metadata_order']['target_body_read'] = True
    with pytest.raises(ValueError, match='reconstruct exactly'):
        EvidenceDocumentCandidates.from_configuration(config)


def test_v13_uses_bound_visible_card_title_without_reading_the_target_body():
    first = article('club-update', 'Club update', observed_anchors=[])
    card = article('match-update', 'Arsenal injury preview 2026-10-10',
                   observed_anchors=[], directory_context_basis='liverpool_visible_article_card_v1',
                   target_article_fetched_for_context=False)
    payload = public([document([first, card])])
    old, new = generator(12), generator(13)
    assert old(payload)[0]['url'] == first['url']
    assert new(payload)[0]['url'] == card['url']
    restored = EvidenceDocumentCandidates.from_configuration(new.configuration())
    assert restored.configuration() == new.configuration()
    assert restored(payload) == new(payload)
    for change in [{'directory_context_basis': 'unknown'}, {'target_article_fetched_for_context': True}]:
        changed = deepcopy(payload)
        changed['documents'][0]['links'][1].update(change)
        assert new(changed)[0]['url'] == first['url']
