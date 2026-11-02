from copy import deepcopy

from football_research.acquisition.news_link_projection import project_news_links


def link(title, path):
    return {'title': title, 'url': 'https://example.org'+path,
            'parent_url': 'https://example.org/', 'available_at': '2026-10-01T00:00:00Z',
            'title_basis': 'observed_parent_link'}


def test_navigation_is_not_news_evidence_even_when_it_targets_an_article():
    rows = [link('Read', '/en/news/actual-article'), link('Next', '/en/news/next-story'),
            link('Privacy Policy', '/en/privacy'), link('News', '/en/news')]
    kept, audit = project_news_links(rows)
    assert not kept and audit['excluded_labels'] == 4


def test_unrelated_or_negative_news_is_retained_without_topic_or_value_rules():
    rows = [link('Weather at the international training camp', '/en/news/weather'),
            link('Under-21 squad shares community photos', '/articles/community'),
            link('A player will not start', '/2026/10/01/player-update/')]
    original = deepcopy(rows)
    kept, audit = project_news_links(rows)
    assert kept == original == rows
    assert audit['retained_labels'] == 3
    kept[0]['title'] = 'changed elsewhere'
    assert rows == original


def test_unknown_routes_are_explicitly_outside_projection_not_fabricated_titles():
    rows = [link('Actual article title', '/en?article=1'),
            link('Latest news', '/en/news'), link('Category content', '/en/news/category/team')]
    kept, audit = project_news_links(rows)
    assert not kept
    assert audit['excluded'][0]['reason'] == 'outside_declared_article_path_projection'


def test_only_identical_observations_are_deduplicated():
    first = link('Observed title', '/news/item')
    later = {**first, 'available_at': '2026-10-02T00:00:00Z'}
    kept, audit = project_news_links([first, deepcopy(first), later])
    assert kept == [first, later]
    assert audit['excluded'][0]['reason'] == 'duplicate_observed_label'
