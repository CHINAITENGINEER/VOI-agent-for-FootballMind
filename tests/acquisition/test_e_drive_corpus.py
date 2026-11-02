"""Behavior checks for the E-drive research corpus, independent of network."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(TOOLS))
from build_e_drive_football_corpus import parse_csv, write_rows
from build_e_drive_news_corpus import parse_article
from vectorize_e_drive_football_corpus import build_lagged_features, public_article_view, mentions, link_matches


def test_match_outcome_and_closing_prices_are_separate():
    csv = b'Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA,AvgCH,AvgCD,AvgCA,HS\n01/01/2020,A,B,2,1,H,2,3,4,1.8,3.2,4.5,99\n'
    public, labels, odds, stats = parse_csv(csv, '1920', 'E0', {})
    assert not {'FTR', 'outcome', 'home_goals', 'HS', 'postmatch_statistics'} & public[0].keys()
    assert labels[0]['outcome'] == 'H'
    assert odds[0]['snapshots']['AvgC']['snapshot_kind'] == 'closing'
    assert odds[0]['strict_prequery_eligible'] is False
    assert sum(odds[0]['snapshots']['Avg']['no_vig_probabilities']) == pytest.approx(1)
    assert stats[0]['postmatch_statistics']['HS'] == '99'


def test_rejects_summary_without_body():
    data = b'<script type="application/ld+json">{"@type":"NewsArticle","headline":"Preview","description":"a"}</script><nav>injury news</nav>'
    with pytest.raises(RuntimeError, match='no_verified_article_body'):
        parse_article(data, 'www.liverpoolfc.com')


def test_dom_body_excludes_navigation_and_retains_null_dates():
    body = 'Player is still recovering and the manager will decide tomorrow. ' * 4
    html = '<meta property="og:title" content="Team news"><nav>SECRET NAVIGATION</nav><article><div data-testid="rte-content"><p>' + body + '</p></div></article>'
    row = parse_article(html.encode(), 'www.liverpoolfc.com')
    assert 'SECRET' not in row['body']
    assert row['published_at'] is None
    assert row['body'] == body.strip()


def test_nonfree_body_not_ingested():
    html = '<script type="application/ld+json">{"@type":"NewsArticle","isAccessibleForFree":false}</script>'
    with pytest.raises(RuntimeError, match='not_freely_accessible'):
        parse_article(html.encode(), 'www.arsenal.com')


def test_visible_intro_included_once_and_nested_paragraph_not_duplicated():
    intro = 'An update before the next match.'
    paragraph = 'The manager will assess the player tomorrow before deciding on the squad. ' * 3
    schema = json.dumps({'@type': 'NewsArticle', 'headline': 'Team news', 'description': intro})
    html = '<script type="application/ld+json">' + schema + '</script><p>' + intro + '</p>'
    html += '<article><div data-testid="rte-content"><ul><li><p>' + paragraph + '</p></li></ul></div></article>'
    row = parse_article(html.encode(), 'www.liverpoolfc.com')
    assert row['paragraphs'] == [intro, paragraph.strip()]
    assert row['body'].count(intro) == 1


def test_visible_summary_alone_is_not_full_body():
    intro = 'A public introductory summary of the team news. ' * 5
    schema = json.dumps({'@type': 'NewsArticle', 'headline': 'Team news', 'description': intro})
    html = '<script type="application/ld+json">' + schema + '</script><p>' + intro + '</p>'
    with pytest.raises(RuntimeError, match='no_verified_article_body'):
        parse_article(html.encode(), 'www.liverpoolfc.com')


def test_unread_view_does_not_expose_body_or_labels():
    article = {'article_id': 'a', 'title': 'Training', 'body': 'secret', 'body_vector': [1], 'outcome': 'H', 'g_gain': 1}
    view = public_article_view(article)
    assert view['title'] == 'Training'
    assert not {'body', 'body_vector', 'outcome', 'g_gain'} & view.keys()


def test_alias_boundaries():
    assert mentions('Man Utd team news', 'Man United')
    assert mentions('Brighton & Hove Albion preview', 'Brighton')
    assert not mentions('Newcastleshire festival', 'Newcastle')


def test_explicit_womens_source_keyword_excluded_from_mens_match_bag(tmp_path):
    write_rows(tmp_path / 'structured/matches.jsonl', [dict(match_id='m', division='E0',
               home_team='Liverpool', away_team='Arsenal', match_date='2025-04-06')])
    common = dict(title='Injury update', source_team='Liverpool', published_at='2025-04-05T11:36:00Z')
    write_rows(tmp_path / 'news/articles.jsonl', [dict(common, article_id='w', source_keywords=['Liverpool FC Women']),
                                                dict(common, article_id='m', source_keywords=['First team'])])
    link_matches(tmp_path)
    bags = [json.loads(s) for s in (tmp_path / 'research/match_article_bags.jsonl').read_text().splitlines()]
    assert [x['article_id'] for x in bags[0]['candidates']] == ['m']


@pytest.mark.parametrize('title', ['U18s preview: Liverpool v Arsenal', 'U21s team news', 'Youth internationals'])
def test_plural_youth_titles_not_linked_to_senior_match(tmp_path, title):
    write_rows(tmp_path / 'structured/matches.jsonl', [dict(match_id='m', division='E0',
               home_team='Liverpool', away_team='Arsenal', match_date='2025-04-06')])
    write_rows(tmp_path / 'news/articles.jsonl', [dict(article_id='a', title=title,
               source_team='Liverpool', published_at='2025-04-05T11:36:00Z')])
    link_matches(tmp_path)
    assert not (tmp_path / 'research/match_article_bags.jsonl').read_text().strip()


def test_same_day_results_cannot_affect_features(tmp_path):
    matches = [dict(match_id=str(i), division='E0', home_team='A', away_team='B', match_date=date)
               for i, date in enumerate(['2020-01-01', '2020-01-01', '2020-01-02'])]
    labels = [dict(match_id=str(i), outcome='H', home_goals=9, away_goals=0) for i in range(3)]
    write_rows(tmp_path / 'structured/matches.jsonl', matches)
    write_rows(tmp_path / 'structured/labels/results.jsonl', labels)
    build_lagged_features(tmp_path)
    values = np.load(tmp_path / 'vectors/prematch_numeric.npy')
    assert np.array_equal(values[0], values[1])
    assert values[0, 3] == 0
    assert values[2, 3] == 2
    assert values[2, 7] == 18


def test_corpus_body_requires_explicit_read(tmp_path):
    from football_research.data_adapters.e_drive_corpus import ExternalFootballCorpus
    body = tmp_path / 'body.txt'
    body.write_text('real local evidence', encoding='utf-8')
    write_rows(tmp_path / 'news/articles.jsonl', [{'article_id': 'x', 'title': 'News', 'body_path': str(body),
               'provenance': 'historical_reconstruction'}])
    write_rows(tmp_path / 'vectors/article_titles.rows.jsonl', [{'row': 0, 'article_id': 'x'}])
    write_rows(tmp_path / 'vectors/article_body_mean.rows.jsonl', [{'row': 0, 'article_id': 'x'}])
    np.save(tmp_path / 'vectors/article_titles.npy', np.ones((1, 384), dtype=np.float32))
    np.save(tmp_path / 'vectors/article_body_mean.npy', np.ones((1, 384), dtype=np.float32))
    corpus = ExternalFootballCorpus(tmp_path)
    assert 'body' not in corpus.unread_candidate('x')
    with pytest.raises(ValueError, match='unread article'):
        corpus.read_evidence('x', already_read_ids=set())
    assert corpus.read_evidence('x', already_read_ids={'x'})['body'] == 'real local evidence'
    write_rows(tmp_path / 'vectors/article_body_mean.rows.jsonl', [{'row': 0, 'article_id': 'wrong-article'}])
    with pytest.raises(ValueError, match='alignment'):
        ExternalFootballCorpus(tmp_path)
