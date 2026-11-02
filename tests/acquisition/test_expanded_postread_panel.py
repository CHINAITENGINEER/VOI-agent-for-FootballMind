from tools.prepare_expanded_postread_panel import prepare
import pytest


def fixture(dates, hashes):
    matches, articles, links = {}, {}, []
    for i,(date,bodyhash) in enumerate(zip(dates,hashes)):
        mid,aid=f'm{i}',f'a{i}'
        matches[mid]={'match_date':date}
        articles[aid]={'title':'Team news', 'published_at':date[:7]+'-01', 'body_sha256':bodyhash}
        links.append({'match_id':mid,'article_id':aid,'days_before_match_date':2})
    return matches,links,articles


def test_body_duplicate_across_f_and_g_removed_from_both():
    data=fixture(['2023-11-12','2024-01-05','2024-02-05'],['same','same','unique'])
    panel,rejected=prepare(*data)
    assert [p['match_id'] for p in panel]==['m2']
    assert sum(r['reason']=='shared_body_crosses_four_temporal_groups' for r in rejected)==2


def test_numeric_base_training_exclusion():
    data=fixture(['2019-06-01','2020-01-01','2022-02-01'],['a','b','c'])
    panel,_=prepare(*data,base_ids={'m1'})
    assert [p['match_id'] for p in panel]==['m2']


def test_cap_keeps_latest_declared_publications_without_labels():
    matches,links,articles=fixture(['2023-11-12']*3,['a','b','c'])
    for i,link in enumerate(links):
        link['match_id']='m0'
        articles[link['article_id']]['published_at']=f'2023-11-0{i+1}'
    panel,rejected=prepare(matches,links,articles,cap=2)
    assert [c['article_id'] for c in panel[0]['candidates']]==['a1','a2']
    assert rejected[-1]['article_id']=='a0'
    with pytest.raises(ValueError):
        prepare(matches,links,articles,cap=0)
