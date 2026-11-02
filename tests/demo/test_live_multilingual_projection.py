"""Regression contracts for source discovery/body ingestion, not model value."""
from datetime import datetime, timezone
import json

import pytest

from demo.live_sources import LiveNewsTools, article_link_reason, search_team_binding
from football_spider.storage.raw_archive import RawArchive

AT = datetime(2026,10,25,9,0,tzinfo=timezone.utc)


def project(tmp_path, url, html, *, index=False):
    from urllib.parse import urlsplit
    archive=RawArchive(tmp_path)
    obs=archive.save("open_page",url,{},html.encode(),status=200,observed_at=AT)
    tools=LiveNewsTools(archive,[urlsplit(url).hostname],clock=lambda:AT,
        index_urls=[url] if index else [],projection_version=13)
    return tools.project(obs,{"tool":"read_url","endpoint":url,"params":{},"source":"open_page"})


def test_literal_vue_directory_cards_do_not_require_rendered_href(tmp_path):
    url="https://fortunasittard.nl/nieuws"
    records=[{"url":url+"/bezoekersinformatie-fortuna-sittard-excelsior",
        "title":"Bezoekersinformatie: Fortuna Sittard - Excelsior","postDate":"23 October 2026"}]
    payload=json.dumps(records).replace('"','&quot;')
    html=f'<html><body><load-filter-component :items="{payload}"><a :href="item.url">${{item.title}}</a></load-filter-component></body></html>'
    d=project(tmp_path,url,html,index=True)
    assert len(d["links"])==1
    assert d["links"][0]["url"]==records[0]["url"]
    assert d["links"][0]["publication_raw"]=="23 October 2026"
    assert d["links"][0]["source_published_at"] is None
    assert d["live_directory_projection"]["target_article_requests"]==0
    assert not d["live_directory_projection"]["rendered_visibility_verified"]


def test_literal_directory_rejects_cross_host_and_javascript(tmp_path):
    url="https://fortunasittard.nl/nieuws"
    records=[{"url":"https://private.example/news/story","title":"A legitimate looking article"}]
    html='<load-filter-component :items="'+json.dumps(records).replace('"','&quot;')+'"></load-filter-component>'
    d=project(tmp_path,url,html,index=True)
    assert not d["links"]
    assert d["live_directory_projection"]["rejected"]
    d=project(tmp_path,url,'<load-filter-component :items="window.loadNews()"></load-filter-component>',index=True)
    assert d["live_directory_projection"]["status"]=="rejected"


@pytest.mark.parametrize("path,title",[
    ("/nieuws/fortuna-sittard-excelsior-voorbeschouwing","Fortuna Sittard ontvangt Excelsior"),
    ("/actualites/le-groupe-pour-dimanche","Le groupe pour dimanche"),
    ("/noticias/parte-medico-del-equipo","Parte médico del equipo"),
    ("/notizie/allenamento-e-convocati","Allenamento e convocati"),
    ("/2026/10/excelsior-neemt-nieuwe-trainingsaccommodatie-in-gebruik/","Excelsior neemt nieuwe trainingsaccommodatie in gebruik"),
    ("/news/20261025-injuries","伤停消息"),
])
def test_localized_article_shapes_and_short_cjk_titles(path,title):
    assert article_link_reason("https://club.example"+path,title,["https://club.example/news"],set()) is None


@pytest.mark.parametrize("path,title",[("/category/nieuws/","Nieuws"),("/nieuws/","Laatste nieuws"),
    ("/partners-excelsior-rotterdam-vrouwen/","Partners Vrouwen"),("/contact/","Neem contact met ons op")])
def test_navigation_and_sponsor_pages_stay_out(path,title):
    assert article_link_reason("https://club.example"+path,title,["https://club.example/news"],set()) is not None


def test_explicit_cjk_fixture_alias_can_be_embedded_in_headline():
    fixture={"home_name":"Shanghai Shenhua","away_name":"Beijing Guoan","home_aliases":["上海申花"],"away_aliases":["北京国安"]}
    assert search_team_binding("上海申花发布赛前消息","https://sports.example/news/a",fixture,[])
    assert not search_team_binding("上海天气预报","https://sports.example/news/a",fixture,[])


def test_registered_bare_homepage_uses_the_same_canonical_url_as_tool(tmp_path):
    tools=LiveNewsTools(RawArchive(tmp_path),["club.example"],index_urls=["https://club.example"],projection_version=13)
    public={"task":{"seed_urls":["https://club.example"]},"documents":[]}
    prepared=tools.prepare({"tool":"read_url","url":"https://club.example"},public)
    assert prepared["endpoint"]=="https://club.example/"
    assert tools.index_urls==["https://club.example/"]
    assert public["task"]["seed_urls"]==["https://club.example"]
    with pytest.raises(ValueError,match="seeds or acquired links"):
        tools.prepare({"tool":"read_url","url":"https://club.example/news/unseen-article"},public)


def test_observed_link_fragment_is_canonicalized_without_admitting_unseen_url(tmp_path):
    tools=LiveNewsTools(RawArchive(tmp_path),["club.example"],projection_version=13)
    public={"task":{"seed_urls":[]},"documents":[{"links":[{"url":"https://club.example/news/preview#body"}]}]}
    prepared=tools.prepare({"tool":"read_url","url":"https://club.example/news/preview#body"},public)
    assert prepared["endpoint"]=="https://club.example/news/preview"


@pytest.mark.parametrize("competition,word,site",[("荷甲","voetbal","eredivisie.nl"),
    ("Spanish LALIGA","fútbol","laliga.com"),("Italian Serie A","calcio","legaseriea.it"),
    ("French Ligue 1","actualités","ligue1.com"),("中超","足球","thecfa.cn")])
def test_navigation_uses_competition_language_and_sources(competition,word,site):
    from demo.live_sources import discovery_queries
    q=discovery_queries({"home_name":"Test Home","away_name":"Test Away","kickoff":"2026-10-25T12:00:00Z","competition_name":competition})
    assert any(word in x for x in q)
    assert any(site in x for x in q)


def test_unregistered_premier_league_club_has_verified_league_navigation():
    from demo.live_sources import profile
    p=profile({"canonical":{"home_name":"Unknown Home","away_name":"Unknown Away","competition_name":"Premier League"}})
    assert p["seed_urls"]==p["league_wide_seed_urls"]==["https://www.premierleague.com/en/news"]
    assert p["unknown_teams"]==["Unknown Home","Unknown Away"]


def test_generic_visible_article_is_bound_and_excludes_footer(tmp_path):
    url="https://club.example/news/players"
    first="The manager confirmed that the named player will miss the forthcoming fixture because he has not recovered."
    second="The rest of the squad completed the session, while the coach said that the replacement has not yet been selected."
    html=f'<html><head><link rel="canonical" href="{url}"><meta property="og:type" content="article"><meta property="og:title" content="Squad update"></head><body><article><h1>Squad update</h1><p>{first}</p><p>{second}</p><footer><p>Buy merchandise.</p></footer></article></body></html>'
    d=project(tmp_path,url,html)
    assert d["structured_news"]["role"]=="article"
    assert d["text"]==first+"\n"+second
    assert d["source_published_at"] is None
    assert all(d["text"][s["start"]:s["end"]]==s["text"] for s in d["structured_news"]["segments"])


def test_generic_fallback_does_not_override_conflicting_article_jsonld(tmp_path):
    url="https://club.example/news/players"
    html=f'<html><head><link rel="canonical" href="{url}"><meta property="og:type" content="article"><meta property="og:title" content="Squad update"><script type="application/ld+json">{{"@type":"Article","headline":"Different title","url":"https://club.example/news/other"}}</script></head><body><article><h1>Squad update</h1><p>{"actual paragraph "*15}</p><p>{"second paragraph "*15}</p></article></body></html>'
    d=project(tmp_path,url,html)
    assert d["structured_news"]["role"]!="article"


def test_live_body_keeps_explicit_language_and_bound_publication(tmp_path):
    url="https://club.example/news/players"
    html=f'<html lang="nl"><head><link rel="canonical" href="{url}"><meta property="og:type" content="article"><meta property="og:title" content="Squad update"><meta property="article:published_time" content="2026-10-24T15:00:00+02:00"></head><body><article><h1>Squad update</h1><p>{"First actual paragraph. "*8}</p><p>{"Second actual paragraph. "*8}</p></article></body></html>'
    d=project(tmp_path,url,html)
    assert d["structured_news"]["role"]=="article"
    assert d["structured_news"]["source_language"]=="nl"
    assert d["source_published_at"]=="2026-10-24T15:00:00+02:00"
    assert all(s["source_published_at"]==d["source_published_at"] for s in d["structured_news"]["segments"])
