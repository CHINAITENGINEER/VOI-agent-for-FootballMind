"""Real saved HTML and bounded malformed variants; all network calls forbidden."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import escape
import importlib.util
import json
from pathlib import Path
import socket

import pytest

from football_agent.news_discovery import NewsDiscoveryTools
from football_research.acquisition.visible_state import encode_visible_state
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive


ROOT = Path(__file__).resolve().parents[2]
CAPTURED = datetime(2026, 10, 6, 19, tzinfo=timezone.utc)
PARSED = CAPTURED+timedelta(hours=1)
URL = 'https://source.example/news/2026/10/06/an-observed-source-story'
TITLE = 'An unfamiliar source subject'
LEAD = 'This source describes an unresolved practical matter without assigning its value.'
P1 = 'Its participants offered an account of their own observations and remaining questions.'
P2 = 'A later paragraph preserves details that may or may not concern the current task.'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Projection v6 may only read saved raw bytes or constructed fixtures')
    for obj, name in ((socket, 'getaddrinfo'), (socket, 'create_connection'),
                      (socket.socket, 'connect'), (socket.socket, 'connect_ex')):
        monkeypatch.setattr(obj, name, blocked)


def html(*, url=URL, day='6.10.2026', extra_head='', content=None, extra='', extra_root=''):
    content = '<p>'+P1+'</p><p>'+P2+'</p>' if content is None else content
    return ('<html lang="fi"><head><meta property="og:url" content="'+escape(url, quote=True)+'">'
        '<meta property="og:title" content="'+TITLE+'"><meta property="og:type" content="website">'
        '<meta property="og:description" content="'+LEAD+'">'+extra_head+'</head><body>'
        '<nav><p>NAV_SENTINEL</p></nav><main><section><div class="text-content padded">'
        '<h1>'+TITLE+'</h1><div class="details"><span class="date">'+day+'</span></div>'
        '<div class="ingress"><p>'+LEAD+'</p></div><div class="news-text">'+content+'</div>'
        '<div class="related"><div class="text-content"><p>RELATED_SENTINEL</p></div></div>'
        '<div class="source"><p>SOURCE_SENTINEL</p></div>'+extra_root+'</div></section></main>'
        +extra+'</body></html>')


def project(tmp_path, source=None, *, url=URL, observed_url=None, version=6, parsed=PARSED):
    archive = RawArchive(tmp_path/'archive')
    obs = archive.save('open_page', observed_url or url, {}, (html() if source is None else source).encode(),
                       status=200, observed_at=CAPTURED)
    prepared = {'tool':'read_url', 'source':'open_page', 'endpoint':url, 'params':{}}
    tools = NewsDiscoveryTools(archive, ['source.example'], projection_version=version, clock=lambda:parsed)
    return tools.project(obs, prepared), tools, obs, prepared


def test_metadata_title_lead_calendar_and_paragraphs_are_bound_without_subject_keywords(tmp_path):
    doc, tools, obs, prepared = project(tmp_path)
    news = doc['structured_news']
    assert doc['kind'] == news['role'] == 'article' and doc['status'] == 'OK'
    assert doc['news_projection_version'] == 'structured-news-discovery-v6'
    assert news['version'] == 'structured-news-discovery-v3'
    assert doc['text'] == '\n'.join((LEAD, P1, P2))
    assert TITLE not in doc['text'] and 'SENTINEL' not in doc['text']
    assert doc['source_published_at'] is None
    for i, segment in enumerate(news['segments']):
        assert segment['content_source'] == 'semantic_html_article_paragraph'
        assert segment['original_dom_paragraph_index'] == i
        assert segment['source_publication_date'] == '2026-10-06'
        assert segment['source_published_at'] is segment['source_publication_local_time'] is None
        assert segment['publication_timezone'] == 'unknown'
        assert doc['text'][segment['start']:segment['end']] == segment['text']
        assert segment['body_binding']['metadata_json_pointer'] is None
        assert segment['body_binding']['metadata_css_selectors']['title']
    assert doc['observed_at'] == CAPTURED.isoformat() and doc['available_at'] == PARSED.isoformat()
    restored = NewsDiscoveryTools.from_configuration(tools.archive, tools.configuration(), clock=lambda:PARSED)
    assert restored.project(obs, prepared) == doc


@pytest.mark.parametrize('case', ['duplicate_url', 'wrong_url', 'duplicate_title', 'wrong_title', 'missing_title',
    'wrong_description', 'canonical_conflict', 'duplicate_canonical', 'invalid_date', 'date_mismatch',
    'invalid_permalink_date', 'directory', 'future_day', 'future_publication', 'future_modification',
    'invalid_publication', 'timezone_missing', 'modified_before_published', 'published_day_conflict',
    'hidden_root', 'hidden_title', 'hidden_body', 'hidden_lead', 'hidden_date', 'template_root',
    'two_roots', 'competing_article', 'missing_main', 'two_main', 'missing_body', 'missing_lead_paragraph',
    'missing_body_paragraph', 'bad_jsonld', 'other_jsonld', 'body_before_lead', 'observation_url_conflict',
    'hidden_title_matching_metadata', 'hidden_lead_matching_metadata', 'modified_before_source_day'])
def test_ambiguous_unbound_hidden_directory_or_future_metadata_has_no_typed_body(tmp_path, case):
    source, url, observed_url = html(), URL, None
    if case == 'duplicate_url': source = html(extra_head='<meta property="og:url" content="'+URL+'">')
    elif case == 'wrong_url': source = html(url=URL+'-other')
    elif case == 'duplicate_title': source = html(extra_head='<meta property="og:title" content="'+TITLE+'">')
    elif case == 'wrong_title': source = source.replace('<h1>'+TITLE, '<h1>Another source heading')
    elif case == 'missing_title': source = source.replace('<meta property="og:title" content="'+TITLE+'">', '')
    elif case == 'wrong_description': source = source.replace('content="'+LEAD+'"', 'content="Other lead"')
    elif case == 'canonical_conflict': source = html(extra_head='<link rel="canonical" href="'+URL+'-other">')
    elif case == 'duplicate_canonical': source = html(extra_head=2*('<link rel="canonical" href="'+URL+'">'))
    elif case == 'invalid_date': source = html(day='31.2.2026')
    elif case == 'date_mismatch': source = html(day='5.10.2026')
    elif case == 'invalid_permalink_date': url = URL.replace('/10/06/', '/02/31/'); source = html(url=url)
    elif case == 'directory': url = 'https://source.example/news/'; source = html(url=url)
    elif case == 'future_day': url = URL.replace('/10/06/', '/10/08/'); source = html(url=url, day='8.10.2026')
    elif case == 'future_publication': source = html(extra_head='<meta property="article:published_time" content="2026-10-06T20:00:00Z">')
    elif case == 'future_modification': source = html(extra_head='<meta property="article:modified_time" content="2026-10-06T20:00:00Z">')
    elif case == 'invalid_publication': source = html(extra_head='<meta property="article:published_time" content="not a date">')
    elif case == 'timezone_missing': source = html(extra_head='<meta property="article:published_time" content="2026-10-06T10:00:00">')
    elif case == 'modified_before_published': source = html(extra_head='<meta property="article:published_time" content="2026-10-06T10:00:00Z"><meta property="article:modified_time" content="2026-10-06T09:00:00Z">')
    elif case == 'published_day_conflict': source = html(extra_head='<meta property="article:published_time" content="2026-10-05T10:00:00Z">')
    elif case == 'hidden_root': source = source.replace('class="text-content padded"', 'class="text-content padded" hidden')
    elif case == 'hidden_title': source = source.replace('<h1>', '<h1 aria-hidden="true">')
    elif case == 'hidden_body': source = source.replace('class="news-text"', 'class="news-text" style="display:none"')
    elif case == 'hidden_lead': source = source.replace('class="ingress"', 'class="ingress" hidden')
    elif case == 'hidden_date': source = source.replace('class="date"', 'class="date" hidden')
    elif case == 'template_root': source = source.replace('<section>', '<template><section>').replace('</section>', '</section></template>')
    elif case == 'two_roots': source = source.replace('</main>', source[source.index('<section>'):source.index('</main>')]+'</main>')
    elif case == 'competing_article': source = html(extra='<article><p>'+P1+'</p><p>'+P2+'</p></article>')
    elif case == 'missing_main': source = source.replace('<main>', '<div>').replace('</main>', '</div>')
    elif case == 'two_main': source = html(extra='<main><p>Other page area</p></main>')
    elif case == 'missing_body': source = source.replace('class="news-text"', 'class="generic-div"')
    elif case == 'missing_lead_paragraph': source = source.replace('<p>'+LEAD+'</p>', LEAD)
    elif case == 'missing_body_paragraph': source = html(content=P1)
    elif case == 'bad_jsonld': source = html(extra_head='<script type="application/ld+json">bad JSON</script>')
    elif case == 'other_jsonld': source = html(extra_head='<script type="application/ld+json">{"@type":"WebPage"}</script>')
    elif case == 'body_before_lead': source = source.replace('<div class="ingress"><p>'+LEAD+'</p></div>', '').replace('<div class="related">', '<div class="ingress"><p>'+LEAD+'</p></div><div class="related">')
    elif case == 'observation_url_conflict': observed_url = URL+'-actually-requested'
    elif case == 'hidden_title_matching_metadata': source = source.replace('content="'+TITLE+'"', 'content="'+TITLE+' HIDDEN"').replace('</h1>', '<span hidden> HIDDEN</span></h1>')
    elif case == 'hidden_lead_matching_metadata': source = source.replace('content="'+LEAD+'"', 'content="'+LEAD+' HIDDEN"').replace('<p>'+LEAD+'</p>', '<p>'+LEAD+'<span hidden> HIDDEN</span></p>')
    elif case == 'modified_before_source_day': source = html(extra_head='<meta property="article:modified_time" content="2020-01-01T00:00:00Z">')
    doc, _, _, _ = project(tmp_path, source, url=url, observed_url=observed_url)
    assert doc['structured_news']['role'] == 'unknown'
    assert doc['structured_news']['segments'] == []


def test_advertisements_hidden_cards_and_template_text_are_excluded_but_br_and_strong_survive(tmp_path):
    body = ('<p>'+P1+'<span class="ad">INLINE_AD_SENTINEL</span></p>'
        '<div class="advertisement"><p>AD_SENTINEL</p></div><aside><p>ASIDE_SENTINEL</p></aside>'
        '<p>1 First Person<br>2 <strong>Second Person</strong></p>'
        '<p hidden>HIDDEN_SENTINEL</p><template><p>TEMPLATE_SENTINEL</p></template>'
        '<p><a href="/card">LINK_CARD_SENTINEL</a></p><p>'+P2+'</p><p class="banner">BANNER_SENTINEL</p>')
    doc, _, _, _ = project(tmp_path, html(content=body))
    assert doc['structured_news']['role'] == 'article'
    assert 'SENTINEL' not in doc['text']
    assert '1 First Person 2 Second Person' in doc['text']
    assert [s['original_dom_paragraph_index'] for s in doc['structured_news']['segments']] == [0, 1, 4, 8]


def _real_receipts():
    parent = ROOT/'var/parallel/agent/round198/live-cohort'
    if not parent.is_dir():
        pytest.skip('Original R198 raw acquisition archives are not installed')
    return parent, [(path.parent, json.loads(path.read_text(encoding='utf-8')))
                    for path in sorted(parent.glob('task-*/analysis/match-*/receipt-*.json'))]


def test_four_actual_raw_articles_reach_visible_state_without_rewriting_originals():
    parent, receipts = _real_receipts()
    frozen = {p:sha256(p.read_bytes()).hexdigest() for p in parent.rglob('*') if p.is_file()}
    at, recovered, rejected_directories = datetime.now(timezone.utc), [], []
    for folder, returned in receipts:
        obs = Observation.model_validate(returned['receipt']['observation'])
        prepared = returned['receipt']['prepared']
        original = returned['document']
        config = json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
        archive = RawArchive(folder/'source-archive')
        old = NewsDiscoveryTools.from_configuration(archive, config['supervisor_metadata']['transport'],
            clock=lambda d=original:datetime.fromisoformat(d['available_at']))
        assert old.project(obs, prepared) == original
        tools = NewsDiscoveryTools(archive, old.allowed_hosts, provider=old.provider, projection_version=6,
                                  clock=lambda:at, index_urls=old.index_urls)
        doc = tools.project(obs, prepared)
        if '/uutiset/2026/' not in obs.endpoint:
            if prepared['tool'] == 'read_url' and doc['status'] == 'OK':
                assert doc['structured_news']['role'] != 'article'
                rejected_directories.append(obs.endpoint)
            continue
        assert original['structured_news']['role'] == 'unknown'
        assert doc['structured_news']['role'] == 'article'
        assert doc['available_at'] == at.isoformat() and doc['observed_at'] == original['observed_at']
        assert doc['raw_sha256'] == original['raw_sha256']
        assert doc['source_published_at'] is None
        public = deepcopy(json.loads((folder/'run.json').read_text(encoding='utf-8'))['cycles'][0]['public_before'])
        public.pop('candidate_contexts', None)  # The original menu sidecar binds its original clock/documents.
        public.update(documents=[doc], decision_at=at.isoformat())
        encoded = encode_visible_state(public)
        assert encoded['before']['current_probabilities'] is None
        assert encoded['audit']['excluded_documents'] == encoded['audit']['excluded_segments'] == []
        assert encoded['before']['documents'][0]['source_role'] == 'article'
        assert not encoded['before']['documents'][0]['mask']['body_unavailable']
        texts = {row['text'] for row in encoded['before']['contents']}
        assert all(segment['text'] in texts for segment in doc['structured_news']['segments'])
        recovered.append((obs.endpoint, len(doc['structured_news']['segments'])))
    assert len(recovered) == 4 and sorted(n for _, n in recovered) == [5, 6, 7, 11]
    assert len(set(rejected_directories)) >= 3
    assert all(sha256(p.read_bytes()).hexdigest() == digest for p, digest in frozen.items())


def test_versions_one_to_five_keep_archived_configuration_and_projection_bytes(tmp_path):
    path = ROOT/'var/parallel/agent/round199/source-before/football_agent/news_discovery.py'
    if not path.is_file():
        pytest.skip('Original pre-v6 implementation source archive is absent')
    spec = importlib.util.spec_from_file_location('football_agent._pre_r199_news', path)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    from test_news_discovery_v3 import html as jsonld_html
    for i, source in enumerate((html(), jsonld_html())):
        _, _, obs, prepared = project(tmp_path/str(i), source)
        archive = RawArchive(tmp_path/str(i)/'archive')
        for version in range(1, 6):
            new = NewsDiscoveryTools(archive, ['source.example'], projection_version=version, clock=lambda:PARSED)
            old = previous.NewsDiscoveryTools(archive, ['source.example'], projection_version=version, clock=lambda:PARSED)
            assert new.configuration() == old.configuration()
            serialize = lambda value:json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            assert serialize(new.project(obs, prepared)) == serialize(old.project(obs, prepared))
