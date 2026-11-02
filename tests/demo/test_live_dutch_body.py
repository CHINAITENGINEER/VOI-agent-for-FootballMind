"""Registered body boundaries, same-page identity, and real archived regressions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demo.live_dutch_body import project_dutch_body, VERSION

AZ = 'https://www.az.nl/inside-az/nieuws/2026/oktober/preview'
TWENTE = 'https://fctwente.nl/nieuws/preview'
TEXT = 'This is the actual first team article. The coach discussed the upcoming fixture and the available players.'


def az_html():
    return f'''<html lang="nl"><head><title>Actual preview</title>
    <script>window.dataLayer.push({{"page":{{"page_type":"articleDetailPage"}},"event":"page_data"}});</script></head>
    <body><nav><p>NAVIGATION DO NOT CONSUME</p></nav><main id="main-content">
    <section class="article-image-hero"><div class="article-image-hero-body"><div class="container">
    <h1>Actual preview</h1><p aria-label="Publicatiedatum">24.10.2026</p></div></div></section>
    <section><div class="text-block-column-wysiwyg"><div class="content"><p>{TEXT}</p><p>Second real paragraph.</p></div></div></section>
    <div class="article-card"><p>RELATED CARD DO NOT CONSUME</p></div></main>
    <footer><p>FOOTER DO NOT CONSUME</p><form data-analytics-form_id="newsletter" action="/inside-az/nieuws/2026/oktober/preview"></form></footer></body></html>'''


def twente_html():
    return f'''<html lang="nl-NL"><head><title>Preview</title>
    <link rel="canonical" href="{TWENTE}"><meta property="og:url" content="{TWENTE}">
    <meta property="og:title" content="Actual preview"><meta property="og:type" content="">
    <meta property="og:datePublished" content="2026-10-24T11:00:00+02:00">
    <meta property="og:dateModified" content="2026-10-24T12:00:00+02:00"><meta name="author" content="FC Twente"></head>
    <body class="NewsPost"><nav><p>NAVIGATION DO NOT CONSUME</p></nav><main><article><h1>Actual preview</h1>
    <span itemprop="datePublished">24 oktober 2026</span><section><div class="prose"><p>{TEXT}</p></div></section>
    <div class="app__elements__textelement"><div class="prose"><p>Second real paragraph.</p><div class="prose"><p>Nested text once only.</p></div></div></div>
    </article><aside><p>RELATED CARD DO NOT CONSUME</p></aside></main><footer><p>FOOTER DO NOT CONSUME</p></footer></body></html>'''


def project(html, url, **kwargs):
    observation = SimpleNamespace(endpoint=url, params={}, http_status=200, error_class=None,
                                  observed_at=datetime(2026, 10, 25, 13, tzinfo=timezone.utc))
    for key, value in kwargs.items(): setattr(observation, key, value)
    return project_dutch_body({'url': url, 'status': 'OK', 'text': 'Unmodeled page text',
                               'structured_news': {'role': 'unknown'}}, html.encode('utf-8'), observation)


@pytest.mark.parametrize('html,url', [(az_html(), AZ), (twente_html(), TWENTE)])
def test_registered_body_only_and_exact_offsets(html, url):
    result = project(html, url)
    assert result['structured_news']['role'] == 'article'
    assert TEXT in result['text']
    for excluded in ('NAVIGATION', 'FOOTER', 'RELATED CARD'):
        assert excluded not in result['text']
    assert result['body_sha256'] == hashlib.sha256(result['text'].encode()).hexdigest()
    assert not result['text_truncated']
    for row in result['structured_news']['segments']:
        assert result['text'][row['start']:row['end']] == row['text']
    if url == TWENTE:
        assert result['text'].count('Nested text once only.') == 1
        assert result['source_authors'] == ['FC Twente']
    else:
        assert result['source_published_at'] is None
        assert result['source_publication_local_time'] == '2026-10-24'
        assert result['publication_status'] == 'date_only_time_zone_unknown'


@pytest.mark.parametrize('html,url', [
    (az_html().replace('<title>Actual preview</title>', '<title>Other page</title>'), AZ),
    (az_html().replace('action="/inside-az/nieuws/2026/oktober/preview"', 'action="/another-page"'), AZ),
    (az_html().replace('articleDetailPage', 'newsTagPage'), AZ),
    (az_html().replace('class="content"', 'class="navigation"'), AZ),
    (az_html().replace('<section class="article-image-hero">', '<section hidden class="article-image-hero">'), AZ),
    (az_html().replace('24.10.2026', '26.10.2026'), AZ),
    (az_html(), 'https://www.az.nl/inside-az/categorieen/laatste-nieuws'),
    (twente_html().replace('rel="canonical" href="'+TWENTE+'"', 'rel="canonical" href="https://fctwente.nl/nieuws/another"'), TWENTE),
    (twente_html().replace('property="og:title" content="Actual preview"', 'property="og:title" content="Other title"'), TWENTE),
    (twente_html().replace('property="og:type" content=""', 'property="og:type" content="website"'), TWENTE),
    (twente_html().replace('<article>', '<article hidden>'), TWENTE),
    (twente_html().replace('class="prose"', 'class="navigation"'), TWENTE),
    (twente_html().replace('2026-10-24T11:00:00+02:00', '2026-10-26T11:00:00+02:00'), TWENTE),
    (twente_html().replace('</main>', '<article><h1>Other article</h1></article></main>'), TWENTE),
])
def test_unbound_hidden_directory_or_empty_body_rejected(html, url):
    result = project(html, url)
    assert result['structured_news']['role'] != 'article'
    assert result['live_dutch_body_projection']['status'] == 'rejected'


def test_conflicting_visible_publication_keeps_both_and_no_invented_timestamp():
    result = project(twente_html().replace('24 oktober 2026', '25 oktober 2026'), TWENTE)
    assert result['structured_news']['role'] == 'article'
    assert result['source_published_at'] is None
    assert result['source_publication_date'] == '2026-10-25'
    assert result['live_dutch_body_projection']['source_metadata_published_at'] == '2026-10-24T11:00:00+02:00'
    assert result['live_dutch_body_projection']['publication_date_conflict']


@pytest.mark.parametrize('changes', [{'http_status': 302}, {'http_status': 500}, {'endpoint': 'https://fctwente.nl/other'}, {'params': {'q': 'wrong'}}])
def test_wrong_transport_cannot_be_rescued(changes):
    assert project(twente_html(), TWENTE, **changes)['structured_news']['role'] != 'article'


def test_real_twenty_archives_recover_fourteen_without_network(monkeypatch):
    import requests
    from demo.live_sources import LiveNewsTools
    from football_spider.models.contracts import Observation
    from football_spider.storage.raw_archive import RawArchive
    root = Path(__file__).resolve().parents[2] / 'var/parallel/agent/round2/runs/live-g/runs/lg-a2adac7a536247a193693ce5af43ea9c'
    if not (root / 'run.json').exists():
        pytest.skip('Local incident archive unavailable; no download attempted')
    before = (root / 'run.json').read_bytes()
    run = json.loads(before)
    monkeypatch.setattr(requests.Session, 'request', lambda *a, **kw: pytest.fail('Zero-network archive replay required'))
    archive = RawArchive(root / 'sources')
    tools = LiveNewsTools(archive, run['source_profile']['allowed_hosts'],
                         index_urls=run['source_profile']['seed_urls'], projection_version=13,
                         clock=lambda: datetime.now(timezone.utc))
    recovered, rejected = [], []
    for attempt in run['attempts']:
        if attempt['purpose'] != 'candidate_body_prefetch': continue
        observation = Observation.model_validate(attempt['response']['observation'])
        assert hashlib.sha256(archive.read_body(observation)).hexdigest() == observation.content_hash
        result = tools.project(observation, {'tool': 'read_url', 'endpoint': observation.endpoint, 'params': {}, 'source': 'open_page'})
        if result['structured_news']['role'] == 'article':
            assert result['structured_news']['html_body_adapter'] == VERSION
            assert result['body_sha256'] == hashlib.sha256(result['text'].encode()).hexdigest()
            assert 'Door je te abonneren gaat u akkoord' not in result['text']
            assert 'Deze link gaat naar een pagina binnen de app.' not in result['text']
            for row in result['structured_news']['segments']:
                assert result['text'][row['start']:row['end']] == row['text']
            recovered.append(result)
        else:
            rejected.append(result)
    assert len(recovered) == 14 and len(rejected) == 6
    assert sum('/categorieen/' in d['url'] for d in rejected) == 3
    preview = next(d for d in recovered if 'vooruitblik-we-kennen' in d['url'])
    assert len(preview['text']) == 1526
    assert 'Stav Lemkin en Max Bruns' in preview['text']
    assert preview['body_sha256'] == '2e7ae4b4cf1a5bb14c830d5247567d35e922e8fd529df0fec76c40c7de8b9842'
    az = next(d for d in recovered if '20-jaar-afas' in d['url'])
    assert len(az['text']) == 4559
    assert az['body_sha256'] == '54411df4015446f73a1f65198f917862fe2b92a21dbc31653d01080742837945'
    assert (root / 'run.json').read_bytes() == before
