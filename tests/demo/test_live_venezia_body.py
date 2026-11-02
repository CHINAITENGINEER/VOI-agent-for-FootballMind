"""Real source template and negative controls for URL-less NewsArticle binding."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from demo.live_sources import LiveNewsTools
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive


URL = "https://www.veneziafc.it/news/i-convocati-per-genoa-venezia-fc"
AT = datetime(2026, 10, 25, 13, 26, tzinfo=timezone.utc)
TITLE = "I convocati per Genoa - Venezia FC"
PARAGRAPHS = [
    "Sono 29 i giocatori convocati dal tecnico Davide Nicola per la Giornata 8 di Serie A Enilive, che vedrà il Venezia affrontare il Genoa, domani alle ore 15:00 allo Stadio Luigi Ferraris.",
    "Portieri: Matteo Grandi, Lorenzo Montipò, Filip Stankovic.",
]


def html(*, record_changes=None, webpage_changes=None, body=True):
    webpage = {"@type": "WebPage", "url": URL, "@id": URL, "headline": TITLE}
    record = {"@type": "NewsArticle", "headline": TITLE, "datePublished": "2026-10-24T15:18:27.627Z",
              "dateModified": "2026-10-24T15:18:27.627Z", "author": [{"@type": "Person", "name": "Andrea Torelli"}]}
    webpage.update(webpage_changes or {})
    record.update(record_changes or {})
    paragraphs = "".join("<p>" + x + "</p>" for x in PARAGRAPHS) if body else ""
    return f'''<html lang="it"><head><link rel="canonical" href="{URL}">
    <meta property="og:url" content="{URL}"><meta property="og:title" content="{TITLE}">
    <meta property="og:type" content="article"><script type="application/ld+json">{json.dumps([webpage,record])}</script></head>
    <body><nav><p>Navigation must never enter the text.</p></nav><article>
    <header><div style="opacity:0;transform:translateY(40px)"><h1>{TITLE}</h1></div></header>
    <time datetime="2026-10-24T15:18:27.627Z">24/10/2026</time>
    <div class="markdown-story-part"><div class="html-content">{paragraphs}</div></div>
    <aside><p>Related stories must not become the body.</p></aside></article>
    <footer><p>Footer must never enter the text.</p></footer></body></html>'''


def project(tmp_path, value, *, status=200):
    archive = RawArchive(tmp_path)
    obs = archive.save("open_page", URL, {}, value.encode(), status=status, observed_at=AT)
    tools = LiveNewsTools(archive, ["www.veneziafc.it"], projection_version=13, clock=lambda: AT)
    return tools.project(obs, {"tool": "read_url", "endpoint": URL, "params": {}, "source": "open_page"})


def test_same_page_webpage_binds_url_less_news_article_to_real_markdown_body(tmp_path):
    doc = project(tmp_path, html())
    assert doc["status"] == "OK" and doc["structured_news"]["role"] == "article"
    assert doc["text"] == "\n".join(PARAGRAPHS)
    assert doc["title"] not in doc["text"]
    assert doc["body_sha256"] == hashlib.sha256(doc["text"].encode()).hexdigest()
    assert doc["source_authors"] == ["Andrea Torelli"]
    assert doc["source_published_at"] == "2026-10-24T15:18:27.627000+00:00"
    assert not doc["text_truncated"]
    for segment in doc["structured_news"]["segments"]:
        assert doc["text"][segment["start"]:segment["end"]] == segment["text"]
    assert doc["structured_news"]["body_binding"]["article_identifier_absent"]
    assert doc["live_venezia_body_projection"]["network_requests"] == 0


@pytest.mark.parametrize("record_changes", [
    {"url": "https://www.veneziafc.it/news/other-article"},
    {"@id": "https://www.veneziafc.it/news/other-article#article"},
    {"mainEntityOfPage": {"@id": "https://www.veneziafc.it/news/other-article"}},
    {"headline": "Unrelated article headline"},
    {"datePublished": "2026-10-26T15:18:27Z"},
    {"dateModified": "2026-10-26T15:18:27Z"},
    {"dateModified": "2026-10-23T15:18:27Z"},
    {"datePublished": "2026-10-24T15:18:27.627"},
])
def test_conflicting_article_metadata_is_not_rescued(tmp_path, record_changes):
    doc = project(tmp_path, html(record_changes=record_changes))
    assert doc.get("structured_news", {}).get("role") != "article"
    assert doc["live_venezia_body_projection"]["status"] == "rejected"


@pytest.mark.parametrize("changes", [{"url": "https://www.veneziafc.it/news/other"},
                                       {"@id": "https://www.veneziafc.it/news/other"},
                                       {"headline": "A different story"}])
def test_conflicting_webpage_metadata_is_not_rescued(tmp_path, changes):
    doc = project(tmp_path, html(webpage_changes=changes))
    assert doc.get("structured_news", {}).get("role") != "article"


@pytest.mark.parametrize("value", [
    html(body=False),
    html().replace('class="html-content"', 'class="related-content"'),
    html().replace('class="html-content"', 'class="html-content" hidden'),
    html().replace('<article>', '<article hidden>'),
    html().replace('<h1>', '<h1 hidden>'),
    html().replace('>24/10/2026</time>', '>23/10/2026</time>'),
])
def test_title_navigation_or_hidden_body_never_becomes_article(tmp_path, value):
    doc = project(tmp_path, value)
    assert doc.get("structured_news", {}).get("role") != "article"
    assert not doc.get("text")


def test_failed_http_is_not_rescued(tmp_path):
    doc = project(tmp_path, html(), status=503)
    assert doc.get("structured_news", {}).get("role") != "article"


def test_three_archived_rejections_recover_real_bodies_without_network(monkeypatch):
    import requests
    from bs4 import BeautifulSoup
    from football_agent.inquiry_contract import normalized
    root = Path(__file__).resolve().parents[2] / "var/parallel/agent/round2/runs/live-g/runs/lg-ce4e02daacf14f6aa16af076e9cea394"
    if not (root / "run.json").exists():
        pytest.skip("Original local incident archive is not distributed with source")
    original = (root / "run.json").read_bytes()
    run = json.loads(original)
    monkeypatch.setattr(requests.Session, "request", lambda *a, **kw: pytest.fail("Archived replay must not use network"))
    archive = RawArchive(root / "sources")
    tools = LiveNewsTools(archive, run["source_profile"]["allowed_hosts"],
                         index_urls=run["source_profile"]["seed_urls"], projection_version=13,
                         clock=lambda: datetime.now(timezone.utc))
    recovered = []
    for attempt in run["attempts"]:
        if attempt["purpose"] != "g_selected_body":
            continue
        observation = Observation.model_validate(attempt["response"]["observation"])
        raw = archive.read_body(observation)
        assert hashlib.sha256(raw).hexdigest() == observation.content_hash
        doc = tools.project(observation, {"tool": "read_url", "endpoint": attempt["action"]["url"], "params": {}, "source": "open_page"})
        assert doc["status"] == "OK" and doc["structured_news"]["role"] == "article"
        assert len(doc["text"]) > 160 and not doc["text_truncated"]
        assert doc["source_authors"]
        source_paragraphs = BeautifulSoup(raw, "html.parser").select("article .markdown-story-part > .html-content p")
        assert doc["text"] == "\n".join(normalized(p.get_text(" ")) for p in source_paragraphs)
        for seg in doc["structured_news"]["segments"]:
            assert doc["text"][seg["start"]:seg["end"]] == seg["text"]
        recovered.append(doc)
    assert len(recovered) == 3
    assert [len(x["text"]) for x in recovered] == [702, 1425, 638]
    assert [x["body_sha256"] for x in recovered] == [
        "49fba6acc8b5532653616835f7330a55a77394063b596058d1d34bd28ec76c72",
        "6b8eef6fbf368a0f76f4532d0dc0c4337922f97d8974c26d58aa20a195803916",
        "26564b57c6185083ce1cb06bdb87cf3d175ada3570157a8135753184a5223ba9",
    ]
    assert "Sono 29 i giocatori convocati" in recovered[0]["text"]
    assert "Davide Nicola" in recovered[1]["text"]
    assert "PREZZO BIGLIETTI SETTORE OSPITI" in recovered[2]["text"]
    assert (root / "run.json").read_bytes() == original
