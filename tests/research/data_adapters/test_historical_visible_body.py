import json
import pytest
from football_research.data_adapters.historical_visible_body import extract_spurs_visible_body

URL = "https://www.tottenhamhotspur.com/news/123/story"


def page(url=URL, access="free"):
    metadata = {"@type": "NewsArticle", "url": url, "mainEntityOfPage": url, "headline": "Preview", "articleBody": "Intro only"}
    return f'''<link rel="canonical" href="{url}"><script type="application/ld+json">{json.dumps(metadata)}</script>
    <article data-widget="article/article" data-article-url="{url}" data-tracking-content-access-type="{access}">
    <h1>Preview</h1><nav>Navigation</nav><div class="w-article__content w-article__intro"><p>Intro only</p></div>
    <div class="w-article__content js-article-content"><p>He said: last season we lost 0:5.</p><h2>Team</h2><p>Starting XI: actual source list.</p>
    <aside><p>Unrelated later result</p></aside></div></article>'''.encode()


def test_full_visible_body_preserves_quotation_and_excludes_navigation():
    result = extract_spurs_visible_body(page(), URL)
    assert result["paragraphs"] == ["Intro only", "He said: last season we lost 0:5.", "Team", "Starting XI: actual source list."]
    assert result["jsonld_intro_as_declared"] == "Intro only"
    assert "claims" not in result


@pytest.mark.parametrize("raw", [page("https://www.tottenhamhotspur.com/news/another"), page(access="paid")])
def test_rejects_unbound_or_nonpublic_body(raw):
    with pytest.raises(ValueError):
        extract_spurs_visible_body(raw, URL)
