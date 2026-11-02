"""Small archived-HTML adapter for the observed public Spurs news template.

The site's JSON-LD articleBody is only its intro. Its visibly rendered full
article is a different representation, retained as a new extraction version.
No runtime/model changes, requests, or semantic claim inference happen here.
"""
import json
from urllib.parse import urlsplit
from bs4 import BeautifulSoup

VERSION = "historical-spurs-visible-article-v1"


def extract_spurs_visible_body(raw, url):
    if urlsplit(url).hostname != "www.tottenhamhotspur.com":
        raise ValueError("unregistered visible news template")
    soup = BeautifulSoup(raw, "html.parser")
    articles = soup.select('article[data-widget="article/article"]')
    canonical = soup.select('link[rel="canonical"]')
    if len(articles) != 1 or len(canonical) != 1 or canonical[0].get("href") != url:
        raise ValueError("unique canonical article required")
    article = articles[0]
    if article.get("data-article-url") != url or article.get("data-tracking-content-access-type") != "free":
        raise ValueError("public same-page article binding required")
    headings = article.find_all("h1")
    if len(headings) != 1:
        raise ValueError("unique article heading required")
    title = headings[0].get_text(" ", strip=True)
    nodes = []
    for tag in soup.find_all("script", type="application/ld+json"):
        node = json.loads(tag.get_text())
        if isinstance(node, dict) and node.get("@type") == "NewsArticle":
            nodes.append(node)
    if (len(nodes) != 1 or nodes[0].get("url") != url or nodes[0].get("headline") != title
            or nodes[0].get("mainEntityOfPage") != url):
        raise ValueError("article metadata conflicts with visible body")
    containers = article.select(".w-article__content")
    if not containers or not article.select(".js-article-content"):
        raise ValueError("registered full article content missing")
    paragraphs, pointers = [], []
    for ci, container in enumerate(containers):
        for ni, block in enumerate(container.find_all(["p", "h2", "h3", "li"])):
            if any(parent.name in {"aside", "nav", "script", "style", "template"}
                   or parent.has_attr("hidden") or parent.get("aria-hidden") == "true"
                   for parent in [block, *block.parents] if getattr(parent, "attrs", None) is not None):
                continue
            text = block.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)
                pointers.append({"container_index": ci, "block_index": ni,
                    "selector": "article[data-widget='article/article'] .w-article__content",
                    "text": text})
    if len(paragraphs) < 2:
        raise ValueError("full article body has fewer than two blocks")
    return {"title": title, "body": "\n\n".join(paragraphs), "paragraphs": paragraphs,
        "body_pointers": pointers, "parser_version": VERSION,
        "jsonld_intro_as_declared": nodes[0].get("articleBody"), "source_time_evidence": {
            "datePublished": nodes[0].get("datePublished"), "dateModified": nodes[0].get("dateModified")},
        "binding_scope": "same-page visible article identity only; historical version and claim truth unverified"}
