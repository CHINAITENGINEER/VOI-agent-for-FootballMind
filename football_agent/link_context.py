"""Preserve observed link labels without fetching or valuing target pages."""
from copy import deepcopy
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .inquiry_contract import normalized
from .inquiry_tools import public_url


VERSION = "observed-html-link-labels-v1"
MAX_LABELS = 8
MAX_CHARACTERS = 512


def _hidden(node):
    attrs = getattr(node, "attrs", None)
    if attrs is None:
        return False
    if node.name in {"script", "style", "noscript", "template"}:
        return True
    if node.has_attr("hidden") or str(node.get("aria-hidden", "")).lower() == "true":
        return True
    declarations = {}
    for declaration in str(node.get("style", "")).split(";"):
        key, separator, value = declaration.partition(":")
        if separator:
            declarations[key.strip().lower()] = value.replace("!important", "").strip().lower()
    return declarations.get("display") == "none" or declarations.get("visibility") in {"hidden", "collapse"}


def retain_link_labels(document, html, allowed_hosts):
    """Annotate already admitted links; never expand the acquisition menu.

    Anchor indexes refer to all original ``a[href]`` nodes, before navigation
    removal or main-content sorting. Parent observation identity and clocks
    remain on the document. Labels are source assertions, not article facts.
    """
    if document.get("status") != "OK" or not document.get("links"):
        return document
    soup = BeautifulSoup(html, "html.parser")
    base = soup.find("base", href=True)
    base_url = urljoin(document["url"], base["href"]) if base else document["url"]
    links = {}
    for link in document["links"]:
        try:
            url = public_url(link["url"], allowed_hosts)
        except (KeyError, TypeError, ValueError):
            continue
        links.setdefault(url, []).append(link)
    observed = {url: [] for url in links}
    seen = {url: set() for url in links}
    truncated = set()
    for index, anchor in enumerate(soup.find_all("a", href=True)):
        # Inspect explicit DOM visibility only; external CSS is not rendered.
        if any(_hidden(node) for node in [anchor, *anchor.parents]):
            continue
        try:
            url = public_url(urljoin(base_url, anchor["href"]), allowed_hosts)
        except (TypeError, ValueError):
            continue
        if url not in observed:
            continue
        visible = deepcopy(anchor)
        for node in list(visible.find_all(True)):
            if _hidden(node):
                node.decompose()
        label = normalized(visible.get_text(" "))
        if not label or label in seen[url]:
            continue
        seen[url].add(label)
        if len(observed[url]) >= MAX_LABELS:
            truncated.add(url)
            continue
        observed[url].append({"anchor_index": index, "text": label[:MAX_CHARACTERS],
                              "text_truncated": len(label) > MAX_CHARACTERS})
    for url, entries in links.items():
        for link in entries:
            link["observed_anchors"] = [dict(item) for item in observed[url]]
            link["observed_anchors_truncated"] = url in truncated
    document["link_context_version"] = VERSION
    return document
