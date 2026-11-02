"""Read the bounded literal Nuxt payload of archived Fulham news, without JS.

Only literals (including minified !0/!1 and void 0), arrays, objects and IIFE parameter
references are accepted. No code, calls or property access is evaluated. This adapter proves
the current body's presence, not its availability at a historical kickoff.
"""
from datetime import datetime
import json
import re
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

VERSION = "fulham-nuxt-literal-news-v2"
_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")


class _LiteralReader:
    def __init__(self, text, names=None):
        if len(text) > 2_000_000:
            raise ValueError("literal payload too large")
        self.text, self.names, self.i, self.nodes = text, names or {}, 0, 0

    def space(self):
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def take(self, token):
        self.space()
        if not self.text.startswith(token, self.i):
            raise ValueError("unsupported literal syntax")
        self.i += len(token)

    def value(self, depth=0):
        self.space()
        self.nodes += 1
        if depth > 50 or self.nodes > 50000 or self.i >= len(self.text):
            raise ValueError("literal traversal bound exceeded")
        c = self.text[self.i]
        undefined = re.match(r"void\s+0\b", self.text[self.i:])
        if undefined:
            self.i += undefined.end()
            return None
        if self.text[self.i:self.i+2] in {"!0", "!1"}:
            value = self.text[self.i+1] == "0"
            self.i += 2
            return value
        if c == '"':
            value, end = json.JSONDecoder().raw_decode(self.text, self.i)
            self.i = end
            return value
        if c == "'":
            self.i += 1
            out = []
            escapes = {"'": "'", '"': '"', "\\": "\\", "/": "/", "n": "\n", "r": "\r",
                       "t": "\t", "b": "\b", "f": "\f"}
            while self.i < len(self.text) and self.text[self.i] != "'":
                char = self.text[self.i]
                self.i += 1
                if char == "\\":
                    esc = self.text[self.i]
                    self.i += 1
                    if esc == "u":
                        token = self.text[self.i:self.i+4]
                        if not re.fullmatch(r"[0-9a-fA-F]{4}", token):
                            raise ValueError("invalid string escape")
                        out.append(chr(int(token, 16)))
                        self.i += 4
                    elif esc in escapes:
                        out.append(escapes[esc])
                    else:
                        raise ValueError("unsupported string escape")
                elif char in "\r\n":
                    raise ValueError("unescaped string newline")
                else:
                    out.append(char)
            self.take("'")
            return "".join(out)
        if c in "[{":
            obj = c == "{"
            self.i += 1
            result = {} if obj else []
            end = "}" if obj else "]"
            self.space()
            while not self.text.startswith(end, self.i):
                if obj:
                    if self.text[self.i] == '"':
                        key = self.value(depth+1)
                    else:
                        match = _IDENT.match(self.text, self.i)
                        if not match:
                            raise ValueError("invalid object key")
                        key, self.i = match[0], match.end()
                    self.take(":")
                    if key in result:
                        raise ValueError("duplicate literal key")
                    result[key] = self.value(depth+1)
                else:
                    result.append(self.value(depth+1))
                self.space()
                if self.text.startswith(end, self.i):
                    break
                self.take(",")
                self.space()
            self.take(end)
            return result
        m = _NUMBER.match(self.text, self.i)
        if m:
            self.i = m.end()
            return json.loads(m[0])
        m = _IDENT.match(self.text, self.i)
        if m:
            self.i = m.end()
            if m[0] in {"true", "false", "null"}:
                return {"true": True, "false": False, "null": None}[m[0]]
            if m[0] in self.names:
                return self.names[m[0]]
        raise ValueError("nonliteral JavaScript is not supported")

    def whole(self):
        result = self.value()
        self.space()
        if self.i != len(self.text):
            raise ValueError("trailing executable syntax")
        return result


def literal_nuxt(script):
    match = re.fullmatch(r"\s*window\.__NUXT__\s*=\s*function\(([^()]*)\)\{return(.*)\}\((.*)\);?\s*",
                         script, re.S)
    if not match:
        raise ValueError("unrecognized Nuxt literal envelope")
    names = match[1].split(",") if match[1] else []
    if any(not _IDENT.fullmatch(k) for k in names) or len(names) != len(set(names)):
        raise ValueError("invalid Nuxt literal parameters")
    values = _LiteralReader("["+match[3]+"]").whole()
    if len(values) != len(names):
        raise ValueError("Nuxt parameter arity mismatch")
    return _LiteralReader(match[2], dict(zip(names, values))).whole()


def extract_archived_news(raw, url):
    if urlsplit(url).hostname != "www.fulhamfc.com":
        raise ValueError("unregistered source template")
    soup = BeautifulSoup(raw, "html.parser")
    scripts = [n.get_text() for n in soup.find_all("script") if "window.__NUXT__=" in n.get_text()]
    if len(scripts) != 1:
        raise ValueError("unique Nuxt payload required")
    data = literal_nuxt(scripts[0]).get("data", [])
    if len(data) != 1 or data[0].get("status") != 200:
        raise ValueError("unique successful page record required")
    post = data[0]["data"]
    headings = soup.find_all("h1")
    canonical = soup.select('link[rel="canonical"]')
    titles = soup.select('meta[property="og:title"]')
    if (len(headings) != 1 or len(canonical) != 1 or len(titles) != 1
            or canonical[0].get("href", "").rstrip("/") != url.rstrip("/")):
        raise ValueError("page identity binding failed")
    title = headings[0].get_text(" ", strip=True)
    # Some archived records carry an explicit null `title` and the visible
    # title in `postTitle`; null must not mask the fallback field.
    nuxt_title = post.get("title") or post.get("postTitle")
    og_title = titles[0].get("content")
    if nuxt_title != title:
        raise ValueError("Nuxt title differs from visible page title")
    stamp = post.get("publishedDateTime")
    published = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if published.tzinfo is None:
        raise ValueError("publication timezone missing")
    fragments, paragraphs = [], []
    for index, row in enumerate(post.get("content", [])):
        widget = row.get("rowData", {})
        if widget.get("widgetType") != "TextBlockWidget":
            continue
        text = widget.get("widgetData", {}).get("content")
        if not isinstance(text, str):
            raise ValueError("unsupported text block content")
        parsed = BeautifulSoup(text, "html.parser")
        for node in parsed(["script", "style", "iframe", "noscript", "form"]):
            node.decompose()
        blocks = [n.get_text(" ", strip=True) for n in parsed.find_all(["p", "h2", "h3", "h4", "h5", "h6", "li"])]
        if not blocks:
            text = parsed.get_text(" ", strip=True)
            if text:
                blocks = [text]
        blocks = [s for s in blocks if s]
        fragments.append({"pointer": f"data[0].data.content[{index}].rowData.widgetData.content",
                          "paragraph_count": len(blocks)})
        paragraphs.extend(blocks)
    if not paragraphs:
        raise ValueError("no literal article text blocks")
    return {"title": title, "open_graph_title": og_title,
            "title_binding_basis": "visible_h1_equals_nuxt_title; open_graph_title_retained_separately",
            "body": "\n\n".join(paragraphs), "paragraphs": paragraphs,
            "source_published_at": published.isoformat(), "source_modified_at": None,
            "source_cms_saved_at": post.get("savedTimestamp"),
            "original_time_text": stamp, "published_precision": "timestamp_with_timezone",
            "fixture_tags": post.get("tagsData", {}).get("fixtureTags", []),
            "parser_version": VERSION, "body_pointers": fragments,
            "body_version_at_historical_kickoff": "unknown", "javascript_executed": False}
