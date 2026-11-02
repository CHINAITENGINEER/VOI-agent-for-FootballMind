"""Append four non-Leeds Premier League historical-reconstruction cases.

This collector keeps public official page bytes and later HTTP receipts.  It
does not read or emit probabilities, VoI labels, or agent-analysis actions.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "footballmind-information-stream"
RESULT_EXPORT = ROOT / "var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda/matches.jsonl"
UA = "FootballMind-research/1.0 (public-source-collection)"

CASES = [
    {
        "source_id": "28870", "source_index": "acceptance-20261007/matches/0001/analysis.json",
        "home_team": "Crystal Palace", "away_team": "Brentford",
        "kickoff": "2025-11-01T15:00:00+00:00",
        "query": "official Crystal Palace Brentford preview team news 1 November 2025",
        "documents": [
            {"url": "https://www.cpfc.co.uk/news/match-previews/view-from-the-opposition-brentford-preview-andrews/",
             "expected_title": "View from the Opposition: \"I have real respect for what Oliver has achieved\"",
             "kind": "official_prematch_opposition_preview", "parser": "cp", "key": "preview"},
            {"url": "https://www.cpfc.co.uk/news/first-team/team-news-announcement-crystal-palace-team-to-face-brentford/",
             "expected_title": "Team News: Your Crystal Palace team to face Brentford",
             "kind": "official_prematch_team_news", "parser": "cp", "key": "team_news"},
        ],
        "candidates": [
            ("preview", "https://www.cpfc.co.uk/news/match-previews/view-from-the-opposition-brentford-preview-andrews/", "View from the Opposition: \"I have real respect for what Oliver has achieved\"", "31 Oct 2025"),
            ("team_news", "https://www.cpfc.co.uk/news/first-team/team-news-announcement-crystal-palace-team-to-face-brentford/", "Team News: Your Crystal Palace team to face Brentford", "1 Nov 2025"),
            ("report", "https://www.cpfc.co.uk/news/first-team/live-blog-report-highlights-crystal-palace-brentford-november-2025/", "Report & Highlights: Crystal Palace vs Brentford", ""),
            ("gallery", "https://www.cpfc.co.uk/news/first-team/matchday-gallery-the-best-images-as-brentford-visit-selhurst-park/", "Matchday Gallery: The best images as Brentford visit Selhurst Park", ""),
            ("match_page", "https://www.cpfc.co.uk/matches/men/epl/2025-26/crystal-palace-vs-brentford-2025-11-01/2561987/", "Crystal Palace vs Brentford", ""),
            ("programme", "https://www.cpfc.co.uk/news/programme/", "Programme", ""),
        ],
    },
    {
        "source_id": "28869", "source_index": "acceptance-20261007/matches/0002/analysis.json",
        "home_team": "Burnley", "away_team": "Arsenal",
        "kickoff": "2025-11-01T15:00:00+00:00",
        "query": "site:arsenal.com Burnley Arsenal pre-match injury team news 1 November 2025",
        "documents": [
            {"url": "https://www.arsenal.com/news/arteta-gives-injury-updates-on-martinelli-saliba-avvaA9p6sm5U",
             "expected_title": "Arteta gives injury updates on Martinelli, Saliba",
             "kind": "official_prematch_fitness_update", "parser": "arsenal", "key": "fitness"},
        ],
        "candidates": [
            ("fitness", "https://www.arsenal.com/news/arteta-gives-injury-updates-on-martinelli-saliba-avvaA9p6sm5U", "Arteta gives injury updates on Martinelli, Saliba", "31 Oct 2025"),
            ("wrong_presser", "https://www.arsenal.com/news/every-word-from-artetas-pre-burnley-presser-aZiQ14f7Xw7i", "Every word from Arteta's pre-Burnley presser", ""),
            ("fixtures", "https://www.arsenal.com/news/arsenals-202526-premier-league-fixture-list-audEc7z6hhrU", "Arsenal's 2025/26 Premier League fixture list", ""),
            ("results", "https://www.arsenal.com/results-and-fixtures-list?field_arsenal_team_target_id=1", "Arsenal results and fixtures", ""),
            ("report", "https://www.arsenal.com/news/report-arsenal-1-0-crystal-palace-a4rW15r72JRJ", "Report: Arsenal 1-0 Crystal Palace", ""),
        ],
    },
    {
        "source_id": "28871", "source_index": "acceptance-20261007/matches/0003/analysis.json",
        "home_team": "Fulham", "away_team": "Wolverhampton Wanderers",
        "kickoff": "2025-11-01T15:00:00+00:00",
        "query": "site:wolves.co.uk Fulham Wolves preview 1 November 2025",
        "documents": [
            {"url": "https://www.wolves.co.uk/news/features/20251031-preview-fulham-vs-wolves/",
             "expected_title": "Preview | Fulham vs Wolves | Features | Latest News | Wolverhampton Wanderers FC",
             "kind": "official_prematch_preview", "parser": "wolves", "key": "preview"},
        ],
        "candidates": [
            ("preview", "https://www.wolves.co.uk/news/features/20251031-preview-fulham-vs-wolves/", "Preview | Fulham vs Wolves", "1st November 2025"),
            ("watch", "https://www.wolves.co.uk/news/mens-first-team/20251101-how-to-watch-fulham-vs-wolves/", "How to watch Fulham vs Wolves", ""),
            ("guide", "https://www.wolves.co.uk/news/club/20251031-matchday-guide-fulham-vs-wolves/", "Matchday guide | Fulham vs Wolves", ""),
            ("report", "https://www.wolves.co.uk/news/mens-first-team/20251101-report-fulham-3-0-wolves/", "Report | Fulham 3-0 Wolves", ""),
            ("blog", "https://www.wolves.co.uk/news/mens-first-team/20251101-matchday-blog-fulham-vs-wolves/", "Matchday blog | Fulham vs Wolves", ""),
        ],
    },
    {
        "source_id": "28874", "source_index": "acceptance-20261007/matches/0005/analysis.json",
        "home_team": "Nottingham Forest", "away_team": "Manchester United",
        "kickoff": "2025-11-01T15:00:00+00:00",
        "query": "site:manutd.com Nottingham Forest Man Utd preview 1 Nov 2025",
        "documents": [
            {"url": "https://www.manutd.com/en/news/detail/how-to-follow-match-preview-nottingham-forest-v-man-utd-1-nov-2025",
             "expected_title": "Forest preview: KO time, form guide, team news",
             "kind": "official_prematch_preview_and_team_news", "parser": "manutd", "key": "preview"},
        ],
        "candidates": [
            ("preview", "https://www.manutd.com/en/news/detail/how-to-follow-match-preview-nottingham-forest-v-man-utd-1-nov-2025", "Forest preview: KO time, form guide, team news", "29 Oct 2025"),
            ("presser_one", "https://www.manutd.com/en/videos/detail/press-conference-part-one-ruben-amorim-previews-forest-v-man-utd-on-1-nov-2025", "Press conference part one: Ruben Amorim previews Forest v Man Utd", ""),
            ("presser_two", "https://www.manutd.com/en/videos/detail/press-conference-part-two-ruben-amorim-forest-v-man-utd-on-1-nov-2025", "Press conference part two: Ruben Amorim on Forest v Man Utd", ""),
            ("interview", "https://www.manutd.com/en/videos/detail/ruben-amorim-pre-match-interview-nottingham-forest-v-man-utd-1-november-2025", "Ruben Amorim pre-match interview: Nottingham Forest v Man Utd", ""),
            ("listing", "https://www.manutd.com/en/news/listing/mens-news?page=55", "Men's news", ""),
            ("other_match", "https://www.manutd.com/en/news/match-preview-how-to-watch-man-utd-v-nottingham-forest-17-may", "Match preview: How to watch Man Utd v Nottingham Forest", ""),
        ],
    },
]


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else None


def norm_time(value: Any) -> str | None:
    dt = parse_time(value)
    return dt.isoformat() if dt else (str(value).strip() if value else None)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\\n" for row in rows), encoding="utf-8")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:90] or "document"


def ld_article(soup: BeautifulSoup) -> dict[str, Any]:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            value = json.loads(script.get_text() or script.string or "")
        except Exception:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, dict) and (item.get("headline") or item.get("@type") in {"NewsArticle", "Article"}):
                return item
    return {}


def clean(text: str) -> str:
    return text.replace("\\xa0", " ").replace("\\r", "").strip()


def block_text(container: Any) -> str:
    if not container:
        return ""
    out = []
    for tag in container.find_all(["h1", "h2", "h3", "h4", "p", "blockquote", "li"]):
        text = clean(tag.get_text(" ", strip=True))
        if text and (not out or out[-1] != text):
            out.append(text)
    return "\\n".join(out)


def parse_cp(raw: bytes) -> dict[str, Any]:
    soup = BeautifulSoup(raw, "html.parser")
    ld = ld_article(soup)
    return {
        "title": ld.get("headline") or (soup.title.get_text(" ", strip=True) if soup.title else ""),
        "publisher": (ld.get("publisher") or {}).get("name") if isinstance(ld.get("publisher"), dict) else "Crystal Palace F.C.",
        "body": block_text(soup.find("main")),
        "published": norm_time(ld.get("datePublished")),
        "modified": norm_time(ld.get("dateModified")),
        "original": ld.get("datePublished"),
        "basis": "article_schema_datePublished",
        "precision": "timestamp" if parse_time(ld.get("datePublished")) else "unknown_or_date_only",
        "body_basis": "BeautifulSoup main h1-h4/p/blockquote/li text blocks in source order",
    }


def parse_arsenal(raw: bytes) -> dict[str, Any]:
    soup = BeautifulSoup(raw, "html.parser")
    ld = ld_article(soup)
    article = {}
    for script in soup.find_all("script"):
        if "articleBody" not in script.get_text():
            continue
        try:
            article = json.loads(script.get_text()).get("props", {}).get("pageProps", {}).get("article", {})
            if article:
                break
        except Exception:
            continue
    body = "\\n".join(clean(item["innerText"]) for item in article.get("articleBody", []) if item.get("type") == "TEXT" and item.get("innerText"))
    published = ld.get("datePublished") or article.get("publicationDate")
    return {
        "title": ld.get("headline") or article.get("title") or "",
        "publisher": (ld.get("publisher") or {}).get("name") if isinstance(ld.get("publisher"), dict) else "Arsenal FC",
        "body": body, "published": norm_time(published), "modified": norm_time(ld.get("dateModified")),
        "original": ld.get("datePublished") or article.get("publicationDate"),
        "basis": "article_schema_datePublished; articleBody publicationDate cross-check",
        "precision": "timestamp" if parse_time(published) else "unknown_or_date_only",
        "body_basis": "Next.js article.articleBody TEXT innerText blocks; promoted blocks excluded",
    }


def parse_wolves(raw: bytes) -> dict[str, Any]:
    soup = BeautifulSoup(raw, "html.parser")
    article = soup.select_one("article.article.container.container--narrow")
    date = soup.select_one(".article--date")
    original = date.get_text(" ", strip=True) if date else "1st November 2025"
    return {
        "title": soup.title.get_text(" ", strip=True) if soup.title else "Preview | Fulham vs Wolves",
        "publisher": "Wolverhampton Wanderers FC", "body": block_text(article),
        "published": None, "modified": None, "original": original,
        "basis": "official_page_date_without_time", "precision": "date_only_or_unknown_time",
        "body_basis": "BeautifulSoup article.article.container.container--narrow h1-h4/p/blockquote/li text blocks",
    }


def find_bodycopy(node: Any) -> Any:
    if isinstance(node, dict):
        candidate = node.get("bodyCopy")
        if isinstance(candidate, dict) and isinstance(candidate.get("json", {}).get("content"), list):
            return candidate["json"]
        for value in node.values():
            found = find_bodycopy(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_bodycopy(value)
            if found is not None:
                return found
    return None


def text_nodes(node: Any) -> list[str]:
    if isinstance(node, dict):
        out = [node["value"]] if node.get("nodeType") == "text" and isinstance(node.get("value"), str) else []
        for key, value in node.items():
            if key != "data":
                out.extend(text_nodes(value))
        return out
    if isinstance(node, list):
        out = []
        for value in node:
            out.extend(text_nodes(value))
        return out
    return []


def parse_manutd(raw: bytes) -> dict[str, Any]:
    soup = BeautifulSoup(raw, "html.parser")
    ld = ld_article(soup)
    body_json = None
    script_index = None
    for index, script in enumerate(soup.find_all("script")):
        text = script.get_text()
        if "bodyCopy" not in text or "self.__next_f.push([1," not in text:
            continue
        try:
            prefix = "self.__next_f.push([1,"
            payload, _ = json.JSONDecoder().raw_decode(text[text.index(prefix) + len(prefix):])
            data = json.loads(payload[payload.find(":") + 1:])
            body_json = find_bodycopy(data)
            if body_json is not None:
                script_index = index
                break
        except Exception:
            continue
    values = [v.replace("\\r", "") for v in text_nodes((body_json or {}).get("content", [])) if v.strip()]
    return {
        "title": ld.get("headline") or (soup.title.get_text(" ", strip=True) if soup.title else ""),
        "publisher": (ld.get("publisher") or {}).get("name") if isinstance(ld.get("publisher"), dict) else "Manchester United",
        "body": "\\n".join(values), "published": norm_time(ld.get("datePublished")),
        "modified": norm_time(ld.get("dateModified")), "original": ld.get("datePublished"),
        "basis": "article_schema_datePublished", "precision": "timestamp" if parse_time(ld.get("datePublished")) else "unknown_or_date_only",
        "body_basis": f"Next.js __next_f bodyCopy JSON text nodes; script_index={script_index}",
    }


PARSERS = {"cp": parse_cp, "arsenal": parse_arsenal, "wolves": parse_wolves, "manutd": parse_manutd}


def fetch(url: str) -> tuple[bytes, dict[str, Any]]:
    last = None
    for attempt in range(1, 4):
        requested, verify = now(), True
        try:
            response = requests.get(url, headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}, timeout=45, verify=True)
        except requests.RequestException as exc:
            last, requested, verify = repr(exc), now(), False
            try:
                response = requests.get(url, headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}, timeout=45, verify=False)
            except requests.RequestException as exc2:
                last = repr(exc2)
                if attempt < 3:
                    time.sleep(1.5 * attempt)
                    continue
                raise RuntimeError(f"GET failed for {url}: {last}") from exc2
        responded = now()
        if response.status_code == 200 and response.content:
            return response.content, {
                "requested_at": iso(requested), "response_at": iso(responded),
                "status_code": response.status_code, "response_url": response.url,
                "request_count": attempt, "tls_verification": verify, "error_type": None,
                "content_type": response.headers.get("content-type"),
                "content_length_header": response.headers.get("content-length"),
                "etag": response.headers.get("etag"), "last_modified_header": response.headers.get("last-modified"),
            }
        last = f"HTTP {response.status_code}; bytes={len(response.content)}"
        if attempt < 3:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"GET failed for {url}: {last}")


def result_rows() -> dict[str, tuple[dict[str, Any], bytes]]:
    out = {}
    with RESULT_EXPORT.open("rb") as f:
        for raw in f:
            if raw.strip():
                row = json.loads(raw.decode("utf-8"))
                out[str(row["source_id"])] = (row, raw if raw.endswith(b"\\n") else raw + b"\\n")
    return out
