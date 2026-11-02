"""Bounded, persistent online experiment using the frozen, paired research f/g.

This deployment is explicitly experimental. It does not promote the historical
model to calibrated production status or rewrite any archived experiment.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import threading
import unicodedata
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4

from .research_g import ROOT, digest, read, now, observed_snapshot
from football_research.acquisition import entity_evidence_updates as f_model
from football_research.acquisition import state_candidate_gain as g_features
from football_research.acquisition import policy_return_continuation_g as g_model
from football_research.acquisition import reviewed_rich_f as rich
from football_research.acquisition.semantic_f import public_input
from football_agent.live_source_assertions import (
    LiveSourceAssertionProtocol as Protocol, RepairLiveSourceAssertionProtocol,
    provider_input, bind_response, with_source_context)
from football_agent import live_source_spans
from football_agent import live_claim_review_v2 as live_claim_review
from .live_f_consumption import consumption_audit, pipeline_summary
from .live_candidate_time import candidate_time_scope
from football_spider.storage.raw_archive import atomic_write, json_bytes, RawArchive

VERSION = "footballmind-live-g-experiment-v3"
SCHEDULE_UI_CONTRACT = "footballmind-daily-schedule-ui-v1"
RUN_ID = re.compile(r"lg-[a-f0-9]{32}\Z")
TERMINAL = {"STOPPED", "FAILED", "CANCELLED", "INTERRUPTED"}
MAX_SOURCE_REQUESTS = 10
MAX_BODY_READS = 5
MAX_PROVIDER_CALLS = 3 * MAX_BODY_READS
MAX_REVIEW_SECONDS = 180
RUN_TIMEOUT_MINUTES = 15
MAX_DYNAMIC_HOSTS = 2
RUNTIME_LOADED_AT = now()
RUNTIME_CODE_FILES = ("demo/live_g.py", "demo/live_sources.py", "demo/live_article_projection.py",
    "demo/live_official_directory_cards.py", "demo/live_venezia_body.py",
    "demo/live_f_consumption.py", "demo/live_candidate_time.py", "demo/live_coverage.py", "demo/daily_fixtures.py",
    "football_agent/live_source_assertions.py", "football_agent/live_source_spans.py", "football_agent/live_claim_review.py", "football_agent/live_claim_review_v2.py", "football_agent/news_discovery.py",
    "football_research/data_adapters/espn_scoreboard.py",
    "football_research/data_adapters/combined_schedule.py",
    "football_research/data_adapters/kleague_official.py",
    "football_research/data_adapters/qiumibao_schedule.py")
RUNTIME_CODE_SHA256 = {name: digest(ROOT / name) for name in RUNTIME_CODE_FILES}
LIMITATIONS = [
    "在线实验：g 由历史研究数据训练，尚未证明在未来比赛上优于固定顺序。",
    "当前在线 g 使用标题与已采用证据状态，不是全文筛选研究模型；非英语候选可能出现特征相同、评分无法区分。",
    "使用与 g 训练绑定的 f；市场先验与实时语义抽取相对训练分布存在变化，尚未校准。",
    "正文经显式语义协议进入固定 f 推理；每步显示实际消费、支持特征和概率变化。当前运行不在线训练模型。",
    "目录发现后最多补充五种搜索意图；正文读取仍由数值 g 自主选择，所有来源请求受有界预算约束。",
    "搜索结果中的未知域名先进入 exact-host 待审核队列；每轮最多批准两个新域名，未通过审核的链接不会被读取。",
    "发布时间未知时，g 的文章年龄采用明确的 168 小时保守插补；不是来源事实。",
    "在线导航保留近30日新闻，来源仅有日期时留2日时区余量；被排除的旧卡仍完整保存在菜单审计，日期不会补成精确发布时间。",
    "查询成本使用声明的 log-loss 换算单位；真实 API 金额未知。",
    "赛果、损失和实际收益需独立赛后结算；当前运行不自动产生训练标签。",
    "跨赛事实验复用英超研究模型，未验证跨联赛效果；来源身份校验与独立官方核验分开显示。",
]


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp requires timezone")
    return result.astimezone(timezone.utc)


def _team_tokens(name):
    """Return distinctive lower-case team tokens for bounded host review."""
    generic = {"united", "city", "county", "town", "club", "football", "fc", "afc", "cf"}
    normalized = unicodedata.normalize("NFKD", str(name).casefold())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return {token for token in re.findall(r"[^\W_]+", normalized)
            if len(token) >= 4 and token not in generic}


def _review_pending_host(item, fixture, allowed_hosts, reviewed_hosts, approved_count):
    """Review one search result whose host was absent from the source profile.

    This is a host admission check, not a relevance or information-value
    judgment.  It only permits a small number of exact HTTPS hosts when the
    observed title/path visibly binds to one of the fixture teams and looks
    like an article.  The target is still fetched as a normal one-attempt
    action and remains subject to the article projection.
    """
    url = str(item.get("url") or "").strip()
    title = str(item.get("title") or "").strip()
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    result = {"url": url, "title": title[:200], "host": host, "status": "rejected", "reason": None}
    if not host or parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in {None, 443}:
        result["reason"] = "unsafe_public_url"
        return result
    from football_agent.inquiry_tools import public_url
    try:
        public_url(url, [host])
    except ValueError:
        result["reason"] = "unsafe_public_url"
        return result
    if host in {"www.bing.com", "cn.bing.com"}:
        result["reason"] = "search_provider_host"
        return result
    if host in {str(h).lower() for h in allowed_hosts}:
        result.update(status="already_allowed", reason="host_already_in_source_profile")
        return result
    if approved_count >= MAX_DYNAMIC_HOSTS:
        result["reason"] = "dynamic_host_review_budget_exhausted"
        return result
    query_keys = {key.lower() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}
    if query_keys & {"token", "api_key", "key", "password", "authorization"}:
        result["reason"] = "credential_query_parameter"
        return result
    path = (parsed.path or "/").rstrip("/").lower()
    navigation_paths = {"", "/", "/news", "/articles", "/stories", "/team", "/teams",
                        "/schedule", "/fixtures", "/results", "/category", "/listing", "/all"}
    article_markers = ("/news/", "/article/", "/articles/", "/story/", "/stories/",
                       "/preview", "/team-news", "/injur", "/lineup", "/squad",
                       "/nieuws/", "/noticias/", "/notizie/", "/actualites/", "/meldungen/")
    sports_markers = ("football", "soccer", "match", "fixture", "preview", "lineup", "injur",
                      "availability", "squad", "team-news", "starting-xi", "manager", "coach",
                      "press-conference", "training", "suspension", "bundesliga", "premier-league",
                      "league", "club", "voetbal", "fútbol", "futbol", "calcio", "futebol",
                      "足球", "中超", "中甲", "赛前", "球队", "伤停", "首发", "サッカー", "スタメン", "축구")
    slug = path.rsplit("/", 1)[-1]
    if path in navigation_paths or not (any(marker in path for marker in article_markers)
            or slug.count("-") >= 2 or re.search(r"\.(?:s?html?)$", path)):
        result["reason"] = "navigation_or_non_article_path"
        return result
    haystack = " ".join((title, host, parsed.path)).lower()
    if not any(marker in haystack for marker in sports_markers):
        result["reason"] = "non_sports_article_signal"
        return result
    from .live_sources import search_team_binding
    if not search_team_binding(title, url, fixture, []):
        result["reason"] = "team_binding_missing"
        return result
    result.update(status="approved", reason="exact_host_article_shape_and_team_binding")
    return result


def _review_pending_host_links(run, tools, doc):
    """Convert bounded RSS pending links into auditable exact-host candidates."""
    structured = doc.get("structured_news") or {}
    pending = structured.get("pending_host_links") or []
    if not pending:
        return
    run.setdefault("host_reviews", [])
    run.setdefault("dynamic_allowed_hosts", [])
    reviewed_hosts = {str(row.get("host") or "").lower() for row in run["host_reviews"] if row.get("host")}
    approved_count = len(run["dynamic_allowed_hosts"])
    existing_urls = {str(link.get("url")) for link in doc.get("links", [])}
    for item in pending:
        review = _review_pending_host(item, run["fixture"], tools.allowed_hosts, reviewed_hosts, approved_count)
        run["host_reviews"].append(review)
        host = review.get("host")
        if host:
            reviewed_hosts.add(host)
        if review["status"] not in {"approved", "already_allowed"} or item.get("url") in existing_urls:
            continue
        tools.allowed_hosts = sorted(set(tools.allowed_hosts) | {host})
        if review["status"] == "approved":
            run["dynamic_allowed_hosts"].append(host)
            approved_count += 1
        existing_urls.add(item["url"])
        doc.setdefault("links", []).append({"url": item["url"], "title": item.get("title", ""),
            "source_published_at": item.get("source_published_at"),
            "publication_raw": item.get("publication_raw"), "host_review": review})


def _directory_candidate_reason(url, title, fixture, entries, seen):
    """Keep directory links that visibly relate to this fixture context.

    This is a navigation projection, not a value score.  It removes obvious
    galleries, fan/business pages and old match reports from the g menu while
    retaining every observed link in ``run['menus']`` with its reason.
    """
    from .live_sources import article_link_reason
    reason = article_link_reason(url, title, entries, seen, search=False)
    if reason is not None:
        return reason
    text = (str(title) + " " + str(url)).lower()
    target_text = (str(title) + " " + (urlsplit(url).path or "")).lower()
    # Explicit separate squad paths must not be folded into a men's fixture,
    # even when the club name matches. All links remain in the observed menu.
    if any(part in urlsplit(url).path.lower().split("/") for part in
           {"damlaget", "women", "frauen", "youth", "u17", "u19", "u21", "u23",
            "vrouwen", "jeugd", "o17", "o19", "onder21", "fortuniors"}):
        return "directory_other_squad_path"
    if re.search(r"\b(?:u|o|onder)[-_ ]?(?:17|18|19|20|21|23)\b|\b(?:women|frauen|feminin\w*|femenin\w*|damlaget|dames|vrouwen|jeugd|fortuniors)\b", str(title), re.I):
        return "directory_other_squad_label"
    def name_tokens(names):
        values = set()
        for name in names:
            values.update(_team_tokens(name))
        return values - {"united", "city", "club", "team"}
    home_tokens = name_tokens([fixture.get("home_name"), *(fixture.get("home_aliases") or [])])
    away_tokens = name_tokens([fixture.get("away_name"), *(fixture.get("away_aliases") or [])])
    host_text = (urlsplit(url).hostname or "").lower()
    if any(token in host_text for token in home_tokens):
        target_tokens = away_tokens
    elif any(token in host_text for token in away_tokens):
        target_tokens = home_tokens
    else:
        target_tokens = home_tokens | away_tokens
    target_bound = any(token in target_text for token in target_tokens)
    category_words = ("bildergalerie", "gallery", "fans", "business", "verein", "nachwuchs",
                      "jugend", "ticket", "kidsclub", "teensclub", "u19", "u17", "u23",
                      "frauen", "innen", "women", "junior", "damlaget", "biljetter", "publikinformation")
    if any(word in str(title).lower() for word in category_words) and not target_bound:
        return "directory_non_match_category"
    context_markers = ("injur", "verlet", "kader", "aufstell", "lineup", "availability",
                        "training", "trainer", "coach", "presse", "press conference",
                        "sperre", "suspension", "einsatz")
    official_host = host_text in {urlsplit(entry).hostname for entry in entries}
    league_hosts = {"www.premierleague.com", "www.bundesliga.com", "www.laliga.com", "spfl.co.uk", "www.jleague.jp"}
    if official_host and host_text in league_hosts:
        from .live_sources import search_team_binding
        return None if search_team_binding(title, url, fixture, []) else "league_directory_team_binding_missing"
    if target_bound or official_host or ("profis" in text and any(marker in text for marker in context_markers)):
        return None
    return "directory_context_not_match_bound"


def _evidence_highlights(claims, body):
    """Build exact, claim-backed excerpts without summarizing the article.

    The priority fields are a *presentation projection* only.  They do not
    change the provider response, f input, or g action choice.  Their purpose
    is to put fixture-bound, model-relevant claims before generic context in
    the audit view while retaining every claim (including unverified ones).
    """
    if not isinstance(claims, list) or not isinstance(body, str):
        return []

    def priority(claim, verified):
        scope = str(claim.get("scope") or "unknown").lower()
        kind = str(claim.get("kind") or "unknown").lower()
        value = str(claim.get("value") or "unknown").lower()
        score, reasons = 0, []
        if scope == "this_match":
            score += 5
            reasons.append("this_match_scope")
        elif scope in {"team_context", "team"}:
            score += 2
            reasons.append("team_context")
        if kind in {"availability", "lineup", "suspension", "tactical",
                    "training", "schedule_pressure", "motivation", "odds"}:
            score += 3
            reasons.append("supported_kind:" + kind)
        if value in {"unavailable", "suspended", "starting", "benched",
                     "available", "ruled_out", "injured", "returning"}:
            score += 2
            reasons.append("explicit_state:" + value)
        elif value in {"uncertain", "unknown", "pending", "conditional"}:
            score += 1
            reasons.append("conditional_state:" + value)
        elif value in {"unmodeled", "other"}:
            score -= 1
            reasons.append("unmodeled_context")
        if verified:
            score += 2
            reasons.append("quote_verified")
        else:
            reasons.append("quote_unverified")
        label = "高" if score >= 9 else "中" if score >= 5 else "低"
        return score, label, reasons

    highlights, seen = [], set()
    for claim_index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        quotes = claim.get("quotes") or []
        if not isinstance(quotes, list):
            quotes = []
        if not quotes:
            score, label, reasons = priority(claim, False)
            highlights.append({"claim_index": claim_index, "entity": claim.get("entity"),
                "team": claim.get("team"), "kind": claim.get("kind"), "scope": claim.get("scope"),
                "value": claim.get("value"), "statement": claim.get("statement"),
                "quote": None, "quote_verified": False, "body_start": None, "body_end": None,
                "body_context": None, "importance_score": score,
                "importance_label": label, "importance_reasons": reasons,
                "priority_basis": "deterministic_claim_priority_projection_v1"})
            continue
        for quote in quotes:
            if not isinstance(quote, str) or not quote.strip():
                continue
            key = (claim_index, quote)
            if key in seen:
                continue
            seen.add(key)
            start = body.find(quote)
            verified = start >= 0
            score, label, reasons = priority(claim, verified)
            context = None
            if verified:
                context_start = max(0, start - 120)
                context_end = min(len(body), start + len(quote) + 120)
                context = body[context_start:context_end]
            highlights.append({"claim_index": claim_index, "entity": claim.get("entity"),
                "team": claim.get("team"), "kind": claim.get("kind"), "scope": claim.get("scope"),
                "value": claim.get("value"), "statement": claim.get("statement"),
                "quote": quote, "quote_verified": verified,
                "body_start": start if verified else None,
                "body_end": start + len(quote) if verified else None,
                "body_context": context, "importance_score": score,
                "importance_label": label, "importance_reasons": reasons,
                "priority_basis": "deterministic_claim_priority_projection_v1"})
    for rank, item in enumerate(sorted(highlights,
                                       key=lambda h: (-h["importance_score"], h["claim_index"])), 1):
        item["importance_rank"] = rank
    return sorted(highlights, key=lambda h: h["importance_rank"])


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json_bytes(value))


def key_from_environment():
    """Desktop may predate User environment updates; never serialize this value."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
                key = winreg.QueryValueEx(handle, "DEEPSEEK_API_KEY")[0]
        except OSError:
            pass
    return key.strip() if isinstance(key, str) and key.strip() else None


class LiveBackend:
    """Existing collectors, archive, source projection and typed provider seams."""

    def __init__(self, *, catalog_root=None):
        self.catalog_root = catalog_root

    def inventory(self, folder):
        from football_research.data_adapters.prospective_fixture_inventory import collect_inventory
        return collect_inventory(folder, minimum_fixtures=1, preferred_fixtures=3)

    def scoreboard(self, folder):
        from football_research.data_adapters.combined_schedule import collect_schedule
        return collect_schedule(folder)

    def fixture_scoreboard(self, fixture, folder):
        from football_research.data_adapters.espn_scoreboard import ESPNScoreboard
        return ESPNScoreboard(folder).collect_fixture(fixture, catalog_root=self.catalog_root)

    @staticmethod
    def _secondary_espn_fixture(fixture):
        """Return only the independently stored, uniquely exact ESPN join."""
        candidate = fixture.get("espn_initialization_fixture")
        if candidate is None:
            return None
        from .daily_fixtures import key
        if (not isinstance(candidate, dict)
                or fixture.get("cross_source_match_basis") != "unique_exact_competition_home_away_kickoff_explicit_aliases"
                or candidate.get("source") != "espn"
                or candidate.get("match_id") != fixture.get("espn_match_id")
                or candidate.get("match_id") != "espn:match:" + str(candidate.get("source_match_id"))
                or candidate.get("fixture_id") != candidate.get("match_id")
                or not all(candidate.get(s+"_team_source_id") for s in ("home", "away"))
                or candidate.get("home_team_source_id") == candidate.get("away_team_source_id")
                or key(candidate) != key(fixture)):
            raise ValueError("ESPN备用来源缺少唯一精确赛事、主客队、开球时间或来源ID绑定")
        return deepcopy(candidate)

    def _initialize_espn(self, fixture, folder, fallback_error):
        """Recheck an ESPN-only fixture with its observed 1X2 moneyline.

        ESPN is accepted here only when the fresh response contains a complete
        home/draw/away moneyline.  A schedule without that snapshot remains
        blocked; no historical or synthetic prior is invented.
        """
        report = self.fixture_scoreboard(fixture, Path(folder) / "espn")
        target_id = str(fixture.get("fixture_id") or fixture.get("match_id"))
        match = next((item for item in report.get("matches", []) if item.get("match_id") == target_id), None)
        if match is None and report.get("status") == "failed":
            raise ValueError("初始化失败：ESPN 来源请求失败，未能核验该场报价；不是已确认缺少赔率") from fallback_error
        if match is None:
            raise ValueError("初始化失败：ESPN 本次响应没有返回所选比赛") from fallback_error
        if match.get("status") != "scheduled":
            raise ValueError("初始化失败：所选比赛已不处于未开赛状态") from fallback_error
        if match.get("time_confirmed") is False:
            raise ValueError("初始化失败：ESPN 本次响应尚未确认开球时间") from fallback_error
        if not match.get("quote_snapshot"):
            raise ValueError("初始化失败：ESPN 未返回该场完整赛前 1X2 先验") from fallback_error
        quote = match["quote_snapshot"]
        probabilities = [quote["no_vig_probabilities"][side] for side in ("home", "draw", "away")]
        kickoff = utc(match["kickoff"])
        cutoff = kickoff - timedelta(minutes=1)
        raw = match.get("raw_observation") or {}
        source_manifest = [f"espn:observation:{raw.get('id')}" if raw.get("id") else "espn:observation:unknown"]
        identity = {
            "match_id": target_id,
            "home_team_id": "espn:team:" + str(match["home_team_source_id"]),
            "away_team_id": "espn:team:" + str(match["away_team_source_id"]),
            "home_team_name": match["home"], "away_team_name": match["away"],
            "competition_name": match["competition_key"], "kickoff": kickoff.isoformat(),
            "status": match["status"], "input_manifest": source_manifest,
        }
        result = {
            "schema": "footballmind-espn-market-initialization-v1",
            "market_initialization": {
                "fixture": identity, "prediction_cutoff": cutoff.isoformat(),
                "probabilities": probabilities, "quote_snapshot": quote,
                "source": "espn", "source_observed_at": match.get("observed_at"),
                "raw_sha256": (raw.get("content_hash") or None),
                "input_manifest": source_manifest,
            },
            "acquisition_mode": report.get("acquisition_mode", "fresh_target_day"),
            "fallback_from": type(fallback_error).__name__ if fallback_error else None,
            "fallback_reason": str(fallback_error)[:180] if fallback_error else None,
        }
        save(Path(folder) / "espn-initialization.json", result)
        return result

    def initialize(self, fixture, folder):
        if str(fixture.get("source")) in {"qiumibao", "kleague_official"}:
            raise ValueError("该来源目前只有赛程、尚无真实完整90分钟胜平负先验，不能启动预测")
        if str(fixture.get("source")) == "espn" or str(fixture.get("match_id", "")).startswith("espn:"):
            return self._initialize_espn(fixture, folder, None)
        secondary = self._secondary_espn_fixture(fixture)
        def initialize_secondary(error):
            if secondary is None:
                raise error
            result = self._initialize_espn(secondary, folder, error)
            result["analysis_fixture"] = secondary
            result["source_switch"] = {
                "display_source": fixture.get("source", "spftll"),
                "display_match_id": fixture.get("match_id"),
                "initialization_source": "espn", "initialization_match_id": secondary["match_id"],
                "binding_basis": fixture["cross_source_match_basis"],
                "reason": "primary_quote_missing" if fixture.get("initialization_source") == "espn"
                          else "primary_initialization_failed"}
            save(Path(folder) / "source-switch.json", result["source_switch"])
            return result
        if fixture.get("initialization_source") == "espn":
            return initialize_secondary(ValueError("主源目录无完整赔率，改由精确匹配ESPN比赛重新获取先验"))
        from football_research.data_adapters.prospective_fixture_inventory import _run_collection
        from football_research.data_adapters.prospective_text import select_initializations
        from football_research.data_adapters.spftll import load_spftll_archive
        from football_spider.platform_config import PlatformSettings
        from zoneinfo import ZoneInfo
        started = datetime.now(timezone.utc)
        local = started.astimezone(ZoneInfo("Asia/Shanghai")).date()
        cfg = PlatformSettings(data_dir=folder, timezone="Asia/Shanghai", sources=["spftll"],
                               max_requests=2, detail_limit=1)
        try:
            result = _run_collection(cfg, start=local, end=local+timedelta(days=42),
                                     match_ids=[fixture["source_match_id"]])
        except Exception as exc:
            if secondary is None:
                raise
            return initialize_secondary(exc)
        save(folder / "collection.json", result.model_dump(mode="json"))
        completed = datetime.now(timezone.utc)
        data = load_spftll_archive(folder / "raw", as_of=completed)
        selection = select_initializations(data, as_of=completed, limit=1, horizon_hours=1008,
                    cutoff_lead_minutes=1, fixture_ids=[fixture["fixture_id"]])
        save(folder / "initialization-selection.json", selection)
        if not selection["selected"]:
            error = ValueError("初始化失败：缺少新鲜完整的 90 分钟赔率，或距离开球不足 30 分钟")
            if secondary is not None:
                return initialize_secondary(error)
            raise error
        return selection["selected"][0]

    def tools(self, folder, seeds, hosts):
        from .live_sources import LiveNewsTools
        official_hosts = {urlsplit(seed).hostname for seed in seeds if urlsplit(seed).hostname}
        route_overrides = {host: "direct" for host in official_hosts}
        return LiveNewsTools(RawArchive(folder), hosts, index_urls=seeds, projection_version=13,
                             route_overrides=route_overrides)

    def identity(self, manifest, folder, expected):
        from .live_sources import SCHEDULE_URL, bind_identity
        from .live_coverage import bind_source_identity, PL_NAMES, TEAMS
        if manifest.get("source_switch"):
            secondary = self._secondary_espn_fixture(expected)
            switch = manifest["source_switch"]
            if (secondary is None or manifest.get("analysis_fixture") != secondary
                    or switch.get("display_match_id") != expected.get("match_id")
                    or switch.get("initialization_source") != "espn"
                    or switch.get("initialization_match_id") != secondary["match_id"]):
                raise ValueError("备用来源初始化与原始目录精确身份不一致")
            expected = secondary
        init_fixture = manifest["market_initialization"]["fixture"]
        if str(init_fixture.get("match_id", "")).startswith("espn:") or str(expected.get("source", "")) == "espn":
            if (init_fixture.get("match_id") != expected.get("match_id")
                    or init_fixture.get("home_team_id") != "espn:team:" + str(expected.get("home_team_source_id"))
                    or init_fixture.get("away_team_id") != "espn:team:" + str(expected.get("away_team_source_id"))
                    or init_fixture.get("home_team_name") != expected.get("home")
                    or init_fixture.get("away_team_name") != expected.get("away")
                    or init_fixture.get("competition_name") != expected.get("competition_key")
                    or utc(init_fixture.get("kickoff")) != utc(expected.get("kickoff"))
                    or init_fixture.get("status") != "scheduled"):
                raise ValueError("ESPN 比赛身份、主客队、赛事或开球时间与所选目录不一致")
            binding = {
                "schema": "footballmind-live-source-identity-espn-v1", "status": "SOURCE_CONFIRMED",
                "identity_level": "espn_event_and_observed_moneyline", "independently_confirmed": False,
                "canonical": {"source_event_id": init_fixture["match_id"],
                    "home_name": init_fixture["home_team_name"], "away_name": init_fixture["away_team_name"],
                    "kickoff": init_fixture["kickoff"], "competition_name": init_fixture["competition_name"],
                    "season": None},
                "aliases": {"home": [init_fixture["home_team_name"]], "away": [init_fixture["away_team_name"]]},
                "verified_at": now(), "source_event_id_is_local_namespace": False,
                "basis": "exact_espn_event_team_ids_names_competition_kickoff_and_observed_moneyline",
                "initialization_inputs": init_fixture.get("input_manifest", []),
                "limitation": "ESPN公开响应的身份与观察到的moneyline；尚无第二独立身份源",
            }
            save(folder / "identity.json", binding)
            return binding
        source_binding = bind_source_identity(manifest, expected, now())
        # Keep the established official check when its declared scope applies.
        official_scope = (expected["competition_key"] in PL_NAMES
            and utc(expected["kickoff"]).strftime("%Y-%m") == "2026-10"
            and all(TEAMS.get(expected[s+"_team_source_id"], [None])[0] == expected[s+"_team"]
                    for s in ("home", "away")))
        if not official_scope:
            save(folder / "identity.json", source_binding)
            return source_binding
        tools = self.tools(folder, [SCHEDULE_URL], ["www.premierleague.com"])
        prepared = tools.prepare({"tool": "read_url", "url": SCHEDULE_URL},
                                 {"task": {"seed_urls": [SCHEDULE_URL]}, "documents": []})
        returned = tools.execute(prepared, run_id=folder.parent.name, request_id="identity-"+uuid4().hex)
        binding = bind_identity(manifest, returned, RawArchive(folder), now())
        binding.update(identity_level="independent_official_schedule", independently_confirmed=True,
                       source_binding=source_binding)
        save(folder / "identity.json", binding)
        if binding["status"] != "CONFIRMED":
            raise ValueError("官方比赛身份/开球核验未通过："+binding["reason"])
        return binding

    def semantics(self, folder, public, timeout):
        return self._semantics(folder, public, timeout, Protocol)

    def repair_semantics(self, folder, public, timeout):
        return self._semantics(folder, public, timeout, live_source_spans.SpanRepairLiveSourceAssertionProtocol)

    def review_claims(self, folder, public, claims, timeout):
        from football_agent.deepseek_provider import DeepSeekInquiryProvider
        self.last_semantic_attempt = None
        protocol=live_claim_review.LiveClaimReviewProtocol
        request=live_claim_review.provider_input(public,claims,protocol=protocol)
        provider=DeepSeekInquiryProvider(folder,protocol=protocol,api_key=key_from_environment(),
            thinking="enabled",max_tokens=16384,timeout_seconds=min(MAX_REVIEW_SECONDS,timeout))
        try:
            returned=provider.decide(request)
            try:
                review,binding=live_claim_review.bind_response(public,claims,request,returned['response'],protocol=protocol)
            except ValueError as exc:
                raise ValueError('Live claim review binding rejected; no claims admitted') from exc
        finally:
            self.last_semantic_attempt={"folder":provider.last_attempt_folder,"receipt":provider.last_receipt}
        return {"input":request,"response":returned['response'],"review":review,"binding":binding,"receipt":returned['receipt']}

    def _semantics(self, folder, public, timeout, protocol):
        from football_agent.deepseek_provider import DeepSeekInquiryProvider
        self.last_semantic_attempt = None
        provider = DeepSeekInquiryProvider(folder, protocol=protocol, api_key=key_from_environment(),
                    thinking="disabled", max_tokens=8192, timeout_seconds=min(120, timeout))
        span_repair = protocol is live_source_spans.SpanRepairLiveSourceAssertionProtocol
        request_builder = live_source_spans.provider_input if span_repair else provider_input
        response_binder = live_source_spans.bind_response if span_repair else bind_response
        request = request_builder(public, protocol=protocol)
        try:
            returned = provider.decide(request)
            bound, binding = response_binder(public, request, returned["response"], protocol=protocol)
        finally:
            self.last_semantic_attempt = {"folder": provider.last_attempt_folder, "receipt": provider.last_receipt}
        return {"input": public, "response": bound, "binding": binding, "receipt": returned["receipt"]}


class LiveGApplication:
    def __init__(self, store_dir, *, root=ROOT, backend=None, clock=now):
        self.root, self.store_dir = Path(root), Path(store_dir)
        self.backend, self.clock = backend or LiveBackend(catalog_root=Path(store_dir) / "espn"), clock
        self.lock = threading.RLock()
        self.workers, self.cancel = {}, {}
        self.catalog_worker = None
        self.auto_catalog = backend is None
        self.refresh_not_before = None
        self.f_path = self.root / "var/reviews/fm-historical-transition-g-training-20261014-v2/f-labeler.json"
        self.g_path = self.root / "var/reviews/fm-g-policy-return-continuation-20261014-v1/g-policy-return.json"
        self.f, self.g = f_model.load(self.f_path), g_model.load(self.g_path)
        self.bindings = {"f": digest(self.f_path), "g": digest(self.g_path)}
        if self.g["f_labeler_sha256"] != self.bindings["f"]:
            raise ValueError("g 与 f 工件不配套")

    def _folder(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid live run ID")
        return self.store_dir / "runs" / run_id

    def _save(self, run):
        run["updated_at"] = self.clock()
        run["revision"] += 1
        run["body_pipeline"] = pipeline_summary(run)
        with self.lock:
            save(self._folder(run["run_id"]) / "run.json", run)

    def catalog(self):
        from .live_coverage import availability, summarize
        with self.lock:
            path = self.store_dir / "catalog.json"
            result = read(path) if path.exists() else {"matches": [], "status": "not_refreshed", "updated_at": None}
            stamp = self.clock()
            check_at = result.get("last_checked_at") or result.get("updated_at")
            stale = not check_at or utc(stamp)-utc(check_at) >= timedelta(seconds=60)
            if (self.auto_catalog and stale
                    and not (self.catalog_worker and self.catalog_worker.is_alive())
                    and (not self.refresh_not_before or utc(stamp) >= utc(self.refresh_not_before))):
                self.refresh_not_before = (utc(stamp)+timedelta(seconds=60)).isoformat()
                self.catalog_worker = threading.Thread(target=self._refresh, daemon=True)
                self.catalog_worker.start()
            result.update(schema_version=VERSION, mode="live_experiment", limitations=LIMITATIONS,
                          provider_configured=bool(key_from_environment()), model_bindings=self.bindings)
            result["runtime_bindings"] = {"loaded_at": RUNTIME_LOADED_AT, "process_id": os.getpid(),
                "code_sha256": dict(RUNTIME_CODE_SHA256),
                "reload_required": any(digest(ROOT / p) != value for p, value in RUNTIME_CODE_SHA256.items())}
            for match in result["matches"]:
                match["can_start"], match["availability_reason"] = availability(match, self.clock())
            result["coverage"] = summarize(result["matches"])
            result["refreshing"] = bool(self.catalog_worker and self.catalog_worker.is_alive())
            result["auto_refresh_seconds"] = 60 if self.auto_catalog else None
            # Loaded Python code advertises this capability, never a disk cache.
            # Static assets can update while an older server process stays alive.
            result["schedule_ui_contract"] = SCHEDULE_UI_CONTRACT
            result["server_time"] = stamp
            result["runs"] = []
            for p in sorted((self.store_dir / "runs").glob("lg-*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
                r = read(p)
                result["runs"].append({k: r[k] for k in ("run_id", "match", "state", "created_at")})
            return result

    def refresh(self, payload):
        if payload:
            raise ValueError("刷新不接受客户端来源配置")
        with self.lock:
            if self.catalog_worker and self.catalog_worker.is_alive():
                return self.catalog()
            if self.refresh_not_before and utc(self.clock()) < utc(self.refresh_not_before):
                return self.catalog()
            self.refresh_not_before = (utc(self.clock())+timedelta(seconds=60)).isoformat()
            self.catalog_worker = threading.Thread(target=self._refresh, daemon=True)
            self.catalog_worker.start()
            return self.catalog()

    def _refresh(self):
        from .live_coverage import catalog_match
        from .daily_fixtures import merge_schedule
        from concurrent.futures import ThreadPoolExecutor
        folder = self.store_dir / "inventories" / uuid4().hex
        cache = self.store_dir / "spftll-catalog-cache.json"
        prior = read(cache) if cache.exists() else None
        def inventory():
            if prior and utc(self.clock())-utc(prior["observed_at"]) < timedelta(minutes=5):
                return prior["matches"], {"status": "ready", "cache_hit": True,
                    "observed_at": prior["observed_at"], "http_attempts": 0}
            try:
                report = self.backend.inventory(folder)
                matches = [catalog_match(row) for row in report["candidate_registrations"]]
                observed = self.clock()
                for match in matches:
                    match["quote_catalog_observed_at"] = observed
                save(cache, {"observed_at": observed, "matches": matches})
                return matches, {"status": "ready", "cache_hit": False, "observed_at": observed,
                    "http_attempts": report["request_budget"]["actual_archived_attempts"]}
            except Exception as exc:
                # A failed refresh must not erase the last known schedule.
                return (prior or {}).get("matches", []), {"status": "failed", "error": type(exc).__name__,
                    "observed_at": (prior or {}).get("observed_at"), "http_attempts": None}
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                primary = pool.submit(inventory)
                other = pool.submit(self.backend.scoreboard, self.store_dir / "espn") if hasattr(self.backend, "scoreboard") else None
                matches, source_status = primary.result()
                try:
                    board = other.result() if other else None
                except Exception as exc:
                    # Source outages are independent: usable primary fixtures
                    # must still refresh even if the secondary service fails.
                    old_path = self.store_dir / "catalog.json"
                    old = read(old_path) if old_path.exists() else {}
                    board = {"status": "failed", "error": type(exc).__name__, "dates": [],
                        "matches": [{**m, "source_stale": True} for m in old.get("matches", [])
                                    if m.get("source") == "espn"]}
            sources = {"spftll": source_status}
            merging = None
            if board:
                matches, merging = merge_schedule(matches, board["matches"], self.clock())
                sources["espn"] = {k: v for k, v in board.items() if k not in {"matches", "dates"}}
                if "primary_source_status" in board:
                    sources["espn"]["status"] = board["primary_source_status"]
                    sources["espn"]["http_attempts"] = board["primary_source_http_attempts"]
                for source, status in board.get("supplemental_sources", {}).items():
                    sources[source] = status
                sources["espn"]["dates"] = [{k: r.get(k) for k in ("date_utc", "observed_at", "stale", "error", "cache_hit")}
                                             for r in board["dates"]]
            result = {"matches": matches,
                      "status": "ready" if all(s["status"] == "ready" for s in sources.values()) else "partial" if matches else "failed",
                      "updated_at": self.clock(), "last_checked_at": self.clock(), "inventory_path": str(folder),
                      "setup_attempts": source_status["http_attempts"], "sources": sources, "schedule_merge": merging}
        except Exception as exc:
            path = self.store_dir / "catalog.json"
            result = read(path) if path.exists() else {"matches": [], "updated_at": None}
            result.update(status="refresh_failed", last_checked_at=self.clock(), error=type(exc).__name__)
        with self.lock:
            save(self.store_dir / "catalog.json", result)

    def start(self, payload):
        if set(payload) - {"match_id", "max_reads", "cost_loss_units"}:
            raise ValueError("仅接受比赛、阅读预算与 loss 单位成本")
        budget, cost = payload.get("max_reads", MAX_BODY_READS), payload.get("cost_loss_units", .001)
        if type(budget) is not int or not 1 <= budget <= MAX_BODY_READS:
            raise ValueError(f"在线实验范围：最多读取 1–{MAX_BODY_READS} 篇")
        if type(cost) not in (int, float) or not math.isfinite(cost) or not 0 <= cost <= .01:
            raise ValueError("成本必须为 0–0.01 loss 单位")
        with self.lock:
            catalog = self.catalog()
            if catalog["runtime_bindings"]["reload_required"]:
                raise ValueError("服务加载的源码与磁盘版本不同，请重启 FootballMind 后再运行")
            if not catalog["updated_at"] or utc(self.clock())-utc(catalog["updated_at"]) > timedelta(minutes=30):
                raise ValueError("请先刷新未来比赛目录（有效期 30 分钟）")
            match = next((r for r in catalog["matches"] if r["match_id"] == payload.get("match_id")), None)
            if not match or not match["can_start"]:
                raise ValueError("比赛不在当前可执行赛前目录内")
            if any(t.is_alive() for t in self.workers.values()):
                raise ValueError("已有在线实验执行中，请等待结束或停止当前实验")
            if not key_from_environment():
                raise ValueError("执行进程或 Windows 用户环境缺少 DEEPSEEK_API_KEY")
            rid = "lg-"+uuid4().hex
            folder = self._folder(rid)
            folder.mkdir(parents=True)
            for name, path in (("f", self.f_path), ("g", self.g_path)):
                if digest(path) != self.bindings[name]:
                    raise ValueError("模型文件已变化，请重启服务后核对")
                (folder / (name+".json")).write_bytes(path.read_bytes())
            run = {"schema_version": VERSION, "run_id": rid, "match": match, "created_at": self.clock(),
                   "updated_at": self.clock(), "revision": 0, "state": "PREPARING", "reason": None,
                   "stage": "刷新目标比赛先验", "model_bindings": self.bindings, "limitations": LIMITATIONS,
                   "selection_authority": "experimental_paired_g", "production_ready": False,
                   "runtime_bindings": deepcopy(catalog["runtime_bindings"]),
                   "max_reads": budget, "cost_loss_units": cost, "current": None, "before": None,
                   "decisions": [], "attempts": [], "steps": [], "documents": [], "menus": [],
                   "semantic_attempts": [], "body_rejections": [],
                   "evidence_events": [], "used_input_ids": [], "prediction_cutoff": None,
                   "forecast_at": None, "pending": None, "outcome": None, "gain": None,
                   "counts": {"source": 0, "provider": 0, "body_attempts": 0},
                   "host_reviews": [], "dynamic_allowed_hosts": [], "discovery_diagnostics": [],
                   "g_age_policy": "published_age_else_explicit_168h_imputation",
                   "source_request_budget": MAX_SOURCE_REQUESTS,
                   "provider_request_budget": MAX_PROVIDER_CALLS,
                   "run_timeout_minutes": RUN_TIMEOUT_MINUTES,
                   "semantic_protocol": Protocol.version,
                   "semantic_repair_protocol": live_source_spans.SpanRepairLiveSourceAssertionProtocol.version,
                   "claim_review_protocol": live_claim_review.LiveClaimReviewProtocol.version,
                   "g_training_eligible": False}
            self.cancel[rid] = threading.Event()
            self._save(run)
            worker = threading.Thread(target=self._work, args=(run,), daemon=True)
            self.workers[rid] = worker
            worker.start()
            return deepcopy(run)

    def get(self, rid):
        with self.lock:
            run = read(self._folder(rid) / "run.json")
            if run["state"] not in TERMINAL and not (rid in self.workers and self.workers[rid].is_alive()):
                run.update(state="INTERRUPTED", reason="服务已中断；已发出的请求不会自动重试", stage="需要新建独立运行")
                self._save(run)
            # Read-only projection for archived runs: unknown route counts stay unknown.
            run["body_pipeline"] = pipeline_summary(run)
            return run

    def stop(self, rid, payload):
        if payload:
            raise ValueError("停止操作不接受额外配置")
        with self.lock:
            run = self.get(rid)
            if run["state"] not in TERMINAL:
                self.cancel[rid].set()
            return {**run, "cancel_requested": run["state"] not in TERMINAL}

    def _check(self, run):
        if self.cancel[run["run_id"]].is_set():
            raise InterruptedError("用户停止；在途请求返回后停止，不发新请求")
        if utc(self.clock()) >= utc(run["created_at"])+timedelta(minutes=RUN_TIMEOUT_MINUTES):
            raise TimeoutError(f"运行超过 {RUN_TIMEOUT_MINUTES} 分钟实验预算")
        if run["prediction_cutoff"] and utc(self.clock()) >= utc(run["prediction_cutoff"]):
            raise TimeoutError("已到赛前截止时间；迟到结果仅归档，不更新预测")

    def _request(self, run, tools, action, seeds, purpose, *, allow_route_retry=True):
        self._check(run)
        if run["counts"]["source"] >= MAX_SOURCE_REQUESTS:
            raise TimeoutError("来源请求预算已用完")
        prepared = tools.prepare(action, {"task": {"seed_urls": seeds}, "documents": run["documents"]})
        attempt = {"request_id": uuid4().hex, "purpose": purpose, "action": action,
                   "requested_at": self.clock(), "status": "PENDING", "response": None}
        if action["tool"] == "search_web":
            attempt["transport_route"] = getattr(tools, "search_route", getattr(tools, "default_route", "environment"))
        run["counts"]["source"] += 1
        run["attempts"].append(attempt)
        run["pending"] = attempt["request_id"]
        self._save(run)
        returned = tools.execute(prepared, run_id=run["run_id"], request_id=attempt["request_id"])
        attempt.update(status=returned["document"]["status"], response=returned, handoff_at=self.clock())
        run["pending"] = None
        run["documents"].append(returned["document"])
        self._save(run)
        self._check(run)
        # One separately budgeted, archived direct-route retry for failed RSS
        # transport. Never retry empty/filtered successful responses or bodies.
        if (allow_route_retry and action["tool"] == "search_web"
                and returned["document"]["status"] == "FAILED"
                and hasattr(tools, "default_route")
                and attempt["transport_route"] != "direct"
                and run["counts"]["source"] < MAX_SOURCE_REQUESTS):
            tools.search_route = "direct"
            attempt["retry_reason"] = "search_transport_failed_try_direct_once"
            self._save(run)
            return self._request(run, tools, action, seeds, "search_transport_retry", allow_route_retry=False)
        return returned["document"]

    def _row(self, run):
        return {"physical_match_id": run["fixture"]["physical_match_id"], "decision_at": self.clock(),
                "prior": run["prior"], "used_input_ids": run["used_input_ids"], "evidence_events": run["evidence_events"]}

    def _decision(self, run, candidates):
        visible = [c for c in candidates if c["action_id"] not in {d["action_id"] for d in run["decisions"] if d["action_id"]}]
        snapshot = observed_snapshot(self._row(run))
        menu = []
        stamp = self.clock()
        for c in visible:
            published = c.get("source_published_at")
            try:
                age = (utc(stamp)-utc(published)).total_seconds()/3600 if published else 168.
                if age < 0:
                    continue
            except (ValueError, TypeError):
                age = 168.
            public = {"schema": "historical-prequery-visible-features-v1", "probabilities": run["current"],
                      "acquired_documents": len(run["steps"]), "visible_candidates": len(visible),
                      "article_age_hours": age, "title": c["title"]}
            menu.append({"action_id": c["action_id"], "query": g_features.build_query(public, snapshot)})
        body_remaining = run["max_reads"]-run["counts"]["body_attempts"]
        semantic_remaining = MAX_PROVIDER_CALLS-run["counts"].get("provider",0)
        if hasattr(self.backend,"review_claims"):
            # A nonempty extraction must still have one request available for
            # its source-entailment review before f can consume any claim.
            semantic_remaining //= 2
        remaining = max(0,min(body_remaining,semantic_remaining))
        result = g_model.choose(self.g, menu, remaining, run["cost_loss_units"]) if menu and remaining else {
            "action_id": None, "reason": ("semantic_budget_exhausted" if semantic_remaining<=0 else
                "budget_exhausted" if body_remaining<=0 else "no_feasible_candidate"), "candidates": []}
        by_id = {c["action_id"]: c for c in visible}
        for c in result["candidates"]:
            source = by_id[c["action_id"]]
            timing = source.get("publication_time_scope") or candidate_time_scope(source, stamp)
            c.update(title=source["title"], url=source["url"],
                     source_publication_day=timing.get("calendar_day"))
        if result["action_id"] is not None and result["candidates"]:
            top = max(c["two_step_net_value"] for c in result["candidates"])
            tied = [c for c in result["candidates"] if abs(c["two_step_net_value"]-top) < 1e-12]
            result["top_score_tie_count"] = len(tied)
            if len(tied)>1 and any(c.get("source_publication_day") for c in tied):
                latest = max(c.get("source_publication_day") or "" for c in tied)
                best = min((c for c in tied if c.get("source_publication_day")==latest),key=lambda c:c["action_id"])
                result.update(action_id=best["action_id"],
                    tie_break="observed_publication_day_desc_then_original_action_id",
                    original_model_tie_choice=result["action_id"])
                # This changes only exact score ties, never an estimated gain or a STOP.
        result.update(decision_at=stamp, prequery_menu=menu)
        scores = [c["two_step_net_value"] for c in result["candidates"] if "two_step_net_value" in c]
        result["ranking_diagnostic"] = ("no_score_separation" if len(scores) > 1 and max(scores)-min(scores) < 1e-12
                                        else "scores_available" if scores else "not_scored")
        run["decisions"].append(result)
        self._save(run)
        return result

    def _extract_body(self, run, public, folder):
        """One ordinary call, with at most one explicitly counted validation repair."""
        from jsonschema import ValidationError
        from football_agent.deepseek_provider import DeepSeekProviderError
        # Pure local input validation happens before reserving/counting any HTTP
        # request, and cannot trigger a provider repair or reuse an old receipt.
        provider_input(public, protocol=Protocol)
        calls = [("initial", self.backend.semantics)]
        if hasattr(self.backend, "repair_semantics"):
            calls.append(("validation_repair", self.backend.repair_semantics))
        for index, (kind, execute) in enumerate(calls):
            self._check(run)
            if run["counts"]["provider"] >= MAX_PROVIDER_CALLS:
                raise TimeoutError("语义请求预算已用完；未生成新的 after")
            current_time = utc(self.clock())
            remaining = min((utc(run["prediction_cutoff"])-current_time).total_seconds(),
                (utc(run["created_at"])+timedelta(minutes=RUN_TIMEOUT_MINUTES)-current_time).total_seconds())
            if remaining < 1:
                raise TimeoutError("剩余运行时间不足一次语义请求；未生成新的 after")
            run["counts"]["provider"] += 1
            attempt = {"input_id": public["input_id"], "kind": kind,
                "action_id": run.get("active_body", {}).get("action_id"),
                "document_id": run.get("active_body", {}).get("document_id"),
                "requested_at": self.clock(), "status": "PENDING"}
            run.setdefault("semantic_attempts", []).append(attempt)
            run["pending"] = "semantic-" + public["input_id"]
            run["stage"] = ("正文已取得：正在抽取陈述，f 尚未执行" if kind == "initial" else
                            "正文已取得：正在修复语义协议，f 尚未执行")
            self._save(run)
            try:
                record = execute(folder / "provider", public, min(120, remaining))
                attempt.update(status="OK", completed_at=self.clock(),
                               provider_artifact=getattr(self.backend, "last_semantic_attempt", None))
                save(folder / "semantics" / (public["input_id"]+".json"), record)
                self._save(run)
                return record
            except Exception as exc:
                attempt.update(status="FAILED", completed_at=self.clock(), error_class=type(exc).__name__,
                               provider_artifact=getattr(self.backend, "last_semantic_attempt", None))
                if isinstance(exc, ValidationError):
                    attempt.update(validation_rule=exc.validator, validation_path=list(exc.absolute_path))
                self._save(run)
                repairable = isinstance(exc, (ValueError, ValidationError)) or (
                    isinstance(exc, DeepSeekProviderError) and str(exc) in {
                        "DEEPSEEK_INVALID_JSON", "DEEPSEEK_EMPTY_CONTENT", "DEEPSEEK_JSON_OBJECT_REQUIRED",
                        "DEEPSEEK_DUPLICATE_JSON_KEY", "DEEPSEEK_NONFINITE_JSON"})
                if index or not repairable or len(calls) < 2 or run["counts"]["provider"] >= MAX_PROVIDER_CALLS:
                    raise
                self._check(run)
        raise RuntimeError("No semantic extraction returned")

    def _review_body(self, run, record, folder):
        """Source entailment admission, separate from exact-quote/schema checks.

        The first extraction stays immutable inside this record. Review only
        selects original indices; it cannot rewrite a claim, quote or forecast.
        """
        public,claims=record['input'],record['response']['claims']
        if not claims:
            review,binding=live_claim_review.skip_review(public,claims)
            admission={'review':review,'binding':binding,'receipt':None}
        elif not hasattr(self.backend,'review_claims'):
            # Archived-provider replays and dependency-injected tests preserve
            # their original interpretation. LiveBackend always reviews.
            admission={'review':{'retained_indices':list(range(len(claims))),'rejected':[],
                'status':'archived_or_test_backend_without_new_semantic_review'}}
        else:
            live_claim_review.provider_input(public,claims)
            self._check(run)
            remaining=min((utc(run['prediction_cutoff'])-utc(self.clock())).total_seconds(),
                (utc(run['created_at'])+timedelta(minutes=RUN_TIMEOUT_MINUTES)-utc(self.clock())).total_seconds())
            if run['counts']['provider']>=MAX_PROVIDER_CALLS or remaining<1:
                raise ValueError('Live claim review lacks request/time budget; no claims admitted')
            run['counts']['provider']+=1
            attempt={'input_id':public['input_id'],'kind':'claim_review','requested_at':self.clock(),'status':'PENDING',
                'action_id':run.get('active_body',{}).get('action_id'),
                'document_id':run.get('active_body',{}).get('document_id')}
            run['semantic_attempts'].append(attempt);run['pending']='claim-review-'+public['input_id']
            run['stage']='正文抽取完成：正在复核陈述，复核通过后执行 f'
            self._save(run)
            try:
                admission=self.backend.review_claims(folder/'provider',public,claims,min(MAX_REVIEW_SECONDS,remaining))
                attempt.update(status='OK',completed_at=self.clock(),provider_artifact=getattr(self.backend,'last_semantic_attempt',None))
            except Exception as exc:
                from jsonschema import ValidationError
                attempt.update(status='FAILED',completed_at=self.clock(),error_class=type(exc).__name__,
                    provider_artifact=getattr(self.backend,'last_semantic_attempt',None))
                if isinstance(exc,ValidationError):
                    attempt.update(validation_rule=exc.validator,validation_path=list(exc.absolute_path))
                self._save(run);raise
            self._save(run)
        # Every retained object is copied unchanged from the already valid
        # extraction. An empty supported set is a valid consumed neutral body.
        record={**record,'claim_admission':admission}
        save(folder/'semantics'/(public['input_id']+'.json'),record)
        return record

    def _work(self, run):
        from .live_sources import profile as source_profile, discovery_queries
        folder = self._folder(run["run_id"])
        try:
            self._check(run)
            manifest = self.backend.initialize(run["match"], folder / "setup")
            run["initialization"] = manifest
            run["prediction_cutoff"] = manifest["market_initialization"]["prediction_cutoff"]
            self._check(run)
            run["stage"] = "核验目标比赛身份与新鲜报价"
            self._save(run)
            binding = self.backend.identity(manifest, folder / "identity", run["match"])
            run["identity"] = binding
            canonical = binding["canonical"]
            run["fixture"] = {"physical_match_id": canonical["source_event_id"],
                "home_name": canonical["home_name"], "away_name": canonical["away_name"],
                "kickoff": canonical["kickoff"], "home_aliases": binding["aliases"]["home"],
                "away_aliases": binding["aliases"]["away"],
                "competition_name": canonical.get("competition_name") or run["match"].get("competition_key"),
                "season": canonical.get("season")}
            run["prior"] = manifest["market_initialization"]["probabilities"]
            self._check(run)
            run["current"] = f_model.predict(self.f, self._row(run)).tolist()
            run["before"] = list(run["current"])
            run["forecast_at"] = self.clock()
            run.update(state="RUNNING", stage="读取球队目录，建立查询前候选菜单")
            profile = source_profile(binding)
            entries = profile["seed_urls"][:2]
            run["source_profile"] = profile
            tools = self.backend.tools(folder / "sources", entries, profile["allowed_hosts"])
            if hasattr(tools, "configuration"):
                run["source_transport"] = tools.configuration()
            candidates, seen = [], set()
            directory_actions = [{"tool": "read_url", "url": entry} for entry in entries]
            search_queries = discovery_queries(canonical)
            run["discovery_queries"] = search_queries
            self._save(run)
            for action in directory_actions:
                doc = self._request(run, tools, action, entries,
                    "directory_discovery" if action["tool"] == "read_url" else "search_discovery")
                self._record_discovery_diagnostics(run, action, doc)
                for link in doc.get("links", []):
                    url, title = link.get("url", ""), link.get("title", "").strip()
                    if action["tool"] == "read_url":
                        reason = _directory_candidate_reason(url, title, run["fixture"], entries, seen)
                    else:
                        from .live_sources import article_link_reason
                        reason = article_link_reason(url, title, entries, seen, search=True)
                    candidate = {**link, "action_id": "article-"+sha256(url.encode()).hexdigest()[:24],
                        "observed_at": self.clock(), "parent_document_id": doc["document_id"],
                        "admission_reason": reason or "observed_article_link", "age_missing": not bool(link.get("source_published_at"))}
                    candidate["publication_time_scope"] = candidate_time_scope(link, self.clock())
                    if reason is None and candidate["publication_time_scope"]["status"] in {"stale","future"}:
                        reason = "outside_live_publication_window"
                        candidate["admission_reason"] = reason
                    run["menus"].append(candidate)
                    if reason is None:
                        seen.add(url)
                        candidates.append(candidate)
                self._save(run)
            # Directory pages are often navigation shells. Before declaring
            # that no article exists, use bounded query diversity: all three
            # intents (up to five) when no candidate was exposed, one extra when only one
            # candidate was found. g still decides which admitted body to read.
            search_limit = 5 if not candidates else 1 if len(candidates) < 2 else 0
            for query in search_queries[:search_limit]:
                if run["counts"]["source"] >= MAX_SOURCE_REQUESTS-(1 if candidates else 0):
                    break
                action = {"tool": "search_web", "query": query}
                doc = self._request(run, tools, action, entries, "search_discovery")
                _review_pending_host_links(run, tools, doc)
                if hasattr(tools, "configuration"):
                    run["source_transport"] = tools.configuration()
                self._record_discovery_diagnostics(run, action, doc)
                for link in doc.get("links", []):
                    url, title = link.get("url", ""), link.get("title", "").strip()
                    from .live_sources import article_link_reason
                    reason = article_link_reason(url, title, entries, seen, search=True)
                    if reason is None:
                        from .live_sources import search_team_binding
                        if not search_team_binding(title, url, run["fixture"], entries):
                            reason = "search_team_binding_missing"
                    candidate = {**link, "action_id": "article-"+sha256(url.encode()).hexdigest()[:24],
                        "observed_at": self.clock(), "parent_document_id": doc["document_id"],
                        "admission_reason": reason or "observed_article_link", "age_missing": not bool(link.get("source_published_at"))}
                    candidate["publication_time_scope"] = candidate_time_scope(link, self.clock())
                    if reason is None and candidate["publication_time_scope"]["status"] in {"stale","future"}:
                        reason = "outside_live_publication_window"
                        candidate["admission_reason"] = reason
                    run["menus"].append(candidate)
                    if reason is None:
                        seen.add(url)
                        candidates.append(candidate)
                self._save(run)
                if len(candidates) >= 5:
                    break
            if not candidates:
                discovery = [a for a in run["attempts"] if a["purpose"] in
                             {"directory_discovery", "search_discovery", "search_transport_retry"}]
                failed = sum(a["status"] in {"FAILED", "INVALID_CONTENT"} for a in discovery)
                reason = ("discovery_sources_failed" if discovery and failed == len(discovery) else
                          "discovery_incomplete_no_candidates" if failed else "discovery_no_candidates")
                run.update(state="FAILED" if reason == "discovery_sources_failed" else "STOPPED",
                           reason=reason, stage="来源发现结束；未形成可供 g 评分的候选")
                run["discovery_summary"] = {"attempts": len(discovery), "failed": failed, "admitted": 0,
                                             "g_scored": False}
                self._save(run)
                return
            while True:
                self._check(run)
                run["stage"] = "g 计算当前信息价值与后续价值"
                decision = self._decision(run, candidates)
                if decision["action_id"] is None:
                    run.update(state="STOPPED", reason=decision["reason"], stage="自主分析结束")
                    break
                candidate = next(c for c in candidates if c["action_id"] == decision["action_id"])
                run["counts"]["body_attempts"] += 1
                run["stage"] = "g 选择正文："+candidate["title"]
                doc = self._request(run, tools, {"tool": "read_url", "url": candidate["url"]}, entries, "g_selected_body")
                structured = doc.get("structured_news", {})
                if doc["status"] != "OK" or structured.get("role") != "article" or not doc.get("text", "").strip():
                    run.setdefault("body_rejections", []).append({"action_id": candidate["action_id"],
                        "document_id": doc.get("document_id"), "status": doc.get("status"),
                        "role": structured.get("role"), "reason": "no_admitted_article_body", "at": self.clock()})
                    self._save(run)
                    continue
                public = public_input(run["fixture"], {"url": doc["url"], "title": candidate["title"], "text": doc["text"]}, "body")
                public = with_source_context(public, doc, candidate, run["fixture"])
                if public["input_id"] in run["used_input_ids"]:
                    continue
                self._check(run)
                run["active_body"] = {"action_id": candidate["action_id"],
                    "document_id": doc.get("document_id"), "input_id": public["input_id"],
                    "title": candidate["title"], "status": "awaiting_semantics"}
                run["stage"] = "正文已取得：准备语义抽取，f 尚未执行"
                try:
                    record = self._extract_body(run, public, folder)
                    record = self._review_body(run,record,folder)
                except Exception as exc:
                    from jsonschema import ValidationError
                    quote_or_pair_error = isinstance(exc, ValidationError) or (
                        isinstance(exc, ValueError) and str(exc).startswith((
                            "Claim must cite", "Source predicate/value", "Unknown source quote", "Unknown quote span",
                            "Unknown acquired body quote_id", "Live claim review")))
                    if not quote_or_pair_error:
                        raise
                    # A candidate-level validation failure does not invalidate
                    # the other acquired candidates, even before the first f
                    # update. Invalid input gets no after and no zero gain.
                    run.setdefault("body_rejections", []).append({"action_id": candidate["action_id"],
                        "document_id": doc.get("document_id"), "input_id": public["input_id"],
                        "raw_sha256": doc.get("raw_sha256"), "reason": "semantic_validation_failed",
                        "error_class": type(exc).__name__, "at": self.clock(), "prediction_after": None,
                        "gain": None, "f_executed": False})
                    run["pending"] = None
                    run["active_body"]["status"] = "semantic_validation_failed"
                    self._save(run)
                    continue
                self._check(run)
                # Commit input consumption, events and after together only after a valid f call.
                trial = deepcopy(run)
                trial["used_input_ids"].append(public["input_id"])
                admitted_indices=record['claim_admission']['review']['retained_indices']
                admitted_claims=[record['response']['claims'][i] for i in admitted_indices]
                for claim in admitted_claims:
                    event = {"physical_match_id": run["fixture"]["physical_match_id"], "input_id": public["input_id"],
                        "entity": claim["entity"], "document_url": public["url"], "source_host": urlsplit(public["url"]).hostname,
                        "source_published_at": public["source_context"]["source_published_at"], "historical_available_at": None,
                        "claim": claim, "evidence": rich.evidence_tokens(claim, "body", run["fixture"],
                            concept=claim.get("concept"))}
                    event["event_id"] = sha256(json_bytes(event)).hexdigest()
                    trial["evidence_events"].append(event)
                before = list(run["current"])
                run["pending"] = None
                run["stage"] = "陈述复核完成：正在执行固定 f，尚未提交本次 after"
                run["active_body"]["status"] = "f_running"
                self._save(run)
                after = f_model.predict(self.f, self._row(trial)).tolist()
                consumption = consumption_audit(self.f, self._row(run), self._row(trial),
                    public["input_id"], doc["text"], admitted_claims, before, after)
                self._check(run)
                run.update(current=after, forecast_at=self.clock(), evidence_events=trial["evidence_events"],
                           used_input_ids=trial["used_input_ids"], pending=None)
                run["active_body"]["status"] = "f_committed"
                run["steps"].append({"step": len(run["steps"])+1, "action_id": candidate["action_id"],
                    "title": candidate["title"], "url": doc["url"], "body": doc["text"],
                    "raw_sha256": doc["raw_sha256"], "input_id": public["input_id"],
                    "claims": admitted_claims, "extracted_claims_count":len(record['response']['claims']),
                    "claim_review":record['claim_admission']['review'],
                    "rejected_claims":[{**item,'original_claim':record['response']['claims'][item['index']]}
                        for item in record['claim_admission']['review'].get('rejected',[]) if 'index' in item],
                    "prediction_before": before, "prediction_after": after,
                    "f_consumption": consumption,
                    "evidence_highlights": _evidence_highlights(admitted_claims, doc["text"]),
                    "executed_at": run["forecast_at"], "f_executed": True, "loss_reduction": None,
                    "semantic_record": "semantics/"+public["input_id"]+".json"})
                self._save(run)
        except InterruptedError as exc:
            run.update(state="CANCELLED", reason=str(exc), stage="已停止")
        except TimeoutError as exc:
            run.update(state="STOPPED", reason=str(exc), stage="时间或请求预算截止")
        except Exception as exc:
            # Provider errors are sanitized at their transport seam. No raw response/body is an error message.
            reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            run.update(state="FAILED", reason=reason[:500], stage="本轮失败，已保留此前真实结果")
        finally:
            self._save(run)

    def _record_discovery_diagnostics(self, run, action, doc):
        structured = doc.get("structured_news") or {}
        retrieval = structured.get("retrieval_diagnostics") or {}
        run.setdefault("discovery_diagnostics", []).append({
            "document_id": doc.get("document_id"),
            "tool": action.get("tool"),
            "query": action.get("query"),
            "url": action.get("url") or doc.get("url"),
            "status": doc.get("status"),
            "filter_status": doc.get("filter_status"),
            "provider_status": retrieval.get("provider_status"),
            "provider_result_count": retrieval.get("provider_result_count"),
            "filtered_result_count": retrieval.get("filtered_result_count"),
            "eligible_result_count": retrieval.get("eligible_result_count", len(doc.get("links", []))),
            "pending_host_review_count": retrieval.get("pending_host_review_count", len(structured.get("pending_host_links") or [])),
            "menu_link_count": len(doc.get("links", [])),
        })

