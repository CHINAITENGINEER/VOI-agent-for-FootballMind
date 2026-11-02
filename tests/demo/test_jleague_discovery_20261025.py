"""J 联赛发现链路根因回归测试（2026-10-25 RCA）。

对应故障：Cerezo Osaka vs JEF United Ichihara-Chiba（赛事"日职"）
- 运行 lg-0ea55264bb37422fb22a4fcc9513330a / lg-1ef82118f0464440a7a3a0348ff1486d
  均以 discovery_no_candidates 停止：5 次搜索、0 篇候选、body_attempts=0、g_scored=false。

测试覆盖四件事：
1. 该比赛必须能通过"官方俱乐部目录"拿到 seed（当前为空 → 修复目标）；
2. 该比赛的查询必须包含日语表述与 J 联赛官方站点（当前没有 → 修复目标）；
3. 真实发生的 40 条无关结果必须继续被现有过滤链拒绝（防止将来放宽过滤）；
4. 俱乐部官网的真实日文文章必须能被现有审核链接受（证明目录通道对日文有效）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo.live_g import _review_pending_host  # noqa: E402
from demo.live_sources import article_link_reason, discovery_queries, profile, search_team_binding  # noqa: E402

FIXTURE = {"home_name": "Cerezo Osaka", "away_name": "JEF United Ichihara-Chiba",
           "home_aliases": ["Cerezo Osaka"], "away_aliases": ["JEF United Ichihara-Chiba"]}
CANONICAL = {**FIXTURE, "kickoff": "2026-10-25T06:00:00+00:00", "competition_name": "日职"}
BINDING = {"status": "SOURCE_CONFIRMED", "canonical": CANONICAL}

# 来自运行归档的**真实**无关结果（10+10+10+10 条中各取代表；完整表见 RCA 目录）
ARCHIVED_JUNK = [
    {"url": "https://www.bilibili.com/c/music/", "title": "音乐-哔哩哔哩 (゜-゜)つロ 干杯~-bilibili"},
    {"url": "https://game.bilibili.com/kf/", "title": "游戏中心客服丨你的幻想世界"},
    {"url": "https://www.zhihu.com/question/22675146", "title": "陈冠希为什么能和众多明星发生关系并拍照? - 知乎"},
    {"url": "https://jingyan.baidu.com/article/d169e186214ea1026711d87e.html", "title": "抖音电脑版如何清理缓存 - 百度经验"},
    {"url": "https://zhidao.baidu.com/question/90342114.html", "title": "添加或删除程序里有个程序删不掉！_百度知道"},
    {"url": "https://www.zhihu.com/question/444910062", "title": "下载安装Steam 一直更新巨慢怎么办? - 知乎"},
]


def test_jleague_fixture_resolves_to_official_directory_seeds():
    config = profile(BINDING)
    assert config["unknown_teams"] == [], "日职两支球队都必须登记，否则只剩搜索通道"
    assert config["seed_urls"], "必须有官方目录 seed，否则目录发现通道不可用"
    assert config["discovery_mode"] == "club_directories_with_search_fallback"
    hosts = config["allowed_hosts"]
    assert "www.cerezo.jp" in hosts and "jefunited.co.jp" in hosts


def test_jleague_queries_cover_japanese_and_official_site():
    queries = discovery_queries(CANONICAL)
    assert len(queries) == len(set(queries)) == 5
    assert any("セレッソ大阪" in q for q in queries), "必须有一条日语查询（英文队名在日文来源上召回极差）"
    assert any("jleague.jp" in q for q in queries), "需要用 J 联赛官方站点限定查询"
    assert not any("kicker.de" in q or "bundesliga.com" in q for q in queries), "不得回退到德甲站点"


def test_archived_unrelated_results_stay_rejected():
    """真实跑偏结果必须继续被拒绝：不得为了让本轮成功而放宽过滤。"""
    entries = profile(BINDING)["seed_urls"]
    for item in ARCHIVED_JUNK:
        review = _review_pending_host(item, FIXTURE, profile(BINDING)["allowed_hosts"], set(), 0)
        assert review["status"] == "rejected", item["url"]
        assert review["reason"] in {"navigation_or_non_article_path", "non_sports_article_signal",
                                    "team_binding_missing", "search_provider_host"}
        shape = article_link_reason(item["url"], item["title"], entries, set(), search=True)
        bound = search_team_binding(item["title"], item["url"], FIXTURE, entries)
        assert shape is not None or not bound, f"无关结果不得被接受：{item['url']}"


def test_official_japanese_article_is_admissible():
    """俱乐部官网真实日文文章的 URL/标题必须能通过现有审核与绑定规则。"""
    entries = profile(BINDING)["seed_urls"]
    url = "https://www.cerezo.jp/news/2026-1024-1900/"
    title = "2026.10.24 10/25 千葉戦 Match Preview"
    review = _review_pending_host({"url": url, "title": title}, FIXTURE,
                                  profile(BINDING)["allowed_hosts"], set(), 0)
    assert review["status"] == "already_allowed" or review["status"] == "approved"
    assert article_link_reason(url, title, entries, set(), search=True) is None
    assert search_team_binding(title, url, FIXTURE, entries)


@pytest.mark.parametrize("other", [{"competition_name": "USL League One", "home_name": "AV Alta FC",
                                    "away_name": "Corpus Christi FC"}])
def test_other_competitions_keep_their_query_behaviour(other):
    queries = discovery_queries({**CANONICAL, **other})
    assert not any("セレッソ大阪" in q for q in queries)
    assert not any("jleague.jp" in q for q in queries)
