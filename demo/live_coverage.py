"""Catalog and identity for multi-competition online experiments.

Source identity is deliberately separate from an independent official check.
These aliases improve discovery; they do not verify a competition or a fixture.
"""
from collections import Counter
from datetime import timedelta
from .live_sources import TEAMS
from football_research.data_adapters.espn_scoreboard import competition_group

PL_NAMES = {"英超", "英格兰超级联赛", "Premier League", "PL"}
NAME_ALIASES = {
    **{zh: en for zh, en in TEAMS.values()},
    "蒙扎": "Monza", "卡利亚里": "Cagliari", "帕尔马": "Parma", "都灵": "Torino",
    "代格福什": "Degerfors", "卡尔马": "Kalmar", "罗德兹": "Rodez",
    "蒙彼利埃": "Montpellier", "赫塔费": "Getafe", "巴列卡诺": "Rayo Vallecano",
    "布拉加": "Braga", "吉维森特": "Gil Vicente", "科林蒂安": "Corinthians",
    "维多利亚": "Vitoria", "费内巴切": "Fenerbahce", "斯拉维亚": "Slavia Prague",
    "巴黎圣曼": "Paris Saint-Germain", "巴萨": "Barcelona", "多特蒙德": "Borussia Dortmund",
    "斯图加特": "Stuttgart", "马竞": "Atletico Madrid", "比利亚雷": "Villarreal",
    "波尔图": "Porto", "埃因霍温": "PSV Eindhoven", "罗马": "Roma",
    "那不勒斯": "Napoli", "博德闪耀": "Bodo Glimt", "拜仁": "Bayern Munich",
    "皇马": "Real Madrid", "贝蒂斯": "Real Betis", "费耶诺德": "Feyenoord",
    "尤文图斯": "Juventus", "AC米兰": "AC Milan", "皇家社会": "Real Sociedad",
    "马赛": "Marseille", "里昂": "Lyon", "勒沃库森": "Bayer Leverkusen",
    "摩纳哥": "Monaco", "亚特兰大": "Atalanta", "阿贾克斯": "Ajax",
    "凯尔特人": "Celtic", "桑德兰": "Sunderland", "弗拉门戈": "Flamengo",
    "奥格斯堡": "Augsburg", "柏林联合": "Union Berlin",
}


def competition_key(name):
    from football_research.data_adapters.kleague_official import ALIASES
    return "Premier League" if name in PL_NAMES else ALIASES.get(name, name)


def catalog_match(row):
    quality = row.get("inventory_quality", {})
    ids = [row.get(s + "_team_source_id") for s in ("home", "away")]
    names = [row.get(s + "_team") for s in ("home", "away")]
    reason = None
    if (not all(ids) or ids[0] == ids[1] or not all(names)
            or not row.get("source_match_id") or not row.get("competition_key")):
        reason = "来源比赛或球队身份不完整"
    elif quality.get("schedule_conflict_observed"):
        reason = "来源开球时间存在冲突，需要重新获取一致赛程"
    elif row.get("status") != "scheduled":
        reason = "来源尚未确认该场处于未开赛状态"
    supported = reason is None
    canonical_competition = competition_key(row["competition_key"])
    return {**row, "match_id": row["fixture_id"], "kickoff": row["kickoff_at"],
        "home": NAME_ALIASES.get(names[0], names[0]), "away": NAME_ALIASES.get(names[1], names[1]),
        "competition_key": canonical_competition,
        "competition_group": competition_group(canonical_competition),
        "supported": supported, "reason": reason,
        "quote_observed": quality.get("complete_1x2_observed", False),
        "identity_status": "source_observed_recheck_on_start",
        "model_scope": "premier_league_research" if canonical_competition == "Premier League"
                       else "cross_competition_experimental"}


def availability(match, at, *, minimum_lead_minutes=30):
    from .live_g import utc
    if match.get("status") in {"live", "finished", "postponed", "cancelled", "suspended"}:
        return False, {"live": "比赛进行中", "finished": "比赛已结束", "postponed": "比赛延期",
                       "cancelled": "比赛取消", "suspended": "比赛中断"}[match["status"]]
    quote_at = match.get("quote_catalog_observed_at") or match.get("quote_observed_at")
    if quote_at and utc(at)-utc(quote_at) > timedelta(minutes=30):
        return False, "先验来源缓存超过 30 分钟，等待刷新；赛程仍可查看"
    if not match.get("supported"):
        return False, match.get("reason") or "来源身份尚未就绪"
    if utc(match["kickoff"]) <= utc(at) + timedelta(minutes=minimum_lead_minutes):
        return False, f"已开赛或距离开球不足 {minimum_lead_minutes} 分钟"
    # Old catalog compatibility: new inventories always declare this explicitly.
    if match.get("quote_observed") is False:
        return False, "目录尚无完整 90 分钟赔率，等待来源更新后刷新"
    return True, "可启动实验 · 启动时重新核验身份与赔率"


def summarize(matches):
    counts = Counter(m.get("competition_key", "未知赛事") for m in matches)
    ready = Counter(m.get("competition_key", "未知赛事") for m in matches if m["can_start"])
    return {"discovered": len(matches), "can_start": sum(ready.values()),
        "waiting_quote": sum(m.get("quote_observed") is False for m in matches),
        "competition_count": len(counts),
        "competitions": [{"name": k, "discovered": v, "can_start": ready[k]} for k, v in counts.items()],
        "blocking_reasons": [{"reason": reason, "matches": count}
            for reason, count in Counter(m.get("availability_reason") or m.get("reason") or "未核验"
                for m in matches if not m["can_start"]).most_common()],
        "mainstream_competitions": [{"name": k, "discovered": counts[k], "can_start": ready[k],
            "catalog_status": "observed" if counts[k] else "not_observed_in_current_window",
            "absence_meaning": None if counts[k] else "本目录未观察到比赛；不能据此断言没有赛程或没有分析能力"}
            for k in ("中超", "中甲", "Premier League", "西甲", "意甲", "德甲", "法甲",
                      "欧冠", "欧罗巴", "欧协联", "葡超", "荷甲", "土超", "苏超", "比甲",
                      "日职", "K联赛", "沙特联", "美职联", "巴甲", "阿甲", "墨超")],
        "schedule_scope": "北京时间今天起 7 天，ESPN 公开赛程与足彩来源合并；不声称覆盖全部赛事"}


def bind_source_identity(manifest, expected, at):
    """Compare a freshly acquired fixture to the exact selected catalog row.

    This cannot grant independent official verification. No name-only joins,
    invented season, or quote substitution from another match are permitted.
    """
    from .live_g import utc
    fixture = manifest["market_initialization"]["fixture"]
    if (fixture["match_id"] != expected["fixture_id"]
            or fixture["match_id"] != "spftll:match:" + expected["source_match_id"]
            or competition_key(fixture["competition_name"]) != competition_key(expected["competition_key"])
            or utc(fixture["kickoff"]) != utc(expected["kickoff"])
            or fixture.get("status") != "scheduled"
            or utc(at) >= utc(fixture["kickoff"])):
        raise ValueError("新鲜比赛详情与所选目录的身份、赛事或开球时间不一致，请刷新目录")
    aliases, names = {}, {}
    for side in ("home", "away"):
        source_name = expected[side + "_team"]
        if (fixture[side + "_team_id"] != "spftll:team:" + expected[side + "_team_source_id"]
                or fixture[side + "_team_name"] != source_name):
            raise ValueError("新鲜比赛详情与所选目录的主客队 ID 或名称不一致")
        names[side] = NAME_ALIASES.get(source_name, source_name)
        aliases[side] = list(dict.fromkeys([source_name, names[side]]))
    if fixture["home_team_id"] == fixture["away_team_id"]:
        raise ValueError("主客队来源身份重复")
    return {"schema": "footballmind-live-source-identity-v2", "status": "SOURCE_CONFIRMED",
        "identity_level": "source_id_and_fresh_quote", "independently_confirmed": False,
        "canonical": {"source_event_id": fixture["match_id"], "home_name": names["home"],
            "away_name": names["away"], "kickoff": fixture["kickoff"],
            "competition_name": fixture["competition_name"], "season": None},
        "aliases": aliases, "verified_at": at, "source_event_id_is_local_namespace": False,
        "basis": "exact_source_match_team_ids_names_competition_kickoff_in_fresh_initialization",
        "catalog_observation_id": expected.get("registration_observation_id"),
        "initialization_inputs": fixture.get("input_manifest"),
        "limitation": "同一数据源的身份与报价校验；尚无跨源官方赛程确认"}
