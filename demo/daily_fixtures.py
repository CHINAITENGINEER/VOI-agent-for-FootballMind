"""Join the daily display schedule to existing analysable source fixtures.

Never route an ESPN ID to the spftll initializer or copy odds by approximate
names. A unique home+away+competition+exact kickoff match is required.
"""
from copy import deepcopy
from collections import defaultdict
import unicodedata

from .live_coverage import NAME_ALIASES, competition_key
from football_research.data_adapters.espn_scoreboard import competition_group
from football_research.data_adapters.espn_scoreboard import utc, window

TEAM_EQUIVALENTS = {
    "Tottenham": "Tottenham Hotspur", "Coventry": "Coventry City",
    "Manchester United": "Manchester United", "Man Utd": "Manchester United",
    "Wolverhampton": "Wolverhampton Wanderers", "Wolves": "Wolverhampton Wanderers",
    "Ipswich": "Ipswich Town", "Hull": "Hull City", "Leeds United": "Leeds",
    "Newcastle United": "Newcastle", "West Ham United": "West Ham",
    "Nottingham": "Nottingham Forest", "Leicester City": "Leicester",
    "Brighton & Hove Albion": "Brighton", "AFC Bournemouth": "Bournemouth",
    "Atlético Madrid": "Atletico Madrid", "Villarreal CF": "Villarreal",
    "Internazionale": "Inter Milan", "Internazionale Milan": "Inter Milan",
    "Paris Saint Germain": "Paris Saint-Germain", "Sporting CP": "Sporting Lisbon",
    "Degerfors IF": "Degerfors", "Kalmar FF": "Kalmar", "Rodez Aveyron": "Rodez",
    "VfB Stuttgart": "Stuttgart", "FC Porto": "Porto", "AS Roma": "Roma",
    "Ajax Amsterdam": "Ajax", "AS Monaco": "Monaco", "Feyenoord Rotterdam": "Feyenoord",
}

# Explicit display-source aliases reviewed against the two observed catalogs.
# These do not change the model, news sources, or the initializer's source IDs.
CATALOG_NAMES = {
    "萨巴赫": "Sabah FK", "伍尔弗": "Wolverhampton Wanderers", "朴次茅斯": "Portsmouth",
    "雅典AEK": "AEK Athens", "布拉迪斯": "Slovan Bratislava", "科莫": "Como",
    "里尔": "Lille", "加拉塔萨": "Galatasaray", "莱万特": "Levante",
    "毕尔巴鄂": "Athletic Club", "莱红牛": "RB Leipzig", "布鲁日": "Club Brugge",
    "朗斯": "Lens", "维京": "Viking FK", "里斯本": "Sporting Lisbon", "LASK林茨": "LASK Linz",
    "帕梅拉斯": "Palmeiras", "弗鲁米嫩": "Fluminense", "雷恩": "Stade Rennais",
    "克里特": "OFI Crete", "波兹南": "Lech Poznan", "格风暴": "SK Sturm Graz",
    "萨迪纳摩": "Dinamo Zagreb", "奈梅亨": "NEC Nijmegen", "弗赖堡": "SC Freiburg",
    "霍芬海姆": "TSG Hoffenheim", "利勒斯特": "Lillestrom", "塞尔塔": "Celta Vigo",
    "贝西克塔": "Besiktas", "拉普大学": "Estudiantes de La Plata",
    "东京FC": "FC Tokyo", "柏太阳神": "Kashiwa Reysol", "川崎前锋": "Kawasaki Frontale",
    "神户胜利": "Vissel Kobe", "大宫松鼠": "Omiya Ardija", "富山胜利": "Kataller Toyama",
}


def team_key(name):
    name = CATALOG_NAMES.get(name, NAME_ALIASES.get(name, name))
    name = TEAM_EQUIVALENTS.get(name, name)
    return "".join(c for c in unicodedata.normalize("NFKD", name.casefold()) if c.isalnum())


def key(match):
    return (competition_key(match["competition_key"]), utc(match["kickoff"]),
            team_key(match["home"]), team_key(match["away"]))


def merge_schedule(spftll, espn, at):
    start, end, _ = window(at)
    primary = [deepcopy(m) for m in spftll if start <= utc(m["kickoff"]) < end]
    secondary = [deepcopy(m) for m in espn if m.get("source", "espn") == "espn" and start <= utc(m["kickoff"]) < end]
    supplementary = [deepcopy(m) for m in espn if m.get("source", "espn") != "espn" and start <= utc(m["kickoff"]) < end]
    left, right = defaultdict(list), defaultdict(list)
    for m in primary:
        left[key(m)].append(m)
        m.update(source="spftll", sources=["spftll"], analysis_source="spftll")
    for m in secondary:
        right[key(m)].append(m)
    joined, consumed, ambiguous = 0, set(), 0
    for identity, a in left.items():
        b = right.get(identity, [])
        if len(a) == len(b) == 1:
            p, s = a[0], b[0]
            primary_quote = p.get("quote_observed") is True
            secondary_quote = s.get("quote_observed") is True and not s.get("source_stale")
            p.update(sources=["spftll", "espn"], espn_match_id=s["match_id"],
                     source_url=s.get("source_url"), schedule_observed_at=s["observed_at"],
                     espn_observation=s["raw_observation"], source_competition_name=s.get("source_competition_name"),
                     quote_observed=(primary_quote or secondary_quote),
                     quote_snapshot=(s.get("quote_snapshot") if secondary_quote else None) or p.get("quote_snapshot"),
                     quote_observed_at=(s.get("quote_observed_at") if secondary_quote else None) or p.get("quote_observed_at"),
                     cross_source_match_basis="unique_exact_competition_home_away_kickoff_explicit_aliases")
            p["spftll_quote_observed"] = primary_quote
            # The exact secondary identity stays intact for initialization;
            # odds are never attached to an unrelated primary source ID.
            p["espn_initialization_fixture"] = deepcopy(s)
            p["initialization_source"] = "espn" if not primary_quote and secondary_quote else "spftll"
            if p["initialization_source"] == "espn":
                p["quote_catalog_observed_at"] = s.get("quote_observed_at")
            if not s.get("source_stale"):
                p.update(status=s["status"], status_detail=s.get("status_detail"),
                         home_score=s.get("home_score"), away_score=s.get("away_score"),
                         time_confirmed=s.get("time_confirmed"))
                if s["status"] != "scheduled" or s.get("time_confirmed") is False:
                    p.update(supported=False, reason="赛程源显示已开赛、已结束、延期或时间待定，不能启动赛前分析")
            consumed.add(s["match_id"])
            joined += 1
        elif b:
            ambiguous += 1
    result = primary + [s for s in secondary if s["match_id"] not in consumed]
    # Supplementary schedules have their own IDs and no market prior. Never
    # send those IDs into either the ESPN or spftll initializer by implication.
    known_ids = {m["match_id"] for m in result}
    for match in supplementary:
        if match["match_id"] not in known_ids:
            result.append(match);known_ids.add(match["match_id"])
    # Display translations are aliases only, never evidence of entity matching.
    display = {team_key(en): zh for zh, en in {**NAME_ALIASES, **CATALOG_NAMES}.items()}
    for match in result:
        match["competition_key"] = competition_key(match.get("competition_key"))
        match["competition_group"] = competition_group(match.get("competition_key"))
        if match.get("source") == "espn":
            for side in ("home", "away"):
                match[side+"_team"] = display.get(team_key(match[side]), match[side])
        match["display_date"] = utc(match["kickoff"]).astimezone(start.tzinfo).date().isoformat()
    return sorted(result, key=lambda m: (utc(m["kickoff"]), m["competition_key"], m["match_id"])), {
        "spftll_rows": len(primary), "espn_rows": len(secondary), "matched_rows": joined,
        "ambiguous_groups_not_merged": ambiguous,
        "espn_schedule_only": len(secondary)-len(consumed),
        "supplemental_schedule_rows": len(supplementary),
        "window_start": start.isoformat(), "window_end_exclusive": end.isoformat(),
    }
