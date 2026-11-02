"""Public ESPN daily football scoreboard, with bounded I/O and replayable raw.

ESPN's public site feed is not a contracted API. It is primarily a schedule
source, but some events expose a complete 1X2 moneyline snapshot. That
snapshot is kept as an explicitly observed fallback market prior; it is never
treated as a historical odds timestamp or as a prediction by ESPN.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re
import time
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import requests
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes

ENDPOINT = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"
SCHEMA = "footballmind-espn-daily-fixtures-v1"
PROJECTION_VERSION = "footballmind-espn-scoreboard-projection-v2"
SHANGHAI = ZoneInfo("Asia/Shanghai")
MAX_BYTES = 12_000_000

# Exact competition labels; women's/youth/reserve competitions are never folded
# into their men's equivalent. Stages are retained separately in the raw label.
COMPETITIONS = {
    "English Premier League": "Premier League", "LALIGA": "西甲",
    "Spanish LALIGA": "西甲", "Italian Serie A": "意甲", "Serie A": "意甲",
    "German Bundesliga": "德甲", "French Ligue 1": "法甲",
    "UEFA Champions League": "欧冠", "UEFA Europa League": "欧罗巴",
    "UEFA Conference League": "欧协联", "Portuguese Primeira Liga": "葡超",
    "Portuguese Liga": "葡超", "Brazilian Serie A": "巴甲",
    "Brazil Serie A": "巴甲", "EFL Championship": "英冠",
    "English League Championship": "英冠", "French Ligue 2": "法乙",
    "Swedish Allsvenskan": "瑞超", "Japanese J.League": "日职",
    "Japanese J2 League": "日乙", "CONMEBOL Libertadores": "解放者杯",
    "Liga MX": "墨超", "MLS": "美职联", "Saudi Pro League": "沙特联",
    "Dutch Eredivisie": "荷甲", "EFL League One": "英甲",
    "EFL League Two": "英乙", "Brazil Serie B": "巴乙",
    "Bundesliga": "德甲", "Ligue 1": "法甲", "Ligue 2": "法乙",
    "Liga Portugal": "葡超", "Eredivisie": "荷甲", "Japanese J1 League": "日职",
    "Chinese Super League": "中超", "LALIGA 2": "西乙", "Italian Serie B": "意乙",
    # Additional mainstream competitions kept as display identities.  These
    # mappings only classify the ESPN schedule; they do not grant a market
    # prior, an official identity binding, or f/g analysis eligibility.
    "Chinese League One": "中甲", "China League One": "中甲",
    "Chinese Football Association League": "中甲",
    "K League 1": "K联赛", "South Korean K League 1": "K联赛",
    "Korean K League 1": "K联赛", "K League Classic": "K联赛",
    "Turkish Super Lig": "土超", "Turkish Super League": "土超",
    "Scottish Premiership": "苏超", "Scottish Premier League": "苏超",
    "SPFL Premiership": "苏超",
    "Belgian Pro League": "比甲", "Belgian First Division A": "比甲",
    "Argentine Primera Division": "阿甲", "Argentine Primera División": "阿甲",
    "Liga Profesional de Fútbol": "阿甲", "Argentine Liga Profesional": "阿甲",
    "Argentine LPF": "阿甲",
    "Major League Soccer": "美职联",
    "Campeonato Brasileiro Série B": "巴乙",
    "CONMEBOL Sudamericana": "南美杯", "Copa Sudamericana": "南美杯",
    "AFC Champions League Elite": "亚冠精英", "AFC Champions League": "亚冠精英",
    "AFC Champions League Two": "亚冠二级",
    "Saudi First Division": "沙甲", "Saudi Division 1": "沙甲",
    "A-League Men": "澳超", "Australian A-League": "澳超",
    "Eliteserien": "挪超", "Norwegian Eliteserien": "挪超",
    "Veikkausliiga": "芬超", "Finnish Veikkausliiga": "芬超",
    "Liga BetPlay": "哥甲", "Colombian Primera A": "哥甲",
    "Ecuadorian LigaPro": "厄甲", "LigaPro Ecuador": "厄甲",
}

# UI and audit grouping.  This is deliberately separate from COMPETITIONS:
# adding a schedule label must never silently make that competition analysable.
COMPETITION_GROUPS = {
    "Premier League": "europe_top5", "西甲": "europe_top5", "意甲": "europe_top5",
    "德甲": "europe_top5", "法甲": "europe_top5",
    "葡超": "europe_other", "荷甲": "europe_other", "土超": "europe_other",
    "苏超": "europe_other", "比甲": "europe_other", "英冠": "europe_other",
    "英甲": "europe_other", "英乙": "europe_other", "法乙": "europe_other",
    "西乙": "europe_other", "意乙": "europe_other", "瑞超": "europe_other",
    "挪超": "europe_other", "芬超": "europe_other",
    "欧冠": "continental_club", "欧罗巴": "continental_club", "欧协联": "continental_club",
    "解放者杯": "continental_club", "南美杯": "continental_club",
    "亚冠精英": "continental_club", "亚冠二级": "continental_club",
    "中超": "asia", "中甲": "asia", "日职": "asia", "日乙": "asia",
    "K联赛": "asia", "K2联赛": "asia", "沙特联": "asia", "沙甲": "asia",
    "澳超": "asia", "巴甲": "americas", "巴乙": "americas", "阿甲": "americas",
    "墨超": "americas", "美职联": "americas", "哥甲": "americas", "厄甲": "americas",
}


def competition_group(name: str | None) -> str:
    """Return a display/research group without changing analysis eligibility."""
    return COMPETITION_GROUPS.get(name or "", "other")


def utc(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if value.tzinfo is None:
        raise ValueError("Timezone required")
    return value.astimezone(timezone.utc)


def _american_to_decimal(value):
    """Convert a displayed American moneyline to decimal odds."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number == 0 or not number == number or number in (float("inf"), float("-inf")):
        return None
    return 1.0 + (number / 100.0 if number > 0 else 100.0 / abs(number))


def _moneyline_snapshot(competition, observation):
    """Return a complete observed 1X2 snapshot, or ``None``.

    The source exposes open/close displays without a bookmaker event clock.
    We therefore bind availability to the ESPN response observation only and
    preserve the provider name and raw observation reference.
    """
    for offered in competition.get("odds", []) or []:
        if not isinstance(offered, dict):
            continue
        market = offered.get("moneyline") or {}
        values = {}
        for side in ("home", "draw", "away"):
            side_data = market.get(side) or {}
            chosen = side_data.get("close") or side_data.get("open") or {}
            decimal = _american_to_decimal(chosen.get("odds"))
            if decimal is None:
                values = {}
                break
            values[side] = decimal
        value_basis = "american_moneyline_open_close"
        if len(values) != 3:
            # Some ESPN soccer feeds (observed Bet365 SPFL responses) use
            # decimal homeTeamOdds/drawOdds/awayTeamOdds instead of moneyline.
            # All three outcomes and their same-offer team IDs must agree.
            values = {}
            expected = {str(c.get("homeAway")): str((c.get("team") or {}).get("id"))
                        for c in competition.get("competitors", []) if isinstance(c, dict)}
            for side, field in (("home", "homeTeamOdds"), ("draw", "drawOdds"), ("away", "awayTeamOdds")):
                offered_side = offered.get(field)
                if not isinstance(offered_side, dict):
                    break
                number = offered_side.get("value")
                if (isinstance(number, bool) or not isinstance(number, (int, float))
                        or not 1 < number < float("inf")):
                    break
                handicap = offered_side.get("handicap")
                if handicap is not None and (isinstance(handicap, bool) or handicap != 0):
                    break
                offered_id = str((offered_side.get("team") or {}).get("id"))
                if side != "draw" and offered_id != "None" and offered_id != expected.get(side):
                    break
                values[side] = float(number)
            if len(values) != 3:
                continue
            value_basis = "decimal_homeTeamOdds_drawOdds_awayTeamOdds"
        inverse = {side: 1.0 / price for side, price in values.items()}
        total = sum(inverse.values())
        return {
            "provider": (offered.get("provider") or {}).get("name") or "ESPN odds feed",
            "market": "moneyline_1x2",
            "value_basis": value_basis,
            "decimal_odds": values,
            "no_vig_probabilities": {side: inverse[side] / total for side in ("home", "draw", "away")},
            "observed_at": observation.get("observed_at"),
            "source_timestamp": None,
            "timestamp_basis": "ESPN_response_observed_at; bookmaker_event_clock_unknown",
        }
    return None


def window(at, days=7):
    """Seven Beijing calendar dates, including the overlapping UTC date."""
    start = datetime.combine(utc(at).astimezone(SHANGHAI).date(), datetime.min.time(), SHANGHAI)
    end = start + timedelta(days=days)
    first, last = start.astimezone(timezone.utc).date(), (end-timedelta(microseconds=1)).astimezone(timezone.utc).date()
    return start, end, [first+timedelta(days=i) for i in range((last-first).days+1)]


def parse(payload, observation):
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("ESPN events list missing")
    matches, rejected = [], []
    for event in payload["events"]:
        try:
            eid = str(event["id"])
            if not eid.isdigit() or len(event["competitions"]) != 1:
                raise ValueError("ambiguous event")
            comp = event["competitions"][0]
            competitors = comp["competitors"]
            if len(competitors) != 2 or {c["homeAway"] for c in competitors} != {"home", "away"}:
                raise ValueError("ambiguous home/away")
            sides = {c["homeAway"]: c for c in competitors}
            home, away = sides["home"]["team"], sides["away"]["team"]
            if str(home["id"]) == str(away["id"]) or not home["displayName"] or not away["displayName"]:
                raise ValueError("invalid teams")
            kickoff = utc(comp.get("date") or event["date"])
            status = comp.get("status", event.get("status", {}))
            st = status.get("type", {})
            name = st.get("name", "")
            state = ("postponed" if name == "STATUS_POSTPONED" else
                     "cancelled" if name == "STATUS_CANCELED" else
                     "suspended" if name in {"STATUS_SUSPENDED", "STATUS_ABANDONED"} else
                     "finished" if st.get("completed") is True else
                     "live" if st.get("state") == "in" else
                     "scheduled" if name in {"STATUS_SCHEDULED", "STATUS_PRE_GAME"} else "unknown")
            label = comp.get("altGameNote") or ""
            if not label:
                leagues = payload.get("leagues", [])
                label = leagues[0].get("name", "") if len(leagues) == 1 else ""
            league_id = re.search(r"~l:([^~]+)", event.get("uid", ""))
            label = label or ("ESPN 赛事 " + (league_id[1] if league_id else "未知"))
            league = COMPETITIONS.get(label.split(",")[0].strip(), label)
            quote = _moneyline_snapshot(comp, observation)
            link = next((l.get("href") for l in event.get("links", [])
                         if "summary" in l.get("rel", []) and
                         urlsplit(l.get("href", "")).hostname == "www.espn.com"), None)
            def score(side):
                value = sides[side].get("score")
                return int(value) if state in {"live", "finished"} and str(value).isdigit() else None
            supported = state == "scheduled" and comp.get("timeValid") is True and quote is not None
            matches.append({
                "match_id": "espn:match:"+eid, "fixture_id": "espn:match:"+eid,
                "source": "espn", "source_match_id": eid, "source_url": link,
                "home": home["displayName"], "away": away["displayName"],
                "home_team": home["displayName"], "away_team": away["displayName"],
                "home_team_source_id": str(home["id"]), "away_team_source_id": str(away["id"]),
                "competition_key": league, "source_competition_name": label,
                "competition_group": competition_group(league),
                "source_competition_id": league_id[1] if league_id else None,
                "kickoff": kickoff.isoformat(), "kickoff_at": kickoff.isoformat(),
                "status": state, "status_detail": st.get("detail"),
                "time_confirmed": comp.get("timeValid") is True,
                "home_score": score("home"), "away_score": score("away"),
                "observed_at": observation["observed_at"], "raw_observation": observation,
                "sources": ["espn"], "supported": supported, "quote_observed": quote is not None,
                "quote_snapshot": quote, "quote_observed_at": quote.get("observed_at") if quote else None,
                "reason": None if supported else ("赛程可查看；尚未发现完整 ESPN 1X2 先验" if quote is None
                          else "ESPN 比赛状态或开球时间尚未满足赛前启动条件"),
                "identity_status": "espn_schedule_observed", "analysis_source": "espn" if supported else None,
                "model_scope": "premier_league_research" if league == "Premier League" else "cross_competition_experimental",
            })
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            rejected.append({"event_id": event.get("id") if isinstance(event, dict) else None,
                             "reason": type(exc).__name__+": "+str(exc)[:160]})
    return matches, rejected


class ESPNScoreboard:
    def __init__(self, root, *, transport=None, clock=None):
        self.root = Path(root)
        self.transport = transport or self._get
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.archive = RawArchive(self.root / "archive")

    @staticmethod
    def _get(params):
        with requests.Session() as session:
            # Retain OS proxy routing, never silently acquire netrc credentials.
            session.auth = lambda request: request
            with session.get(ENDPOINT, params=params, headers={"User-Agent": "FootballMind/0.1 (fixture research)"},
                             timeout=(6, 15), allow_redirects=False, stream=True) as response:
                chunks, size, started = [], 0, time.monotonic()
                for chunk in response.iter_content(16384):
                    size += len(chunk)
                    if size > MAX_BYTES or time.monotonic()-started > 20:
                        raise ValueError("ESPN response limit exceeded")
                    chunks.append(chunk)
                return response.status_code, b"".join(chunks)

    def _day(self, day, at, *, force=False):
        path = self.root / "cache" / (day.isoformat()+".json")
        try:
            previous = json.loads(path.read_text(encoding="utf8")) if path.exists() else None
        except (ValueError, OSError):
            previous = None
        ttl = 60 if day == at.date() else 900 if day > at.date() else 21600
        if not force and previous and 0 <= (at-utc(previous["observed_at"])).total_seconds() < ttl:
            if previous.get("projection_version") == PROJECTION_VERSION:
                return {**previous, "cache_hit": True, "http_attempts": 0, "stale": False}
            try:
                # Code upgrades must not keep treating an already archived
                # complete legacy market as absent until the HTTP cache expires.
                # Reproject exact source bytes, preserving their original clock.
                from football_spider.models.contracts import Observation
                cached_observation = Observation.model_validate(previous["observation"])
                if (cached_observation.endpoint != ENDPOINT or cached_observation.http_status != 200
                        or cached_observation.params.get("dates") != day.strftime("%Y%m%d")):
                    raise ValueError("Cached source does not bind the selected endpoint/day")
                matches, rejected = parse(json.loads(self.archive.read_body(cached_observation)),
                                          cached_observation.model_dump(mode="json"))
                previous = {**previous, "matches": matches, "rejected": rejected,
                            "projection_version": PROJECTION_VERSION}
                atomic_write(path, json_bytes(previous))
                return {**previous, "cache_hit": True, "http_attempts": 0,
                        "stale": False, "projection_replayed": True}
            except (KeyError, ValueError, TypeError, OSError):
                # Missing or mismatched source bytes require a fresh response.
                pass
        params = {"dates": day.strftime("%Y%m%d"), "limit": 1000}
        obs = None
        try:
            status, body = self.transport(params)
            obs = self.archive.save("espn", ENDPOINT, params, body, status=status,
                                    observed_at=utc(self.clock())).model_dump(mode="json")
            if status != 200:
                raise ValueError("ESPN HTTP "+str(status))
            payload = json.loads(body)
            matches, rejected = parse(payload, obs)
            if payload["events"] and not matches:
                raise ValueError("All ESPN events failed schema validation")
            result = {"schema": SCHEMA, "date_utc": day.isoformat(), "observed_at": obs["observed_at"],
                      "matches": matches, "rejected": rejected, "observation": obs,
                      "projection_version": PROJECTION_VERSION,
                      "response_may_be_truncated": len(payload["events"]) >= 1000}
            atomic_write(path, json_bytes(result))
            return {**result, "cache_hit": False, "http_attempts": 1, "stale": False}
        except Exception as exc:
            error = type(exc).__name__+": "+str(exc)[:180]
            if obs is None:
                obs = self.archive.save("espn", ENDPOINT, params, b"", error_class=type(exc).__name__,
                                        observed_at=utc(self.clock())).model_dump(mode="json")
            return {**(previous or {"matches": [], "observed_at": None}), "date_utc": day.isoformat(),
                    "stale": True, "error": error, "failure_observation": obs,
                    "http_attempts": 1, "cache_hit": False}

    def collect_fixture(self, fixture, *, catalog_root=None, at=None):
        """One target day, never the entire catalogue, for run initialization.

        On transport failure only, a <=300 second immutable catalogue response
        may be replayed. Its original observation time is retained. A successful
        fresh response withdrawing odds/status always takes precedence.
        """
        from football_spider.models.contracts import Observation
        at = utc(at or self.clock())
        kickoff = utc(fixture["kickoff"])
        raw = fixture.get("raw_observation") or {}
        day = kickoff.date()
        try:
            observed_day = datetime.strptime(raw.get("params", {}).get("dates", ""), "%Y%m%d").date()
            if abs((observed_day-day).days) <= 1:
                day = observed_day
        except (TypeError, ValueError):
            pass
        daily = self._day(day, at, force=True)
        target = str(fixture.get("fixture_id") or fixture["match_id"])
        matches = [] if daily.get("stale") else daily.get("matches", [])
        mode, fallback_error = "fresh_target_day", None
        failed = daily.get("failure_observation") or {}
        # Only a failed request can use the last known quote; not fresh absence,
        # a malformed successful response, a started game or expired evidence.
        if (daily.get("stale") and failed.get("http_status") != 200 and catalog_root
                and at < kickoff-timedelta(minutes=1)):
            try:
                source = RawArchive(Path(catalog_root) / "archive")
                obs = Observation.model_validate(raw)
                metadata = (source.root / obs.payload_path).with_suffix("").with_suffix(".observation.json")
                if not metadata.resolve().is_relative_to(source.root):
                    raise ValueError("Catalogue archive path outside source")
                saved = Observation.model_validate_json(metadata.read_bytes())
                if saved.model_dump() != obs.model_dump():
                    raise ValueError("Catalogue observation differs from immutable metadata")
                if (obs.endpoint != ENDPOINT or obs.http_status != 200 or obs.error_class
                        or not 0 <= (at-obs.observed_at).total_seconds() <= 300):
                    raise ValueError("Catalogue observation is not a recent successful source response")
                body = source.read_body(obs)
                replay, _ = parse(json.loads(body), raw)
                candidate = next((m for m in replay if m["match_id"] == target), None)
                if (not candidate or candidate["status"] != "scheduled" or not candidate.get("time_confirmed")
                        or not candidate.get("quote_snapshot") or utc(candidate["kickoff"]) != kickoff):
                    raise ValueError("Catalogue response cannot initialize the selected prematch event")
                copied = self.archive.save(obs.source, obs.endpoint, obs.params, body,
                    status=obs.http_status, headers=obs.response_headers, request_id=obs.request_id,
                    context=obs.context, observed_at=obs.observed_at, observation_id=obs.id)
                matches, _ = parse(json.loads(body), copied.model_dump(mode="json"))
                mode = "recent_catalogue_raw_replay_after_transport_failure"
            except (ValueError, OSError, KeyError, TypeError) as exc:
                fallback_error = type(exc).__name__+": "+str(exc)[:180]
        selected = [m for m in matches if m.get("match_id") == target]
        report = {"schema": "footballmind-espn-target-initialization-v1", "source": "espn",
            "target_match_id": target, "matches": selected, "http_attempts": daily["http_attempts"],
            "checked_at": utc(self.clock()).isoformat(), "acquisition_mode": mode,
            "catalogue_replay_max_age_seconds": 300, "fallback_error": fallback_error,
            "status": "ready" if selected else "failed" if daily.get("stale") else "target_not_returned",
            "dates": [{k: v for k, v in daily.items() if k != "matches"}]}
        atomic_write(self.root / "last-report.json", json_bytes(report))
        return report

    def collect(self, at=None):
        at = utc(at or self.clock())
        start, end, dates = window(at)
        with ThreadPoolExecutor(max_workers=3) as pool:
            reports = list(pool.map(lambda day: self._day(day, at), dates))
        matches = {}
        for report in reports:
            for match in report["matches"]:
                if not start <= utc(match["kickoff"]) < end:
                    continue
                value = {**match, "source_stale": report["stale"]}
                label = value.get("source_competition_name", value["competition_key"])
                value["competition_key"] = COMPETITIONS.get(label.split(",")[0].strip(), label)
                old = matches.get(match["match_id"])
                if old is None or utc(value["observed_at"]) > utc(old["observed_at"]):
                    matches[match["match_id"]] = value
        report = {"schema": SCHEMA, "source": "espn", "endpoint": ENDPOINT,
                  "status": "ready" if not any(r["stale"] for r in reports) else "partial" if matches else "failed",
                  "checked_at": utc(self.clock()).isoformat(), "display_timezone": "Asia/Shanghai",
                  "window_start": start.isoformat(), "window_end_exclusive": end.isoformat(),
                  "http_attempts": sum(r["http_attempts"] for r in reports),
                  "matches": sorted(matches.values(), key=lambda m: (m["kickoff"], m["match_id"])),
                  "dates": [{k: v for k, v in r.items() if k != "matches"} for r in reports]}
        atomic_write(self.root / "last-report.json", json_bytes(report))
        return report
