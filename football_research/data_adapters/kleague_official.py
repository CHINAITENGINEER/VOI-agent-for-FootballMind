"""Official K League monthly schedule, independent of the live analysis runtime.

Observed contract: https://www.kleague.com/schedule.do uses a JSON POST to
getScheduleList.do. No credentials or anti-bot bypass. This source provides
fixtures and club IDs/homepages, not a market prior. K1 and K2 stay distinct.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import requests

from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes

ENDPOINT = "https://www.kleague.com/getScheduleList.do"
SCHEMA = "footballmind-kleague-official-schedule-v1"
KST = ZoneInfo("Asia/Seoul")
MAX_BYTES = 2_000_000
ALIASES = {
    "K League 1": "K联赛", "K League Classic": "K联赛", "K리그1": "K联赛",
    "韩职": "K联赛", "韩K联": "K联赛", "韩国职业联赛": "K联赛",
    "K League 2": "K2联赛", "K리그2": "K2联赛", "韩K2联": "K2联赛",
}


def parameters(year, month, league_id=1):
    if type(year) is not int or not 1983 <= year <= 2100:
        raise ValueError("Explicit valid season required")
    if type(month) is not int or not 1 <= month <= 12:
        raise ValueError("Explicit calendar month required")
    if type(league_id) is not int or league_id not in {1, 2}:
        raise ValueError("Only distinct K League 1 / K League 2 contracts are supported")
    return dict(leagueId=str(league_id), teamId=None, year=str(year), month=f"{month:02d}", ticketYn="")


def parse(payload, observation, *, year, month, league_id=1):
    parameters(year, month, league_id)
    if (not isinstance(payload, dict) or str(payload.get("resultCode")) != "200"
            or not isinstance(payload.get("data"), dict)
            or not isinstance(payload["data"].get("scheduleList"), list)):
        raise ValueError("Official K League schedule response contract rejected")
    raw_clock = observation["observed_at"]
    observed = datetime.fromisoformat(raw_clock.replace("Z", "+00:00"))
    if observed.tzinfo is None:
        raise ValueError("Observation time needs an explicit timezone")
    club_by = {str(c.get("teamId")): c for c in payload["data"].get("clubList", [])
               if isinstance(c, dict) and c.get("teamId")}
    by_id, rejected, duplicate_ids, conflicted = {}, [], [], set()
    for raw in payload["data"]["scheduleList"]:
        if not isinstance(raw, dict):
            rejected.append(dict(game_id=None, reason="invalid_source_row"))
            continue
        try:
            if (type(raw.get("year")) is not int or raw["year"] != year
                    or type(raw.get("leagueId")) is not int or raw["leagueId"] != league_id):
                raise ValueError("requested_season_or_division_mismatch")
            game_id = raw["gameId"]
            if type(game_id) is not int or game_id <= 0:
                raise ValueError("missing_or_invalid_source_game_id")
            if (not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", str(raw.get("gameDate")))
                    or not re.fullmatch(r"\d{2}:\d{2}", str(raw.get("gameTime")))):
                raise ValueError("unconfirmed_kickoff")
            local = datetime.strptime(raw["gameDate"]+" "+raw["gameTime"], "%Y.%m.%d %H:%M").replace(tzinfo=KST)
            if (local.year, local.month) != (year, month):
                raise ValueError("requested_month_mismatch")
            kickoff = local.astimezone(timezone.utc)
            home_id, away_id = str(raw.get("homeTeam") or ""), str(raw.get("awayTeam") or "")
            if (not home_id or not away_id or home_id == away_id
                    or not raw.get("homeTeamName") or not raw.get("awayTeamName")):
                raise ValueError("incomplete_home_away_identity")
            status_code = str(raw.get("gameStatus") or "").strip()
            if status_code == "FE" or raw.get("endYn") == "Y":
                status = "finished"
            elif re.fullmatch(r"[12]S(?: \d+)?", status_code):
                status = "live"
            elif not status_code and raw.get("endYn") == "N" and kickoff > observed:
                status = "scheduled"
            else:
                status = "unknown"
            def score(key):
                value = raw.get(key)
                return value if status in {"live", "finished"} and type(value) is int and value >= 0 else None
            source_id = f"{year}:{league_id}:{game_id}"
            mid = "kleague:match:"+source_id
            row = dict(match_id=mid, fixture_id=mid, source="kleague_official", source_match_id=source_id,
                source_game_id=str(game_id), source_league_id=str(league_id), season=str(year),
                source_url=f"https://www.kleague.com/schedule.do?leagueId={league_id}",
                home=raw["homeTeamName"], away=raw["awayTeamName"],
                home_team=raw["homeTeamName"], away_team=raw["awayTeamName"],
                home_team_source_id=home_id, away_team_source_id=away_id,
                competition_key="K联赛" if league_id == 1 else "K2联赛",
                source_competition_name=raw.get("meetName"), competition_group="asia",
                kickoff=kickoff.isoformat(), kickoff_at=kickoff.isoformat(),
                source_timezone="Asia/Seoul", original_time=raw["gameDate"]+" "+raw["gameTime"],
                timing_policy="official Korean local schedule interpreted in Asia/Seoul, UTC export",
                time_confirmed=True, status=status, status_detail=status_code,
                home_score=score("homeGoal"), away_score=score("awayGoal"),
                observed_at=raw_clock, raw_observation=observation,
                sources=["kleague_official"], quote_observed=False, quote_snapshot=None,
                quote_observed_at=None, source_available_at=None, supported=False,
                analysis_source=None, identity_status="official_source_event_ids_observed",
                reason="官方赛程可查看；本来源未提供完整90分钟胜平负市场先验",
                club_metadata={side: {"source_team_id": tid,
                    "full_name": club_by.get(tid, {}).get("teamNameFull"),
                    "homepage": club_by.get(tid, {}).get("homepage")}
                    for side, tid in (("home", home_id), ("away", away_id))})
            if mid in conflicted:
                rejected.append(dict(game_id=game_id, reason="conflicting_duplicate_event"));continue
            if mid in by_id:
                if by_id[mid] != row:
                    by_id.pop(mid);conflicted.add(mid)
                    rejected.append(dict(game_id=game_id, reason="conflicting_duplicate_event"))
                else:
                    duplicate_ids.append(mid)
                continue
            by_id[mid] = row
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(dict(game_id=raw.get("gameId") if isinstance(raw, dict) else None,
                                 reason=str(exc)))
    return sorted(by_id.values(), key=lambda r:(r["kickoff"],r["match_id"])), {
        "rejected": rejected, "identical_duplicate_ids": duplicate_ids,
        "response_rows": len(payload["data"]["scheduleList"]), "market_prior_available": False}


class KLeagueOfficialSchedule:
    def __init__(self, root, *, transport=None, clock=None):
        self.root = Path(root)
        self.archive = RawArchive(self.root/"archive")
        self.transport = transport or self._post
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _post(params):
        with requests.Session() as session:
            session.auth = lambda request: request
            with session.post(ENDPOINT, json=params, timeout=(8,20), allow_redirects=False, stream=True,
                              headers={"User-Agent":"FootballMind/0.1 official fixture research"}) as response:
                chunks, size = [], 0
                for chunk in response.iter_content(16384):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError("Official schedule response byte budget exceeded")
                    chunks.append(chunk)
                return response.status_code, b"".join(chunks)

    def collect_month(self, year, month, league_id=1):
        params = parameters(year, month, league_id)
        cache = self.root/"cache"/f"{year}-{month:02d}-{league_id}.json"
        if cache.exists():
            try:
                old = json.loads(cache.read_text(encoding="utf-8"))
                observed = datetime.fromisoformat(old["observation"]["observed_at"].replace("Z", "+00:00"))
                if old.get("status") == "ready" and 0 <= (self.clock()-observed).total_seconds() < 900:
                    return {**old, "requests": 0, "cache_hit": True}
            except (ValueError, KeyError, TypeError, OSError):
                pass
        obs = None
        try:
            status, body = self.transport(params)
            obs = self.archive.save("kleague_official", ENDPOINT, params, body,
                status=status, observed_at=self.clock())
            if status != 200:
                raise ValueError("Official schedule HTTP "+str(status))
            matches, audit = parse(json.loads(body), obs.model_dump(mode="json"),
                                   year=year, month=month, league_id=league_id)
            result = dict(schema=SCHEMA, status="ready", matches=matches, audit=audit,
                          requests=1, cache_hit=False, method="POST", params=params, observation=obs.model_dump(mode="json"))
            atomic_write(cache, json_bytes(result))
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            if obs is None:
                obs = self.archive.save("kleague_official", ENDPOINT, params, b"",
                    error_class=type(exc).__name__, observed_at=self.clock())
            result = dict(schema=SCHEMA, status="failed", matches=[], requests=1,
                method="POST", params=params, error_class=type(exc).__name__,
                observation=obs.model_dump(mode="json"))
        atomic_write(self.root/"last-report.json", json_bytes(result))
        return result
