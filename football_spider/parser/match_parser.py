"""Translate source JSON fields into the canonical Pydantic model."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta
from typing import Any

from football_spider.models.match import (
    Handicap,
    HistoryMatch,
    Match,
    MatchHistory,
    Odds,
    Score,
    TeamRank,
)


class SourceResponseError(ValueError):
    pass


def _assert_success(payload: dict[str, Any], source: str) -> Any:
    if payload.get("code") != 0:
        raise SourceResponseError(
            f"{source} 返回业务错误 code={payload.get('code')}: "
            f"{payload.get('message') or payload.get('info')}"
        )
    return payload.get("data")


def _to_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise SourceResponseError(f"无法解析比赛时间: {value!r}") from exc


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _status(state: Any, title: Any = None) -> str:
    if title:
        return str(title)
    # Only states directly observed in captured responses are named. Unknown
    # values are retained losslessly instead of being guessed.
    if state == 0:
        return "未开"
    if state == -1:
        return "完场"
    return "unknown" if state is None else f"state_{state}"


class MatchParser:
    def parse_match_list(
        self, payload: dict[str, Any], start_date: date, end_date: date
    ) -> list[Match]:
        groups = _assert_success(payload, "GetSimpleMatchsAll")
        if not isinstance(groups, list):
            raise SourceResponseError("GetSimpleMatchsAll.data 不是数组")

        window_start = datetime.combine(start_date, time.min)
        window_end = datetime.combine(end_date + timedelta(days=1), time.min)
        matches: dict[str, Match] = {}
        for group in groups:
            if not isinstance(group, dict):
                continue
            for item in group.get("list") or []:
                parsed = self._parse_list_item(item)
                if window_start <= parsed.time < window_end:
                    matches[parsed.id] = parsed
        return sorted(matches.values(), key=lambda match: (match.time, match.id))

    def _parse_list_item(self, item: dict[str, Any]) -> Match:
        match_time = _to_datetime(item.get("matchTime"))
        if match_time is None:
            raise SourceResponseError(f"比赛 {item.get('matchId')} 缺少 matchTime")
        match_id = item.get("matchId")
        if match_id is None:
            raise SourceResponseError("比赛列表项缺少 matchId")

        return Match(
            id=str(match_id),
            time=match_time,
            league=str(item.get("leagueChs") or ""),
            home_team=str(item.get("homeChs") or ""),
            away_team=str(item.get("awayChs") or ""),
            status=_status(item.get("footMatchState")),
            odds=Odds(
                home=_to_float(item.get("spfWinFoot")),
                draw=_to_float(item.get("spfEqualFoot")),
                away=_to_float(item.get("spfLoseFoot")),
            ),
            handicap=Handicap(
                line=None if item.get("goalFoot") is None else str(item["goalFoot"]),
                home=_to_float(item.get("rspfWinFoot")),
                draw=_to_float(item.get("rspfEqualFoot")),
                away=_to_float(item.get("rspfLoseFoot")),
            ),
            team_rank=TeamRank(
                home=None if item.get("homeRank") in (None, "") else str(item["homeRank"]),
                away=None if item.get("awayRank") in (None, "") else str(item["awayRank"]),
            ),
            extras={
                "match_guid": item.get("matchGuid"),
                "lottery_id": item.get("lotteryId"),
                "home_id": item.get("homeId"),
                "away_id": item.get("awayId"),
                "source_state": item.get("footMatchState"),
            },
        )

    def merge_match_detail(
        self,
        base_match: Match,
        detail_payload: dict[str, Any],
        odds_payload: dict[str, Any],
    ) -> Match:
        detail = _assert_success(detail_payload, "match/GetInfo")
        odds_data = _assert_success(odds_payload, "GetMoreSpInfo")
        if not isinstance(detail, dict):
            raise SourceResponseError("match/GetInfo.data 不是对象")
        if not isinstance(odds_data, dict):
            odds_data = {}
        more = odds_data.get("footballMoreSpInfo") or {}

        result = base_match.model_copy(deep=True)
        result.home_team = str(detail.get("home") or more.get("hometeamchs") or result.home_team)
        result.away_team = str(detail.get("away") or more.get("awayteamchs") or result.away_team)
        result.league = str(detail.get("league") or more.get("leaguechs") or result.league)
        result.time = _to_datetime(detail.get("matchTime")) or result.time
        result.status = _status(detail.get("footMatchState"), detail.get("finishedTitle"))
        result.score = Score(
            home=_to_int(detail.get("homeScore")),
            away=_to_int(detail.get("awayScore")),
        )
        result.half_score = Score(
            home=_to_int(detail.get("homeHalfScore")),
            away=_to_int(detail.get("awayHalfScore")),
        )

        if more:
            result.odds = Odds(
                home=_to_float(more.get("spf_win")),
                draw=_to_float(more.get("spf_draw")),
                away=_to_float(more.get("spf_lost")),
            )
            result.handicap = Handicap(
                line=None if more.get("rfspf_goal") is None else str(more["rfspf_goal"]),
                home=_to_float(more.get("rfspf_win")),
                draw=_to_float(more.get("rfspf_draw")),
                away=_to_float(more.get("rfspf_lost")),
            )
            result.team_rank = TeamRank(
                home=None if more.get("homerank") in (None, "") else str(more["homerank"]),
                away=None if more.get("awayrank") in (None, "") else str(more["awayrank"]),
            )

        result.history = MatchHistory(
            head_to_head=self._parse_history_list(detail.get("battleList")),
            home_recent=self._parse_history_list(detail.get("homeBattleList")),
            away_recent=self._parse_history_list(detail.get("awayBattleList")),
            summary={
                "head_to_head": detail.get("battleReport"),
                "home_recent": detail.get("homeBattleReport"),
                "away_recent": detail.get("awayBattleReport"),
            },
        )

        total_goal_odds = {
            key: more.get(key) for key in ("t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7")
            if more.get(key) is not None
        }
        result.extras = {
            **deepcopy(result.extras),
            "source": {
                "list": "GetSimpleMatchsAll",
                "detail": "match/GetInfo",
                "odds": "GetMoreSpInfo",
            },
            "start_time": detail.get("startTime"),
            "total_goals_exact_odds": total_goal_odds,
            "correct_score_odds": {
                key: value
                for key, value in more.items()
                if (key.startswith("sw") or key.startswith("sd") or key.startswith("sl"))
                and value is not None
            },
            "half_full_odds": {
                key: value
                for key, value in more.items()
                if key.startswith("ht") and value is not None
            },
        }
        # The verified responses expose exact-total-goals odds, not an Asian
        # over/under line. over_under therefore remains null rather than being
        # populated with a semantically different market.
        return result

    def _parse_history_list(self, values: Any) -> list[HistoryMatch]:
        if not isinstance(values, list):
            return []
        result: list[HistoryMatch] = []
        for item in values:
            if not isinstance(item, dict) or item.get("matchId") is None:
                continue
            result.append(
                HistoryMatch(
                    id=str(item["matchId"]),
                    time=_to_datetime(item.get("matchTime")),
                    league=item.get("league"),
                    home_team=item.get("home"),
                    away_team=item.get("away"),
                    score=Score(
                        home=_to_int(item.get("homeScore")),
                        away=_to_int(item.get("awayScore")),
                    ),
                    half_score=Score(
                        home=_to_int(item.get("homeHalfScore")),
                        away=_to_int(item.get("awayHalfScore")),
                    ),
                    status=_status(item.get("footMatchState"), item.get("finishedTitle")),
                )
            )
        return result
