"""Fetch one known match ID and save a complete structured snapshot."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))

from football_spider.config import settings  # noqa: E402
from football_spider.models.match import Match, Score  # noqa: E402
from football_spider.parser.match_parser import MatchParser  # noqa: E402
from football_spider.spiders.base import AiohttpJsonClient  # noqa: E402
from football_spider.storage.database import Database  # noqa: E402


def locate_match(
    payload: dict[str, Any], match_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if payload.get("code") != 0 or not isinstance(payload.get("data"), list):
        raise ValueError(f"match/GetList 返回异常: {payload.get('code')}")

    for group in payload["data"]:
        source_match = next(
            (
                item
                for item in group.get("matchList") or []
                if str(item.get("matchId")) == match_id
            ),
            None,
        )
        if source_match is None:
            continue
        source_events = next(
            (
                item
                for item in group.get("messageList") or []
                if str(item.get("matchId")) == match_id
            ),
            {},
        )
        group_meta = {
            key: value
            for key, value in group.items()
            if key not in {"matchList", "messageList"}
        }
        return source_match, source_events, group_meta
    raise LookupError(f"match/GetList 中没有比赛 {match_id}")


async def fetch(match_id: str) -> dict[str, Any]:
    async with AiohttpJsonClient(settings) as client:
        detail_payload, odds_payload, list_payload = await asyncio.gather(
            client.get_json(
                settings.detail_api_url,
                {"gameKind": 0, "matchId": match_id},
            ),
            client.get_json(
                settings.more_odds_api_url,
                {"matchId": match_id, "mKind": 0},
            ),
            client.get_json(settings.generic_match_list_api_url, {}),
        )

    source_match, source_events, group_meta = locate_match(list_payload, match_id)
    match_time = datetime.fromisoformat(source_match["matchTime"])
    base = Match(
        id=match_id,
        time=match_time,
        league=str(source_match.get("league") or ""),
        home_team=str(source_match.get("home") or ""),
        away_team=str(source_match.get("away") or ""),
        status=str(source_match.get("finishedTitle") or "unknown"),
        score=Score(
            home=source_match.get("homeScore"),
            away=source_match.get("awayScore"),
        ),
        half_score=Score(
            home=source_match.get("homeHalfScore"),
            away=source_match.get("awayHalfScore"),
        ),
        extras={
            "source_state": source_match.get("footMatchState"),
            "start_time": source_match.get("startTime"),
            "lottery_id": source_match.get("lotteryId"),
            "sale_date": source_match.get("saledate"),
        },
    )
    match = MatchParser().merge_match_detail(base, detail_payload, odds_payload)
    home_events = source_events.get("homeMessageList") or []
    away_events = source_events.get("awayMessageList") or []
    match.extras.update(
        {
            "generic_list_group": group_meta,
            "events": {
                "home": home_events,
                "away": away_events,
            },
        }
    )

    return {
        "fetched_at": datetime.now(UTC).isoformat(),
        "source_urls": {
            "schedule": settings.generic_match_list_api_url,
            "detail": settings.detail_api_url,
            "odds": settings.more_odds_api_url,
        },
        "match": match.model_dump(mode="json"),
        "source_match": source_match,
        "event_count": {
            "home": len(home_events),
            "away": len(away_events),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("match_id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or settings.database_path.parent / f"match_{args.match_id}.json"
    snapshot = asyncio.run(fetch(args.match_id))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with Database(settings.database_path) as database:
        database.upsert_match(
            Match.model_validate(snapshot["match"]), full_detail=True
        )
    print(json.dumps({
        "output": str(output.resolve()),
        "id": snapshot["match"]["id"],
        "time": snapshot["match"]["time"],
        "league": snapshot["match"]["league"],
        "home": snapshot["match"]["home_team"],
        "away": snapshot["match"]["away_team"],
        "status": snapshot["match"]["status"],
        "score": snapshot["match"]["score"],
        "half_score": snapshot["match"]["half_score"],
        "odds": snapshot["match"]["odds"],
        "handicap": snapshot["match"]["handicap"],
        "team_rank": snapshot["match"]["team_rank"],
        "history_count": {
            "head_to_head": len(snapshot["match"]["history"]["head_to_head"]),
            "home_recent": len(snapshot["match"]["history"]["home_recent"]),
            "away_recent": len(snapshot["match"]["history"]["away_recent"]),
        },
        "event_count": snapshot["event_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
