"""Concurrent detail and expanded-odds collection."""

from __future__ import annotations

import asyncio

from football_spider.config import Settings
from football_spider.models.match import Match
from football_spider.parser.match_parser import MatchParser
from football_spider.spiders.base import AiohttpJsonClient


class MatchDetailSpider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.parser = MatchParser()
        self.client = AiohttpJsonClient(settings)

    async def __aenter__(self) -> "MatchDetailSpider":
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.client.__aexit__(*args)

    async def fetch(self, match: Match) -> Match:
        detail_task = asyncio.create_task(
            self.client.get_json(
                self.settings.detail_api_url,
                {"gameKind": 0, "matchId": match.id},
            )
        )
        odds_task = asyncio.create_task(
            self.client.get_json(
                self.settings.more_odds_api_url,
                {"matchId": match.id, "mKind": 0},
            )
        )
        detail_payload, odds_payload = await asyncio.gather(detail_task, odds_task)
        return self.parser.merge_match_detail(match, detail_payload, odds_payload)
