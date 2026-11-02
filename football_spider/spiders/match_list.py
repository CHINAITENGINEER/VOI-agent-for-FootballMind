"""Fetch the real football match list endpoint discovered in DevTools."""

from __future__ import annotations

from datetime import date
from typing import Any

from football_spider.config import Settings
from football_spider.models.match import Match
from football_spider.parser.match_parser import MatchParser
from football_spider.spiders.base import RequestsJsonClient


class MatchListSpider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.parser = MatchParser()

    def fetch_raw(self) -> dict[str, Any]:
        # dateFormat is intentionally empty: the production page sends exactly
        # this and the server returns all currently published date groups.
        params = {
            "dateFormat": "",
            "notSingle": "false",
            "mKind": 0,
            "reqtype": 0,
        }
        with RequestsJsonClient(self.settings) as client:
            return client.get_json(self.settings.list_api_url, params)

    def fetch(self, start_date: date, end_date: date) -> list[Match]:
        return self.parser.parse_match_list(
            self.fetch_raw(), start_date=start_date, end_date=end_date
        )
