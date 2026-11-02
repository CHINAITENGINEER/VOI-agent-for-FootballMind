"""Runtime configuration for the football collector."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseModel):
    page_url: str = "https://xx.spftll.cn/"
    list_api_url: str = (
        "https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll"
    )
    detail_api_url: str = "https://justpost.haoyun999.cn/api/match/GetInfo"
    generic_match_list_api_url: str = (
        "https://justpost.haoyun999.cn/api/match/GetList"
    )
    more_odds_api_url: str = (
        "https://slb.haoyun999.cn/api/Game/GetMoreSpInfo"
    )

    database_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("FOOTBALL_DB_PATH", BASE_DIR / "data" / "matches.db")
        )
    )
    cookie_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("FOOTBALL_COOKIE_PATH", BASE_DIR / "data" / "cookies.json")
        )
    )
    log_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("FOOTBALL_LOG_PATH", BASE_DIR / "logs" / "spider.log")
        )
    )

    future_days: int = int(os.getenv("FOOTBALL_FUTURE_DAYS", "7"))
    concurrency: int = int(os.getenv("FOOTBALL_CONCURRENCY", "4"))
    retries: int = int(os.getenv("FOOTBALL_RETRIES", "5"))
    connect_timeout: float = float(os.getenv("FOOTBALL_CONNECT_TIMEOUT", "30"))
    read_timeout: float = float(os.getenv("FOOTBALL_READ_TIMEOUT", "90"))
    min_delay: float = float(os.getenv("FOOTBALL_MIN_DELAY", "0.35"))
    max_delay: float = float(os.getenv("FOOTBALL_MAX_DELAY", "1.20"))
    retry_backoff: float = float(os.getenv("FOOTBALL_RETRY_BACKOFF", "1.8"))

    retry_statuses: set[int] = {403, 429, 500, 502, 503, 504}
    user_agents: tuple[str, ...] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36",
    )

    def browser_headers(self, user_agent: str) -> dict[str, str]:
        # These are the headers observed in Chrome DevTools on 2026-09-20.
        return {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "XMIME-TYPE": "3",
            "Access-Control-Allow-Origin": "*",
            "AppType": "0",
            "DeviceKind": "1",
            "Referer": self.page_url,
            "token": "null",
            "rkey": "null",
            "User-Agent": user_agent,
        }


settings = Settings()
