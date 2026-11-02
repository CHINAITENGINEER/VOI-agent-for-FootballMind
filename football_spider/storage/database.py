"""SQLite storage with idempotent upserts and per-day checkpoints."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from football_spider.models.match import Match


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS matches (
                id TEXT PRIMARY KEY,
                match_time TEXT NOT NULL,
                league TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                odds_home REAL,
                odds_draw REAL,
                odds_away REAL,
                created_time TEXT NOT NULL,
                updated_time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'unknown',
                home_half_score INTEGER,
                away_half_score INTEGER,
                handicap_line TEXT,
                handicap_home REAL,
                handicap_draw REAL,
                handicap_away REAL,
                details_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_matches_match_time
            ON matches(match_time);

            CREATE TABLE IF NOT EXISTS crawl_progress (
                run_key TEXT NOT NULL,
                match_id TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                updated_time TEXT NOT NULL,
                PRIMARY KEY (run_key, match_id)
            );
            """
        )
        self.connection.commit()

    def upsert_match(self, match: Match, *, full_detail: bool = False) -> None:
        now = datetime.now(UTC).isoformat()
        details = match.model_dump(mode="json")
        if not full_detail:
            existing_row = self.connection.execute(
                "SELECT details_json FROM matches WHERE id=?", (match.id,)
            ).fetchone()
            if existing_row:
                try:
                    existing = Match.model_validate_json(existing_row["details_json"])
                    existing.time = match.time
                    existing.league = match.league
                    existing.home_team = match.home_team
                    existing.away_team = match.away_team
                    existing.status = match.status
                    for field in ("home", "draw", "away"):
                        value = getattr(match.odds, field)
                        if value is not None:
                            setattr(existing.odds, field, value)
                    for field in ("line", "home", "draw", "away"):
                        value = getattr(match.handicap, field)
                        if value is not None:
                            setattr(existing.handicap, field, value)
                    if match.team_rank.home is not None:
                        existing.team_rank.home = match.team_rank.home
                    if match.team_rank.away is not None:
                        existing.team_rank.away = match.team_rank.away
                    existing.extras = {**existing.extras, **match.extras}
                    details = existing.model_dump(mode="json")
                except (TypeError, ValueError):
                    # A future schema migration may make an older JSON document
                    # invalid. Scalar columns still update and the next full
                    # detail refresh replaces this document.
                    pass
        self.connection.execute(
            """
            INSERT INTO matches (
                id, match_time, league, home_team, away_team,
                home_score, away_score, odds_home, odds_draw, odds_away,
                created_time, updated_time, status,
                home_half_score, away_half_score,
                handicap_line, handicap_home, handicap_draw, handicap_away,
                details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                match_time=excluded.match_time,
                league=excluded.league,
                home_team=excluded.home_team,
                away_team=excluded.away_team,
                home_score=COALESCE(excluded.home_score, matches.home_score),
                away_score=COALESCE(excluded.away_score, matches.away_score),
                odds_home=COALESCE(excluded.odds_home, matches.odds_home),
                odds_draw=COALESCE(excluded.odds_draw, matches.odds_draw),
                odds_away=COALESCE(excluded.odds_away, matches.odds_away),
                updated_time=excluded.updated_time,
                status=excluded.status,
                home_half_score=COALESCE(excluded.home_half_score, matches.home_half_score),
                away_half_score=COALESCE(excluded.away_half_score, matches.away_half_score),
                handicap_line=COALESCE(excluded.handicap_line, matches.handicap_line),
                handicap_home=COALESCE(excluded.handicap_home, matches.handicap_home),
                handicap_draw=COALESCE(excluded.handicap_draw, matches.handicap_draw),
                handicap_away=COALESCE(excluded.handicap_away, matches.handicap_away),
                details_json=excluded.details_json
            """,
            (
                match.id,
                match.time.isoformat(),
                match.league,
                match.home_team,
                match.away_team,
                match.score.home,
                match.score.away,
                match.odds.home,
                match.odds.draw,
                match.odds.away,
                now,
                now,
                match.status,
                match.half_score.home,
                match.half_score.away,
                match.handicap.line,
                match.handicap.home,
                match.handicap.draw,
                match.handicap.away,
                json.dumps(details, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        self.connection.commit()

    def completed_ids(self, run_key: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT match_id FROM crawl_progress WHERE run_key=? AND status='done'",
            (run_key,),
        )
        return {str(row["match_id"]) for row in rows}

    def mark_progress(
        self, run_key: str, match_id: str, status: str, error: str | None = None
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self.connection.execute(
            """
            INSERT INTO crawl_progress(run_key, match_id, status, error, updated_time)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_key, match_id) DO UPDATE SET
                status=excluded.status,
                error=excluded.error,
                updated_time=excluded.updated_time
            """,
            (run_key, match_id, status, error, now),
        )
        self.connection.commit()

    def count_matches(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM matches").fetchone()
        return int(row["count"])
