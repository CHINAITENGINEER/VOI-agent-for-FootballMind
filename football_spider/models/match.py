"""Canonical, source-independent football match model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Score(BaseModel):
    home: int | None = None
    away: int | None = None


class Odds(BaseModel):
    home: float | None = None
    draw: float | None = None
    away: float | None = None


class Handicap(BaseModel):
    line: str | None = None
    home: float | None = None
    draw: float | None = None
    away: float | None = None


class OverUnder(BaseModel):
    line: float | None = None
    over: float | None = None
    under: float | None = None


class TeamRank(BaseModel):
    home: str | None = None
    away: str | None = None


class HistoryMatch(BaseModel):
    id: str
    time: datetime | None = None
    league: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    score: Score = Field(default_factory=Score)
    half_score: Score = Field(default_factory=Score)
    status: str | None = None


class MatchHistory(BaseModel):
    head_to_head: list[HistoryMatch] = Field(default_factory=list)
    home_recent: list[HistoryMatch] = Field(default_factory=list)
    away_recent: list[HistoryMatch] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class Match(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    time: datetime
    league: str
    home_team: str
    away_team: str
    status: str = "unknown"
    score: Score = Field(default_factory=Score)
    half_score: Score = Field(default_factory=Score)
    odds: Odds = Field(default_factory=Odds)
    handicap: Handicap = Field(default_factory=Handicap)
    over_under: OverUnder = Field(default_factory=OverUnder)
    team_rank: TeamRank = Field(default_factory=TeamRank)
    history: MatchHistory = Field(default_factory=MatchHistory)

    red_yellow_cards: list[dict[str, Any]] | None = None
    lineup: dict[str, Any] | None = None
    players: list[dict[str, Any]] | None = None
    weather: dict[str, Any] | None = None
    odds_changes: list[dict[str, Any]] | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

