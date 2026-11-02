"""Versioned, source-neutral contracts. Unknown values are never zero-filled."""
from __future__ import annotations
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt

SCHEMA_VERSION = '1.0'
PARSER_VERSION = '1.0'


def now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(StrEnum):
    OK = 'OK'
    PARTIAL = 'PARTIAL'
    NO_DATA = 'NO_DATA'
    SKIPPED_DISABLED = 'SKIPPED_DISABLED'
    NEEDS_CREDENTIALS = 'NEEDS_CREDENTIALS'
    NEEDS_DATABASE = 'NEEDS_DATABASE'
    RATE_LIMITED = 'RATE_LIMITED'
    FAILED = 'FAILED'


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Observation(Contract):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    endpoint: str
    method: str = 'GET'
    params: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    observed_at: AwareDatetime = Field(default_factory=now)
    http_status: int | None = None
    payload_path: str
    content_hash: str
    payload_format: str
    response_headers: dict[str, str] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    error_class: str | None = None
    schema_version: str = SCHEMA_VERSION


# Single public type; RawObservation is a vocabulary alias, not another schema.
RawObservation = Observation


class EntityRef(Contract):
    source_id: str
    name: str
    country: str | None = None
    category: str = 'unknown'
    id_quality: Literal['stable', 'source_name', 'match_scoped'] = 'stable'


class Score(Contract):
    home: StrictInt | None = Field(default=None, ge=0)
    away: StrictInt | None = Field(default=None, ge=0)


class MatchRecord(Contract):
    source_id: str
    competition: EntityRef
    season: str | None = None
    home: EntityRef
    away: EntityRef
    kickoff: AwareDatetime
    status: Literal['scheduled', 'live', 'finished', 'postponed', 'cancelled', 'suspended', 'unknown']
    neutral: bool | None = None
    stage: str | None = None
    score_period: Literal['regulation', 'extra_time', 'penalties', 'unknown'] = 'unknown'
    score: Score = Field(default_factory=Score)
    half_score: Score = Field(default_factory=Score)
    source_published_at: AwareDatetime | None = None
    trusted_snapshot_at: AwareDatetime | None = None
    original_time: str
    source_timezone: str
    missing: dict[str, str] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    coverage: str = 'unknown'


class Quote(Contract):
    source_match_id: str
    provider: str
    bookmaker: str | None = None
    market_type: Literal['1x2', 'jc_handicap_3way', 'asian_handicap', 'exact_total_goals', 'over_under']
    period: Literal['regulation', 'first_half', 'extra_time', 'unknown']
    selection: str
    line: float | None = None
    decimal_price: float = Field(gt=1)
    source_timestamp: AwareDatetime | None = None
    trusted_snapshot_at: AwareDatetime | None = None
    settlement_rule: str


class Issue(Contract):
    rule: str
    severity: Literal['error', 'warning', 'recoverable']
    record_key: str
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    rule_version: str = '1.0'


class TeamStatistic(Contract):
    source_match_id: str
    source_team_id: str
    metric: Literal['xg', 'xga']
    value: float = Field(ge=0)
    unit: str = 'expected_goals'
    provider: str
    definition_version: str
    source_published_at: AwareDatetime | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class NormalizedBatch(Contract):
    matches: list[MatchRecord] = Field(default_factory=list)
    quotes: list[Quote] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    statistics: list[TeamStatistic] = Field(default_factory=list)
    parser_version: str = PARSER_VERSION


class CollectionResult(Contract):
    source: str
    status: RunStatus = RunStatus.OK
    observations: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    requests: int = 0
    scope: dict[str, Any] = Field(default_factory=dict)
