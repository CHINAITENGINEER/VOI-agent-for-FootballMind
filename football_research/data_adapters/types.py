"""Private research projections; these do not replace platform public contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Timezone-aware timestamps are required')
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ResearchMatch:
    match_id: str
    competition_id: str
    season: str | None
    home_team_id: str
    away_team_id: str
    kickoff: datetime
    home_goals: int | None = None
    away_goals: int | None = None
    result_available_at: datetime | None = None
    fixture_available_at: datetime | None = None
    information_mode: str = 'operational'
    source: str = 'unknown'
    input_manifest: tuple[str, ...] = ()
    score_period: str = 'unknown'
    status: str = 'unknown'
    time_basis: str = 'observed_at_only'
    entity_quality: str = 'source_id'
    competition_name: str = ''
    home_team_name: str = ''
    away_team_name: str = ''
    # Separate clocks: source publication, receipt, database commit and system
    # availability. None preserves compatibility with earlier research exports.
    source_published_at: datetime | None = None
    observed_at: datetime | None = None
    ingested_at: datetime | None = None
    operational_available_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ('kickoff', 'result_available_at', 'fixture_available_at',
                     'source_published_at', 'observed_at', 'ingested_at', 'operational_available_at'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, utc(value))
        object.__setattr__(self, 'input_manifest', tuple(self.input_manifest))
        if not all((self.match_id, self.competition_id, self.home_team_id, self.away_team_id)):
            raise ValueError('Research identifiers must be nonempty')
        if self.home_team_id == self.away_team_id:
            raise ValueError('A team cannot play itself')
        for value in (self.home_goals, self.away_goals):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('Goals must be nonnegative integers or null')
        if (self.home_goals is None) != (self.away_goals is None):
            raise ValueError('Both score selections must be present or both null')
        if self.result_available_at is not None:
            if self.home_goals is None or self.score_period != 'regulation' or self.status != 'finished':
                raise ValueError('Available results require finished regulation scores')
            if self.result_available_at < self.kickoff:
                raise ValueError('A result cannot be available before kickoff')
        if self.information_mode not in {'operational', 'historical_reconstruction', 'synthetic_test'}:
            raise ValueError('Unknown information_mode')

    @property
    def outcome(self) -> int | None:
        """Class index: home=0, draw=1, away=2; only accepted regulation labels."""
        if self.score_period != 'regulation' or self.status != 'finished' or self.home_goals is None:
            return None
        return 0 if self.home_goals > self.away_goals else 1 if self.home_goals == self.away_goals else 2


@dataclass(frozen=True)
class ResearchOddsSnapshot:
    match_id: str
    observed_at: datetime
    home_price: float
    draw_price: float
    away_price: float
    provider: str = 'spftll'
    bookmaker: str | None = None
    source_timestamp: datetime | None = None
    period: str = 'regulation'
    settlement_rule: str = '90_minutes_plus_stoppage_excludes_extra_time_penalties'
    information_mode: str = 'operational'
    input_manifest: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'observed_at', utc(self.observed_at))
        if self.source_timestamp is not None:
            object.__setattr__(self, 'source_timestamp', utc(self.source_timestamp))
        object.__setattr__(self, 'input_manifest', tuple(self.input_manifest))
        if any(isinstance(p, bool) or not math.isfinite(p) or p <= 1 for p in self.prices):
            raise ValueError('All decimal odds must be finite and greater than one')
        if self.period != 'regulation':
            raise ValueError('Only regulation 1x2 prices are accepted')
        if not self.match_id or not self.provider or not self.settlement_rule:
            raise ValueError('Odds provenance and settlement must be present')

    @property
    def prices(self) -> tuple[float, float, float]:
        return self.home_price, self.draw_price, self.away_price

    @property
    def available_at(self) -> datetime:
        # Source update times do not prove historical system observation.
        return self.observed_at


@dataclass(frozen=True)
class ResearchDataset:
    matches: tuple[ResearchMatch, ...]
    odds: tuple[ResearchOddsSnapshot, ...]
    dataset_version: str
    input_manifest: tuple[str, ...]
    audit: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [m.match_id for m in self.matches]
        if len(ids) != len(set(ids)):
            raise ValueError('ResearchDataset requires one current projection per match ID')


def json_record(value: Any) -> dict[str, Any]:
    record = asdict(value)
    return {key: item.isoformat() if isinstance(item, datetime) else item for key, item in record.items()}
