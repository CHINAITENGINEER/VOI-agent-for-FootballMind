"""Public, allow-listed information acquisition protocol (no private payload paths)."""
from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator
from football_spider.models.contracts import Contract, AwareDatetime

PROTOCOL_VERSION = 'information-v1'
Mask = Literal['UNACQUIRED', 'OBSERVED', 'SOURCE_MISSING', 'STALE', 'FAILED']
AcquisitionStatus = Literal['ACQUIRED', 'UNCHANGED', 'CACHE_HIT', 'NO_DATA', 'TIMEOUT',
    'BAD_DATA', 'STALE', 'FAILED', 'UNAVAILABLE', 'BUDGET_EXCEEDED', 'NEEDS_NEW_CUTOFF']


class CostComponent(Contract):
    amount: float | None = Field(default=None, ge=0)
    basis: Literal['known', 'estimated', 'simulated', 'unknown'] = 'unknown'
    unit: str

    @model_validator(mode='after')
    def coherent(self):
        if (self.basis == 'unknown') != (self.amount is None):
            raise ValueError('Unknown cost requires null; other bases require a finite amount')
        return self


class CostVector(Contract):
    money: CostComponent = Field(default_factory=lambda: CostComponent(unit='USD'))
    requests: CostComponent = Field(default_factory=lambda: CostComponent(unit='HTTP_request'))
    latency_ms: CostComponent = Field(default_factory=lambda: CostComponent(unit='ms'))
    tokens: CostComponent = Field(default_factory=lambda: CostComponent(unit='token'))


class BudgetState(Contract):
    limits: dict[str, float] = Field(default_factory=dict)
    spent: dict[str, float] = Field(default_factory=lambda: dict(money=0, requests=0, latency_ms=0, tokens=0))
    unknown_components: list[str] = Field(default_factory=list)


class ActionSpec(Contract):
    action_id: str
    source: str
    tool: str
    group_ids: list[str]
    # Public operation semantics let policies distinguish an initial query
    # from a refresh, cross-source check, or a retry without inspecting any
    # owner-side implementation details.  Defaults preserve older catalogs.
    operation: Literal['query', 'refresh', 'cross_check', 'retry', 'verify'] = 'query'
    required_inputs: list[str] = Field(default_factory=lambda: ['episode_id', 'request_id'])
    capability_status: Literal['AVAILABLE', 'NOT_IMPLEMENTED', 'NEEDS_CONFIGURATION'] = 'AVAILABLE'
    prerequisites: list[str] = Field(default_factory=list)
    expected_cost: CostVector
    cost_scope: Literal['cached_projection', 'simulated_reveal', 'real_external_request']
    billing_key: str
    dependency_key: str
    cache_policy: Literal['cache_view', 'shared_response', 'refresh']
    idempotency: str = 'same request_id returns receipt with zero incremental charge'
    refresh_supported: bool = False
    # Capability only: no target-specific missingness or success forecast.
    quality_policy: str = 'unscored_source_quality; validate_after_acquisition'
    refresh_window: str | None = None
    cost_range: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    interaction_key: str | None = None


class EvidenceTimes(Contract):
    source_published_at: AwareDatetime | None = None
    observed_at: AwareDatetime | None = None
    ingested_at: AwareDatetime | None = None
    operational_available_at: AwareDatetime | None = None
    feature_available_at: AwareDatetime
    basis: str


class EvidenceGroup(Contract):
    group_id: str
    values: dict[str, float | int | str | bool | None]
    source: str
    times: EvidenceTimes
    quality: Literal['VALIDATED', 'PARTIAL', 'SOURCE_MISSING', 'STALE', 'INVALID', 'UNVERIFIED',
                     'DUPLICATE', 'CONFLICT', 'PROVIDER_ERROR', 'NO_CHANGE']
    missing_fields: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    version: str
    cache_reused: bool = True
    confidence: float | None = None
    confidence_basis: str = 'not_calibrated'
    source_id: str = ''
    provider: str | None = None
    observed_at: AwareDatetime | None = None
    available_at: AwareDatetime | None = None
    expires_at: AwareDatetime | None = None
    freshness_age: float | None = Field(default=None, ge=0)
    quality_status: str = 'unknown'
    content_hash: str = ''
    dependency_key: str = ''


class Fixture(Contract):
    match_id: str
    competition_id: str
    season: str | None = None
    home_team_id: str
    away_team_id: str
    kickoff: AwareDatetime
    home_team_name: str = ''
    away_team_name: str = ''


class ActionReceipt(Contract):
    action_id: str
    request_id: str
    status: AcquisitionStatus
    group_ids: list[str]
    charged_cost: CostVector
    sequence: int
    synthetic_perturbation: bool = False


class VisibleState(Contract):
    protocol_version: str = PROTOCOL_VERSION
    episode_id: str
    match_id: str
    prediction_cutoff: AwareDatetime
    information_mode: Literal['operational', 'historical_reconstruction']
    fixture: Fixture
    base_prior: dict[str, float]
    prior_basis: str = 'declared_uniform_prior_not_fitted'
    groups: dict[str, EvidenceGroup] = Field(default_factory=dict)
    mask: dict[str, Mask]
    history: list[ActionReceipt] = Field(default_factory=list)
    budget: BudgetState = Field(default_factory=BudgetState)
    state_version: int = 0
    experiment_role: Literal['exploratory', 'training_validation', 'prospective'] = 'exploratory'
    catalog_version: str
    cost_config_version: str


class AcquisitionResult(Contract):
    episode_id: str
    action_id: str
    request_id: str
    status: AcquisitionStatus
    revealed_groups: dict[str, EvidenceGroup] = Field(default_factory=dict)
    mask_updates: dict[str, Mask] = Field(default_factory=dict)
    charged_cost: CostVector
    sequence: int
    evidence_refs: list[str] = Field(default_factory=list)
    repeated_request: bool = False
    content_changed: bool = False
    attempts: int = 1
    retry_after_seconds: float | None = None
    synthetic_perturbation: bool = False
    reason: str | None = None
    resulting_cutoff: AwareDatetime
    suggested_new_cutoff: AwareDatetime | None = None
    request_statistics: dict[str, int | float | str | None] = Field(default_factory=dict)
    processing_status: str | None = None
