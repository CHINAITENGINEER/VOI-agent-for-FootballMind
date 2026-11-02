"""Frozen experiment costs; no inference of provider money from request count."""
from pydantic import Field, model_validator
from football_spider.models.contracts import Contract
from football_spider.storage.raw_archive import digest
from .contracts import CostComponent, CostVector

DIMENSIONS = ('money', 'requests', 'latency_ms', 'tokens')


def zero_cost():
    return CostVector(**{key: CostComponent(amount=0, basis='known', unit=unit)
        for key, unit in zip(DIMENSIONS, ('USD', 'HTTP_request', 'ms', 'token'))})


def simulated_cost(money=0.01, requests=1, latency_ms=100, tokens=0):
    return CostVector(**{key: CostComponent(amount=value, basis='simulated', unit=unit)
        for key, value, unit in zip(DIMENSIONS, (money, requests, latency_ms, tokens),
                                   ('simulation_credit', 'projection_call', 'ms', 'token'))})


class ExperimentConfig(Contract):
    mode: str = 'cache_views'
    # A provider response contains all three groups. The two/three-action
    # projection design measures controlled reveal cost, not actual HTTP saving.
    bindings: dict[str, list[str]] = Field(default_factory=lambda: {
        'strength': ['strength'], 'recent_form': ['recent_form'], 'rolling_xg': ['rolling_xg']})
    costs: dict[str, CostVector] = Field(default_factory=dict)
    budget: dict[str, float] = Field(default_factory=dict)
    seed: int = 0
    base_prior: dict[str, float] = Field(default_factory=lambda: {'home': 1/3, 'draw': 1/3, 'away': 1/3})
    experiment_role: str = 'exploratory'

    @model_validator(mode='after')
    def validate_experiment(self):
        import math
        if self.mode not in {'cache_views', 'simulated_reveal', 'source_bundle'}:
            raise ValueError('Unknown acquisition experiment mode')
        valid = {'strength', 'recent_form', 'rolling_xg'}
        if not self.bindings or any(not groups or not set(groups) <= valid or len(set(groups)) != len(groups)
                                    for groups in self.bindings.values()):
            raise ValueError('Bindings must name implemented evidence groups')
        if self.mode == 'source_bundle' and (len(self.bindings) != 1 or set(next(iter(self.bindings.values()))) != valid):
            raise ValueError('One Understat source response must reveal all dependent groups')
        if self.mode != 'cache_views' and set(self.costs) != set(self.bindings):
            raise ValueError('Controlled costs must be explicitly injected for every action')
        if self.mode == 'cache_views' and self.costs:
            raise ValueError('Existing offline cache is free of external query fees; use simulated_reveal explicitly')
        for cost in self.costs.values():
            if any(getattr(cost, k).basis != 'simulated' for k in DIMENSIONS):
                raise ValueError('Controlled reveal components must be labelled simulated')
        if set(self.base_prior) != {'home', 'draw', 'away'} or any(not math.isfinite(v) or v < 0 for v in self.base_prior.values()) or abs(sum(self.base_prior.values())-1) > 1e-9:
            raise ValueError('Declared prior must be three normalized finite probabilities')
        if any(k not in DIMENSIONS or not math.isfinite(v) or v < 0 for k, v in self.budget.items()):
            raise ValueError('Invalid budget')
        if self.experiment_role not in {'exploratory', 'training_validation', 'prospective'}:
            raise ValueError('Unknown experiment role')
        return self

    @property
    def version(self):
        return 'cost-config-v1:' + digest(self.model_dump(mode='json'))

    def cost(self, action_id):
        return self.costs[action_id].model_copy(deep=True) if action_id in self.costs else zero_cost()
