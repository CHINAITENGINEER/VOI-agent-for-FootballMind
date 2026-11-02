"""Deterministic dynamic evidence episodes for value-estimation research.

The scenarios use projections from a verified COMPLETE export and apply only
explicit, synthetic response/clock perturbations. They never modify the raw
archive or the frozen production bundle. A policy receives only the public
Pydantic objects returned by this class.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import RLock
import hashlib

from football_spider.storage.raw_archive import digest
from .contracts import (ActionReceipt, ActionSpec, AcquisitionResult, BudgetState,
                        EvidenceGroup, EvidenceTimes, VisibleState)
from .costs import ExperimentConfig, simulated_cost, zero_cost
from .frozen import FrozenEvidenceEnvironment, utc

SCENARIOS = ('fresh', 'stale', 'duplicate', 'no_change', 'conflict',
             'no_data_then_available', 'provider_error', 'complementary',
             'shared_response', 'refresh_cycle')


class DynamicEpisodeEnvironment:
    """Owner-side dynamic episode with explicit quality/time/dependency changes."""

    @classmethod
    def from_export(cls, export_dir, scenario='fresh', config=None, *, match_id=None,
                    prediction_cutoff=None, seed=0):
        if scenario not in SCENARIOS:
            raise ValueError('Unknown dynamic scenario')
        base = FrozenEvidenceEnvironment.from_export(export_dir, config or ExperimentConfig())
        choices = base.list_fixtures()
        if not choices:
            raise ValueError('Completed export has no fixtures')
        chosen = next((v for v in choices if v['match_id'] == match_id), None) if match_id else None
        if chosen is None and match_id:
            raise ValueError('Requested fixture is not in completed export')
        if chosen is None:
            # Prefer a deterministic snapshot with all three implemented groups
            # supported; this keeps quality episodes about dynamic state rather
            # than accidentally selecting an opening-round missing window.
            for candidate in choices:
                snap = base._features.get((candidate['match_id'], utc(candidate['prediction_cutoff']), candidate['information_mode']))
                features = (snap or {}).get('features', {})
                if (features.get('elo', {}).get('home') is not None
                        and (features.get('home_form') or {}).get('matches', 0) > 0
                        and features.get('home_rolling_xg_for') is not None
                        and features.get('away_rolling_xg_for') is not None):
                    chosen = candidate
                    break
            chosen = chosen or choices[0]
        cutoff = prediction_cutoff or chosen['prediction_cutoff']
        base.start(chosen['match_id'], cutoff, chosen['information_mode'])
        return cls(base, scenario, seed=seed)

    def __init__(self, base, scenario, *, seed=0):
        self._base = base
        self._scenario = scenario
        self._seed = seed
        self._lock = RLock()
        self._state = None
        self._receipts = {}
        self._attempts = {}
        self._clock = None
        self._published = {}
        # A dependency cache is deliberately owner-side.  The public state
        # exposes only the requested projection and a cache-hit receipt; it
        # never exposes the complete response or hidden groups.  This allows
        # replay tests to distinguish one physical response from several
        # feature views without claiming real provider savings.
        self._dependency_cache = {}
        self._catalog = self._build_catalog()
        self.reset()

    def _cost(self, amount=0.01, *, requests=1, latency_ms=25):
        return simulated_cost(money=amount, requests=requests, latency_ms=latency_ms, tokens=0)

    def _spec(self, aid, groups, *, dependency='synthetic:source-response', provider='synthetic', amount=.01,
              interaction=None, refresh='episode_window', operation='query', cache_policy='shared_response',
              requests=1):
        cost = self._cost(amount, requests=requests, latency_ms=0 if requests == 0 else 25)
        return ActionSpec(action_id=aid, source=provider, tool='dynamic_evidence_fixture', group_ids=groups,
            operation=operation,
            expected_cost=cost, cost_scope='simulated_reveal', billing_key='simulated:' + dependency,
            dependency_key=dependency, cache_policy=cache_policy, refresh_supported=True,
            refresh_window=refresh, interaction_key=interaction,
            cost_range={'money': {'min': amount, 'max': amount},
                        'requests': {'min': float(requests), 'max': float(requests)}},
            prerequisites=['public_fixture_and_cutoff'])

    def _build_catalog(self):
        if self._scenario == 'conflict':
            return [self._spec('provider_a.strength', ['strength'], dependency='provider_a:strength', provider='provider_a'),
                    self._spec('provider_b.strength', ['strength'], dependency='provider_b:strength', provider='provider_b',
                               operation='cross_check')]
        if self._scenario == 'complementary':
            key = 'complementary_pair:strength+recent_form'
            return [self._spec('complementary.strength', ['strength'], dependency='understat:strength', interaction=key),
                    self._spec('complementary.recent_form', ['recent_form'], dependency='understat:recent_form', interaction=key)]
        if self._scenario == 'no_data_then_available':
            return [self._spec('first.no_data', ['strength'], dependency='provider:strength'),
                    self._spec('retry.strength', ['strength'], dependency='provider:strength', refresh='retry_window', operation='retry')]
        if self._scenario == 'provider_error':
            return [self._spec('provider.error', ['strength'], dependency='provider:strength'),
                    self._spec('provider.retry', ['strength'], dependency='provider:strength', refresh='retry_window',
                               operation='retry')]
        if self._scenario == 'duplicate':
            return [self._spec('primary.strength', ['strength']), self._spec('duplicate.strength', ['strength'])]
        if self._scenario == 'no_change':
            return [self._spec('primary.strength', ['strength']), self._spec('refresh.no_change', ['strength'], refresh='refresh_window', operation='refresh')]
        if self._scenario == 'stale':
            return [self._spec('stale.strength', ['strength'], refresh='expires_after_1h')]
        if self._scenario == 'shared_response':
            dependency = 'understat:league_response'
            interaction = 'shared_response:understat:league_response'
            return [
                # The first request obtains one physical response but reveals
                # only the requested projection.
                self._spec('shared.query.strength', ['strength'], dependency=dependency,
                           interaction=interaction, amount=.02, operation='query'),
                # A second view of the same response is free of incremental
                # simulated request cost and still has a separate action id.
                self._spec('shared.cache.recent_form', ['recent_form'], dependency=dependency,
                           interaction=interaction, amount=0.0, requests=0,
                           operation='query', cache_policy='cache_view', refresh='same_response'),
                self._spec('shared.refresh.rolling_xg', ['rolling_xg'], dependency=dependency,
                           interaction=interaction, amount=.02, operation='refresh',
                           cache_policy='refresh', refresh='new_response'),
            ]
        if self._scenario == 'refresh_cycle':
            return [
                self._spec('refresh.query.strength', ['strength'], dependency='understat:strength',
                           amount=.01, operation='query', refresh='expires_after_1h'),
                self._spec('refresh.refresh.strength', ['strength'], dependency='understat:strength',
                           amount=.015, operation='refresh', refresh='new_response'),
            ]
        return [self._spec('fresh.strength', ['strength'])]

    def catalog(self):
        return [v.model_copy(deep=True) for v in self._catalog]

    def configuration(self):
        return {'protocol_version': 'information-v1', 'environment': 'dynamic_episode',
                'scenario': self._scenario, 'scenario_classification': 'synthetic_mechanism_over_real_frozen_projection',
                'seed': self._seed, 'catalog_version': 'dynamic-catalog-v2:' + digest([v.model_dump(mode='json') for v in self._catalog]),
                'action_registry_hash': digest([v.model_dump(mode='json') for v in self._catalog]),
                'registry_semantics': 'operation+dependency+cache_policy+interaction_key',
                'quality_statuses': ['valid', 'duplicate', 'stale', 'source_missing', 'conflict', 'provider_error', 'no_change'],
                'confidence_calibration': 'NOT_CALIBRATED'}

    def action_registry(self):
        """Return the stable public action registry used by replay runners."""
        return [v.model_copy(deep=True) for v in sorted(self._catalog, key=lambda item: item.action_id)]

    def reset(self):
        with self._lock:
            base = self._base.visible_state()
            self._state = VisibleState(episode_id='dynamic:' + digest([base.fixture.model_dump(mode='json'),
                        base.prediction_cutoff.isoformat(), self._scenario, self._seed]),
                match_id=base.match_id, prediction_cutoff=base.prediction_cutoff,
                information_mode=base.information_mode, fixture=base.fixture,
                base_prior=base.base_prior, prior_basis=base.prior_basis,
                mask={g: 'UNACQUIRED' for g in ('strength', 'recent_form', 'rolling_xg')},
                budget=BudgetState(limits={'money': .10, 'requests': 4}),
                experiment_role=base.experiment_role, catalog_version=self.configuration()['catalog_version'],
                cost_config_version='dynamic-cost-v1:' + digest([self._scenario, self._seed]))
            self._receipts, self._attempts, self._published = {}, {}, {}
            self._dependency_cache = {}
            self._clock = base.prediction_cutoff
            return self.visible_state()

    def start(self):
        return self.reset()

    def visible_state(self):
        with self._lock:
            return self._state.model_copy(deep=True)

    def _base_group(self, group, *, provider='understat', dependency='understat_league_response'):
        value = self._base._project(group).model_copy(deep=True)
        value.source_id = provider + ':' + self._state.match_id
        value.provider = provider
        value.dependency_key = dependency
        value.quality_status = {
            'SOURCE_MISSING': 'source_missing', 'PARTIAL': 'partial',
            'STALE': 'stale', 'INVALID': 'provider_error',
        }.get(value.quality, 'valid')
        # Synthetic freshness window starts at the episode's public cutoff so
        # the first acquisition is valid and the owner can advance the clock
        # deterministically into an expired state.
        value.expires_at = max(value.available_at, self._state.prediction_cutoff) + timedelta(hours=1)
        value.freshness_age = max(0.0, (self._clock - value.available_at).total_seconds())
        value.content_hash = digest([group, value.values])
        value.version = 'dynamic-group-v1:' + digest([value.content_hash, provider, dependency])
        return value

    def _cache_dependency(self, dependency, *, provider='understat'):
        """Materialize a synthetic shared response in owner memory only."""
        if dependency not in self._dependency_cache:
            self._dependency_cache[dependency] = {
                group: self._base_group(group, provider=provider, dependency=dependency)
                for group in ('strength', 'recent_form', 'rolling_xg')
            }
        return self._dependency_cache[dependency]

    def _conflict_group(self, first, second):
        values = {key: None for key in first.values}
        combined = digest([first.content_hash, second.content_hash])
        return first.model_copy(update={
            'values': values, 'quality': 'CONFLICT', 'quality_status': 'conflict',
            'source_id': first.source_id + '|conflict|' + second.source_id,
            'provider': first.provider + '|conflict|' + second.provider,
            'content_hash': combined, 'dependency_key': first.dependency_key + '+' + second.dependency_key,
            'version': 'dynamic-conflict-v1:' + combined,
            'missing_fields': list(values),
            'evidence_refs': sorted(set(first.evidence_refs + second.evidence_refs + ['conflict:' + combined]))})

    def _failure_group(self, group, quality_status):
        source = self._base_group(group)
        return source.model_copy(update={'values': {}, 'quality': 'SOURCE_MISSING' if quality_status == 'source_missing' else 'PROVIDER_ERROR',
            'quality_status': quality_status, 'missing_fields': ['all'], 'content_hash': '',
            'version': 'dynamic-failure-v1:' + quality_status})

    def advance_time(self, at):
        """Owner clock injection; the public state only receives the resulting stale mask."""
        with self._lock:
            current = utc(at)
            if current < self._clock:
                raise ValueError('Dynamic clock cannot move backwards')
            if current >= self._state.fixture.kickoff:
                raise ValueError('Dynamic prematch clock must precede kickoff')
            self._clock = current
            self._state.prediction_cutoff = current
            for gid, evidence in list(self._state.groups.items()):
                if evidence.expires_at and current >= evidence.expires_at:
                    self._state.groups[gid] = evidence.model_copy(update={
                        'quality': 'STALE', 'quality_status': 'stale',
                        'freshness_age': max(0.0, (current - evidence.available_at).total_seconds())})
                    self._state.mask[gid] = 'STALE'
            self._state.state_version += 1
            return self.visible_state()

    def acquire(self, action_id, request_id):
        with self._lock:
            if request_id in self._receipts:
                old = self._receipts[request_id]
                if old.action_id != action_id:
                    raise ValueError('Request ID belongs to another action')
                return old.model_copy(update={'charged_cost': zero_cost(), 'repeated_request': True, 'attempts': 0}, deep=True)
            spec = next((v for v in self._catalog if v.action_id == action_id), None)
            if spec is None:
                raise ValueError('Unknown dynamic action')
            cost = spec.expected_cost.model_copy(deep=True)
            if any(self._state.budget.spent.get(unit, 0) + getattr(cost, unit).amount > limit + 1e-12
                   for unit, limit in self._state.budget.limits.items()):
                result = AcquisitionResult(episode_id=self._state.episode_id, action_id=action_id,
                    request_id=request_id, status='BUDGET_EXCEEDED', charged_cost=zero_cost(),
                    sequence=len(self._state.history)+1, attempts=0, reason='dynamic_budget_exhausted',
                    resulting_cutoff=self._state.prediction_cutoff, synthetic_perturbation=True)
                self._receipts[request_id] = result.model_copy(deep=True)
                return result
            attempt = self._attempts.get(action_id, 0)
            self._attempts[action_id] = attempt + 1
            status, reason, revealed, masks = 'ACQUIRED', None, {}, {}
            # A cache-view action projects a group from an already queried
            # physical response.  It is legal only after that dependency has
            # been populated; no hidden full response is returned.
            if (self._scenario == 'shared_response' and spec.cache_policy == 'cache_view'
                    and spec.dependency_key not in self._dependency_cache):
                status, reason = 'NO_DATA', 'shared_response_not_queried'
                for group_id in spec.group_ids:
                    masks[group_id] = 'SOURCE_MISSING'
                    revealed[group_id] = self._failure_group(group_id, 'source_missing')
            elif (self._scenario == 'shared_response' and spec.cache_policy == 'cache_view'):
                cached = self._dependency_cache[spec.dependency_key]
                for group_id in spec.group_ids:
                    group = cached[group_id].model_copy(deep=True)
                    self._state.groups[group_id] = group
                    self._state.mask[group_id] = 'OBSERVED'
                    revealed[group_id] = group
                cost = zero_cost()
                status, reason = 'CACHE_HIT', 'shared_response_projection'
            if self._scenario == 'no_data_then_available' and action_id == 'first.no_data' and attempt == 0:
                status, reason = 'NO_DATA', 'synthetic_source_missing'
                masks['strength'] = 'SOURCE_MISSING'
                revealed['strength'] = self._failure_group('strength', 'source_missing')
            elif self._scenario == 'provider_error' and action_id == 'provider.error' and attempt == 0:
                status, reason = 'FAILED', 'synthetic_provider_error'
                masks['strength'] = 'FAILED'
                revealed['strength'] = self._failure_group('strength', 'provider_error')
            elif status != 'CACHE_HIT' and self._scenario == 'shared_response' and action_id == 'shared.query.strength':
                cached = self._cache_dependency(spec.dependency_key)
                group = cached['strength'].model_copy(deep=True)
                self._state.groups['strength'] = group
                self._state.mask['strength'] = 'OBSERVED'
                revealed['strength'] = group
            elif status != 'CACHE_HIT' and self._scenario == 'shared_response' and action_id == 'shared.refresh.rolling_xg':
                cached = self._cache_dependency(spec.dependency_key)
                # A refresh is a new version of the same response.  Values are
                # deliberately unchanged, so downstream tests can separate a
                # new observation from a new signal.
                group = cached['rolling_xg'].model_copy(deep=True, update={
                    'version': cached['rolling_xg'].version + ':refresh',
                    'quality_status': 'valid', 'quality': 'VALIDATED',
                    'available_at': self._clock, 'times': cached['rolling_xg'].times.model_copy(update={'feature_available_at': self._clock, 'basis': 'synthetic_refresh_clock_over_frozen_features'}), 'expires_at': self._clock + timedelta(hours=1),
                    'freshness_age': 0.0})
                self._dependency_cache[spec.dependency_key]['rolling_xg'] = group
                self._state.groups['rolling_xg'] = group
                self._state.mask['rolling_xg'] = 'OBSERVED'
                revealed['rolling_xg'] = group
            elif status != 'CACHE_HIT' and self._scenario == 'refresh_cycle' and action_id == 'refresh.refresh.strength':
                group = self._base_group('strength', dependency=spec.dependency_key).model_copy(update={
                    'available_at': self._clock, 'times': self._base_group('strength').times.model_copy(update={'feature_available_at': self._clock, 'basis': 'synthetic_refresh_clock_over_frozen_features'}), 'expires_at': self._clock + timedelta(hours=1),
                    'freshness_age': 0.0, 'version': self._base_group('strength', dependency=spec.dependency_key).version + ':refresh'})
                self._state.groups['strength'] = group
                self._state.mask['strength'] = 'OBSERVED'
                revealed['strength'] = group
            elif status != 'CACHE_HIT' and self._scenario in ('duplicate', 'no_change') and attempt == 0 and action_id.startswith('primary.'):
                group = self._base_group('strength')
                self._state.groups['strength'] = group
                self._state.mask['strength'] = 'OBSERVED'
                revealed['strength'] = group
            elif self._scenario == 'conflict' and action_id == 'provider_a.strength':
                group = self._base_group('strength', provider='provider_a', dependency='provider_a:strength')
                self._state.groups['strength'] = group
                self._state.mask['strength'] = 'OBSERVED'
                revealed['strength'] = group
            elif self._scenario == 'conflict' and action_id == 'provider_b.strength':
                first = self._state.groups.get('strength') or self._base_group('strength', provider='provider_a', dependency='provider_a:strength')
                second = self._base_group('strength', provider='provider_b', dependency='provider_b:strength')
                second.values = {k: (v + 7.0 if isinstance(v, (int, float)) else v) for k, v in second.values.items()}
                conflict = self._conflict_group(first, second)
                self._state.groups['strength'] = conflict
                # Keep the conflict evidence auditable in the owner state, but
                # prevent policy models from treating it as a valid observation.
                self._state.mask['strength'] = 'FAILED'
                revealed['strength'] = conflict
                status, reason = 'FAILED', 'SOURCE_CONFLICT'
            elif self._scenario in ('duplicate', 'no_change') and action_id in ('duplicate.strength', 'refresh.no_change'):
                group = self._base_group('strength')
                group = group.model_copy(update={'quality': 'DUPLICATE' if self._scenario == 'duplicate' else 'NO_CHANGE',
                    'quality_status': 'duplicate' if self._scenario == 'duplicate' else 'no_change',
                    'evidence_refs': group.evidence_refs + ['duplicate_of:' + self._state.groups['strength'].content_hash]})
                revealed['strength'] = group
                status, reason = 'UNCHANGED', 'DUPLICATE_CONTENT' if self._scenario == 'duplicate' else 'CONTENT_NO_CHANGE'
            elif status not in ('CACHE_HIT', 'NO_DATA', 'FAILED'):
                for group_id in spec.group_ids:
                    group = self._base_group(group_id)
                    self._state.groups[group_id] = group
                    self._state.mask[group_id] = 'OBSERVED'
                    revealed[group_id] = group
            self._state.mask.update(masks)
            for key in ('money', 'requests', 'latency_ms', 'tokens'):
                part = getattr(cost, key)
                if part.amount is not None:
                    self._state.budget.spent[key] += part.amount
            self._state.state_version += 1
            result = AcquisitionResult(episode_id=self._state.episode_id, action_id=action_id,
                request_id=request_id, status=status, revealed_groups=revealed, mask_updates=masks,
                charged_cost=cost, sequence=len(self._state.history) + 1,
                evidence_refs=sorted({ref for group in revealed.values() for ref in group.evidence_refs}),
                content_changed=status == 'ACQUIRED', attempts=1, reason=reason, synthetic_perturbation=True,
                resulting_cutoff=self._state.prediction_cutoff)
            self._state.history.append(ActionReceipt(action_id=action_id, request_id=request_id,
                status=status, group_ids=spec.group_ids, charged_cost=cost, sequence=result.sequence, synthetic_perturbation=True))
            self._receipts[request_id] = result.model_copy(deep=True)
            return result.model_copy(deep=True)
