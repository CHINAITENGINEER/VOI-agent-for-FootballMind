"""Current operational acquisition over the existing bounded RefreshService.

The owner supplies a previously observed fixture. Refresh never moves a cutoff
implicitly; advance_cutoff publishes only evidence actually available by then.
"""
from datetime import datetime, timezone
import json
from threading import RLock
import time
from uuid import uuid5, NAMESPACE_URL

from football_spider.models.contracts import Observation
from football_spider.normalization.sources import normalize_spftll
from football_spider.quality.rules import validate_batch
from football_spider.storage.raw_archive import digest
from .contracts import (ActionSpec, AcquisitionResult, ActionReceipt, BudgetState,
                        CostVector, CostComponent, EvidenceGroup, EvidenceTimes, Fixture, VisibleState)
from .costs import zero_cost, DIMENSIONS
from .frozen import utc


class LiveEvidenceEnvironment:
    def __init__(self, refresh_service, *, fixture, fixture_observed_at, max_requests=2, timeout=45):
        if type(max_requests) is not int or not 1 <= max_requests <= 2 or not 0 < timeout <= 60:
            raise ValueError('Live episode permits at most two attempts and 60 seconds per action')
        self._service = refresh_service
        self._fixture = Fixture.model_validate(fixture)
        self._known = utc(fixture_observed_at)
        self._limit, self._timeout = max_requests, timeout
        self._lock = RLock()
        self._state = None
        self._receipts = {}
        self._pending = {}
        self._used = 0
        self._latest_kickoff = self._fixture.kickoff
        self._latest_status = 'scheduled'

    def catalog(self):
        return [ActionSpec(action_id=aid, source='spftll', tool='bounded_refresh', group_ids=groups,
            expected_cost=CostVector(requests=CostComponent(amount=1, basis='estimated', unit='HTTP_request')),
            cost_scope='real_external_request', billing_key='spftll:' + dep, dependency_key=dep,
            cache_policy='refresh', refresh_supported=True,
            prerequisites=['operational_prematch_fixture', 'explicit_new_cutoff_to_publish_new_observations'],
            refresh_window='bounded_current_observation',
            cost_range={'requests': {'min': 0.0, 'max': float(self._limit)}},
            interaction_key='spftll_response:' + dep)
            for aid, groups, dep in [('fixture_history', ['fixture_refresh', 'history'], 'GetInfo'),
                                    ('odds', ['odds'], 'GetMoreSpInfo')]]

    def start(self, prediction_cutoff):
        with self._lock:
            if self._state is not None:
                raise ValueError('One bounded live episode per owner instance; use advance_cutoff')
            cutoff = utc(prediction_cutoff)
            if not self._known <= cutoff < self._fixture.kickoff:
                raise ValueError('Known operational prematch fixture required')
            self._state = VisibleState(episode_id='live:' + digest([self._fixture.model_dump(mode='json'), cutoff.isoformat()]),
                match_id=self._fixture.match_id, prediction_cutoff=cutoff, information_mode='operational',
                fixture=self._fixture, base_prior={'home': 1/3, 'draw': 1/3, 'away': 1/3},
                mask={g: 'UNACQUIRED' for g in ('fixture_refresh', 'history', 'odds')},
                budget=BudgetState(limits={'requests': self._limit}), experiment_role='prospective',
                catalog_version='live-catalog-v1:' + digest([s.model_dump(mode='json') for s in self.catalog()]),
                cost_config_version='real-fees-unknown-v1')
            return self.visible_state()

    def visible_state(self):
        with self._lock:
            if self._state is None:
                raise ValueError('Start a live episode first')
            return self._state.model_copy(deep=True)

    def _ingested(self, observation_id):
        # Actual warehouse timestamp when available, never substitute kickoff.
        if not self._service.config.database_url:
            return None
        from football_spider.storage.postgres import Repository
        repo = Repository(self._service.config.database_url)
        try:
            row = repo.conn.execute('SELECT ingested_at AS t FROM fm.raw_observations WHERE id=%s',
                                   (observation_id,)).fetchone()
            return row['t'] if row else None
        finally:
            repo.close()

    def _project(self, report, group_ids):
        output = {}
        sid = self._fixture.match_id.removeprefix('spftll:match:').removeprefix('spftll:')
        reported_id = str(report.get('match_id', '')).removeprefix('spftll:match:').removeprefix('spftll:')
        if reported_id != sid or report.get('status') not in ('OK', 'PARTIAL', 'NO_DATA', 'NEEDS_DATABASE'):
            raise ValueError('Incomplete operation or response identity conflict')
        processing = {p['observation_id']: p for p in report.get('processing', [])}
        for raw in report.get('observations', []):
            obs = Observation.model_validate(raw)
            if obs.http_status != 200 or obs.error_class:
                continue
            if obs.source != 'spftll' or obs.endpoint.rsplit('/', 1)[-1] not in ('GetInfo', 'GetMoreSpInfo'):
                raise ValueError('Unexpected source or endpoint')
            processed = processing.get(obs.id)
            if processed and processed.get('status') not in ('OK', 'NO_DATA', 'SKIPPED'):
                raise ValueError('Observation is failed or quarantined')
            payload = json.loads(self._service.archive.read_body(obs))
            data = payload.get('data')
            root = data if obs.endpoint.endswith('GetInfo') else (data or {}).get('footballMoreSpInfo')
            if root and str(root.get('matchId')) != sid:
                raise ValueError('Response root belongs to a different match')
            batch = normalize_spftll(obs, payload)
            invalid = {i.record_key for i in validate_batch(batch, obs.observed_at) if i.severity == 'error'}
            if sid in invalid:
                raise ValueError('Target quality quarantine')
            ingested = self._ingested(obs.id)
            # finished_at is a conservative publication/processing-complete bound;
            # it is NOT called the actual database ingestion timestamp.
            available = max(obs.observed_at, utc(report['finished_at']), ingested or obs.observed_at)
            times = EvidenceTimes(observed_at=obs.observed_at, ingested_at=ingested,
                operational_available_at=available, feature_available_at=available,
                basis='actual_observation_plus_refresh_completion_bound; warehouse_ingestion_nullable')
            def group(gid, values, quality='VALIDATED', refs=()):
                content = digest([gid, values])
                candidate = EvidenceGroup(group_id=gid, values=values, source='spftll', times=times,
                    quality=quality, evidence_refs=['observation:' + obs.id, 'content:' + content, *refs],
                    version='live-group-v1:' + digest([content, obs.id, times.model_dump(mode='json')]),
                    cache_reused=bool(report.get('cached')), source_id='spftll:' + sid,
                    provider='spftll', observed_at=obs.observed_at, available_at=available,
                    freshness_age=max(0.0, (utc(report['finished_at']) - available).total_seconds()),
                    quality_status={'VALIDATED': 'valid', 'PARTIAL': 'partial',
                                    'SOURCE_MISSING': 'source_missing'}.get(quality, quality.lower()),
                    content_hash=content, dependency_key='spftll:' + obs.endpoint.rsplit('/', 1)[-1])
                prior = output.get(gid)
                if prior is None or candidate.times.feature_available_at >= prior.times.feature_available_at:
                    output[gid] = candidate
            target = next((m for m in batch.matches if m.source_id == sid), None)
            if target and 'fixture_refresh' in group_ids:
                self._latest_kickoff = target.kickoff
                self._latest_status = target.status
                group('fixture_refresh', {'kickoff': target.kickoff.isoformat(),
                    'home_team_name': target.home.name, 'away_team_name': target.away.name,
                    'competition_name': target.competition.name, 'status': target.status,
                    'score_period': target.score_period})  # No target score ever projected.
                history = [m for m in batch.matches if m.source_id != sid and m.source_id not in invalid
                    and m.kickoff < min(target.kickoff, obs.observed_at) and m.status == 'finished'
                    and m.score_period == 'regulation' and m.score.home is not None and m.score.away is not None]
                values = {'coverage': 'selected_source_history_not_complete_league',
                          'missing_reason': 'unverified_phases_and_incomplete_scores_excluded'}
                refs = []
                for side, team in [('home', target.home), ('away', target.away)]:
                    recent = sorted([m for m in history if team.source_id in (m.home.source_id, m.away.source_id)],
                                    key=lambda m: (m.kickoff, m.source_id))[-5:]
                    values[side + '_matches'] = len(recent)
                    values[side + '_goals_for'] = sum(m.score.home if m.home.source_id == team.source_id else m.score.away for m in recent) if recent else None
                    values[side + '_goals_against'] = sum(m.score.away if m.home.source_id == team.source_id else m.score.home for m in recent) if recent else None
                    refs += ['source_match:spftll:' + m.source_id for m in recent]
                group('history', values, 'PARTIAL' if history else 'SOURCE_MISSING', sorted(set(refs)))
            if 'odds' in group_ids:
                quotes = [q for q in batch.quotes if q.source_match_id == sid and q.market_type == '1x2' and q.period == 'regulation']
                if len(quotes) == 3 and {q.selection for q in quotes} == {'home', 'draw', 'away'}:
                    if any(q.source_timestamp and q.source_timestamp > available for q in quotes):
                        raise ValueError('Future quoted timestamp')
                    group('odds', {**{q.selection: q.decimal_price for q in quotes}, 'market_type': '1x2',
                        'period': 'regulation', 'bookmaker': None, 'provider': 'spftll',
                        'settlement_rule': quotes[0].settlement_rule, 'quote_time_semantics': 'current_observation_only'})
        return output

    def acquire(self, action_id, request_id):
        with self._lock:
            s = self.visible_state()
            if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
                raise ValueError('Bounded nonempty request ID required')
            if request_id in self._receipts:
                old = self._receipts[request_id]
                if old.action_id != action_id:
                    raise ValueError('Request ID belongs to another action')
                return old.model_copy(deep=True, update={'charged_cost': zero_cost(), 'repeated_request': True, 'attempts': 0})
            spec = next((v for v in self.catalog() if v.action_id == action_id), None)
            if spec is None:
                raise ValueError('Unknown live action')
            status, cost, reason, refs, attempts = 'BUDGET_EXCEEDED', zero_cost(), 'NO_REQUEST_EXECUTED', [], 0
            suggested, revealed, masks = None, {}, {}
            statistics, processing_status = {}, None
            if self._used < self._limit:
                operation = str(uuid5(NAMESPACE_URL, s.episode_id + ':' + action_id + ':' + request_id))
                started = time.perf_counter()
                report = self._service.refresh(s.match_id, topics=('fixture', 'history') if action_id == 'fixture_history' else ('odds',),
                    request_id=operation, max_requests=1, timeout=self._timeout)
                elapsed_ms = (time.perf_counter() - started) * 1000
                attempts = report.get('requests_this_call', 0)
                self._used += attempts
                cost = CostVector(requests=CostComponent(amount=attempts, basis='known', unit='HTTP_request'),
                    latency_ms=CostComponent(amount=elapsed_ms, basis='known', unit='ms'),
                    tokens=CostComponent(amount=0, basis='known', unit='token'))
                refs = ['observation:' + o['id'] for o in report.get('observations', [])]
                network = report.get('network_status', 'FAILED')
                statistics = {k: v for k, v in report.get('request_statistics', {}).items() if isinstance(v, (str, int, float)) or v is None}
                processing_status = report.get('status')
                status, reason = ('NO_DATA' if network == 'NO_DATA' else 'FAILED'), report.get('error') or network
                if any('TIMEOUT' in error.get('category', '').upper() for error in report.get('errors', [])):
                    status, reason = 'TIMEOUT', 'BOUNDED_REQUEST_TIMEOUT'
                try:
                    projected = (self._project(report, spec.group_ids) if any(
                        o.get('http_status') == 200 and not o.get('error_class')
                        for o in report.get('observations', [])) else {})
                    if projected:
                        for gid, evidence in projected.items():
                            prior = self._pending.get(gid)
                            if prior is None or evidence.times.feature_available_at >= prior.times.feature_available_at:
                                self._pending[gid] = evidence
                        suggested = max(v.times.feature_available_at for v in projected.values())
                        if suggested > s.prediction_cutoff:
                            status, reason = 'NEEDS_NEW_CUTOFF', 'NEW_OBSERVATION_NOT_IN_FIXED_CUTOFF'
                        else:
                            revealed = {g: v for g, v in projected.items() if v.quality != 'SOURCE_MISSING'
                                        and (g not in s.groups or
                                             (v.times.feature_available_at > s.groups[g].times.feature_available_at) or
                                             (v.times.feature_available_at == s.groups[g].times.feature_available_at
                                              and v.values != s.groups[g].values))}
                            masks = {g: 'SOURCE_MISSING' if v.quality == 'SOURCE_MISSING' else 'OBSERVED' for g, v in projected.items()}
                            status, reason = ('ACQUIRED' if revealed else 'UNCHANGED'), report.get('database', {}).get('status')
                    elif network in ('OK', 'PARTIAL'):
                        status, reason = 'BAD_DATA', 'NO_VALID_REQUESTED_PROJECTION'
                except Exception:
                    # HTTP has already happened. Parsing/archive/DB read failures
                    # must retain its receipt and costs, not trigger blind retry.
                    status, reason = 'BAD_DATA', 'NORMALIZATION_OR_QUALITY_FAILED'
                if status in ('NO_DATA', 'FAILED', 'BAD_DATA', 'TIMEOUT'):
                    masks = {g: 'SOURCE_MISSING' if status == 'NO_DATA' else 'FAILED' for g in spec.group_ids if self._state.mask[g] != 'OBSERVED'}
            self._state.groups.update(revealed)
            self._state.mask.update(masks)
            for k in DIMENSIONS:
                part = getattr(cost, k)
                if part.amount is None:
                    if k not in self._state.budget.unknown_components:
                        self._state.budget.unknown_components.append(k)
                else:
                    self._state.budget.spent[k] += part.amount
            if status == 'ACQUIRED' and not revealed:
                status, reason = 'UNCHANGED', 'CONTENT_UNCHANGED_OR_OLDER_OBSERVATION'
            result = AcquisitionResult(episode_id=s.episode_id, action_id=action_id, request_id=request_id,
                status=status, charged_cost=cost, sequence=len(s.history) + 1, revealed_groups=revealed,
                mask_updates=masks, resulting_cutoff=s.prediction_cutoff, suggested_new_cutoff=suggested,
                reason=reason, evidence_refs=refs, attempts=attempts, content_changed=bool(revealed),
                request_statistics=statistics, processing_status=processing_status)
            self._state.state_version += 1
            self._state.history.append(ActionReceipt(action_id=action_id, request_id=request_id,
                status=status, group_ids=spec.group_ids, charged_cost=cost, sequence=result.sequence))
            self._receipts[request_id] = result.model_copy(deep=True)
            return result.model_copy(deep=True)

    def advance_cutoff(self, new_cutoff):
        with self._lock:
            s = self.visible_state()
            cutoff = utc(new_cutoff)
            if (not s.prediction_cutoff < cutoff < min(self._fixture.kickoff, self._latest_kickoff)
                    or cutoff > datetime.now(timezone.utc) or self._latest_status != 'scheduled'):
                raise ValueError('Advance explicitly within current operational prematch window')
            self._state.prediction_cutoff = cutoff
            self._state.state_version += 1
            for gid, evidence in self._pending.items():
                if evidence.times.feature_available_at <= cutoff:
                    self._state.mask[gid] = 'SOURCE_MISSING' if evidence.quality == 'SOURCE_MISSING' else 'OBSERVED'
                    if evidence.quality != 'SOURCE_MISSING':
                        self._state.groups[gid] = evidence
            return self.visible_state()
