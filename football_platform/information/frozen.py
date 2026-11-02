"""Owner-side frozen environment. Only serialized public methods reach a policy.

This is an information-flow boundary, not a Python/filesystem security sandbox.
The experiment runner retains the complete bundle and labels, not the policy.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
from threading import RLock

from football_spider.storage.raw_archive import digest
from .contracts import (ActionSpec, AcquisitionResult, ActionReceipt, BudgetState,
                        EvidenceGroup, EvidenceTimes, Fixture, VisibleState)
from .costs import ExperimentConfig, DIMENSIONS, zero_cost

GROUPS = ('strength', 'recent_form', 'rolling_xg')


def utc(value):
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Explicit timezone required')
    return result.astimezone(timezone.utc)


def completed_bundle(directory):
    """Verify publication, paths, checksums and row counts before private loading."""
    root = Path(directory).resolve()
    body = (root / 'manifest.json').read_bytes()
    marker = json.loads((root / 'COMPLETE.json').read_bytes())
    if hashlib.sha256(body).hexdigest() != marker['manifest_sha256']:
        raise ValueError('Completed manifest mismatch')
    manifest = json.loads(body)
    if manifest.get('schema') != 'footballmind.platform.history-export.v1':
        raise ValueError('Unsupported completed export')
    records = {}
    for name, item in manifest['files'].items():
        file = (root / item['path']).resolve()
        if not file.is_relative_to(root):
            raise ValueError('Export file is outside publication')
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Completed file mismatch')
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        if len(rows) != item['row_count']:
            raise ValueError('Completed row count mismatch')
        records[name] = rows
    return manifest, records


class FrozenEvidenceEnvironment:
    """An episode, with no public full-feature, target-label or raw-reader API."""

    @classmethod
    def from_export(cls, export_dir, config=None, *, perturbations=None):
        manifest, records = completed_bundle(export_dir)
        return cls(manifest, records, config or ExperimentConfig(), perturbations=perturbations)

    def __init__(self, manifest, records, config, *, perturbations=None):
        self._config = ExperimentConfig.model_validate(config).model_copy(deep=True)
        self._manifest = manifest
        self._matches = {r['match_id']: r for r in records['matches']}
        self._features = {}
        for row in records['features']:
            key = (row['match_id'], utc(row['prediction_cutoff']), row['information_mode'])
            if key in self._features:
                raise ValueError('Ambiguous feature snapshot')
            self._features[key] = row
        self._faults = dict(perturbations or {})  # owner-only action -> ordered outcomes
        permitted = {'NO_DATA', 'TIMEOUT', 'BAD_DATA', 'STALE', 'UNCHANGED'}
        if any(not isinstance(v, list) or not set(v) <= permitted for v in self._faults.values()):
            raise ValueError('Invalid explicit synthetic perturbations')
        self._lock = RLock()
        self._state = None
        self._receipts = {}
        self._attempts = {}

    def catalog(self):
        entries = []
        for aid, groups in self._config.bindings.items():
            entries.append(ActionSpec(action_id=aid, source='understat', tool='frozen_evidence_projection',
                group_ids=groups, expected_cost=self._config.cost(aid),
                cost_scope='cached_projection' if self._config.mode == 'cache_views' else 'simulated_reveal',
                billing_key=('cached:' if self._config.mode == 'cache_views' else 'simulated-reveal:') + aid,
                dependency_key='understat_league_response', cache_policy='shared_response',
                refresh_window='frozen_complete_snapshot',
                cost_range={'money': {'min': self._config.cost(aid).money.amount,
                                      'max': self._config.cost(aid).money.amount}},
                interaction_key='understat_response_projection'))
        for group in ('weather', 'injuries'):
            entries.append(ActionSpec(action_id='unavailable.' + group, source='not_connected',
                tool='not_implemented', group_ids=[group], capability_status='NOT_IMPLEMENTED',
                expected_cost=zero_cost(), cost_scope='cached_projection', billing_key='none',
                dependency_key='not_connected', cache_policy='cache_view'))
        return entries

    def configuration(self):
        """Public manifest summary, intentionally independent of hidden content."""
        return {'protocol_version': 'information-v1', 'environment': 'frozen_evidence',
                'source': self._manifest['source'], 'season': self._manifest['season'],
                'information_mode': self._manifest['information_mode'],
                'feature_version': self._manifest['feature_version'],
                'cost_config_version': self._config.version,
                'experiment_role': self._config.experiment_role,
                'physical_requests': 0, 'confidence_calibration': 'NOT_CALIBRATED'}

    def list_fixtures(self):
        """Only fixture identity and committed prediction cutoffs, no coverage flags."""
        return [{'fixture': Fixture(**{k: self._matches[mid][k] for k in Fixture.model_fields
                                      if k in self._matches[mid]}).model_dump(mode='json'),
                 'match_id': mid, 'prediction_cutoff': cutoff.isoformat(), 'information_mode': mode}
                for mid, cutoff, mode in sorted(self._features, key=lambda k: (k[1], k[0], k[2]))]

    def start(self, match_id, prediction_cutoff, information_mode='historical_reconstruction'):
        with self._lock:
            cutoff = utc(prediction_cutoff)
            row = self._matches.get(match_id)
            snapshot = self._features.get((match_id, cutoff, information_mode))
            if row is None or snapshot is None:
                raise ValueError('No committed snapshot for requested match, cutoff and mode')
            fixture = Fixture(**{k: row[k] for k in Fixture.model_fields if k in row})
            if cutoff >= fixture.kickoff:
                raise ValueError('Prematch cutoff required')
            known = (utc(row['fixture_available_at']) if information_mode == 'historical_reconstruction'
                     else max(utc(row['observed_at']), utc(row['ingested_at'])))
            if known > cutoff:
                raise ValueError('Fixture unavailable at requested cutoff')
            if row.get('source_published_at') and utc(row['source_published_at']) > cutoff:
                raise ValueError('Fixture unavailable: source publication is after cutoff')
            catalog_version = 'catalog-v1:' + digest([a.model_dump(mode='json') for a in self.catalog()])
            # No hidden-value, source-payload, label or full-manifest hash in public IDs.
            identity = [fixture.model_dump(mode='json'), cutoff.isoformat(), information_mode,
                        self._config.version, catalog_version]
            self._state = VisibleState(episode_id='episode:' + digest(identity), match_id=match_id,
                prediction_cutoff=cutoff, information_mode=information_mode, fixture=fixture,
                base_prior=self._config.base_prior, prior_basis='declared_experiment_prior_not_fitted',
                mask={g: 'UNACQUIRED' for g in GROUPS},
                budget=BudgetState(limits=self._config.budget), experiment_role=self._config.experiment_role,
                catalog_version=catalog_version, cost_config_version=self._config.version)
            self._snapshot = snapshot
            self._receipts = {}
            self._attempts = {}
            return self.visible_state()

    reset = start

    def visible_state(self):
        with self._lock:
            if self._state is None:
                raise ValueError('Start an episode first')
            return self._state.model_copy(deep=True)

    def _project(self, group):
        state, f = self._state, self._snapshot['features']
        m = self._snapshot['manifest']
        row = self._matches[state.match_id]
        if group == 'strength':
            values = {'home_elo': f['elo']['home'], 'away_elo': f['elo']['away'],
                      'elo_difference': f['elo']['difference']}
            inputs = m['history_inputs']
        elif group == 'recent_form':
            values = {f'{side}_rolling_{key}': f.get(f'{side}_rolling_{key}')
                      for side in ('home', 'away') for key in ('goals_for', 'goals_against', 'points', 'matches')}
            values.update({side + '_rest_days': f[side + '_form'].get('rest_days') for side in ('home', 'away')})
            versions = set(f['home_form']['version_ids'] + f['away_form']['version_ids'])
            history_versions = {r['version_id'] for r in m['history_inputs']}
            if not versions <= history_versions or m.get('target_version') in versions:
                raise ValueError('Form version is outside historical lineage')
            inputs = [r for r in m['history_inputs'] if r['version_id'] in versions]
        else:
            values = {f'{side}_rolling_{key}': f.get(f'{side}_rolling_{key}')
                      for side in ('home', 'away') for key in ('xg_for', 'xg_against', 'xg_difference')}
            values['provider'] = 'understat'
            values['definition_version'] = 'understat-match-xg-v1'
            by_mid = {r['match_id']: r for r in m['history_inputs']}
            for item in m['xg_inputs']:
                linked = by_mid.get(item['match_id'])
                if (item['match_id'] == state.match_id or linked is None or
                        utc(linked['kickoff']) >= state.prediction_cutoff or
                        utc(linked['result_available_at']) > state.prediction_cutoff):
                    raise ValueError('XG input is outside eligible historical lineage')
            mids = {r['match_id'] for r in m['xg_inputs']}
            inputs = [r for r in m['history_inputs'] if r['match_id'] in mids]
        if any(r['match_id'] == state.match_id or utc(r['kickoff']) >= state.prediction_cutoff for r in inputs):
            raise ValueError('Target or future result entered lineage')
        if any(r.get('source_published_at') and utc(r['source_published_at']) > state.prediction_cutoff for r in inputs):
            raise ValueError('Source publication is after cutoff')
        observed = max([utc(row['observed_at'])] + [utc(r['observed_at']) for r in inputs])
        ingested = max([utc(row['ingested_at'])] + [utc(r['ingested_at']) for r in inputs])
        times = [utc(row['fixture_available_at'])] if state.information_mode == 'historical_reconstruction' else [max(observed, ingested)]
        times += [utc(r['result_available_at']) for r in inputs]
        if group == 'rolling_xg':
            times += [utc(r['xg_available_at']) for r in m['xg_inputs']]
        available = max(times)
        if available > state.prediction_cutoff or (state.information_mode == 'operational' and max(observed, ingested) > state.prediction_cutoff):
            raise ValueError('Evidence violates source-time contract')
        refs = sorted({'observation:' + row['observation_id']} |
                      {'match_version:' + r['version_id'] for r in inputs})
        if group == 'rolling_xg':
            refs += sorted({'statistic:' + r['statistic_id'] for r in m['xg_inputs']})
        missing = [k for k, v in values.items() if v is None]
        meaningful = [v for k, v in values.items() if k not in ('provider', 'definition_version') and not k.endswith('_matches')]
        quality = 'SOURCE_MISSING' if all(v is None for v in meaningful) else 'PARTIAL' if missing else 'VALIDATED'
        assumptions = m.get('assumptions') or {}
        reconstruction_basis = ('explicit_reconstruction_result_' + str(assumptions.get('result_delay_hours'))
            + 'h_xg_' + str(assumptions.get('xg_delay_hours')) + 'h_snapshot_revision_risk')
        stamp = EvidenceTimes(observed_at=observed, ingested_at=ingested,
            operational_available_at=max(observed, ingested), feature_available_at=available,
            basis=reconstruction_basis if state.information_mode == 'historical_reconstruction' else 'actual_observation_and_ingestion')
        content_hash = digest([group, values])
        version = 'group-v1:' + digest([group, values, stamp.model_dump(mode='json'), refs])
        quality_status = 'source_missing' if quality == 'SOURCE_MISSING' else 'partial' if quality == 'PARTIAL' else 'valid'
        age = max(0.0, (state.prediction_cutoff - available).total_seconds())
        return EvidenceGroup(group_id=group, values=values, source='understat', times=stamp,
            quality=quality, missing_fields=missing, evidence_refs=refs, version=version,
            source_id='understat:EPL:' + str(self._manifest.get('season')),
            provider='understat', observed_at=observed, available_at=available,
            freshness_age=age, quality_status=quality_status, content_hash=content_hash,
            dependency_key='understat_league_response')

    def acquire(self, action_id, request_id):
        with self._lock:
            self.visible_state()
            if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
                raise ValueError('Nonempty bounded request_id required')
            if request_id in self._receipts:
                previous = self._receipts[request_id]
                if previous.action_id != action_id:
                    raise ValueError('Idempotency key belongs to another action')
                return previous.model_copy(deep=True, update={'charged_cost': zero_cost(), 'repeated_request': True, 'attempts': 0})
            spec = next((a for a in self.catalog() if a.action_id == action_id), None)
            if spec is None:
                raise ValueError('Action is not in the public catalog')
            s = self._state
            cost = self._config.cost(action_id)
            revealed, masks, reason = {}, {}, None
            synthetic = False
            if spec.capability_status != 'AVAILABLE':
                status, cost, reason = 'UNAVAILABLE', zero_cost(), 'SOURCE_NOT_CONNECTED'
            elif all(s.mask.get(g) == 'OBSERVED' for g in spec.group_ids):
                status, cost = 'CACHE_HIT', zero_cost()
                revealed = {g: s.groups[g] for g in spec.group_ids}
            elif any(s.budget.spent[k] + getattr(cost, k).amount > limit + 1e-12 for k, limit in s.budget.limits.items()):
                status, cost, reason = 'BUDGET_EXCEEDED', zero_cost(), 'NO_ACQUISITION_EXECUTED'
            else:
                n = self._attempts.get(action_id, 0)
                self._attempts[action_id] = n + 1
                faults = self._faults.get(action_id, [])
                fault = faults[n] if n < len(faults) else None
                synthetic = fault is not None
                if fault:
                    status, reason = fault, 'EXPLICIT_SYNTHETIC_PERTURBATION'
                    mask = 'SOURCE_MISSING' if fault == 'NO_DATA' else 'STALE' if fault == 'STALE' else 'FAILED'
                    if fault != 'UNCHANGED':
                        masks = {g: mask for g in spec.group_ids if s.mask[g] != 'OBSERVED'}
                else:
                    try:
                        projected = {g: self._project(g) for g in spec.group_ids}
                        revealed = {g: v for g, v in projected.items() if v.quality != 'SOURCE_MISSING'}
                        masks = {g: 'SOURCE_MISSING' if v.quality == 'SOURCE_MISSING' else 'OBSERVED' for g, v in projected.items()}
                        status = 'ACQUIRED' if revealed else 'NO_DATA'
                        reason = 'MISSING_FIELDS_REMAIN_NULL' if any(v.missing_fields for v in projected.values()) else None
                    except (ValueError, KeyError, TypeError):
                        status, reason = 'BAD_DATA', 'SOURCE_CONTRACT_FAILED'
                        masks = {g: 'FAILED' for g in spec.group_ids if s.mask[g] != 'OBSERVED'}
            changed = any(g not in s.groups or s.groups[g] != v for g, v in revealed.items())
            s.groups.update(revealed)
            s.mask.update(masks)
            for k in DIMENSIONS:
                part = getattr(cost, k)
                if part.amount is None:
                    if k not in s.budget.unknown_components:
                        s.budget.unknown_components.append(k)
                else:
                    s.budget.spent[k] += part.amount
            s.state_version += 1
            result = AcquisitionResult(episode_id=s.episode_id, action_id=action_id, request_id=request_id,
                status=status, revealed_groups=revealed, mask_updates=masks, charged_cost=cost,
                sequence=len(s.history) + 1, evidence_refs=sorted({r for v in revealed.values() for r in v.evidence_refs}),
                content_changed=changed, synthetic_perturbation=synthetic, reason=reason,
                resulting_cutoff=s.prediction_cutoff, attempts=0 if status in ('CACHE_HIT', 'UNAVAILABLE', 'BUDGET_EXCEEDED') else 1)
            s.history.append(ActionReceipt(action_id=action_id, request_id=request_id, status=status,
                group_ids=spec.group_ids, charged_cost=cost, sequence=result.sequence, synthetic_perturbation=synthetic))
            self._receipts[request_id] = result.model_copy(deep=True)
            return result.model_copy(deep=True)

    def restore(self, public_state):
        """Trusted runner recovery: replay receipts, then verify every public field.

        Serialized state is not authority to insert values, refund cost, or set a
        mask. A changed bundle/config/perturbation rejects the recovery.
        """
        supplied = VisibleState.model_validate(public_state)
        with self._lock:
            previous = (self._state, self._receipts, self._attempts, getattr(self, '_snapshot', None))
            try:
                self.start(supplied.match_id, supplied.prediction_cutoff, supplied.information_mode)
                for receipt in supplied.history:
                    self.acquire(receipt.action_id, receipt.request_id)
                if self._state != supplied:
                    raise ValueError('Public recovery state differs from deterministic replay')
                return self.visible_state()
            except Exception:
                self._state, self._receipts, self._attempts, self._snapshot = previous
                raise

    def handle(self, request):
        """Allow-listed JSON RPC. No filesystem/SQL/manifest/label operations."""
        op = request.get('op')
        fields = {'start': {'op', 'match_id', 'prediction_cutoff', 'information_mode'},
                  'catalog': {'op'}, 'state': {'op'}, 'acquire': {'op', 'action_id', 'request_id'}}
        if op not in fields or set(request) - fields[op]:
            raise ValueError('Unknown public operation or parameters')
        if op == 'start':
            if self._state is not None:
                raise ValueError('Public worker permits one start; lifecycle reset belongs to trusted owner')
            return self.start(request['match_id'], request['prediction_cutoff'], request.get('information_mode', 'historical_reconstruction')).model_dump(mode='json')
        if op == 'catalog':
            return [v.model_dump(mode='json') for v in self.catalog()]
        if op == 'state':
            return self.visible_state().model_dump(mode='json')
        return self.acquire(request['action_id'], request['request_id']).model_dump(mode='json')
