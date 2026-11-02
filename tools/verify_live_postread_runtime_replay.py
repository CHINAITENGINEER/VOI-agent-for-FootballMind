"""Run the real new live application against exact saved HTTP responses only.

The injected clock belongs to a declared offline replay. Missing archived URLs
or searches return offline_missing; they never become invented HTTP responses.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.verify_live_postread_migration import RUN, digest_file, no_network, now, read_json, write_json

OUTPUT = ROOT / 'var/work/live-postread-migration-20261025-v1/offline-runtime-replay'


class ReplayClock:
    def __init__(self, value):
        self.value = datetime.fromisoformat(value.replace('Z', '+00:00'))

    def __call__(self):
        self.value += timedelta(milliseconds=1)
        return self.value.isoformat()

    def at_least(self, value):
        self.value = max(self.value, datetime.fromisoformat(value.replace('Z', '+00:00')))


class RecordedTools:
    """Prepare/project with production tools; execute only matches the archive."""
    def __init__(self, backend, entries, allowed):
        from demo.live_sources import LiveNewsTools
        from football_spider.storage.raw_archive import RawArchive
        self.backend = backend
        self.delegate = LiveNewsTools(RawArchive(backend.original / 'sources'), allowed,
            index_urls=entries, projection_version=13,
            clock=lambda: datetime.fromisoformat(backend.clock()))
        self.allowed_hosts = list(allowed)

    def configuration(self):
        return {**self.delegate.configuration(), 'execution_mode': 'offline_archive_replay',
                'network_requests': 0, 'missing_archive_policy': 'explicit_offline_missing'}

    def prepare(self, action, public):
        self.delegate.allowed_hosts = self.allowed_hosts
        return self.delegate.prepare(action, public)

    def execute(self, prepared, *, run_id, request_id):
        from football_spider.models.contracts import Observation
        matches = [r for r in self.backend.observations if
                   r['endpoint'] == prepared['endpoint'] and r['params'] == prepared.get('params', {})]
        entry = {'tool': prepared['tool'], 'endpoint': prepared['endpoint'],
                 'params': prepared.get('params', {}), 'new_http': 0,
                 'replay_request_id': request_id, 'wall_clock_at': now()}
        self.backend.calls.append(entry)
        if not matches:
            entry['status'] = 'offline_missing'
            return {'observation': None, 'offline_replay': True, 'network_requests': 0,
                    'document': {'document_id': None, 'status': 'FAILED', 'url': prepared['endpoint'],
                                 'links': [], 'text': '', 'reason': 'offline_missing',
                                 'error_class': 'ArchivedResponseUnavailable', 'http_status': None,
                                 'structured_news': {'role': 'unavailable'}, 'offline_replay': True}}
        if len(matches) != 1:
            raise ValueError('Ambiguous saved response version; no arbitrary archive selection')
        record = matches[0]
        observation = Observation.model_validate(record)
        raw = self.delegate.archive.read_body(observation)
        assert hashlib.sha256(raw).hexdigest() == observation.content_hash
        self.backend.clock.at_least(record['observed_at'])
        doc = self.delegate.project(observation, prepared)
        doc.update(offline_replay=True, replay_clock_basis='injected_original_source_clock_not_new_capture',
                   replay_wall_clock_projected_at=now(), new_http_requests=0)
        entry.update(status='archive_reprojected', raw_sha256=observation.content_hash,
                     document_id=doc.get('document_id'), projection_status=doc['status'],
                     body_characters=len(doc.get('text', '')), original_observed_at=record['observed_at'])
        return {'observation': deepcopy(record), 'document': doc, 'offline_replay': True,
                'network_requests': 0}


class RecordedBackend:
    def __init__(self, original, clock):
        self.original, self.clock = original, clock
        self.calls = []
        self.observations = [read_json(p) for p in sorted((original / 'sources/raw').rglob('*.observation.json'))]
        assert len(self.observations) == 5

    def initialize(self, match, folder):
        value = read_json(self.original / 'setup/espn-initialization.json')
        value.update(replay_mode='offline_archived_initialization', new_http_requests=0,
                     original_initialization_path=str(self.original / 'setup/espn-initialization.json'))
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'replayed-initialization.json', value)
        return value

    def identity(self, manifest, folder, expected):
        value = read_json(self.original / 'identity/identity.json')
        assert value['canonical']['source_event_id'] == expected['match_id']
        value.update(replay_mode='offline_archived_identity', fresh_independent_check=False,
                     new_http_requests=0)
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'replayed-identity.json', value)
        return value

    def tools(self, folder, entries, allowed):
        return RecordedTools(self, entries, allowed)

    def semantics(self, *args, **kwargs):
        raise AssertionError('No external semantic API belongs in this migration')


def parameter_state(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.f.network.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {'f_sha256': digest_file(model.release_dir / 'models/f/model.pt'),
            'g_sha256': digest_file(model.release_dir / 'models/g/g-models.joblib'),
            'f_state_sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = now()
    write_json(output / 'status.json', {'status': 'running', 'scope': 'offline_runtime_replay', 'started_at': started})
    original_hashes = {str(p): digest_file(p) for p in RUN.rglob('*') if p.is_file()}
    identity = read_json(RUN / 'identity/identity.json')
    init = read_json(RUN / 'setup/espn-initialization.json')['market_initialization']
    c = identity['canonical']
    # This catalog projection is derived solely from the saved source fixture,
    # not the old run's predictions, gains, outcomes, or selected policy.
    match = {'match_id': c['source_event_id'], 'fixture_id': c['source_event_id'],
             'source_match_id': c['source_event_id'].split(':')[-1], 'source': 'espn',
             'home': c['home_name'], 'away': c['away_name'], 'kickoff': c['kickoff'],
             'competition_key': c['competition_name'], 'supported': True, 'status': init['fixture']['status'],
             'quote_observed': True, 'quote_catalog_observed_at': init['source_observed_at'],
             'quote_snapshot': init['quote_snapshot'], 'offline_replay': True}
    import demo.live_postread as runtime
    from demo.live_postread_model import LivePostreadModel, DEFAULT_HEAD

    class ReplayApplication(runtime.LivePostreadApplication):
        def _save(self, record):
            record.update(offline_replay=True, prospective=False,
                replay_scope='new_application_control_flow_using_exact_old_archived_responses',
                replay_wall_clock_saved_at=now(),
                execution_clock_basis='injected historical replay time; never a new prospective observation',
                actual_new_http_requests=0, original_archive_run_id=RUN.name)
            super()._save(record)

    code_before = {str(ROOT / p): digest_file(ROOT / p) for p in runtime.CODE_FILES}
    reports = []
    with no_network() as network_attempts, patch('demo.live_g.key_from_environment', lambda: None):
        for head in (DEFAULT_HEAD, 'fixed_order'):
            store = output / head
            store.mkdir()
            clock = ReplayClock(identity['verified_at'])
            backend = RecordedBackend(RUN, clock)
            model = LivePostreadModel(head=head)
            parameters = parameter_state(model)
            app = ReplayApplication(store, backend=backend, clock=clock, model=model)
            write_json(store / 'catalog.json', {'matches': [deepcopy(match)], 'status': 'ready',
                       'updated_at': clock(), 'offline_replay': True})
            creation = app.start({'match_id': match['match_id'], 'max_reads': 20,
                                  'cost_loss_units': 0.0, 'model_name': head})
            worker = app.workers[creation['run_id']]
            worker.join(90)
            if worker.is_alive():
                app.cancel[creation['run_id']].set()
                worker.join(30)
                raise RuntimeError('Replay exceeded bound; cancellation requested')
            result = app.get(creation['run_id'])
            assert result['state'] == 'STOPPED', (result['state'], result.get('failure_detail'))
            assert result['counts']['encoded'] == 3
            assert result['counts']['provider'] == 0
            assert result['outcome'] is None and result['gain'] is None and not result['g_training_eligible']
            encoded = [r for r in result['candidate_pool'] if r['status'] == 'encoded']
            for row in encoded:
                assert row['encoding']['cache_hit']
                assert row['text_sha256'] == hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
                assert row['captured_at'] is None
            if head == DEFAULT_HEAD:
                assert len(result['steps']) == 0
                assert result['reason'] == 'predicted_gain_below_frozen_threshold'
                assert all(not r['f_adopted'] and r['prediction_after'] is None for r in encoded)
                assert len(result['decisions'][0]['candidates']) == 3
            else:
                assert len(result['steps']) == 3
                assert all(s['f_executed'] and s['f_used_document_ids'] for s in result['steps'])
                assert all(not d['candidates'] and not d['g_selection_executed'] for d in result['decisions'])
                for i, step in enumerate(result['steps']):
                    assert step['used_input_ids'] == [s['text_sha256'] for s in result['steps'][:i+1]]
                    if i: assert step['prediction_before'] == result['steps'][i-1]['prediction_after']
            assert parameter_state(model) == parameters
            write_json(store / 'archive-replay-ledger.json', backend.calls)
            run_path = store / 'runs' / result['run_id'] / 'run.json'
            # Stable export for an explicit offline browser fixture; identical
            # bytes to the new application record, never copied to production.
            (store / 'browser-replay-run.json').write_bytes(run_path.read_bytes())
            report = {'policy': head, 'run_id': result['run_id'], 'run_path': str(run_path),
                      'browser_fixture': str(store / 'browser-replay-run.json'),
                      'state': result['state'], 'stop_reason': result['reason'],
                      'archive_responses_reprojected': sum(x['status'] == 'archive_reprojected' for x in backend.calls),
                      'offline_missing_actions': sum(x['status'] == 'offline_missing' for x in backend.calls),
                      'source_actions_are_replay_attempts_not_http': result['counts']['source'],
                      'candidate_pool_rows': len(result['candidate_pool']), 'encoded_documents': 3,
                      'body_characters': [len(r['text']) for r in encoded], 'f_body_updates': len(result['steps']),
                      'initial_probabilities': result['before'], 'final_probabilities': result['current'],
                      'frozen_parameters_unchanged': True, 'model_hashes': parameters,
                      'new_http_requests': 0, 'external_api_requests': 0, 'training_rows': 0}
            reports.append(report)
            write_json(store / 'verification.json', report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        assert not network_attempts
    assert original_hashes == {str(p): digest_file(p) for p in RUN.rglob('*') if p.is_file()}
    code_after = {p: digest_file(p) for p in code_before}
    assert code_before == code_after, 'Runtime changed during the acceptance run'
    write_json(output / 'verification.json', {'status': 'passed', 'scope': 'offline_runtime_replay',
        'started_at': started, 'finished_at': now(), 'new_http_requests': 0,
        'old_run_unchanged': True, 'original_files_sha256': original_hashes, 'runtime_sha256': code_after,
        'original_outcome_values_used': False, 'original_gain_values_used': False,
        'clock_basis': 'injected original historical time, not newly measured prospective time', 'runs': reports})
    write_json(output / 'status.json', {'status': 'passed', 'finished_at': now()})
    write_json(output / 'manifest.json', {'script_sha256': digest_file(__file__),
        'files': [{'path': str(p.relative_to(output)), 'sha256': digest_file(p)}
                  for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json']})


if __name__ == '__main__':
    main()
