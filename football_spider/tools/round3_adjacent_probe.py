"""Source-evidenced adjacent-season capture, never a future holdout claim.

This command first reuses a checksum-verified official response. Otherwise it
allows one logical GET and at most two attempts, within a four-request ceiling.
It does not open a database, normalize into a production repository, or train.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filelock import FileLock
from football_platform.collectors.http_transport import RequestTransport, TransportError
from football_spider.collectors.quota import request_slot, reserve
from football_spider.collectors.understat import TIME_POLICY
from football_spider.normalization.understat import audit_season, normalize_understat
from football_spider.platform_config import PlatformSettings
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes

PAGE_ID = 'b7212198-7610-44d4-9f52-4a51dae38c05'
ENDPOINT = 'https://understat.com/getLeagueData/EPL/2024'
OUT = ROOT / 'var/parallel/collectors/round3/adjacent'


def discovery(archive):
    """Check both source inputs rather than synthesizing a plausible endpoint."""
    page = next(o for o in archive.observations() if o.id == PAGE_ID)
    html = archive.read_body(page).decode('utf-8')
    option = re.search(r'<option\s+value="2024"[^>]*>2024/2025</option>', html)
    epl = re.search(r'<option\s+value="EPL"[^>]*selected[^>]*>', html)
    if not option or not epl:
        raise ValueError('Official page no longer proves EPL and 2024 season selection')
    manifest_path = ROOT / 'var/parallel/collectors/understat-research/manifest.json'
    metadata = json.loads(manifest_path.read_bytes())
    js_path = (ROOT / metadata['payload']).resolve()
    if not js_path.is_relative_to(ROOT / 'var/parallel/collectors/understat-research'):
        raise ValueError('JS archive escaped original research directory')
    raw = js_path.read_bytes()
    if metadata['http_status'] != 200 or hashlib.sha256(raw).hexdigest() != metadata['sha256']:
        raise ValueError('Original JS hash/status mismatch')
    script = raw.decode('utf-8')
    construct = 'url:"getLeagueData/"+league+"/"+season'
    position = script.find(construct)
    if position < 0 or 'js/league.min.js?t=1765269520' not in html:
        raise ValueError('Official archived JS does not prove endpoint construction')
    return {
        'page_observation_id': page.id,
        'page_content_hash': page.content_hash,
        'page_observed_at': page.observed_at.isoformat(),
        'season_option': option.group(),
        'league_option': epl.group(),
        'script_evidence_id': metadata['evidence_id'],
        'script_content_hash': metadata['sha256'],
        'script_observed_at': metadata['observed_at'],
        'script_excerpt': script[max(0, position - 30):position + len(construct) + 100],
        'timezone_evidence_observation_id': '69874328-a5d6-4e32-af33-f971a909eb2e',
        'endpoint': ENDPOINT,
    }


def run(*, allow_network=True):
    archive = RawArchive(PlatformSettings().data_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        'run_id': str(uuid4()), 'created_at': datetime.now(timezone.utc).isoformat(),
        'source': 'understat', 'league': 'EPL', 'season': 2024,
        'status': 'FAILED', 'database_accessed': False,
        'is_test_fixture': False, 'requests_this_run': 0,
        'cost': {'money': None, 'currency': None, 'money_status': 'unknown',
                 'provider_quota': None, 'provider_quota_status': 'unknown',
                 'local_request_ceiling': 4, 'local_daily_ceiling': 20},
        'role': 'earlier-season training/validation candidate; not an unseen future confirmation set',
    }
    observations = []
    started = time.monotonic()
    try:
        report['discovery'] = discovery(archive)
        with FileLock(str(OUT / 'capture.lock')):
            existing = [o for o in archive.observations('understat')
                        if o.endpoint == ENDPOINT and o.http_status == 200
                        and o.context.get('league') == 'EPL'
                        and int(o.context.get('season', -1)) == 2024]
            if existing:
                observation = existing[0]
                raw = archive.read_body(observation)
                report['reused_existing_archive'] = True
            elif not allow_network:
                report['status'] = 'NO_DATA'
                report['reason'] = 'No registered EPL2024 archive; networking disabled'
                return report
            else:
                def persist(**facts):
                    observations.append(archive.save(
                        'understat', facts['endpoint'], facts['params'], facts['body'],
                        status=facts['http_status'], headers=facts['response_headers'],
                        request_id=facts['request_id'], observation_id=facts['observation_id'],
                        observed_at=facts['observed_at'], error_class=facts['error_class'],
                        context={'league': 'EPL', 'season': 2024, 'kickoff_timezone': 'UTC',
                                 'timezone_policy': TIME_POLICY,
                                 'request_started_at': facts['requested_at'].isoformat(),
                                 'attempt': facts['attempt'], 'round3_adjacent_capture': True,
                                 'discovered_from_page_observation': PAGE_ID,
                                 'discovered_from_script_evidence': report['discovery']['script_evidence_id'],
                                 'credential_echo_redacted': facts['credential_echo_redacted']}))

                def quota(source, _estimated_credits):
                    if not reserve(archive.root, source, 1, 20):
                        return False
                    delay = request_slot(archive.root, source, 3.0)
                    if delay is None:
                        return False
                    time.sleep(delay)
                    return True

                with RequestTransport('understat', on_attempt=persist, max_attempts=2,
                                      request_budget=4, reserve=quota, interval=3,
                                      timeout=(10, 35), deadline=time.monotonic() + 100,
                                      cookie_path=archive.root / 'cookies/understat.txt') as transport:
                    try:
                        raw = transport.get(ENDPOINT, headers={
                            'Referer': 'https://understat.com/league/EPL/2024',
                            'X-Requested-With': 'XMLHttpRequest'})
                    finally:
                        report['requests_this_run'] = transport.stats['attempts']
                        report['transport_statistics'] = dict(transport.stats)
                observation = observations[-1]
                report['reused_existing_archive'] = False

            payload = json.loads(raw)
            audit = audit_season(payload, 'EPL', 2024)
            normalized = normalize_understat(observation, payload)
            finished = sum(r['isResult'] is True for r in payload['dates'])
            report.update({
                'status': 'OK' if audit['structural_complete'] and not audit['history_mismatches']
                         and len(normalized.matches) == 380 and finished == 380 else 'PARTIAL',
                'observation_id': observation.id, 'request_id': observation.request_id,
                'observed_at': observation.observed_at.isoformat(),
                'source_published_at': None, 'ingested_at': None,
                'ingestion_status': 'NOT_PERFORMED_BY_PROBE',
                'content_hash': observation.content_hash, 'payload_bytes': len(raw),
                'payload_path': observation.payload_path,
                'archive_root': str(archive.root), 'context': observation.context,
                'http_status': observation.http_status, 'audit': audit,
                'finished_matches': finished,
                'normalization_preview': {'valid_matches': len(normalized.matches),
                    'statistics': len(normalized.statistics),
                    'issues': [i.model_dump(mode='json') for i in normalized.issues],
                    'parser_version': normalized.parser_version,
                    'persisted_to_database': False},
            })
    except (TransportError, ValueError, KeyError, StopIteration, OSError) as exc:
        report['status'] = 'RATE_LIMITED' if isinstance(exc, TransportError) and exc.category == 'RATE_LIMITED' else 'FAILED'
        report['error'] = {'category': getattr(exc, 'category', type(exc).__name__), 'message': str(exc)[:400]}
    finally:
        report['attempt_observation_ids'] = [o.id for o in observations]
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        atomic_write(OUT / 'runs' / (report['run_id'] + '.json'), json_bytes(report))
        atomic_write(OUT / 'report.json', json_bytes(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Only reuse an existing real archive')
    args = parser.parse_args()
    result = run(allow_network=not args.offline)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'OK' else 1)
