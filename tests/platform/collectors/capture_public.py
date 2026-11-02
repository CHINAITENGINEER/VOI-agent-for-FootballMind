"""Small, bounded public-source verification for A; no credentials or DB.

This writes evidence under A's private work directory, NOT a production archive
and NOT a replacement RawObservation schema. No retry/challenge bypass.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4
import requests

ROOT = Path(__file__).resolve().parents[3]
TARGETS = {
    'football_data_competition_docs': 'https://docs.football-data.org/general/v4/competition.html',
    'football_data_pricing': 'https://www.football-data.org/pricing',
    'football_data_public_catalog': 'https://api.football-data.org/v4/competitions',
    'odds_api_history_docs': 'https://the-odds-api.com/historical-odds-data/',
    'api_football_official_guide': 'https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide',
    'api_football_pricing': 'https://www.api-football.com/pricing',
    'fbref_data_announcement': 'https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/',
}


def capture(session, folder, name, url, purpose='official_documentation'):
    """One real request, stored only in role A's research evidence folder."""
    record = {'evidence_id': str(uuid4()), 'name': name, 'url': url,
              'request_started_at': datetime.now(timezone.utc).isoformat(),
              'credential_sent': False, 'purpose': purpose}
    try:
        response = session.get(url, timeout=(10, 20), allow_redirects=True)
        body = response.content
        record.update(http_status=response.status_code, payload_bytes=len(body),
                      content_type=response.headers.get('Content-Type'), final_url=response.url,
                      observed_at=datetime.now(timezone.utc).isoformat(), sha256=hashlib.sha256(body).hexdigest())
        content_type = response.headers.get('Content-Type', '').lower()
        extension = '.json' if 'json' in content_type else '.js' if 'javascript' in content_type else '.html'
        name_with_id = name + '_' + record['evidence_id']
        (folder / (name_with_id + extension)).write_bytes(body)
        record['payload'] = str((folder / (name_with_id + extension)).relative_to(ROOT))
    except requests.RequestException as exc:
        record.update(http_status=None, error_class=type(exc).__name__, observed_at=datetime.now(timezone.utc).isoformat())
    (folder / (record['evidence_id'] + '.metadata.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({k: record.get(k) for k in ('name', 'http_status', 'error_class', 'payload_bytes')}), flush=True)
    return record


def main():
    folder = ROOT / 'var' / 'parallel' / 'collectors' / 'public-verification'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    with requests.Session() as session:
        for name, url in TARGETS.items():
            record = capture(session, folder, name, url, 'public_catalog_only' if 'catalog' in name else 'official_documentation')
            records.append(record)
            time.sleep(1)
    (folder / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
