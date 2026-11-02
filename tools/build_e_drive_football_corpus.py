"""Resumable research corpus on an explicitly supplied data volume.

Acquisition is separate from production Agent/f/g. Historical prices are not
timestamp-proven pre-query observations. Outcomes never enter public records.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path

import requests

MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def fetch(root, url, relative, max_bytes=15_000_000):
    path = root / relative
    receipt_path = path.with_suffix(path.suffix + '.receipt.json')
    if path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('sha256') == digest(path.read_bytes()) and receipt.get('status') == 200:
            return path, receipt
    start = now()
    receipt = {'url': url, 'request_started_at': start, 'status': None}
    try:
        with requests.get(url, timeout=(10, 45), stream=True,
                          headers={'User-Agent': 'FootballMind-Research/1.0'}) as r:
            receipt.update(status=r.status_code, final_url=r.url)
            chunks, size = [], 0
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError('response_size_limit')
                chunks.append(chunk)
            data = b''.join(chunks)
            receipt.update(response_completed_at=now(), bytes=len(data), sha256=digest(data))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            receipt['save_return_at'] = now()
    except Exception as exc:
        receipt.update(error=f'{type(exc).__name__}: {exc}', failed_at=now())
    write_json(receipt_path, receipt)
    with LOCK:
        with (root / 'http-ledger.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(receipt) + '\n')
    if receipt.get('status') != 200 or receipt.get('error'):
        raise RuntimeError(f"fetch failed {url}: {receipt.get('status')} {receipt.get('error')}")
    return path, receipt


def parse_csv(data, season, division, source):
    public, labels, odds, stats = [], [], [], []
    for row in csv.DictReader(io.StringIO(data.decode('utf-8-sig', errors='replace'))):
        if not row.get('HomeTeam') or not row.get('AwayTeam') or not row.get('Date'):
            continue
        date = None
        for fmt in ('%d/%m/%Y', '%d/%m/%y'):
            try:
                date = datetime.strptime(row['Date'], fmt).date().isoformat()
                break
            except ValueError:
                continue
        if not date:
            raise ValueError(f"Unrecognized date {row['Date']}")
        match_id = f"fd:{division}:{season}:{date}:{row['HomeTeam']}:{row['AwayTeam']}"
        public.append({'match_id': match_id, 'division': division, 'season': season,
                       'match_date': date, 'kickoff_local_time': row.get('Time') or None,
                       'kickoff_timezone': None, 'home_team': row['HomeTeam'], 'away_team': row['AwayTeam'],
                       'source': source, 'provenance': 'historical_reconstruction'})
        if row.get('FTR') in {'H', 'D', 'A'}:
            labels.append({'match_id': match_id, 'outcome': row['FTR'],
                           'home_goals': int(float(row['FTHG'])), 'away_goals': int(float(row['FTAG'])),
                           'source': source, 'label_available_at': None})
        snapshots = {}
        for prefix in ('Avg', 'B365', 'PS', 'Max', 'AvgC', 'B365C', 'PSC', 'MaxC'):
            try:
                values = [float(row[prefix + x]) for x in ('H', 'D', 'A')]
                if all(math.isfinite(x) and x > 1 for x in values):
                    inverse = [1/x for x in values]
                    snapshots[prefix] = {'decimal_odds': values, 'no_vig_probabilities': [x/sum(inverse) for x in inverse],
                                         'snapshot_kind': 'closing' if prefix.endswith('C') else 'preclosing_or_legacy',
                                         'quote_observed_at': None}
            except (KeyError, ValueError, TypeError):
                continue
        odds.append({'match_id': match_id, 'snapshots': snapshots, 'source': source,
                     'point_in_time_verified': False, 'strict_prequery_eligible': False})
        stats.append({'match_id': match_id, 'source': source,
                      'postmatch_statistics': {k: row.get(k) for k in ('HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR')}})
    return public, labels, odds, stats


def structured(root, first=2014, last=2024):
    jobs = [(f'{year%100:02d}{(year+1)%100:02d}', division)
            for year in range(first, last + 1) for division in ('E0', 'D1', 'I1', 'SP1', 'F1')]
    all_parts = [[], [], [], []]
    failed = []
    def one(job):
        season, division = job
        url = f'https://www.football-data.co.uk/mmz4281/{season}/{division}.csv'
        path, receipt = fetch(root, url, f'raw/football-data/{season}/{division}.csv')
        source = {'url': url, 'sha256': receipt['sha256'], 'path': str(path), 'downloaded_at': receipt['response_completed_at']}
        return parse_csv(path.read_bytes(), season, division, source)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(one, j): j for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                parts = future.result()
                for target, part in zip(all_parts, parts):
                    target.extend(part)
                print('CSV', job, len(parts[0]), flush=True)
            except Exception as exc:
                failed.append({'job': job, 'error': str(exc)})
    for rows, filename in zip(all_parts, ('matches.jsonl', 'labels/results.jsonl', 'market_snapshots.jsonl', 'labels/postmatch_statistics.jsonl')):
        rows.sort(key=lambda x: x['match_id'])
        ids = [r['match_id'] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate match identity')
        write_rows(root / 'structured' / filename, rows)
    write_json(root / 'reports/structured.json', {'matches': len(all_parts[0]), 'labels': len(all_parts[1]),
               'odds_covered_matches': sum(bool(r['snapshots']) for r in all_parts[2]),
               'csv_requested': len(jobs), 'failed': failed,
               'strict_prospective': False, 'g_gain_labels_generated': 0})


def download_model(root):
    target = root / 'models' / 'all-MiniLM-L6-v2' / MODEL_REVISION
    files = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'vocab.txt', 'README.md']
    for filename in files:
        fetch(root, f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{filename}',
              target.relative_to(root) / filename, max_bytes=150_000_000)
        print('MODEL', filename, flush=True)
    write_json(target / 'provenance.json', {'model_id': MODEL_ID, 'revision': MODEL_REVISION,
               'dimension': 384, 'pooling': 'attention_mask_mean_then_l2', 'max_length': 256})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['structured', 'model'], required=True)
    p.add_argument('--first-season', type=int, default=2014)
    p.add_argument('--last-season', type=int, default=2024)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'structured':
        structured(args.root, args.first_season, args.last_season)
    else:
        download_model(args.root)


if __name__ == '__main__':
    main()
