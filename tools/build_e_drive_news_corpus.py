"""Collect public club news and build historical, read-gated vector research data."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from build_e_drive_football_corpus import fetch, digest, write_json, write_rows, now, MODEL_REVISION

TOPICS = re.compile(r'team-news|injur|preview|press-conference|training|confirmed.*line|suspens|fitness|rotation|availability', re.I)
HOST_TEAM = {'www.liverpoolfc.com': 'Liverpool', 'www.arsenal.com': 'Arsenal',
             'www.chelseafc.com': 'Chelsea', 'www.cpfc.co.uk': 'Crystal Palace'}
EXCLUDED_PATH = re.compile(r'/news/(tag|category|listing|page)/')


def read_rows(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def parse_article(data, host):
    soup = BeautifulSoup(data, 'html.parser')
    schema = None
    for node in soup.select('script[type="application/ld+json"]'):
        try:
            obj = json.loads(node.get_text())
        except (ValueError, TypeError):
            continue
        objects = obj if isinstance(obj, list) else [obj]
        if isinstance(obj, dict):
            objects += obj.get('@graph', [])
        schema = next((o for o in objects if isinstance(o, dict) and o.get('@type') in ('NewsArticle', 'Article')), schema)
    info = {}
    for node in soup.select('[data-props]') if host == 'www.chelseafc.com' else []:
        try:
            obj = json.loads(node['data-props'])
            if obj.get('requiresLogin') is True:
                raise RuntimeError('article_requires_login')
            if 'page_info' in obj:
                info = obj['page_info']
        except (ValueError, TypeError):
            continue
    if schema and schema.get('isAccessibleForFree') in (False, 'False', 'false'):
        raise RuntimeError('article_not_freely_accessible')
    selectors = {'www.liverpoolfc.com': 'article [data-testid="rte-content"] p, article [data-testid="rte-content"] li',
                 'www.arsenal.com': '[class*="Text_text__"] p, [class*="Text_text__"] li',
                 'www.cpfc.co.uk': '.richtext-root p, .richtext-root li',
                 'www.chelseafc.com': '.news-article-page__body p, .news-article-page__body li'}
    nodes = soup.select(selectors[host])
    nodes = [n for n in nodes if not (n.name == 'li' and n.find('p'))]
    paragraphs = [n.get_text(' ', strip=True) for n in nodes if n.get_text(' ', strip=True)]
    if not paragraphs:
        raise RuntimeError('no_verified_article_body')
    # Some clubs render the introductory paragraph separately from rich text.
    # Include a description only when the same words actually occur in a DOM p.
    description = (schema or {}).get('description')
    if description and description not in paragraphs:
        normalized = ' '.join(description.split())
        if any(' '.join(p.get_text(' ', strip=True).split()) == normalized for p in soup.select('p')):
            paragraphs.insert(0, description)
    # Do not substitute a JSON-LD description or navigation for a missing body.
    body = '\n\n'.join(paragraphs)
    if len(body) < 150:
        raise RuntimeError('no_verified_article_body')
    meta = schema or {}
    keywords = meta.get('keywords') or []
    if isinstance(keywords, str):
        keywords = [x.strip() for x in keywords.split(',') if x.strip()]
    og = soup.select_one('meta[property="og:title"]')
    title = meta.get('headline') or info.get('content_title') or (og.get('content') if og else None)
    if not title:
        raise RuntimeError('missing_article_title')
    return {'title': title, 'body': body, 'paragraphs': paragraphs,
            'published_at': meta.get('datePublished') or info.get('content_publish_date'),
            'modified_at': meta.get('dateModified'),
            'source_description': meta.get('description'),
            'source_category': info.get('content_sub_category') or meta.get('articleSection'),
            'source_keywords': keywords,
            'source_native_match_id': info.get('match_id'),
            'extraction': 'registered_dom_paragraphs_v2', 'body_sha256': digest(body.encode('utf-8'))}


def discover(root, probe_root, per_host=220):
    maps = [('lfc_allmap.txt', 'https://www.liverpoolfc.com/sitemaps/sitemap/0.xml'),
            ('chelsea_map.txt', 'https://www.chelseafc.com/en/sitemap.xml'),
            ('cpfc_sitemap.txt', 'https://www.cpfc.co.uk/server-sitemap.xml'),
            ('arsenal_map1.txt', 'https://www.arsenal.com/sitemaps/articles/1/sitemap.xml')]
    # These are real observed child URLs from Arsenal's index, not guessed routes.
    index = BeautifulSoup((probe_root / 'arsenal_articlesmap.txt').read_bytes(), 'xml')
    child_urls = [x.text for x in index.select('loc')][1:4]
    for i, url in enumerate(child_urls, start=2):
        path, receipt = fetch(root, url, f'raw/discovery/arsenal-{i}.xml')
        maps.append((path, url))
    candidates = []
    for filename, map_url in maps:
        path = filename if isinstance(filename, Path) else probe_root / filename
        data = path.read_bytes()
        map_sha = digest(data)
        soup = BeautifulSoup(data, 'xml')
        for row in soup.select('url'):
            if row.loc is None:
                continue
            url = row.loc.text
            host = urlsplit(url).hostname
            if host not in HOST_TEAM or '/news/' not in url or EXCLUDED_PATH.search(url):
                continue
            candidates.append({'url': url, 'host': host, 'source_team': HOST_TEAM[host],
                               'sitemap_lastmod': row.lastmod.text if row.lastmod else None,
                               'sitemap_url': map_url, 'sitemap_raw_path': str(path), 'sitemap_sha256': map_sha,
                               'sitemap_lastmod_is_publication_time': False,
                               'discovery_phase': 'offline_corpus_not_agent_menu',
                               'topic_stratum': 'evidence_topic' if TOPICS.search(url) else 'other_topic'})
    unique = {c['url']: c for c in candidates}
    candidates = sorted(unique.values(), key=lambda c: c['url'])
    selected = set()
    for host in HOST_TEAM:
        for stratum, quota in [('evidence_topic', int(per_host*.9)), ('other_topic', per_host-int(per_host*.9))]:
            pool = sorted((c for c in candidates if c['host'] == host and c['topic_stratum'] == stratum),
                          key=lambda c: digest(c['url'].encode()))
            selected.update(c['url'] for c in pool[:quota])
    for c in candidates:
        c['selected_for_corpus'] = c['url'] in selected
        c['selection_basis'] = 'deterministic_url_hash_within_host_topic_stratum_no_outcome_or_gain'
    write_rows(root / 'news/discovery.jsonl', candidates)
    return [c for c in candidates if c['selected_for_corpus']]


def collect(root, probe_root, per_host, offline=False):
    plan = root / 'news/discovery.jsonl'
    candidates = [x for x in read_rows(plan) if x['selected_for_corpus']] if plan.exists() else discover(root, probe_root, per_host)
    documents, failures = [], []
    def host_worker(host):
        accepted, rejected = [], []
        group = [c for c in candidates if c['host'] == host]
        for index, c in enumerate(group):
            article_id = digest(c['url'].encode())[:24]
            try:
                if offline:
                    path = root / f'raw/news/{host}/{article_id}.html'
                    receipt = json.loads(path.with_suffix('.html.receipt.json').read_text(encoding='utf-8'))
                    if receipt.get('status') != 200 or receipt.get('error') or digest(path.read_bytes()) != receipt['sha256']:
                        raise RuntimeError('archived_response_not_successful_or_digest_mismatch')
                else:
                    path, receipt = fetch(root, c['url'], f'raw/news/{host}/{article_id}.html', max_bytes=8_000_000)
                record = parse_article(path.read_bytes(), host)
                body_path = root / 'news/bodies' / (article_id + '.txt')
                body_path.parent.mkdir(parents=True, exist_ok=True)
                body_path.write_text(record.pop('body'), encoding='utf-8', newline='\n')
                record.pop('paragraphs')
                record.update(c, article_id=article_id, body_path=str(body_path), raw_path=str(path),
                              raw_sha256=receipt['sha256'], captured_at=receipt['response_completed_at'],
                              available_at=None, historical_version_unknown=True,
                              provenance='historical_reconstruction', first_public_available_at=None,
                              historical_prequery_title_availability_verified=False,
                              strict_prospective=False, semantic_review_status='not_reviewed',
                              g_gain=None, training_eligibility='requires_match_binding_and_f_transition')
                write_json(root / 'news/records' / (article_id + '.json'), record)
                accepted.append(record)
            except Exception as exc:
                receipt_path = root / f'raw/news/{host}/{article_id}.html.receipt.json'
                observed = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
                rejected.append({**c, 'article_id': article_id, 'reason': str(exc),
                                 'receipt_path': str(receipt_path), 'http_status': observed.get('status'),
                                 'transport_error': observed.get('error'), 'raw_sha256': observed.get('sha256'),
                                 'response_completed_at': observed.get('response_completed_at')})
            if (index+1) % 20 == 0:
                print('NEWS', host, index+1, '/', len(group), 'bodies', len(accepted), 'failed', len(rejected), flush=True)
            if not offline:
                time.sleep(.25)
        return accepted, rejected
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for good, bad in pool.map(host_worker, HOST_TEAM):
            documents.extend(good)
            failures.extend(bad)
    documents.sort(key=lambda x: x['article_id'])
    write_rows(root / 'news/articles.jsonl', documents)
    write_rows(root / 'news/failures.jsonl', failures)
    write_json(root / 'reports/news.json', {'selected': len(candidates), 'bodies': len(documents),
               'failures': len(failures), 'hosts': dict(Counter(x['host'] for x in documents)),
               'unique_body_hashes': len({x['body_sha256'] for x in documents}),
               'semantic_review_performed': False, 'g_training_rows': 0})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--probe-root', type=Path, default=Path('E:/FootballMind/source-probes'))
    p.add_argument('--per-host', type=int, default=220)
    p.add_argument('--offline-reparse', action='store_true')
    args = p.parse_args()
    collect(args.root, args.probe_root, args.per_host, offline=args.offline_reparse)


if __name__ == '__main__':
    main()
