"""Bounded historical source recovery, separate from prospective acquisition."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URLS = [
    'https://www.chelseafc.com/en/news/article/enzo-maresca-delivers-chelsea-team-news-ahead-of-burnley',
    'https://www.chelseafc.com/en/news/article/enzo-maresca-confirms-chelsea-team-news-for-arsenal',
    'https://www.chelseafc.com/en/news/article/confirmed-chelsea-line-up-versus-arsenal-web-november-2025',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'collection-plan.json', {
        'created_at': now(), 'fixture_id': 'understat:28900',
        'fixture': 'Chelsea–Arsenal, 2025-11-30', 'urls': URLS,
        'maximum_direct_requests': 3, 'retries': 0, 'follow_redirects': False,
        'selection_basis': 'official pre-match availability update and lineup; before f/g scoring',
        'review_model': 'current Codex session; no provider API',
        'historical_version_unknown': True, 'prospective_training_eligible': False,
        'analyst_blind': False,
        'disclosure': 'Discovery search also exposed a post-match result snippet; not strictly blind selection. No result article is fetched.',
    })
    ledger = []
    for i, url in enumerate(URLS, 1):
        key = f'C{i:02d}'
        entry = {'id': key, 'url': url, 'request_started_at': now()}
        try:
            response = requests.get(url, timeout=35, allow_redirects=False,
                                    headers={'User-Agent': 'Mozilla/5.0'})
            entry.update(response_received_at=now(), http_status=response.status_code,
                         response_url=response.url, content_type=response.headers.get('Content-Type'))
            raw = response.content
            entry['raw_sha256'] = sha256(raw).hexdigest()
            entry['raw_path'] = f'{key}.html'
            entry['save_started_at'] = now()
            (out / entry['raw_path']).write_bytes(raw)
            entry['save_returned_at'] = now()
            if response.status_code != 200:
                entry['status'] = 'NON_200_NO_RETRY'
            else:
                soup = BeautifulSoup(raw, 'html.parser')
                headers = soup.select('[data-component="ArticleHeader"][data-props]')
                roots = soup.select('article.news-article-page')
                assert len(headers) == len(roots) == 1, 'unique registered article required'
                details = json.loads(headers[0]['data-props'])['articleHeaderDetails']
                paragraphs = [p.get_text(' ', strip=True) for p in roots[0].select(
                    'section.news-article-page__body contentful-rich-text p')]
                paragraphs = [p for p in paragraphs if p]
                assert len(paragraphs) >= 2, 'full article paragraph container required'
                body = '\n'.join(paragraphs)
                (out / f'{key}.txt').write_text(body, encoding='utf-8', newline='\n')
                entry.update(status='BODY_RECOVERED', title=details['title'],
                             source_published_at=details.get('date'), header_details=details,
                             paragraphs=paragraphs, body_path=f'{key}.txt',
                             body_sha256=sha256(body.encode()).hexdigest(),
                             historical_available_at=None, historical_version_unknown=True)
        except Exception as exc:
            entry.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', ended_at=now())
        write(out / f'{key}.json', entry)
        ledger.append(entry)
        print(json.dumps({k: entry.get(k) for k in ('id', 'status', 'title', 'source_published_at', 'error')}, ensure_ascii=False), flush=True)
    write(out / 'ledger.json', ledger)


if __name__ == '__main__':
    main()
