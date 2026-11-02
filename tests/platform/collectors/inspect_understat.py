"""Fetch an exact script referenced by previously archived official HTML.

Research evidence only. No endpoint guessing, challenge solving or ingestion.
"""
import gzip
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urljoin

import requests
from capture_public import capture, ROOT


class ScriptLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.base = None
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'base':
            self.base = attrs.get('href')
        if tag == 'script' and attrs.get('src'):
            self.scripts.append(attrs['src'])


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--verify-data', action='store_true', help='Use saved JS discovery for one real league-data request')
    args = args.parse_args()
    archive = ROOT / 'football_spider/platform_data'
    records = [json.loads(p.read_bytes()) for p in (archive / 'raw').glob('understat_probe/*/*.observation.json')]
    record = max(records, key=lambda o: o['observed_at'])
    page = gzip.decompress((archive / record['payload_path']).read_bytes()).decode('utf-8')
    links = ScriptLinks()
    links.feed(page)
    script = next(s for s in links.scripts if 'js/league.min.js?' in s)
    url = urljoin(links.base or record['endpoint'], script)
    folder = ROOT / 'var/parallel/collectors/understat-research'
    folder.mkdir(parents=True, exist_ok=True)
    if args.verify_data:
        discovery = json.loads((folder / 'manifest.json').read_bytes())
        js = (ROOT / discovery['payload']).read_text(encoding='utf-8')
        # This string was observed in the official script, not guessed.
        if 'url:"getLeagueData/"+league+"/"+season,type:"get"' not in js:
            raise ValueError('Official JS discovery changed; re-inspect before requesting')
        league, season = record['endpoint'].rstrip('/').split('/')[-2:]
        data_url = urljoin(links.base, f'getLeagueData/{league}/{season}')
        with requests.Session() as session:
            session.headers.update({'Referer': record['endpoint'], 'X-Requested-With': 'XMLHttpRequest'})
            data_record = capture(session, folder, 'understat_league_data', data_url, 'discovered_xhr_league_data')
        data_record['discovered_from_evidence'] = discovery['evidence_id']
        data_record['page_observation_id'] = record['id']
        (folder / 'data-manifest.json').write_text(json.dumps(data_record, indent=2), encoding='utf-8')
        if data_record.get('http_status') == 200:
            data = json.loads((ROOT / data_record['payload']).read_bytes())
            print(json.dumps({'keys': list(data), 'sizes': {k: len(v) for k,v in data.items() if isinstance(v,(list,dict))}}))
        return
    with requests.Session() as session:
        evidence = capture(session, folder, 'understat_league_js', url, 'official_page_linked_javascript')
    evidence['discovered_from_observation'] = record['id']
    evidence['discovery'] = {'html_base': links.base, 'script_src': script}
    (folder / 'manifest.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    if evidence.get('http_status') == 200:
        js = (ROOT / evidence['payload']).read_text(encoding='utf-8')
        for term in ('ajax', 'getData', 'fetch(', 'leagueData', 'datesData', 'url:', '/get', 'getLeague'):
            for match in list(re.finditer(re.escape(term), js))[:6]:
                print(term, js[max(0, match.start()-100):match.start()+300])


if __name__ == '__main__':
    main()
