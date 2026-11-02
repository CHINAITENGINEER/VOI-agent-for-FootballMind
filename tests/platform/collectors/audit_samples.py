"""Generate an A-only, evidence-backed report without network or DB writes."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT/'tests/fixtures/collectors'
OUTPUT = ROOT/'var/parallel/collectors'


def read(relative):
    return json.loads((ROOT/relative).read_bytes())


def main():
    fixture_manifest = read('tests/fixtures/collectors/manifest.json')
    for item in fixture_manifest['samples']:
        metadata_path = ROOT/item['metadata']
        metadata = json.loads(metadata_path.read_bytes())
        body = (metadata_path.parent/metadata['payload_file']).read_bytes()
        if hashlib.sha256(body).hexdigest() != metadata['fixture_payload_sha256']:
            raise ValueError('Fixture checksum mismatch')
    data = read('tests/fixtures/collectors/understat/epl_2025.json')
    matches = data['dates']
    appearances = Counter(ref['id'] for match in matches for ref in (match['h'],match['a']))
    goals_complete = sum(all(m.get('goals',{}).get(side) is not None for side in ('h','a')) for m in matches)
    xg_complete = sum(all(m.get('xG',{}).get(side) is not None for side in ('h','a')) for m in matches)
    catalog = read('tests/fixtures/collectors/football_data/public_competitions.json')
    public = read('var/parallel/collectors/public-verification/manifest.json')
    public += [read('var/parallel/collectors/understat-research/manifest.json'),
               read('var/parallel/collectors/understat-research/data-manifest.json')]
    transport_live_path = OUTPUT/'transport-live/report.json'
    transport_live = json.loads(transport_live_path.read_bytes()) if transport_live_path.exists() else None
    if transport_live:
        public += transport_live['attempts']
    for record in public:
        if record.get('payload'):
            if hashlib.sha256((ROOT/record['payload']).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Original research payload checksum mismatch')
    credentials = {name:bool(os.environ.get(name)) for name in (
        'FOOTBALL_DATA_API_TOKEN','THE_ODDS_API_KEY','API_FOOTBALL_KEY')}
    tests = {}
    for name in ['tests','legacy-tests']:
        path = OUTPUT/f'{name}.xml'
        if path.exists():
            suite = ET.parse(path).getroot()
            suites = [suite] if suite.tag == 'testsuite' else list(suite.iter('testsuite'))
            tests[name] = {k: sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    records = {}
    for kind in ['list','recent','detail','odds']:
        payload = read(f'tests/fixtures/collectors/spftll/{kind}.json')['data']
        records[kind] = sum(len(g['list' if kind=='list' else 'matchList']) for g in payload) if isinstance(payload,list) else int(bool(payload))
    repeated = [read(f'tests/fixtures/collectors/spftll/repeated_detail_{n}.metadata.json') for n in (1,2)]
    report = {
        'report_kind':'role_A_collectors_evidence', 'generated_at':datetime.now(timezone.utc).isoformat(),
        'status':'PARTIAL', 'public_contract_integration':'PENDING_B_C_E',
        'production_database_writes_this_role_turn':0,
        'new_requests_recorded_by_A_scripts':len(public),
        'http_status_counts':dict(Counter(str(p.get('http_status')) for p in public)),
        'new_authenticated_data_requests':sum(bool(r.get('credential_sent',False)) for r in public),
        'credential_environment_presence_only':credentials,
        'fixture_samples_verified':len(fixture_manifest['samples']),
        'sources':{
            'spftll':{'implementation':'EXISTING_ADAPTER_PRESERVED','real_network':'PRIOR_VERIFIED_ARCHIVES_RECHECKED',
                      'sample_record_counts':records,'two_distinct_observations_same_content':
                      repeated[0]['observation']['id'] != repeated[1]['observation']['id'] and
                      repeated[0]['original_payload_sha256'] == repeated[1]['original_payload_sha256']},
            'football_data':{'implementation':'EXISTING_MATCH_COLLECTOR','status':'NEEDS_CREDENTIALS',
                             'public_catalog_count':len(catalog['competitions']),
                             'authenticated_matches_verified':False,
                             'pilot_catalog_codes':[c['code'] for c in catalog['competitions'] if c.get('code') in ('PL','BL1')]},
            'odds_api':{'implementation':'EXISTING_CURRENT_AND_HISTORICAL_COLLECTOR','status':'NEEDS_CREDENTIALS','real_odds_verified':False},
            'api_football':{'implementation':'RESEARCH_ONLY','status':'NEEDS_CREDENTIALS','paging':'SYNTHETIC_TESTED','real_data_verified':False},
            'understat':{'implementation':'DISCOVERY_AND_CAPTURE_TOOL','status':'PARTIAL',
                        'http_json_verified':True,'league':'EPL','season_start_year':2025,
                        'matches':len(matches),'unique_match_ids':len({m['id'] for m in matches}),
                        'teams':len(data['teams']),'player_aggregate_rows':len(data['players']),
                        'result_matches':sum(m['isResult'] is True for m in matches),
                        'complete_goal_pairs':goals_complete,'complete_xg_pairs':xg_complete,
                        'source_datetime_min':min(m['datetime'] for m in matches),'source_datetime_max':max(m['datetime'] for m in matches),
                        'raw_datetime_timezone':'UNCONFIRMED', 'source_published_at':None,
                        'team_history_lengths':{tid:len(t['history']) for tid,t in data['teams'].items()},
                        'team_appearance_counts':dict(appearances),'production_normalization_integrated':False},
            'clubelo':{'implementation':'RESEARCH_ONLY','network':'PRIOR_HTTP_502_EMPTY_BODY','real_elo_verified':False},
            'fbref':{'implementation':'RESEARCH_ONLY','network':'PRIOR_HTTP_403_CHALLENGE','current_advanced_fields_verified':False}},
        'tests':tests,
        'new_http_component_live_smoke':transport_live,
        'limitations':['No B contracts / C RawArchive / E path mapping published at verification time',
                       'Paid-provider keys absent at scoped audit; fixtures never substitute for authenticated acceptance',
                       'Small Understat request sample is not a long-term availability measurement',
                       'Historical rows were observed today; no backdated operational availability'],
    }
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'source-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'samples':report['fixture_samples_verified'],
                      'recorded_requests':len(public),'understat_matches':len(matches),'tests':tests}))


if __name__ == '__main__':
    main()
