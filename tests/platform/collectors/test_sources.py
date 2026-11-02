from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest
from football_platform.collectors.policy import retry_delay, football_data_windows, odds_credit_estimate, api_football_next_page
from football_platform.collectors.envelopes import response_count, SourceEnvelopeError

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT/'tests/fixtures/collectors'


def test_retry_after_http_date_and_invalid_values():
    now = datetime(2026,9,20,tzinfo=timezone.utc)
    assert retry_delay(1,status=429,retry_after='Sun, 20 Sep 2026 00:00:05 GMT',now=now) == 5
    for invalid in ['not-a-date','NaN','inf','-10']:
        assert retry_delay(1,status=429,retry_after=invalid,now=now) == 1
    assert retry_delay(3,status=500) is None


def test_local_day_expands_to_utc_without_gaps():
    assert football_data_windows(date(2026,9,20),date(2026,9,20)) == [('2026-09-19','2026-09-21')]
    windows=football_data_windows(date(2026,9,1),date(2026,9,28))
    assert windows[0][0] == '2026-08-31' and windows[-1][1] == '2026-09-29'
    assert all(a[1] == b[0] for a,b in zip(windows,windows[1:]))
    assert football_data_windows(date(2026,9,20),date(2026,9,20),'UTC') == [('2026-09-20','2026-09-21')]


def test_dst_day_edges_are_timezone_aware():
    assert football_data_windows(date(2026,11,1),date(2026,11,1),'America/New_York') == [('2026-11-01','2026-11-03')]
    with pytest.raises(ValueError):
        football_data_windows(date(2026,9,21),date(2026,9,20))


def test_odds_credit_budget_separates_actual_billing():
    assert odds_credit_estimate('h2h,spreads,totals','uk,eu') == 6
    assert odds_credit_estimate('h2h,h2h','eu',historical=True) == 10
    with pytest.raises(ValueError):
        odds_credit_estimate('unknown','eu')


def test_documented_paging_synthetic_cases():
    assert api_football_next_page({'paging':{'current':1,'total':2},'errors':[]},1) == 2
    assert api_football_next_page({'paging':{'current':2,'total':2}},2) is None
    for payload,page in [({'paging':{'current':1,'total':3}},2), ({},1), ({'paging':{'current':0,'total':1}},1), ({'errors':{'quota':'reached'}},1)]:
        with pytest.raises(ValueError): api_football_next_page(payload,page)
    with pytest.raises(ValueError):
        api_football_next_page({'paging':{'current':2,'total':3}},2,max_pages=2)


@pytest.mark.parametrize('source,kind,body',[
    ('football_data','matches',b'{"matches":[]}'),
    ('odds_api','current',b'[]'),
    ('odds_api','historical',b'{"timestamp":"2026-09-20T00:00:00Z","data":[]}'),
    ('spftll','list',b'{"code":0,"data":[]}'),
    ('spftll','odds',b'{"code":0,"data":{"footballMoreSpInfo":null}}'),
    ('understat','league',b'{"dates":[],"teams":{},"players":[]}'),
])
def test_synthetic_empty_data_not_failure(source,kind,body):
    assert response_count(source,kind,body) == 0


@pytest.mark.parametrize('source,kind,body,category',[
    ('football_data','matches',b'{"matches":{}}','SCHEMA_DRIFT'),
    ('football_data','matches',b'{"errorCode":403,"message":"denied"}','PROVIDER_ERROR'),
    ('odds_api','current',b'{"error_code":"INVALID_KEY"}','PROVIDER_ERROR'),
    ('spftll','list',b'{"code":5,"data":[]}','PROVIDER_ERROR'),
    ('spftll','list',b'{"code":0,"data":[{"renamed_list":[]}]}','SCHEMA_DRIFT'),
    ('understat','league',b'{"renamed_dates":[],"teams":{},"players":[]}','SCHEMA_DRIFT'),
])
def test_synthetic_error_and_field_drift(source,kind,body,category):
    with pytest.raises(SourceEnvelopeError,match=category): response_count(source,kind,body)


def test_all_exported_real_fixture_hashes_and_provenance():
    manifest=json.loads((FIXTURES/'manifest.json').read_bytes())
    assert len(manifest['samples']) >= 12
    for item in manifest['samples']:
        path=ROOT/item['metadata']; metadata=json.loads(path.read_bytes())
        assert metadata['sample_kind'] == 'real_capture_redacted_copy'
        assert metadata['production_ingestion_allowed'] is False
        body=(path.parent/metadata['payload_file']).read_bytes()
        assert hashlib.sha256(body).hexdigest() == metadata['fixture_payload_sha256']
        assert datetime.fromisoformat(item['observed_at']).tzinfo


def test_actual_spftll_envelopes_and_repeated_observations():
    for kind in ['list','recent','detail','odds']:
        assert response_count('spftll',kind,(FIXTURES/f'spftll/{kind}.json').read_bytes()) > 0
    first=json.loads((FIXTURES/'spftll/repeated_detail_1.metadata.json').read_bytes())
    second=json.loads((FIXTURES/'spftll/repeated_detail_2.metadata.json').read_bytes())
    assert first['original_payload_sha256'] == second['original_payload_sha256']
    assert first['observation']['id'] != second['observation']['id']
    assert first['observation']['observed_at'] < second['observation']['observed_at']


def test_actual_catalog_does_not_prove_match_entitlement():
    body=(FIXTURES/'football_data/public_competitions.json').read_bytes()
    data=json.loads(body)
    assert response_count('football_data','catalog',body) == len(data['competitions']) == data['count']
    assert {'PL','BL1'} <= {c.get('code') for c in data['competitions']}
    with pytest.raises(SourceEnvelopeError): response_count('football_data','matches',body)


def test_actual_understat_full_season_structural_coverage():
    body=(FIXTURES/'understat/epl_2025.json').read_bytes(); data=json.loads(body)
    assert response_count('understat','league',body) == 380
    assert len({m['id'] for m in data['dates']}) == 380
    assert len(data['teams']) == 20
    assert all(len(team['history']) == 38 for team in data['teams'].values())
    for match in data['dates']:
        assert match['isResult'] is True
        assert match['h']['id'] != match['a']['id']
        for side in ['h','a']:
            assert int(match['goals'][side]) >= 0 and float(match['xG'][side]) >= 0


def test_synthetic_handoff_samples_are_explicitly_not_real():
    manifest=json.loads((FIXTURES/'synthetic-manifest.json').read_bytes())
    assert manifest['sample_kind']=='synthetic_test_only'
    assert manifest['production_ingestion_allowed'] is False and manifest['real_network_acceptance'] is False
    fd=(FIXTURES/'football_data/synthetic_matches.json').read_bytes()
    assert response_count('football_data','matches',fd)==3
    records=json.loads(fd)['matches']
    assert records[0]['score']['fullTime']['home']==0
    assert records[1]['score']['regularTime'] != records[1]['score']['fullTime']
    assert records[2]['score']['fullTime']['home'] is None
    odds=(FIXTURES/'odds_api/synthetic_current.json').read_bytes()
    assert response_count('odds_api','current',odds)==1
