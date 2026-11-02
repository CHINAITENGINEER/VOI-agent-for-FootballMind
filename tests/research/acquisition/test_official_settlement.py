from datetime import timedelta
import gzip
import json

import pytest

from football_research.data_adapters.official_settlement import (
    official_api_event_id, project_official_api_result, SETTLEMENT_VERSION)
from football_spider.storage.raw_archive import RawArchive
from football_research.data_adapters.verified_fixtures import main
from test_verified_fixtures import (
    AFTER, KICKOFF, identity, save, verify, source_objects, body, result)

API = 'https://match.uefa.com/v5/matches/2047946'


def native_result():
    value = source_objects(status='Finished')[1]
    value['status'] = 'FINISHED'
    value['competition'] = {'id':'2014', 'code':'UNL', 'age':'ADULT', 'sex':'MALE',
                           'sportsType':'FOOTBALL', 'teamCategory':'NATIONAL'}
    value['round'] = {'mode':'GROUP', 'competitionId':'2014', 'seasonYear':'2027',
                     'metaData':{'name':'League phase', 'type':'GROUP_STANDINGS'}}
    value['group'] = {'competitionId':'2014', 'seasonYear':'2027', 'teams':['56370','122']}
    value['competitionPhase'] = 'TOURNAMENT'
    value['type'] = 'GROUP_STAGE'
    value['score'] = {'regular':{'home':0,'away':0}, 'total':{'home':0,'away':0}}
    return value


def acquire(archive, value=None, *, endpoint=API, at=AFTER, raw=None):
    return save(archive, raw if raw is not None else json.dumps(value or native_result()).encode(),
                endpoint=endpoint, at=at)


def test_api_exact_identity_regular_zero_draw_exports_one_raw_bound_label(tmp_path):
    original, binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    obs = acquire(results)
    checked = verify(binding, original, results)
    assert checked['status'] == 'VERIFIED'
    assert checked['schema'] == SETTLEMENT_VERSION
    row, = checked['settlement_rows']
    assert row['verification_schema'] == SETTLEMENT_VERSION
    assert (row['home_goals'],row['away_goals'],row['score_period']) == (0,0,'regulation')
    assert row['observed_at'] == AFTER.isoformat()
    assert row['available_at'] == (AFTER+timedelta(minutes=1)).isoformat()
    assert len(row['input_manifest']) == 1 and obs.id in row['input_manifest'][0]


@pytest.mark.parametrize('attack', ['event','teams','name','case','competition','category','season',
                                   'knockout','group','kickoff','future_finish','short_duration'])
def test_api_identity_stage_and_final_clocks_fail_closed(tmp_path, attack):
    original, binding = identity(tmp_path)
    results, data = RawArchive(tmp_path/'results'), native_result()
    if attack == 'event': data['id'] = '2047948'
    elif attack == 'teams': data['homeTeam'],data['awayTeam'] = data['awayTeam'],data['homeTeam']
    elif attack == 'name': data['homeTeam']['internationalName'] = 'France'
    elif attack == 'case': data['homeTeam']['internationalName'] = 'CROATIA'
    elif attack == 'competition': data['competition']['id'] = '1'
    elif attack == 'category': data['competition']['age'] = 'UNDER21'
    elif attack == 'season': data['seasonYear'] = '2025'
    elif attack == 'knockout': data['type'] = 'KNOCK_OUT'
    elif attack == 'group': data['group']['teams'] = ['56370','1']
    elif attack == 'kickoff': data['kickOffTime']['dateTime'] = (KICKOFF+timedelta(hours=1)).isoformat()
    elif attack == 'future_finish': data['fullTimeAt'] = (AFTER+timedelta(minutes=5)).isoformat()
    else: data['fullTimeAt'] = (KICKOFF+timedelta(minutes=80)).isoformat()
    acquire(results, data)
    checked = verify(binding, original, results)
    assert checked['status'] == 'DISPUTED' and checked['settlement_rows'] == []


@pytest.mark.parametrize('score', [True,False,-1,100,0.0,'0',None,float('nan'),float('inf')])
def test_api_bool_string_nonfinite_and_invalid_scores_never_settle(tmp_path, score):
    original, binding = identity(tmp_path)
    results, data = RawArchive(tmp_path/'results'), native_result()
    data['score']['regular']['home'] = score
    data['score']['total']['home'] = score
    acquire(results,data)
    checked = verify(binding, original, results)
    assert checked['status'] != 'VERIFIED' and checked['settlement_rows'] == []


@pytest.mark.parametrize('kind', ['missing_regular','total_conflict','extra_time','penalties','duplicate'])
def test_api_period_ambiguity_and_duplicate_fields_never_settle(tmp_path, kind):
    original, binding = identity(tmp_path)
    results, data = RawArchive(tmp_path/'results'), native_result()
    if kind == 'missing_regular': del data['score']['regular']
    elif kind == 'total_conflict': data['score']['total']['home'] = 1
    elif kind == 'extra_time': data['score']['extraTime'] = {'home':0,'away':0}
    elif kind == 'penalties': data['score']['penalty'] = {'home':5,'away':4}
    raw = json.dumps(data).encode()
    if kind == 'duplicate': raw = raw[:-1]+b',"status":"FINISHED"}'
    acquire(results,raw=raw)
    checked = verify(binding, original, results)
    assert checked['status'] != 'VERIFIED' and checked['settlement_rows'] == []


@pytest.mark.parametrize('status,expected', [('LIVE','PENDING'),('POSTPONED','POSTPONED'),
    ('CANCELLED','CANCELLED'),('ABANDONED','DISPUTED'),('AWARDED','DISPUTED')])
def test_api_nonfinished_numeric_scores_cannot_create_labels(tmp_path,status,expected):
    original, binding = identity(tmp_path)
    results, data = RawArchive(tmp_path/'results'), native_result()
    data['status'] = status
    acquire(results,data)
    checked = verify(binding, original, results)
    assert checked['status'] == expected and checked['settlement_rows'] == []


def test_api_wrong_endpoint_ignored_and_conflicting_supported_score_disputed(tmp_path):
    original, binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    acquire(results, endpoint=API.replace('2047946','2047948'))
    assert verify(binding,original,results)['status'] == 'PENDING'
    acquire(results)
    result(results,text='Croatia 1-0 Spain')
    checked = verify(binding,original,results)
    assert checked['status'] == 'DISPUTED' and checked['settlement_rows'] == []


def test_api_future_receipt_and_raw_mutation_are_rejected(tmp_path):
    original,binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    obs = acquire(results)
    with pytest.raises(ValueError,match='unavailable'):
        verify(binding,original,results,as_of=AFTER-timedelta(seconds=1))
    (results.root/obs.payload_path).write_bytes(gzip.compress(b'changed'))
    with pytest.raises(ValueError,match='checksum'):
        verify(binding,original,results)


def test_api_missing_request_clock_and_query_receipts_are_not_promoted(tmp_path):
    original,binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    raw = json.dumps(native_result()).encode()
    obs = results.save('open_page',API,{},raw,status=200,observed_at=AFTER)
    with pytest.raises(ValueError,match='request/receipt'):
        project_official_api_result(obs,raw,binding['canonical'],as_of=AFTER)
    queried = obs.model_copy(update={'params':{'fields':['score']},
        'context':{'requested_at':(AFTER-timedelta(seconds=1)).isoformat()}})
    assert project_official_api_result(queried,raw,binding['canonical'],as_of=AFTER)['status'] == 'UNCONFIRMED'


@pytest.mark.parametrize('url', [API+'/',API+'?score=1',API+'#score',API.replace('https:','http:'),
    API.replace('match.uefa.com','match.uefa.com.example.org'),API.replace('match.uefa.com','user@match.uefa.com'),
    API.replace('match.uefa.com','match.uefa.com:443')])
def test_api_endpoint_is_exact(url):
    assert official_api_event_id(url) is None


def test_legacy_html_remains_v1_when_no_target_api_is_present(tmp_path):
    original,binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    save(results,body(status='Finished'),at=AFTER)
    result(results)
    checked = verify(binding,original,results)
    assert checked['status'] == 'VERIFIED'
    assert checked['schema'] == 'raw-backed-independent-settlement-v1'
    assert len(checked['settlement_rows'][0]['input_manifest']) == 2


def test_cli_marks_api_verification_wrapper_and_rows_v2(tmp_path):
    original,binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    acquire(results)
    bundle,output = tmp_path/'bundle.json',tmp_path/'verified.json'
    bundle.write_text(json.dumps({'bindings':{binding['match_id']:binding}}),encoding='utf-8')
    assert main(['verify-results','--identity-bundle',str(bundle),
        '--identity-archive',str(original.root),'--result-archive',str(results.root),
        '--output',str(output)]) == 0
    checked = json.loads(output.read_text(encoding='utf-8'))
    assert checked['schema'] == SETTLEMENT_VERSION
    assert checked['settlement_rows'][0]['verification_schema'] == SETTLEMENT_VERSION
