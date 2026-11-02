"""Constructed official responses: no real samples and no HTTP in these tests."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path

import pytest

from football_research.data_adapters.official_fixture import (
    VERSION, _digest, prepare_official_fixture, verify_official_fixture,
)
from football_research.data_adapters.finland_stage_fixture import RESULTS_URL, RESULTS_PARAMS
from football_spider.storage.raw_archive import RawArchive
from test_finland_stage_fixture import schedule

NOW = datetime(2026,10,7,7,tzinfo=timezone.utc)
KICKOFF = datetime(2026,10,8,15,tzinfo=timezone.utc)
EVENT = 'veikkausliiga:2026:match:4353553'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def blocked(*args, **kwargs):
        raise AssertionError('This fixture contract test must not use the network')
    monkeypatch.setattr(socket,'getaddrinfo',blocked)
    monkeypatch.setattr(socket,'create_connection',blocked)
    monkeypatch.setattr(socket.socket,'connect',blocked)


class Transport:
    def __init__(self,body=None,*,status=200,error=None,interrupt=None,complete=True,extra=False):
        self.body = schedule() if body is None else body
        self.status,self.error,self.interrupt,self.complete,self.extra=status,error,interrupt,complete,extra
        self.calls=[]

    def __call__(self,spec,*,archive,clock):
        self.calls.append(deepcopy(spec))
        # Intent exists durably before any trusted transport is invoked.
        assert json.loads((archive.root/'requests/official_schedule.intent.json').read_text())==spec
        if self.interrupt=='before_raw':
            raise KeyboardInterrupt('constructed interruption before original raw commit')
        obs=archive.save(spec['source'],spec['url'],spec['params'],self.body,
            status=self.status,error_class=self.error,observed_at=clock(),context={
                'requested_at':spec['requested_at'],'preparation_request_digest':spec['request_digest'],
                'response_complete':self.complete})
        if self.extra:
            archive.save(spec['source'],spec['url'],spec['params'],self.body,status=200,observed_at=clock())
        if self.interrupt=='after_raw':
            raise KeyboardInterrupt('constructed interruption after original raw commit')
        return obs


def prepare(root,transport=None,query=EVENT,**kwargs):
    return prepare_official_fixture(query,output=root/'prepare',clock=lambda:NOW,
        transport=transport or Transport(),**kwargs)


def read_binding(result):
    return json.loads(Path(result['identity_path']).read_text(encoding='utf-8'))


def test_native_identity_uses_only_official_raw_with_no_market_initialization(tmp_path,monkeypatch):
    from football_research.data_adapters import prospective_text
    monkeypatch.setattr(prospective_text.MarketPriorPolicy,'from_manifest',
        lambda *a,**k:pytest.fail('An official identity must not instantiate a market policy'))
    transport=Transport()
    result=prepare(tmp_path,transport)
    assert result['status']=='ready' and result['match_id']==EVENT
    assert result['archived_http_responses']==result['request_intents']==len(transport.calls)==1
    assert transport.calls[0]['url']==RESULTS_URL and transport.calls[0]['params']==RESULTS_PARAMS
    binding=read_binding(result)
    assert binding['schema']==VERSION and binding['status']=='CONFIRMED'
    assert binding['canonical']['home_id']=='veikkausliiga:team:60731'
    assert binding['canonical']['away_id']=='veikkausliiga:team:60901'
    assert binding['canonical']['kickoff']==KICKOFF.isoformat()
    assert binding['prediction_cutoff']==(KICKOFF-timedelta(minutes=1)).isoformat()
    assert binding['observed_generic_entry_urls']==['https://www.veikkausliiga.com/','https://www.veikkausliiga.com/uutiset/']
    assert binding['canonical']['settlement_rule']==binding['canonical']['score_period']=='unknown'
    assert binding['settlement_rows']==[]
    encoded=json.dumps(binding)
    assert all(word not in encoded for word in ('initialization_manifest','probabilities','odds','spftll:match:'))
    assert verify_official_fixture(binding,archive_root=result['identity_archive'],as_of=NOW)==binding


@pytest.mark.parametrize('home,away',[('HJK','VPS'),('赫尔辛基','VPS瓦萨')])
def test_exact_registered_aliases_with_explicit_date_are_officially_resolved(tmp_path,home,away):
    result=prepare(tmp_path,query={'home':home,'away':away,'date':'2026-10-08'})
    assert result['status']=='ready' and result['source_event_id']==EVENT
    assert read_binding(result)['query']=={'home':'HJK','away':'VPS','date':'2026-10-08','timezone':'Asia/Shanghai'}


def test_named_query_date_uses_its_declared_timezone(tmp_path):
    q={'home':'HJK','away':'VPS','date':'2026-10-09','timezone':'Asia/Tokyo'}
    assert prepare(tmp_path/'tokyo',query=q)['status']=='ready'
    q['timezone']='UTC'
    assert prepare(tmp_path/'utc',query=q)['status']=='not_found'


def test_named_query_without_date_resolves_only_one_future_official_match(tmp_path):
    # An earlier occurrence of the pair remains in the raw page. Only the
    # predeclared future time window excludes it, not score or article quality.
    body=schedule()
    old=schedule(native='4353000',date='20261001',clock='18:00')
    old_row=old[old.index(b'<tr '):old.index(b'</tr>')+5]
    transport=Transport(body.replace(b'</table>',old_row+b'</table>'))
    result=prepare(tmp_path,transport,query={'home':'HJK','away':'VPS'})
    assert result['status']=='ready' and result['match_id']==EVENT
    assert result['query']['date'] is None and len(transport.calls)==1


def test_named_query_without_date_keeps_every_eligible_occurrence_and_reports_ambiguity(tmp_path):
    body=schedule()
    other=schedule(native='4353999',date='20261009',clock='19:00',score='-')
    other_row=other[other.index(b'<tr '):other.index(b'</tr>')+5]
    transport=Transport(body.replace(b'</table>',other_row+b'</table>'))
    result=prepare(tmp_path,transport,query={'home':'HJK','away':'VPS','date':None,'timezone':'UTC'})
    assert result['status']=='ambiguous' and len(result['candidates'])==2
    assert len(transport.calls)==1 and not Path(result['identity_path']).exists()


@pytest.mark.parametrize('query',['spftll:match:3085211','4353553','veikkausliiga:2025:match:4353553',
    {'home':'HJK Women','away':'VPS','date':'2026-10-08'},
    {'home':'H J K','away':'VPS','date':'2026-10-08'}])
def test_unregistered_namespace_or_team_does_not_trigger_http(tmp_path,query):
    transport=Transport()
    result=prepare(tmp_path,transport,query)
    assert result['status']=='unsupported' and len(transport.calls)==0
    assert result['archived_http_responses']==result['request_intents']==0
    assert not Path(result['identity_path']).exists()


@pytest.mark.parametrize('old,new',[
    ('Otteluohjelma | Veikkausliiga','Youth League'),
    ('selected value="2026">2026','selected value="2025">2025'),
    ('selected value="2">Mestaruussarja','selected value="1">Runkosarja'),
    ('value="60731">HJK','value="777">HJK'),
    ('>HJK - VPS</a>','>HJK U17 - VPS</a>'),
    ('4353553/seuranta/','777/seuranta/'),
])
def test_identity_scope_and_native_team_registry_are_checked_from_body(tmp_path,old,new):
    result=prepare(tmp_path,Transport(schedule().replace(old.encode(),new.encode())))
    assert result['status']=='unsupported' and not Path(result['identity_path']).exists()


@pytest.mark.parametrize('score',['0 — 0','2 - 1','Siirretty'])
def test_score_or_non_scheduled_state_never_becomes_a_prematch_identity(tmp_path,score):
    result=prepare(tmp_path,Transport(schedule(score=score)))
    assert result['status']=='not_prematch' and not Path(result['identity_path']).exists()


def test_duplicate_rows_remain_ambiguous_without_outcome_or_quality_selection(tmp_path):
    body=schedule()
    row=body[body.index(b'<tr '):body.index(b'</tr>')+5]
    result=prepare(tmp_path,Transport(body.replace(b'</table>',row+b'</table>')))
    assert result['status']=='ambiguous' and len(result['candidates'])==2
    assert not Path(result['identity_path']).exists()


@pytest.mark.parametrize('kickoff,expected',[
    (NOW+timedelta(minutes=29),'not_prematch'),
    (NOW+timedelta(minutes=30),'ready'),
    (NOW+timedelta(days=7),'ready'),
    (NOW+timedelta(days=7,minutes=1),'not_prematch'),
])
def test_source_kickoff_defines_the_fixed_future_horizon(tmp_path,kickoff,expected):
    from zoneinfo import ZoneInfo
    local=kickoff.astimezone(ZoneInfo('Europe/Helsinki'))
    result=prepare(tmp_path,Transport(schedule(date=local.strftime('%Y%m%d'),clock=local.strftime('%H:%M'))))
    assert result['status']==expected


def test_legacy_physical_event_keeps_the_existing_palloliitto_namespace(tmp_path):
    body=schedule('IF Gnistan','FC Inter','4353542','20261007','19:00')
    result=prepare(tmp_path,Transport(body),'veikkausliiga:2026:match:4353542')
    binding=read_binding(result)
    assert result['match_id']==binding['canonical']['source_event_id']=='palloliitto:match:4353542'
    assert binding['canonical']['home_id']=='finland:veikkausliiga:2026:men:if-gnistan'
    assert binding['canonical']['away_id']=='finland:veikkausliiga:2026:men:fc-inter'
    assert binding['native_event_id']=='4353542'
    assert verify_official_fixture(binding,archive_root=result['identity_archive'],as_of=NOW)==binding
    wrong=prepare(tmp_path/'wrong',Transport(schedule(native='4353542')),'veikkausliiga:2026:match:4353542')
    assert wrong['status']=='unsupported'


def test_unknown_fixture_id_and_reversed_named_match_not_guessed(tmp_path):
    assert prepare(tmp_path/'id',query='veikkausliiga:2026:match:9999')['status']=='not_found'
    assert prepare(tmp_path/'order',query={'home':'VPS','away':'HJK','date':'2026-10-08'})['status']=='not_found'


def test_successful_resume_uses_same_raw_snapshot_and_cannot_change_query_or_budget(tmp_path):
    transport=Transport()
    first=prepare(tmp_path,transport)
    second=prepare_official_fixture(EVENT,output=tmp_path/'prepare',clock=lambda:NOW+timedelta(days=3),
        transport=transport,resume=True)
    assert second==first and len(transport.calls)==1
    assert read_binding(second)['verified_at']==NOW.isoformat()
    with pytest.raises(ValueError,match='Resume|resume'):
        prepare(tmp_path,transport,query='veikkausliiga:2026:match:999',resume=True)
    with pytest.raises(ValueError,match='resume'):
        prepare(tmp_path,transport,max_http_requests=0,resume=True)


@pytest.mark.parametrize('transport',[Transport(status=503),Transport(error='TLSFailure'),Transport(complete=False)])
def test_failed_or_incomplete_actual_response_is_kept_and_never_retried(tmp_path,transport):
    first=prepare(tmp_path,transport)
    assert first['status']=='unavailable' and first['archived_http_responses']==1
    assert prepare(tmp_path,transport,resume=True)==first and len(transport.calls)==1
    assert not Path(first['identity_path']).exists()


def test_empty_budget_does_not_create_a_request_or_identity(tmp_path):
    transport=Transport()
    result=prepare(tmp_path,transport,max_http_requests=0)
    assert result['status']=='budget_exhausted' and len(transport.calls)==0


def test_pending_request_without_raw_requires_reconciliation_not_retry(tmp_path):
    transport=Transport(interrupt='before_raw')
    with pytest.raises(KeyboardInterrupt):
        prepare(tmp_path,transport)
    result=prepare(tmp_path,transport,resume=True)
    assert result['status']=='interrupted' and len(transport.calls)==1
    assert result['request_intents']==1 and result['archived_http_responses']==0


def test_raw_committed_before_interruption_can_be_adopted_without_http(tmp_path):
    transport=Transport(interrupt='after_raw')
    with pytest.raises(KeyboardInterrupt):
        prepare(tmp_path,transport)
    result=prepare(tmp_path,transport,resume=True)
    assert result['status']=='ready' and len(transport.calls)==1
    assert verify_official_fixture(read_binding(result),archive_root=result['identity_archive'],as_of=NOW)


def test_extra_raw_response_cannot_be_hidden_by_the_one_request_receipt(tmp_path):
    with pytest.raises(ValueError,match='exactly'):
        prepare(tmp_path,Transport(extra=True))
    assert not (tmp_path/'prepare/identity.json').exists()


@pytest.mark.parametrize('change',['canonical','alias','cutoff','receipt','raw','extra_raw','extra_intent','future_audit'])
def test_verification_reparses_raw_and_rejects_rehashed_fabrication(tmp_path,change):
    result=prepare(tmp_path)
    binding=read_binding(result)
    archive=RawArchive(Path(result['identity_archive']))
    observation=archive.observations()[0]
    audit=NOW
    if change=='canonical':binding['canonical']['home_name']='KuPS'
    elif change=='alias':binding['aliases']['home'].append('HJK Women')
    elif change=='cutoff':binding['prediction_cutoff']=(KICKOFF+timedelta(minutes=1)).isoformat()
    elif change=='receipt':
        path=Path(result['request_receipt_path'])
        receipt=json.loads(path.read_text());receipt['http_attempts']=0
        path.write_text(json.dumps(receipt))
    elif change=='raw':
        (archive.root/observation.payload_path).write_bytes(gzip.compress(schedule()+b' changed',mtime=0))
    elif change=='extra_raw':
        archive.save('open_page',RESULTS_URL,RESULTS_PARAMS,schedule(),status=200,observed_at=NOW)
    elif change=='extra_intent':
        (archive.root/'requests/unaccounted.intent.json').write_text('{}')
    else:audit=NOW-timedelta(seconds=1)
    binding['binding_sha256']=_digest({k:v for k,v in binding.items() if k!='binding_sha256'})
    with pytest.raises(ValueError):
        verify_official_fixture(binding,archive_root=archive.root,as_of=audit)


@pytest.mark.parametrize('offset',[-1,1])
def test_recovery_cannot_write_identity_using_an_unavailable_derivation_clock(tmp_path,offset):
    transport=Transport(interrupt='after_raw')
    with pytest.raises(KeyboardInterrupt):
        prepare(tmp_path,transport)
    (tmp_path/'prepare/identity-clock.json').write_text(json.dumps({'at':(NOW+timedelta(seconds=offset)).isoformat()}))
    with pytest.raises(ValueError,match='derivation'):
        prepare(tmp_path,transport,resume=True)
    assert len(transport.calls)==1 and not (tmp_path/'prepare/identity.json').exists()
