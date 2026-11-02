"""Synthetic transport fixtures test admission; they are never real samples."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import shutil

import pytest

from football_research.data_adapters.finland_fixture import (
    CALENDAR_LINK, CALENDAR_URL, FIXTURES_URL, bind_finland_fixture,
)
from football_research.data_adapters.finland_results import (
    RESULTS_PARAMS, RESULTS_URL, RULE_PACKAGE_ROOT, RULE_URLS, VERSION,
    copy_rule_observations, match_followup_url, project_championship_schedule,
    project_league_match, verify_finland_settlement, verify_rule_package,
)
from football_research.data_adapters.prospective_text import MarketPriorPolicy
from football_research.data_adapters.types import ResearchMatch, ResearchOddsSnapshot, json_record
from football_spider.storage.raw_archive import RawArchive

UTC = timezone.utc
AS_OF = datetime.now(UTC)
BEFORE = datetime(2026, 10, 5, 10, tzinfo=UTC)
KICKOFF = datetime(2026, 10, 5, 16, tzinfo=UTC)
AFTER = KICKOFF + timedelta(hours=2, minutes=10)
DETAIL_URL = RESULTS_URL+'4353542/seuranta/'


def save(archive, url, body, *, at=AFTER, params=None, status=200):
    return archive.save('open_page', url, params or {}, body, status=status, observed_at=at,
        context={'requested_at':(at-timedelta(seconds=1)).isoformat()})


def identity(root):
    match=ResearchMatch('spftll:match:3085210','spftll:competition:13',None,
        'spftll:team:2429','spftll:team:386',KICKOFF,source='spftll',status='scheduled',
        competition_name='芬超',home_team_name='赫尔火花',away_team_name='国际图尔',
        fixture_available_at=BEFORE-timedelta(minutes=2),input_manifest=('test:fixture',))
    quote=ResearchOddsSnapshot(match.match_id,BEFORE-timedelta(minutes=1),3.58,3.55,1.76,
                              input_manifest=('test:quote',))
    initial=MarketPriorPolicy(json_record(match),json_record(quote),frozen_at=BEFORE,
                             prediction_cutoff=KICKOFF-timedelta(minutes=1)).manifest
    schedule=('''<h1 class="entry-title">Miesten ottelut</h1><h2>Mestaruussarja 2026</h2>
        <figure><table><tr><td>Päivä</td><td>PVM.</td><td>Aika</td><td>Paikka</td><td>Koti</td>
        <td>Vieras</td><td>Tulos</td><td>Ennakko</td><td>Raportti</td></tr>
        <tr><td>Ma</td><td>5.10</td><td>19.00</td><td>Stadion</td><td>IF Gnistan</td>
        <td>FC Inter</td><td></td><td></td><td></td></tr></table></figure>
        <a href="'''+CALENDAR_LINK+'">Calendar</a>').encode()
    calendar=b'''BEGIN:VCALENDAR
VERSION:2.0
NAME:IF Gnistan \xe2\x80\x93 Veikkausliiga
X-WR-CALNAME:IF Gnistan \xe2\x80\x93 Veikkausliiga
X-WR-TIMEZONE:Europe/Helsinki
BEGIN:VEVENT
UID:4353542@spl.torneopal.fi
CATEGORIES:Veikkausliiga
SUMMARY:IF Gnistan \xe2\x80\x93 FC Inter, Veikkausliiga
URL:https://tulospalvelu.palloliitto.fi/match/4353542
DESCRIPTION:https://tulospalvelu.palloliitto.fi/match/4353542
STATUS:CONFIRMED
DTSTART:20261005T160000Z
END:VEVENT
END:VCALENDAR'''
    archive=RawArchive(root/'identity')
    ids=[save(archive,FIXTURES_URL,schedule,at=BEFORE+timedelta(minutes=1)).id,
         save(archive,CALENDAR_URL,calendar,at=BEFORE+timedelta(minutes=1)).id]
    binding=bind_finland_fixture(initial,archive_root=archive.root,observation_ids=ids,
                                as_of=BEFORE+timedelta(minutes=2))
    assert binding['status']=='CONFIRMED'
    return archive,binding


def league_schedule(score='2 — 0'):
    return f'''<html><select id="season"><option value="2026" selected>2026</option></select>
    <select id="category"><option value="veikkausliiga" selected>Veikkausliiga</option></select>
    <select name="group"><option value="2" selected>Mestaruussarja</option></select>
    <table><tr data-time="20261005"><td>10</td><td>Ma 5.10.2026</td><td>19:00</td><td>5.10.2026</td>
    <td><a href="/tilastot/2026/veikkausliiga/ottelut/4353542/">IF Gnistan - FC Inter</a></td>
    <td><a href="/tilastot/2026/veikkausliiga/ottelut/4353542/seuranta/">Seuranta</a></td>
    <td><span>{score}</span></td><td>1500</td></tr></table></html>'''.encode()


def terminal(home=2, away=0):
    return f'''<html><head><meta property="og:url" content="{DETAIL_URL}"></head><main>
    <section class="game-header"><div class="head finished">
    <div class="home"><div class="team-name"><a>IF Gnistan</a></div></div>
    <div class="away"><div class="team-name"><a>FC Inter</a></div></div>
    <span id="home-goals">{home}</span><span id="away-goals">{away}</span>
    <div class="game-state">Pelattu 05.10.2026</div></div></section>
    <div class="game-events"><p class="info center">Ottelu päättyi</p>
    <p class="info center">Puoliaika päättyi</p>
    <div class="regular-event time"><span>90' + 3</span></div>
    <div class="regular-event time"><span>46'</span></div>
    <p class="info center">Puoliaika alkoi</p><p class="info center">Puoliaika päättyi</p>
    <div class="regular-event time"><span>45'</span></div>
    <p class="info center">Puoliaika alkoi</p><p class="info center">Ottelu alkoi</p></div>
    </main></html>'''.encode()


def verify(root, *, schedule_body=None, terminal_body=None, rules=True, at=AFTER):
    archived_identity,binding=identity(root)
    results=RawArchive(root/'results')
    if rules: copy_rule_observations(results,as_of=AS_OF)
    save(results,RESULTS_URL,league_schedule() if schedule_body is None else schedule_body,
         params=RESULTS_PARAMS,at=at+timedelta(seconds=1))
    save(results,DETAIL_URL,terminal() if terminal_body is None else terminal_body,at=at)
    report=verify_finland_settlement(binding,identity_archive_root=archived_identity.root,
        result_archive_root=results.root,observation_ids=[o.id for o in results.observations()],as_of=AS_OF)
    return report,results,binding,archived_identity


def test_complete_result_uses_actual_clocks_and_independent_rule_raw(tmp_path):
    report,results,binding,ident=verify(tmp_path)
    assert report['status']=='VERIFIED' and report['schema']==VERSION
    row=report['settlement_rows'][0]
    assert (row['home_goals'],row['away_goals'])==(2,0)
    assert row['available_at']==AS_OF.isoformat()
    assert row['score_period']=='regulation'
    assert row['identity_binding_sha256']==binding['binding_sha256']
    terminal_projection=next(p for p in report['projections'] if p['status']=='FINISHED_SCORE')
    assert terminal_projection['full_time_at'] is None
    assert terminal_projection['finished_available_at']==AFTER.isoformat()
    assert len(row['input_manifest'])==5
    assert match_followup_url(binding)==DETAIL_URL
    repeated=verify_finland_settlement(binding,identity_archive_root=ident.root,result_archive_root=results.root,
        observation_ids=[o.id for o in results.observations()],as_of=AS_OF)
    assert repeated==report


def test_zero_zero_is_valid_and_final_minute_event_is_not_required(tmp_path):
    report,_,_,_=verify(tmp_path,schedule_body=league_schedule('0 — 0'),
        terminal_body=terminal(0,0).replace(b"90' + 3",b"89'"))
    assert report['status']=='VERIFIED'
    assert report['settlement_rows'][0]['home_goals']==report['settlement_rows'][0]['away_goals']==0


@pytest.mark.parametrize('before,after',[
    ('head finished','head live'),
    ('Pelattu 05.10.2026','Pelattu 06.10.2026'),
    ('<a>IF Gnistan</a>','<a>FC Inter</a>'),
    ('<span id="home-goals">2</span>','<span id="home-goals">False</span>'),
    ('<span id="home-goals">2</span>','<span id="home-goals">NaN</span>'),
    ('<span id="home-goals">2</span>','<span id="home-goals">-1</span>'),
    ('<span id="home-goals">2</span>','<span id="home-goals">2.0</span>'),
    ('Ottelu päättyi','Ottelu keskeytettiin'),
    ('90\x27 + 3','105\x27'),
    ('90\x27 + 3','46\x27 + 10'),
    ('Puoliaika alkoi','Jatkoaika alkoi'),
    ('Puoliaika päättyi','Rangaistuspotkukilpailu päättyi'),
    ('<div class="game-events">','<div class="missing-events">'),
    ('<p class="info center">Ottelu päättyi</p>',''),
    ('content="'+DETAIL_URL+'"','content="'+DETAIL_URL.replace('4353542','123')+'"'),
])
def test_terminal_identity_period_or_finish_must_be_explicit(tmp_path,before,after):
    assert before.encode() in terminal()
    report,_,_,_=verify(tmp_path,terminal_body=terminal().replace(before.encode(),after.encode()))
    assert report['status']!='VERIFIED' and report['settlement_rows']==[]


@pytest.mark.parametrize('before,after',[
    ('value="2" selected>Mestaruussarja','value="3" selected>Karsintasarja'),
    ('data-time="20261005"','data-time="20261006"'),
    ('19:00','18:00'),
    ('IF Gnistan - FC Inter','FC Inter - IF Gnistan'),
    ('4353542/','9999/'),
    ('2026" selected>2026','2025" selected>2025'),
    ('2 — 0','-'),
    ('2 — 0','3 — 0'),
    ('2 — 0','2 — 0 (penalties 5-4)'),
])
def test_native_group_time_order_and_scores_must_agree(tmp_path,before,after):
    assert before.encode() in league_schedule()
    report,_,_,_=verify(tmp_path,schedule_body=league_schedule().replace(before.encode(),after.encode()))
    assert report['status']!='VERIFIED' and report['settlement_rows']==[]


def test_future_or_premature_finished_response_cannot_be_a_label(tmp_path):
    report,_,_,_=verify(tmp_path,at=KICKOFF+timedelta(minutes=30))
    assert report['status']=='DISPUTED' and report['settlement_rows']==[]
    _,binding=identity(tmp_path/'future')
    archive=RawArchive(tmp_path/'future-results')
    obs=save(archive,DETAIL_URL,terminal(),at=AS_OF+timedelta(seconds=1))
    with pytest.raises(ValueError,match='unavailable'):
        project_league_match(obs,archive.read_body(obs),binding['canonical'],as_of=AS_OF)


def test_missing_rules_never_inherits_uefa_rule(tmp_path):
    report,_,_,_=verify(tmp_path,rules=False)
    assert report['status']=='PENDING' and report['regulation_rule'] is None
    assert set(report['missing_rule_sources'])==set(RULE_URLS)
    assert report['settlement_rows']==[]


def test_rule_copy_preserves_original_ids_and_clocks_and_old_asof_is_pending(tmp_path):
    old=datetime(2026,10,7,5,39,tzinfo=UTC)
    check=verify_rule_package(as_of=old)
    assert check['status']=='PENDING' and check['observation_ids']==[]
    copied=copy_rule_observations(tmp_path/'old',as_of=old)
    assert copied['copied_observation_ids']==[] and copied['http_requests']==0
    assert not (tmp_path/'old').exists()
    copied=copy_rule_observations(tmp_path/'now',as_of=AS_OF)
    originals={o.id:o.model_dump() for o in RawArchive(RULE_PACKAGE_ROOT).observations()}
    assert {o.id:o.model_dump() for o in RawArchive(tmp_path/'now').observations()}==originals
    assert copied['http_requests']==0
    ident,binding=identity(tmp_path/'identity-case')
    unrelated=RawArchive(tmp_path/'unrelated')
    obs=save(unrelated,'https://www.uefa.com/example/',b'other',at=BEFORE)
    report=verify_finland_settlement(binding,identity_archive_root=ident.root,
        result_archive_root=unrelated.root,observation_ids=[obs.id],as_of=old)
    assert report['status']=='PENDING' and report['settlement_rows']==[]


def test_raw_and_rule_package_tamper_are_rejected(tmp_path):
    _,results,binding,ident=verify(tmp_path/'case')
    obs=next(o for o in results.observations() if o.endpoint==DETAIL_URL)
    (results.root/obs.payload_path).write_bytes(gzip.compress(b'changed'))
    with pytest.raises(ValueError,match='checksum'):
        verify_finland_settlement(binding,identity_archive_root=ident.root,result_archive_root=results.root,
            observation_ids=[o.id for o in results.observations()],as_of=AS_OF)
    shutil.copytree(RULE_PACKAGE_ROOT,tmp_path/'package')
    (tmp_path/'package/manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        verify_rule_package(package_root=tmp_path/'package',as_of=AS_OF)


def test_unrelated_event_ignored_but_conflicting_same_event_score_disputed(tmp_path):
    _,results,binding,ident=verify(tmp_path)
    save(results,DETAIL_URL.replace('4353542','99999'),b'other event')
    ids=[o.id for o in results.observations()]
    check=lambda: verify_finland_settlement(binding,identity_archive_root=ident.root,
        result_archive_root=results.root,observation_ids=[o.id for o in results.observations()],as_of=AS_OF)
    report=check()
    assert report['status']=='VERIFIED' and len(report['ignored'])==1
    save(results,DETAIL_URL,terminal(3,0),at=AFTER+timedelta(minutes=1))
    assert check()['status']=='DISPUTED'


def test_later_live_revision_cannot_leave_old_finished_label_valid(tmp_path):
    _,results,binding,ident=verify(tmp_path)
    save(results,DETAIL_URL,terminal().replace(b'head finished',b'head live'),at=AFTER+timedelta(minutes=2))
    report=verify_finland_settlement(binding,identity_archive_root=ident.root,result_archive_root=results.root,
        observation_ids=[o.id for o in results.observations()],as_of=AS_OF)
    assert report['status']=='DISPUTED' and report['settlement_rows']==[]
