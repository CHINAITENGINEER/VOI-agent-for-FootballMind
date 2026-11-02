"""Raw-backed identities for the six registered 2026 Finnish championship teams.

The league's observed stage schedule supplies event and league-native team IDs.
Provider entities are reviewed once across historical ordered fixtures, rather
than inferred afresh from a similar spelling or patched for each future match.
The existing Gnistan/Inter pair keeps its original calendar identity namespace.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from football_platform.information.fixture_corroboration import compare_assertion
from football_research.acquisition.state import parse_time
from football_research.acquisition.value import _canonical
from football_spider.storage.raw_archive import RawArchive
from .prospective_text import MarketPriorPolicy

VERSION = 'finland-stage-fixture-identity-v1'
PARSER_VERSION = 'veikkausliiga-2026-observed-championship-fixture-v1'
HOME_URL = 'https://www.veikkausliiga.com/'
NEWS_URL = HOME_URL + 'uutiset/'
RESULTS_URL = HOME_URL + 'tilastot/2026/veikkausliiga/ottelut/'
RESULTS_PARAMS = {'group': '2'}
REGISTERED_TEAMS = {
    'spftll:team:2429': {'name':'IF Gnistan', 'slug':'if-gnistan',
        'provider_names':('赫尔火花','格尼斯坦'), 'native_team_id':'48'},
    'spftll:team:386': {'name':'FC Inter', 'slug':'fc-inter',
        'provider_names':('国际图尔','图尔库国际'), 'native_team_id':'60987'},
    'spftll:team:383': {'name':'HJK', 'slug':'hjk',
        'provider_names':('赫尔辛基',), 'native_team_id':'60731'},
    'spftll:team:2226': {'name':'VPS', 'slug':'vps',
        'provider_names':('瓦萨','VPS瓦萨'), 'native_team_id':'60901'},
    'spftll:team:1986': {'name':'KuPS', 'slug':'kups',
        'provider_names':('库奥皮奥','古比斯'), 'native_team_id':'60756'},
    'spftll:team:2228': {'name':'AC Oulu', 'slug':'ac-oulu',
        'provider_names':('AC奥卢','奥卢'), 'native_team_id':'34'},
}
_LEGACY_PAIR = {'spftll:team:2429','spftll:team:386'}
_MATCH_PATH = re.compile(r'/tilastot/2026/veikkausliiga/ottelut/([1-9][0-9]*)/')
_MAPPING_PROVENANCE = {
    'basis':'reviewed_stable_provider_ids_with_ordered_historical_fixture_corroboration',
    'purpose':'identity_mapping_only_not_prospective_training_labels',
    'official_regular_stage_sha256':'9733f6431a41e4c50b390ba59ac93cc4a20c192801bbb82a08e8206e56c79f63',
    'official_mens_competition_description_sha256':'8f84bc039d85f156b061a12f5edef778dd3734c122899694ff0c1047ff8d9d5b',
    'pairs':[
        {'provider_team_ids':['383','2226'],
         'provider_observation_id':'977e783d-084e-4696-8275-ef5ef12c654b',
         'provider_raw_sha256':'621d5b1cffdc1ed2c99ff0ab14efd8a3f4fb7ee4c07a6bbf3919c6f67af628f8',
         'provider_fixture_ids':['2913664','2913616'], 'official_fixture_ids':['4036942','4036894']},
        {'provider_team_ids':['1986','2228'],
         'provider_observation_id':'ca06ab5c-0f1f-4172-ae6c-2c59f019f855',
         'provider_raw_sha256':'0ac72eb8af6034212559c48bfea9bf5bc6c0729a5fa24b87fd82099274b790d6',
         'provider_fixture_ids':['2913601','2913652'], 'official_fixture_ids':['4036879','4036930']},
        {'provider_team_ids':['2429','386'],
         'provider_observation_id':'7d47259e-0f05-41dc-8163-10cdd22934a4',
         'provider_raw_sha256':'085d79be980b0cd099e1f1b2136db17e3a16e42ab47a4c0745fef327b41b1449',
         'provider_fixture_ids':['2913603','2913674'],
         'official_club_schedule_sha256':'41c0c15cbc288ce27e990cdcc68ac985a72573930d841556b846596d17a8ab7f'}],
}


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _text(node):
    return ' '.join(node.get_text(' ', strip=True).split())


def _failure(reason, *, disputed=False):
    return {'status':'DISPUTED' if disputed else 'UNCONFIRMED', 'reason':reason}


def _registered(fixture):
    teams = [fixture.get(side+'_team_id') for side in ('home','away')]
    return (isinstance(fixture.get('match_id'),str)
        and re.fullmatch(r'spftll:match:[1-9][0-9]*',fixture['match_id']) is not None
        and fixture.get('competition_id') == 'spftll:competition:13'
        and fixture.get('competition_name') == '芬超'
        and fixture.get('season') in (None,'2026')
        and parse_time(fixture['kickoff']).year == 2026
        and fixture.get('status') == 'scheduled'
        and len(set(teams)) == 2 and all(team in REGISTERED_TEAMS for team in teams)
        and all(fixture.get(side+'_team_name') in REGISTERED_TEAMS[fixture[side+'_team_id']]['provider_names']
                for side in ('home','away')))


def _schedule(body, fixture):
    soup = BeautifulSoup(body,'html.parser')
    if soup.title is None or _text(soup.title) != 'Otteluohjelma | Veikkausliiga':
        return _failure('official_league_schedule_identity_missing')
    for selector,value,title in [('select#season option[selected]','2026','2026'),
                                ('select#category option[selected]','veikkausliiga','Veikkausliiga'),
                                ('select[name=group] option[selected]','2','Mestaruussarja')]:
        nodes = soup.select(selector)
        if len(nodes) != 1 or nodes[0].get('value') != value or _text(nodes[0]) != title:
            return _failure('official_schedule_season_competition_or_stage_missing')
    registered = {side:REGISTERED_TEAMS[fixture[side+'_team_id']] for side in ('home','away')}
    for team in registered.values():
        nodes = [node for node in soup.select('select[name=team] option')
                 if node.get('value') == team['native_team_id'] or _text(node) == team['name']]
        if len(nodes) != 1 or nodes[0].get('value') != team['native_team_id'] or _text(nodes[0]) != team['name']:
            return _failure('official_native_team_registry_conflict', disputed=True)
    home, away = registered['home']['name'], registered['away']['name']
    local = parse_time(fixture['kickoff']).astimezone(ZoneInfo('Europe/Helsinki'))
    targets = []
    for row in soup.select('tr[data-time]'):
        cells = row.find_all('td',recursive=False)
        if len(cells) != 8 or row.get('data-time') != local.strftime('%Y%m%d'):
            continue
        links = cells[4].find_all('a',href=True)
        if len(links) == 1 and _text(links[0]) in (home+' - '+away, away+' - '+home):
            targets.append((row,cells,links[0]))
    if len(targets) != 1:
        return _failure('official_ordered_fixture_missing_or_ambiguous')
    row,cells,link = targets[0]
    native = _MATCH_PATH.fullmatch(link['href'])
    if native is None:
        return _failure('official_native_fixture_url_missing')
    if len(soup.find_all('a',href=link['href'])) != 1:
        return _failure('official_native_fixture_id_ambiguous')
    if _text(link) != home+' - '+away or _text(cells[2]) != local.strftime('%H:%M'):
        return _failure('official_ordered_teams_or_local_clock_conflict', disputed=True)
    if _text(cells[6]) not in ('','-','–','—'):
        return _failure('official_target_has_score_or_non_scheduled_state')
    # The source's observed match and follow-up paths share an exact native ID.
    if not any(a.get('href') == link['href']+'seuranta/' for a in cells[5].find_all('a',href=True)):
        return _failure('observed_native_followup_link_missing')
    generic = []
    if soup.find('a',href='/') is not None:
        generic.append(HOME_URL)
    if any(a.get('href') == '/uutiset/' and _text(a) == 'Uutiset' for a in soup.find_all('a',href=True)):
        generic.append(NEWS_URL)
    return {'status':'EXTRACTED','role':'official_championship_schedule',
        'source_event_id':'veikkausliiga:2026:match:'+native.group(1),
        'source_url':urljoin(HOME_URL,link['href']), 'native_event_id':native.group(1),
        'home_name':home,'away_name':away,'kickoff':parse_time(fixture['kickoff']).isoformat(),
        'local_kickoff':local.replace(tzinfo=None).isoformat(),'timezone':'Europe/Helsinki',
        'competition':'Veikkausliiga','season':'2026','stage':'Mestaruussarja',
        'home_native_team_id':registered['home']['native_team_id'],
        'away_native_team_id':registered['away']['native_team_id'],
        'score_present':False,'publication_time':None,'observed_generic_entry_urls':generic}


def _project(observation, body, fixture, *, as_of):
    if observation.observed_at > as_of:
        raise ValueError('Official stage response was unavailable at audit time')
    requested = observation.context.get('requested_at')
    if requested is None or parse_time(requested) > observation.observed_at:
        raise ValueError('Official receipt must follow its recorded request')
    if (observation.source != 'open_page' or observation.endpoint != RESULTS_URL
            or observation.params != RESULTS_PARAMS):
        raise ValueError('Expected the registered official championship schedule URL and group')
    if observation.http_status != 200 or observation.error_class:
        return _failure('official_response_unavailable')
    if len(body)>2_000_000 or observation.context.get('response_complete') is False:
        return _failure('official_identity_response_incomplete_or_oversized')
    if not _registered(fixture):
        return _failure('provider_fixture_not_in_registered_2026_mens_stage')
    if {fixture['home_team_id'],fixture['away_team_id']} == _LEGACY_PAIR:
        return _failure('require_existing_pair_identity_provider')
    if observation.observed_at >= parse_time(fixture['kickoff']):
        return _failure('official_identity_response_not_observed_prematch')
    return _schedule(body,fixture)


def bind_finland_stage_fixture(initialization_manifest, *, archive_root, observation_ids, as_of):
    """Reconstruct a future fixture from an explicit official stage observation."""
    as_of = parse_time(as_of)
    policy = MarketPriorPolicy.from_manifest(initialization_manifest)
    fixture = policy.model.fixture
    if parse_time(policy.model.frozen_at)>as_of:
        raise ValueError('Market initialization was unavailable at identity verification')
    if (not isinstance(observation_ids,(list,tuple)) or not 1<=len(observation_ids)<=8
            or any(not isinstance(i,str) or not i for i in observation_ids)
            or len(set(observation_ids)) != len(observation_ids)):
        raise ValueError('Distinct explicit source observations are required')
    archive = RawArchive(Path(archive_root))
    observations = archive.observations()
    by_id = {o.id:o for o in observations}
    if len(by_id)!=len(observations) or any(i not in by_id for i in observation_ids):
        raise ValueError('Source observations missing or ambiguous')
    facts,evidence = [],[]
    for ident in sorted(observation_ids):
        observation = by_id[ident]
        facts.append(_project(observation,archive.read_body(observation),fixture,as_of=as_of))
        evidence.append({'observation_id':ident,'raw_sha256':observation.content_hash,
                         'url':observation.endpoint,'observed_at':observation.observed_at.isoformat()})
    status,reason = 'UNCONFIRMED','official_stage_identity_missing_or_ambiguous'
    canonical,comparison = None,None
    if any(f['status']=='DISPUTED' for f in facts):
        status,reason='DISPUTED','official_order_identity_or_clock_conflict'
    elif any(f['status']!='EXTRACTED' for f in facts):
        reason=next(f['reason'] for f in facts if f['status']!='EXTRACTED')
    elif len(facts)==1:
        fact=facts[0]
        comparison=compare_assertion({key:fixture[key] for key in (
            'match_id','competition_id','season','home_team_id','away_team_id','kickoff',
            'home_team_name','away_team_name')}, source_event_id=fact['source_event_id'],
            explicitly_mapped_match_id=fixture['match_id'], fields={
                'home_team_id':fixture['home_team_id'],'away_team_id':fixture['away_team_id'],
                'competition_id':fixture['competition_id'],'kickoff':fact['kickoff']},
            evidence_basis='acquired_structured', available_at=parse_time(evidence[0]['observed_at']),
            cutoff=min(as_of,parse_time(policy.model.cutoff)))
        status,reason=comparison['status'],comparison['reason']
        if status=='CONFIRMED':
            canonical={key:fact[key] for key in ('home_name','away_name','kickoff','competition',
                'source_url','source_event_id','stage','season')}
            canonical.update(category='senior_men_club',event_status='EventScheduled',
                parser_version=PARSER_VERSION, team_id_basis='official_league_native_team_ids',
                category_basis='registered_official_mens_league_and_explicit_championship_stage',
                score_period='unknown',settlement_rule='unknown')
            for side in ('home','away'):
                canonical[side+'_id']='veikkausliiga:team:'+fact[side+'_native_team_id']
    aliases={side:list(REGISTERED_TEAMS[fixture[side+'_team_id']]['provider_names'])
                    +[REGISTERED_TEAMS[fixture[side+'_team_id']]['name']]
             if fixture.get(side+'_team_id') in REGISTERED_TEAMS else [] for side in ('home','away')}
    native_ids={side:REGISTERED_TEAMS[fixture[side+'_team_id']]['native_team_id']
                if fixture.get(side+'_team_id') in REGISTERED_TEAMS else None for side in ('home','away')}
    result={'schema':VERSION,'status':status,'reason':reason,'match_id':fixture['match_id'],
        'initialization_manifest':deepcopy(initialization_manifest),'initialization_sha256':_digest(initialization_manifest),
        'canonical':canonical,'aliases':aliases,'observations':evidence,'verified_at':as_of.isoformat(),
        'comparison':comparison,'source_projections':facts,'mapping_provenance':deepcopy(_MAPPING_PROVENANCE),
        'mapping_basis':'reviewed_stable_provider_ids_with_official_stage_native_event',
        'native_team_ids':native_ids,'native_team_id_namespace':'veikkausliiga',
        'native_numeric_team_ids_complete':all(native_ids.values()),
        'observed_generic_entry_urls':facts[0].get('observed_generic_entry_urls',[]) if canonical else [],
        'independent_from_initial_quote_provider':True,'retroactive_actor_evidence':False,
        'settlement_status':'PENDING','settlement_rows':[],'regulation_rule':None}
    result['binding_sha256']=_digest(result)
    return result


def verify_finland_stage_identity(binding, *, archive_root, as_of):
    """Validate the entire binding against the exact original raw bytes again."""
    if (not isinstance(binding,dict) or binding.get('schema')!=VERSION
            or parse_time(binding['verified_at'])>parse_time(as_of)
            or binding.get('binding_sha256')!=_digest({k:v for k,v in binding.items() if k!='binding_sha256'})):
        raise ValueError('Finland stage identity binding changed or is not yet available')
    rebuilt=bind_finland_stage_fixture(binding['initialization_manifest'],archive_root=archive_root,
        observation_ids=[e['observation_id'] for e in binding['observations']],as_of=binding['verified_at'])
    if rebuilt!=binding:
        raise ValueError('Finland stage identity does not reproduce official raw evidence')
    return rebuilt


__all__=['VERSION','PARSER_VERSION','REGISTERED_TEAMS','HOME_URL','NEWS_URL','RESULTS_URL','RESULTS_PARAMS',
         'bind_finland_stage_fixture','verify_finland_stage_identity']
