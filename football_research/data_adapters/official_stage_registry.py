"""Observed native entities for the explicit 2026 lower six-team league stage.

These are league-native names and IDs, not inferred market-provider mappings.
Membership and event rows were observed on the official group=3 schedule.
"""
from copy import deepcopy
import re
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from football_research.acquisition.state import parse_time
from .finland_stage_fixture import HOME_URL, NEWS_URL, _text

GROUP3_TEAMS = {
    'FC Lahti': {'native_id':'60969', 'slug':'fc-lahti', 'aliases':['FC Lahti']},
    'FF Jaro': {'native_id':'60974', 'slug':'ff-jaro', 'aliases':['FF Jaro']},
    'IFK Mariehamn': {'native_id':'60759', 'slug':'ifk-mariehamn', 'aliases':['IFK Mariehamn']},
    'Ilves': {'native_id':'60745', 'slug':'ilves', 'aliases':['Ilves']},
    'SJK': {'native_id':'60970', 'slug':'sjk', 'aliases':['SJK']},
    'TPS': {'native_id':'7', 'slug':'tps', 'aliases':['TPS']},
}
GROUP3_PROVENANCE = {
    'url':'https://www.veikkausliiga.com/tilastot/2026/veikkausliiga/ottelut/',
    'params':{'group':'3'}, 'observed_at':'2026-10-07T10:54:32.591840+00:00',
    'raw_sha256':'f4e6d75feb154e58a31c433dc262be5b98ef31748cdd5689c4ff2f08d45fc31b',
    'basis':'native_team_options_and_ordered_Karsintasarja_fixture_rows',
    'provider_mapping_inferred':False,
}
PARSER_VERSION = 'official-veikkausliiga-2026-karsintasarja-no-market-v1'
_PATH = re.compile(r'/tilastot/2026/veikkausliiga/ottelut/([1-9][0-9]{0,11})/')


def schedule_group(value):
    if type(value) is not str or value not in {'2','3'}:
        raise ValueError("Official schedule_group must be exactly '2' or '3'")
    return value


def group3_registry():
    return deepcopy(GROUP3_TEAMS)


def project_group3_schedule(body, candidate):
    """Strict selected-stage/native-ID projection; no market alias is needed."""
    def fail(reason):
        return {'status':'UNCONFIRMED', 'reason':reason}
    soup=BeautifulSoup(body,'html.parser')
    if soup.title is None or _text(soup.title)!='Otteluohjelma | Veikkausliiga':
        return fail('official_league_schedule_identity_missing')
    for selector, value, title in (
        ('select#season option[selected]','2026','2026'),
        ('select#category option[selected]','veikkausliiga','Veikkausliiga'),
        ('select[name=group] option[selected]','3','Karsintasarja')):
        nodes=soup.select(selector)
        if len(nodes)!=1 or nodes[0].get('value')!=value or _text(nodes[0])!=title:
            return fail('official_schedule_season_competition_or_stage_missing')
    for team, spec in GROUP3_TEAMS.items():
        nodes=[node for node in soup.select('select[name=team] option')
               if node.get('value')==spec['native_id'] or _text(node)==team]
        if len(nodes)!=1 or nodes[0].get('value')!=spec['native_id'] or _text(nodes[0])!=team:
            return fail('official_native_team_registry_conflict')
    native=candidate['native_event_id']
    path='/tilastot/2026/veikkausliiga/ottelut/'+native+'/'
    if _PATH.fullmatch(path) is None or len(soup.find_all('a',href=path))!=1:
        return fail('official_native_fixture_id_ambiguous')
    targets=[]
    for row in soup.select('tr[data-time]'):
        cells=row.find_all('td',recursive=False)
        if len(cells)!=8:
            continue
        links=cells[4].find_all('a',href=True)
        if len(links)==1 and links[0]['href']==path:
            targets.append((row,cells,links[0]))
    if len(targets)!=1:
        return fail('official_ordered_fixture_missing_or_ambiguous')
    row,cells,link=targets[0]
    home,away=candidate['home_name'],candidate['away_name']
    local=parse_time(candidate['kickoff']).astimezone(ZoneInfo('Europe/Helsinki'))
    if (home not in GROUP3_TEAMS or away not in GROUP3_TEAMS or home==away
            or row.get('data-time')!=local.strftime('%Y%m%d') or _text(cells[2])!=local.strftime('%H:%M')
            or _text(link)!=home+' - '+away):
        return fail('official_ordered_teams_or_local_clock_conflict')
    if _text(cells[6]) not in ('','-','–','—'):
        return fail('official_target_has_score_or_non_scheduled_state')
    if not any(a.get('href')==path+'seuranta/' for a in cells[5].find_all('a',href=True)):
        return fail('observed_native_followup_link_missing')
    generic=[]
    if soup.find('a',href='/') is not None:
        generic.append(HOME_URL)
    if any(a.get('href')=='/uutiset/' and _text(a)=='Uutiset' for a in soup.find_all('a',href=True)):
        generic.append(NEWS_URL)
    return {'status':'EXTRACTED', 'role':'official_relegation_series_schedule',
        'source_event_id':'veikkausliiga:2026:match:'+native, 'source_url':urljoin(HOME_URL,path),
        'native_event_id':native,'home_name':home,'away_name':away,
        'kickoff':parse_time(candidate['kickoff']).isoformat(),'local_kickoff':local.replace(tzinfo=None).isoformat(),
        'timezone':'Europe/Helsinki','competition':'Veikkausliiga','season':'2026','stage':'Karsintasarja',
        'home_native_team_id':GROUP3_TEAMS[home]['native_id'], 'away_native_team_id':GROUP3_TEAMS[away]['native_id'],
        'score_present':False,'publication_time':None,'observed_generic_entry_urls':generic}


__all__=['GROUP3_TEAMS','GROUP3_PROVENANCE','PARSER_VERSION','schedule_group',
         'group3_registry','project_group3_schedule']
