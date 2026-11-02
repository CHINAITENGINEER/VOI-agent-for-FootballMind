"""Prepare a real future fixture from a user query using registered sources only.

Every request has an immutable intent before I/O. A resumed invocation reuses
raw observations; an intent without a receipt or raw response is never retried.
This module selects capabilities and verifies identity, not information value.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from filelock import FileLock

from football_research.acquisition.state import parse_time
from football_research.acquisition.value import _canonical
from football_spider.storage.raw_archive import RawArchive, atomic_write
from .prospective_text import select_initializations, verify_prior_archive
from .spftll import load_spftll_archive
from .types import json_record
from .finland_fixture import REGISTERED_TEAMS as FINLAND_TEAMS

VERSION = 'live-fixture-preparation-v3'
LEGACY_VERSION = 'live-fixture-preparation-v2'
DEFAULT_TIMEZONE = 'Asia/Shanghai'
# Explicit provider entity mappings previously verified against official raw
# fixtures. Adding a name requires a reviewed mapping, never fuzzy inference.
UEFA_TEAMS = {
    '768': ('克罗地亚', 'Croatia', '56370'), '772': ('西班牙', 'Spain', '122'),
    '641': ('苏格兰', 'Scotland', '117'), '632': ('斯洛文尼', 'Slovenia', '57163'),
    '763': ('哈萨克', 'Kazakhstan', '65146'), '754': ('法罗群岛', 'Faroe Islands', '51218'),
    '751': ('摩尔多瓦', 'Moldova', '57160'), '752': ('斯洛伐克', 'Slovakia', '58836'),
    '749': ('卢森堡', 'Luxembourg', '82'), '769': ('保加利亚', 'Bulgaria', '20'),
    '636': ('爱沙尼亚', 'Estonia', '56556'), '756': ('冰岛', 'Iceland', '58'),
}
REGISTERED_IDENTITY_PROVIDERS = {
    'finland_gnistan_2026': {'competition_id': 'spftll:competition:13', 'competition_name': '芬超',
        'category': 'senior_men_club', 'scope': 'registered_gnistan_inter_pair_2026_championship_phase'},
    'uefa_unl_2027': {'competition_id': 'spftll:competition:1864', 'competition_name': '欧国联',
        'category': 'senior_men_national', 'scope': 'registered_teams_2027_league_phase'},
    'canada_6630': {'match_id': 'spftll:match:3097883', 'scope': 'existing_exact_match_gateway'},
    'finland_championship_2026': {'competition_id': 'spftll:competition:13', 'competition_name': '芬超',
        'category': 'senior_men_club', 'scope': 'registered_six_teams_2026_championship_phase'},
}
_MATCH_PATH = re.compile(r'/uefanationsleague/match/([1-9][0-9]*)--[a-z0-9-]+/$')


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _registry(version):
    """Old preparations keep their exact capability inventory on recovery."""
    providers = deepcopy(REGISTERED_IDENTITY_PROVIDERS)
    if version == LEGACY_VERSION:
        providers.pop('finland_championship_2026')
        return {'providers': providers, 'uefa_teams': UEFA_TEAMS, 'finland_teams': FINLAND_TEAMS}
    if version != VERSION:
        raise ValueError('Unknown preparation capability version')
    from .finland_stage_fixture import REGISTERED_TEAMS
    return {'providers': providers, 'uefa_teams': UEFA_TEAMS,
            'finland_teams': FINLAND_TEAMS, 'finland_stage_teams': REGISTERED_TEAMS}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _put(path, value):
    """Immutable derived checkpoint; missing files may be reconstructed."""
    path = Path(path)
    if path.exists():
        if _canonical(_read(path)) != _canonical(value):
            raise ValueError('Preparation checkpoint changed: ' + path.name)
        return
    atomic_write(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False) + '\n').encode())


def _query(value):
    if isinstance(value, str):
        value = {'provider_id': value}
    if not isinstance(value, dict):
        raise ValueError('Use a provider ID or home/away/date query')
    if set(value) == {'provider_id'}:
        ident = str(value['provider_id']) if type(value['provider_id']) in {str, int} else ''
        match = re.fullmatch(r'(?:spftll:match:)?([1-9][0-9]{0,11})', ident)
        if not match:
            raise ValueError('Expected an exact positive SPFTLL match ID')
        return {'provider_id': match.group(1)}
    if not {'home','away'} <= set(value) or set(value) - {'home','away','date','timezone'}:
        raise ValueError('Named query requires home, away and optional date/timezone')
    result = {key: value.get(key) for key in ('home', 'away', 'date')}
    if any(not isinstance(result[k], str) or not result[k].strip() or len(result[k]) > 120
           for k in ('home', 'away')):
        raise ValueError('Nonempty bounded team names are required')
    result.update(home=result['home'].strip(), away=result['away'].strip(),
                  timezone=value.get('timezone', DEFAULT_TIMEZONE))
    if result['home'].casefold() == result['away'].casefold():
        raise ValueError('Home and away teams must differ')
    if result['date'] is not None:
        date.fromisoformat(result['date'])
    ZoneInfo(result['timezone'])
    return result


class _Stop(Exception):
    def __init__(self, status, reason, **details):
        self.status, self.reason, self.details = status, reason, details


def _default_transport(spec, *, archive, clock):
    """Use existing collectors/transport once, with no automatic retries."""
    if spec['source'] == 'spftll':
        from football_agent.inquiry_transport import SPFTLL_HEADERS, RESEARCH_USER_AGENT
        from football_spider.collectors.aio_transport import AioSession
        from football_spider.collectors.http import ArchiveHttp, FetchError
        client = ArchiveHttp(archive, 'spftll', max_requests=1, retries=1,
            session=AioSession(archive.root/'cookies'/'spftll-aio.json'))
        try:
            try:
                observation, _ = client.get(spec['url'], spec['params'],
                    headers={**SPFTLL_HEADERS, 'User-Agent': RESEARCH_USER_AGENT},
                    context={'requested_at': spec['requested_at'], 'preparation_stage': spec['stage'],
                             'preparation_request_digest': spec['request_digest']})
            except FetchError as exc:
                observation = exc.observation
            return observation
        finally:
            client.close()
    from football_agent.inquiry_transport import RoutedOpenWebTools
    if spec['stage'] in {'finland_schedule', 'finland_calendar', 'finland_championship_schedule'}:
        from .system_tls_source import capture_system_tls_page
        return capture_system_tls_page(spec, archive=archive, clock=clock)
    tools = RoutedOpenWebTools(archive, [urlsplit(spec['url']).hostname], route='direct',
                              clock=clock, extraction_version=3)
    prepared = tools.prepare({'tool':'read_url', 'url':spec['url']},
        {'task':{'seed_urls':[spec['url']]}, 'documents':[]})
    returned = tools.execute(prepared, run_id=spec['request_digest'], request_id=spec['stage'])
    ident = returned['observation']['id']
    return next(o for o in archive.observations() if o.id == ident)


class _Preparation:
    def __init__(self, output, plan, clock, transport):
        self.output, self.plan, self.clock, self.transport = output, plan, clock, transport
        self.registry = _registry(plan['schema'])
        self.market = RawArchive(output/'market-archive')
        self.identity = RawArchive(output/'identity-archive')

    def checkpoint_time(self, name):
        path = self.output/(name+'.json')
        if not path.exists():
            _put(path, {'at':parse_time(self.clock()).isoformat()})
        return parse_time(_read(path)['at'])

    def fetch(self, stage, url, params, *, official=False):
        archive = self.identity if official else self.market
        source = 'open_page' if official else 'spftll'
        directory = self.output/'requests'
        intent_path, receipt_path = directory/(stage+'.intent.json'), directory/(stage+'.receipt.json')
        request = {'stage':stage, 'url':url, 'params':params, 'source':source}
        request_digest = _digest(request)
        existing = intent_path.exists()
        if existing:
            spec = _read(intent_path)
            if any(spec.get(k) != v for k,v in request.items()) or spec.get('request_digest') != request_digest:
                raise ValueError('Preparation intent was changed')
        else:
            if len(list(directory.glob('*.intent.json'))) >= self.plan['max_http_requests']:
                raise _Stop('budget_exhausted', 'maximum_http_requests_reached', stage=stage)
            spec = {**request, 'request_digest':request_digest,
                    'requested_at':parse_time(self.clock()).isoformat()}
            _put(intent_path, spec)
        observations = [o for o in archive.observations() if o.endpoint == url and o.params == params
            and (o.context.get('preparation_request_digest') == request_digest
                 or o.context.get('inquiry_run_id') == request_digest)]
        if not observations and existing:
            raise _Stop('interrupted', 'request_intent_without_archived_response_never_retried', stage=stage)
        if not observations:
            observation = self.transport(deepcopy(spec), archive=archive, clock=self.clock)
            observations = [o for o in archive.observations() if observation is not None and o.id == observation.id]
        if len(observations) != 1:
            raise _Stop('interrupted', 'request_archive_not_exactly_one_response', stage=stage)
        observation = observations[0]
        body = archive.read_body(observation)
        if (observation.endpoint != url or observation.params != params or observation.source != source
                or observation.observed_at < parse_time(spec['requested_at'])
                or observation.observed_at > parse_time(self.clock())):
            raise ValueError('Request identity or actual receipt clock mismatch')
        receipt = {'stage':stage, 'observation_id':observation.id, 'raw_sha256':observation.content_hash,
            'observed_at':observation.observed_at.isoformat(), 'http_status':observation.http_status,
            'error_class':observation.error_class, 'request_digest':request_digest}
        _put(receipt_path, receipt)
        if observation.http_status != 200 or observation.error_class:
            raise _Stop('unavailable', 'source_request_failed', receipt=receipt)
        if not official:
            try:
                payload = json.loads(body)
                if not isinstance(payload,dict) or payload.get('code') != 0:
                    raise ValueError('Source business status failed')
            except (ValueError,TypeError):
                raise _Stop('unavailable', 'source_payload_not_successful_json', receipt=receipt)
        return observation, body

    def resolve(self):
        from football_spider.config import settings
        query = self.plan['query']
        if 'provider_id' in query:
            ident = query['provider_id']
            self.fetch('fixture', settings.detail_api_url, {'gameKind':0, 'matchId':ident})
        else:
            self.fetch('schedule', settings.generic_match_list_api_url, {})
            data = load_spftll_archive(self.market.root/'raw', as_of=self.clock())
            def named(match, side):
                name, team = getattr(match,side+'_team_name'), getattr(match,side+'_team_id')
                aliases = {name.casefold()}
                registered = UEFA_TEAMS.get(team.rsplit(':',1)[-1])
                if registered and name == registered[0]:
                    aliases.add(registered[1].casefold())
                finnish = {**FINLAND_TEAMS, **self.registry.get('finland_stage_teams', {})}.get(team)
                if finnish and name in finnish['provider_names']:
                    aliases.add(finnish['name'].casefold())
                    aliases.update(n.casefold() for n in finnish['provider_names'])
                return query[side].casefold() in aliases
            now=parse_time(self.clock())
            found = [m for m in data.matches if named(m,'home') and named(m,'away')
                     and (m.kickoff.astimezone(ZoneInfo(query['timezone'])).date().isoformat() == query['date']
                          if query['date'] is not None else m.status=='scheduled'
                          and now+timedelta(minutes=30)<m.kickoff<=now+timedelta(hours=168))]
            if not found:
                raise _Stop('not_found', 'no_exact_fixture_in_observed_provider_schedule')
            if len(found) != 1:
                raise _Stop('ambiguous', 'multiple_exact_fixtures_need_provider_id',
                            candidates=[json_record(m) for m in found])
            ident = found[0].match_id.rsplit(':',1)[-1]
        data = load_spftll_archive(self.market.root/'raw', as_of=self.clock())
        found = [m for m in data.matches if m.match_id == 'spftll:match:'+ident]
        if len(found) != 1:
            raise _Stop('not_found', 'provider_fixture_missing_or_quarantined', audit=data.audit)
        match = found[0]
        now = parse_time(self.clock())
        if (match.status != 'scheduled' or match.home_goals is not None
                or not now+timedelta(minutes=30) < match.kickoff <= now+timedelta(hours=168)):
            raise _Stop('not_prematch', 'future_scheduled_fixture_requires_30_minutes_to_168_hours',
                        fixture=json_record(match))
        return match

    def identity_provider(self, fixture):
        providers = self.registry['providers']
        finnish = providers['finland_gnistan_2026']
        if (fixture.competition_id == finnish['competition_id']
                and fixture.competition_name == finnish['competition_name']
                and {fixture.home_team_id, fixture.away_team_id} == set(FINLAND_TEAMS)
                and all(getattr(fixture, side+'_team_name') in FINLAND_TEAMS[getattr(fixture, side+'_team_id')]['provider_names']
                        for side in ('home', 'away'))):
            return 'finland_gnistan_2026'
        stage = providers.get('finland_championship_2026')
        stage_teams = self.registry.get('finland_stage_teams', {})
        if (stage and fixture.competition_id == stage['competition_id']
                and fixture.competition_name == stage['competition_name']
                and fixture.home_team_id != fixture.away_team_id
                and all(getattr(fixture, side+'_team_id') in stage_teams
                    and getattr(fixture, side+'_team_name') in stage_teams[getattr(fixture, side+'_team_id')]['provider_names']
                    for side in ('home', 'away'))):
            return 'finland_championship_2026'
        if fixture.match_id == providers['canada_6630']['match_id']:
            return 'canada_6630'
        registered = providers['uefa_unl_2027']
        if (fixture.competition_id == registered['competition_id']
                and fixture.competition_name == registered['competition_name']):
            for side in ('home','away'):
                row = UEFA_TEAMS.get(getattr(fixture,side+'_team_id').rsplit(':',1)[-1])
                if row is None or row[0] != getattr(fixture,side+'_team_name'):
                    raise _Stop('unsupported', 'official_team_mapping_not_registered', fixture=json_record(fixture))
            return 'uefa_unl_2027'
        raise _Stop('unsupported', 'official_identity_provider_not_registered', fixture=json_record(fixture),
                    registered_providers=deepcopy(providers))

    def bind(self, provider, manifest):
        if provider == 'finland_championship_2026':
            from .finland_stage_fixture import RESULTS_URL, RESULTS_PARAMS, bind_finland_stage_fixture
            schedule, _ = self.fetch('finland_championship_schedule', RESULTS_URL, RESULTS_PARAMS, official=True)
            return bind_finland_stage_fixture(manifest, archive_root=self.identity.root,
                observation_ids=[schedule.id], as_of=self.checkpoint_time('identity-as-of'))
        if provider == 'finland_gnistan_2026':
            from .finland_fixture import FIXTURES_URL, CALENDAR_URL, CALENDAR_LINK, bind_finland_fixture
            schedule, body = self.fetch('finland_schedule', FIXTURES_URL, {}, official=True)
            links = [a for a in BeautifulSoup(body, 'html.parser').find_all('a', href=True)
                     if a['href'] == CALENDAR_LINK]
            if len(links) != 1:
                raise _Stop('unavailable', 'registered_calendar_link_not_uniquely_observed')
            calendar, _ = self.fetch('finland_calendar', CALENDAR_URL,
                {'t': CALENDAR_LINK.split('?t=', 1)[1]}, official=True)
            return bind_finland_fixture(manifest, archive_root=self.identity.root,
                observation_ids=[schedule.id, calendar.id], as_of=self.checkpoint_time('identity-as-of'))
        fixture = manifest['market_initialization']['fixture']
        if provider == 'canada_6630':
            from .canada_fixture import MATCH_URL, bind_canada_fixture
            observation,_ = self.fetch('official_match',MATCH_URL,{},official=True)
            return bind_canada_fixture(manifest,archive_root=self.identity.root,
                observation_ids=[observation.id],as_of=self.checkpoint_time('identity-as-of'))
        from .verified_fixtures import SCHEDULE_URL, verify_fixture_identity
        mappings = {side:UEFA_TEAMS[fixture[side+'_team_id'].rsplit(':',1)[-1]] for side in ('home','away')}
        _, body = self.fetch('official_schedule', SCHEDULE_URL, {}, official=True)
        soup = BeautifulSoup(body,'html.parser')
        links=[]
        for anchor in soup.find_all('a',href=True):
            url=urljoin(SCHEDULE_URL,anchor['href'])
            parts=urlsplit(url)
            if (parts.scheme!='https' or parts.netloc!='www.uefa.com' or parts.query or parts.fragment
                    or not _MATCH_PATH.fullmatch(parts.path)):
                continue
            text=' '.join(anchor.get_text(' ',strip=True).split())
            if text == mappings['home'][1]+' vs '+mappings['away'][1] and url not in links:
                links.append(url)
        if not links:
            raise _Stop('unavailable','official_schedule_has_no_exact_ordered_fixture_link')
        if len(links)!=1:
            raise _Stop('ambiguous','official_schedule_has_multiple_exact_pair_links',official_urls=links)
        observation,_ = self.fetch('official_match',links[0],{},official=True)
        binding = verify_fixture_identity(manifest,archive_root=self.identity.root,observation_ids=[observation.id],
            aliases={side:list(mappings[side][:2]) for side in ('home','away')},
            as_of=self.checkpoint_time('identity-as-of'))
        if binding['status']=='CONFIRMED' and any(binding['canonical'][side+'_id'] != 'uefa:team:'+mappings[side][2]
                                                for side in ('home','away')):
            raise _Stop('unavailable','official_team_ids_conflict_with_registered_mapping')
        if binding['status']=='CONFIRMED' and binding['canonical']['season']!='2027':
            raise _Stop('unsupported','official_season_not_registered')
        return binding

    def run(self):
        from football_spider.config import settings
        from football_agent.match_source_profiles import resolve_verified_profile
        from .verified_fixtures import verify_identity_binding
        fixture = self.resolve()
        provider = self.identity_provider(fixture)
        self.fetch('quote',settings.more_odds_api_url,{'mKind':0,'matchId':fixture.match_id.rsplit(':',1)[-1]})
        frozen = self.checkpoint_time('initialization-as-of')
        data = load_spftll_archive(self.market.root/'raw',as_of=frozen)
        data = replace(data,matches=tuple(m for m in data.matches if m.match_id==fixture.match_id))
        prepared=select_initializations(data,as_of=frozen,limit=1,horizon_hours=168)
        if len(prepared['selected'])!=1:
            raise _Stop('unavailable','complete_recent_regulation_initialization_unavailable',exclusions=prepared['excluded'])
        manifest=prepared['selected'][0]
        verify_prior_archive(manifest,self.market.root)
        binding=self.bind(provider,manifest)
        if binding['status']!='CONFIRMED':
            raise _Stop('unavailable','independent_fixture_identity_not_confirmed',identity_status=binding['status'],
                        identity_reason=binding['reason'])
        verify_identity_binding(binding,archive_root=self.identity.root,as_of=self.clock())
        profile_version = 4 if binding['schema'] == 'finland-stage-fixture-identity-v1' else 3
        profile=resolve_verified_profile(binding,version=profile_version)
        _put(self.output/'initializations.json',prepared)
        _put(self.output/'identity-bundle.json',{'schema':'independent-fixture-identity-bundle-v1',
            'created_at':binding['verified_at'],'archive_root':str(self.identity.root),
            'bindings':{fixture.match_id:binding}})
        return {'status':'ready','reason':'current_market_and_independent_official_identity_verified',
            'match_id':fixture.match_id,'provider':provider,'fixture':json_record(fixture),
            'initial_probabilities':manifest['market_initialization']['probabilities'],
            'quote_observed_at':manifest['market_initialization']['quote']['observed_at'],
            'quote_valid_for_new_freeze_until':(parse_time(manifest['market_initialization']['quote']['observed_at'])
                                              +timedelta(hours=1)).isoformat(),
            'prediction_cutoff':manifest['market_initialization']['prediction_cutoff'],
            'identity_binding_sha256':binding['binding_sha256'],
            'collection':{'allowed_hosts':profile['allowed_hosts'],'seed_urls':profile['seed_urls'],
                'source_profile_version':profile_version,'news_provider':'bing_rss','news_projection_version':5,
                'evidence_version':'match-evidence-extraction-v5',
                'candidate_discovery_version':10 if profile_version == 4 else 8},
            'labels':[],'numeric_voi_training_eligible':False}


def prepare_live_fixture(query, *, output, max_http_requests=6, clock=None, resume=False, transport=None):
    """Resolve ID or exact teams/date, acquire raw inputs, and prepare one run.

    Dates default to Asia/Shanghai. ``transport`` is a trusted test seam taking
    ``(request_spec, archive=RawArchive, clock=callable)`` and returning the
    already archived Observation; application user input cannot override it.
    Resume never retries an issued request. Use a new output for an explicitly
    requested fresh acquisition, retaining the original failed preparation.
    """
    query=_query(query)
    if type(max_http_requests) is not int or not 0<=max_http_requests<=12:
        raise ValueError('Preparation HTTP budget must be an integer from 0 to 12')
    if type(resume) is not bool:
        raise ValueError('resume must be boolean')
    clock=clock or (lambda:datetime.now(timezone.utc))
    parse_time(clock())
    output=Path(output).resolve()
    if output.exists() and not resume:
        raise ValueError('Use a new preparation directory or explicit resume')
    output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/'.prepare.lock'),timeout=0):
        plan_path=output/'prepare-plan.json'
        if plan_path.exists():
            plan=_read(plan_path)
            if (plan.get('schema') not in {VERSION, LEGACY_VERSION} or plan['query']!=query or plan['max_http_requests']!=max_http_requests
                    or plan['registry_sha256']!=_digest(_registry(plan['schema']))):
                raise ValueError('Resume query, budget, or registered capabilities changed')
        else:
            if resume and any(p.name!='.prepare.lock' for p in output.iterdir()):
                raise ValueError('Cannot resume an unrecognised preparation directory')
            plan={'schema':VERSION,'query':query,'max_http_requests':max_http_requests,
                'created_at':parse_time(clock()).isoformat(),'default_date_timezone':DEFAULT_TIMEZONE,
                'registry_sha256':_digest(_registry(VERSION)),
                'automatic_retries':0,'horizon_hours':168,'minimum_lead_minutes':30}
            _put(plan_path,plan)
        (output/'requests').mkdir(exist_ok=True)
        worker=_Preparation(output,plan,clock,transport or _default_transport)
        try:
            result=worker.run()
        except _Stop as stop:
            result={'status':stop.status,'reason':stop.reason,**stop.details}
        result.update(schema=plan['schema'],query=query,prepare_output=str(output),
            initializations_path=str(output/'initializations.json'),
            identity_bundle_path=str(output/'identity-bundle.json'),market_archive=str(worker.market.root),
            identity_archive=str(worker.identity.root),
            request_intents=len(list((output/'requests').glob('*.intent.json'))),
            archived_http_responses=len(worker.market.observations())+len(worker.identity.observations()),
            preparation_plan_sha256=sha256(plan_path.read_bytes()).hexdigest())
        # The terminal report is recomputed from archived evidence on resume.
        result=json.loads(_canonical(result))
        result_path=output/'prepare-result.json'
        if result_path.exists() and _read(result_path)!=result:
            previous=_read(result_path)
            if previous.get('status')!='interrupted':
                raise ValueError('A completed preparation result changed')
            _put(output/'result-history'/(_digest(previous)+'.json'),previous)
            atomic_write(result_path,(json.dumps(result,ensure_ascii=False,sort_keys=True,
                indent=2,allow_nan=False)+'\n').encode())
        else:
            _put(result_path,result)
        return result


__all__=['prepare_live_fixture','REGISTERED_IDENTITY_PROVIDERS','VERSION']
