"""One-request official fixture preparation without a market initialization.

Only the registered 2026 men's championship schedule is supported. Raw bytes,
the pre-I/O intent and its receipt are sufficient to reconstruct the identity.
Recovery never reissues an existing intent or refreshes its observation time.
"""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from bs4 import BeautifulSoup
from filelock import FileLock

from football_research.acquisition.state import parse_time
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes
from .finland_stage_fixture import (
    REGISTERED_TEAMS, RESULTS_URL, RESULTS_PARAMS, _schedule, _text,
)
from .official_stage_registry import (
    schedule_group as _group, group3_registry, project_group3_schedule,
    PARSER_VERSION as GROUP3_PARSER_VERSION,
)

VERSION = 'official-fixture-evidence-identity-v1'
PREPARATION_VERSION = 'official-fixture-preparation-v1'
PARSER_VERSION = 'official-veikkausliiga-2026-championship-no-market-v1'
DEFAULT_TIMEZONE = 'Asia/Shanghai'
_MATCH = re.compile(r'veikkausliiga:2026:match:([1-9][0-9]{0,11})')
_PATH = re.compile(r'/tilastot/2026/veikkausliiga/ottelut/([1-9][0-9]{0,11})/')
_LEGACY_EVENT = '4353542'
_STAGE = 'official_schedule'


def _digest(value):
    return sha256(json_bytes(value)).hexdigest()


def _horizon(value):
    if type(value) is not int or value not in {7, 14}:
        raise ValueError('Official preparation horizon_days must be explicitly 7 or 14')
    return value


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _put(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError('Immutable official preparation checkpoint changed: '+path.name)
    else:
        atomic_write(path, json_bytes(value))


def _registry(schedule_group='2'):
    # Source IDs here are league-native. No provider fixture ID is inferred.
    if _group(schedule_group) == '3':
        return group3_registry()
    return {team['name']:{'native_id':team['native_team_id'], 'slug':team['slug'],
        'aliases':list(team['provider_names'])+[team['name']]} for team in REGISTERED_TEAMS.values()}


def _query(value, *, schedule_group='2'):
    if isinstance(value, str):
        matched = _MATCH.fullmatch(value)
        if matched:
            return {'native_event_id':matched.group(1)}
        if value == 'palloliitto:match:'+_LEGACY_EVENT:
            return {'native_event_id':_LEGACY_EVENT}
        return {'unsupported_reason':'official_fixture_namespace_not_registered'}
    if (not isinstance(value, dict) or not {'home','away'} <= set(value)
            or set(value)-{'home','away','date','timezone'}):
        raise ValueError('Use an official native ID or home/away/date/timezone query')
    names = {alias:name for name,team in _registry(schedule_group).items() for alias in team['aliases']}
    if any(not isinstance(value[k],str) or value[k].strip() not in names for k in ('home','away')):
        return {'unsupported_reason':'official_team_alias_not_registered'}
    home, away = (names[value[k].strip()] for k in ('home','away'))
    if home == away:
        raise ValueError('Home and away must be different registered official teams')
    try:
        day = date.fromisoformat(value['date']).isoformat() if value.get('date') is not None else None
        tz = value.get('timezone', DEFAULT_TIMEZONE)
        ZoneInfo(tz)
    except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError('An exact ISO date and recognized explicit timezone are required') from exc
    return {'home':home,'away':away,'date':day,'timezone':tz}


def _local_kickoff(day, clock):
    naive = datetime.strptime(day+' '+clock, '%Y%m%d %H:%M')
    if naive.year != 2026:
        raise ValueError('Official fixture season conflicts with its date')
    zone = ZoneInfo('Europe/Helsinki')
    values = {naive.replace(tzinfo=zone,fold=fold).astimezone(timezone.utc) for fold in (0,1)
        if naive.replace(tzinfo=zone,fold=fold).astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)==naive}
    if len(values) != 1:
        raise ValueError('Official local kickoff is ambiguous or nonexistent')
    return values.pop()


def _resolve(body, query, at, *, horizon_days=7, schedule_group='2'):
    horizon_days = _horizon(horizon_days)
    schedule_group = _group(schedule_group)
    soup = BeautifulSoup(body,'html.parser')
    registry = _registry(schedule_group)
    internal_keys = {team['name']:key for key,team in REGISTERED_TEAMS.items()}
    candidates = []
    for row in soup.select('tr[data-time]'):
        cells = row.find_all('td',recursive=False)
        if len(cells) != 8:
            continue
        links = cells[4].find_all('a',href=True)
        if len(links) != 1 or not (match := _PATH.fullmatch(links[0]['href'])):
            continue
        if 'native_event_id' in query and match.group(1) != query['native_event_id']:
            continue
        pair = _text(links[0]).split(' - ')
        if len(pair)!=2 or pair[0]==pair[1] or any(name not in registry for name in pair):
            if 'native_event_id' in query:
                return {'status':'unsupported','reason':'official_target_team_or_squad_not_registered'}
            continue
        if 'home' in query and pair != [query['home'],query['away']]:
            continue
        try:
            kickoff = _local_kickoff(row['data-time'], _text(cells[2]))
        except ValueError:
            return {'status':'unsupported','reason':'official_target_kickoff_missing_or_ambiguous'}
        if query.get('date') is not None and kickoff.astimezone(ZoneInfo(query['timezone'])).date().isoformat()!=query['date']:
            continue
        if 'home' in query and query.get('date') is None and not timedelta(minutes=30)<=kickoff-at<=timedelta(days=horizon_days):
            continue
        candidates.append({'native_event_id':match.group(1), 'home_name':pair[0],
            'away_name':pair[1], 'kickoff':kickoff.isoformat()})
    if not candidates:
        return {'status':'not_found','reason':'no_exact_official_fixture_in_observed_schedule'}
    if len(candidates)!=1:
        return {'status':'ambiguous','reason':'multiple_official_fixture_rows','candidates':candidates}
    candidate = candidates[0]
    # Reuse the existing strict season, stage, team-ID, ordered-row, score and
    # follow-up-link parser. These internal registry keys are never a provider
    # fixture assertion and are not returned as official identities.
    fact = (project_group3_schedule(body,candidate) if schedule_group=='3' else
        _schedule(body, {'home_team_id':internal_keys[candidate['home_name']],
            'away_team_id':internal_keys[candidate['away_name']], 'kickoff':candidate['kickoff']}))
    if fact.get('status') != 'EXTRACTED':
        status = 'not_prematch' if fact.get('reason')=='official_target_has_score_or_non_scheduled_state' else 'unsupported'
        return {'status':status, 'reason':fact.get('reason','official_identity_not_confirmed')}
    if fact['native_event_id'] != candidate['native_event_id']:
        return {'status':'unsupported','reason':'official_native_fixture_id_conflict'}
    if fact['native_event_id']==_LEGACY_EVENT and (fact['home_name'],fact['away_name'])!=('IF Gnistan','FC Inter'):
        return {'status':'unsupported','reason':'existing_official_event_namespace_identity_conflict'}
    lead = parse_time(fact['kickoff'])-at
    if not timedelta(minutes=30)<=lead<=timedelta(days=horizon_days):
        return {'status':'not_prematch','reason':f'outside_registered_30min_to_{horizon_days}day_preparation_horizon'}
    return {'status':'ready','fact':fact}


def _request_digest(spec):
    return _digest({k:v for k,v in spec.items() if k!='request_digest'})


def _receipt(archive, spec, observation, *, as_of):
    params = {'group':_group(spec.get('schedule_group','2'))}
    if (spec.get('source')!='open_page' or spec.get('url')!=RESULTS_URL or spec.get('params')!=params
            or spec.get('stage')!=_STAGE or spec.get('request_digest')!=_request_digest(spec)
            or observation.source!='open_page' or observation.endpoint!=RESULTS_URL
            or observation.params!=params
            or observation.context.get('preparation_request_digest')!=spec['request_digest']
            or parse_time(observation.context.get('requested_at'))!=parse_time(spec['requested_at'])
            or not parse_time(spec['requested_at'])<=observation.observed_at<=parse_time(as_of)):
        raise ValueError('Official identity request, raw response or actual clocks disagree')
    archive.read_body(observation)
    return {'stage':_STAGE,'observation_id':observation.id,'raw_sha256':observation.content_hash,
        'url':observation.endpoint,'params':deepcopy(observation.params),'http_attempts':1,
        'requested_at':spec['requested_at'],'observed_at':observation.observed_at.isoformat(),
        'http_status':observation.http_status,'error_class':observation.error_class,
        'request_digest':spec['request_digest']}


def _request_files(archive, *, complete):
    directory = archive.root/'requests'
    intents = {p.name for p in directory.glob('*.intent.json')}
    receipts = {p.name for p in directory.glob('*.receipt.json')}
    if (not intents <= {_STAGE+'.intent.json'} or not receipts <= {_STAGE+'.receipt.json'}
            or complete and (len(intents)!=1 or len(receipts)!=1)):
        raise ValueError('Official archive must account for exactly one request intent and receipt')


def _binding(query, observation, body, *, at, horizon_days=7, schedule_group='2'):
    horizon_days = _horizon(horizon_days)
    schedule_group = _group(schedule_group)
    if observation.http_status!=200 or observation.error_class:
        return {'status':'unavailable','reason':'official_schedule_request_failed'}
    if len(body)>2_000_000 or observation.context.get('response_complete') is False:
        return {'status':'unavailable','reason':'official_schedule_incomplete_or_oversized'}
    resolved = _resolve(body, query, at, horizon_days=horizon_days, schedule_group=schedule_group)
    if resolved['status']!='ready':
        return resolved
    fact, registry = resolved['fact'], _registry(schedule_group)
    legacy = fact['native_event_id']==_LEGACY_EVENT
    canonical = {key:fact[key] for key in ('source_event_id','source_url','home_name','away_name',
        'kickoff','competition','season','stage')}
    if legacy:
        canonical['source_event_id']='palloliitto:match:'+_LEGACY_EVENT
    canonical.update(category='senior_men_club',event_status='EventScheduled',
        parser_version=PARSER_VERSION if schedule_group=='2' else GROUP3_PARSER_VERSION,
        category_basis='registered_official_mens_league_and_explicit_championship_stage' if schedule_group=='2'
            else 'registered_official_mens_league_and_explicit_relegation_stage',
        team_id_basis='preserved_registered_name_ids' if legacy else 'official_league_native_team_ids',
        score_period='unknown',settlement_rule='unknown')
    for side in ('home','away'):
        team = registry[fact[side+'_name']]
        canonical[side+'_id'] = ('finland:veikkausliiga:2026:men:'+team['slug'] if legacy
                                else 'veikkausliiga:team:'+team['native_id'])
    result = {'schema':VERSION,'status':'CONFIRMED','reason':'official_native_schedule_identity_confirmed',
        'match_id':canonical['source_event_id'],'native_event_id':fact['native_event_id'],
        'canonical':canonical,'aliases':{side:registry[fact[side+'_name']]['aliases'] for side in ('home','away')},
        'query':deepcopy(query),'registry_sha256':_digest(registry),'verified_at':at.isoformat(),
        'prediction_cutoff':(parse_time(fact['kickoff'])-timedelta(minutes=1)).isoformat(),
        'observed_generic_entry_urls':fact['observed_generic_entry_urls'],
        'observations':[{'observation_id':observation.id,'raw_sha256':observation.content_hash,
            'url':observation.endpoint,'params':deepcopy(observation.params),
            'requested_at':observation.context['requested_at'],'observed_at':observation.observed_at.isoformat()}],
        'source_projections':[fact],'namespace_basis':'preserved_existing_official_event_identity' if legacy
            else 'observed_league_native_event','settlement_status':'PENDING','settlement_rows':[],
        'regulation_rule':None,'retroactive_actor_evidence':False}
    if horizon_days != 7:
        result['preparation_horizon_days'] = horizon_days
    if schedule_group != '2':
        result['schedule_group'] = schedule_group
    result['binding_sha256']=_digest(result)
    return {'status':'ready','binding':result}


def verify_official_fixture(binding, *, archive_root, as_of):
    """Reconstruct this version from raw bytes and its single actual receipt."""
    at = parse_time(as_of)
    if (not isinstance(binding,dict) or binding.get('schema')!=VERSION or binding.get('status')!='CONFIRMED'
            or binding.get('binding_sha256')!=_digest({k:v for k,v in binding.items() if k!='binding_sha256'})
            or parse_time(binding['verified_at'])>at):
        raise ValueError('Official identity is changed or not available at verification')
    archive = RawArchive(Path(archive_root))
    _request_files(archive,complete=True)
    observations = archive.observations()
    if len(observations)!=1 or len(binding['observations'])!=1 or observations[0].id!=binding['observations'][0]['observation_id']:
        raise ValueError('Exactly one official schedule HTTP response is required')
    spec = _read(archive.root/'requests'/(_STAGE+'.intent.json'))
    horizon_days = _horizon(spec.get('preparation_horizon_days', 7))
    schedule_group = _group(spec.get('schedule_group','2'))
    if _group(binding.get('schedule_group','2')) != schedule_group:
        raise ValueError('Official identity group differs from its pre-query request commitment')
    if _horizon(binding.get('preparation_horizon_days', 7)) != horizon_days:
        raise ValueError('Official identity horizon differs from its pre-query request commitment')
    expected = _receipt(archive,spec,observations[0],as_of=binding['verified_at'])
    if _read(archive.root/'requests'/(_STAGE+'.receipt.json')) != expected:
        raise ValueError('Official identity receipt changed')
    rebuilt = _binding(binding['query'],observations[0],archive.read_body(observations[0]),
                       at=parse_time(binding['verified_at']),horizon_days=horizon_days,schedule_group=schedule_group)
    if rebuilt.get('status')!='ready' or rebuilt['binding']!=binding:
        raise ValueError('Official identity does not reproduce its original raw evidence')
    return rebuilt['binding']


def prepare_official_fixture(query, *, output, max_http_requests=1, clock=None, transport=None, resume=False,
                             horizon_days=7, schedule_group='2'):
    """Prepare one official fixture; transport is a trusted offline test seam.

    A successful resume returns the same original snapshot, not a live refresh.
    Consumers must still enforce its prediction cutoff before executing actions.
    No odds, initial probabilities, labels or market manifest are constructed.
    The default seven-day contract preserves its original archive shape. An
    explicit fourteen-day window is bound before I/O and cannot change on resume.
    """
    horizon_days = _horizon(horizon_days)
    schedule_group = _group(schedule_group)
    query = _query(query,schedule_group=schedule_group)
    if type(max_http_requests) is not int or not 0<=max_http_requests<=1 or type(resume) is not bool:
        raise ValueError('Official preparation allows zero or one HTTP request and an explicit boolean resume')
    clock = clock or (lambda:datetime.now(timezone.utc))
    now = parse_time(clock())
    output = Path(output).resolve()
    if output.exists() and not resume:
        raise ValueError('Use a new official preparation directory or explicit resume')
    output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/'.prepare.lock'),timeout=0):
        plan_path = output/'preparation-plan.json'
        if plan_path.exists():
            plan = _read(plan_path)
            if (plan.get('schema')!=PREPARATION_VERSION or plan.get('query')!=query
                    or plan.get('max_http_requests')!=max_http_requests or plan.get('registry_sha256')!=_digest(_registry(schedule_group))
                    or plan.get('horizon_days', 7)!=horizon_days or plan.get('horizon_hours')!=24*horizon_days
                    or plan.get('schedule_group','2')!=schedule_group
                    or plan.get('minimum_lead_minutes')!=30
                    or plan.get('automatic_retries')!=0 or parse_time(plan['created_at'])>now):
                raise ValueError('Official resume query, budget, horizon, capabilities or clock changed')
        else:
            if resume and any(p.name!='.prepare.lock' for p in output.iterdir()):
                raise ValueError('Cannot resume an unrecognized official preparation directory')
            plan = {'schema':PREPARATION_VERSION,'query':query,'created_at':now.isoformat(),
                'max_http_requests':max_http_requests,'registry_sha256':_digest(_registry(schedule_group)),
                'minimum_lead_minutes':30,'horizon_hours':24*horizon_days,'automatic_retries':0}
            if horizon_days != 7:
                plan['horizon_days'] = horizon_days
            if schedule_group != '2':
                plan['schedule_group'] = schedule_group
            _put(plan_path,plan)
        archive = RawArchive(output/'identity-archive')
        _request_files(archive,complete=False)
        intent = archive.root/'requests'/(_STAGE+'.intent.json')
        receipt_path = archive.root/'requests'/(_STAGE+'.receipt.json')
        result = None
        if 'unsupported_reason' in query:
            result = {'status':'unsupported','reason':query['unsupported_reason']}
        elif not max_http_requests:
            result = {'status':'budget_exhausted','reason':'official_preparation_http_budget_is_zero'}
        else:
            existed = intent.exists()
            if existed:
                spec = _read(intent)
                if _horizon(spec.get('preparation_horizon_days', 7)) != horizon_days:
                    raise ValueError('Official request horizon differs from the frozen preparation plan')
                if _group(spec.get('schedule_group','2')) != schedule_group:
                    raise ValueError('Official request group differs from the frozen preparation plan')
            else:
                if archive.observations() or receipt_path.exists():
                    raise ValueError('Official raw response or receipt has no precommitted request')
                spec = {'stage':_STAGE,'source':'open_page','url':RESULTS_URL,'params':{'group':schedule_group},
                    'requested_at':parse_time(clock()).isoformat()}
                if horizon_days != 7:
                    spec['preparation_horizon_days'] = horizon_days
                if schedule_group != '2':
                    spec['schedule_group'] = schedule_group
                spec['request_digest']=_request_digest(spec)
                _put(intent,spec)
            observations = archive.observations()
            if not observations and existed:
                result = {'status':'interrupted','reason':'existing_request_intent_without_raw_never_retried'}
            elif not observations:
                if transport is None:
                    from .system_tls_source import capture_system_tls_page
                    transport = capture_system_tls_page
                returned = transport(deepcopy(spec),archive=archive,clock=clock)
                observations = archive.observations()
                if len(observations)!=1 or returned is None or observations[0].id!=returned.id:
                    raise ValueError('Official transport must archive exactly its one actual response')
            if result is None:
                if len(observations)!=1:
                    raise ValueError('Official preparation cannot hide extra archived HTTP responses')
                observation = observations[0]
                if parse_time(spec['requested_at'])<parse_time(plan['created_at']):
                    raise ValueError('Official actual request cannot precede the saved preparation plan')
                _put(receipt_path,_receipt(archive,spec,observation,as_of=clock()))
                checkpoint = output/'identity-clock.json'
                if not checkpoint.exists():
                    _put(checkpoint,{'at':parse_time(clock()).isoformat()})
                verified = parse_time(_read(checkpoint)['at'])
                if not observation.observed_at<=verified<=parse_time(clock()):
                    raise ValueError('Official identity derivation must follow observation and not precede actual availability')
                result = _binding(query,observation,archive.read_body(observation),at=verified,
                                  horizon_days=horizon_days,schedule_group=schedule_group)
                if result['status']=='ready':
                    binding = result.pop('binding')
                    _put(output/'identity.json',binding)
                    verify_official_fixture(binding,archive_root=archive.root,as_of=clock())
                    result.update(match_id=binding['match_id'],source_event_id=binding['canonical']['source_event_id'],
                        native_event_id=binding['native_event_id'],prediction_cutoff=binding['prediction_cutoff'])
        result.update(schema=PREPARATION_VERSION,query=query,prepare_output=str(output),
            identity_path=str(output/'identity.json'),identity_archive=str(archive.root),
            request_intent_path=str(intent),request_receipt_path=str(receipt_path),
            request_intents=int(intent.exists()),archived_http_responses=len(archive.observations()),
            preparation_plan_sha256=sha256(plan_path.read_bytes()).hexdigest())
        if horizon_days != 7:
            result['preparation_horizon_days'] = horizon_days
        if schedule_group != '2':
            result['schedule_group'] = schedule_group
        result_path = output/'prepare-result.json'
        if result_path.exists() and _read(result_path)!=result and _read(result_path).get('status')=='interrupted':
            previous = _read(result_path)
            _put(output/'result-history'/(_digest(previous)+'.json'),previous)
            atomic_write(result_path,json_bytes(result))
        else:
            _put(result_path,result)
        return result


__all__ = ['VERSION','PREPARATION_VERSION','prepare_official_fixture','verify_official_fixture']
