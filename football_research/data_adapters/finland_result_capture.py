"""Bounded official list/detail result capture, separate from Actor evidence."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re

from football_research.acquisition.state import parse_time
from football_spider.storage.raw_archive import RawArchive, json_bytes
from .system_tls_source import capture_system_tls_page

VERSION = "finland-results-capture-v1"
LIST_URL = "https://www.veikkausliiga.com/tilastot/2026/veikkausliiga/ottelut/"

def detail_url(binding):
    from .finland_results import match_followup_url
    if binding['canonical']['season'] != '2026':
        raise ValueError('A verified registered 2026 Finnish event is required')
    return match_followup_url(binding)

def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(json_bytes(data))

def verify_capture_receipts(binding, output, *, as_of):
    """Reconcile actual attempts separately from unchanged cached rule bytes."""
    from .finland_results import verify_rule_package, project_championship_schedule, result_schedule_group
    from .prospective_text import _digest
    output, as_of = Path(output), parse_time(as_of)
    read=lambda name: json.loads((output/name).read_text(encoding='utf-8'))
    plan, report=read('capture-plan.json'),read('capture-receipts.json')
    group=result_schedule_group(binding)
    schedule_params={'group':group}
    package=verify_rule_package(as_of=as_of,schedule_group=group)
    rule_ids=set(package['observation_ids'])
    archive=RawArchive(output)
    saved={o.id:o for o in archive.observations()}
    receipts=report['receipts']
    if (plan.get('schema')!=VERSION or report.get('schema')!=VERSION
            or plan.get('identity_binding_sha256')!=binding['binding_sha256']
            or plan.get('source_event_id')!=binding['canonical']['source_event_id']
            or plan.get('max_http_requests')!=2 or report.get('max_http_requests')!=2
            or plan.get('schedule_url')!=LIST_URL or plan.get('schedule_params')!=schedule_params
            or plan.get('possible_detail_url')!=detail_url(binding)
            or not parse_time(plan['created_at'])<=parse_time(report['completed_at'])<=as_of
            or plan.get('due') is not (parse_time(plan['created_at'])>=parse_time(binding['canonical']['kickoff'])+timedelta(minutes=90))
            or report['http_requests_actual']!=len(receipts) or not (1<=len(receipts)<=2 if plan['due'] else len(receipts)==0)
            or set(plan.get('cached_rule_observation_ids',[]))!=rule_ids
            or set(report.get('cached_rule_observation_ids',[]))!=rule_ids
            or set(saved)!=rule_ids|{r['observation_id'] for r in receipts}):
        raise ValueError('Finnish result manifest, request count or cached rules disagree')
    previous=parse_time(plan['created_at'])
    for index,receipt in enumerate(receipts):
        stage='schedule' if index==0 else 'detail'
        expected=(LIST_URL,schedule_params) if index==0 else (detail_url(binding),{})
        spec=read(f'requests/{stage}.intent.json')
        obs=saved[receipt['observation_id']]
        request_digest=_digest({k:v for k,v in spec.items() if k!='request_digest'})
        if (spec['request_digest']!=request_digest or receipt!=read(f'requests/{stage}.receipt.json')
                or (spec['url'],spec['params'])!=expected or spec.get('source')!='open_page'
                or spec.get('stage')!=stage or receipt.get('http_attempts')!=1
                or (obs.endpoint,obs.params)!=(receipt['url'],receipt['params'])
                or (obs.endpoint,obs.params)!=expected or receipt['raw_sha256']!=obs.content_hash
                or parse_time(receipt['observed_at'])!=obs.observed_at
                or parse_time(receipt['requested_at'])!=parse_time(spec['requested_at'])
                or parse_time(obs.context['requested_at'])!=parse_time(spec['requested_at'])
                or not previous<=parse_time(spec['requested_at'])<=obs.observed_at<=parse_time(report['completed_at'])):
            raise ValueError('Finnish result receipt is not its planned actual request')
        archive.read_body(obs)
        if index==0 and len(receipts)==2:
            projection=project_championship_schedule(obs,archive.read_body(obs),binding['canonical'],
                as_of=as_of,schedule_group=group)
            if projection.get('status') not in {'SCORE','PENDING'} or projection.get('followup_url')!=detail_url(binding):
                raise ValueError('Result detail was not discovered in the actual verified schedule')
        previous=obs.observed_at
    return report

def capture_finland_result(binding, *, identity_archive_root, output, clock=None, transport=None):
    """At most two HTTP attempts; a stored intent is never automatically retried."""
    from .finland_results import (
        verify_finland_settlement, project_championship_schedule, copy_rule_observations, result_schedule_group,
    )
    from .prospective_text import _digest
    from .verified_fixtures import verify_identity_binding
    clock = clock or (lambda: datetime.now(timezone.utc))
    now = parse_time(clock())
    verified = verify_identity_binding(binding, archive_root=identity_archive_root, as_of=now)
    if verified['status'] != 'CONFIRMED':
        raise ValueError('Raw identity must be confirmed before result capture')
    group=result_schedule_group(verified)
    schedule_params={'group':group}
    url = detail_url(binding)
    due = now >= parse_time(binding['canonical']['kickoff'])+timedelta(minutes=90)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = RawArchive(output)
    rules = copy_rule_observations(output, as_of=now, schedule_group=group)
    rule_ids = set(rules['observation_ids'])
    _write(output/'capture-plan.json', {'schema':VERSION, 'created_at':now.isoformat(),
        'max_http_requests':2, 'requests_planned':2 if due else 0, 'due':due,
        'identity_binding_sha256':binding['binding_sha256'], 'source_event_id':binding['canonical']['source_event_id'],
        'schedule_url':LIST_URL, 'schedule_params':schedule_params, 'possible_detail_url':url,
        'detail_condition':'exact_native_event_href_observed_in_actual_schedule_response',
        'automatic_retries':0, 'actor_evidence':False, 'cached_rule_observation_ids':sorted(rule_ids),
        'cached_rule_http_requests':0})
    receipts=[]
    transport = transport or capture_system_tls_page
    def fetch(endpoint, params, stage):
        spec={'url':endpoint,'params':params,'source':'open_page','stage':stage,
              'requested_at':parse_time(clock()).isoformat()}
        spec['request_digest']=_digest(spec)
        _write(output/'requests'/f'{stage}.intent.json',spec)
        obs=transport(spec,archive=archive,clock=clock)
        archive.read_body(obs)
        if (obs.endpoint,obs.params,obs.source)!=(endpoint,params,'open_page') or not (
                parse_time(spec['requested_at']) <= obs.observed_at <= parse_time(clock())):
            raise ValueError('Result response differs from its actual request or clocks')
        receipt={'url':obs.endpoint,'params':obs.params,'observation_id':obs.id,'raw_sha256':obs.content_hash,
            'http_attempts':1,'requested_at':spec['requested_at'],'observed_at':obs.observed_at.isoformat()}
        _write(output/'requests'/f'{stage}.receipt.json',receipt)
        receipts.append(receipt)
        return obs
    if due:
        obs=fetch(LIST_URL,schedule_params,'schedule')
        if obs.http_status==200 and not obs.error_class:
            projection=project_championship_schedule(obs,archive.read_body(obs),binding['canonical'],
                as_of=clock(),schedule_group=group)
            if projection.get('status') in {'SCORE','PENDING'} and projection.get('followup_url')==url:
                fetch(url,{},'detail')
    if len([o for o in archive.observations() if o.id not in rule_ids])!=len(receipts):
        raise ValueError('Result captures must account for every archived HTTP attempt')
    _write(output/'capture-receipts.json',{'schema':VERSION,'completed_at':parse_time(clock()).isoformat(),
        'http_requests_actual':len(receipts),'max_http_requests':2,'receipts':receipts,
        'cached_rule_observation_ids':sorted(rule_ids)})
    report=verify_finland_settlement(binding,identity_archive_root=identity_archive_root,
        result_archive_root=output,observation_ids=[o.id for o in archive.observations()],as_of=clock())
    _write(output/'verification.json',{'schema':report['schema'],'created_at':parse_time(clock()).isoformat(),
        'reports':{binding['match_id']:report},'settlement_rows':report['settlement_rows']})
    return report
