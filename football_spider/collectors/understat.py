"""Proven league XHR, and explicit immutable research-archive registration."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from uuid import UUID

from filelock import FileLock
from football_platform.collectors.http_transport import RequestTransport, TransportError
from football_platform.collectors.envelopes import response_count
from football_spider.collectors.quota import request_slot,reserve
from football_spider.identity.resolver import stable_id
from football_spider.models.contracts import CollectionResult,RunStatus


TIME_POLICY='UTC interpreted from official frontend timezoneOffset conversion; source strings retained; see round2 timezone evidence'


def register_research(archive, manifest_path, *, workspace_root):
    root=Path(workspace_root).resolve()
    path=Path(manifest_path).resolve()
    if not path.is_relative_to(root/'var/parallel/collectors'):
        raise ValueError('Only explicitly registered real research archives are accepted, not fixtures')
    manifest=json.loads(path.read_bytes())
    payload=(root/manifest['payload']).resolve()
    if not payload.is_relative_to(root/'var/parallel/collectors'):
        raise ValueError('Research payload outside approved evidence root')
    raw=payload.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['sha256'] or manifest['http_status']!=200:
        raise ValueError('Research evidence integrity/status failed')
    endpoint=manifest.get('url') or manifest.get('endpoint')
    if endpoint!='https://understat.com/getLeagueData/EPL/2025':
        raise ValueError('Research endpoint is not the verified league-season')
    oid=manifest.get('evidence_id') or manifest.get('observation_id')
    UUID(oid)
    requested_id=manifest.get('request_id') or stable_id('understat','research-request',oid)
    context={'research_archive_registration':True,'original_evidence_id':oid,
             'original_manifest':str(path.relative_to(root)),'original_payload_sha256':manifest['sha256'],
             'league':'EPL','season':2025,'kickoff_timezone':'UTC','timezone_policy':TIME_POLICY}
    with FileLock(str(archive.root/'understat-registration.lock')):
        return archive.save('understat',endpoint,{},raw,status=200,headers={'Content-Type':manifest.get('content_type','text/javascript')},
            request_id=requested_id,observation_id=oid,observed_at=datetime.fromisoformat(manifest['observed_at'].replace('Z','+00:00')),
            context=context)


def collect_understat(config,archive,*,league='EPL',season=2025):
    # 2024 was verified from archived official season selector + JS constructor,
    # then observation 843eefe4-1c04-45c4-a3ae-b319bf6096e9 (380/20/38 audit).
    if league!='EPL' or season not in (2024,2025):
        raise ValueError('Only EPL/2024 and EPL/2025 have been source-verified; extend coverage explicitly')
    observations=[]
    def persist(**facts):
        observations.append(archive.save(facts['source'],facts['endpoint'],facts['params'],facts['body'],
            status=facts['http_status'],headers=facts['response_headers'],request_id=facts['request_id'],
            observation_id=facts['observation_id'],observed_at=facts['observed_at'],error_class=facts['error_class'],
            context={'league':league,'season':season,'kickoff_timezone':'UTC','timezone_policy':TIME_POLICY,
                     'request_started_at':facts['requested_at'].isoformat(),'attempt':facts['attempt'],
                     'credential_echo_redacted':facts['credential_echo_redacted']}))
    def quota(source,cost):
        if not reserve(archive.root,source,1,20):
            return False
        delay=request_slot(archive.root,source,3.0)
        if delay is None:return False
        import time
        time.sleep(delay)
        return True
    result=CollectionResult(source='understat',scope={'league':league,'season':season})
    archive.root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(archive.root/'understat-collector.lock')):
        with RequestTransport('understat',on_attempt=persist,max_attempts=2,request_budget=min(config.max_requests,2),reserve=quota,
                              interval=3.0,cookie_path=archive.root/'cookies/understat.txt') as client:
            try:
                body=client.get(f'https://understat.com/getLeagueData/{league}/{season}',
                    headers={'Referer':f'https://understat.com/league/{league}/{season}','X-Requested-With':'XMLHttpRequest'})
                count=response_count('understat','league',body)
                result.scope['response_matches']=count
                if not count:result.status=RunStatus.NO_DATA
            except TransportError as exc:
                result.status=RunStatus.RATE_LIMITED if exc.category=='RATE_LIMITED' else RunStatus.FAILED
                result.errors.append({'category':exc.category,'observation_id':exc.observation_id})
            except ValueError as exc:
                result.status=RunStatus.FAILED
                result.errors.append({'category':type(exc).__name__})
            result.requests=client.stats['attempts']
            result.scope['request_statistics']=dict(client.stats)
    result.observations=[o.id for o in observations]
    return result
