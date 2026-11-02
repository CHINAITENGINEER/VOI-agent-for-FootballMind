"""Bounded spftll refresh with durable operation journal; no Agent dependency."""
from datetime import datetime,timezone
from pathlib import Path
import json
import random
import time
from uuid import UUID
from filelock import FileLock,Timeout

from football_spider.collectors.aio_transport import AioSession
from football_spider.collectors.quota import reserve,request_slot
from football_spider.config import settings
from football_spider.storage.raw_archive import RawArchive,atomic_write,json_bytes,digest
from football_spider.commands.pipeline import open_repository
from football_platform.collectors.http_transport import RequestTransport,TransportError
from football_platform.collectors.envelopes import response_count,SourceEnvelopeError


class RefreshService:
    def __init__(self,config):
        self.config=config
        self.archive=RawArchive(config.data_dir)
        self.root=self.archive.root/'refresh_operations'
        self.root.mkdir(parents=True,exist_ok=True)

    def configuration(self):
        return {'service':'spftll-refresh-v1','archive_root':str(self.archive.root),'environment':self.config.environment,
                'database_configured':bool(self.config.database_url),'max_requests_per_call':2,'billing':'UNKNOWN_NOT_ASSUMED_ZERO'}

    def estimate_requests(self,inputs):
        return min(2,max(0,int(inputs.get('max_requests',2))))

    def estimate_cost(self,inputs):return None

    def refresh(self,match_id,topics=('result',),request_id=None,max_requests=2,timeout=60):
        if not request_id:raise ValueError('Caller supplies stable request_id for idempotence')
        UUID(str(request_id))
        sid=str(match_id).removeprefix('spftll:match:').removeprefix('spftll:')
        if not sid.isdigit():raise ValueError('Refresh requires a spftll source match ID, not an unverified name/canonical ID')
        if not 1<=max_requests<=2 or not 0<timeout<=120:raise ValueError('Invalid bounded refresh budget')
        topics=tuple(sorted(set(topics)))
        if not topics or not set(topics)<={'fixture','result','history','odds'}:raise ValueError('Unsupported refresh topic')
        spec={'match_id':sid,'topics':topics,'max_requests':max_requests,'timeout':timeout}
        path=self.root/(str(request_id)+'.json')
        lock=FileLock(str(path)+'.lock')
        try:
            with lock.acquire(timeout=0):
                if path.exists():
                    previous=json.loads(path.read_bytes())
                    if previous['spec_hash']!=digest(spec):raise ValueError('request_id reused with different operation')
                    previous['cached']=True
                    previous['requests_this_call']=0
                    if previous['status']=='RUNNING':
                        # A crash may have happened after provider receipt but before local save.
                        # Do not replay an unaccounted request; reconcile retained evidence only.
                        previous['status']='PARTIAL'
                        previous['error']='INTERRUPTED_OPERATION_REQUIRES_RECONCILIATION'
                        previous['observations']=[o.model_dump(mode='json') for o in self.archive.observations('spftll')
                            if o.context.get('refresh_operation_id')==str(request_id)]
                    return previous
                return self._run(sid,topics,str(request_id),max_requests,float(timeout),path,spec)
        except Timeout:
            return {'status':'PARTIAL','error':'OPERATION_LOCKED','request_id':str(request_id),'requests_this_call':0,'cost':None}

    def _run(self,sid,topics,request_id,max_requests,timeout,path,spec):
        start=time.monotonic();deadline=start+timeout
        report={'status':'RUNNING','request_id':request_id,'spec_hash':digest(spec),'match_id':'spftll:'+sid,
                'topics':topics,'started_at':datetime.now(timezone.utc).isoformat(),'observations':[],
                'errors':[],'processing':[],'cost':None,'cost_currency':None,'cost_status':'UNKNOWN',
                'provider_credits':None,'requests_reserved':0,'requests_this_call':0,'cached':False,
                'environment':self.config.environment,'odds_time_semantics':'current_at_observation_never_backdated_to_prematch'}
        atomic_write(path,json_bytes(report))
        def persist(**f):
            obs=self.archive.save('spftll',f['endpoint'],f['params'],f['body'],status=f['http_status'],headers=f['response_headers'],
                request_id=f['request_id'],observation_id=f['observation_id'],observed_at=f['observed_at'],error_class=f['error_class'],
                context={'refresh_operation_id':request_id,'requested_source_match_id':sid,'topics':list(topics),
                         'attempt':f['attempt'],'requested_at':f['requested_at'].isoformat(),'credential_echo_redacted':f['credential_echo_redacted']})
            report['observations'].append(obs.model_dump(mode='json'))
            report['requests_this_call']+=1
            atomic_write(path,json_bytes(report))
        def quota(source,cost):
            if time.monotonic()>=deadline:return False
            if not reserve(self.archive.root,source,1,200):return False
            delay=request_slot(self.archive.root,source,1.2)
            if delay is None or time.monotonic()+delay>=deadline:return False
            time.sleep(delay)
            report['requests_reserved']+=1
            atomic_write(path,json_bytes(report))
            return True
        plans=[]
        if set(topics)&{'fixture','result','history'}:
            plans.append(('detail',settings.detail_api_url,{'gameKind':0,'matchId':sid}))
        if 'odds' in topics:plans.append(('odds',settings.more_odds_api_url,{'mKind':0,'matchId':sid}))
        try:
            session=AioSession(self.archive.root/'cookies/spftll-refresh-aio.json')
            with RequestTransport('spftll',on_attempt=persist,session=session,request_budget=max_requests,max_attempts=2,
                                  reserve=quota,interval=1.2,deadline=deadline) as client:
                for kind,url,params in plans:
                    try:
                        body=client.get(url,params,headers=settings.browser_headers(random.choice(settings.user_agents)))
                        count=response_count('spftll',kind,body)
                        if count:
                            data=json.loads(body)['data']
                            returned=data if kind=='detail' else data['footballMoreSpInfo']
                            if str(returned.get('matchId'))!=sid:raise SourceEnvelopeError('TARGET_ID_MISMATCH')
                        report.setdefault('response_counts',{})[kind]=count
                    except (TransportError,ValueError) as exc:
                        report['errors'].append({'category':getattr(exc,'category',type(exc).__name__)})
                report['request_statistics']=dict(client.stats)
            repo,database=open_repository(self.config)
            report['database']=database
            if repo:
                try:
                    from football_spider.models.contracts import Observation
                    for raw in report['observations']:
                        obs=Observation.model_validate(raw)
                        report['processing'].append({'observation_id':obs.id,**repo.process(obs,self.archive)})
                finally:repo.close()
            has_data=any(report.get('response_counts',{}).values())
            failures=bool(report['errors']) or any(p['status'] not in ('OK','NO_DATA','SKIPPED') for p in report['processing'])
            report['status']='PARTIAL' if failures else 'NEEDS_DATABASE' if not repo else 'OK' if has_data else 'NO_DATA'
            report['network_status']='OK' if has_data and not report['errors'] else 'PARTIAL' if has_data else 'FAILED' if report['errors'] else 'NO_DATA'
        except Exception as exc:
            report['status']='FAILED';report['errors'].append({'category':type(exc).__name__})
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        report['elapsed_seconds']=round(time.monotonic()-start,3)
        atomic_write(path,json_bytes(report))
        return report
