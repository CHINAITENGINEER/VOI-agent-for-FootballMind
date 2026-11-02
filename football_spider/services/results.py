"""Bounded result refresh and immutable normalized result-version export."""
from datetime import datetime,timezone
from pathlib import Path
from uuid import UUID,uuid5
import hashlib
import json
from filelock import FileLock
from football_spider.services.refresh import RefreshService
from football_spider.services.reader import match_projection,utc
from football_spider.storage.raw_archive import atomic_write,json_bytes,digest
from football_spider.models.contracts import Observation

def prediction_match_ids(path):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    return sorted({r['match_id'] for r in rows})

def refresh_results(config,match_ids,*,operation_id,max_requests=18,timeout_per_match=30):
    namespace=UUID(str(operation_id));ids=sorted(set(match_ids))
    if not 1<=max_requests<=50:raise ValueError('Result batch needs 1..50 request budget')
    service=RefreshService(config);reports=[];used=0;skipped=[]
    for mid in ids:
        if used>=max_requests:skipped.append({'match_id':mid,'reason':'BATCH_REQUEST_BUDGET'});continue
        # Each child gets a fixed single-attempt budget; retries require a new
        # explicit operation. Replay returns the old journal without new HTTP.
        result=service.refresh(mid,('result',),str(uuid5(namespace,mid)),max_requests=1,timeout=timeout_per_match)
        reports.append(result);used+=result['requests_this_call']
    statuses=[r['status'] for r in reports]
    return {'operation_id':str(namespace),'status':'PARTIAL' if skipped or any(s!='OK' for s in statuses) else 'OK',
            'requested_matches':len(ids),'requests_this_call':used,'max_requests':max_requests,'reports':reports,'skipped':skipped,
            'cost':None,'cost_status':'UNKNOWN','provider_credits':None}

def export_results(repo,archive,match_ids,output_root,*,as_of=None):
    cutoff=utc(as_of or datetime.now(timezone.utc));matches=[];versions=[];missing=[];mappings=[];quality=[]
    for requested in sorted(set(match_ids)):
        sid=requested.removeprefix('spftll:match:').removeprefix('spftll:')
        if not sid.isdigit():raise ValueError('Expected spftll source ID')
        rows=repo.conn.execute('''SELECT * FROM fm.match_versions WHERE source='spftll' AND source_id=%s
            AND observed_at<=%s AND ingested_at<=%s ORDER BY observed_at,ingested_at,id''',(sid,cutoff,cutoff)).fetchall()
        if not rows:missing.append({'match_id':requested,'reason':'NO_OBSERVED_VERSION'});continue
        # Keep consumer source-scoped IDs and explicitly export canonical mapping.
        for row in rows:
            projected=match_projection(row);projected['canonical_match_id']=projected['match_id'];projected['match_id']=requested
            projected['settlement_status']='final' if row['status']=='finished' and row['record']['score_period']=='regulation' and None not in row['record']['score'].values() else 'pending'
            versions.append(projected)
        matches.append(dict(versions[-1]))
        problems=repo.conn.execute('''SELECT q.*,o.observed_at,o.metadata FROM fm.quality_issues q
            JOIN fm.raw_observations o ON o.id=q.observation_id WHERE q.record_key=%s AND o.source='spftll'
            AND q.severity='error' AND o.observed_at<=%s AND o.ingested_at<=%s ORDER BY o.observed_at,q.id''',(sid,cutoff,cutoff)).fetchall()
        for issue in problems:
            quality.append({'match_id':requested,'rule':issue['rule'],'detail':issue['detail'],
                            'observation_id':str(issue['observation_id']),'observed_at':issue['observed_at'],
                            'payload_path':issue['metadata']['payload_path'],'content_hash':issue['metadata']['content_hash']})
        if problems and problems[-1]['observed_at']>rows[-1]['observed_at']:
            matches[-1]['settlement_status']='pending'
            matches[-1]['pending_reason']='newer_observation_quarantined; retained prior accepted fact'
            matches[-1]['blocked_observation_id']=str(problems[-1]['observation_id'])
            if matches[-1]['status']=='finished':
                matches[-1].update(status='disputed',home_goals=None,away_goals=None,result_available_at=None)
        elif matches[-1]['settlement_status']=='pending':matches[-1]['pending_reason']='not_finished_or_unverified_regulation_score'
        mappings.append({'requested_id':requested,'canonical_match_id':str(rows[-1]['match_id']),
                         'source':'spftll','source_id':sid,'evidence_version_id':str(rows[-1]['id'])})
    if not matches:raise ValueError('No requested match available')
    version='results-export-v1:'+digest([matches,versions,missing,mappings,quality])
    dest=Path(output_root)/version.replace(':','-');dest.mkdir(parents=True,exist_ok=True)
    with FileLock(str(dest/'.export.lock')):
        if (dest/'COMPLETE.json').exists():
            body=(dest/'manifest.json').read_bytes();manifest=json.loads(body)
            if hashlib.sha256(body).hexdigest()!=json.loads((dest/'COMPLETE.json').read_bytes())['manifest_sha256']:
                raise ValueError('Result manifest checksum mismatch')
            for spec in manifest['files'].values():
                if hashlib.sha256((dest/spec['path']).read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Result file checksum mismatch')
            return {'directory':str(dest),'manifest':manifest,'reused':True}
        files={}
        for name,rows in [('matches',matches),('result_versions',versions),('id_mappings',mappings),('missing',missing),('quality',quality)]:
            body=b''.join(json_bytes(r)+b'\n' for r in rows);atomic_write(dest/(name+'.jsonl'),body)
            files[name]={'path':name+'.jsonl','sha256':hashlib.sha256(body).hexdigest(),'row_count':len(rows)}
        ids={r['observation_id'] for r in versions+quality};evidence=[]
        for o in archive.observations('spftll'):
            if o.id in ids:
                archive.read_body(o)
                evidence.append({'observation_id':o.id,'content_hash':o.content_hash,'observed_at':o.observed_at.isoformat(),'payload_path':o.payload_path})
        if len(evidence)!=len(ids):raise ValueError('Result raw evidence missing')
        manifest={'schema':'footballmind.platform.history-export.v1','source':'spftll','information_mode':'operational',
            'dataset_version':version,'files':files,'source_observed_at':max(r['observed_at'] for r in versions),
            'source_evidence':evidence,'as_of':cutoff.isoformat(),'generated_at':datetime.now(timezone.utc).isoformat(),
            'timing':'actual source response observed_at and database ingested_at are distinct; availability=max(both)',
            'identity_policy':'requested source-scoped match_id preserved; explicit canonical_match_id mappings',
            'final_matches':sum(r['settlement_status']=='final' for r in matches),
            'pending_matches':sum(r['settlement_status']=='pending' for r in matches),'missing_matches':len(missing)}
        body=json_bytes(manifest);atomic_write(dest/'manifest.json',body)
        atomic_write(dest/'COMPLETE.json',json_bytes({'manifest_sha256':hashlib.sha256(body).hexdigest()}))
        return {'directory':str(dest),'manifest':manifest,'reused':False}
