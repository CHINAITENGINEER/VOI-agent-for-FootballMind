"""Immutable consumer bundles, publication committed with COMPLETE.json last."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import hashlib
import json
from filelock import FileLock

from football_spider.features.store import FeatureStore,select_matches
from football_spider.features.season import VERSION,DEFAULT_ASSUMPTIONS,assumptions_checked
from football_spider.services.reader import match_projection,plain
from football_spider.storage.raw_archive import atomic_write,json_bytes,digest


def export_season(repo,archive,output_root,*,source='understat',season='2025',observation_id=None,
                  information_mode='historical_reconstruction',lead_hours=(24,1),assumptions=None):
    if information_mode not in ('operational','historical_reconstruction'):raise ValueError('Unknown information mode')
    a=assumptions_checked(assumptions or DEFAULT_ASSUMPTIONS) if information_mode=='historical_reconstruction' else None
    rows=repo.conn.execute('SELECT * FROM fm.match_versions WHERE source=%s AND record->>\'season\'=%s',(source,str(season))).fetchall()
    if observation_id:rows=[r for r in rows if str(r['observation_id'])==str(observation_id)]
    rows=sorted(select_matches(rows).values(),key=lambda r:(r['event_time'],str(r['match_id'])))
    if not rows:raise ValueError('No normalized season rows')
    ids=sorted(str(r['id']) for r in rows)
    # A restored immutable warehouse reproduces this identity. Re-ingesting raw
    # into a different database has new ingestion/mapping times: a new bundle.
    dataset_version='season-export-v1:'+digest([[match_projection(r) for r in rows],information_mode,a,list(lead_hours),VERSION])
    dest=Path(output_root)/f'{source}-{season}-{information_mode}-{dataset_version.split(":")[1][:16]}'
    dest.mkdir(parents=True,exist_ok=True)
    with FileLock(str(dest/'.export.lock')):
        if (dest/'COMPLETE.json').exists():
            body=(dest/'manifest.json').read_bytes()
            marker=json.loads((dest/'COMPLETE.json').read_bytes())
            if hashlib.sha256(body).hexdigest()!=marker['manifest_sha256']:
                raise ValueError('Completed manifest checksum mismatch')
            manifest=json.loads(body)
            for item in manifest['files'].values():
                file=(dest/item['path']).resolve()
                if not file.is_relative_to(dest.resolve()) or hashlib.sha256(file.read_bytes()).hexdigest()!=item['sha256']:
                    raise ValueError('Completed data checksum mismatch')
            return {'directory':str(dest),'manifest':manifest,'reused':True}
        store=FeatureStore(repo,archive.root/'features')
        matches=[];features=[];exclusions=[]
        for row in rows:
            match=match_projection(row);match['information_mode']=information_mode
            if a:
                match['fixture_available_at']=(row['event_time']-timedelta(hours=a['fixture_lead_hours'])).isoformat()
                match['result_available_at']=(row['event_time']+timedelta(hours=a['result_delay_hours'])).isoformat() if row['status']=='finished' else None
            matches.append(match)
            for lead in lead_hours:
                if lead<=0:raise ValueError('Positive lead required')
                cutoff=row['event_time']-timedelta(hours=lead)
                try:
                    feature=store.get_features(str(row['match_id']),cutoff,VERSION,information_mode,version_ids=ids,assumptions=a)
                    features.append(feature)
                except ValueError as exc:
                    exclusions.append({'match_id':str(row['match_id']),'cutoff':cutoff.isoformat(),'reason':str(exc)})
        source_ids=sorted({str(r['observation_id']) for r in rows})
        observations={o.id:o for o in archive.observations(source)}
        for oid in source_ids:
            archive.read_body(observations[oid])  # Verify the immutable original payload.
        evidence=[{'observation_id':oid,'content_hash':observations[oid].content_hash,'payload_path':observations[oid].payload_path,
                   'observed_at':observations[oid].observed_at.isoformat(),'original_context':observations[oid].context} for oid in source_ids]
        mappings=repo.conn.execute('SELECT * FROM fm.entity_source_map WHERE source=%s ORDER BY entity_type,source_id,version',(source,)).fetchall()
        issues=repo.conn.execute('SELECT rule,severity,record_key,detail FROM fm.quality_issues WHERE observation_id=ANY(%s::uuid[]) ORDER BY issue_key',(source_ids,)).fetchall()
        files={}
        for name,records in [('matches',matches),('features',features),('id_mappings',mappings),('quality',issues),('feature_exclusions',exclusions)]:
            body=b''.join(json_bytes(r)+b'\n' for r in records)
            filename=name+'.jsonl';atomic_write(dest/filename,body)
            files[name]={'path':filename,'sha256':hashlib.sha256(body).hexdigest(),'row_count':len(records)}
        manifest={'schema':'footballmind.platform.history-export.v1','dataset_version':dataset_version,'source':source,'season':str(season),
            'information_mode':information_mode,'source_observed_at':max(o.observed_at for o in observations.values() if o.id in source_ids).isoformat(),
            'source_evidence':evidence,'assumptions':a,'feature_version':VERSION,'lead_hours':list(lead_hours),'version_ids':ids,'files':files,
            'generated_at':datetime.now(timezone.utc).isoformat(),'source_revision_bias':a['revision_bias'] if a else 'actual_ingestion_gating_no_backdating',
            'feature_leakage_policy':'target_results_target_xg_forecast_and_season_aggregates_excluded',
            'scope':'frozen_source_season; no_claim_of_historical_operational_availability'}
        body=json_bytes(manifest);atomic_write(dest/'manifest.json',body)
        atomic_write(dest/'COMPLETE.json',json_bytes({'manifest_sha256':hashlib.sha256(body).hexdigest()}))
        return {'directory':str(dest),'manifest':manifest,'reused':False}
