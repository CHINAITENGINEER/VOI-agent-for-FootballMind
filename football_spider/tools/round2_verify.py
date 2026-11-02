"""Reproducible real-data acceptance in retained, isolated PostgreSQL clusters."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from uuid import uuid4

from football_spider.tools.pg_test_cluster import local_cluster
from football_spider.storage.postgres import Repository
from football_spider.storage.raw_archive import RawArchive,atomic_write,json_bytes
from football_spider.collectors.understat import register_research
from football_spider.normalization.understat import audit_season
from football_spider.services.export import export_season
from football_spider.services.reader import PlatformReader
from football_spider.services.refresh import RefreshService
from football_spider.platform_config import PlatformSettings
from football_spider.features.store import FeatureStore
from football_spider.identity.reviewed_links import apply_reviewed_epl_links
from football_spider.services.results import prediction_match_ids,refresh_results,export_results

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'var/parallel/collectors/round2'
DATA=ROOT/'football_spider/platform_data'

def pg(name,*args):
    subprocess.run([str(Path(os.getenv('PG_BIN','D:/PostgreSQL/bin'))/name),*map(str,args)],check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,timeout=90)

def counts(repo):
    return {t:repo.conn.execute('SELECT count(*) AS n FROM fm.'+t).fetchone()['n'] for t in
            ('raw_observations','match_versions','team_statistics','feature_snapshots','matches','odds_history','mapping_candidates')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--inspect-only',action='store_true');p.add_argument('--refresh',action='store_true')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    report={'run_id':str(uuid4()),'started_at':datetime.now(timezone.utc).isoformat(),'environment':'isolated_validation',
            'status':'RUNNING','credentials':PlatformSettings().public()['configured']}
    def save(stage):
        report['stage']=stage;atomic_write(OUT/'acceptance.json',json_bytes(report));print(stage,flush=True)
    archive=RawArchive(DATA)
    observations=[register_research(archive,ROOT/path,workspace_root=ROOT) for path in
        ('var/parallel/collectors/understat-research/data-manifest.json',
         'var/parallel/collectors/transport-live/5c90a971-37e4-40a1-bd3f-f83bb76d1f29.metadata.json')]
    body=json.loads(archive.read_body(observations[0]));report['audit']=audit_season(body,'EPL',2025)
    report['observations']=[o.model_dump(mode='json') for o in observations]
    save('registered_original_research_observations')
    with local_cluster(OUT/'pg') as (url,cluster):
        report['validation_cluster']=str(cluster)
        pg('pg_restore','--no-owner','--no-privileges','-d',url,DATA/'validation/footballmind-real-validation.dump')
        repo=Repository(url);repo.migrate();repo.migrate();report['before']=counts(repo)
        old=repo.conn.execute("SELECT DISTINCT ON(source_id) * FROM fm.match_versions WHERE source='spftll' ORDER BY source_id,observed_at DESC").fetchall()
        overlaps=[]
        for m in body['dates']:
            kickoff=datetime.fromisoformat(m['datetime']).replace(tzinfo=timezone.utc)
            for r in old:
                rec=r['record']
                if (r['event_time']==kickoff and rec['competition']['name']=='英超' and
                    rec['score']=={'home':int(m['goals']['h']),'away':int(m['goals']['a'])}):
                    overlaps.append({'understat_id':m['id'],'spftll_id':r['source_id'],'kickoff':kickoff,'score':rec['score'],
                        'understat_teams':[m['h'],m['a']],'spftll_teams':[rec['home'],rec['away']],
                        'spftll_version_id':r['id'],'spftll_observation_id':r['observation_id']})
        report['cross_source_overlap_candidates']=overlaps
        atomic_write(OUT/'cross-source-candidates.json',json_bytes(overlaps))
        if args.inspect_only:
            repo.close();save('inspection_complete');return
        with repo.conn.transaction():
            report['reviewed_identity_links']=apply_reviewed_epl_links(repo,overlaps,observations[0].id)
        atomic_write(OUT/'reviewed-identity-links.json',json_bytes(report['reviewed_identity_links']))
        report['processing']=[]
        for obs in observations:
            result=repo.process(obs,archive);report['processing'].append(result)
            if result['status']!='OK':raise RuntimeError(f'Ingest failed: {result}')
        report['replay']=[repo.process(o,archive) for o in observations]
        report['after_ingest']=counts(repo);save('two_observations_ingested_replay_idempotent')
        exports={}
        for mode in ('historical_reconstruction','operational'):
            exports[mode]=export_season(repo,archive,OUT/'exports',observation_id=observations[0].id,information_mode=mode)
            save('exported_'+mode)
        report['exports']={m:{'directory':v['directory'],'files':v['manifest']['files'],'dataset_version':v['manifest']['dataset_version']} for m,v in exports.items()}
        # Consumer code is invoked by this acceptance tool, never a platform dependency.
        from football_research.data_adapters.platform_history import load_completed_history_export
        from football_research.data_adapters.batch_features import load_feature_index
        dataset=load_completed_history_export(exports['historical_reconstruction']['directory'])
        index=load_feature_index(exports['historical_reconstruction']['directory'],strict=True)
        joined,join_report=index.assemble_training_rows(dataset.matches,feature_version='prematch-season-v2.0',lead_hours=24,
            required_features=('home_elo','away_elo','home_rolling_xg_for','away_rolling_xg_for'),
            information_mode='historical_reconstruction',allow_missing_values=False)
        report['baseline_consumer']={'matches':len(dataset.matches),'feature_index':index.report(),'training_join':join_report}
        atomic_write(OUT/'consumer-training-join.json',json_bytes({'rows':joined,'report':join_report}))
        save('baseline_read_real_completed_export')
        features=repo.conn.execute("SELECT * FROM fm.feature_snapshots WHERE feature_version='prematch-season-v2.0' ORDER BY prediction_cutoff").fetchall()
        store=FeatureStore(repo,DATA/'features')
        report['feature_replay']=[store.replay_snapshot(str(r['id'])) for r in (features[0],features[len(features)//2],features[-1])]
        with PlatformReader(url,DATA) as reader:
            current=datetime.now(timezone.utc)
            matches=reader.list_matches(current,source='understat',season=2025)
            report['reader']={'matches':len(matches),'id_mapping':reader.resolve_id('understat',body['dates'][0]['id']),
                'evidence_versions':len(reader.get_evidence('understat:'+body['dates'][0]['id'],current))}
        if args.refresh:
            service=RefreshService(PlatformSettings(database_url=url,data_dir=DATA,environment='validation'))
            op='f7a3cfbb-3b37-46d8-a169-6d393e71ee64'
            result=service.refresh('spftll:3013707',request_id=op,max_requests=2,timeout=60)
            report['refresh']=result
            report['refresh_repeat']=service.refresh('spftll:3013707',request_id=op,max_requests=2,timeout=60)
            # Cached request journals remain valid after a database recovery.
            from football_spider.models.contracts import Observation
            report['refresh_reprocess']=[repo.process(Observation.model_validate(o),archive) for o in result['observations']]
            target_ids=prediction_match_ids(ROOT/'var/parallel/baselines/real-asof-final/predictions.jsonl')
            report['result_refresh']=refresh_results(PlatformSettings(database_url=url,data_dir=DATA,environment='validation'),target_ids,
                operation_id='6b9c6d92-233f-483a-8974-19cf18a0c3e6',max_requests=18)
            for child in report['result_refresh']['reports']:
                for raw in child.get('observations',[]):repo.process(Observation.model_validate(raw),archive)
            results=export_results(repo,archive,target_ids,OUT/'results')
            report['result_export']={'directory':results['directory'],'manifest':results['manifest']}
            result_data=load_completed_history_export(results['directory'])
            from football_research.data_adapters.settlement import export_results as consume_results
            consumed,result_report=consume_results(result_data,target_ids)
            report['result_consumer']={'matches_read':len(result_data.matches),'settlement_report':result_report}
            atomic_write(OUT/'result-consumer-check.json',json_bytes(report['result_consumer']))
            save('bounded_live_refresh_finished')
        report['final_counts']=counts(repo)
        backup=OUT/f"footballmind-round2-{report['run_id']}.dump"
        pg('pg_dump','--format=custom','--file',backup,url)
        report['backup']=str(backup);repo.close()
    report['validation_database_state']='STOPPED_RETAINED_NOT_PRODUCTION'
    with local_cluster(OUT/'pg') as (restore_url,restore_cluster):
        pg('pg_restore','--no-owner','--no-privileges','-d',restore_url,report['backup'])
        recovered=Repository(restore_url);recovered.migrate()
        report['recovered_counts']=counts(recovered)
        assert report['recovered_counts']==report['final_counts']
        report['restored_feature_replay']=FeatureStore(recovered,DATA/'features').replay_snapshot(str(features[-1]['id']))
        recovered.close();report['restore_cluster']=str(restore_cluster)
    report['status']='OK';report['finished_at']=datetime.now(timezone.utc).isoformat();save('acceptance_complete')
    print(json.dumps({'status':report['status'],'counts':report['final_counts'],'exports':report['exports']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
