"""Consume the completed warehouse backup without refetching a season."""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import subprocess

from football_spider.tools.round2_verify import ROOT,OUT,DATA,pg,counts
from football_spider.tools.pg_test_cluster import local_cluster
from football_spider.storage.postgres import Repository
from football_spider.storage.raw_archive import RawArchive,atomic_write,json_bytes
from football_spider.services.results import export_results
from football_spider.models.contracts import Observation
from football_spider.platform_config import PlatformSettings
from football_spider.collectors.sources import collect
from football_spider.normalization.sources import normalize
from datetime import timedelta

def main():
    original=json.loads((OUT/'acceptance.json').read_bytes())
    report={'environment':'isolated_validation','started_at':datetime.now(timezone.utc).isoformat(),
            'input_acceptance_run':original['run_id'],'source_backup':original['backup']}
    original_observations={o.id for o in RawArchive(DATA).observations()}
    with local_cluster(OUT/'pg') as (url,cluster):
        pg('pg_restore','--no-owner','--no-privileges','-d',url,original['backup'])
        repo=Repository(url);repo.migrate();archive=RawArchive(DATA)
        inputs={'match_id':'spftll:3013698','topic':'fixture','request_id':'ca41d678-18ee-4619-8745-bebf14277d07',
                'prediction_cutoff':report['started_at'],'max_requests':1,'timeout_seconds':40}
        previous_input=OUT/'agent-adapter-check'/inputs['request_id']/'refresh.json'
        if previous_input.exists():inputs=json.loads(previous_input.read_bytes())['inputs']
        prior_url=os.environ.get('DATABASE_URL');prior_environment=os.environ.get('PLATFORM_ENVIRONMENT')
        try:
            os.environ['DATABASE_URL']=url
            os.environ['PLATFORM_ENVIRONMENT']='validation'
            atomic_write(OUT/'agent-consumer-input.json',json_bytes(inputs))
            subprocess.run([os.getenv('CONSUMER_PYTHON','D:/Develop/Python/python.exe'),'-m',
                'football_spider.tools.round2_agent_probe',str(OUT/'agent-consumer-input.json')],check=True,cwd=ROOT,timeout=55)
            result=json.loads((OUT/'agent-consumer-result.json').read_bytes())
            if result['status']!='OK':
                # One archived schedule request selects a real future ID; no
                # guessing from stale fixtures or repeated empty detail calls.
                selection_path=OUT/'live-fixture-selection.json'
                if selection_path.exists():selection=json.loads(selection_path.read_bytes())
                else:
                    today=datetime.now(timezone.utc).date()
                    collection=collect('spftll',PlatformSettings(database_url=url,data_dir=DATA,max_requests=1,detail_limit=0),
                        archive,start=today,end=today+timedelta(days=7))
                    candidates=[];obs_ids=set(collection.observations)
                    for obs in archive.observations('spftll'):
                        if obs.id in obs_ids and obs.http_status==200:
                            repo.process(obs,archive)
                            candidates += [m for m in normalize(obs,json.loads(archive.read_body(obs))).matches
                                if m.status=='scheduled' and m.kickoff>datetime.now(timezone.utc)+timedelta(hours=1)]
                    candidates.sort(key=lambda m:m.kickoff)
                    selection={'collection':collection.model_dump(mode='json'),
                        'match':candidates[0].model_dump(mode='json') if candidates else None}
                    atomic_write(selection_path,json_bytes(selection))
                report['live_fixture_selection']=selection
                for obs in archive.observations('spftll'):
                    if obs.id in selection['collection']['observations']:repo.process(obs,archive)
                if selection['match']:
                    inputs2={**inputs,'match_id':'spftll:'+selection['match']['source_id'],
                        'request_id':'9a4ee2b9-e27e-4a4c-9574-9c783eb9c6ae'}
                    prior_valid=OUT/'agent-adapter-check'/inputs2['request_id']/'refresh.json'
                    if prior_valid.exists():inputs2=json.loads(prior_valid.read_bytes())['inputs']
                    atomic_write(OUT/'agent-valid-input.json',json_bytes(inputs2))
                    subprocess.run([os.getenv('CONSUMER_PYTHON','D:/Develop/Python/python.exe'),'-m',
                        'football_spider.tools.round2_agent_probe',str(OUT/'agent-valid-input.json')],check=True,cwd=ROOT,timeout=55)
                    report['agent_valid_live_call']=json.loads((OUT/'agent-consumer-result.json').read_bytes())
                    valid_journal=DATA/'refresh_operations'/f"{inputs2['request_id']}.json"
                    if valid_journal.exists():
                        valid_raw=json.loads(valid_journal.read_bytes());report['valid_domain_refresh']=valid_raw
                        for raw in valid_raw['observations']:repo.process(Observation.model_validate(raw),archive)
        finally:
            if prior_url is None:os.environ.pop('DATABASE_URL',None)
            else:os.environ['DATABASE_URL']=prior_url
            if prior_environment is None:os.environ.pop('PLATFORM_ENVIRONMENT',None)
            else:os.environ['PLATFORM_ENVIRONMENT']=prior_environment
        report['agent_live_call']=result
        # Reconcile an existing idempotent journal after backup restoration.
        journal=DATA/'refresh_operations'/f"{inputs['request_id']}.json"
        if journal.exists():
            report['domain_refresh']=json.loads(journal.read_bytes())
            report['journal_replay']=[repo.process(Observation.model_validate(o),archive) for o in report['domain_refresh']['observations']]
        requested=json.loads((ROOT/'var/parallel/baselines/round2/settlement-20260922/results.report.json').read_bytes())
        ids=[r['match_id'] for r in requested['skipped']]
        frozen=export_results(repo,archive,ids,OUT/'results')
        report['prospective_result_export']={'directory':frozen['directory'],'manifest':frozen['manifest']}
        from football_research.data_adapters.platform_history import load_completed_history_export
        from football_research.data_adapters.settlement import export_results as consumer
        dataset=load_completed_history_export(frozen['directory'])
        _,report['consumer_settlement']=consumer(dataset,ids)
        # Verify cross-source reads return one canonical match and both providers.
        from football_platform.api import PlatformReader
        with PlatformReader(url,DATA) as platform:
            cutoff=datetime.now(timezone.utc)
            source_a=platform.resolve_id('understat','28778');source_b=platform.resolve_id('spftll','2789129')
            report['cross_source_read']={'same_canonical_id':source_a['entity_id']==source_b['entity_id'],
                'understat':platform.get_match('understat:28778',cutoff,source='understat'),
                'spftll':platform.get_match('spftll:2789129',cutoff,source='spftll'),
                'evidence_sources':sorted({r['source'] for r in platform.get_evidence('understat:28778',cutoff)})}
        report['final_counts']=counts(repo)
        backup=OUT/'footballmind-round2-followup-validation.dump';pg('pg_dump','--format=custom','--file',backup,url)
        report['backup']=str(backup);repo.close()
    report['database_state']='STOPPED_RETAINED_NOT_PRODUCTION';report['finished_at']=datetime.now(timezone.utc).isoformat()
    report['new_observation_ids_this_execution']=sorted({o.id for o in RawArchive(DATA).observations()}-original_observations)
    report['http_requests_this_execution']=len(report['new_observation_ids_this_execution'])
    report['status']='PARTIAL' if report['consumer_settlement']['skipped_matches'] else 'OK'
    atomic_write(OUT/'followup.json',json_bytes(report))
    print(json_bytes({'status':report['status'],'agent_empty_source_status':result['status'],
                     'agent_valid_status':report.get('agent_valid_live_call',{}).get('status'),
                     'http_requests_this_execution':report['http_requests_this_execution'],'export':frozen['directory'],
                     'final_matches':frozen['manifest']['final_matches'],'pending_matches':frozen['manifest']['pending_matches'],
                     'settlement_consumer':report['consumer_settlement']['skip_reason_counts']}).decode(),flush=True)

if __name__=='__main__':main()
