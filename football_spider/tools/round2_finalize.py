"""Read-only evidence summary and final backup recovery check; no HTTP."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
from football_spider.tools.round2_verify import ROOT,OUT,DATA,pg,counts
from football_spider.tools.pg_test_cluster import local_cluster
from football_spider.storage.postgres import Repository
from football_spider.storage.raw_archive import RawArchive,atomic_write,json_bytes
from football_spider.features.store import FeatureStore
from football_spider.platform_config import PlatformSettings

def append_consumer_regression():
    suites=list(ET.parse(OUT/'consumer-regression.xml').getroot().iter('testsuite'))
    record={field:sum(int(s.get(field,0)) for s in suites) for field in ('tests','failures','errors','skipped')}
    report=json.loads((OUT/'final-manifest.json').read_bytes())
    report['consumer_regression']={'report':str(OUT/'consumer-regression.xml'),'counts':record,
        'environment':'existing Python3.13 consumer runtime; platform PG tests separately Python3.11',
        'scope':'read_only_execution_of_other_roles_tests_no_code_changes'}
    additional=OUT/'consumer-platform-env.xml'
    if additional.exists():
        other=list(ET.parse(additional).getroot().iter('testsuite'))
        report['consumer_regression']['platform_environment_followup']={
            'report':str(additional),'reason':'module skipped in Python3.13 without psycopg; rerun with installed platform Python3.11',
            'counts':{field:sum(int(s.get(field,0)) for s in other) for field in ('tests','failures','errors','skipped')}}
    atomic_write(OUT/'final-manifest.json',json_bytes(report));print(json_bytes(record).decode())
def main():
    base=json.loads((OUT/'acceptance.json').read_bytes());follow=json.loads((OUT/'followup.json').read_bytes())
    with local_cluster(OUT/'pg') as (url,cluster):
        pg('pg_restore','--no-owner','--no-privileges','-d',url,follow['backup'])
        repo=Repository(url);repo.migrate();repo.migrate()
        recovered=counts(repo);assert recovered==follow['final_counts']
        rows=repo.conn.execute("SELECT id FROM fm.feature_snapshots WHERE feature_version='prematch-season-v2.0' ORDER BY prediction_cutoff").fetchall()
        store=FeatureStore(repo,DATA/'features')
        replays=[store.replay_snapshot(str(rows[i]['id'])) for i in (0,len(rows)//2,-1)]
        registered=repo.conn.execute('SELECT metadata FROM fm.raw_observations').fetchall()
        from football_spider.models.contracts import Observation
        for r in registered:RawArchive(DATA).read_body(Observation.model_validate(r['metadata']))
        dbstatus=repo.status();repo.close()
    suites=list(ET.parse(OUT/'platform-tests.xml').getroot().iter('testsuite'))
    tests={field:sum(int(s.get(field,0)) for s in suites) for field in ('tests','failures','errors','skipped')}
    assert tests['tests'] and not tests['failures'] and not tests['errors']
    operations=[base['refresh'],*base['result_refresh']['reports'],follow['domain_refresh'],follow['valid_domain_refresh']]
    obs={o['id']:o for op in operations for o in op['observations']}
    refs=follow['live_fixture_selection']['collection']['observations']
    request_ids=sorted(set(obs)|set(refs))
    report={'status':'PARTIAL','platform_verification':'PASSED','generated_at':datetime.now(timezone.utc).isoformat(),
        'environment':'isolated_validation','production_configuration':PlatformSettings().public()['configured'],
        'sources':{'spftll':{'implementation':'IMPLEMENTED','live':'OK','quality':'PARTIAL',
                            'bounded_refresh_observations':len(request_ids),'observation_ids':request_ids,
                            'prospective_final':follow['prospective_result_export']['manifest']['final_matches'],
                            'prospective_pending':follow['prospective_result_export']['manifest']['pending_matches']},
                   'understat':{'implementation':'IMPLEMENTED','live':'OK_REGISTERED_ORIGINAL_RESPONSES','league':'EPL','season':'2025/26',
                                'observations':len(base['observations']),'unique_matches':base['audit']['unique_ids'],'teams':base['audit']['teams'],
                                'history_field_checks':base['audit']['history_value_checks'],'statistics':recovered['team_statistics']},
                   'football_data':{'implementation':'IMPLEMENTED','live':'NEEDS_CREDENTIALS'},
                   'odds_api':{'implementation':'IMPLEMENTED','live':'NEEDS_CREDENTIALS'},
                   'api_football':{'implementation':'RESEARCH_ONLY','live':'NEEDS_CREDENTIALS'},
                   'clubelo':{'implementation':'RESEARCH_ONLY','live':'HTTP_502_NOT_ACCEPTED'},
                   'fbref':{'implementation':'RESEARCH_ONLY','live':'HTTP_403_NOT_ACCEPTED'}},
        'tests':tests,'history_export':base['exports']['historical_reconstruction'],
        'operational_control':base['exports']['operational'],
        'consumer_join':{k:base['baseline_consumer']['training_join'][k] for k in ('accepted_rows','excluded_rows','missing_values')},
        'confirmed_entities':{'teams':base['reviewed_identity_links']['confirmed_teams'],'matches':base['reviewed_identity_links']['confirmed_matches']},
        'prospective_result_export':follow['prospective_result_export'],
        'agent':{'status':follow['agent_valid_live_call']['status'],'request_id':follow['valid_domain_refresh']['request_id'],
                 'requests':follow['valid_domain_refresh']['requests_this_call'],
                 'observations':[o['id'] for o in follow['valid_domain_refresh']['observations']],
                 'evidence_count':len(follow['agent_valid_live_call']['data']['evidence']),'cost':None,'cost_status':'UNKNOWN',
                 'scope':'existing_consumer_adapter_actual_call; no_full_parent_child_prediction_claim'},
        'recovery':{'status':'OK','counts':recovered,'archive_hashes_checked':len(registered),'feature_replays':replays,
                    'backup':follow['backup'],'backup_sha256':hashlib.sha256(Path(follow['backup']).read_bytes()).hexdigest(),
                    'validation_database_state':'STOPPED_RETAINED','final_cluster':str(cluster)},
        'database_status_query':dbstatus,'scheduling':'NOT_REGISTERED',
        'open_dependencies':['production_DATABASE_URL','API_keys_optional_for_other_sources',
                             'baseline_settlement_available_vs_observed_time_contract','full_live_agent_decision_demo_owned_by_agent'],
        'no_new_http_on_followup_replay':follow['http_requests_this_execution']==0}
    atomic_write(OUT/'final-manifest.json',json_bytes(report))
    print(json_bytes({'status':report['status'],'platform_verification':'PASSED','tests':tests,'archive_hashes_checked':len(registered),'counts':recovered}).decode())

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--consumer-report-only',action='store_true')
    args=parser.parse_args()
    if args.consumer_report_only:append_consumer_regression()
    else:main()
