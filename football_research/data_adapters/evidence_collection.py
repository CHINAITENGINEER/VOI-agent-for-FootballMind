"""Acquire actual visible evidence independently of numerical initialization.

The existing runtime and tools own actions and receipts. No f/g target, quote,
uniform prior or numerical prediction is constructed by this entry point.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
from uuid import uuid4
from filelock import FileLock

from football_research.acquisition.state import parse_time
from football_agent.learned_voi_runtime import digest
from football_spider.storage.raw_archive import digest as identity_digest
from .prospective_learning import (
    EVIDENCE_PLAN_VERSION, FACTS, SCOPE, _check_plan, _previous_inventory, _own_inventory, _merge_inventory)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare_evidence_plan(binding, *, previous_plans, created_at):
    at = parse_time(created_at)
    if (binding.get('schema') != 'official-fixture-evidence-identity-v1' or binding.get('status') != 'CONFIRMED'
            or binding.get('binding_sha256') != identity_digest({k:v for k,v in binding.items() if k!='binding_sha256'})
            or not parse_time(binding['verified_at']) <= at < parse_time(binding['prediction_cutoff'])):
        raise ValueError('Evidence acquisition needs a current verified official identity')
    prior = _previous_inventory(previous_plans, at)
    canonical = binding['canonical']
    if canonical['source_event_id'] in {row['source_event_id'] for row in prior}:
        raise ValueError('Physical fixture already registered; evidence collection cannot repeat its budget')
    plan = {'schema':EVIDENCE_PLAN_VERSION, 'purpose':'f_train', 'created_at':at.isoformat(),
        'matches':[{'match_id':binding['match_id'],'source_event_id':canonical['source_event_id'],
            'fixture_facts':{key:canonical[key] for key in FACTS}, 'kickoff_at':canonical['kickoff'],
            'prediction_cutoff':binding['prediction_cutoff'],'identity_binding_sha256':binding['binding_sha256']}],
        'artifacts':{},'prior_inventory':prior,'inventory_scope':SCOPE,'prior_status':'prior_unavailable',
        'numeric_training_eligible':False,'scope':'prospective_A_exposure_only_pending_a_valid_predictor_input_contract'}
    _merge_inventory(prior+_own_inventory(plan))
    plan['cohort_id']=EVIDENCE_PLAN_VERSION+':'+digest(plan)[:24]
    return plan


def validate_evidence_plan(binding, plan, previous_plans, *, as_of):
    _check_plan(plan)
    if (plan['schema']!=EVIDENCE_PLAN_VERSION or parse_time(plan['created_at'])>parse_time(as_of)
            or plan!=prepare_evidence_plan(binding,previous_plans=previous_plans,created_at=plan['created_at'])):
        raise ValueError('Evidence exposure differs from its pre-query registration')
    return plan


def _components(binding, settings, archive):
    from football_agent.evidence_candidates import EvidenceDocumentCandidates
    from football_agent.evidence_only_policy import EvidenceOnlyPolicy
    from football_agent.match_source_profiles import resolve_verified_profile
    from football_agent.news_discovery import NewsDiscoveryTools
    from football_spider.storage.raw_archive import RawArchive
    keys={'requests_per_match','seed','route','news_route_overrides'}
    projection=settings.get('news_projection_version',5)
    if (set(settings) not in (keys,keys|{'news_projection_version'})
            or type(projection) is not int or projection not in {5,6}
            or type(settings['requests_per_match']) is not int or not 1<=settings['requests_per_match']<=4):
        raise ValueError('Evidence acquisition requires explicit bounded settings')
    policy=EvidenceOnlyPolicy(binding)
    profile=resolve_verified_profile(binding,version=5)
    generator=EvidenceDocumentCandidates(match_terms=[binding['canonical'][s+'_name'] for s in ('home','away')],
        seed_urls=profile['seed_urls'],allowed_hosts=profile['allowed_hosts'],numeric_actions=[],
        web_cost={'money':.01,'requests':1.},match=policy.evidence_match,profile=profile,discovery_version=11)
    tools=NewsDiscoveryTools(RawArchive(archive),profile['allowed_hosts'],provider='bing_rss',
        default_route=settings['route'],route_overrides=settings['news_route_overrides'],extraction_version=3,
        index_urls=[e['url'] for e in profile['entries'] if e['kind'] in {'news_index','club_home'}],projection_version=projection)
    return policy,profile,generator,tools


def verify_evidence_run(folder, config):
    from .official_fixture import verify_official_fixture
    from football_agent.inquiry_exploration import UniformInquiryExploration
    folder=Path(folder)
    metadata=config['supervisor_metadata']
    if config['schema']!='learned-voi-open-execution-v5':
        raise ValueError('Evidence-only acquisition requires the explicit v5 runtime')
    context=metadata['evidence_collection']
    binding=metadata['verified_fixture_identity']
    verify_official_fixture(binding,archive_root=folder/'identity-archive',as_of=config['created_at'])
    validate_evidence_plan(binding,context['learning_plan'],context['previous_plans'],as_of=config['created_at'])
    policy,_,generator,tools=_components(binding,context['settings'],folder/'source-archive')
    if (config['policy_manifest']!=policy.manifest or config['initial_state']!=policy.initial_state
            or config['evidence_projection']!={'match':policy.evidence_match,'version':'match-evidence-extraction-v5'}
            or metadata['generator']!=generator.configuration() or metadata['transport']!=tools.configuration()
            or any(k in metadata for k in ('learning_cohort','context_calibration','context_confirmation'))):
        raise ValueError('Evidence-only identity, policy or capabilities differ')
    settings=context['settings']
    claim=context['execution_claim']
    if (set(claim)!={'schema','cohort_id','learning_plan_sha256','source_event_id','run_id','output','claimed_at','settings'}
            or claim['schema']!='official-evidence-execution-claim-v1'
            or claim['cohort_id']!=context['learning_plan']['cohort_id']
            or claim['learning_plan_sha256']!=digest(context['learning_plan'])
            or claim['source_event_id']!=binding['match_id'] or claim['run_id']!=config['run_id']
            or not Path(claim['output']).is_absolute() or claim['settings']!=settings
            or not parse_time(context['learning_plan']['created_at'])<=parse_time(claim['claimed_at'])<=parse_time(config['created_at'])):
        raise ValueError('Evidence execution differs from its single pre-query claim')
    caps={'money':.01*settings['requests_per_match'],'requests':settings['requests_per_match']}
    if (config['limits']!=caps or config['exploration_limits']!=caps or config['max_steps']!=16
            or config['max_exploration_actions']!=settings['requests_per_match']
            or config['inquiry_exploration']!=UniformInquiryExploration(settings['seed']).configuration()
            or parse_time(config['created_at'])>=parse_time(binding['prediction_cutoff'])):
        raise ValueError('Evidence-only budget, randomization or cutoff differs')
    return binding


def collect_official_evidence(identity_path, *, identity_archive, learning_plan_path,
                              previous_plan_paths, output, requests_per_match=4, seed=198,
                              route='direct', news_route_overrides=None, news_projection_version=None):
    from .official_fixture import verify_official_fixture
    from football_agent.learned_voi_runtime import LearnedVOIRuntime
    from football_agent.learned_voi_web import ArchivedAndWebExecutor
    from football_agent.match_analysis_report import write_analysis_report
    output=Path(output)
    if output.exists():
        raise FileExistsError('Completed evidence acquisitions are replayed, never repeated')
    binding,plan=_read(identity_path),_read(learning_plan_path)
    previous=[_read(p) for p in previous_plan_paths]
    now=datetime.now(timezone.utc)
    verify_official_fixture(binding,archive_root=identity_archive,as_of=now)
    validate_evidence_plan(binding,plan,previous,as_of=now)
    if now>=parse_time(binding['prediction_cutoff']):
        raise ValueError('Evidence acquisition cutoff already elapsed')
    settings={'requests_per_match':requests_per_match,'seed':seed,'route':route,
        'news_route_overrides':deepcopy(news_route_overrides)}
    if news_projection_version is not None:
        settings['news_projection_version']=news_projection_version
    policy,profile,generator,tools=_components(binding,settings,output/'source-archive')
    row={'match_id':binding['match_id'],'prediction_cutoff':binding['prediction_cutoff'],
        'features_available_at':binding['verified_at'],'information_mode':'operational',
        **policy.initial_state['base']}
    executor=ArchivedAndWebExecutor(row,[],tools,seed_urls=profile['seed_urls'])
    run_id='official-evidence-'+uuid4().hex
    executor.run_id=run_id
    # An original registration owns one execution, including interrupted I/O.
    # Moving only its output path cannot allocate another exploration budget.
    # Global physical-fixture admission remains under the live registry lock.
    claim_path=Path(learning_plan_path).with_name(Path(learning_plan_path).name+'.acquisition.json')
    claim={'schema':'official-evidence-execution-claim-v1','cohort_id':plan['cohort_id'],
        'learning_plan_sha256':digest(plan),'source_event_id':binding['match_id'],
        'run_id':run_id,'output':str(output.resolve()),'claimed_at':now.isoformat(),'settings':settings}
    with FileLock(str(claim_path)+'.lock',timeout=0):
        with claim_path.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(claim,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    metadata={'verified_fixture_identity':binding,'generator':generator.configuration(),'transport':tools.configuration(),
        'evidence_collection':{'learning_plan':plan,'previous_plans':previous,'settings':settings,'execution_claim':claim},
        'initialization':{'status':'prior_unavailable','probabilities':None,'numeric_voi':None}}
    def copy_identity():
        if not (output/'identity-archive').exists():
            shutil.copytree(identity_archive,output/'identity-archive')
        verify_official_fixture(binding,archive_root=output/'identity-archive',as_of=now)
    def execute(action,public):
        # Runtime has already persisted config/reservation. Save independent
        # identity raw bytes before the first external evidence request.
        copy_identity()
        return executor(action,public)
    caps={'money':.01*requests_per_match,'requests':requests_per_match}
    result=LearnedVOIRuntime(policy,seed=seed,exploration_limits=caps,max_exploration_actions=requests_per_match).run(
        policy.initial_state,run_id=run_id,limits=caps,propose=generator,execute=execute,output=output,
        provider_version=executor.version+':'+digest(tools.configuration()),
        generator_version=generator.version+':'+digest(generator.configuration()),run_metadata=metadata,
        evidence_match=policy.evidence_match,evidence_version='match-evidence-extraction-v5')
    copy_identity()
    verify_evidence_run(output,_read(output/'configuration.json'))
    write_analysis_report(result,match=policy.evidence_match,output=output)
    return {'status':'collected_prior_unavailable','run_directory':str(output.resolve()),'run_id':run_id,
        'match_id':binding['match_id'],'source_event_id':binding['canonical']['source_event_id'],
        'http_requests':result['measured_resources']['requests'],'documents':len(result['documents']),
        'initial_probabilities':None,'final_probabilities':None,'numeric_voi_training_eligible':False,
        'terminal_selection':result['terminal_selection'],'independent_labels':0}


__all__=['prepare_evidence_plan','validate_evidence_plan','collect_official_evidence','verify_evidence_run']
