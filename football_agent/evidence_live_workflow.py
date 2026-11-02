"""Persistent official evidence jobs without numerical prior or training join.

The public workflow keeps ownership of locking/state. This branch reuses the
official raw gateway, registered collector, exact replayer and result gateway.
"""
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

from football_research.acquisition.state import parse_time
from football_research.data_adapters.official_fixture import prepare_official_fixture, verify_official_fixture
from football_research.data_adapters.evidence_collection import (
    prepare_evidence_plan, validate_evidence_plan, collect_official_evidence, verify_evidence_run,
)
from football_research.data_adapters.finland_result_capture import capture_finland_result, verify_capture_receipts
from football_research.data_adapters.verified_fixtures import verify_independent_settlement
from football_research.data_adapters.prospective_learning import _previous_inventory
from football_spider.storage.raw_archive import RawArchive

STATES = {'preparing_evidence','prepared_evidence','collecting_evidence','awaiting_result_no_prior',
          'settling_evidence','evidence_settled_prior_unavailable'}


def _workflow():
    from . import live_match_workflow
    return live_match_workflow


def is_official_query(query):
    return isinstance(query,str) and query.startswith(('veikkausliiga:2026:match:','palloliitto:match:'))


def handles(request,state):
    return (state['status'] in STATES or state.get('analysis_mode')=='evidence_only'
            or state['status']=='new' and is_official_query(request['query']))


def _pins(output, folders):
    w=_workflow()
    return {p.relative_to(output).as_posix():w._hash(p) for folder in folders
            for p in sorted(folder.rglob('*')) if p.is_file() and not p.name.endswith('.lock')}


def try_fallback(output,request,state,result):
    """Only a completed current named preparation may spend its remaining cap."""
    w=_workflow()
    if (not request['settings'].get('official_evidence_fallback',False)
            or not isinstance(w._prepare_query(request),dict)
            or result.get('status') in {'ready','interrupted'}):
        return False
    base=output/'preparation'
    intents=sorted((base/'requests').glob('*.intent.json'))
    receipts=sorted((base/'requests').glob('*.receipt.json'))
    if ({p.name.replace('.intent.json','') for p in intents}
            !={p.name.replace('.receipt.json','') for p in receipts}):
        return False
    actual=sum(len(RawArchive(base/name).observations()) for name in ('market-archive','identity-archive'))
    declared=result.get('request_intents')
    if (type(declared) is not int or declared!=len(intents) or actual!=len(receipts)
            or result.get('archived_http_responses')!=actual
            or request['settings']['preparation_http_budget']-declared<1):
        return False
    w._stage(output,state,'preparing_evidence',analysis_mode='evidence_only',prior_status='prior_unavailable',
        numeric_voi_status='prior_unavailable',
        market_preparation_result='preparation/prepare-result.json',market_preparation_requests=actual,
        numeric_voi_training_eligible=False,qualified_response_rows=0)
    return True


def _identity(output):
    w=_workflow()
    base=output/'official-preparation'
    binding=w._read(base/'identity.json')
    return verify_official_fixture(binding,archive_root=base/'identity-archive',as_of=w._now())


def _register_and_collect(output,request,state):
    w=_workflow()
    binding=_identity(output)
    if w._now()>=parse_time(binding['prediction_cutoff']):
        w._stage(output,state,'expired',reason='official_evidence_cutoff_elapsed_before_acquisition')
        return
    history=[output/ref['path'] for ref in request['previous_plans']]
    previous=[w._read(p) for p in history]
    learning_path=output/'learning-plan.json'
    event=binding['canonical']['source_event_id']
    if not learning_path.exists():
        registry=Path(request['settings']['task_root'])
        if not registry.is_absolute():registry=w.ROOT/registry
        registry.mkdir(parents=True,exist_ok=True)
        with FileLock(str(registry/'.registry.lock'),timeout=0):
            registered={r['source_event_id'] for r in _previous_inventory(previous,w._now())}
            if (registry/'registry.json').exists():
                for known in w._read(registry/'registry.json')['tasks']:
                    known=Path(known)
                    if known!=output and (known/'learning-plan.json').exists():
                        w._load_job(known)
                        registered.update(r['source_event_id'] for r in _previous_inventory(
                            [w._read(known/'learning-plan.json')],w._now()))
            if event in registered:
                w._stage(output,state,'duplicate_fixture',reason='physical_fixture_already_registered_no_additional_actor_budget',
                    match_id=binding['match_id'],source_event_id=event)
                return
            learning=prepare_evidence_plan(binding,previous_plans=previous,created_at=w._now())
            w._write(learning_path,learning)
    validate_evidence_plan(binding,w._read(learning_path),previous,as_of=w._now())
    folder=output/'analysis'/('match-'+binding['native_event_id'])
    config=request['settings']
    w._stage(output,state,'collecting_evidence',match_id=binding['match_id'],source_event_id=event,
        learning_plan_sha256=w._hash(learning_path),actor_intent_at=w._now().isoformat(),
        run_directory_pending=folder.relative_to(output).as_posix())
    try:
        collect_official_evidence(output/'official-preparation/identity.json',
            identity_archive=output/'official-preparation/identity-archive',learning_plan_path=learning_path,
            previous_plan_paths=history,output=folder,requests_per_match=config['actor_requests_per_match'],
            seed=config['seed'],route=config['route'],news_route_overrides=config.get('news_route_overrides'),
            news_projection_version=config.get('news_projection_version'))
    except Exception as error:
        w._stage(output,state,'reconciliation_required',reason='evidence_acquisition_incomplete_never_automatically_repeat',
            failed_stage='collecting_evidence',error_type=type(error).__name__,error_message=str(error))
        return
    adopt_collection(output,request,state)


def adopt_collection(output,request,state):
    w=_workflow()
    try:
        folder=output/state['run_directory_pending']
        run,config=w._read(folder/'run.json'),w._read(folder/'configuration.json')
        binding=verify_evidence_run(folder,config)
        context=config['supervisor_metadata']['evidence_collection']
        settings=request['settings']
        expected={'requests_per_match':settings['actor_requests_per_match'],'seed':settings['seed'],
                  'route':settings['route'],'news_route_overrides':settings.get('news_route_overrides')}
        if 'news_projection_version' in settings:
            expected['news_projection_version']=settings['news_projection_version']
        if (context['learning_plan']!=w._read(output/'learning-plan.json')
                or context['execution_claim']!=w._read(output/'learning-plan.json.acquisition.json')
                or context['previous_plans']!=[w._read(output/ref['path']) for ref in request['previous_plans']]
                or context['settings']!=expected or binding!=_identity(output)
                or run['configuration_digest']!=w.digest(config)
                or run.get('initial_probabilities') is not None or run.get('final_probabilities') is not None
                or run['measured_resources']['requests']>settings['actor_requests_per_match']
                or 'reconciliation' in str(run.get('terminal_selection',{}).get('reason',''))):
            raise ValueError('Evidence actor differs from its fixed identity, registration or budget')
        if not (folder/'report.md').is_file() or not (folder/'analysis.json').is_file():
            raise ValueError('Evidence actor report is incomplete')
        from .learned_voi_replay import replay_saved_run
        with TemporaryDirectory(prefix='evidence-verify-',dir=output) as temporary:
            verification=replay_saved_run(folder,output=Path(temporary)/'replay',project_root=w.ROOT)
        due=parse_time(binding['canonical']['kickoff'])+timedelta(minutes=settings['settlement_delay_minutes'])
        w._stage(output,state,'awaiting_result_no_prior',actor_requests=run['measured_resources']['requests'],
            run_directory=folder.relative_to(output).as_posix(),result_due_at=due.isoformat(),
            actor_sha256={**_pins(output,[folder]),'learning-plan.json.acquisition.json':
                w._hash(output/'learning-plan.json.acquisition.json')},
            numeric_voi_training_eligible=False,qualified_response_rows=0,
            initial_probabilities=None,final_probabilities=None,independent_labels=0,
            actor_verification={'identical_replay':verification['identical_replay'],
                'raw_responses_reprojected':verification['raw_responses_reprojected'],'network_requests':0},
            report=(folder/'report.md').relative_to(output).as_posix())
    except (ValueError,OSError,KeyError,TypeError) as error:
        w._stage(output,state,'reconciliation_required',reason='evidence_actor_archive_verification_failed_no_repeat',
            failed_stage='collecting_evidence',error_type=type(error).__name__,error_message=str(error))


def _settle(output,request,state):
    w=_workflow()
    if w._now()<parse_time(state['result_due_at']):return
    attempt=state['settlement_attempts']+1
    if attempt>request['settings']['settlement_max_attempts']:
        w._stage(output,state,'result_unconfirmed',reason='bounded_result_attempts_exhausted')
        return
    binding=_identity(output)
    destination=output/'results'/f'attempt-{attempt:02}'
    w._stage(output,state,'settling_evidence',settlement_attempts=attempt,
        settlement_path=destination.relative_to(output).as_posix(),settlement_intent_at=w._now().isoformat())
    try:
        capture_finland_result(binding,identity_archive_root=output/'official-preparation/identity-archive',
                              output=destination,clock=w._now)
    except Exception as error:
        w._stage(output,state,'reconciliation_required',reason='evidence_result_acquisition_incomplete_no_automatic_retry',
            failed_stage='settling_evidence',error_type=type(error).__name__,error_message=str(error))
        return
    adopt_settlement(output,request,state)


def adopt_settlement(output,request,state):
    w=_workflow()
    try:
        destination=output/state['settlement_path']
        binding=_identity(output)
        receipts=verify_capture_receipts(binding,destination,as_of=w._now())
        if not receipts['receipts']:
            raise ValueError('Due evidence settlement requires an actual result request')
        archive=RawArchive(destination)
        verified=verify_independent_settlement(binding,identity_archive_root=output/'official-preparation/identity-archive',
            result_archive_root=destination,observation_ids=[o.id for o in archive.observations()],as_of=w._now())
        qualification={'schema':'evidence-only-independent-settlement-v1','independent_result_verification':verified,
            'numeric_training_eligible':False,'qualified_response_rows':0,'training_records':[],
            'reason':'predictive_prior_unavailable_independent_result_is_not_a_numeric_training_row'}
        path=destination/'evidence-qualification.json'
        if path.exists():
            # A lost completion marker may be adopted later. Preserve the
            # original qualification clock and independently recheck its raw.
            previous=w._read(path)
            original=previous['independent_result_verification']
            checked=verify_independent_settlement(binding,identity_archive_root=output/'official-preparation/identity-archive',
                result_archive_root=destination,observation_ids=[o.id for o in archive.observations()],as_of=original['verified_at'])
            if checked!=original or any(previous.get(k)!=v for k,v in qualification.items() if k!='independent_result_verification'):
                raise ValueError('Frozen independent evidence qualification changed')
            qualification=previous
        else:w._write(path,qualification)
        count=sum(w._read(p)['http_requests_actual'] for p in (output/'results').glob('*/capture-receipts.json'))
        common={'result_requests':count,'qualification_path':path.relative_to(output).as_posix(),
            'qualified_response_rows':0,'numeric_voi_training_eligible':False,
            'independent_labels':len(verified['settlement_rows']),
            'result_sha256':{**state.get('result_sha256',{}),**_pins(output,[destination])}}
        if verified['status']=='VERIFIED':
            w._stage(output,state,'evidence_settled_prior_unavailable',**common)
        elif state['settlement_attempts']>=request['settings']['settlement_max_attempts']:
            w._stage(output,state,'result_unconfirmed',reason='bounded_result_attempts_exhausted',**common)
        else:
            w._stage(output,state,'awaiting_result_no_prior',reason='independent_result_not_final_or_unavailable',
                result_due_at=(w._now()+timedelta(minutes=request['settings']['settlement_retry_minutes'])).isoformat(),**common)
    except (ValueError,OSError,KeyError,TypeError) as error:
        w._stage(output,state,'reconciliation_required',reason='evidence_result_archive_verification_failed_no_repeat',
            failed_stage='settling_evidence',error_type=type(error).__name__,error_message=str(error))


def advance(output,request,state,*,settle=False):
    w=_workflow()
    if state['status']=='reconciliation_required':
        stage=state.get('failed_stage')
        if stage=='collecting_evidence':adopt_collection(output,request,state)
        elif stage=='settling_evidence':adopt_settlement(output,request,state)
        elif stage=='preparing_evidence':w._stage(output,state,'preparing_evidence')
        return w._summary(output,state) if stage!='preparing_evidence' else advance(output,request,state,settle=settle)
    if state['status'] in {'new','preparing_evidence'}:
        w._stage(output,state,'preparing_evidence',analysis_mode='evidence_only',prior_status='prior_unavailable',
            numeric_voi_status='prior_unavailable',numeric_voi_training_eligible=False,qualified_response_rows=0)
        base=output/'official-preparation'
        scope = ({'schedule_group': request['settings']['official_schedule_group']}
                 if 'official_schedule_group' in request['settings'] else {})
        result=prepare_official_fixture(w._prepare_query(request),output=base,max_http_requests=1,
                                       resume=base.exists(),clock=w._now,
                                       horizon_days=request['settings'].get('official_horizon_days',7), **scope)
        if result['status']!='ready':
            status='reconciliation_required' if result['status']=='interrupted' else result['status']
            w._stage(output,state,status,reason=result.get('reason',result['status']),failed_stage='preparing_evidence',
                preparation_result='official-preparation/prepare-result.json')
            return w._summary(output,state)
        binding=_identity(output)
        w._stage(output,state,'prepared_evidence',match_id=binding['match_id'],source_event_id=binding['canonical']['source_event_id'],
            preparation_requests=state.get('market_preparation_requests',0)+result['archived_http_responses'],
            prepared_sha256=_pins(output,[base,output/'preparation']))
    if state['status']=='prepared_evidence':_register_and_collect(output,request,state)
    elif state['status']=='collecting_evidence':adopt_collection(output,request,state)
    elif state['status']=='settling_evidence':adopt_settlement(output,request,state)
    if settle and state['status']=='awaiting_result_no_prior':_settle(output,request,state)
    return w._summary(output,state)
