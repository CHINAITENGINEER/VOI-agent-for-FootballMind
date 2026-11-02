"""Original v5 evidence prefixes for predictors with an explicit missing-prior contract.

No archive is reprojected with a later extractor. Independent outcomes, when
available, remain supervisor-only; the original actor forecast stays null.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from football_agent.learned_voi_replay import replay_saved_run
from football_research.acquisition.state import parse_time
from .context_training import _bindings, _immediate_context_prefixes, _read, _write
from .evidence_collection import verify_evidence_run
from .open_text_episodes import _digest, document_role
from .prospective_learning import _previous_inventory
from .verified_fixtures import verify_independent_settlement
from football_spider.storage.raw_archive import RawArchive

VERSION = 'raw-backed-prior-unavailable-context-v1'


def _episodes(folder, run):
    """Describe actual, already raw-replayed returns without a numeric prior."""
    episodes = []
    for cycle in run['cycles']:
        if cycle['selected_action'] is None:
            continue
        receipt = _read(folder/f"receipt-{cycle['step']:03}.json")
        document = receipt['document']
        exploration = cycle['exploration_record']
        cutoff = parse_time(cycle['public_before']['state']['prediction_cutoff'])
        late = parse_time(cycle['returned_at']) > cutoff
        episodes.append({'step':cycle['step'], 'action':deepcopy(cycle['selected_action']),
            'decision_at':cycle['public_before']['decision_at'], 'observed_at':document['observed_at'],
            'available_at':document['available_at'],
            'response':{'status':receipt['status'],'kind':document_role(document),
                'document_id':document['document_id'],'raw_sha256':document['raw_sha256']},
            'selection_probability':exploration['selected_p'] if exploration else None,
            'selection_scope':'conditional_on_generated_menu_not_open_world_propensity',
            'selection_basis':cycle['selection_basis'],
            'training_exclusion_reasons':['response_not_operationally_available_before_cutoff'] if late else [],
            'response_observation_eligible':not late})
    return episodes


def build_evidence_context_from_runs(run_folders, previous_plans, output, *, result_archives=None, as_of=None):
    """Convert authenticated original v5 states; keep missing labels pending.

    result_archives explicitly maps canonical physical IDs to independent raw
    result directories. This function never acquires results or accepts labels
    from callers. It does not authorize training the old market-offset model.
    """
    started = datetime.now(timezone.utc)
    at = parse_time(as_of or started)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Evidence context output exists; preserve original corpus versions')
    if not isinstance(run_folders,(list,tuple)) or not run_folders:
        raise ValueError('Explicit original evidence run folders are required')
    results = {} if result_archives is None else result_archives
    if not isinstance(results,dict):
        raise ValueError('Independent result archives must be keyed by physical match')
    inventory = _previous_inventory(previous_plans,at)
    registered = {row['source_event_id']:row for row in inventory}
    supplied = {p['cohort_id'] for p in previous_plans if p.get('cohort_id')}
    folders = sorted({Path(p).resolve() for p in run_folders})
    bindings = _bindings(folders+list(results.values()))
    prefixes, transitions, checked, seen, labels_audit = [], [], [], set(), []
    for folder in folders:
        config, run = _read(folder/'configuration.json'), _read(folder/'run.json')
        if (config['schema']!='learned-voi-open-execution-v5'
                or run['configuration_digest']!=_digest(config)
                or not parse_time(config['created_at'])<=parse_time(run['finished_at'])<=at):
            raise ValueError('A completed original v5 run before the audit time is required')
        binding = verify_evidence_run(folder,config)
        physical = binding['canonical']['source_event_id']
        plan = config['supervisor_metadata']['evidence_collection']['learning_plan']
        entry = registered.get(physical)
        if (plan['cohort_id'] not in supplied or not entry or entry['purpose']!='f_train'
                or entry['model_bindings'] or binding['match_id'] not in entry['match_ids']):
            raise ValueError('Original evidence exposure must remain explicitly registered in A')
        if run['run_id'] in seen:
            raise ValueError('Repeated evidence run identity cannot duplicate its prefix history')
        seen.add(run['run_id'])
        if physical not in results:
            check = {'source_event_id':physical,'status':'PENDING','settlement_rows':[],
                'reason':'independent_result_archive_not_supplied','verified_at':at.isoformat()}
        else:
            check = verify_independent_settlement(binding,identity_archive_root=folder/'identity-archive',
                result_archive_root=results[physical],
                observation_ids=[o.id for o in RawArchive(Path(results[physical])).observations()],as_of=at)
            if check['status']!='VERIFIED' and check['settlement_rows']:
                raise ValueError('Unverified independent results cannot attach supervised outcomes')
        labels_audit.append(check)
        with TemporaryDirectory(prefix='evidence-context-replay-') as temporary:
            replay = replay_saved_run(folder,output=Path(temporary)/'replay',
                project_root=Path(__file__).resolve().parents[2])
        checked.append({key:replay[key] for key in
            ('run_id','identical_replay','raw_responses_reprojected','replay_network_requests')})
        identity = {'original_match_id':binding['match_id'],'source_event_id':physical}
        labels = {binding['match_id']:check['settlement_rows']}
        rows,pairs = _immediate_context_prefixes(run,_episodes(folder,run),identity,labels,at)
        for row in rows:
            if (row['visible_state']['before']['prior_probabilities'] is not None
                    or row['visible_state']['before']['current_probabilities'] is not None):
                raise ValueError('Original unavailable forecast cannot acquire a retrospective probability')
            row.update(schema=VERSION,purpose='f_train',predictive_input_contract='optional_market_prior_required',
                original_actor_prior_status='prior_unavailable',original_numeric_training_eligible=False)
        prefixes.extend(rows)
        transitions.extend({**pair,'schema':VERSION,'original_gain_target':None,
            'predictive_input_contract':'optional_market_prior_required'} for pair in pairs)
    if set(results)-{r['physical_match_id'] for r in prefixes}:
        raise ValueError('Result archive mapping contains a fixture outside this corpus')
    if bindings != _bindings(folders+list(results.values())):
        raise ValueError('Original evidence/result inputs changed during conversion')
    labelled = [r for r in prefixes if 'outcome' in r]
    pending = [r for r in prefixes if 'outcome' not in r]
    report = {'schema':VERSION,'status':'converted','created_at':started.isoformat(),'as_of':at.isoformat(),
        'input_runs':len(folders),'physical_matches':len({r['physical_match_id'] for r in prefixes}),
        'prefixes':len(prefixes),'supervised_prefixes':len(labelled),'pending_prefixes':len(pending),
        'actual_transitions':len(transitions),'response_statuses':dict(Counter(t['response_status'] for t in transitions)),
        'independent_result_verification':labels_audit,'replay_acceptance':checked,
        'input_bindings':bindings,'registration_inputs_sha256':_digest(previous_plans),
        'records_sha256':_digest(labelled),'pending_sha256':_digest(pending),'transitions_sha256':_digest(transitions),
        'network_requests':0,'model_training_performed':False,'model':None,
        'original_actor_forecasts_changed':False,'later_body_projections_used':False,
        'numeric_gain_supervision_created':False,'legacy_market_model_training_eligible':False,
        'split_requirement':'all_prefixes_aliases_runs_of_one_physical_match_share_one_time_partition'}
    output.mkdir(parents=True,exist_ok=False)
    for name,value in (('prefix-records.json',labelled),('pending-prefixes.json',pending),
                       ('actual-transitions.json',transitions),('registration-inputs.json',previous_plans),
                       ('conversion-report.json',report)):
        _write(output/name,value)
    return report


__all__ = ['VERSION','build_evidence_context_from_runs']
