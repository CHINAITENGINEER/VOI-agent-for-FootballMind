"""Research-only optional-prior f fitting through both original raw gateways.

No caller-supplied labels, records, later projections, or synthetic masked
views enter this adapter. Pending fixtures remain unlabelled A exposures.
"""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import shutil
from tempfile import TemporaryDirectory

from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor, training_readiness
from football_research.acquisition.state import parse_time
from .context_training import build_context_training_from_runs, _bindings, _read, _write
from .evidence_context import build_evidence_context_from_runs
from .open_text_episodes import _digest
from .prospective_learning import _previous_inventory


VERSION = 'raw-backed-optional-prior-context-training-v1'
_OLD = {f'learned-voi-open-execution-v{i}' for i in range(1, 5)}
_NEW = 'learned-voi-open-execution-v5'


def _real_origin(value):
    if isinstance(value, dict):
        if 'evidence_origin' in value and value['evidence_origin'] != 'real_observational':
            raise ValueError('Synthetic or unverified evidence origin cannot enter real optional-prior training')
        for child in value.values():
            _real_origin(child)
    elif isinstance(value, list):
        for child in value:
            _real_origin(child)


def _domain_counts(rows):
    result = {}
    for domain in ('provided_prior', 'missing_prior'):
        members = [r for r in rows if ('missing_prior' if r['visible_state']['before']['prior_probabilities'] is None
                                      else 'provided_prior') == domain]
        physical = sorted({r['physical_match_id'] for r in members})
        result[domain] = {'prefixes':len(members), 'physical_matches':len(physical), 'physical_match_ids':physical}
    return result


def _build_optional_context_from_runs(run_folders, previous_plans, result_archive, output, *, as_of=None, fit_model):
    """Rebuild both registered A domains and fit only after the real fit gate.

    A shared result directory is mapped to each v5 physical fixture. Independent
    result verification decides which observations actually concern that event.
    This function never fetches data or changes any existing prediction entry.
    """
    started = datetime.now(timezone.utc)
    at = parse_time(started if as_of is None else as_of)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Optional-prior training output exists; preserve earlier research artifacts')
    if at > started:
        raise ValueError('Training audit time cannot follow the actual execution clock')
    if not isinstance(run_folders, (list, tuple)) or not run_folders or not result_archive:
        raise ValueError('Explicit original run folders, registered plans and a raw result archive are required')
    if not isinstance(previous_plans, (list, tuple)) or not previous_plans:
        raise ValueError('Explicit registered A exposure plans are required')
    _real_origin(previous_plans)
    inventory = _previous_inventory(previous_plans, at)
    registered = {row['source_event_id']:row for row in inventory}
    folders = sorted({Path(p).resolve() for p in run_folders})
    result_archive = Path(result_archive).resolve()
    if any(output.resolve().is_relative_to(source) for source in folders+[result_archive]):
        raise ValueError('Training output must not modify an original run or result archive')
    old, new, configs, seen, events = [], [], {}, set(), {}
    for folder in folders:
        config, run = _read(folder/'configuration.json'), _read(folder/'run.json')
        if config.get('schema') not in _OLD | {_NEW}:
            raise ValueError('Unsupported original run schema for optional-prior training')
        if (run.get('schema') != config['schema'] or run.get('configuration_digest') != _digest(config)
                or run.get('run_id') != config.get('run_id')):
            raise ValueError('Original run/configuration binding is inconsistent')
        if run['run_id'] in seen:
            raise ValueError('Repeated run identity cannot duplicate training exposure')
        seen.add(run['run_id'])
        _real_origin(config)
        if 'synthetic' in str(config.get('provider_version', '')).casefold():
            raise ValueError('Synthetic archives cannot enter real optional-prior training')
        metadata = config.get('supervisor_metadata', {})
        if any(name in metadata for name in ('context_calibration', 'context_confirmation')):
            raise ValueError('Only explicitly registered A runs may train f; B/C cannot be converted')
        cohort = metadata.get('learning_cohort')
        if cohort is not None and cohort.get('purpose') != 'f_train':
            raise ValueError('Only explicitly registered A runs may train f')
        identity = metadata.get('verified_fixture_identity')
        if not isinstance(identity, dict):
            raise ValueError('Original raw-verified physical fixture identity is required')
        physical = identity['canonical']['source_event_id']
        entry = registered.get(physical)
        if not entry or entry['purpose'] != 'f_train' or identity['match_id'] not in entry['match_ids']:
            raise ValueError('Every original run must remain in the explicitly registered physical A inventory')
        events[run['run_id']] = physical
        configs[run['run_id']] = config
        (new if config['schema'] == _NEW else old).append(folder)
    source_bindings = _bindings(folders+[result_archive])
    # Both gateways finish, and the merged data validate, before publication.
    # An error never leaves a partial output looking like a completed corpus.
    with TemporaryDirectory(prefix='optional-context-training-') as temporary:
        staging = Path(temporary)
        conversions = {}
        if old:
            conversions['provided_prior_gateway'] = build_context_training_from_runs(
                old, previous_plans, result_archive, staging/'market-context', as_of=at)
        if new:
            result_map = {events[_read(folder/'run.json')['run_id']]:result_archive for folder in new}
            conversions['missing_prior_gateway'] = build_evidence_context_from_runs(
                new, previous_plans, staging/'evidence-context', result_archives=result_map, as_of=at)
        labelled, pending, transitions = [], [], []
        for name in ('market-context', 'evidence-context'):
            folder = staging/name
            if folder.exists():
                labelled.extend(_read(folder/'prefix-records.json'))
                pending.extend(_read(folder/'pending-prefixes.json'))
                transitions.extend(_read(folder/'actual-transitions.json'))
        ids, groups, labels = set(), defaultdict(list), {}
        for row in labelled+pending:
            physical = row['physical_match_id']
            if (row['prefix_id'] in ids or row['run_id'] not in events or events[row['run_id']] != physical
                    or registered[physical]['purpose'] != 'f_train'
                    or row.get('purpose', 'f_train') != 'f_train'):
                raise ValueError('Merged prefix identity, uniqueness or registered A purpose differs')
            ids.add(row['prefix_id'])
            groups[physical].append(row)
            if 'outcome' in row:
                fact = (row['outcome'], row['kickoff_at'])
                if physical in labels and labels[physical] != fact:
                    raise ValueError('Physical fixture has conflicting raw-verified labels or kickoff')
                labels[physical] = fact
        if any('outcome' in row or 'result_available_at' in row for row in pending):
            raise ValueError('Pending prefixes cannot acquire training labels')
        readiness = training_readiness(labelled, as_of=at, evidence_origin='real_observational')
        data_version = VERSION+':'+_digest({'records':labelled, 'pending':pending,
            'registrations':previous_plans, 'source_bindings':source_bindings})[:24]
        model, model_info = None, None
        if fit_model and readiness['ready_to_fit']:
            model = OptionalPriorContextPredictor().fit(labelled, as_of=at,
                dataset_version=data_version, evidence_origin='real_observational')
            model.save(staging/'f.json')
            restored = OptionalPriorContextPredictor.load(staging/'f.json')
            if restored.model_version != model.model_version:
                raise ValueError('Saved optional-prior model did not reconstruct exactly')
            model_info = {'file':'f.json', 'sha256':sha256((staging/'f.json').read_bytes()).hexdigest(),
                          'model_version':model.model_version,
                          'fit_execution_at':model.metadata['fit_execution_at'], 'production_ready':False}
        physical_groups = [{'physical_match_id':physical,
            'provider_match_ids':sorted({row['match_id'] for row in rows}),
            'run_ids':sorted({row['run_id'] for row in rows}),
            'prefix_ids':[row['prefix_id'] for row in rows], 'prefixes':len(rows),
            'supervised_prefixes':sum('outcome' in row for row in rows),
            'pending_prefixes':sum('outcome' not in row for row in rows),
            'outcome':labels[physical][0] if physical in labels else None,
            'split':'registered_A', 'split_unit':'complete_physical_match'}
            for physical, rows in sorted(groups.items())]
        splits = {name:sorted(row['source_event_id'] for row in inventory if row['purpose'] == purpose)
                  for name, purpose in (('A', 'f_train'), ('B', 'calibration'), ('C', 'confirmation'))}
        projection_counts = defaultdict(int)
        for config in configs.values():
            projection_counts[config['supervisor_metadata'].get('transport', {}).get(
                'news_discovery', {}).get('version', 'unstructured_original_projection')] += 1
        report = {'schema':VERSION, 'status':'fitted_research_only' if model else 'not_ready',
            'created_at':started.isoformat(), 'as_of':at.isoformat(), 'dataset_version':data_version,
            'evidence_origin':'real_observational', 'input_runs':len(folders),
            'runtime_versions':{version:sum(c['schema'] == version for c in configs.values())
                                for version in sorted({c['schema'] for c in configs.values()})},
            'physical_matches':len(groups), 'supervised_prefixes':len(labelled), 'pending_prefixes':len(pending),
            'supervised_physical_matches':len(labels),
            'pending_physical_matches':len({r['physical_match_id'] for r in pending}),
            'actual_transitions':len(transitions), 'readiness':readiness,
            'domain_counts':{'all':_domain_counts(labelled+pending), 'labelled':_domain_counts(labelled),
                             'pending':_domain_counts(pending)},
            'registered_splits':splits, 'registered_split_counts':{name:len(ids) for name,ids in splits.items()},
            'fitted_physical_match_ids':sorted(labels) if model else [],
            'split_scope':'explicit_registered_A_B_C_physical_inventory_no_new_random_or_independent_test_claim',
            'independent_evaluation_performed':False, 'projection_versions':dict(sorted(projection_counts.items())),
            'source_bindings':source_bindings, 'source_bindings_sha256':_digest(source_bindings),
            'registration_inputs_sha256':_digest(previous_plans),
            'records_sha256':_digest(labelled), 'pending_sha256':_digest(pending),
            'transitions_sha256':_digest(transitions), 'conversion_reports':conversions,
            'model_training_performed':model is not None, 'model':model_info,
            'production_ready':False, 'calibration_status':'not_established',
            'network_requests':0, 'synthetic_mask_views':0,
            'original_actor_forecasts_changed':False, 'later_body_projections_used':False,
            'numeric_information_value':None, 'numeric_gain_supervision_created':False}
        if not fit_model:
            report.update(status='raw_corpus_only', corpus_only=True)
        if source_bindings != _bindings(folders+[result_archive]):
            raise ValueError('Original raw inputs changed during optional-prior training')
        for name, value in (('prefix-records.json', labelled), ('pending-prefixes.json', pending),
                            ('actual-transitions.json', transitions), ('physical-groups.json', physical_groups),
                            ('registration-inputs.json', previous_plans), ('report.json', report)):
            _write(staging/name, value)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, output)
    return report


def train_optional_context_from_runs(run_folders, previous_plans, result_archive, output, *, as_of=None):
    """Reconstruct original A evidence and fit only after the existing gate."""
    return _build_optional_context_from_runs(run_folders, previous_plans, result_archive, output,
                                            as_of=as_of, fit_model=True)


def build_optional_context_corpus_from_runs(run_folders, previous_plans, result_archive, output, *, as_of=None):
    """Reconstruct real A evidence without exposing any held-out data to fitting.

    This uses precisely the same raw identity, source, replay, settlement and
    physical registration checks as the existing training entry point.
    """
    return _build_optional_context_from_runs(run_folders, previous_plans, result_archive, output,
                                            as_of=as_of, fit_model=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--previous-plan', type=Path, action='append', required=True)
    parser.add_argument('--result-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--as-of')
    args = parser.parse_args(argv)
    report = train_optional_context_from_runs(args.run, [_read(p) for p in args.previous_plan],
        args.result_archive, args.output, as_of=args.as_of)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = ['VERSION', 'train_optional_context_from_runs', 'build_optional_context_corpus_from_runs', 'main']
