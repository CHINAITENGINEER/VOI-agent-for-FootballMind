"""Raw-backed, time-separated A development evaluation, never a B/C claim.

Both training-time and audit-time labels are reconstructed independently from
their then-observed raw result bytes. No full-corpus model is fitted first.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from football_research.acquisition.state import parse_time
from football_spider.storage.raw_archive import RawArchive
from .context_training import _bindings, _read, _write
from .optional_context_training import build_optional_context_corpus_from_runs
from .open_text_episodes import _digest

VERSION = 'raw-optional-context-development-v1'


def _result_snapshot(source, target, at):
    """Copy only original committed observations available at this cutoff."""
    source, target = Path(source).resolve(), Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    archive, copied, omitted = RawArchive(source), [], []
    for observation in archive.observations():
        archive.read_body(observation)
        if observation.observed_at > at:
            omitted.append(observation.id)
            continue
        payload = (source/observation.payload_path).resolve()
        metadata = payload.with_name(payload.name.removesuffix('.body.gz')+'.observation.json')
        for path in (payload, metadata):
            if not path.is_relative_to(source):
                raise ValueError('Original result observation escapes its archive')
            local = target/path.relative_to(source)
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, local)
        copied.append({'observation_id': observation.id, 'observed_at': observation.observed_at.isoformat(),
                       'raw_sha256': observation.content_hash})
    return {'as_of': at.isoformat(), 'observations': copied, 'later_observations_excluded': omitted}


def evaluate_optional_context_from_runs(run_folders, previous_plan_paths, result_archive, output,
                                        *, training_cutoff, as_of=None):
    """Reconstruct temporal development inputs using the existing real gateway.

    Input registration paths, raw runs and results are mandatory. A label file
    or already-fitted all-A model cannot substitute for these source records.
    The cutoff is explicit and this analysis remains retrospective development.
    """
    return _evaluate_context_from_runs(run_folders, previous_plan_paths, result_archive, output,
        training_cutoff=training_cutoff, as_of=as_of, role_context=False)


def _evaluate_context_from_runs(run_folders, previous_plan_paths, result_archive, output,
                                *, training_cutoff, as_of, role_context):
    if role_context:
        from football_research.evaluation.role_context_development import evaluate_role_context_development as evaluate_context_development
        version_schema = 'raw-role-context-development-v1'
    else:
        from football_research.evaluation.optional_context_development import evaluate_context_development
        version_schema = VERSION
    started = datetime.now(timezone.utc)
    cutoff, at = parse_time(training_cutoff), parse_time(started if as_of is None else as_of)
    if not cutoff < at <= started:
        raise ValueError('Explicit training cutoff must precede the reached audit time')
    if not isinstance(run_folders, (list, tuple)) or not run_folders:
        raise ValueError('Original registered A runs are required')
    if not isinstance(previous_plan_paths, (list, tuple)) or not previous_plan_paths:
        raise ValueError('Original A registration paths are required')
    folders = sorted({Path(path).resolve() for path in run_folders})
    paths = [Path(path).resolve() for path in previous_plan_paths]
    result_archive, output = Path(result_archive).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new immutable context development output')
    if any(output.is_relative_to(path) for path in folders+[result_archive]):
        raise ValueError('Development output cannot overwrite original raw inputs')
    bindings = _bindings(folders+[result_archive]+paths)
    plans = [_read(path) for path in paths]
    early_plans = [plan for plan in plans if parse_time(plan['created_at']) <= cutoff]
    early_folders = [folder for folder in folders if parse_time(_read(folder/'run.json')['finished_at']) <= cutoff]
    # Raw snapshots need permanent addresses in the downstream source manifests.
    # An interrupted output has no top-level report and is never a completed run.
    output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix='optional-context-development-') as tmp:
        staging = Path(tmp)
        late_source = _result_snapshot(result_archive, output/'audit-results', at)
        late_report = build_optional_context_corpus_from_runs(folders, plans, output/'audit-results',
                                                             output/'audit-corpus', as_of=at)
        full_rows = _read(output/'audit-corpus/prefix-records.json')
        pending = _read(output/'audit-corpus/pending-prefixes.json')
        early_source = _result_snapshot(result_archive, output/'training-results', cutoff)
        early_report, early_rows, early_pending = None, [], []
        if early_folders and early_plans:
            early_report = build_optional_context_corpus_from_runs(early_folders, early_plans,
                output/'training-results', output/'training-corpus', as_of=cutoff)
            early_rows = _read(output/'training-corpus/prefix-records.json')
            early_pending = _read(output/'training-corpus/pending-prefixes.json')
        # Later verification must agree with an already-known label and state.
        # Its new export timestamp cannot erase earlier proven availability.
        earlier = {row['prefix_id']: row for row in early_rows}
        latest = {row['prefix_id']: row for row in full_rows}
        latest_states = {row['prefix_id']: row for row in full_rows+pending}
        for previous in early_rows+early_pending:
            current = latest_states.get(previous['prefix_id'])
            if current is None or any(previous[k] != current[k] for k in
                                     ('physical_match_id', 'run_id', 'visible_state', 'kickoff_at')):
                raise ValueError('An earlier visible prefix changed or vanished at the later audit')
        if set(earlier)-set(latest):
            raise ValueError('A previously verified training label vanished at the later audit')
        merged = []
        for row in full_rows:
            previous = earlier.get(row['prefix_id'])
            if previous is not None:
                if any(previous[k] != row[k] for k in ('physical_match_id', 'run_id', 'visible_state', 'outcome', 'kickoff_at')):
                    raise ValueError('An earlier training label or visible state conflicts with its audit-time reconstruction')
                if not parse_time(previous['result_available_at']) <= cutoff:
                    raise ValueError('Training label was not proven before its cutoff')
            merged.append(deepcopy(previous if previous is not None else row))
        if role_context:
            from .role_context_training import role_records_from_verified_runs
            projected = role_records_from_verified_runs(merged+pending, folders)
            merged, pending = projected[:len(merged)], projected[len(merged):]
        version = version_schema+':'+_digest({'bindings': bindings, 'training_cutoff': cutoff.isoformat(),
                                      'as_of': at.isoformat(), 'records': merged})[:24]
        evaluation = evaluate_context_development(merged+pending, training_cutoff=cutoff, as_of=at,
            evidence_origin='real_observational', dataset_version=version)
        report = {'schema': version_schema, 'status': evaluation['status'], 'created_at': started.isoformat(),
            'training_cutoff': cutoff.isoformat(), 'as_of': at.isoformat(), 'dataset_version': version,
            'evidence_origin': 'real_observational', 'scope': 'retrospective_registered_A_development_not_B_or_C',
            'input_runs': len(folders), 'training_time_input_runs': len(early_folders),
            'supervised_prefixes': len(merged), 'pending_prefixes': len(pending),
            'earlier_proven_label_prefixes': len(earlier), 'evaluation': evaluation,
            'input_bindings': bindings, 'records_sha256': _digest(merged), 'pending_sha256': _digest(pending),
            'training_result_snapshot': early_source, 'audit_result_snapshot': late_source,
            'raw_corpus_reports': {'training': early_report, 'audit': late_report},
            'full_corpus_model_fitted': False, 'actual_fit_clock_rewritten': False,
            'numeric_query_gain_created': False, 'production_ready': False, 'network_requests': 0}
        if role_context:
            report.update(original_actor_role_features_changed=False,
                role_representation_scope='later_research_projection_of_then_available_original_identity_and_text')
        if bindings != _bindings(folders+[result_archive]+paths):
            raise ValueError('Original sources changed during development reconstruction')
        _write(staging/'development-records.json', merged)
        _write(staging/'pending-prefixes.json', pending)
        _write(staging/'evaluation.json', evaluation)
        _write(staging/'report.json', report)
        _write(staging/'source-inputs.json', {'schema': version_schema, 'run_folders': [str(p) for p in folders],
            'previous_plan_paths': [str(p) for p in paths], 'result_archive': str(result_archive),
            'training_cutoff': cutoff.isoformat(), 'as_of': at.isoformat(), 'input_bindings': bindings})
        # Publish report last; preserve all partial inputs if execution fails.
        for path in sorted(staging.iterdir(), key=lambda p: (p.name == 'report.json', p.name)):
            with (output/path.name).open('xb') as stream:
                stream.write(path.read_bytes())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--previous-plan', type=Path, action='append', required=True)
    parser.add_argument('--result-archive', type=Path, required=True)
    parser.add_argument('--training-cutoff', required=True)
    parser.add_argument('--as-of')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_optional_context_from_runs(args.run, args.previous_plan, args.result_archive,
        args.output, training_cutoff=args.training_cutoff, as_of=args.as_of), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
