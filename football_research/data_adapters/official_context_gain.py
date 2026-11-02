"""Signed same-f supervision from registered official, market-free B runs.

Only original raw-replayed before/immediate-after states and independently
verified outcomes enter targets. Unknown predictor domains remain excluded.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor
from football_research.acquisition.state import parse_time
from .context_training import _bindings, _immediate_context_prefixes, _read, _write
from .evidence_context import _episodes
from .open_text_episodes import _digest
from .verified_fixtures import verify_independent_settlement
from football_spider.storage.raw_archive import RawArchive

VERSION = 'official-optional-context-gain-v1'
SOURCE_VERSION = 'raw-official-context-gain-inputs-v1'


def build_official_context_gain_from_runs(run_folders, result_archives, output, *, predictor_path, as_of=None):
    """Reconstruct registered B losses without caller-provided labels or gains.

    ``result_archives`` maps physical fixture IDs to existing independent raw
    result directories. Missing results remain pending, never zero targets.
    The explicit origin comes from the frozen model and registration, not a
    parameter that can relabel a constructed run as a real experiment.
    """
    from football_research.acquisition.optional_context_query import query_input
    return _build_context_gain_from_runs(run_folders, result_archives, output,
        predictor_path=predictor_path, as_of=as_of, predictor_loader=OptionalPriorContextPredictor.load,
        query_builder=query_input, version=VERSION, source_version=SOURCE_VERSION)


def _build_context_gain_from_runs(run_folders, result_archives, output, *, predictor_path, as_of,
                                  predictor_loader, query_builder, version, source_version,
                                  bind_visible=None):
    """Shared raw B verification; model-specific representation is explicit."""
    from football_agent.learned_voi_replay import replay_saved_run
    from .official_context_collection import verify_official_context_run

    started = datetime.now(timezone.utc)
    at = parse_time(started if as_of is None else as_of)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new immutable official context gain directory')
    if at > started:
        raise ValueError('Supervision audit time cannot follow actual execution')
    if not isinstance(run_folders, (list, tuple)) or not run_folders or not isinstance(result_archives, dict):
        raise ValueError('Explicit original run folders and physical result archive mapping are required')
    folders = sorted({Path(path).resolve() for path in run_folders})
    results = {physical: Path(path).resolve() for physical, path in result_archives.items()}
    if any(not isinstance(physical, str) or not physical for physical in results):
        raise ValueError('Independent result archives require canonical physical IDs')
    if any(not path.is_dir() for path in folders + list(results.values())):
        raise ValueError('Original run and result directories must exist')
    if any(output.is_relative_to(path) for path in folders + list(results.values())):
        raise ValueError('Gain output cannot be placed inside original raw inputs')
    predictor_path = Path(predictor_path).resolve()
    model = predictor_loader(predictor_path)
    model_sha = sha256(predictor_path.read_bytes()).hexdigest()
    origin = model.metadata['evidence_origin']
    if not parse_time(model.metadata['fit_execution_at']) <= at:
        raise ValueError('Outcome predictor must already have been fitted')
    bindings = _bindings(folders + list(results.values()) + [predictor_path])
    features, targets, excluded, prefixes, transitions = [], [], [], [], []
    registrations, result_checks, replays, seen_runs, physicals, facts = [], [], [], set(), set(), {}
    for folder in folders:
        run, config = _read(folder / 'run.json'), _read(folder / 'configuration.json')
        if (config.get('schema') != 'learned-voi-open-execution-v6' or run.get('schema') != config['schema']
                or run.get('run_id') != config.get('run_id') or run.get('configuration_digest') != _digest(config)
                or not parse_time(config['created_at']) <= parse_time(run['finished_at']) <= at):
            raise ValueError('A completed original official context run is required')
        if run['run_id'] in seen_runs:
            raise ValueError('Repeated run identity cannot duplicate actual gain exposure')
        seen_runs.add(run['run_id'])
        checked = verify_official_context_run(folder, config)
        binding, plan = checked['identity'], checked['registration']
        physical = binding['canonical']['source_event_id']
        if (plan.get('purpose') != 'calibration' or 'context_value_model' in config
                or plan.get('evidence_origin') != origin
                or physical in model.metadata['training_physical_match_ids']):
            raise ValueError('Only registered independent B runs of the same origin can supply query gains')
        artifact = config['context_predictor']
        if (artifact.get('sha256') != model_sha
                or sha256((folder / artifact['file']).read_bytes()).hexdigest() != model_sha
                or checked['predictor'].model_version != model.model_version):
            raise ValueError('Every actual query must use the same frozen predictor bytes')
        if not (parse_time(model.metadata['training_as_of']) <= parse_time(model.metadata['fit_execution_at'])
                <= parse_time(plan['created_at']) < parse_time(config['created_at'])):
            raise ValueError('Actual fitting and registration must precede acquisition')
        physicals.add(physical)
        registrations.append(deepcopy(plan))
        if physical in results:
            archive = RawArchive(results[physical])
            verified = verify_independent_settlement(binding, identity_archive_root=folder / 'identity-archive',
                result_archive_root=archive.root, observation_ids=[obs.id for obs in archive.observations()], as_of=at)
        else:
            verified = {'source_event_id': physical, 'status': 'PENDING', 'settlement_rows': [],
                'reason': 'independent_result_archive_not_supplied', 'verified_at': at.isoformat()}
        if verified['status'] != 'VERIFIED' and verified['settlement_rows']:
            raise ValueError('Unverified outcomes cannot create gain labels')
        result_checks.append(verified)
        with TemporaryDirectory(prefix='official-gain-replay-') as tmp:
            replay = replay_saved_run(folder, output=Path(tmp) / 'replay',
                                     project_root=Path(__file__).resolve().parents[2])
        if not replay['identical_replay'] or replay['replay_network_requests'] != 0:
            raise ValueError('Original raw context execution does not replay exactly')
        replays.append({key: replay[key] for key in ('run_id', 'identical_replay',
                       'raw_responses_reprojected', 'replay_network_requests')})
        rows, pairs = _immediate_context_prefixes(run, _episodes(folder, run),
            {'original_match_id': binding['match_id'], 'source_event_id': physical},
            {binding['match_id']: verified['settlement_rows']}, at)
        if bind_visible is not None:
            for row in rows:
                row['visible_state'] = bind_visible(row['visible_state'], binding,
                                                    folder/'identity-archive')
        if any(row['visible_state']['before']['prior_probabilities'] is not None for row in rows):
            raise ValueError('Official no-market B states must retain their original missing prior')
        prefixes.extend(rows); transitions.extend(pairs)
        by_id = {row['prefix_id']: row for row in rows}
        cycles = {cycle['step']: cycle for cycle in run['cycles']}
        for pair in pairs:
            reason = pair.get('gain_exclusion_reason')
            before = by_id[pair['before_prefix_id']]
            after = by_id.get(pair['after_prefix_id'])
            if reason is None and (after is None or 'outcome' not in before or 'outcome' not in after):
                reason = 'independent_settlement_missing'
            if reason:
                excluded.append({'run_id': run['run_id'], 'physical_match_id': physical, 'step': pair['step'],
                    'reason': reason, 'response_status': pair['response_status'],
                    'actual_cost': deepcopy(pair['actual_cost']), 'returned_at': pair['returned_at']})
                continue
            y = before['outcome']
            fact = (y, before['kickoff_at'])
            if (type(y) is not int or y not in (0, 1, 2) or y != after['outcome']
                    or before['result_available_at'] != after['result_available_at']
                    or physical in facts and facts[physical] != fact):
                raise ValueError('Conflicting independently verified physical outcome')
            facts[physical] = fact
            p = model.predict(before['visible_state'], physical_match_id=physical)
            q = model.predict(after['visible_state'], physical_match_id=physical)
            cycle = cycles[pair['step']]
            if (p['probabilities'] != cycle['public_before']['prediction']['probabilities']
                    or q['probabilities'] != cycle['public_after']['prediction']['probabilities']):
                raise ValueError('Signed gain must use the same actual before and immediate-return forecasts')
            if p['prior_domain_context_supported'] is not True or q['prior_domain_context_supported'] is not True:
                excluded.append({'run_id': run['run_id'], 'physical_match_id': physical, 'step': pair['step'],
                    'reason': 'predictor_prior_domain_context_unsupported', 'before_support': p['support_status'],
                    'after_support': q['support_status'], 'actual_cost': deepcopy(pair['actual_cost']),
                    'response_status': pair['response_status'], 'signed_gain': None})
                continue
            inputs = query_builder(before['visible_state'], pair['action'], p)
            sample_id = 'official-context-gain:' + _digest([run['run_id'], pair['step']])[:24]
            features.append({'sample_id': sample_id, 'inputs': inputs})
            targets.append({'sample_id': sample_id, 'run_id': run['run_id'], 'physical_match_id': physical,
                'before_prefix_id': before['prefix_id'], 'after_prefix_id': after['prefix_id'],
                'outcome': y, 'result_available_at': before['result_available_at'],
                'signed_gain': math.log(q['probabilities'][y]) - math.log(p['probabilities'][y]),
                'prediction_before': p['probabilities'], 'prediction_after': q['probabilities'],
                'before_support': p['support_status'], 'after_support': q['support_status'],
                'response_status': pair['response_status'], 'response_kind': pair['response_kind'],
                'actual_cost': deepcopy(pair['actual_cost']), 'selection_probability': pair['selection_probability'],
                'selection_scope': pair['selection_scope'], 'selection_basis': pair['selection_basis'],
                'registration_id': plan['registration_id'],
                'transition_sha256': _digest({'transition': pair, 'before': before, 'after': after}),
                'target_kind': 'realized_fixed_predictor_log_loss_reduction_not_expected_or_causal_value'})
    if set(results) - physicals:
        raise ValueError('Independent result mapping includes an unregistered corpus fixture')
    if bindings != _bindings(folders + list(results.values()) + [predictor_path]):
        raise ValueError('Original raw or predictor bytes changed during reconstruction')
    counts = Counter(row['physical_match_id'] for row in targets)
    for row in targets:
        row['physical_match_weight'] = 1. / (len(counts) * counts[row['physical_match_id']])
    source = {'schema': source_version, 'as_of': at.isoformat(), 'evidence_origin': origin,
        'run_folders': [str(path) for path in folders],
        'result_archives': {physical: str(path) for physical, path in results.items()},
        'input_bindings': bindings, 'exact_replay': replays}
    report = {'schema': version, 'as_of': at.isoformat(), 'evidence_origin': origin,
        'predictor_sha256': model_sha, 'predictor_model_version': model.model_version,
        'input_runs': len(folders), 'registered_B_physical_matches': len(physicals),
        'supervised_physical_matches': len(counts), 'actual_transitions': len(transitions),
        'supervised_transitions': len(targets), 'pending_prefixes': sum('outcome' not in row for row in prefixes),
        'negative_targets': sum(row['signed_gain'] < -1e-12 for row in targets),
        'zero_targets': sum(abs(row['signed_gain']) <= 1e-12 for row in targets),
        'positive_targets': sum(row['signed_gain'] > 1e-12 for row in targets),
        'excluded_transitions': excluded, 'independent_result_verification': result_checks,
        'features_sha256': _digest(features), 'targets_sha256': _digest(targets),
        'network_requests': 0, 'g_trained': False, 'production_ready': False,
        'scope': 'actual_registered_B_same_f_signed_loss_difference_not_unobserved_or_causal_effect',
        'original_actor_forecasts_changed': False, 'source_prior_missing_preserved': True}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (('source-inputs.json', source), ('calibration-registrations.json', registrations),
                       ('features.json', features), ('targets.json', targets),
                       ('prefix-records.json', prefixes), ('actual-transitions.json', transitions)):
        _write(output / name, data)
    report['source_inputs_file_sha256'] = sha256((output / 'source-inputs.json').read_bytes()).hexdigest()
    report['registrations_file_sha256'] = sha256((output / 'calibration-registrations.json').read_bytes()).hexdigest()
    _write(output / 'report.json', report)
    return report


__all__ = ['VERSION', 'SOURCE_VERSION', 'build_official_context_gain_from_runs']
