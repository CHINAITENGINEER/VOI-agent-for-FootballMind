"""Retrospective A-only development assessment of the existing optional-prior f.

This consumes caller-verified original visible prefixes, never a records-file
CLI claiming to verify raw evidence. It does not register B/C, publish a model,
search parameters, or reconstruct an actor's historical numerical forecasts.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import math
from statistics import mean

from football_research.acquisition.context_predictor import ORIGINS, _digest, _state
from football_research.acquisition.optional_prior_predictor import (
    OptionalPriorContextPredictor, training_readiness,
)
from football_research.acquisition.state import parse_time
from .metrics import LOG_LOSS_EPSILON

VERSION = 'optional-context-retrospective-development-v1'


def _groups(records, as_of, *, state_validator=None):
    if not isinstance(records, (list, tuple)):
        raise ValueError('Original visible prefix records must be a list or tuple')
    groups, aliases, runs, unique = defaultdict(list), {}, {}, {}
    for row in records:
        if not isinstance(row, dict) or row.get('purpose', 'f_train') != 'f_train':
            raise ValueError('Only A/f_train records are permitted in development')
        physical, run, index = row.get('physical_match_id'), row.get('run_id'), row.get('prefix_index')
        if (not isinstance(physical, str) or not physical or not isinstance(run, str) or not run
                or type(index) is not int or index < 0):
            raise ValueError('Canonical physical identity, original run_id and prefix_index are required')
        state = (state_validator(row['visible_state']) if state_validator is not None
                 else _state(row['visible_state'], allow_missing_prior=True))
        if state.get('physical_match_id', physical) != physical:
            raise ValueError('Verified visible identity differs from the physical fixture')
        if parse_time(state['decision_at']) > as_of:
            raise ValueError('Actual visible prefix follows evaluation as_of')
        provider = state['provider_match_id']
        if provider in aliases and aliases[provider] != physical:
            raise ValueError('One provider identity cannot represent different physical fixtures')
        aliases[provider] = physical
        run_facts = (physical, provider, state['kickoff_at'], state['prediction_cutoff'], state['prior'])
        if run in runs and runs[run] != run_facts:
            raise ValueError('One original run cannot change fixture, cutoff or original prior')
        runs[run] = run_facts
        y, label = row.get('outcome'), row.get('result_available_at')
        if (y is None) != (label is None):
            raise ValueError('Pending rows must have neither outcome nor result_available_at')
        if y is not None:
            if type(y) is not int or y not in (0, 1, 2):
                raise ValueError('An independent three-class outcome is required')
            if parse_time(label) < parse_time(state['kickoff_at']):
                raise ValueError('Independent settlement cannot precede kickoff')
        item = {'row': row, 'state': state}
        key = (run, index)
        identity = _digest([physical, row['visible_state'], y, label])
        if key in unique:
            if unique[key] != identity:
                raise ValueError('Conflicting original records for one run/prefix')
            continue
        unique[key] = identity
        groups[physical].append(item)
    for physical, items in groups.items():
        if len({item['state']['kickoff_at'] for item in items}) != 1:
            raise ValueError('Conflicting kickoff for one physical fixture; resolve postponement upstream')
        if len({item['row']['outcome'] for item in items if item['row'].get('outcome') is not None}) > 1:
            raise ValueError('Conflicting outcomes for one physical fixture')
        by_run = defaultdict(list)
        for item in items:
            by_run[item['row']['run_id']].append(item)
        for run_items in by_run.values():
            ordered = sorted(run_items, key=lambda item: item['row']['prefix_index'])
            if [item['row']['prefix_index'] for item in ordered] != list(range(len(ordered))):
                raise ValueError('Original run must retain its initial and contiguous actual prefixes')
            clocks = [parse_time(item['state']['decision_at']) for item in ordered]
            if clocks != sorted(clocks):
                raise ValueError('Actual prefix decision clocks cannot run backwards')
    return dict(sorted(groups.items())), len(records) - len(unique)


def _partition(groups, cutoff, as_of):
    train, development, excluded = {}, {}, []
    for physical, items in groups.items():
        clocks = [parse_time(item['state']['decision_at']) for item in items]
        reasons = []
        if min(clocks) <= cutoff < max(clocks):
            reasons.append('physical_fixture_crosses_actual_decision_boundary')
        pending = any(item['row'].get('outcome') is None for item in items)
        if pending:
            reasons.append('independent_settlement_pending_for_some_or_all_prefixes')
        labels = [parse_time(item['row']['result_available_at']) for item in items
                  if item['row'].get('result_available_at') is not None]
        if labels and max(labels) > as_of:
            reasons.append('independent_settlement_not_available_at_evaluation_as_of')
        training_side = max(clocks) <= cutoff
        if training_side and labels and max(labels) > cutoff:
            reasons.append('independent_settlement_not_available_at_training_cutoff')
        if reasons:
            excluded.append({'physical_match_id': physical, 'reasons': reasons,
                'run_ids': sorted({item['row']['run_id'] for item in items}),
                'prefixes': len(items), 'first_decision_at': min(clocks).isoformat(),
                'last_decision_at': max(clocks).isoformat()})
        else:
            (train if training_side else development)[physical] = items
    return train, development, excluded


def _text_inventory(groups):
    inventory = {}
    for physical, items in groups.items():
        for item in items:
            row, before = item['row'], item['row']['visible_state']['before']
            content = {entry['content_id']: entry['text'] for entry in before.get('contents', [])}
            entries = [(ref['role'], content[ref['content_id']])
                for doc in before.get('documents', []) for ref in doc.get('content_refs', [])]
            entries += [('link', anchor['text']) for link in before.get('observed_links', [])
                        for anchor in link.get('observed_anchors', [])]
            for role, text in entries:
                if not text.strip():
                    continue
                digest = sha256(text.encode()).hexdigest()
                entry = inventory.setdefault(digest, {'roles': set(), 'physical_match_ids': set(),
                    'prefixes': set(), 'occurrences': 0})
                entry['roles'].add(role)
                entry['physical_match_ids'].add(physical)
                entry['prefixes'].add((row['run_id'], row['prefix_index']))
                entry['occurrences'] += 1
    return inventory


def _duplication(train, development):
    left, right = _text_inventory(train), _text_inventory(development)
    def describe(inventory):
        return {'unique_texts': len(inventory), 'occurrences_across_prefixes': sum(x['occurrences'] for x in inventory.values()),
            'unique_texts_by_role': {role: sum(role in entry['roles'] for entry in inventory.values())
                                     for role in ('body', 'headline', 'link')},
            'texts_shared_by_physical_matches': sum(len(x['physical_match_ids']) > 1 for x in inventory.values()),
            'texts_repeated_across_prefixes': sum(len(x['prefixes']) > 1 for x in inventory.values()),
            'same_text_body_and_title_or_anchor': sum('body' in x['roles'] and bool(x['roles'] & {'headline', 'link'})
                                                     for x in inventory.values())}
    shared = [{'text_sha256': key, 'training_roles': sorted(left[key]['roles']),
        'development_roles': sorted(right[key]['roles']),
        'training_physical_match_ids': sorted(left[key]['physical_match_ids']),
        'development_physical_match_ids': sorted(right[key]['physical_match_ids'])}
        for key in sorted(left.keys() & right.keys())]
    return {'matching': 'exact_utf8_text_sha256_no_semantic_or_near_duplicate_claim',
        'training': describe(left), 'development': describe(right), 'shared_texts': shared,
        'shared_text_count': len(shared), 'rows_removed_for_text_overlap': 0,
        'interpretation': 'Cumulative prefixes, repeated titles and reposts are not new independent fixtures; overlap is disclosed, not proof of leakage or utility.'}


def _loss(probabilities, y):
    return {'log_loss': -math.log(max(probabilities[y], LOG_LOSS_EPSILON)),
            'brier_score': sum((p - int(i == y)) ** 2 for i, p in enumerate(probabilities))}


def _snapshot(model, item, physical):
    row = item['row']
    forecast = model.predict(row['visible_state'], physical_match_id=physical)
    losses = {'forecast': _loss(forecast['probabilities'], row['outcome']),
              'base': _loss(forecast['base_probabilities'], row['outcome'])}
    eligible = (forecast['prior_domain_context_supported'] and forecast['fallback_reason'] is None
                and not forecast['unknown_sources']
                and not forecast.get('role_support', {}).get('unseen_training_roles', []))
    return {'prefix_index': row['prefix_index'], 'decision_at': item['state']['decision_at'],
        'prediction_cutoff': item['state']['prediction_cutoff'],
        'visible_state_sha256': _digest(row['visible_state']), 'forecast': forecast, 'losses': losses,
        'context_comparison_eligible': bool(eligible),
        'context_gain': {key: losses['base'][key]-losses['forecast'][key] for key in losses['base']}
                        if eligible else None}


def _scores(snapshots):
    if not snapshots:
        return None
    losses = {name: {key: mean(s['losses'][name][key] for s in snapshots)
        for key in ('log_loss', 'brier_score')} for name in ('forecast', 'base')}
    return {'physical_matches': len(snapshots), **losses,
        'base_minus_forecast': {key: losses['base'][key]-losses['forecast'][key] for key in losses['base']}}


def _evaluate_context_development(records, *, training_cutoff, as_of, evidence_origin, dataset_version,
        model_factory=OptionalPriorContextPredictor, readiness_function=training_readiness,
        state_validator=None, representation_limitation=None):
    """Compare the single existing f with its own base on untouched later A.

    Supply complete original runs with run_id/prefix_index. Canonical fixtures
    that straddle the actual decision boundary are excluded in their entirety.
    Missing independent outcomes are pending, never zero-loss examples. The
    caller must separately verify raw identity, A registration and settlements.
    """
    started = datetime.now(timezone.utc)
    cutoff, at = parse_time(training_cutoff), parse_time(as_of)
    if not cutoff <= at <= started:
        raise ValueError('Training cutoff <= evaluation as_of <= actual execution is required')
    if evidence_origin not in ORIGINS or not isinstance(dataset_version, str) or not dataset_version.strip():
        raise ValueError('Explicit evidence origin and dataset version are required')
    groups, duplicate_rows = _groups(records, at, state_validator=state_validator)
    train, development, excluded = _partition(groups, cutoff, at)
    training = [item['row'] for items in train.values() for item in items]
    ready = readiness_function(training, as_of=cutoff, evidence_origin=evidence_origin)
    report = {'schema': VERSION, 'status': 'not_ready', 'scope': 'retrospective_A_development',
        'evidence_origin': evidence_origin, 'dataset_version': dataset_version,
        'training_cutoff': cutoff.isoformat(), 'as_of': at.isoformat(), 'evaluation_started_at': started.isoformat(),
        'input_records_sha256': _digest(records), 'input_records': len(records),
        'duplicate_run_prefix_rows': duplicate_rows, 'readiness': ready,
        'split': {'physical_matches': len(groups), 'training_physical_match_ids': sorted(train),
            'development_physical_match_ids': sorted(development), 'excluded': excluded,
            'boundary': 'all_actual_decisions_and_training_label_availability_not_kickoff_or_deadline',
            'roles': 'all_inputs_remain_A_no_B_or_C_registration'},
        'snapshot_selection': 'earliest_run_by_initial_decision_then_run_id; initial_and_terminal_actual_prefix; one_fixture_one_vote_in_each_table',
        'duplication': _duplication(train, development), 'model': None, 'actual_fit_execution_at': None,
        'summary': None, 'per_match': [], 'parameter_search_performed': False,
        'independent_confirmation': False, 'production_ready': False, 'information_value_established': False,
        'limitations': ['Raw identity, original visible snapshots, A registration and independent labels are verified by the caller gateway.',
            'Retrospective fit time is preserved; this f was not available to the original actor and is not authorized for historical live use.',
            'Losses compare forecasts on the same fixed observed snapshots, not alternative acquisition policies or causal information gains.',
            'Fallback forecasts are scored as fallback; unsupported domain/text/source comparisons are not established context gains.',
            'Exact text overlap includes cumulative prefixes, titles and boilerplate; no semantic duplication or relevance claim.',
            representation_limitation or 'The existing encoder excludes fixture identifiers; this evaluation does not establish home/away entity-role understanding.',
            'No confidence intervals, calibration claim, threshold search or new independent confirmation claim.']}
    if not ready['ready_to_fit']:
        return report
    if not development:
        report['status'] = 'no_development_matches'
        return report
    model = model_factory().fit(training, as_of=cutoff,
        dataset_version=dataset_version, evidence_origin=evidence_origin)
    report['model'] = {'schema': model.version, 'model_version': model.model_version,
        'artifact_content_sha256': _digest(model._payload()), 'metadata': deepcopy(model.metadata),
        'artifact_published': False}
    report['actual_fit_execution_at'] = model.metadata['fit_execution_at']
    for physical, items in development.items():
        runs = defaultdict(list)
        for item in items:
            runs[item['row']['run_id']].append(item)
        for run_items in runs.values():
            run_items.sort(key=lambda item: item['row']['prefix_index'])
        chosen = min(runs, key=lambda run: (parse_time(runs[run][0]['state']['decision_at']), run))
        initial, terminal = runs[chosen][0], runs[chosen][-1]
        report['per_match'].append({'physical_match_id': physical, 'run_id': chosen,
            'outcome': initial['row']['outcome'], 'original_runs': len(runs),
            'other_run_ids_not_scored': sorted(set(runs)-{chosen}), 'selected_run_prefixes': len(runs[chosen]),
            'initial': _snapshot(model, initial, physical), 'terminal': _snapshot(model, terminal, physical)})
    summary = {}
    for position in ('initial', 'terminal'):
        snapshots = [item[position] for item in report['per_match']]
        eligible = [s for s in snapshots if s['context_comparison_eligible']]
        summary[position] = {'all_forecasts_including_explicit_fallbacks': _scores(snapshots),
            'supported_context_comparisons_only': _scores(eligible),
            'domain_unsupported_physical_matches': sum(not s['forecast']['prior_domain_context_supported'] for s in snapshots),
            'fallback_physical_matches': sum(s['forecast']['fallback_reason'] is not None for s in snapshots),
            'source_extrapolation_physical_matches': sum(bool(s['forecast']['unknown_sources']) for s in snapshots),
            'context_comparison_physical_matches': len(eligible)}
    report.update(status='evaluated', summary=summary)
    return report


def evaluate_context_development(records, *, training_cutoff, as_of, evidence_origin, dataset_version):
    """Original optional-f evaluation; explicit versioned variants use private hooks."""
    return _evaluate_context_development(records, training_cutoff=training_cutoff, as_of=as_of,
        evidence_origin=evidence_origin, dataset_version=dataset_version)


__all__ = ['evaluate_context_development', 'VERSION']
