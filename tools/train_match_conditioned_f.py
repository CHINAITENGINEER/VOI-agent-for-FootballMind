"""Bounded offline experiment. Existing models, live runs and g stay untouched."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.historical_context_training import (
    HistoricalContextModel, _discovery_panel, _state_for, prepare_historical_cases,
)
from football_research.acquisition.match_conditioned_f import (
    MatchConditionedF, PARAMETERS, ReconstructedMatchPrior, bind_prior, fit_forward_f, project,
)
from football_research.acquisition.state import parse_time


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def no_network(*args, **kwargs):
    raise AssertionError('This experiment is offline; new HTTP is forbidden')


def run(output):
    socket.socket.connect = no_network
    socket.getaddrinfo = no_network
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        'cases': ROOT/'var/submission/p8-g-stream-v3/cases.json',
        'old_f': ROOT/'var/submission/p9-g-timestamped-replay-20261010/model.json',
        'results': ROOT/'var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda/matches.jsonl',
    }
    bindings = {key: {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()} for key, path in paths.items()}
    original_files = {path: sha256(path.read_bytes()).hexdigest()
        for path in (ROOT/'var/f-text-live-integration-20261010-v1').rglob('*') if path.is_file()}
    old = HistoricalContextModel.load(paths['old_f'])
    splits = {group: old.metadata[group+'_match_ids'] for group in ('A', 'B', 'C')}
    # Freeze the scope before reading evaluation outcomes or fitting anything.
    protocol = {'schema': 'match-conditioned-f-experiment-v1',
        'frozen_at': datetime.now(timezone.utc).isoformat(), 'parameters': PARAMETERS,
        'inputs': bindings, 'splits': splits, 'network_requests_allowed': 0,
        'selection': 'original_A_only_forward_three_match_blocks_after_six_match_warmup',
        'baseline': 'existing_PoissonModel_default_ridge_fitted_before_each_match_information_time',
        'evaluation': 'B_and_C_once_after_freeze_previously_used_development_corpus_not_blind_test',
        'availability': 'source_reconstruction_results_and_text_versions_not_prospective_truth',
        'g': 'no_training_no_runtime_replacement_candidate_B_targets_separate',
        'promotion': 'offline_candidate_only_no_automatic_live_promotion',
        'early_season': 'only_existing_2025_season_no_preseason_strength_available'}
    write(output/'protocol.json', protocol)
    cases, audit = prepare_historical_cases(json.loads(paths['cases'].read_text(encoding='utf-8')))
    if set(c['fixture']['physical_match_id'] for c in cases) != set(sum(splits.values(), [])):
        raise ValueError('Original physical-match partition changed')
    prior = ReconstructedMatchPrior.from_export(paths['results'].parent)
    priors, prepared, states = {}, {}, {}
    # Each baseline is computed from earlier results, never an outcome feature
    # from the case being forecast. All reveal states share that baseline.
    for case in cases:
        mid = case['fixture']['physical_match_id']
        initial = _state_for(case, acquired_indices=[], discovered_indices=[])
        snapshot = prior.snapshot(initial['fixture'], initial['decision_at'])
        priors[mid] = snapshot
        state_rows, state_list = [], []
        for discovered, acquired, _, _, state in _discovery_panel(case):
            projection = project(bind_prior(state, snapshot))
            state_list.append((discovered, acquired, state))
            state_rows.append({'physical_match_id': mid, 'outcome': case['outcome'],
                'result_available_at': case['result_available_at'], 'state': projection})
        prepared[mid], states[mid] = state_rows, state_list
    write(output/'structured-priors.json', priors)
    a_rows = [row for mid in splits['A'] for row in prepared[mid]]
    print(json.dumps({'phase': 'training_A', 'matches': len(splits['A']), 'states': len(a_rows)}, ensure_ascii=False), flush=True)
    model = fit_forward_f(a_rows, dataset_binding={**bindings['cases'], 'split_ids': splits['A']})
    model.save(output/'model.json')
    restored = MatchConditionedF.load(output/'model.json')
    if model.model_version != restored.model_version:
        raise AssertionError('Candidate model save/load differs')
    write(output/'selection.json', model.metadata['selection'])
    print(json.dumps({'phase': 'f_frozen', 'version': model.model_version,
        'selected_regularization': model.metadata['selection']['selected_regularization']}, ensure_ascii=False), flush=True)
    groups = {}
    for group in ('B', 'C'):
        details = []
        for case in cases:
            mid = case['fixture']['physical_match_id']
            if mid not in splits[group]:
                continue
            y = case['outcome']
            scores = {'old_text_f': [], 'structured_prior_only': [], 'candidate_f': []}
            forecasts = []
            for discovered, acquired, state in states[mid]:
                old_p = old._probabilities(state)[1]
                current = restored.predict(state, prior_snapshot=priors[mid])
                if current != model.predict(state, prior_snapshot=priors[mid]):
                    raise AssertionError('Loaded model forecast differs')
                p, p0 = current['probabilities'], current['initial_probabilities']
                for name, values in [('old_text_f', old_p), ('structured_prior_only', p0), ('candidate_f', p)]:
                    scores[name].append(-math.log(values[y]))
                forecasts.append({'discovered_mask': discovered, 'acquired_mask': acquired,
                    'old_f': old_p, 'candidate_f': p, 'prior': p0})
            details.append({'physical_match_id': mid, 'home': case['fixture']['home_name'],
                'away': case['fixture']['away_name'], 'outcome': y, 'states': len(forecasts),
                'mean_state_log_loss': {key: float(np.mean(values)) for key, values in scores.items()},
                'forecasts': forecasts})
        groups[group] = {'physical_matches': len(details), 'matches': details,
            'mean_match_log_loss': {key: float(np.mean([d['mean_state_log_loss'][key] for d in details]))
                for key in ('old_text_f', 'structured_prior_only', 'candidate_f')}}
    write(output/'evaluation.json', groups)
    targets = []
    for case in cases:
        mid = case['fixture']['physical_match_id']
        if mid not in splits['B']:
            continue
        for discovered_mask, acquired_mask, discovered, acquired, before in _discovery_panel(case):
            p = model.predict(before, prior_snapshot=priors[mid])['probabilities']
            for index in discovered:
                if index in acquired:
                    continue
                after = _state_for(case, acquired_indices=acquired+[index], discovered_indices=discovered)
                q = model.predict(after, prior_snapshot=priors[mid])['probabilities']
                targets.append({'sample_id': f'{mid}:{discovered_mask}:{acquired_mask}:{index}',
                    'physical_match_id': mid, 'model_version': model.model_version,
                    'prior_snapshot_sha256': priors[mid]['snapshot_sha256'],
                    'prediction_before': p, 'prediction_after': q,
                    'outcome': case['outcome'], 'gain': math.log(q[case['outcome']]/p[case['outcome']]),
                    'gain_units': 'log_loss_nats', 'cost': None, 'net_gain': None,
                    'evidence_origin': 'historical_reconstruction', 'strict_g_training_eligible': False,
                    'eligibility_reason': 'reconstructed_versions_menus_and_clocks_no_actual_propensity_or_cost'})
    write(output/'B-candidate-targets.json', targets)
    for key, path in paths.items():
        if sha256(path.read_bytes()).hexdigest() != bindings[key]['sha256']:
            raise AssertionError('An input changed during the experiment')
    if not all(sha256(p.read_bytes()).hexdigest() == digest for p, digest in original_files.items()):
        raise AssertionError('A previous live artifact changed')
    report = {'schema': 'match-conditioned-f-result-v1', 'status': 'offline_candidate_complete',
        'model_version': model.model_version, 'network_requests': 0, 'g_training_calls': 0,
        'old_live_files_unchanged': len(original_files), 'source_audit': audit,
        'A_matches': len(splits['A']), 'A_states': len(a_rows),
        'validation_matches': model.metadata['selection']['independent_validation_matches'],
        'selected_regularization': model.metadata['selection']['selected_regularization'],
        'base_only_selected': model.metadata['selection']['base_only_selected'],
        'evaluation': {key: value['mean_match_log_loss'] for key, value in groups.items()},
        'structured_history_rows': len(prior.rows),
        'prior_training_rows_range': [min(len(p['training_match_ids']) for p in priors.values()), max(len(p['training_match_ids']) for p in priors.values())],
        'prior_cold_start_matches': [mid for mid, p in priors.items() if p['unknown_team_roles']],
        'B_candidate_targets': len(targets),
        'B_positive_gains': sum(r['gain'] > 1e-12 for r in targets),
        'B_negative_gains': sum(r['gain'] < -1e-12 for r in targets),
        'strict_g_eligible_rows': 0, 'live_promoted': False,
        'limitations': ['Previously reviewed historical development corpus, not untouched test.',
            'All text cases center on Leeds; no cross-club semantic generalization claim.',
            'Local role TF-IDF is not verified semantic evidence or calibrated uncertainty.',
            'No earlier-season results: early-season structured priors have cold starts.',
            'New f changes g targets; never combine old and new model gain labels.',
            'One-SE is a simplicity heuristic on dependent small-sample match losses.']}
    write(output/'report.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
