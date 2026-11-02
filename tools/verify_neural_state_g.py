"""Reload saved neural g models and reproduce predictions and complete policies."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition import neural_state_gain as neural
from football_research.acquisition import state_candidate_gain as linear
from tools.train_neural_state_g import SOURCE, PREVIOUS, SEEDS, read, metrics, evaluate_policies
from tools.train_reviewed_rich_f import write
from tools.reconstruct_panel_priors import no_network


def verify(out):
    socket.socket.connect = no_network
    socket.getaddrinfo = no_network
    plan, report = read(out/'plan.json'), read(out/'report.json')
    for path, digest in plan['bindings'].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest
    query_map = {(r['state_id'], r['action_id']): r['query'] for r in read(PREVIOUS/'prequery-feature-audit.json')['queries']}
    rows = [{**r, 'query': query_map[r['state_id'], r['action_id']]} for r in read(SOURCE/'transitions.json')]
    train, held = ([r for r in rows if r['g_split'] == split] for split in ('train', 'holdout'))
    states = read(SOURCE/'states.json')
    baseline = linear.load(out/'linear-g.json')
    ensemble = read(out/'ensemble.json')
    for item in ensemble['members']:
        assert sha256((out/item['file']).read_bytes()).hexdigest() == item['sha256']
    models = [neural.load(out/f'mlp-g-seed-{seed}.json') for seed in SEEDS]
    for model in models:
        assert model['f_labeler_sha256'] == ensemble['f_labeler_sha256'] == baseline['f_labeler_sha256']
        assert model['fit_match_ids'] == sorted(plan['train_ids'])
        assert model['input_dimension'] == 54 and model['parameter_count'] == 1025
    scorers = {'linear': lambda q: linear.estimate(baseline, q),
               **{f'mlp_seed_{m["seed"]}': (lambda q, m=m: neural.estimate(m, q)) for m in models},
               'mlp_ensemble': lambda q: float(np.mean([neural.estimate(m, q) for m in models]))}
    predictions = []
    for name, scorer in {**scorers, 'zero': lambda q: 0., 'constant': lambda q: baseline['constant_baseline']}.items():
        result, ps = metrics(held, scorer)
        assert result == report['development_metrics'][name]
        predictions.extend({**p, 'model': name} for p in ps)
    assert predictions == read(out/'development-predictions.json')
    runs = evaluate_policies(held, states, scorers)
    assert runs == read(out/'policy-replays.json')
    oof_predictions, oof_runs = [], []
    for mid in sorted(plan['train_ids']):
        folder = out/'crossfit'/mid.replace(':', '-')
        base = linear.load(folder/'linear.json')
        ms = [neural.load(folder/f'mlp-{s}.json') for s in SEEDS]
        allowed = sorted(set(plan['train_ids'])-{mid})
        assert base['fit_match_ids'] == allowed
        for m in ms:
            assert m['fit_match_ids'] == allowed and m['f_labeler_sha256'] == baseline['f_labeler_sha256']
            fitting = [r for r in train if r['physical_match_id'] != mid]
            counts = Counter(r['physical_match_id'] for r in fitting)
            weights = np.array([1/(len(counts)*counts[r['physical_match_id']]) for r in fitting])
            y = np.array([r['signed_gain'] for r in fitting])
            assert abs(m['target_mean'] - weights@y) < 1e-15
            assert abs(m['target_scale'] - max(float(np.sqrt(weights@((y-weights@y)**2))), 1e-8)) < 1e-15
        testing = [r for r in train if r['physical_match_id'] == mid]
        fold_scorers = {'linear': lambda q: linear.estimate(base, q),
                        'mlp_ensemble': lambda q: float(np.mean([neural.estimate(m, q) for m in ms]))}
        for name, scorer in {**fold_scorers, 'zero': lambda q: 0., 'constant': lambda q: base['constant_baseline']}.items():
            _, ps = metrics(testing, scorer)
            oof_predictions.extend({**p, 'model': name, 'fit_match_ids': base['fit_match_ids']} for p in ps)
        oof_runs += evaluate_policies(testing, states, fold_scorers, budgets=(2,), costs=(.001,))
    assert oof_predictions == read(out/'crossfit-predictions.json')
    assert oof_runs == read(out/'crossfit-policy-replays.json')
    train_metrics = {name: metrics(train, scorers[name])[0] for name in ('linear', 'mlp_ensemble')}
    primary = [r for r in runs if r['budget'] == 2 and r['cost_loss_units'] == .001]
    comparison = []
    for mid in sorted({r['physical_match_id'] for r in held}):
        per_match = {r['model']: r for r in primary if r['physical_match_id'] == mid}
        old, new = per_match['linear'], per_match['mlp_ensemble']
        comparison.append({'physical_match_id': mid, 'linear_reads': old['expected_reads'],
            'neural_reads': new['expected_reads'], 'linear_loss_plus_cost': old['loss_plus_cost'],
            'neural_loss_plus_cost': new['loss_plus_cost'],
            'neural_signed_reduction_in_loss_plus_cost': old['loss_plus_cost']-new['loss_plus_cost'],
            'neural_steps': [{'title': query_map[s['state_id'], s['action_id']]['public']['title'],
                             **s} for s in new['paths'][0]['steps']]})
    write(out/'training-fit-metrics.json', train_metrics)
    write(out/'primary-per-match.json', comparison)
    result = {'status': 'PASS_REPRODUCIBILITY_NOT_GENERALIZATION',
        'full_neural_models_reloaded': len(models), 'crossfit_neural_models_reloaded': 8*len(SEEDS),
        'development_prediction_records_exact': len(predictions),
        'crossfit_prediction_records_exact': len(oof_predictions),
        'development_policy_replays_exact': len(runs), 'crossfit_policy_replays_exact': len(oof_runs),
        'source_bindings_unchanged': len(plan['bindings']), 'train_only_target_normalization': True,
        'whole_match_exclusions_verified': 8, 'network_calls': 0, 'training_performed': False,
        'runtime_changed': False, 'all_seed_primary_decisions_identical': all(
            next(r for r in primary if r['physical_match_id'] == mid and r['model'] == f'mlp_seed_{seed}')['paths'][0]['end_state_id']
            == next(r for r in primary if r['physical_match_id'] == mid and r['model'] == 'mlp_ensemble')['paths'][0]['end_state_id']
            for mid in plan['development_ids'] for seed in SEEDS),
        'interpretation': 'Neural policy improves the reused 5-match development panel, but worsens 8-match crossfit. No promotion.'}
    write(out/'verification.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args().output)
