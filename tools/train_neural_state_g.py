"""Frozen-f, same-feature linear/MLP comparison, with whole-match cross-fitting."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition import neural_state_gain as neural
from football_research.acquisition import state_candidate_gain as linear
from football_research.acquisition.historical_gain_head import signed_gain
from football_research.acquisition.historical_policy_comparison import rollout
from tools.train_state_candidate_g import SOURCE, read
from tools.train_reviewed_rich_f import write
from tools.reconstruct_panel_priors import no_network

PREVIOUS = ROOT/'var/reviews/fm-g-state-candidate-policy-20261014-v1'
SEEDS = tuple(range(20261020, 20261025))


def metrics(rows, scorer):
    by_match, menus, predictions = defaultdict(list), defaultdict(list), []
    for row in rows:
        predicted = scorer(row['query'])
        by_match[row['physical_match_id']].append((predicted-row['signed_gain'])**2)
        menus[row['physical_match_id'], row['state_id']].append((predicted, row['signed_gain']))
        predictions.append({'physical_match_id': row['physical_match_id'], 'state_id': row['state_id'],
                            'action_id': row['action_id'], 'prediction': predicted, 'actual_gain': row['signed_gain']})
    rank = defaultdict(list)
    for (mid, _), values in menus.items():
        for i, (pi, yi) in enumerate(values):
            for pj, yj in values[i+1:]:
                if abs(yi-yj) <= 1e-12:
                    continue
                rank[mid].append(.5 if abs(pi-pj) <= 1e-12 else float((pi-pj)*(yi-yj) > 0))
    return {'match_weighted_mse': float(np.mean([np.mean(v) for v in by_match.values()])),
            'per_match_mse': {m: float(np.mean(v)) for m, v in by_match.items()},
            'within_menu_pair_concordance': float(np.mean([np.mean(v) for v in rank.values()])) if rank else None,
            'non_tied_observed_pairs': sum(map(len, rank.values())),
            'ranking_scope': 'posthoc noisy realized-gain comparison; not true expected-VoI ranking'}, predictions


def evaluate_policies(rows, states, scorers, budgets=(1, 2, 3), costs=(0., .001, .01)):
    graph, runs = defaultdict(list), []
    for r in rows:
        graph[r['state_id']].append(r)
    for mid in sorted({r['physical_match_id'] for r in rows}):
        start = max((sid for sid in graph if states[sid]['physical_match_id'] == mid and states[sid]['read_count'] == 0),
                    key=lambda sid: states[sid]['title_count'])
        first = graph[start][0]
        initial_loss = float(-np.log(first['prediction_before'][first['outcome']]))
        for budget in budgets:
            for cost in costs:
                for name in ['stop', 'fixed', 'random']+list(scorers):
                    scorer = scorers.get(name)
                    run = rollout(graph, start, policy='g' if scorer else name, budget=budget, cost=cost,
                                  scorer=(lambda a, fn=scorer: fn(a['query'])) if scorer else None)
                    for path in run['paths']:
                        assert path['reads'] <= budget
                        if path['steps']:
                            edges = [next(e for e in graph[s['state_id']] if e['action_id'] == s['action_id']) for s in path['steps']]
                            assert abs(signed_gain(edges[0]['prediction_before'], edges[-1]['prediction_after'], first['outcome'])-path['gain']) < 1e-12
                    runs.append({**run, 'model': name, 'physical_match_id': mid,
                                 'final_loss': initial_loss-run['expected_gain'],
                                 'loss_plus_cost': initial_loss-run['expected_net']})
    return runs


def summarize(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[r['model'], r['budget'], r['cost_loss_units']].append(r)
    return [{'model': name, 'budget': budget, 'cost': cost, 'matches': len(rs),
             **{k: float(np.mean([r[k] for r in rs])) for k in
                ('expected_reads', 'expected_gain', 'expected_net', 'final_loss', 'loss_plus_cost')}}
            for (name, budget, cost), rs in groups.items()]


def execute(out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    socket.socket.connect = no_network
    socket.getaddrinfo = no_network
    inputs = [SOURCE/'plan.json', SOURCE/'f-labeler.json', SOURCE/'transitions.json', SOURCE/'states.json',
              PREVIOUS/'g-state-candidate.json', PREVIOUS/'prequery-feature-audit.json']
    bindings = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in inputs}
    old_plan = read(SOURCE/'plan.json')
    write(out/'plan.json', {'schema': 'neural-g-controlled-experiment-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'bindings': bindings,
        'fixed_f': str((SOURCE/'f-labeler.json').relative_to(ROOT)),
        'f_retrained': False, 'h_retrained': False, 'feature_change': False,
        'architecture': [54, 16, 8, 1], 'activation': 'tanh', 'expected_full_parameter_count': 1025,
        'seeds': SEEDS, 'ensemble': 'predeclared arithmetic mean of all five; no best-seed selection',
        'mlp_kernel_l2': .01, 'linear_alpha': .1, 'hyperparameter_search': False,
        'normalization': 'training-only weighted target mean/std; input vectors unchanged',
        'weighting': 'equal matches then equal transitions within match',
        'optimization': 'CPU scipy L-BFGS-B analytic backprop; maxiter=10000 maxfun=30000 ftol=1e-10 gtol=1e-6',
        'numerical_revision': 'Initial v1 attempt hit 2000 training iterations before any held-out evaluation; retained its plan, changed optimizer limits/tolerances only.',
        'train_ids': old_plan['g_training_ids'], 'development_ids': old_plan['g_holdout_ids'],
        'crossfit': 'leave one entire training match out; refit vocabulary and normalization',
        'primary_budget': 2, 'primary_cost': .001, 'all_budgets': [1, 2, 3], 'all_costs': [0., .001, .01],
        'primary_metric': 'equal match final log loss plus research query cost',
        'holdout_reused': True, 'new_independent_matches': 0,
        'online_activation': False, 'network': 0,
        'comparison_limit': 'architecture and its regularizer differ; this is not proof that parameter count alone causes performance changes'})
    queries = {(r['state_id'], r['action_id']): r['query'] for r in read(PREVIOUS/'prequery-feature-audit.json')['queries']}
    rows = [{**r, 'query': queries[r['state_id'], r['action_id']]} for r in read(SOURCE/'transitions.json')]
    train, held = ([r for r in rows if r['g_split'] == name] for name in ('train', 'holdout'))
    train_ids, held_ids = set(r['physical_match_id'] for r in train), set(r['physical_match_id'] for r in held)
    assert train_ids == set(old_plan['g_training_ids']) and held_ids == set(old_plan['g_holdout_ids'])
    assert not train_ids & held_ids
    states = read(SOURCE/'states.json')
    for r in rows:
        assert r['query']['public']['probabilities'] == r['prediction_before']
        assert r['body_input_id'] not in states[r['state_id']]['used_input_ids']
        assert abs(r['signed_gain']-signed_gain(r['prediction_before'], r['prediction_after'], r['outcome'])) < 1e-12
    frozen = linear.load(PREVIOUS/'g-state-candidate.json')
    baseline = linear.fit(train, alpha=.1)
    baseline['f_labeler_sha256'] = bindings[str((SOURCE/'f-labeler.json').relative_to(ROOT))]
    assert all(abs(linear.estimate(baseline, r['query'])-linear.estimate(frozen, r['query'])) < 1e-12 for r in rows)
    linear.save(baseline, out/'linear-g.json')
    models, convergence = [], []
    for seed in SEEDS:
        model = neural.fit(train, baseline, seed=seed)
        assert model['parameter_count'] == 1025
        neural.save(model, out/f'mlp-g-seed-{seed}.json')
        restored = neural.load(out/f'mlp-g-seed-{seed}.json')
        assert all(neural.estimate(model, r['query']) == neural.estimate(restored, r['query']) for r in rows)
        models.append(restored)
        convergence.append({'scope': 'full_train', 'seed': seed, **model['optimizer']})
    print('Five 1025-parameter MLPs fitted; running whole-match cross-fit', flush=True)
    oof_metrics, oof_predictions, oof_runs = [], [], []
    for mid in sorted(train_ids):
        fitting, testing = [r for r in train if r['physical_match_id'] != mid], [r for r in train if r['physical_match_id'] == mid]
        fold = out/'crossfit'/mid.replace(':', '-')
        fold.mkdir(parents=True)
        base = linear.fit(fitting, alpha=.1)
        base['f_labeler_sha256'] = baseline['f_labeler_sha256']
        linear.save(base, fold/'linear.json')
        ns = []
        for seed in SEEDS:
            m = neural.fit(fitting, base, seed=seed)
            neural.save(m, fold/f'mlp-{seed}.json')
            assert mid not in m['fit_match_ids'] and set(m['fit_match_ids']) == train_ids-{mid}
            ns.append(neural.load(fold/f'mlp-{seed}.json'))
            convergence.append({'scope': 'crossfit', 'excluded_match': mid, 'seed': seed, **m['optimizer']})
        scorers = {'linear': lambda q: linear.estimate(base, q),
                   'mlp_ensemble': lambda q: float(np.mean([neural.estimate(m, q) for m in ns]))}
        for name, scorer in {**scorers, 'zero': lambda q: 0., 'constant': lambda q: base['constant_baseline']}.items():
            metric, ps = metrics(testing, scorer)
            oof_metrics.append({'excluded_match': mid, 'model': name, **metric})
            oof_predictions += [{**p, 'model': name, 'fit_match_ids': base['fit_match_ids']} for p in ps]
        oof_runs += evaluate_policies(testing, states, scorers, budgets=(2,), costs=(.001,))
        print('Cross-fit completed: '+mid, flush=True)
    write(out/'crossfit-metrics.json', oof_metrics)
    write(out/'crossfit-predictions.json', oof_predictions)
    write(out/'crossfit-policy-replays.json', oof_runs)
    scorers = {'linear': lambda q: linear.estimate(baseline, q),
               **{f'mlp_seed_{m["seed"]}': (lambda q, m=m: neural.estimate(m, q)) for m in models},
               'mlp_ensemble': lambda q: float(np.mean([neural.estimate(m, q) for m in models]))}
    evaluated, predictions = {}, []
    for name, scorer in {**scorers, 'zero': lambda q: 0., 'constant': lambda q: baseline['constant_baseline']}.items():
        evaluated[name], ps = metrics(held, scorer)
        predictions += [{**p, 'model': name} for p in ps]
    runs = evaluate_policies(held, states, scorers)
    summary = summarize(runs)
    primary = [r for r in summary if r['budget'] == 2 and r['cost'] == .001]
    write(out/'development-predictions.json', predictions)
    write(out/'policy-replays.json', runs)
    write(out/'policy-summary.json', summary)
    write(out/'convergence.json', convergence)
    write(out/'ensemble.json', {'schema': 'neural-g-mean-ensemble-manifest-v1',
        'aggregation': 'arithmetic_mean', 'f_labeler_sha256': baseline['f_labeler_sha256'],
        'members': [{'file': f'mlp-g-seed-{s}.json', 'sha256': sha256((out/f'mlp-g-seed-{s}.json').read_bytes()).hexdigest()} for s in SEEDS],
        'online_activation': False})
    for path, digest in bindings.items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest
    report = {'schema': 'neural-g-controlled-result-v1', 'full_models': len(models),
        'crossfit_models': len(convergence)-len(models), 'all_optimizers_converged': all(r['success'] for r in convergence),
        'parameter_count_per_full_model': 1025, 'train_matches': len(train_ids), 'train_transitions': len(train),
        'development_matches': len(held_ids), 'development_transitions': len(held),
        'frozen_linear_reproduced': True, 'prediction_roundtrip_exact': True,
        'crossfit_metrics': {name: float(np.mean([r['match_weighted_mse'] for r in oof_metrics if r['model'] == name]))
                            for name in ('linear', 'mlp_ensemble', 'zero', 'constant')},
        'crossfit_primary_policy': summarize(oof_runs), 'development_metrics': evaluated,
        'primary_policy': primary, 'elapsed_seconds': time.perf_counter()-started,
        'new_independent_matches': 0, 'gpu_required': False, 'network_calls': 0,
        'f_retrained': False, 'h_retrained': False, 'runtime_activation': False,
        'production_ready': False, 'promotion': 'research_candidate_only',
        'limitations': ['Repeated historical development set, not fresh blind validation.',
            'All g matches involve Leeds; transitions share outcomes within matches.',
            'No neural h attached; one-step adaptive reading isolates g capacity.',
            'Five seeds and eight folds do not create independent matches.',
            'Expected gain concerns the fixed research f, not intrinsic article value.',
            'Historical visibility assumptions unchanged; reconstructed menus and random paths are not prospective logs.']}
    write(out/'report.json', report)
    lines = ['FootballMind：同一 f / 同一输入的神经网络 g 对照',
             '网络 54→16→8→1，tanh，1025 参数；5 个预声明种子，无最优种子挑选。',
             '训练8场106转移，开发5场114转移；8个整场留一折，词表及输出归一化仅用折内训练。',
             'f 与收益标签冻结；本轮未训练 h、未接入旧 h、未部署。', '',
             '主要条件：最多2次读取、每次成本0.001研究损失单位。',
             '策略 | 平均读取 | 最终log loss | 加查询成本']
    for r in primary:
        lines.append(f"{r['model']} | {r['expected_reads']:.2f} | {r['final_loss']:.9f} | {r['loss_plus_cost']:.9f}")
    lines += ['', '开发集收益MSE：']+[f"{k}: {v['match_weighted_mse']:.9f}" for k, v in evaluated.items()]
    lines += ['', '重复开发集，不能据此宣称前瞻优势；收益为负或零的样本全部保留。',
              '参数更多是否更好，按完整策略损失与成本判断，不能只看拟合误差。']
    (out/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(out), 'seconds': report['elapsed_seconds'],
                      'crossfit': report['crossfit_metrics'], 'primary': primary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        execute(args.output)
