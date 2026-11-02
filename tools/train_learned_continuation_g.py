"""Train/evaluate one fixed two-step continuation experiment, offline only."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.train_state_candidate_g import SOURCE, read
from tools.train_reviewed_rich_f import write
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import state_candidate_gain as g1
from football_research.acquisition import learned_continuation_g as g2
from football_research.acquisition import historical_gain_head as labels
from football_research.acquisition.historical_policy_comparison import rollout

PREVIOUS = ROOT/'var/reviews/fm-g-state-candidate-policy-20261014-v1'


def public_menu(edges):
    return [{'action_id': t['action_id'], 'query': t['query']} for t in sorted(edges, key=lambda t: t['action_id'])]


def change(value):
    if abs(value) < 1e-12:
        return '损失不变'
    return f"损失{'降低' if value > 0 else '增加'} {abs(value):.9f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    socket.socket.connect = no_network; socket.getaddrinfo = no_network
    plan = read(PREVIOUS/'plan.json'); costs = [0., .001, .01]
    inputs = [SOURCE/'transitions.json', SOURCE/'states.json', PREVIOUS/'plan.json',
              PREVIOUS/'prequery-feature-audit.json', PREVIOUS/'g-state-candidate.json']
    bindings = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in inputs}
    write(out/'plan.json', {'schema': 'crossfit-continuation-g-plan-v1', 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'source_bindings': bindings, 'train_ids': plan['g_training_ids'], 'holdout_ids': plan['g_holdout_ids'],
        'alpha': .1, 'cost_grid_loss_units': costs, 'budgets': [1, 2, 3],
        'immediate_head': 'frozen_previous_state_candidate_g', 'continuation_target':
        'max(0, max g1_leave_whole_match_out(successor_candidate) - cost)',
        'teacher_fit': 'other seven training matches; no holdout matches',
        'continuation_features': 'current candidate vector, mean remaining current-menu vectors, remaining menu count, cost',
        'runtime': 'receding two-step horizon; actual next state is observed only after the selected action',
        'horizon_one': 'exactly previous one-step g; continuation forced to zero',
        'terminal_current_menu': 'single remaining candidate forces zero continuation',
        'negative_continuation_regression_predictions': 'clip to zero because STOP is available',
        'selection_cost_threshold': 'strictly positive predicted net value',
        'no_hyperparameter_search': True, 'new_independent_holdout': False,
        'no_realized_successor_gain_or_oracle_max_in_training_target': True,
        'network': 0, 'online_selection_allowed': False})
    queries = {(q['state_id'], q['action_id']): q['query'] for q in read(PREVIOUS/'prequery-feature-audit.json')['queries']}
    ts = [{**t, 'query': queries[t['state_id'], t['action_id']]} for t in read(SOURCE/'transitions.json')]
    train = [t for t in ts if t['g_split'] == 'train']
    assert {t['physical_match_id'] for t in train} == set(plan['g_training_ids'])
    train_graph = defaultdict(list)
    for t in train:
        train_graph[t['state_id']].append(t)
    teachers = {}
    (out/'crossfit-teachers').mkdir()
    for mid in plan['g_training_ids']:
        teacher = g1.fit([t for t in train if t['physical_match_id'] != mid], alpha=.1)
        assert set(teacher['fit_match_ids']) == set(plan['g_training_ids'])-{mid}
        path = out/'crossfit-teachers'/f"leave-out-{mid.replace(':', '-')}.json"
        g1.save(teacher, path); teachers[mid] = g1.load(path)
    targets = []
    for t in train:
        teacher = teachers[t['physical_match_id']]
        future = train_graph.get(t['next_state_id'], [])
        future_scores = [{'action_id': edge['action_id'], 'estimated_gain': g1.estimate(teacher, edge['query'])} for edge in future]
        menu = public_menu(train_graph[t['state_id']]); own = next(a for a in menu if a['action_id'] == t['action_id'])
        for cost in costs:
            target = max([0.] + [s['estimated_gain']-cost for s in future_scores])
            targets.append({'physical_match_id': t['physical_match_id'], 'state_id': t['state_id'],
                'action_id': t['action_id'], 'next_state_id': t['next_state_id'],
                'teacher_fit_match_ids': teacher['fit_match_ids'], 'teacher_successor_estimates': future_scores,
                'packet': {'candidate': own, 'menu': menu, 'cost_loss_units': cost},
                'continuation_target': target, 'observed_successor_used_for_training_target_only': True,
                'realized_successor_gain_used': False})
    write(out/'continuation-training-targets.json', targets)
    base = g1.load(PREVIOUS/'g-state-candidate.json')
    model = g2.fit(targets, base, alpha=.1)
    model['f_labeler_sha256'] = base['f_labeler_sha256']
    g2.save(model, out/'g-two-step.json'); model = g2.load(out/'g-two-step.json')
    states = read(SOURCE/'states.json'); graph = defaultdict(list)
    for t in ts:
        graph[t['state_id']].append(t)
    runs = []; comparisons = []
    for mid in plan['g_holdout_ids']:
        start = max(((sid,s) for sid,s in states.items() if s['physical_match_id'] == mid and s['read_count'] == 0), key=lambda p:p[1]['title_count'])[0]
        for budget in (1, 2, 3):
            for cost in costs:
                for policy in ('stop', 'fixed', 'random', 'one_step', 'two_step'):
                    run = rollout(graph, start, policy='g' if policy == 'one_step' else 'menu' if policy == 'two_step' else policy,
                        budget=budget, cost=cost, scorer=lambda p:g1.estimate(base, p['query']),
                        menu_selector=lambda menu, left, c:g2.choose(model, menu, left, c))
                    for path in run['paths']:
                        if path['steps']:
                            chosen = [next(e for e in graph[s['state_id']] if e['action_id'] == s['action_id']) for s in path['steps']]
                            exact = labels.signed_gain(chosen[0]['prediction_before'], chosen[-1]['prediction_after'], chosen[0]['outcome'])
                            assert abs(exact-path['gain']) < 1e-12
                        assert path['reads'] <= budget
                    runs.append({**run, 'physical_match_id': mid, 'comparison_policy': policy})
                one, two = runs[-2:]
                if budget == 1:
                    assert [s['action_id'] for s in one['paths'][0]['steps']] == [s['action_id'] for s in two['paths'][0]['steps']]
                    assert one['expected_net'] == two['expected_net']
    for budget in (1, 2, 3):
        for cost in costs:
            cell = [r for r in runs if r['budget'] == budget and r['cost_loss_units'] == cost]
            refs = {p: {r['physical_match_id']:r for r in cell if r['comparison_policy'] == p} for p in ('one_step', 'random')}
            for policy in ('stop', 'fixed', 'random', 'one_step', 'two_step'):
                selected = [r for r in cell if r['comparison_policy'] == policy]
                result = {'budget': budget, 'cost_loss_units': cost, 'policy': policy,
                    **{k:float(np.mean([r['expected_'+k] for r in selected])) for k in ('reads','gain','net')}}
                for reference in refs:
                    diffs = [r['expected_net']-refs[reference][r['physical_match_id']]['expected_net'] for r in selected]
                    result['paired_net_vs_'+reference] = diffs
                    result['mean_net_vs_'+reference] = float(np.mean(diffs))
                comparisons.append(result)
    write(out/'policy-replays.json', runs)
    report = {'schema':'crossfit-continuation-g-result-v1', 'g_trained': True,
        'training_matches': len(plan['g_training_ids']), 'holdout_matches':len(plan['g_holdout_ids']),
        'original_training_transitions':len(train), 'cost_conditioned_target_rows':len(targets),
        'new_independent_matches':0, 'teacher_models':len(teachers),
        'positive_continuation_targets':sum(r['continuation_target']>1e-12 for r in targets),
        'policy_summary':comparisons, 'production_ready':False, 'online_selection_allowed':False,
        'network':0, 'provider_calls':0,
        'limitations':['Reused development corpus and holdout, all g matches involve Leeds.',
            'Successor value is predicted by seven-match cross-fit teachers, not ground-truth optimal continuation.',
            'Clipped continuation estimates may be optimistic; no uncertainty calibration.',
            'At most two-step fitted continuation, not full transition distribution integration.',
            'Menus contain archived body reads; no new candidate discovery/tool failures modeled.',
            'Historical reconstruction and fixed-f gain, not live or causal validation.']}
    write(out/'report.json', report)
    # Human-facing signs are expressed with direction words, as requested.
    lines=['FootballMind：学习式二步信息价值实验', '',
        'g2净价值 = 当前g1预计收益 - 本次成本 + 预计后续净价值。',
        '后续目标用排除当前整场比赛的g1估计，不使用本场实际后续收益的最大值。',
        '训练8场、留出5场；106条训练转移在3种成本下构成318条目标，不增加独立样本。',
        '使用上一轮冻结f/g1与原划分，预算1时严格退化为g1。', '',
        '预算 | 单次成本 | 策略 | 平均读取 | 不计成本预测损失变化 | 计入成本后变化']
    for r in comparisons:
        lines.append(f"{r['budget']} | {r['cost_loss_units']:.3f} | {r['policy']} | {r['reads']:.2f} | {change(r['gain'])} | {change(r['net'])}")
    lines += ['', '本报告是既有历史开发集实验；所有成本均为假设损失单位，不是美元。',
        '完整各步即时收益、预计后续价值、选择/停止原因见policy-replays.json。',
        '后续价值头尚未校准，不得把这次结果称为真实开放环境最优策略。']
    with (out/'阶段结果.txt').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')
    for path,digest in bindings.items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest
    write(out/'verification.json', {'bound_inputs_unchanged':len(bindings),
        'crossfit_teachers_whole_match_excluded':len(teachers), 'holdout_used_for_training':False,
        'budget_one_equal_to_one_step_cells':15, 'policy_runs':len(runs),
        'path_gain_telescoping_and_budget_checks':True, 'network':0,'provider_calls':0})
    import json
    print(json.dumps({'positive_continuation_targets':report['positive_continuation_targets'],
                      'policy_summary':comparisons},ensure_ascii=False))


if __name__ == '__main__':
    main()
