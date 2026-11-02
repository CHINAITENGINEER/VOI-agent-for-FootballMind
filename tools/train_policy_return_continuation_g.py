"""One declared offline experiment: signed policy-return h, no outcome oracle."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.train_learned_continuation_g import PREVIOUS, SOURCE, read, write, public_menu, change
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import state_candidate_gain as g1
from football_research.acquisition import learned_continuation_g as old
from football_research.acquisition import policy_return_continuation_g as new
from football_research.acquisition.historical_policy_comparison import rollout

OLD = ROOT/'var/reviews/fm-g-learned-two-step-20261014-v1'


def mean_menu_choice(module, model, menu, left, cost):
    decision = module.choose(model, menu, left, cost)
    scores = decision['candidates']
    if not scores:
        return decision
    shared = float(np.mean([s['continuation_net_value'] for s in scores]))
    for s in scores:
        s['continuation_net_value'] = shared
        s['two_step_net_value'] = s['immediate_gain']-cost+shared
    best = max(scores, key=lambda s: s['two_step_net_value'])
    return {'action_id': best['action_id'] if best['two_step_net_value'] > 0 else None,
            'reason': 'current_menu_common_continuation_ablation', 'candidates': scores}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    socket.socket.connect = no_network; socket.getaddrinfo = no_network
    plan = read(OLD/'plan.json'); base = g1.load(PREVIOUS/'g-state-candidate.json')
    old_model = old.load(OLD/'g-two-step.json')
    files = [OLD/'g-two-step.json', OLD/'continuation-training-targets.json', OLD/'plan.json',
             SOURCE/'transitions.json', SOURCE/'states.json', PREVIOUS/'prequery-feature-audit.json',
             PREVIOUS/'g-state-candidate.json']
    bindings = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files}
    write(out/'plan.json', {'schema': 'policy-return-continuation-experiment-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_bindings': bindings,
        'train_ids': plan['train_ids'], 'holdout_ids': plan['holdout_ids'], 'alpha': .1,
        'costs': [0., .001, .01], 'budgets': [1, 2, 3],
        'target_rule': 'teacher chooses by crossfit prediction; label is selected signed actual net return; stop=0',
        'negative_labels_retained': True, 'oracle_realized_max_forbidden': True,
        'fixed_immediate_model_and_f': True, 'feature_and_hyperparameter_search': False,
        'comparison': 'one_step, old_h, signed_return_h, each_h_current_menu_mean, random, fixed',
        'reused_holdout_development_diagnostic': True, 'new_independent_matches': 0,
        'online_selection_allowed': False})
    queries = {(q['state_id'], q['action_id']): q['query'] for q in read(PREVIOUS/'prequery-feature-audit.json')['queries']}
    graph = defaultdict(list)
    for t in read(SOURCE/'transitions.json'):
        graph[t['state_id']].append({**t, 'query': queries[t['state_id'], t['action_id']]})
    teachers = {mid: g1.load(OLD/'crossfit-teachers'/f"leave-out-{mid.replace(':','-')}.json") for mid in plan['train_ids']}
    rows = []
    for original in read(OLD/'continuation-training-targets.json'):
        mid = original['physical_match_id']; teacher = teachers[mid]
        assert set(teacher['fit_match_ids']) == set(plan['train_ids'])-{mid}
        future = graph.get(original['next_state_id'], [])
        assert all(e['physical_match_id'] == mid and e['g_split'] == 'train' for e in future)
        target = new.successor_target(teacher, future, original['packet']['cost_loss_units'])
        rows.append({**original, 'old_predicted_continuation_target': original['continuation_target'],
                     'continuation_target': target['continuation_target'], 'successor_policy_label': target,
                     'realized_successor_gain_used': True, 'realized_gain_used_for_action_selection': False})
    write(out/'training-targets.json', rows)
    model = new.fit(rows, base, alpha=.1); model['f_labeler_sha256'] = base['f_labeler_sha256']
    new.save(model, out/'g-policy-return.json'); model = new.load(out/'g-policy-return.json')
    states = read(SOURCE/'states.json')
    starts = {mid: max(((sid, s) for sid, s in states.items() if s['physical_match_id'] == mid and s['read_count'] == 0),
                      key=lambda pair: pair[1]['title_count'])[0] for mid in plan['holdout_ids']}
    selectors = {'old_h': lambda menu, left, c: old.choose(old_model, menu, left, c),
                 'signed_return_h': lambda menu, left, c: new.choose(model, menu, left, c),
                 'old_h_menu_mean': lambda menu, left, c: mean_menu_choice(old, old_model, menu, left, c),
                 'signed_return_h_menu_mean': lambda menu, left, c: mean_menu_choice(new, model, menu, left, c)}
    runs = []
    for mid, start in starts.items():
        for budget in (1, 2, 3):
            for cost in (0., .001, .01):
                for name in ('one_step', 'old_h', 'signed_return_h', 'old_h_menu_mean', 'signed_return_h_menu_mean', 'random', 'fixed'):
                    r = rollout(graph, start, policy='menu' if name in selectors else 'g' if name == 'one_step' else name,
                                budget=budget, cost=cost, scorer=lambda p: g1.estimate(base, p['query']),
                                menu_selector=selectors.get(name))
                    assert all(p['reads'] <= budget for p in r['paths'])
                    runs.append({**r, 'physical_match_id': mid, 'comparison_policy': name})
    write(out/'policy-replays.json', runs)
    summary = []
    for budget in (1, 2, 3):
        for cost in (0., .001, .01):
            for name in ('one_step', 'old_h', 'signed_return_h', 'old_h_menu_mean', 'signed_return_h_menu_mean', 'random', 'fixed'):
                selected = [r for r in runs if r['budget'] == budget and r['cost_loss_units'] == cost and r['comparison_policy'] == name]
                summary.append({'budget': budget, 'cost': cost, 'policy': name,
                                **{k: float(np.mean([r['expected_'+k] for r in selected])) for k in ('reads', 'gain', 'net')}})
    # A common downstream policy defines the evaluation target for both h models.
    diagnostics = []
    for edges in graph.values():
        for edge in edges:
            if edge['g_split'] != 'holdout': continue
            menu = public_menu(edges); own = next(a for a in menu if a['action_id'] == edge['action_id'])
            for cost in (0., .001, .01):
                label = new.successor_target(base, graph.get(edge['next_state_id'], []), cost)
                packet = {'candidate': own, 'menu': menu, 'cost_loss_units': cost}
                diagnostics.append({'physical_match_id': edge['physical_match_id'], 'state_id': edge['state_id'],
                    'action_id': edge['action_id'], 'cost': cost, 'target': label,
                    'old': old.estimate(old_model, packet, remaining_reads=2),
                    'new': new.estimate(model, packet, remaining_reads=2)})
    write(out/'continuation-diagnostics.json', diagnostics)
    metrics = []
    for cost in (0., .001, .01):
        per = [d for d in diagnostics if d['cost'] == cost]
        for name in ('old', 'new', 'zero'):
            per_match = []
            for mid in plan['holdout_ids']:
                ds = [d for d in per if d['physical_match_id'] == mid]
                pred = np.array([0. if name == 'zero' else d[name]['raw_continuation_estimate'] for d in ds])
                target = np.array([d['target']['continuation_target'] for d in ds])
                per_match.append({'match': mid, 'mse': float(np.mean((pred-target)**2)),
                                  'bias': float(np.mean(pred-target))})
            metrics.append({'cost': cost, 'model': name, 'equal_match_mse': float(np.mean([m['mse'] for m in per_match])),
                            'equal_match_bias': float(np.mean([m['bias'] for m in per_match])), 'per_match': per_match})
    write(out/'continuation-metrics.json', metrics)
    def actions(r): return [s['action_id'] for s in r['paths'][0]['steps']]
    ablations = []
    for name in ('old_h', 'signed_return_h'):
        checks = []
        for r in runs:
            if r['comparison_policy'] != name: continue
            shared = next(q for q in runs if q['comparison_policy'] == name+'_menu_mean' and
                          all(q[k] == r[k] for k in ('physical_match_id', 'budget', 'cost_loss_units')))
            checks.append({'match': r['physical_match_id'], 'budget': r['budget'], 'cost': r['cost_loss_units'],
                           'actions': actions(r), 'common_bonus_actions': actions(shared),
                           'same': actions(r) == actions(shared)})
        ablations.append({'model': name, 'different_cells': sum(not c['same'] for c in checks), 'cells': checks})
    write(out/'menu-dependence-ablation.json', ablations)
    budget_one_checks = 0
    for r in runs:
        if r['budget'] != 1 or r['comparison_policy'] not in selectors: continue
        one = next(q for q in runs if q['comparison_policy'] == 'one_step' and q['budget'] == 1 and
                   q['physical_match_id'] == r['physical_match_id'] and q['cost_loss_units'] == r['cost_loss_units'])
        assert actions(r) == actions(one) and r['expected_net'] == one['expected_net']; budget_one_checks += 1
    for path, digest in bindings.items(): assert sha256((ROOT/path).read_bytes()).hexdigest() == digest
    counts = {k: sum(test(r['continuation_target']) for r in rows) for k, test in
              [('positive', lambda v:v>1e-12), ('negative', lambda v:v < -1e-12), ('zero', lambda v:abs(v)<=1e-12)]}
    report = {'training_matches': len(plan['train_ids']), 'holdout_matches': len(plan['holdout_ids']),
              'original_transitions': len(rows)//3, 'cost_conditioned_rows': len(rows), 'target_counts': counts,
              'policy_summary': summary, 'policy_runs': len(runs), 'budget_one_checks': budget_one_checks,
              'new_independent_matches': 0, 'network': 0, 'api_calls': 0, 'production_ready': False,
              'limitations': ['Reused Leeds-only development holdout; not independent confirmation.',
                              'Signed target evaluates a fixed crossfit successor policy, not optimal future action.',
                              'Clipping signed predictions and switching successor policies are approximations.',
                              'Historical enumerated closed menus; no new-discovery transitions.']}
    write(out/'report.json', report)
    lines = ['h 实际后续收益训练对照', '', '新 h 学习排除整场的 g 所选择动作的真实净收益，包含负收益；不选择事后收益最大的动作。',
             '8场训练、5场重复使用的开发留出；不是新增独立验证。', f'训练目标分布：{counts}', '',
             '预算 | 成本 | 策略 | 平均读取 | 预测损失变化 | 计入成本后变化']
    for r in summary:
        lines.append(f"{r['budget']} | {r['cost']:.3f} | {r['policy']} | {r['reads']:.2f} | {change(r['gain'])} | {change(r['net'])}")
    lines += ['', '将同一菜单所有动作的 h 换成菜单平均值，策略发生变化的格数：']
    for a in ablations: lines.append(f"{a['model']}: {a['different_cells']}/45")
    lines += ['', '菜单均值对照仍保留状态相关停止奖励；相同表现不证明所有菜单信息无用，只说明动作间 h 的差异未改变这些轨迹。',
              '本轮不覆盖原模型，不启用线上选择；成本为假设损失单位。']
    (out/'研究结果.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import json
    print(json.dumps({'target_counts': counts, 'policy_runs': len(runs), 'budget_one_checks': budget_one_checks,
                      'budget2_cost001': [r for r in summary if r['budget'] == 2 and r['cost'] == .001],
                      'ablation_changed': {a['model']:a['different_cells'] for a in ablations}}, ensure_ascii=False))


if __name__ == '__main__': main()
