"""One fixed f representation experiment, with fully regenerated paired g/h.

Uses archived research evidence only. No online activation or new network calls.
Reused evaluation matches are development diagnostics, not a fresh blind test.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
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
from football_research.acquisition import symmetric_semantic_f as candidate_f
from football_research.acquisition import entity_evidence_updates as old_f
from football_research.acquisition import state_candidate_gain as g
from football_research.acquisition import policy_return_continuation_g as h
from football_research.acquisition.historical_policy_comparison import rollout
from football_research.acquisition.historical_gain_head import signed_gain
from tools.train_learned_continuation_g import SOURCE, PREVIOUS, read, write, public_menu, change
from tools.train_state_candidate_g import EVENTS, before_snapshot
from tools.reconstruct_panel_priors import no_network

OLD_H = ROOT/'var/reviews/fm-g-policy-return-continuation-20261014-v1/g-policy-return.json'
EXTRA = ROOT/'var/reviews/fm-g-nonleeds-expansion-20261015-v1/evaluation/prediction-input-rows.json'
EXTRA_LABELS = EXTRA.parent/'loss-and-acceptance.json'
COSTS = (0., .001, .01)


def public_row(row):
    return {k: deepcopy(row[k]) for k in ('physical_match_id', 'prior', 'decision_at',
                                         'used_input_ids', 'evidence_events')}


def scores(rows, predictor, labels=None):
    by = defaultdict(list)
    for row in rows:
        p = predictor(public_row(row))
        # External panel intentionally stores labels separately from prediction input.
        y = labels[row['physical_match_id']] if labels is not None else row['outcome']
        if type(y) is not int or y not in (0, 1, 2):
            raise ValueError('Settled outcome required after prediction')
        by[row['physical_match_id']].append({'loss': float(-np.log(p[y])),
            'brier': float(np.sum((p-np.eye(3)[y])**2))})
    per = {mid: {k: float(np.mean([r[k] for r in rs])) for k in ('loss', 'brier')} for mid, rs in by.items()}
    return {'per_match': per, 'matches': len(per),
            **{k: float(np.mean([p[k] for p in per.values()])) for k in ('loss', 'brier')}}


def execute(out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    socket.socket.connect = no_network
    socket.getaddrinfo = no_network
    plan = read(SOURCE/'plan.json')
    f_ids, train_ids, held_ids = (plan[k] for k in ('f_training_ids', 'g_training_ids', 'g_holdout_ids'))
    assert not (set(f_ids)&set(train_ids) or set(f_ids)&set(held_ids) or set(train_ids)&set(held_ids))
    inputs = [EVENTS, SOURCE/'plan.json', SOURCE/'transitions.json', SOURCE/'states.json',
              SOURCE/'f-labeler.json', PREVIOUS/'g-state-candidate.json', OLD_H, EXTRA, EXTRA_LABELS]
    bindings = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in inputs}
    write(out/'plan.json', {'schema': 'symmetric-f-paired-g-experiment-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_bindings': bindings,
        'f_train_ids': f_ids, 'g_train_ids': train_ids, 'development_holdout_ids': held_ids,
        'f_regularization': 10., 'g_and_h_alpha': .1, 'parameter_search': False,
        'f_change': 'shared home-away typed-presence residual; prior unchanged; no lexical/name parameters',
        'costs': COSTS, 'budgets': [1, 2, 3], 'primary_budget': 2, 'primary_cost': .001,
        'primary_metric': 'equal-match final log loss plus declared query cost',
        'regenerate_all_g_labels': True, 'positive_zero_negative_retained': True,
        'previously_used_development_holdout': True, 'new_independent_matches': 0,
        'extra_nonleeds': 'two previously used diagnostic matches; never used for fitting',
        'historical_visibility_assumptions': plan['time_basis'],
        'network': 0, 'provider_calls': 0, 'runtime_activation': False})
    rows = read(EVENTS)
    facts = {r['physical_match_id']: r for r in rows}
    assert max(old_f.timestamp(facts[m]['result_available_at']) for m in f_ids) < min(old_f.timestamp(facts[m]['decision_at']) for m in train_ids)
    assert max(old_f.timestamp(facts[m]['result_available_at']) for m in train_ids) < min(old_f.timestamp(facts[m]['decision_at']) for m in held_ids)
    fitted_f = candidate_f.fit([r for r in rows if r['physical_match_id'] in f_ids], 10.)
    candidate_f.save(fitted_f, out/'f-labeler.json')
    fitted_f = candidate_f.load(out/'f-labeler.json')
    f_digest = sha256((out/'f-labeler.json').read_bytes()).hexdigest()
    existing_f = old_f.load(SOURCE/'f-labeler.json')
    print('f fitted; regenerating transitions', flush=True)
    keyed = {(r['physical_match_id'], tuple(sorted(r['used_input_ids']))): r for r in rows}
    states = read(SOURCE/'states.json')
    state_rows = {sid: keyed[s['physical_match_id'], tuple(sorted(s['used_input_ids']))] for sid, s in states.items()}
    predictions = {sid: candidate_f.predict(fitted_f, public_row(r)).tolist() for sid, r in state_rows.items()}
    write(out/'state-predictions.json', predictions)
    original = read(SOURCE/'transitions.json')
    enriched, old_enriched = [], []
    for t in original:
        sid, nxt = t['state_id'], t['next_state_id']
        p, q = predictions[sid], predictions[nxt]
        assert set(states[nxt]['used_input_ids'])-set(states[sid]['used_input_ids']) == {t['body_input_id']}
        snapshot = before_snapshot(state_rows[sid])
        assert t['body_input_id'] not in snapshot['used_input_ids']
        old_enriched.append({**t, 'query': g.build_query(t['query'], snapshot)})
        query = {**t['query'], 'probabilities': p}
        gain = signed_gain(p, q, t['outcome'])
        before_features = candidate_f.public_features(public_row(state_rows[sid]))[1:]
        after_features = candidate_f.public_features(public_row(state_rows[nxt]))[1:]
        changed = any(not np.array_equal(a, b) for a, b in zip(before_features, after_features))
        enriched.append({**t, 'query': g.build_query(query, snapshot),
            'prediction_before': p, 'prediction_after': q, 'signed_gain': gain,
            'loss_before': float(-np.log(p[t['outcome']])), 'loss_after': float(-np.log(q[t['outcome']])),
            'f_labeler_sha256': f_digest,
            'effect_class': 'probability_changed' if np.max(np.abs(np.array(p)-q)) > 1e-12 else
                'representation_changed_probability_unchanged' if changed else 'representation_unchanged'})
    write(out/'transitions.json', enriched)
    train = [t for t in enriched if t['g_split'] == 'train']
    assert {t['physical_match_id'] for t in train} == set(train_ids)
    immediate = g.fit(train, alpha=.1)
    immediate['f_labeler_sha256'] = f_digest
    g.save(immediate, out/'g-immediate.json')
    immediate = g.load(out/'g-immediate.json')
    graph = defaultdict(list)
    old_graph = defaultdict(list)
    for t in enriched:
        graph[t['state_id']].append(t)
    for t in old_enriched:
        old_graph[t['state_id']].append(t)
    targets = []
    (out/'crossfit-teachers').mkdir()
    for mid in train_ids:
        teacher = g.fit([t for t in train if t['physical_match_id'] != mid], alpha=.1)
        assert set(teacher['fit_match_ids']) == set(train_ids)-{mid}
        teacher['f_labeler_sha256'] = f_digest
        g.save(teacher, out/'crossfit-teachers'/f"without-{mid.replace(':','-')}.json")
        for t in (t for t in train if t['physical_match_id'] == mid):
            future = graph.get(t['next_state_id'], [])
            assert all(e['g_split'] == 'train' and e['physical_match_id'] == mid for e in future)
            menu = public_menu(graph[t['state_id']])
            own = next(a for a in menu if a['action_id'] == t['action_id'])
            for cost in COSTS:
                target = h.successor_target(teacher, future, cost)
                targets.append({'physical_match_id': mid, 'state_id': t['state_id'], 'action_id': t['action_id'],
                    'teacher_fit_match_ids': teacher['fit_match_ids'],
                    'packet': {'candidate': own, 'menu': menu, 'cost_loss_units': cost},
                    'continuation_target': target['continuation_target'], 'successor_policy_label': target})
    write(out/'continuation-targets.json', targets)
    continuation = h.fit(targets, immediate, alpha=.1)
    continuation['f_labeler_sha256'] = f_digest
    h.save(continuation, out/'g-policy-return.json')
    continuation = h.load(out/'g-policy-return.json')
    existing_h = h.load(OLD_H)
    print('paired g/h fitted; evaluating unchanged development matches', flush=True)
    evaluations = {}
    extra_labels = {m['physical_match_id']: m['outcome'] for m in read(EXTRA_LABELS)['matches']}
    for scope, group in [('development_holdout', [r for r in rows if r['physical_match_id'] in held_ids]),
                         ('nonleeds_reused_diagnostic', read(EXTRA))]:
        assert not set(r['physical_match_id'] for r in group)&set(f_ids+train_ids)
        evaluations[scope] = {name: scores(group, fn, extra_labels if scope == 'nonleeds_reused_diagnostic' else None) for name, fn in {
            'prior': lambda r: np.array(r['prior']),
            'old_f': lambda r: old_f.predict(existing_f, r),
            'symmetric_f': lambda r: candidate_f.predict(fitted_f, r)}.items()}
    write(out/'f-evaluation.json', evaluations)
    replays = []
    for mid in held_ids:
        start = max((sid for sid, s in states.items() if s['physical_match_id'] == mid and s['read_count'] == 0),
                    key=lambda sid: states[sid]['title_count'])
        for budget in (1, 2, 3):
            for cost in COSTS:
                for family, edges, labeler, two in [('old_pair', old_graph, existing_f, existing_h),
                                                   ('symmetric_pair', graph, fitted_f, continuation)]:
                    start_p = old_f.predict(labeler, public_row(state_rows[start])) if family == 'old_pair' else np.array(predictions[start])
                    start_loss = float(-np.log(start_p[state_rows[start]['outcome']]))
                    for policy in ('stop', 'fixed', 'random', 'one_step', 'two_step'):
                        run = rollout(edges, start, policy='g' if policy == 'one_step' else 'menu' if policy == 'two_step' else policy,
                            budget=budget, cost=cost,
                            scorer=lambda a, m=two['immediate_model']: g.estimate(m, a['query']),
                            menu_selector=lambda menu, left, c, m=two: h.choose(m, menu, left, c))
                        for path in run['paths']:
                            assert path['reads'] <= budget
                            if path['steps']:
                                chosen = [next(e for e in edges[s['state_id']] if e['action_id'] == s['action_id']) for s in path['steps']]
                                assert abs(signed_gain(chosen[0]['prediction_before'], chosen[-1]['prediction_after'], chosen[0]['outcome'])-path['gain']) < 1e-12
                        replays.append({**run, 'physical_match_id': mid, 'family': family, 'comparison_policy': policy,
                            'initial_loss': start_loss, 'final_loss': start_loss-run['expected_gain'],
                            'loss_plus_cost': start_loss-run['expected_net']})
    write(out/'policy-replays.json', replays)
    summary = []
    for family in ('old_pair', 'symmetric_pair'):
        for budget in (1, 2, 3):
            for cost in COSTS:
                for policy in ('stop', 'fixed', 'random', 'one_step', 'two_step'):
                    group = [r for r in replays if r['family'] == family and r['budget'] == budget and r['cost_loss_units'] == cost and r['comparison_policy'] == policy]
                    summary.append({'family': family, 'policy': policy, 'budget': budget, 'cost': cost,
                        **{k: float(np.mean([r[k] for r in group])) for k in ('expected_reads', 'expected_gain', 'final_loss', 'loss_plus_cost')}})
    write(out/'policy-summary.json', summary)
    held = [t for t in enriched if t['g_split'] == 'holdout']
    gain_eval = {}
    for name, scorer in [('trained_g', lambda t: g.estimate(immediate, t['query'])),
                         ('zero', lambda t: 0.), ('constant', lambda t: immediate['constant_baseline'])]:
        by_match = defaultdict(list)
        for t in held:
            by_match[t['physical_match_id']].append((scorer(t)-t['signed_gain'])**2)
        gain_eval[name] = float(np.mean([np.mean(v) for v in by_match.values()]))
    primary = [r for r in summary if r['budget'] == 2 and r['cost'] == .001]
    for path, digest in bindings.items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest
    distribution = {split: {name: sum(test(t['signed_gain']) for t in enriched if t['g_split'] == split)
        for name, test in [('positive', lambda v: v > 1e-12), ('zero', lambda v: abs(v) <= 1e-12), ('negative', lambda v: v < -1e-12)]}
        for split in ('train', 'holdout')}
    report = {'schema': 'symmetric-f-paired-g-result-v1', 'f_trained': True, 'g_trained': True, 'h_trained': True,
        'f_training_matches': len(f_ids), 'g_training_matches': len(train_ids), 'development_holdout_matches': len(held_ids),
        'f_parameter_count': fitted_f['parameter_count'], 'f_optimizer': fitted_f['optimizer'],
        'training_transitions': len(train), 'development_transitions': len(held), 'gain_distribution': distribution,
        'continuation_training_rows': len(targets), 'equal_match_gain_mse': gain_eval,
        'primary_comparison': primary, 'f_evaluation': evaluations,
        'elapsed_seconds': time.perf_counter()-started, 'gpu_required': False,
        'network': 0, 'provider_calls': 0, 'new_independent_matches': 0, 'production_ready': False,
        'runtime_activation': False, 'decision': 'candidate_only_no_automatic_promotion',
        'limitations': ['Reused development holdout, not new blind validation.',
            'Historical reconstruction and enumerated closed menus; clocks include declared historical assumptions.',
            'Fixed prior and reviewed semantic extraction; no new raw-body encoder training.',
            'Semantic presence loses player importance and number-of-absentees information.',
            'Symmetric residual sharing is a hypothesis; prior retains home advantage.',
            'Continuation clipping retained from baseline; policy improvement is not assumed.']}
    write(out/'report.json', report)
    lines = ['FootballMind：共享语义 f 与配套 g/h 训练', '',
        f'f 训练 {len(f_ids)} 场，g 训练 {len(train_ids)} 场，重复开发留出 {len(held_ids)} 场；新增独立比赛 0。',
        f'f 参数 {fitted_f["parameter_count"]} 个；真实 CPU 拟合，GPU 不需要。',
        '先固定 f，重新计算全部收益标签，再训练 g 和排除整场教师监督的 h；正负零样本全部保留。',
        '', '主对照：最多读取2篇，每次成本0.001（log-loss研究单位，不是美元）。',
        '配套模型 | 策略 | 平均读取 | 读取带来的损失变化 | 最终loss | loss加查询成本']
    for r in primary:
        lines.append(f"{r['family']} | {r['policy']} | {r['expected_reads']:.2f} | {change(r['expected_gain'])} | {r['final_loss']:.9f} | {r['loss_plus_cost']:.9f}")
    lines += ['', 'f 单独评估（每场等权，对该场归档状态取平均；与完整策略最终loss口径不同）：']
    for scope, models in evaluations.items():
        for name, metric in models.items():
            lines.append(f"{scope} | {name} | {metric['matches']}场 | log loss {metric['loss']:.9f} | Brier {metric['brier']:.9f}")
    lines += ['', '新参数不自动替换现有线上模型。完整逐步轨迹见 policy-replays.json。',
              '重复开发集结果不能证明前瞻优势；非Leeds2场也是既有诊断材料。']
    (out/'training-summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(out), 'seconds': report['elapsed_seconds'], 'primary': primary,
                      'f_evaluation': evaluations, 'distribution': distribution}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        execute(args.output)
