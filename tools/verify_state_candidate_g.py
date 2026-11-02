"""Read back a fixed experiment; materialize human-readable decision evidence."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import socket
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.train_state_candidate_g import SOURCE, EVENTS, before_snapshot, read
from tools.train_reviewed_rich_f import write
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import state_candidate_gain as g
from football_research.acquisition import historical_gain_head as old_g
from football_research.acquisition import entity_evidence_updates as f
from football_research.acquisition.historical_policy_comparison import rollout


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args(); out = args.output
    socket.socket.connect = no_network; socket.getaddrinfo = no_network
    plan = read(out/'plan.json'); report = read(out/'report.json')
    for path, digest in plan['source_bindings'].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest
    ts = read(SOURCE/'transitions.json'); states = read(SOURCE/'states.json'); rows = read(EVENTS)
    keyed = {(r['physical_match_id'], tuple(sorted(r['used_input_ids']))): r for r in rows}
    rows_by_state = {sid: keyed[s['physical_match_id'], tuple(s['used_input_ids'])] for sid, s in states.items()}
    model = g.load(out/'g-state-candidate.json'); frozen_f = f.load(SOURCE/'f-labeler.json')
    assert set(model['fit_match_ids']) == set(plan['g_training_ids'])
    assert not set(model['fit_match_ids']) & set(plan['g_holdout_ids'])
    assert not set(frozen_f['head']['head']['encoder']['fit_match_ids']) & (set(plan['g_training_ids']) | set(plan['g_holdout_ids']))
    audit = read(out/'prequery-feature-audit.json')
    queries = {(q['state_id'], q['action_id']): q['query'] for q in audit['queries']}
    predictions = {sid: f.predict(frozen_f, row).tolist() for sid, row in rows_by_state.items()}
    graph = defaultdict(list)
    for t in ts:
        q = g.build_query(t['query'], before_snapshot(rows_by_state[t['state_id']]))
        assert q == queries[t['state_id'], t['action_id']]
        assert np.allclose(predictions[t['state_id']], t['prediction_before'], atol=1e-14, rtol=0)
        assert np.allclose(predictions[t['next_state_id']], t['prediction_after'], atol=1e-14, rtol=0)
        gain = old_g.signed_gain(predictions[t['state_id']], predictions[t['next_state_id']], t['outcome'])
        assert abs(gain-t['signed_gain']) < 1e-14
        graph[t['state_id']].append({**t, 'query': q})
    scorer = lambda public: g.estimate(model, public['query'])
    old = old_g.load(SOURCE/'g-visible_title.json')
    old_scorer = lambda public: old_g.estimate(old, public['query']['public'])
    starts = {mid: max(((sid, s) for sid, s in states.items() if s['physical_match_id'] == mid and s['read_count'] == 0),
                      key=lambda pair: pair[1]['title_count'])[0] for mid in plan['g_holdout_ids']}
    runs = read(out/'policy-replays.json')
    for run in runs:
        sc = scorer if run['comparison_policy'].startswith('state_g') else old_scorer if run['comparison_policy'].startswith('old_g') else None
        again = rollout(graph, starts[run['physical_match_id']], policy=run['policy'], budget=run['budget'], cost=run['cost_loss_units'], scorer=sc)
        assert all(again[k] == run[k] for k in again)
    cases = []
    for run in runs:
        if run['budget'] != 3 or run['cost_loss_units'] != 0 or run['comparison_policy'] != 'state_g':
            continue
        path = run['paths'][0]; taken = []
        for s in path['steps']:
            edge = next(t for t in graph[s['state_id']] if t['action_id'] == s['action_id'])
            taken.append({**s, 'title': edge['query']['public']['title'], 'body_url': edge['body_url']})
        remaining = [{'action_id': t['action_id'], 'title': t['query']['public']['title'], 'body_url': t['body_url'],
                      'estimated_gain': scorer({'action_id': t['action_id'], 'query': t['query']}),
                      'retrospective_realized_gain_if_read_next': t['signed_gain']}
                     for t in graph.get(path['end_state_id'], [])]
        cases.append({'physical_match_id': run['physical_match_id'], 'read': taken, 'remaining_at_stop': remaining,
                      'stop_reason': path['stop_reason'], 'retrospective_unselected_gain_not_available_to_policy': True})
    distribution = {}
    for split in ('train', 'holdout'):
        ids = plan['g_training_ids'] if split == 'train' else plan['g_holdout_ids']
        first = []
        for mid in ids:
            start = max(((sid, s) for sid, s in states.items() if s['physical_match_id'] == mid and s['read_count'] == 0), key=lambda pair: pair[1]['title_count'])[0]
            first.extend({'physical_match_id': mid, 'title': t['query']['public']['title'], 'gain': t['signed_gain']}
                         for t in graph[start] if t['query']['relations']['title_injury'])
        gains = [t['gain'] for t in first]
        distribution[split] = {'first_read_injury_title_rows': first, 'count': len(first),
                               'mean_gain': float(np.mean(gains)), 'positive': sum(v > 1e-12 for v in gains),
                               'negative': sum(v < -1e-12 for v in gains)}
    write(out/'decision-diagnosis.json', {'cases': cases, 'injury_title_distribution': distribution,
        'interpretation': 'Descriptive small-sample association; neither a causal diagnosis nor a title-class value rule.'})
    lines = ['FootballMind：查询前状态关联 g 与预算策略验收', '',
        '结论：本轮已训练一个新 g 并完成策略对照，但新 g 没有改善结果，不能晋升。',
        '固定原 f、220 条收益标签和比赛划分：8 场/106 条训练，5 场/114 条留出。',
        '新增15个查询前特征；只用读取前已有证据与已曝光标题，不读取候选正文或赛果。',
        '保留零收益、负收益。未联网，未调用 API，未改变正式 f/g。', '',
        '一、收益估计 MSE（按比赛等权；越小越好）']
    for name, result in report['evaluation'].items():
        lines.append(f"{name}: {result['match_weighted_mse']:.12f}")
    lines += ['', '二、同一最多查询预算下的策略结果',
        'gain 是平均 log loss 降低量，正值更好。cost 是假设损失单位，不是美元。',
        'random_exact 穷举所有均匀随机路径求期望，不是抽到一次幸运结果。',
        'g_forced 强制按排序读到预算上限，用来区分排序错误与停止错误。',
        '预算 | 单次cost | 策略 | 平均读取 | 平均gain | 平均net']
    for r in report['policy_summary']:
        lines.append(f"{r['budget']} | {r['cost_loss_units']:.3f} | {r['policy']} | {r['reads']:.2f} | {r['gain']:+.9f} | {r['net']:+.9f}")
    lines += ['', '三、逐场停止诊断（预算3、成本0；未选收益只用于事后分析）']
    for case in cases:
        lines.append(f"比赛 {case['physical_match_id']}；停止原因 {case['stop_reason']}")
        for step in case['read']:
            lines.append(f"  已读：{step['title']}；g预计 {step['estimated_gain']:+.8f}；实际 {step['signed_gain']:+.8f}")
        for step in case['remaining_at_stop']:
            lines.append(f"  停止未读：{step['title']}；g预计 {step['estimated_gain']:+.8f}；事后重建收益 {step['retrospective_realized_gain_if_read_next']:+.8f}")
    lines += ['', '四、现有材料对新增特征的覆盖',
        '本版状态关联特征只统计已读正文；首次读取时这些状态特征为零，标题特征仍有效。',
        '220条中 pending_entity_title_overlap=0，未出现“已知待决球员名字直接命中候选标题”的训练实例。',
        '这限制了本实验验证主动追问的能力；不代表实际信息环境没有这种价值。']
    for split, d in distribution.items():
        lines.append(f"{split} 初始状态伤情标题候选 {d['count']} 条，正 {d['positive']} / 负 {d['negative']}，平均实际收益 {d['mean_gain']:+.9f}")
    lines += ['上述差异仅是小样本描述，不能直接归因为某类文章或推广为伤情新闻无价值。', '',
        '五、下一步决策',
        '保留旧 g 作为研究基线，本轮新模型保留为未胜出的实验；不继续围绕这5场调参。',
        '新增独立多球队训练材料，按查询前状态/候选类型覆盖选择，不按赛果或收益挑选。',
        '优先覆盖：已知出场待定→后续阵容/伤情；已知训练→出场结论；重复信息与无关正文。',
        '保留全部结果正/零/负及失败，使用同一冻结 f 形成标签；预留未使用的新比赛做验证。',
        '再次比较同预算随机、固定、停止和g；先证明排序优势，再证明停止带来净收益。', '',
        '边界：这是已反复使用的历史重建开发集，13场 g 目标都涉及 Leeds，非新盲测。',
        'g 的价值针对固定研究 f；未证明真实开放搜索、实时收益、真实成本或跨球队泛化。']
    with (out/'阶段结果.txt').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')
    write(out/'verification.json', {'source_bindings_unchanged': len(plan['source_bindings']),
        'queries_rebuilt_from_before': len(ts), 'same_f_predictions_and_gain_recomputed': len(ts),
        'policy_runs_exactly_replayed': len(runs), 'f_g_fit_holdout_matches_disjoint': True,
        'query_body_outcome_exclusion': True, 'network': 0, 'provider_calls': 0,
        'formal_runtime_enabled': False})
    print(json.dumps(read(out/'verification.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
