"""Read-back experiment checks and human-readable research decision."""
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.train_policy_return_continuation_g import OLD, PREVIOUS, SOURCE, read, write, mean_menu_choice
from tools.reconstruct_panel_priors import no_network
from football_research.acquisition import policy_return_continuation_g as new
from football_research.acquisition import learned_continuation_g as old
from football_research.acquisition import state_candidate_gain as g1
from football_research.acquisition.historical_policy_comparison import rollout


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    socket.socket.connect = no_network; socket.getaddrinfo = no_network
    plan = read(out/'plan.json'); model = new.load(out/'g-policy-return.json'); base = model['immediate_model']
    assert base == g1.load(PREVIOUS/'g-state-candidate.json')
    old_model = old.load(OLD/'g-two-step.json')
    for path, digest in plan['source_bindings'].items(): assert sha256((ROOT/path).read_bytes()).hexdigest() == digest
    queries = {(q['state_id'], q['action_id']): q['query'] for q in read(PREVIOUS/'prequery-feature-audit.json')['queries']}
    graph = defaultdict(list)
    for t in read(SOURCE/'transitions.json'):
        graph[t['state_id']].append({**t, 'query': queries[t['state_id'], t['action_id']]})
    rows = read(out/'training-targets.json')
    teachers = {mid: g1.load(OLD/'crossfit-teachers'/f"leave-out-{mid.replace(':','-')}.json") for mid in plan['train_ids']}
    for r in rows:
        teacher = teachers[r['physical_match_id']]
        assert set(teacher['fit_match_ids']) == set(plan['train_ids'])-{r['physical_match_id']}
        assert not set(teacher['fit_match_ids']) & set(plan['holdout_ids'])
        following = graph.get(r['next_state_id'], [])
        assert all(e['physical_match_id'] == r['physical_match_id'] and e['g_split'] == 'train' for e in following)
        target = new.successor_target(teacher, following, r['packet']['cost_loss_units'])
        assert target == r['successor_policy_label'] and target['continuation_target'] == r['continuation_target']
    refit = new.fit(rows, base, alpha=.1)
    assert np.allclose(refit['coefficients'], model['coefficients'], atol=1e-15, rtol=1e-12)
    assert abs(refit['intercept']-model['intercept']) < 1e-15
    selectors = {'old_h': lambda m,l,c: old.choose(old_model,m,l,c),
                 'signed_return_h': lambda m,l,c: new.choose(model,m,l,c),
                 'old_h_menu_mean': lambda m,l,c: mean_menu_choice(old,old_model,m,l,c),
                 'signed_return_h_menu_mean': lambda m,l,c: mean_menu_choice(new,model,m,l,c)}
    states = read(SOURCE/'states.json')
    starts = {mid: max(((sid,s) for sid,s in states.items() if s['physical_match_id'] == mid and s['read_count'] == 0),
                      key=lambda p:p[1]['title_count'])[0] for mid in plan['holdout_ids']}
    runs = read(out/'policy-replays.json')
    for r in runs:
        replay = rollout(graph, starts[r['physical_match_id']], policy=r['policy'], budget=r['budget'],
                         cost=r['cost_loss_units'], scorer=lambda p:g1.estimate(base,p['query']),
                         menu_selector=selectors.get(r['comparison_policy']))
        assert all(replay[k] == r[k] for k in replay)
    diagnostics = read(out/'continuation-diagnostics.json')
    for d in diagnostics:
        edge = next(e for e in graph[d['state_id']] if e['action_id'] == d['action_id'])
        assert new.successor_target(base,graph.get(edge['next_state_id'],[]),d['cost']) == d['target']
    decision = {'status':'research_variant_trained_not_promoted', 'reason':
        'No policy improvement; signed continuation regression also worse than zero on the reused development holdout.',
        'target_correction':'negative realized successor returns retained, selected by out-of-match teacher rather than outcome oracle',
        'menu_ablation':'both h variants have identical paths after replacing per-action h by current-menu mean in all 45 cells each',
        'interpretation':'observed prior improvement can be reproduced by state/menu-dependent stopping bonus; action-specific lookahead not demonstrated',
        'next_priority':'improve successor-policy supervision and independent multi-team sequence coverage; do not increase lookahead depth or tune on these five matches',
        'data_selection_rule':'freeze matches and chronological candidate menus without looking at outcome or gain; retain zero/negative and failure cases',
        'production_ready':False,'online_selection_allowed':False}
    write(out/'research-decision.json', decision)
    lines = ['本轮结论与下一步', '',
        '已训练新的 h，但未取得提升，不替换上一版本。g1/f/划分和特征维度保持不变。',
        '旧目标：下一步 g 预测净收益的非负最大值。新目标：排除当前比赛的 g 先选动作，再记录该动作实际净收益；允许负值。',
        '318个成本条件目标：6正、17负、295零；来自106条转移、8场训练比赛，不是318场。',
        '成本0.001下：106个目标中teacher继续14次、全部实际净收益为负；其余92次停止。',
        '所以这批目标教给 h 的主要信号是这个后续策略不值得继续，不能解释为所有信息没有价值。',
        '预算2/成本0.001：旧h计入成本后平均损失降低0.002369866；新h平均损失增加0.000800000，与一步g相同。',
        '相同成本下raw h对固定后续策略真实净收益的比赛等权MSE：旧1.5744055e-7，新1.2567705e-6，零预测7.0370370e-8。',
        '两版本逐动作h替换成当前菜单均值后，各45个场景动作路径完全相同。这里只证明当前轨迹不依赖h的动作间差异。',
        '下一步应优先补足独立多球队的状态转移和改进g1对后续动作的估计；不继续叠深度或用5场结果搜阈值。',
        '按球队/比赛/时间和已见来源类型取样，不能按事后收益挑正例；保留零收益、负收益、失败和上下文更新。',
        '当前为历史重建研究，尚无开放目录发现带来的新动作链条；没有新增网络或API。']
    (out/'结论与后续.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verification = {'source_bindings_unchanged':len(plan['source_bindings']), 'immediate_model_unchanged':True,
        'targets_recomputed':len(rows), 'whole_match_exclusion_teachers':len(teachers),
        'model_refit_matches':True, 'exact_policy_replays':len(runs), 'successor_diagnostic_labels_recomputed':len(diagnostics),
        'network':0,'api_calls':0}
    write(out/'readback-verification.json', verification)
    import json
    print(json.dumps(verification))


if __name__ == '__main__': main()
