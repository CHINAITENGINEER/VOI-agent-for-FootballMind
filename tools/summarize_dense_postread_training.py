"""Readable result and state-dependence diagnostics; never selects another model."""
from collections import defaultdict, Counter
import argparse
import json
from pathlib import Path
import numpy as np


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def main():
    ap = argparse.ArgumentParser()
    for key in ('root', 'old', 'new'):
        ap.add_argument('--'+key, type=Path, required=True)
    a = ap.parse_args()
    report = json.loads((a.new/'report.json').read_text(encoding='utf-8'))
    old = json.loads((a.old/'report.json').read_text(encoding='utf-8'))
    train = rows(a.new/'training-transitions.jsonl')
    by_candidate, by_depth = defaultdict(list), defaultdict(list)
    for r in train:
        s = r['selector_state']
        by_candidate[(r['match_id'], s['candidate_id'])].append(r)
        by_depth[len(s['f_adopted_ids'])].append(r)
    sign_changes = []
    articles = {r['article_id']: r for r in rows(a.root/'news/articles.jsonl')}
    for (mid, aid), rr in by_candidate.items():
        gains = [r['label']['signed_gain'] for r in rr]
        if min(gains) < -1e-12 and max(gains) > 1e-12:
            lo = min(rr, key=lambda r:r['label']['signed_gain'])
            hi = max(rr, key=lambda r:r['label']['signed_gain'])
            sign_changes.append({'match_id':mid, 'candidate_id':aid, 'title':articles[aid]['title'],
                'minimum_gain':min(gains), 'maximum_gain':max(gains),
                'minimum_state':lo['selector_state']['f_adopted_ids'],
                'maximum_state':hi['selector_state']['f_adopted_ids']})
    depth = {}
    for n, rr in sorted(by_depth.items()):
        per_match = defaultdict(list)
        for r in rr:per_match[r['match_id']].append(r['label']['signed_gain'])
        depth[n] = {'transitions':len(rr), 'matches':len(per_match),
            'mean_gain_equal_match_weight':float(np.mean([np.mean(v) for v in per_match.values()])),
            'positive':sum(r['label']['signed_gain'] > 1e-12 for r in rr),
            'negative':sum(r['label']['signed_gain'] < -1e-12 for r in rr)}
    def selected(path):
        return {r['match_id']:r for r in rows(path/'per-match-comparison.jsonl')
                if r['split']=='test' and r['mode']=='body_state' and r['budget']==3}
    before, after = selected(a.old), selected(a.new)
    changes = [{'match_id':mid, 'old_loss':before[mid]['loss']['g'], 'new_loss':r['loss']['g'],
                'loss_increase':r['loss']['g']-before[mid]['loss']['g'],
                'old_adoptions':before[mid]['g_adopted_count'], 'new_adoptions':r['g_adopted_count']}
               for mid,r in after.items()]
    changes.sort(key=lambda r:r['loss_increase'], reverse=True)
    diag = {'training_only_state_diagnostics':{'by_adopted_count':depth,
        'candidate_match_pairs':len(by_candidate), 'sign_changing_pairs':len(sign_changes),
        'sign_changing_matches':len({r['match_id'] for r in sign_changes}),
        'sign_changes':sign_changes}, 'selected_models_test_changes':changes,
        'interpretation':'Realized gain under fixed research f; sign changes are not proof of expected value or causality.',
        'model_or_threshold_changed_by_diagnostics':False}
    (a.new/'state-and-error-diagnostics.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['全文＋状态 g：补密训练候选后的结果', '',
        '本轮目的：增加同场不同证据状态，检验候选池密度对 g 的影响。',
        '固定 f、旧向量、f 训练集、验证/测试候选与标签均保持一致。',
        '训练比赛仍为59场；新增117篇正文，65条关联进入训练候选池。',
        '候选关联83→148，多候选比赛20→45，训练转移151→750（344正收益、406负收益）。',
        '全文分块编码：384维MiniLM，正文未静默截成标题；新正文330个块。',
        '28个配置均已训练；验证选择的全文＋状态模型为16隐层MLP，24865个参数，alpha=0.1。',
        '参数选择使用验证集；测试集此前已在研究中看过，不宣称最终盲测。', '',
        '每场最多交给 f 三篇；log loss 越小越好；本轮查询成本罚项为0。',
        '策略\t验证24场\t测试49场']
    labels={'no_information':'不加入正文','all_information':'全部加入（不受三篇上限约束）',
            'fixed_order':'固定顺序','random_exact':'相同三篇上限随机（精确期望）'}
    for key,title in labels.items():
        vals=[report['comparisons'][s]['body_state']['budgets']['3']['mean_log_loss'][key] for s in ('validation','test')]
        lines.append(f'{title}\t{vals[0]:.6f}\t{vals[1]:.6f}')
    for rr,title in ((old,'旧全文＋状态 g'),(report,'新全文＋状态 g')):
        vals=[rr['comparisons'][s]['body_state']['budgets']['3']['mean_log_loss']['g'] for s in ('validation','test')]
        lines.append(f'{title}\t{vals[0]:.6f}\t{vals[1]:.6f}')
    lines += ['', '结论：验证仅微幅改善，测试退步；没有证明新 g 优于固定顺序或随机。',
        '相同ridge(alpha=10)配置重训也退步，不能将问题全归因于模型种类变化。',
        f'训练中有{len(sign_changes)}/{len(by_candidate)}个比赛—候选对的实际收益随已选状态发生正负翻转，覆盖{len({r["match_id"] for r in sign_changes})}场。',
        '这支持保留状态相关任务，但不证明当前模型已学会它。',
        '新增同场状态不是新增独立样本；日期/球队关联仍为启发式，未逐篇语义审阅。',
        '历史正文版本假设保留，不当作前瞻观测；没有上线替换。',
        '下一步：训练集内按比赛做时间前向验证，诊断状态深度/收益标签分布与策略目标；扩独立比赛及双方覆盖。',
        '不因本轮测试结果重新挑模型或停止阈值。', '', '最大退步场次（仅错误分析）：']
    lines += [f'{r["match_id"]}：损失增加{r["loss_increase"]:.6f}；采用篇数{r["old_adoptions"]}→{r["new_adoptions"]}' for r in changes[:5]]
    (a.new/'训练结果说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'state_depths':depth,'sign_changing_pairs':len(sign_changes),
        'sign_changing_matches':len({r['match_id'] for r in sign_changes}),
        'test_better':sum(r['loss_increase'] < -1e-12 for r in changes),
        'test_worse':sum(r['loss_increase'] > 1e-12 for r in changes),
        'test_unchanged':sum(abs(r['loss_increase'])<=1e-12 for r in changes)},ensure_ascii=False))


if __name__=='__main__':main()
