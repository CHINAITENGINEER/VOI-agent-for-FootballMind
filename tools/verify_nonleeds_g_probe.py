"""Offline independent readback of the new fixed-model historical probe."""
import argparse
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import socket
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_reviewed_rich_f import read, write
from tools.reconstruct_panel_priors import no_network
from tools.train_state_candidate_g import before_snapshot
from tools.evaluate_nonleeds_g_probe import F_PATH, G_DIR, MATCH
from football_research.acquisition import entity_evidence_updates as f
from football_research.acquisition import structured_state_gain as g
from football_research.acquisition.entity_semantic_f import EntityEvidenceBank
from football_research.acquisition.historical_policy_comparison import rollout
from football_research.acquisition.match_conditioned_f import bind_prior


def change(value):
    if abs(value) < 1e-12:
        return '不变'
    return f'损失{"降低" if value > 0 else "增加"} {abs(value):.6f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.root/'sources', args.root/'evaluation'
    socket.socket.connect = no_network
    socket.getaddrinfo = no_network
    plan, report = read(out/'plan.json'), read(out/'report.json')
    ledger = read(source/'ledger.json')
    reviews = read(out/'semantic-review.json')
    spans = 0
    for doc in ledger:
        assert sha256((source/doc['raw_path']).read_bytes()).hexdigest() == doc['raw_sha256']
        body_bytes = (source/doc['body_path']).read_bytes()
        assert sha256(body_bytes).hexdigest() == doc['body_sha256']
        text = body_bytes.decode('utf-8')
        assert text == '\n'.join(doc['paragraphs'])
        for channel in ('title', 'body'):
            content = doc['title'] if channel == 'title' else text
            for assertion in reviews[doc['id']][channel]:
                for span in assertion['spans']:
                    assert content[span['start']:span['end']] == span['quote']
                    spans += 1
        from bs4 import BeautifulSoup
        soup = BeautifulSoup((source/doc['raw_path']).read_bytes(), 'html.parser')
        assert soup.select_one('meta[property="og:title"]')['content'] == doc['title']
        assert soup.select_one('meta[property="og:url"]')['content'].rstrip('/') == doc['url'].rstrip('/')
    model = f.load(F_PATH)
    models = {name: g.load(G_DIR/f'g-{name}.json') for name in ('baseline', 'combined')}
    bank = EntityEvidenceBank(read(out/'semantic-records.json'))
    rows, states = read(out/'prediction-input-rows.json'), read(out/'states.json')
    predictions, transitions = read(out/'fixed-f-predictions.json'), read(out/'transitions.json')
    prior, label = read(out/'prior-snapshot.json'), read(out/'independent-outcome.json')
    assert MATCH not in prior['training_match_ids']
    assert not set(prior['training_match_ids']) & set(plan['base_fit_exclusions'])
    assert not set([MATCH]) & set(model['head']['head']['encoder']['fit_match_ids'])
    assert all(MATCH not in m['fit_match_ids'] for m in models.values())
    # bind_prior independently replays the saved numerical prior parameters.
    bind_prior({'fixture': prior['fixture'], 'information_mode': 'historical_reconstruction',
                'decision_at': prior['as_of']}, prior)
    for sid, row in rows.items():
        assert row['outcome'] is None and row['result_available_at'] is None
        p = f.predict(model, row)
        assert np.array_equal(p, predictions[sid])
        for y in (0, 1, 2):
            altered = {**deepcopy(row), 'outcome': y, 'result_available_at': '2099-01-01T00:00:00Z'}
            assert np.array_equal(f.predict(model, altered), p)
            assert before_snapshot(altered) == before_snapshot(row)
        assert set(row['used_input_ids']) <= set(bank.records)
    graph = {}
    for t in transitions:
        sid, tid = t['state_id'], t['next_state_id']
        before, after = set(rows[sid]['used_input_ids']), set(rows[tid]['used_input_ids'])
        assert before < after and after-before == {t['body_input_id']}
        assert bank.records[t['body_input_id']]['input']['channel'] == 'body'
        query = g.build_query(t['query']['base']['public'], before_snapshot(rows[sid]))
        assert query == t['query']
        p, q = predictions[sid], predictions[tid]
        independent_gain = -np.log(p[label['outcome']])+np.log(q[label['outcome']])
        assert abs(independent_gain-t['signed_gain']) < 1e-12
        for name, m in models.items():
            assert g.estimate(m, query) == t['g_estimates'][name]
        graph.setdefault(sid, []).append(t)
    start = next(sid for sid,s in states.items() if s['read_count'] == 0 and s['title_count'] == 3)
    runs = read(out/'policy-replays.json')
    for record in runs:
        name = record['comparison_policy']
        rebuilt = rollout(graph, start, policy='g' if name in models else name,
            budget=record['budget'], cost=record['cost_loss_units'],
            scorer=lambda a, m=models.get(name): g.estimate(m, a['query']))
        assert rebuilt == {k:v for k,v in record.items() if k != 'comparison_policy'}
    for path, digest in plan['models'].items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest
    write(out/'verification.json', {'status': 'passed', 'raw_and_body_pairs': len(ledger),
        'exact_quote_spans': spans, 'fixed_f_predictions_recomputed': len(rows),
        'outcome_perturbation_checks': 3*len(rows), 'transition_and_query_checks': len(transitions),
        'policy_replays_recomputed': len(runs), 'f_g_models_unchanged': True,
        'test_network_calls': 0, 'prospective_training_eligibility': False})
    display = {'baseline': '基础版 g（已冻结）', 'combined': '结构特征组合版 g（上一轮研究候选）',
               'stop': '不查询', 'fixed': '固定顺序（action ID 排序）', 'random': '均匀随机读取的精确期望'}
    selected = [r for r in runs if r['budget'] == 2 and r['cost_loss_units'] == .001]
    lines = ['Chelsea–Arsenal：新增非 Leeds 比赛的冻结模型检查', '',
        '比赛：2025-11-30，understat:28900。独立归档赛果：1:1。',
        '本次新增三篇官方原文，审阅由当前 Codex 完成。没有调用 DeepSeek 或其他模型 API。',
        '固定 f 与 g 都没有用这场比赛训练；没有根据本场收益改参数或挑选模型。',
        '共 27 个可见信息状态、27 个单篇正文转移；它们来自同一场比赛，不能当 27 个独立样本。',
        '菜单是历史重建；正文当前抓取、历史版本未知。搜索曾露出赛果摘要，不称严格盲测或前瞻实测。',
        '', '来源与明确语义：',
        'C01：11月21日，教练明确说 Palmer 将缺席 Arsenal；Burnley 的其他球员状态没有移植为本场事实。',
        'C02：11月28日，明确 Palmer 可以出场；该标题已透露这一点，读取前状态保留该信息。',
        'C03：11月30日15:15Z，官方首发及替补名单，Palmer 在替补席；比赛16:30Z开始。',
        '名单仅作为 selection，未把入选等同完全康复；Essugo 参加部分训练不等同可以出场。',
        '', '初始状态：全部3条标题已见，正文尚未读取。',
        f"无文本重建先验（不是市场赔率）：{report['no_text_prior']}",
        f"标题进入f后的概率[主胜,平,客胜]：{report['title_only_prediction']}",
        '', '从同一初始状态单独读取每篇正文：']
    url_to_id = {d['url']: d['id'] for d in ledger}
    for action in report['initial_actions']:
        p, q = action['prediction_before'], action['prediction_after']
        lines += [f"{url_to_id[action['body_url']]} | 平局概率 {p[1]:.8f} → {q[1]:.8f} | {change(action['signed_gain'])}",
                  f"  基础g预估收益={action['g_estimates']['baseline']:.8f}；组合g预估收益={action['g_estimates']['combined']:.8f}"]
    lines += ['', '预先声明的一个运行条件：最多读2篇，每篇成本0.001损失单位（不是真实美元）。',
              '策略 | 读取篇数 | 预测损失变化 | 加入查询成本后的变化']
    for r in selected:
        lines.append(f"{display[r['comparison_policy']]} | {r['expected_reads']:.2f} | {change(r['expected_gain'])} | {change(r['expected_net'])}")
    lines += ['', 'g的实际动作路径（每步选择只读query，不读取after/赛果/收益）：']
    for r in selected:
        if r['comparison_policy'] not in models:
            continue
        for path in r['paths']:
            urls = [next(t['body_url'] for t in graph[s['state_id']] if t['action_id'] == s['action_id']) for s in path['steps']]
            lines.append(display[r['comparison_policy']]+': '+' → '.join(url_to_id[u] for u in urls)+f"；实际读取{path['reads']}篇。")
    lines += ['', f"27个转移：损失降低{report['gains']['positive']}条、损失增加{report['gains']['negative']}条、不变{report['gains']['zero']}条。",
        f"收益预测MSE：基础g={report['g_mse']['baseline']:.10f}；组合g={report['g_mse']['combined']:.10f}；永远预测0={report['zero_mse']:.10f}。",
        '', '结论与下一步：',
        '基础g在该条件下选择首发后停止，优于固定顺序和随机；组合版在这场反而更差，不能据此推广升级。',
        '一场实现收益不等于条件期望正确，也不能证明泛化优势。所有成本/预算共45条策略回放保存在policy-replays.json。',
        '旧f仍保留明确unavailable→available两次陈述为冲突；没有为了本场表现修改规则。',
        '标题与正文措辞不同的相同状态在当前f里仍可能产生词汇差异；因此C02收益不是纯新增事实价值。',
        '保留全部正负转移作为下一批历史研究训练候选，本轮不回训；先再补不同比赛，按比赛划分训练和验证。']
    (out/'研究结果.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(out/'stage-decision.json', {'status': 'completed_nonleeds_frozen_model_probe',
        'retain_baseline_g': True, 'promote_combined_g': False, 'f_or_g_retrained': False,
        'new_match_for_g_evaluation': MATCH, 'rows_available_for_future_research_training': len(transitions),
        'counts_as_independent_matches': 1, 'prospective_B_rows': 0,
        'next': 'Collect other independent non-Leeds matches with this reviewed body-to-fixed-f protocol before enlarging the training cohort.'})
    print(f'PASS: 3 sources, {spans} quote spans, {len(rows)} predictions, {len(transitions)} transitions, {len(runs)} policy replays')


if __name__ == '__main__':
    main()
