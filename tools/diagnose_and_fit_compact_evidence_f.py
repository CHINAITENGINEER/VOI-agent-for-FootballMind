"""A-only coverage and low-capacity f ablation. No promotion or g labels."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition.compact_evidence_f import VERSION, STATUSES, REGULARIZATIONS, fit, predict
from football_research.acquisition.entity_semantic_f import EntityEvidenceBank, vector, _modeled
from football_research.acquisition.historical_context_training import prepare_historical_cases, _discovery_panel
from football_research.acquisition.match_conditioned_f import bind_prior
from football_research.acquisition.semantic_f import FEATURES
from football_research.acquisition.state import parse_time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def metrics(beta, rows):
    losses, briers = [], []
    for row in rows:
        p = predict(beta, row['state'])
        losses.append(float(-np.log(p[row['outcome']])))
        briers.append(float(np.sum((p - np.eye(3)[row['outcome']]) ** 2)))
    return {'log_loss': float(np.mean(losses)), 'brier_sum_three_classes': float(np.mean(briers))}


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = read(source / 'protocol.json')
    model = read(source / 'model.json')
    a_ids = protocol['splits']['A']
    paths = {k: Path(protocol['source_bindings'][k]['path']) for k in ('cases', 'records', 'priors')}
    # Deserialization reads the archive; only A objects/labels are used below.
    raw_cases = [c for c in read(paths['cases']) if c['fixture']['physical_match_id'] in a_ids]
    records = [r for r in read(paths['records']) if r['input']['fixture']['physical_match_id'] in a_ids]
    snapshots = {k: v for k, v in read(paths['priors']).items() if k in a_ids}
    sources = {k: {'path': str(p), 'sha256': sha256(p.read_bytes()).hexdigest()} for k, p in paths.items()}
    write(output / 'experiment-plan.json', {
        'schema': VERSION, 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'purpose': 'retrospective_A_development_capacity_ablation',
        'A_ids': a_ids, 'source_bindings': sources,
        'input': 'four availability binary presence differences home-minus-away; title/body union',
        'logit_residual': '[x @ beta, 0, -x @ beta]', 'parameter_count': 4,
        'regularizations': REGULARIZATIONS,
        'objective': 'equal_physical_match_mean_state_log_loss_plus_half_lambda_l2',
        'folds': 'reuse_original_A_purged_forward_folds',
        'selection': 'original_one_paired_SE_rule_in_declared_regularization_order',
        'fresh_holdout_claim': False, 'B_C_used_for_fit_or_evaluation': False,
        'new_semantics': False, 'g_training': False, 'runtime_activation': False,
        'limitations': ['A was previously used for development; not an independent confirmatory test.',
                       'Historical source versions and availability are assumptions, not prospective observations.',
                       'Four-feature ablation does not represent weak training-participation evidence.']})
    cases, _ = prepare_historical_cases(raw_cases)
    bank = EntityEvidenceBank(records)
    rows, panels = [], {}
    for case in cases:
        mid = case['fixture']['physical_match_id']
        panels[mid] = {}
        for dm, am, discovered, acquired, state in _discovery_panel(case):
            projected = bank.project(bind_prior(state, snapshots[mid]))
            row = {'physical_match_id': mid, 'outcome': case['outcome'],
                   'result_available_at': case['result_available_at'], 'state': projected}
            rows.append(row); panels[mid][dm, am] = row
    by_match = {mid: [r for r in rows if r['physical_match_id'] == mid] for mid in a_ids}
    matrix = np.asarray([vector(r['state']) for r in rows])
    match_feature_support = {f: sum(any(r['state']['title'][i] or r['state']['novel_body'][i]
                               for r in by_match[mid]) for mid in a_ids) for i, f in enumerate(FEATURES)}
    record_by_key = {(r['input']['fixture']['physical_match_id'], r['input']['url'], r['input']['channel']): r
                     for r in records}
    samples = []
    for case in cases:
        mid = case['fixture']['physical_match_id']
        for index, doc in enumerate(case['documents']):
            rec = record_by_key[mid, doc['url'], 'body']
            claims = rec['response']['claims']
            samples.append({'physical_match_id': mid, 'document_index': index,
                'fixture': case['fixture'], 'url': doc['url'], 'title': doc['title'],
                'body': rec['input']['text'], 'source_published_at': doc.get('source_published_at'),
                'captured_at': doc.get('captured_at'), 'available_at': doc.get('available_at'),
                'information_mode': 'historical_reconstruction',
                'review_origin': 'existing_Codex_annotations_not_new_independent_review',
                'review_provenance': rec['provenance'], 'claims': claims,
                'modeled_claim_count': sum(_modeled(c) for c in claims),
                'unmodeled_claim_count': sum(not _modeled(c) for c in claims)})
    with (output / 'semantic-samples.jsonl').open('x', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    text = ['A组真实正文语义样本。沿用既有Codex标注，本轮未独立重标；历史重建，不是前瞻轨迹。',
            '没有按是否产生正收益过滤。以下不附赛果，便于后续语义审阅。\n']
    for s in samples:
        text.extend([f"比赛 {s['physical_match_id']} | {s['fixture']['home_name']} - {s['fixture']['away_name']}",
                     f"标题：{s['title']}\nURL：{s['url']}",
                     f"可建模陈述 {s['modeled_claim_count']}；未建模 {s['unmodeled_claim_count']}", s['body']])
        for c in s['claims']:
            text.append(f"标注：{c['entity']} | {c['scope']} | {c['kind']}/{c['value']} | {c['statement']}")
            text.extend('原文依据：' + quote for quote in c['quotes'])
        text.append('\n' + '=' * 60 + '\n')
    (output / 'semantic-samples.txt').write_text('\n'.join(text), encoding='utf-8')
    coverage = {'physical_matches': len(a_ids), 'states': len(rows), 'body_documents': len(samples),
                'matches_by_team': dict(Counter(t for c in cases for t in
                    (c['fixture']['home_name'], c['fixture']['away_name']))),
                'publishers': dict(Counter(d.get('publisher', 'unknown') for c in cases for d in c['documents'])),
                'outcome_counts_A': dict(Counter(c['outcome'] for c in cases)),
                'documents_with_modeled_claims': sum(s['modeled_claim_count'] > 0 for s in samples),
                'documents_without_modeled_claims': sum(s['modeled_claim_count'] == 0 for s in samples),
                'body_claims': sum(len(s['claims']) for s in samples),
                'modeled_body_claims': sum(s['modeled_claim_count'] for s in samples),
                'feature_match_support': match_feature_support,
                'original_feature_columns': matrix.shape[1],
                'original_nonzero_columns': int(np.count_nonzero(np.any(matrix != 0, axis=0))),
                'original_stored_coefficients': int(np.asarray(model['coefficients']).size),
                'new_coefficients': 4}
    write(output / 'coverage.json', coverage)
    folds, score_lists = [], [[] for _ in REGULARIZATIONS]
    for fold in model['metadata']['selection']['folds']:
        train, valid = fold['training_ids'], fold['validation_ids']
        if set(train) & set(valid):
            raise ValueError('Fold overlap')
        boundary = min(parse_time(by_match[mid][0]['state']['decision_at']) for mid in valid)
        if any(parse_time(by_match[mid][0]['result_available_at']) >= boundary for mid in train):
            raise ValueError('Fold result timing leakage under declared reconstruction clocks')
        candidates = []
        for i, lam in enumerate(REGULARIZATIONS):
            beta, fit_info = fit([r for mid in train for r in by_match[mid]], lam)
            scores = {mid: metrics(beta, by_match[mid]) for mid in valid}
            score_lists[i].extend(v['log_loss'] for v in scores.values())
            candidates.append({'regularization': lam, 'beta': beta, 'fit': fit_info, 'matches': scores})
        folds.append({'train': train, 'validation': valid, 'candidates': candidates})
    scores = np.asarray(score_lists); means = scores.mean(axis=1); best = int(means.argmin())
    se = (scores-scores[best]).std(axis=1, ddof=1) / np.sqrt(scores.shape[1])
    chosen = next(i for i in range(len(means)) if means[i] <= means[best] + se[i] + 1e-12)
    candidate_results = []
    for i, lam in enumerate(REGULARIZATIONS):
        brier = [v['brier_sum_three_classes'] for f in folds for v in f['candidates'][i]['matches'].values()]
        candidate_results.append({'regularization': lam, 'mean_match_log_loss': float(means[i]),
            'mean_match_brier': float(np.mean(brier)), 'paired_SE_to_best': float(se[i]),
            'mean_log_loss_gain_vs_base': float(means[0]-means[i]),
            'matches_better_than_base': int(np.sum(scores[i] < scores[0]-1e-12)),
            'matches_worse_than_base': int(np.sum(scores[i] > scores[0]+1e-12)),
            'matches_equal_to_base': int(np.sum(abs(scores[i]-scores[0]) <= 1e-12))})
    best_nonzero = min(range(1, len(means)), key=lambda i: means[i])
    beta, fit_info = fit(rows, REGULARIZATIONS[best_nonzero])
    write(output / 'candidate-model.json', {'schema': VERSION, 'statuses': STATUSES,
        'coefficients': beta, 'regularization': REGULARIZATIONS[best_nonzero], 'fit': fit_info,
        'role': 'A_only_best_nonzero_research_candidate_not_promoted',
        'production_ready': False, 'g_training_enabled': False,
        'training_match_ids': a_ids, 'selection_index': chosen,
        'parameterization': '[x @ beta, 0, -x @ beta]'})
    transitions = []
    for s in samples:
        mid, index = s['physical_match_id'], s['document_index']
        mask = (1 << len(next(c for c in cases if c['fixture']['physical_match_id'] == mid)['documents'])) - 1
        pre, post = panels[mid][mask, 0]['state'], panels[mid][mask, 1 << index]['state']
        p, q = predict(beta, pre), predict(beta, post)
        transitions.append({'physical_match_id': mid, 'url': s['url'], 'title': s['title'],
            'before': p.tolist(), 'after': q.tolist(), 'L1_change': float(abs(q-p).sum()),
            'modeled_claim_count': s['modeled_claim_count'], 'used_input_ids_after': post['used_input_ids'],
            'new_body_input_ids': sorted(set(post['used_input_ids']) - set(pre['used_input_ids'])),
            'scope': 'in_sample_A_response_diagnostic_not_generalization',
            'gain': None, 'g_training_eligible': False})
    write(output / 'body-response-diagnostics.json', transitions)
    report = {'schema': 'compact-evidence-f-development-report-v1',
        'coverage': coverage, 'folds': folds, 'candidates': candidate_results,
        'original_f_candidates': model['metadata']['selection']['candidates'],
        'independent_validation_matches': scores.shape[1],
        'original_base_reproduced': bool(abs(means[0] - model['metadata']['selection']['candidates'][0]['mean_match_log_loss']) < 1e-10),
        'one_SE_selected_regularization': REGULARIZATIONS[chosen],
        'nonzero_candidate_fit_completed': True, 'new_candidate_beta': beta,
        'A_body_responses_changed': sum(r['L1_change'] > 1e-12 for r in transitions),
        'A_body_responses_unchanged': sum(r['L1_change'] <= 1e-12 for r in transitions),
        'selection_note': 'Retrospective A-only development; no fresh heldout success claimed.',
        'empirical_g_training': False, 'formal_f_changed': False, 'runtime_enabled': False,
        'HTTP': 0, 'provider_calls': 0}
    write(output / 'report.json', report)
    lines = ['本轮 f 数据诊断与低参数训练实验',
             f"A组：{len(a_ids)}场独立比赛，{len(samples)}篇正文，{len(rows)}个枚举状态。状态数不等于比赛数。",
             '15场均涉及Leeds，36篇均来自Leeds官网；不能外推为通用足球文本效果。',
             f"已有可建模正文：{coverage['documents_with_modeled_claims']}篇；没有可建模陈述：{coverage['documents_without_modeled_claims']}篇。",
             f"旧表示192列、576个存储系数；本次仅4个系数，不新增语义事实或强制效果方向。",
             '本轮重新拟合A组时间顺序折；基础概率不重训；B/C未用于拟合或评价。',
             'A组此前已用于研发，本次结果属于重复使用开发集的探索，不是新独立验证。',
             '比较（各比赛先平均其状态，再在比赛之间平均）：']
    lines.extend(f"lambda={r['regularization']}：log loss={r['mean_match_log_loss']:.9f}；Brier={r['mean_match_brier']:.9f}；对基础改善={r['mean_log_loss_gain_vs_base']:.9f}；改善/变差/不变比赛={r['matches_better_than_base']}/{r['matches_worse_than_base']}/{r['matches_equal_to_base']}" for r in candidate_results)
    lines.extend([f"沿用one-SE选择规则选中lambda={REGULARIZATIONS[chosen]}（None为基础模型）。",
                  f"非零研究候选已实际拟合并保存，beta={beta}。不替换正式f，不启用g。",
                  f"36篇逐篇正文响应（全A拟合后的样本内诊断）：变化{report['A_body_responses_changed']}，不变{report['A_body_responses_unchanged']}。不作为预测能力证据。",
                  'available系数也是负值：该模型可能在学习单队伤情报道的相关性，不能解释为可出场会伤害球队的因果效应。',
                  '优先补足不同球队/来源和真实语义标注；支持训练参与等弱证据需要新标注与表示，不靠四参数实验假装已解决。',
                  '保留零/负收益。此实验不输出g训练标签，也不将historical重建说成实时轨迹。'])
    (output / 'readable-report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(output), 'coverage': coverage,
                      'candidates': candidate_results, 'selected': REGULARIZATIONS[chosen],
                      'A_body_responses_changed': report['A_body_responses_changed']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'var/f-entity-semantic-20261011-v2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
