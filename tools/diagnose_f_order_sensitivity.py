"""Frozen-f full-pool order diagnosis; no fitting or model selection.

Publication order is a historical ordering assumption, not fact-event order.
Reuse exact previously evaluated complete orders; recompute causal metadata for
every requested order, and infer only missing orders. Labels are attached after
all requested probabilities have been saved to a separate unlabeled file.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from football_research.acquisition.sequential_evidence_f import FrozenF
from football_research.acquisition.sequential_postread_g import EvidencePanel, VectorStore, causal_metadata


EXPERIMENT = Path('E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    with Path(path).open(encoding='utf8') as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def save_json(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def save_rows(path, records):
    with Path(path).open('x', encoding='utf8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.network.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def requested_orders(mid, count, seeds):
    values = [('source_order', None, tuple(range(count)))]
    match_component = int(hashlib.sha256(mid.encode()).hexdigest()[:8], 16)
    for seed in seeds:
        rng = np.random.default_rng(seed + match_component)
        values.append(('random_full_order', seed, tuple(rng.permutation(count).tolist())))
    return values


def existing_complete_orders(path, panel, model_sha):
    """Only project input IDs and saved probabilities; discard label fields.

    The existing file also contains previously evaluated outcomes. It is not a
    physically blind file, but no label enters order planning or f prediction.
    """
    result = {}
    audit = Counter()
    path_sha = sha(path)
    for line, record in rows(path):
        audit['lines_scanned'] += 1
        mid = record.get('physical_match_id')
        if mid not in panel.sequences or record.get('strategy') not in ('all', 'random'):
            continue
        expected = panel.sequences[mid]['document_ids']
        adopted = record.get('selected_ids', [])
        if len(adopted) != len(expected) or set(adopted) != set(expected):
            continue
        if record.get('f_model_sha256') != model_sha:
            raise ValueError('Saved policy path has a different f binding')
        pp = np.asarray(record['p_after'], dtype=float)
        if pp.shape != (3,) or not np.isfinite(pp).all() or (pp <= 0).any() or abs(pp.sum() - 1) > 1e-6:
            raise ValueError('Invalid saved complete-order probability')
        key = (mid, tuple(adopted))
        if key in result:
            if not np.allclose(result[key]['probabilities'], pp, rtol=0, atol=1e-6):
                raise ValueError('Same saved complete order has incompatible predictions')
            audit['duplicate_complete_orders'] += 1
        else:
            result[key] = {'probabilities': pp.tolist(), 'path': str(path), 'line': line,
                           'path_sha256': path_sha, 'strategy': record['strategy']}
    audit['unique_complete_orders'] = len(result)
    return result, dict(audit)


def describe(values):
    a = np.asarray(values, dtype=float)
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95)), 'max': float(a.max()), 'min': float(a.min())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=EXPERIMENT / 'data')
    parser.add_argument('--vectors', type=Path, default=EXPERIMENT / 'gpu-import-v1/embeddings.npz')
    parser.add_argument('--f-root', type=Path, default=EXPERIMENT / 'f-models')
    parser.add_argument('--policies', type=Path, default=EXPERIMENT / 'policy-evaluation')
    parser.add_argument('--output', type=Path, default=EXPERIMENT / 'order-sensitivity-v1')
    parser.add_argument('--seeds', type=int, nargs=5, default=[20261025, 20261026, 20261027, 20261028, 20261029])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Order diagnostic must use a new output directory')
    if len(set(args.seeds)) != 5:
        raise ValueError('Exactly five distinct fixed seeds required')
    torch.set_num_threads(1)

    def offline(*a, **k):
        raise RuntimeError('Offline frozen order diagnostic only')
    socket.create_connection = socket.socket.connect = socket.getaddrinfo = offline
    freeze = read_json(args.f_root / 'selection-freeze.json')
    for path, expected in freeze['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Frozen f input or source changed: ' + path)
    panel = EvidencePanel(args.data, VectorStore(args.vectors), roles=['validation', 'test'],
                          include_outcomes=False, max_candidates=None)
    if len(panel.sequences) != 513:
        raise ValueError('Expected the frozen 513 evaluation news matches')
    models = {'set': FrozenF.load(args.f_root / 'selected-set_residual_mlp'),
              'gru': FrozenF.load(args.f_root / 'selected-gru')}
    for model in models.values():
        if list(model.config.metadata_names) != panel.metadata_names:
            raise ValueError('Metadata contract differs')
    input_files = [args.data / name for name in ('matches.jsonl', 'documents.jsonl', 'sequences.jsonl', 'schema.json')]
    input_files += [args.vectors, Path(__file__), ROOT / 'football_research/acquisition/sequential_postread_g.py',
                    ROOT / 'football_research/acquisition/full_history_sequence_metadata.py']
    input_hashes = {str(p): sha(p) for p in input_files}
    initial_tensor_hashes = {name: tensor_digest(model) for name, model in models.items()}
    args.output.mkdir(parents=True)
    protocol = {'schema': 'footballmind-f-order-sensitivity-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
                'roles': ['validation', 'test'], 'fixed_seeds': args.seeds, 'random_method': 'numpy default_rng(seed + int(sha256(physical_match_id)[:8],16)).permutation(n)',
                'orders_per_match': 6, 'candidate_cap': None, 'identical_full_set_per_order': True,
                'metadata_recomputed_from_each_actual_prefix': True, 'source_and_fact_clocks_modified': False,
                'publication_order_is_fact_update': False, 'source_order_scope': 'saved declared publication then text SHA; historical reconstruction',
                'selection_or_training': False, 'source_hashes': input_hashes,
                'models': {name: model.f_model_sha256 for name, model in models.items()},
                'float_tolerance_set_L1': 1e-6, 'labels_used_after_probability_export': True,
                'reuse_note': 'Existing policy JSONL is previously settled; only IDs/probabilities projected for cache, label fields not used in planning/inference.'}
    save_json(args.output / 'protocol.json', protocol)
    start = time.perf_counter()
    caches = {}; cache_audit = {}
    for name, model in models.items():
        caches[name], cache_audit[name] = existing_complete_orders(args.policies / name / 'per-match-paths.jsonl', panel, model.f_model_sha256)
        # Verify representative reused paths against current frozen inference,
        # including the largest complete pool. This is cache validation only.
        keys = sorted(caches[name])
        check_keys = list(dict.fromkeys(keys[:3] + [max(keys, key=lambda k: len(k[1]))]))
        checked = []
        for key in check_keys:
            mid, ordered_ids = key
            sequence = panel.sequences[mid]
            order = [sequence['document_ids'].index(d) for d in ordered_ids]
            actual = model.predict(sequence['prior'], sequence['vectors'][np.asarray(order, dtype=int)],
                                   causal_metadata(sequence, order), order=list(ordered_ids))
            difference = float(np.abs(actual - np.asarray(caches[name][key]['probabilities'])).sum())
            if difference > 1e-6:
                raise ValueError('Saved complete policy path cannot be reproduced with frozen causal metadata')
            checked.append({'physical_match_id': mid, 'used_ids': list(ordered_ids), 'L1_recomputed_difference': difference})
        cache_audit[name]['independent_frozen_recomputation_checks'] = checked
    output = []; metadata_audit = []; counters = Counter(); metadata_changed = Counter()
    for mid, sequence in sorted(panel.sequences.items()):
        ids = sequence['document_ids']; n = len(ids); source_meta = causal_metadata(sequence, range(n))
        source_prob = {}
        for order_type, seed, order in requested_orders(mid, n, args.seeds):
            ordered_ids = [ids[i] for i in order]
            if len(ordered_ids) != n or set(ordered_ids) != set(ids):
                raise ValueError('Permutation changed complete candidate membership')
            meta = causal_metadata(sequence, order)
            # Align each document back to source order before comparing its
            # prefix-dependent relation metadata, not just matrix positions.
            aligned = np.empty_like(meta)
            for position, original_index in enumerate(order):
                aligned[original_index] = meta[position]
            delta = np.abs(aligned - source_meta)
            changed_columns = [panel.metadata_names[j] for j in np.flatnonzero((delta > 1e-12).any(axis=0))]
            for name in changed_columns:
                metadata_changed[name] += 1
            metadata_audit.append({'physical_match_id': mid, 'role': sequence['split_role'], 'order_type': order_type,
                                   'seed': seed, 'used_ids': ordered_ids, 'metadata_in_adoption_order': meta.tolist(),
                                   'changed_columns_after_document_alignment': changed_columns,
                                   'max_aligned_metadata_difference': float(delta.max()), 'identical_full_candidate_set': True,
                                   'order_basis': sequence['order_basis'], 'source_clocks_unchanged': True})
            for name, model in models.items():
                key = (mid, tuple(ordered_ids))
                cached = caches[name].get(key)
                if cached is not None:
                    pp = np.asarray(cached['probabilities'], dtype=float)
                    reused = {k: v for k, v in cached.items() if k != 'probabilities'}
                    reuse_scope = '_reused_within_diagnostic' if cached['strategy'] == 'new_frozen_inference_this_diagnostic' else '_reused_saved_orders'
                    counters[name + reuse_scope] += 1
                else:
                    trace = model.predict_prefixes(sequence['prior'], sequence['vectors'][np.asarray(order, dtype=int)], meta, order=ordered_ids)
                    if trace[-1]['used_input_ids'] != ordered_ids:
                        raise ValueError('Frozen f did not preserve exact requested adoption order')
                    pp = np.asarray(trace[-1]['probabilities'], dtype=float)
                    reused = None
                    caches[name][key] = {'probabilities': pp.tolist(), 'path': str(args.output / 'order-predictions-unlabeled.jsonl'),
                                         'line': len(output) + 1, 'strategy': 'new_frozen_inference_this_diagnostic'}
                    counters[name + '_new_full_order_inferences'] += 1
                if order_type == 'source_order':
                    source_prob[name] = pp
                output.append({'physical_match_id': mid, 'role': sequence['split_role'], 'model': name,
                               'f_model_sha256': model.f_model_sha256, 'order_type': order_type, 'seed': seed,
                               'candidate_count': n, 'source_order_ids': ids, 'used_ids': ordered_ids,
                               'identical_full_candidate_set': True, 'final_probabilities': pp.tolist(),
                               'L1_vs_source_order': float(np.abs(pp - source_prob[name]).sum()),
                               'predicted_class_vs_source_changed': bool(pp.argmax() != source_prob[name].argmax()),
                               'metadata_changed_columns': changed_columns, 'causal_metadata_recomputed': True,
                               'reused_prediction': reused, 'historical_order_diagnostic': True, 'parameters_updated': False})
    # Complete all requested inference/probability exports before loading labels.
    save_rows(args.output / 'order-predictions-unlabeled.jsonl', output)
    save_rows(args.output / 'metadata-order-audit.jsonl', metadata_audit)
    unlabeled_sha = sha(args.output / 'order-predictions-unlabeled.jsonl')
    labels = {r['physical_match_id']: r for _, r in rows(args.data / 'outcomes.jsonl') if r['physical_match_id'] in panel.sequences}
    source_losses = {}
    for record in output:
        y = labels[record['physical_match_id']]['outcome_class']
        if isinstance(y, bool) or y not in (0, 1, 2):
            raise ValueError('Missing or invalid independent outcome')
        ll = -math.log(max(record['final_probabilities'][y], 1e-12))
        key = (record['model'], record['physical_match_id'])
        if record['order_type'] == 'source_order':
            source_losses[key] = ll
        record.update(outcome_class=y, final_log_loss=ll, log_loss_difference_vs_source=ll - source_losses[key],
                      outcome_source_refs=labels[record['physical_match_id']].get('source_refs'),
                      label_settlement_after_unlabeled_export=True)
    save_rows(args.output / 'order-predictions.jsonl', output)
    groups = defaultdict(list)
    for record in output:
        groups[(record['model'], record['physical_match_id'])].append(record)
    per_match = []
    for (model, mid), records in groups.items():
        source = next(r for r in records if r['order_type'] == 'source_order'); random = [r for r in records if r['seed'] is not None]
        per_match.append({'model': model, 'physical_match_id': mid, 'role': source['role'], 'candidate_count': source['candidate_count'],
                          'unique_orders_among_six': len({tuple(r['used_ids']) for r in records}),
                          'max_L1_vs_source': max(r['L1_vs_source_order'] for r in random),
                          'mean_random_L1_vs_source': float(np.mean([r['L1_vs_source_order'] for r in random])),
                          'final_probability_range_by_class': (np.max([r['final_probabilities'] for r in records], axis=0) - np.min([r['final_probabilities'] for r in records], axis=0)).tolist(),
                          'source_final_log_loss': source['final_log_loss'], 'mean_random_log_loss': float(np.mean([r['final_log_loss'] for r in random])),
                          'log_loss_range_across_six': max(r['final_log_loss'] for r in records) - min(r['final_log_loss'] for r in records),
                          'any_predicted_class_change': any(r['predicted_class_vs_source_changed'] for r in random),
                          'any_metadata_changed': any(r['metadata_changed_columns'] for r in random)})
    save_rows(args.output / 'per-match-order-stats.jsonl', per_match)
    summaries = []
    for model in models:
        for role in ('validation', 'test', 'combined'):
            for scope in ('all_matches', 'multiple_documents'):
                rr = [r for r in per_match if r['model'] == model and (role == 'combined' or r['role'] == role) and (scope == 'all_matches' or r['candidate_count'] > 1)]
                summaries.append({'model': model, 'role': role, 'scope': scope, 'physical_matches': len(rr),
                                  'max_L1_per_match': describe([r['max_L1_vs_source'] for r in rr]),
                                  'mean_random_L1_per_match': describe([r['mean_random_L1_vs_source'] for r in rr]),
                                  'loss_range_per_match': describe([r['log_loss_range_across_six'] for r in rr]),
                                  'mean_source_log_loss': float(np.mean([r['source_final_log_loss'] for r in rr])),
                                  'mean_random_log_loss': float(np.mean([r['mean_random_log_loss'] for r in rr])),
                                  'class_changed_matches': sum(r['any_predicted_class_change'] for r in rr),
                                  'metadata_changed_matches': sum(r['any_metadata_changed'] for r in rr),
                                  'matches_with_L1_above_1e_6': sum(r['max_L1_vs_source'] > 1e-6 for r in rr),
                                  'matches_with_L1_above_0_01': sum(r['max_L1_vs_source'] > .01 for r in rr)})
    seed_metrics = []
    for model in models:
        for role in ('validation', 'test'):
            for seed in [None, *args.seeds]:
                rr = [r for r in output if r['model'] == model and r['role'] == role and r['seed'] == seed]
                seed_metrics.append({'model': model, 'role': role, 'seed': seed, 'physical_matches': len(rr),
                                     'mean_final_log_loss': float(np.mean([r['final_log_loss'] for r in rr])),
                                     'mean_L1_vs_source': float(np.mean([r['L1_vs_source_order'] for r in rr])),
                                     'mean_log_loss_difference_vs_source': float(np.mean([r['log_loss_difference_vs_source'] for r in rr]))})
    final_tensor_hashes = {name: tensor_digest(model) for name, model in models.items()}
    if initial_tensor_hashes != final_tensor_hashes:
        raise ValueError('Model parameters/buffers changed during frozen diagnosis')
    if any(sha(p) != digest for p, digest in input_hashes.items()):
        raise ValueError('Input/source changed during diagnosis')
    set_max = max(r['max_L1_vs_source'] for r in per_match if r['model'] == 'set')
    report = {'schema': 'footballmind-f-order-sensitivity-results-v1', 'physical_matches': len(panel.sequences),
              'roles': dict(Counter(s['split_role'] for s in panel.sequences.values())),
              'full_document_count_histogram': dict(Counter(len(s['document_ids']) for s in panel.sequences.values())),
              'full_document_occurrences': sum(len(s['document_ids']) for s in panel.sequences.values()),
              'requested_prediction_rows': len(output), 'fixed_seeds': args.seeds,
              'all_candidate_sets_preserved': all(r['identical_full_candidate_set'] for r in output),
              'metadata_changed_columns_order_counts': dict(metadata_changed), 'cache_reuse': dict(counters), 'existing_policy_cache_audit': cache_audit,
              'set_full_causal_reordering_L1_below_1e_6': set_max <= 1e-6, 'set_max_L1': set_max,
              'summaries': summaries, 'seed_metrics': seed_metrics, 'initial_tensor_hashes': initial_tensor_hashes,
              'final_tensor_hashes': final_tensor_hashes, 'unlabeled_prediction_file_sha256': unlabeled_sha,
              'outcome_file_sha256': sha(args.data / 'outcomes.jsonl'), 'elapsed_seconds': time.perf_counter() - start,
              'new_training': False, 'model_selection': False, 'network_requests': 0,
              'limitations': ['Random orders preserve the same complete set and original source clocks; they are not fact-event update orders.',
                             'Single-document matches cannot diagnose order sensitivity and are reported separately.',
                             'Five fixed random draws may repeat the source or another order; unique counts are exported.',
                             'Order sensitivity alone does not prove harmful sensitivity or real temporal understanding.',
                             'Set masks history-dependent metadata whereas GRU retains it; this is not an architecture-only comparison.',
                             'Heldout outputs are diagnostic only; no model or threshold is changed.']}
    save_json(args.output / 'summary.json', report)
    plot(args.output, per_match, seed_metrics)
    lines = ['同一完整正文集合的采用顺序诊断', '', f'覆盖{len(panel.sequences)}场比赛（validation193/test320），完整正文出现1041次。189场只有1篇，324场有多篇；未截断8、9、20篇的3场。',
             '每场保留原来源顺序，并用5个固定seed随机排列全部正文。每个排列按真实已采用前缀重算metadata，原始来源/事实时间不变。',
             '复用已有策略中同一f SHA、同一比赛、完全相同完整有序正文ID的预测；缺少的排列才新增离线推理。',
             '所有概率导出后才加载本次独立结果表计算loss；旧策略文件本身含已评结果，但缓存仅投影ID和概率，不能声称整个过程从未打开包含结果的文件。', '',
             '模型 | 集合 | 多文比赛数 | 每场最大L1的均值 | P95 | 全局最大L1 | 概率最大类别变化场数 | 原顺序平均loss | 随机顺序平均loss']
    for r in summaries:
        if r['scope'] == 'multiple_documents':
            lines.append(' | '.join(map(str, [r['model'], r['role'], r['physical_matches'], r['max_L1_per_match']['mean'], r['max_L1_per_match']['p95'], r['max_L1_per_match']['max'], r['class_changed_matches'], r['mean_source_log_loss'], r['mean_random_log_loss']])))
    lines += ['', f'集合模型完整metadata重算后的最大L1={set_max}；是否在1e-6浮点容差内：{set_max <= 1e-6}。',
              'GRU的非零差异表示它依赖采用顺序；随机顺序不是事实发生顺序，不能以此声称学到了真实事件时序，也不能仅凭敏感性断言有害。需结合损失范围和具体输入状态解释。',
              '未训练、未调参、未改模型，不根据这些测试结果再次选择模型。原始时间、正文、分组和向量包不改。']
    with (args.output / 'readable-report.txt').open('x', encoding='utf8') as f:
        f.write('\n'.join(lines) + '\n')
    save_json(args.output / 'manifest.json', {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps({'output': str(args.output), 'matches': len(panel.sequences), 'prediction_rows': len(output),
                      'set_max_L1': set_max, 'cache_reuse': dict(counters), 'elapsed_seconds': report['elapsed_seconds']}))


def plot(output, per_match, seed_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {'set': '#20715d', 'gru': '#c27b1c'}
    for col, model in enumerate(('set', 'gru')):
        rr = [r for r in per_match if r['model'] == model and r['candidate_count'] > 1]
        ax = axes[0, col]
        values = sorted(r['max_L1_vs_source'] for r in rr)
        ax.plot(range(1, len(values) + 1), values, color=colors[model])
        ax.set(title=model + ': max L1 across 5 random full orders', xlabel='Multi-document matches, ranked (n=324)', ylabel='L1 distance from source order')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3));ax.grid(alpha=.2)
        ax.text(.04, .95, 'Max = ' + format(max(values), '.8g'), transform=ax.transAxes, va='top')
    for col, role in enumerate(('validation', 'test')):
        ax = axes[1, col]
        for model in ('set', 'gru'):
            rr = [r for r in seed_metrics if r['model'] == model and r['role'] == role and r['seed'] is not None]
            ax.plot(range(1, 6), [r['mean_log_loss_difference_vs_source'] for r in rr], 'o-', color=colors[model], label=model)
        ax.axhline(0, linestyle='--', color='gray', linewidth=1)
        ax.set(title=role + ': random-order loss minus source-order loss', xlabel='Fixed random seed index (1..5)', ylabel='Mean loss difference (below 0 is lower loss)')
        ax.set_xticks(range(1, 6));ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Full-set order diagnosis: 513 matches; frozen models; no selection', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96));fig.savefig(output / 'order-sensitivity.png', dpi=170);plt.close(fig)


if __name__ == '__main__':
    main()
