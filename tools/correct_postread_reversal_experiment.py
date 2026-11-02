"""Versioned offline correction of the 2026-10-25 postread ablation.

Existing f, labels, models and reports are read-only. New diagnostic Ridge fits
and validation-selected shrinkage are saved in a fresh experiment directory.
All splits are retrospective development checks, not new blind experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import socket
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_postread_shortcut_diagnostics as base

OLD = base.EXP / 'postread-reversal-ablation-20261025-v1'
F_PATH = base.EXP / 'unified-f-postread-g-20261025-v1/replay/frozen-f.json'
DEFAULT_OUTPUT = base.EXP / 'postread-reversal-corrected-20261025-v2'
SHRINK_WEIGHTS = (0., .05, .1, .25, .5, .75, 1.)
RIDGE_KEYS = ('state', 'body', 'semantic', 'title_local', 'title_masked_only',
              'title_masked_local', 'fulltext_state', 'fulltext_masked')
# Frozen vocabulary from the old diagnostic; word boundaries now prevent
# substring replacements such as 'Roma' inside 'aromatic'. Original text stays intact.
_spec = importlib.util.spec_from_file_location('old_ablation_readonly', OLD / 'ablation_lib.py')
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
MASK_PATTERN = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(x) for x in _old.CLUB_RE_LIST) + r')(?!\w)', re.I)


def mask_identity(text):
    value = MASK_PATTERN.sub('team', text)
    return re.sub(r'\bteam(?:\s+team)+\b', 'team', value, flags=re.I)


def normalized_metric_name(name):
    if name.startswith('mlp_') and '_seed' not in name:
        prefix, seed = name.rsplit('_', 1)
        return prefix + '_seed' + seed
    return name


def check_saved_reproduction(old_rows, published):
    lookup = {(r['model'], r['split']): r for r in published}
    results = []
    for row in old_rows:
        name = normalized_metric_name(row['model'])
        # New ablations have no original counterpart. Every original model is
        # required below, so a naming mismatch can no longer silently pass.
        for split in base.SPLITS:
            if (name, split) in lookup:
                error = abs(row[split + '_mse'] - lookup[name, split]['match_macro_mse'])
                results.append(dict(model=name, split=split, abs_mse_difference=error))
    expected = set(lookup)
    observed = {(r['model'], r['split']) for r in results}
    if expected != observed:
        raise ValueError('Missing original metric comparisons: ' + repr(expected - observed))
    if any(r['abs_mse_difference'] > 1e-12 for r in results):
        raise ValueError('Saved metric reproduction differs')
    return dict(comparisons=results, count=len(results),
                scope='saved aggregate metrics; not a new neural refit or per-weight equality claim')


def audit_split(assignments, documents, held_domain=None, dates=None):
    overlap = {}
    for field in ('text_sha256', 'source_url'):
        locations = defaultdict(set)
        for d in documents:
            split = assignments[d['physical_match_id']]
            if split != 'unused':
                locations[d[field]].add(split)
        overlap[field] = [key for key, parts in locations.items() if len(parts) > 1]
        if overlap[field]:
            raise ValueError('Cross-split article: ' + field)
    domains = {s: {d['source_domain'] for d in documents if assignments[d['physical_match_id']] == s}
               for s in base.SPLITS}
    if held_domain:
        if held_domain in domains['train'] or held_domain in domains['validation']:
            raise ValueError('Held-out domain survived exclusion')
        if domains['test'] & (domains['train'] | domains['validation']):
            raise ValueError('Test source overlaps train/validation')
    result = dict(article_overlap=overlap, domains={s: sorted(v) for s, v in domains.items()},
                  matches={s: sum(v == s for v in assignments.values()) for s in (*base.SPLITS, 'unused')})
    if not all(result['matches'][s] for s in base.SPLITS):
        raise ValueError('Empty partition')
    if dates is not None:
        spans = {s: sorted(dates[m] for m, split in assignments.items() if split == s) for s in base.SPLITS}
        if not (spans['train'][-1] < spans['validation'][0] and spans['validation'][-1] < spans['test'][0]):
            raise ValueError('Not globally chronological')
        result['date_spans'] = {s: [v[0], v[-1]] for s, v in spans.items()}
    return result


def correct_source_split(old_assignments, documents, held_domain):
    assignment = dict(old_assignments)
    for d in documents:
        mid = d['physical_match_id']
        if d['source_domain'] == held_domain and assignment[mid] != 'test':
            assignment[mid] = 'unused'
    # Preserve whole matches and quarantine any shared article/URL connected
    # to the test set. Candidate AND adopted evidence come from this doc panel.
    while True:
        forbidden = {(k, d[k]) for d in documents if assignment[d['physical_match_id']] in ('test', 'unused')
                     for k in ('text_sha256', 'source_url')}
        newly_excluded = {d['physical_match_id'] for d in documents
                          if assignment[d['physical_match_id']] in ('train', 'validation')
                          and any((k, d[k]) in forbidden for k in ('text_sha256', 'source_url'))}
        if not newly_excluded:
            break
        for mid in newly_excluded:
            assignment[mid] = 'unused'
    audit_split(assignment, documents, held_domain)
    return assignment


def source_predict(domains, targets, mids, train_indices):
    grouped = defaultdict(list)
    for i in train_indices:
        grouped[domains[i], mids[i]].append(float(targets[i]))
    by_domain = defaultdict(list)
    for (domain, mid), values in grouped.items():
        by_domain[domain].append(float(np.mean(values)))
    means = {d: float(np.mean(v)) for d, v in by_domain.items()}
    fallback = base.macro(targets[train_indices], mids[train_indices])
    pred = np.array([means.get(d, fallback) for d in domains])
    return pred, dict(domain_means=means, fallback_match_equal_training_mean=fallback,
                      fallback_unknown_domains=sorted(set(domains) - set(means)))


def choose_shrinkage(text_val, baseline_val, y_val, mids_val, weights=SHRINK_WEIGHTS):
    values = []
    for weight in weights:
        prediction = weight * text_val + (1 - weight) * baseline_val
        values.append(dict(weight=weight, validation_mse=base.macro((prediction - y_val) ** 2, mids_val)))
    # Smaller weight wins exact ties; test arrays are not accepted by this API.
    best = min(values, key=lambda r: (r['validation_mse'], r['weight']))
    return dict(**best, grid=values, selection_input='validation only')


def centroid_probe(vectors, docs, assignments):
    # Each unique document appears once, never once per state transition.
    domains = [d['source_domain'] for d in docs]
    train_indices = [i for i, d in enumerate(docs) if assignments[d['physical_match_id']] == 'train']
    mids = np.array([docs[i]['physical_match_id'] for i in train_indices])
    weights = base.match_weights(mids)
    centers = {}
    for domain in sorted({domains[i] for i in train_indices}):
        ii = [j for j, i in enumerate(train_indices) if domains[i] == domain]
        center = np.average(vectors[[train_indices[j] for j in ii]], axis=0, weights=weights[ii])
        centers[domain] = center / max(float(np.linalg.norm(center)), 1e-12)
    train_domain_weights = defaultdict(float)
    for i, w in zip(train_indices, weights):
        train_domain_weights[domains[i]] += float(w)
    majority = max(sorted(train_domain_weights), key=train_domain_weights.get)
    records = []
    for i, doc in enumerate(docs):
        split = assignments[doc['physical_match_id']]
        if split == 'unused':
            continue
        prediction = max(centers, key=lambda d: float(vectors[i] @ centers[d]))
        records.append(dict(physical_match_id=doc['physical_match_id'], text_sha256=doc['text_sha256'],
            split=split, actual_domain=domains[i], predicted_domain=prediction,
            known_domain=domains[i] in centers, correct=prediction == domains[i], majority_correct=majority == domains[i]))
    metrics = []
    for split in base.SPLITS:
        rr = [r for r in records if r['split'] == split]
        known = [r for r in rr if r['known_domain']]
        metrics.append(dict(split=split, articles=len(rr), matches=len({r['physical_match_id'] for r in rr}),
            article_accuracy=float(np.mean([r['correct'] for r in rr])),
            match_macro_accuracy=base.macro([r['correct'] for r in rr], [r['physical_match_id'] for r in rr]),
            train_majority_accuracy=float(np.mean([r['majority_correct'] for r in rr])),
            known_domain_articles=len(known), known_domain_accuracy=float(np.mean([r['correct'] for r in known])) if known else None))
    return dict(protocol='train-only centroids, article deduplication, independent match partitions',
                train_domains=sorted(centers), train_majority=majority, metrics=metrics, records=records)


def make_design(features, train_indices):
    if any(set(row) != base.FEATURE_FIELDS for row in features):
        raise ValueError('Feature allowlist differs')
    train = [features[i] for i in train_indices]
    encoders = {}
    for name, fields, masked in [('body', ('candidate_text', 'adopted_text'), False),
                                 ('title', ('candidate_title',), False),
                                 ('body_masked', ('candidate_text', 'adopted_text'), True),
                                 ('title_masked', ('candidate_title',), True)]:
        values = [r[k] for r in train for k in fields]
        if masked:
            values = [mask_identity(v) for v in values]
        encoders[name] = base.fit_encoder(values)
        assert set(encoders[name]['train_text_sha256']) == {base.text_sha(v) for v in values if v.strip()}
    return transform_design(features, encoders), encoders


def transform_design(features, encoders):
    """Inference-only transform, also used to verify saved encoder bytes."""
    if any(set(row) != base.FEATURE_FIELDS for row in features):
        raise ValueError('Feature allowlist differs')
    def vec(key, field, masked=False):
        return base.encode(encoders[key], [mask_identity(r[field]) if masked else r[field] for r in features])
    state = np.array([r['state_numeric'] for r in features], float)
    sem = np.array([r['semantic_numeric'] for r in features], float)
    body, adopted = vec('body', 'candidate_text'), vec('body', 'adopted_text')
    body_m, adopted_m = vec('body_masked', 'candidate_text', True), vec('body_masked', 'adopted_text', True)
    title, title_m = vec('title', 'candidate_title'), vec('title_masked', 'candidate_title', True)
    return dict(state=state, body=body, semantic=np.c_[state, sem], title_local=np.c_[state, title],
        title_masked_only=title_m, title_masked_local=np.c_[state, title_m],
        fulltext_state=np.c_[state, sem, body, adopted, abs(body-adopted), body*adopted],
        fulltext_masked=np.c_[state, sem, body_m, adopted_m, abs(body_m-adopted_m), body_m*adopted_m])


def validate_labels(features, samples, labels, states, documents, f_sha):
    sample_by = {s['sample_id']: s for s in samples}
    label_by = {r['sample_id']: r for r in labels}
    state_by = {(s['physical_match_id'], s['discovered_mask'], s['acquired_mask']): s for s in states}
    docs = {(d['physical_match_id'], d['candidate_index']): d for d in documents}
    assert len(sample_by) == len(samples) == len(label_by) == len(features)
    for f in features:
        s, label = sample_by[f['sample_id']], label_by[f['sample_id']]
        before = state_by[s['physical_match_id'], s['discovered_mask'], s['acquired_mask']]
        after = state_by[s['physical_match_id'], s['discovered_mask'], s['acquired_mask'] | (1 << s['candidate_index'])]
        assert before['f_model_sha256'] == after['f_model_sha256'] == label['f_model_sha256'] == f_sha
        assert np.allclose(before['p'], label['p_before'], atol=1e-14, rtol=0)
        assert np.allclose(after['p'], label['p_after'], atol=1e-14, rtol=0)
        for state, key in ((before, 'loss_before'), (after, 'loss_after')):
            assert abs(-math.log(state['p'][label['outcome_class']]) - label[key]) < 1e-12
        assert abs(before['loss'] - after['loss'] - label['realized_gain']) < 1e-12
        doc = docs[s['physical_match_id'], s['candidate_index']]
        assert base.text_sha(f['candidate_text']) == doc['text_sha256'] == s['text_sha256']
    return dict(rows=len(features), matches=len({s['physical_match_id'] for s in samples}), documents=len(documents),
                same_f_state_and_gain_identity=True, body_hashes_match=True, labels_recomputed=False,
                provenance='historical reconstruction; no new prospective qualification')


def policy_checks(features, samples, states, predictions, assignment):
    score_maps = {k: {r['sample_id']: float(v) for r, v in zip(features, p)} for k, p in predictions.items()}
    detail, _ = base.policy_evaluation(features, samples, states, score_maps, assignment)
    state_by = {(r['physical_match_id'], r['discovered_mask'], r['acquired_mask']): r for r in states}
    n_by = {s['physical_match_id']: s['candidate_count'] for s in samples}
    for mid, split in assignment.items():
        if split != 'test':
            continue
        n = n_by[mid]; dm = (1 << n) - 1
        for budget in (1, 2, 3):
            subsets = [c for count in range(min(n, budget) + 1) for c in itertools.combinations(range(n), count)]
            chosen = min(subsets, key=lambda c: (state_by[mid, dm, sum(1 << i for i in c)]['loss'], len(c), c))
            first, last = state_by[mid, dm, 0], state_by[mid, dm, sum(1 << i for i in chosen)]
            detail.append(dict(physical_match_id=mid, budget=budget, strategy='oracle_analysis_only',
                selected_indices=list(chosen), unselected_indices=[i for i in range(n) if i not in chosen],
                f_updates=len(chosen), p_before=first['p'], p_after=last['p'], initial_loss=first['loss'],
                final_loss=last['loss'], loss_change=last['loss']-first['loss'], stop_reason='uses_outcome_analysis_only',
                budget_compliant=True, g_read_pool_size=n, historical_total_cost=None))
    for r in detail:
        n = n_by[r['physical_match_id']]
        r['cost_breakdown'] = dict(discovery_usd=None, body_acquisition_usd=None, g_encoding_reading_usd=None,
            g_scoring_usd=None, f_update_usd=None, other_usd=None, total_usd=None,
            source_body_pool_size=n, local_source_requests=0, downstream_update_count=r['f_updates'],
            comparison_budget='number of f adoptions; not total economic cost')
        r['economic_superiority_established'] = False
        r['duplicate_adoption'] = len(r['selected_indices']) != len(set(r['selected_indices'])) if r['selected_indices'] is not None else None
        r['missed_improvement_after_stop'] = None
        if r['stop_reason'] == 'predicted_nonpositive':
            mask = sum(1 << i for i in r['selected_indices']); dm = (1 << n) - 1
            remaining = [i for i in range(n) if not mask & (1 << i)]
            possible = [state_by[r['physical_match_id'], dm, mask | sum(1 << i for i in c)]['loss']
                for count in range(min(len(remaining), r['budget'] - len(r['selected_indices'])) + 1)
                for c in itertools.combinations(remaining, count)]
            r['missed_improvement_after_stop'] = r['final_loss'] - min(possible)
    aggregate = []
    for budget in (1, 2, 3):
        for strategy in sorted({r['strategy'] for r in detail}):
            rr = [r for r in detail if r['budget'] == budget and r['strategy'] == strategy]
            aggregate.append(dict(budget=budget, strategy=strategy, matches=len(rr),
                mean_final_loss=float(np.mean([r['final_loss'] for r in rr])),
                mean_loss_change=float(np.mean([r['loss_change'] for r in rr])),
                mean_f_updates=float(np.mean([r['f_updates'] for r in rr])),
                budget_violations=sum(not r['budget_compliant'] for r in rr),
                avoidable_stop_matches=sum((r['missed_improvement_after_stop'] or 0) > 1e-12 for r in rr)))
    return detail, aggregate


def fit_fold(name, assignment, data, destination, f_sha, persist=True):
    all_features, samples, labels, states, documents = data
    features = [{**r, 'split': assignment[r['physical_match_id']]} for r in all_features if assignment[r['physical_match_id']] != 'unused']
    mids = np.array([r['physical_match_id'] for r in features])
    indices = {s: np.array([i for i, r in enumerate(features) if r['split'] == s]) for s in base.SPLITS}
    ti, vi = indices['train'], indices['validation']
    label_by = {r['sample_id']: r for r in labels}; sample_by = {r['sample_id']: r for r in samples}
    y = np.array([label_by[r['sample_id']]['realized_gain'] for r in features])
    domains = [sample_by[r['sample_id']]['source_domain'] for r in features]
    design, encoders = make_design(features, ti)
    weights = base.match_weights(mids[ti]); mean = float(np.average(y[ti], weights=weights))
    scale = max(float(np.sqrt(np.average((y[ti]-mean)**2, weights=weights))), 1e-8)
    predictions = dict(zero=np.zeros(len(y)), train_mean=np.full(len(y), mean))
    predictions['source_mean'], source_meta = source_predict(domains, y, mids, ti)
    fitted = {}
    for key in RIDGE_KEYS:
        scaler = StandardScaler().fit(design[key][ti], sample_weight=weights)
        estimator = Ridge(alpha=10.).fit(scaler.transform(design[key][ti]), (y[ti]-mean)/scale, sample_weight=weights)
        predictions[key+'_ridge'] = estimator.predict(scaler.transform(design[key]))*scale+mean
        fitted[key+'_ridge'] = dict(input_key=key, scaler=scaler, estimator=estimator)
    shrinkage = {}
    for text_name in ('title_local_ridge', 'fulltext_state_ridge', 'fulltext_masked_ridge'):
        for baseline in ('zero', 'train_mean', 'source_mean'):
            key = text_name + '_toward_' + baseline
            choice = choose_shrinkage(predictions[text_name][vi], predictions[baseline][vi], y[vi], mids[vi])
            shrinkage[key] = dict(text_model=text_name, baseline=baseline, **choice)
            predictions[key] = choice['weight']*predictions[text_name] + (1-choice['weight'])*predictions[baseline]
    scores = {k: base.macro((p[vi]-y[vi])**2, mids[vi]) for k, p in predictions.items()}
    selected = min(scores, key=lambda k: (scores[k], k))
    full_models = [k for k in predictions if k.startswith('fulltext')]
    selected_full = min(full_models, key=lambda k: (scores[k], k))
    selection = dict(selected_model=selected, selected_fulltext_family=selected_full, validation_mse=scores,
        shrinkage=shrinkage, test_used=False, fixed_f_sha256=f_sha, retrospective_development_only=True)
    if persist:
        destination.mkdir()
        base.save(destination/'selection-frozen.json', selection)
    # Test metrics and policy outcomes are computed after selection is saved.
    metric_rows = [dict(fold=name, model=k, split=s, **base.metrics(y[ii], p[ii], mids[ii]))
                   for k, p in predictions.items() for s, ii in indices.items()]
    predictions['validation_selected'] = predictions[selected]
    predictions['validation_selected_fulltext'] = predictions[selected_full]
    paths, policy = policy_checks(features, samples, states, predictions, assignment)
    probes = {}
    if persist:
        unique_docs = {}; doc_splits = defaultdict(set)
        for doc in documents:
            if assignment[doc['physical_match_id']] == 'unused':
                continue
            unique_docs.setdefault((doc['physical_match_id'], doc['text_sha256']), doc)
            doc_splits[doc['text_sha256']].add(assignment[doc['physical_match_id']])
        assert not any(len(v) > 1 for v in doc_splits.values())
        docs = list(unique_docs.values())
        for kind, key, field, masked in [('raw_title', 'title', 'title', False), ('masked_title', 'title_masked', 'title', True), ('raw_body', 'body', 'text', False)]:
            values = [mask_identity(d[field]) if masked else d[field] for d in docs]
            probes[kind] = centroid_probe(base.encode(encoders[key], values), docs, assignment)
        bundle = dict(schema='postread-corrected-diagnostic-v2', encoders=encoders, fitted=fitted,
            source_mean=source_meta, target_mean=mean, target_scale=scale, shrinkage=shrinkage,
            selection=selection, production_ready=False, f_model_sha256=f_sha)
        joblib.dump(bundle, destination/'models.joblib', compress=3)
        restored = joblib.load(destination/'models.joblib')
        errors = []
        for key, spec in restored['fitted'].items():
            p = spec['estimator'].predict(spec['scaler'].transform(design[spec['input_key']]))*restored['target_scale']+restored['target_mean']
            errors.append(float(np.max(abs(p-predictions[key]))))
        assert max(errors) <= 1e-12
        base.save(destination/'verification.json', dict(serialized_prediction_max_error=max(errors),
            source_mean_fallback=source_meta, learned_preprocessing_train_only=True, model_sha256=base.sha(destination/'models.joblib')))
        base.save(destination/'metrics.json', metric_rows)
        base.save(destination/'source-probe.json', probes)
        base.save(destination/'policy-comparison.json', policy)
        base.save_rows(destination/'policy-paths.jsonl', paths)
        base.save_rows(destination/'predictions.jsonl', [dict(model=k, sample_id=r['sample_id'],
            physical_match_id=r['physical_match_id'], split=r['split'], predicted_gain=float(p[i]), realized_gain=float(y[i]))
            for k, p in predictions.items() for i, r in enumerate(features)])
    return dict(name=name, selection=selection, metrics=metric_rows, policy=policy,
                probes=probes, predictions=predictions, features=features)


def plots_and_report(output, results, reproduction, label_check, split_audits, random_results, start):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False})
    original = results['P0_original']
    keys = ['zero', 'source_mean', 'title_local_ridge', 'title_masked_local_ridge', 'fulltext_state_ridge', 'fulltext_masked_ridge']
    labels = ['零收益', '来源均值', '标题＋状态', '遮盖标题＋状态', '全文＋状态', '遮盖全文＋状态']
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, split in enumerate(('validation', 'test')):
        vals = [next(r['mse'] for r in original['metrics'] if r['model']==key and r['split']==split) for key in keys]
        ax.bar(np.arange(len(keys))+(j-.5)*.35, vals, .35, label='验证集6场' if j==0 else '原测试集6场（已看过）')
    ax.set_xticks(range(len(keys)), labels, rotation=12); ax.set_ylabel('收益预测 MSE · 越低越好'); ax.legend()
    ax.set_title('修正后的诊断模型；同一冻结 f、历史开发数据'); fig.tight_layout(); fig.savefig(output/'corrected-mse.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, split in enumerate(('validation', 'test')):
        vals=[next(r['match_macro_accuracy'] for r in original['probes'][k]['metrics'] if r['split']==split) for k in ('raw_title','masked_title','raw_body')]
        ax.bar(np.arange(3)+(j-.5)*.35, vals, .35, label=split)
    ax.set_xticks(range(3), ['原标题', '遮盖队名后的标题', '全文']); ax.set_ylim(0,1)
    ax.set_ylabel('独立比赛的来源识别准确率 · 比赛等权'); ax.legend()
    ax.set_title('仅训练集建立中心；每篇正文计一次，不能与旧81.3%混称'); fig.tight_layout(); fig.savefig(output/'heldout-source-probe.png',dpi=160); plt.close(fig)
    chosen = ['P0_original','global_time_51','loco_arsenal','loco_chelsea','loco_liverpool']
    fig, axes = plt.subplots(2,3,figsize=(15,8))
    names=['no_information','all_candidates','random_exact','fixed_order','fulltext_state_ridge','validation_selected_fulltext','validation_selected']
    for ax, fold in zip(axes.flat,chosen):
        for name in names:
            rr=sorted([r for r in results[fold]['policy'] if r['strategy']==name and r['budget_violations']==0],key=lambda r:r['budget'])
            ax.plot([r['budget'] for r in rr],[r['mean_final_loss'] for r in rr],marker='o',label=name)
        ax.set_title(fold);ax.set_xlabel('最多采用正文数');ax.set_ylabel('最终 log loss · 越低越好');ax.set_xticks([1,2,3]);ax.grid(alpha=.2)
    axes.flat[-1].axis('off'); handles, labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',fontsize=9)
    fig.suptitle('完整选择策略：只画预算内结果；真实金额未知，不宣称节省总成本');fig.tight_layout();fig.savefig(output/'corrected-policy-curves.png',dpi=160);plt.close(fig)
    pred_rows=base.rows(output/'P0_original/predictions.jsonl')
    selected=original['selection']['selected_fulltext_family']
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,split in zip(axes,('validation','test')):
        rr=[r for r in pred_rows if r['model']==selected and r['split']==split]
        x=np.array([r['realized_gain'] for r in rr]);y=np.array([r['predicted_gain'] for r in rr]);bound=max(.02,float(max(abs(x).max(),abs(y).max())))
        ax.scatter(x,y,alpha=.65);ax.plot([-bound,bound],[-bound,bound],'--',color='gray',label='预测等于实际的参考线；不是拟合直线')
        ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.set_title(split);ax.set_xlabel('实际 signed gain');ax.set_ylabel('g 预测 signed gain');ax.legend(fontsize=8)
    fig.suptitle('验证集选出的全文模型：'+selected);fig.tight_layout();fig.savefig(output/'selected-gain-scatter.png',dpi=160);plt.close(fig)
    all_metrics=[r for result in results.values() for r in result['metrics']]
    all_policy=[dict(fold=k,**r) for k,result in results.items() for r in result['policy']]
    base.save_csv(output/'metrics.csv',all_metrics);base.save_csv(output/'strategy-comparison.csv',all_policy)
    counts=Counter(r['selected'] for r in random_results)
    summary=dict(status='completed_research_only',label_check=label_check,reproduction_count=reproduction['count'],
        source_holdout_audits=split_audits,random_resplits=random_results,random_validation_winners=dict(counts),
        training_seconds_including_verification_and_plots=time.perf_counter()-start,
        new_independent_matches=0,network_requests=0,external_api_calls=0,formal_models_modified=False,
        neural_refits=0,production_ready=False,all_current_test_sets_previously_viewed=True)
    base.save(output/'summary.json',summary)
    lines=['读后 g 消融修正版 v2：已执行完成','',
        '固定 f 和原始收益标签；仅新建诊断Ridge与收缩模型。零网络、零API、未扩大网络、未改正式模型。',
        f'原数据 {label_check["matches"]} 场 / {label_check["documents"]} 篇 / {label_check["rows"]} 状态转移；没有新增独立比赛。',
        '全部是已查看历史数据上的开发诊断，不是新盲测。',
        '39项旧模型汇总指标对账通过，包含原检查漏掉的18项MLP；没有重新训练MLP。','',
        '已修正：整场全部来源排除；独立文章/比赛的来源识别；未知来源回退比赛等权总体均值；遮盖正文单独训练编码器；遮盖使用词边界。',
        '维持原始正文与实体信息；遮盖仅作对照。全库时间切分单独标明，旧18场队列内时间切分保留为参考。',
        '收缩权重固定候选0/0.05/0.1/0.25/0.5/0.75/1，仅验证集选择。测试结果不参与权重/模型选择。','',
        '按验证集选模型，再看原测试分组：']
    for name,result in results.items():
        sel=result['selection']; best=sel['selected_model']; full=sel['selected_fulltext_family']
        test={r['model']:r['mse'] for r in result['metrics'] if r['split']=='test'}
        pp={r['strategy']:r for r in result['policy'] if r['budget']==2}
        lines.extend([f'{name}: 全部候选模型中选 {best}；全文家族选 {full}。',
            f'  MSE: 零收益 {test["zero"]:.10f}；选中模型 {test[best]:.10f}；选中全文 {test[full]:.10f}。',
            '  最多采用2篇，最终log loss：'+ '；'.join(f'{k}={pp[k]["mean_final_loss"]:.8f}' for k in ('no_information','random_exact','all_candidates','validation_selected','validation_selected_fulltext')),
            f'  全量策略超预算比赛数={pp["all_candidates"]["budget_violations"]}，如非零则不能当同预算对照。'])
    lines.extend(['','来源识别：仅训练比赛拟合中心，独立文章计数，再按比赛等权。'])
    for kind,probe in original['probes'].items():
        for r in probe['metrics']:
            if r['split']!='train':lines.append(f'{kind} {r["split"]}: {r["matches"]}场/{r["articles"]}篇，比赛等权准确率={r["match_macro_accuracy"]:.2%}；训练多数类文章准确率={r["train_majority_accuracy"]:.2%}。')
    lines.extend(['',f'30次重排选中模型次数：{dict(counts)}；同一批比赛的敏感性分析，不是30批独立样本。',
        '来源均值与总体训练均值都可能优于恒零；不能再称来源均值唯一有效。',
        '收益回归MSE与策略log loss分别报告；不从概率变化或回归误差单独推出预测改善。',
        '停止规则：预测边际收益<=0停止；训练仍保留正/零/负收益，不由停止预测过滤标签。',
        '成本：保存各环节未知金额为null，按实际候选数量与f采用次数比较；不冒称全文已读之后仍节省获取成本。',
        '下一步：将当前51场作为开发集；从已有归档准备未参与调参的独立比赛和来源。固定f、同池同预算完成完整策略评估后再决定表示升级。',
        '本轮不把表现较好模型接入正式runtime。'])
    (output/'readable-report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    socket.create_connection=base.forbid_network;socket.socket.connect=base.forbid_network
    source_manifest=base.read(OLD/'manifest.json')
    tracked={str(base.DATA/k):base.sha(base.DATA/k) for k in source_manifest['source_files_sha256_at_end']}
    tracked.update({str(p):base.sha(p) for p in OLD.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    tracked[str(F_PATH)]=base.sha(F_PATH)
    assert all(tracked[str(base.DATA/k)]==v for k,v in source_manifest['source_files_sha256_at_end'].items())
    features,samples,labels,states,documents=[base.rows(base.DATA/n) for n in ('features.jsonl','samples.jsonl','labels.jsonl','states.jsonl','documents.jsonl')]
    f_sha=base.sha(F_PATH); label_check=validate_labels(features,samples,labels,states,documents,f_sha)
    reproduction=check_saved_reproduction(base.read(OLD/'ablation-results.json')['results'],base.read(base.DATA/'training/metrics.json'))
    base.save(output/'reproduction-check.json',reproduction)
    registry=base.read(OLD/'split-registry.json'); audits={}; changes={}
    for club,domain in {'arsenal':'www.arsenal.com','chelsea':'www.chelseafc.com','liverpool':'www.liverpoolfc.com'}.items():
        name='loco_'+club; old=registry[name];registry[name]=correct_source_split(old,documents,domain)
        changes[name]=[dict(physical_match_id=m,before=old[m],after=v) for m,v in registry[name].items() if old[m]!=v]
        audits[name]=audit_split(registry[name],documents,domain)
    plans,_=base.plan_splits(features,samples,documents,base.read(base.EXPANSION_PLAN))
    registry['global_time_51']=plans['global_time_51']['assignments']
    for name,assignment in registry.items():
        audits.setdefault(name,audit_split(assignment,documents,dates=base.match_dates(samples) if name=='global_time_51' else None))
    cohort=sorted(r['physical_match_id'] for r in base.read(base.EXPANSION_PLAN)['groups']);all_mids=sorted(registry['P0_original'])
    random_registry={}
    for seed in range(20261100,20261130):
        order=[cohort[i] for i in np.random.RandomState(seed).permutation(len(cohort))]
        assignment={m:'train' for m in all_mids}
        assignment.update({m:('train' if i<6 else 'validation' if i<12 else 'test') for i,m in enumerate(order)})
        audit_split(assignment,documents);random_registry['rand_'+str(seed)]=assignment
    base.save(output/'experiment-plan.json',dict(schema='postread-reversal-correction-v2',
        named_splits=registry,random_splits=random_registry,corrected_assignments=changes,source_sha256=tracked,
        f_model_sha256=f_sha,mask_tokens=_old.CLUB_RE_LIST,mask_word_boundaries=True,
        ridge_alpha=10.,svd_max_dimensions=32,tfidf_max_features=4000,shrink_weights=SHRINK_WEIGHTS,
        selection='validation match-macro MSE only, saved before test evaluation',
        retain_all_gain_signs=True,production_ready=False,neural_refits=0,new_independent_test_matches=0))
    base.save(output/'split-audit.json',audits)
    data=(features,samples,labels,states,documents);results={};random_results=[]
    with threadpool_limits(limits=1):
        for name,assignment in registry.items():
            print('Fitting diagnostic fold '+name,flush=True)
            results[name]=fit_fold(name,assignment,data,output/name,f_sha)
        for name,assignment in random_registry.items():
            result=fit_fold(name,assignment,data,output/name,f_sha,persist=False)
            rows=result['metrics'];test={r['model']:r['mse'] for r in rows if r['split']=='test'}
            random_results.append(dict(name=name,selected=result['selection']['selected_model'],
                selected_fulltext=result['selection']['selected_fulltext_family'],test_mse=test,
                validation_mse=result['selection']['validation_mse'],shrinkage=result['selection']['shrinkage'],policy=result['policy']))
            if len(random_results)%5==0:print('Completed random resplits '+str(len(random_results)),flush=True)
    base.save(output/'random-resplits.json',random_results)
    plots_and_report(output,results,reproduction,label_check,audits,random_results,start)
    assert all(base.sha(Path(k))==v for k,v in tracked.items()),'Original source changed'
    shutil.copy2(__file__,output/'run-script.py')
    manifest={str(p.relative_to(output)):base.sha(p) for p in output.rglob('*') if p.is_file()}
    base.save(output/'manifest.json',dict(schema='postread-correction-delivery-v2',files=manifest,source_unchanged=True,
        production_ready=False,elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(dict(output=str(output),files=len(manifest),elapsed_seconds=time.perf_counter()-start)))


if __name__=='__main__':
    main()
