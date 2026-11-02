"""Run a time-split, match-weighted, historical text f experiment on E:.

No network, production mutation, outcome-based sample selection or g activation.
Protocol/panel are written before loading any target labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_research.acquisition.corpus_text_f import make_panel, fit_residual, predict_residual, loss_each
from football_research.data_adapters.e_drive_corpus import ExternalFootballCorpus


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def rows(path, values):
    path.write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False)+'\n' for v in values), encoding='utf-8')


def metrics(p, y):
    return {'matches': len(y), 'log_loss': float(loss_each(p, y).mean()),
            'brier': float(((p-np.eye(3)[y])**2).sum(1).mean()),
            'accuracy': float((p.argmax(1) == y).mean())}


def paired_delta(a, b, y):
    gain = loss_each(a, y)-loss_each(b, y)
    rng = np.random.default_rng(20261020)
    samples = rng.choice(gain, (3000, len(gain)), replace=True).mean(1)
    return {'mean_loss_reduction': float(gain.mean()),
            'match_bootstrap_95_interval': np.quantile(samples, [.025, .975]).tolist(),
            'interpretation': 'descriptive_small_historical_sample_not_prospective_significance'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=False)
    corpus = ExternalFootballCorpus(root)
    matches = {m['match_id']: m for m in corpus.rows('structured/matches.jsonl')}
    links = corpus.rows('research/match_article_candidates.jsonl')
    panel, rejected = make_panel(matches, links, corpus.articles)
    # The outcome-free protocol is materialized before target files are opened.
    protocol = {
        'schema': 'footballmind-e-drive-text-f-experiment-v1', 'seed': 20261020,
        'scope': 'EPL_historical_reconstructed_text_candidates_not_verified_prospective',
        'base_train_before': '2019-07-01', 'base_C_selection_validation_start': '2018-07-01',
        'text_train_before': '2025-07-01', 'text_validation_before': '2026-01-01',
        'text_test_on_or_after': '2026-01-01', 'split_unit': 'physical_match',
        'cross_split_article_and_body_cluster': 'purge_from_all_splits',
        'same_date_articles': 'excluded_kickoff_order_unknown',
        'known_on_or_post_match_modification': 'excluded',
        'source_scope': 'explicit_women_youth_U18s_U21s_excluded_from_mens_panel_retained_in_corpus',
        'candidate_scope_review': 'heuristic_club_date_links_may_be_background_not_all_direct_match_facts',
        'alpha_grid': [.001, .01, .1, 1.0], 'hidden_units': 16,
        'branches': ['metadata', 'title', 'body'],
        'selection': 'minimum_validation_log_loss_per_branch_no_test_selection',
        'promotion_candidate': 'body_vs_baseline_by_validation_only_no_online_promotion',
        'training_subsets': 'all_prefixes_and_singletons_deduplicated_equal_match_weight',
        'test_primary': 'all_eligible_articles_same_matches',
        'test_secondary': 'chronological_prefixes_descriptive_never_g_training',
        'g_training': False, 'network': False, 'production_modified': False,
        'corpus_report_sha256': hashlib.sha256((root/'reports/delivery.json').read_bytes()).hexdigest(),
        'source_date_assumption': 'published_date_asserted_current_body_version_unknown',
    }
    dump(out/'protocol.json', protocol)
    rows(out/'panel.jsonl', panel)
    rows(out/'excluded-links.jsonl', rejected)
    print('PANEL', dict(Counter(p['split'] for p in panel)), 'excluded', len(rejected), flush=True)
    assert all(any(p['split'] == s for p in panel) for s in ('train', 'validation', 'test'))
    for s in ('train', 'validation', 'test'):
        for t in ('train', 'validation', 'test'):
            if s != t:
                assert not ({c['body_sha256'] for p in panel if p['split'] == s for c in p['candidates']}
                            & {c['body_sha256'] for p in panel if p['split'] == t for c in p['candidates']})
    numeric_rows = corpus.rows('vectors/prematch_numeric.rows.jsonl')
    positions = {m['match_id']: m['row'] for m in numeric_rows}
    xnum = np.load(root/'vectors/prematch_numeric.npy').astype(float)
    targets = np.load(root/'structured/labels/prematch_numeric_targets.npy')
    target_rows = corpus.rows('structured/labels/prematch_numeric_targets.rows.jsonl')
    assert [r['match_id'] for r in numeric_rows] == [r['match_id'] for r in target_rows]
    assert np.isfinite(xnum).all() and np.all((targets >= 0) & (targets <= 2))
    base_fit = np.array([positions[mid] for mid, m in matches.items()
                         if m['division'] == 'E0' and m['match_date'] < '2019-07-01'])
    base_train = np.array([i for i in base_fit if numeric_rows[i]['date'] < '2018-07-01'])
    base_val = np.array([i for i in base_fit if numeric_rows[i]['date'] >= '2018-07-01'])
    assert max(numeric_rows[i]['date'] for i in base_fit) < min(p['date'] for p in panel)
    scaler = StandardScaler().fit(xnum[base_train])
    base_grid = []
    for c in [.01, .1, 1., 10.]:
        clf = LogisticRegression(C=c, max_iter=2000).fit(scaler.transform(xnum[base_train]), targets[base_train])
        base_grid.append({'C': c, 'validation_log_loss': metrics(clf.predict_proba(scaler.transform(xnum[base_val])), targets[base_val])['log_loss']})
    best_c = min(base_grid, key=lambda v: v['validation_log_loss'])['C']
    scaler = StandardScaler().fit(xnum[base_fit])
    clf = LogisticRegression(C=best_c, max_iter=2000).fit(scaler.transform(xnum[base_fit]), targets[base_fit])
    assert clf.classes_.tolist() == [0, 1, 2]
    base_model = {'schema': 'footballmind-frozen-lagged-logistic-v1',
                  'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                  'coef': clf.coef_.tolist(), 'intercept': clf.intercept_.tolist(),
                  'C': best_c, 'C_validation': base_grid,
                  'fit_match_ids': [numeric_rows[i]['match_id'] for i in base_fit],
                  'fit_last_date': max(numeric_rows[i]['date'] for i in base_fit),
                  'features': json.loads((root/'vectors/prematch_numeric.schema.json').read_text(encoding='utf-8')),
                  'production_ready': False}
    dump(out/'f-base.json', base_model)
    base_by_id = {p['match_id']: clf.predict_proba(scaler.transform(xnum[[positions[p['match_id']]]]))[0] for p in panel}
    host_order = sorted({a['host'] for a in corpus.articles.values()})
    semantic_dim_start = 11+4+len(host_order)

    def feature(p, subset, branch):
        mid = p['match_id']; m = matches[mid]
        vec = list(scaler.transform(xnum[[positions[mid]]])[0])
        selected = [c for c in p['candidates'] if c['article_id'] in subset]
        side_links = [[c for c in selected if corpus.articles[c['article_id']]['source_team'] == m[side]]
                      for side in ('home_team', 'away_team')]
        vec += [float(len(v)) for v in side_links]
        vec += [float(np.mean([c['days_before_match_date'] for c in v])) if v else 0. for v in side_links]
        vec += [float(sum(corpus.articles[c['article_id']]['host'] == host for c in selected)) for host in host_order]
        if branch != 'metadata':
            for side in side_links:
                vs = []
                for c in side:
                    aid = c['article_id']
                    # Explicit already-read guard; no unselected body enters f.
                    vs.append(corpus.read_evidence(aid, already_read_ids=set(subset))['body_vector']
                              if branch == 'body' else corpus.unread_candidate(aid)['title_vector'])
                vec.extend(np.mean(vs, axis=0).tolist() if vs else [0.]*384)
        return np.asarray(vec)

    selected_models, validation_grids, feature_models = {}, {}, {}
    train_panel = [p for p in panel if p['split'] == 'train']
    val_panel = [p for p in panel if p['split'] == 'validation']
    test_panel = [p for p in panel if p['split'] == 'test']
    for branch in protocol['branches']:
        train_x, train_p, train_y, weights, training_rows = [], [], [], [], []
        for p in train_panel:
            ids = [c['article_id'] for c in p['candidates']]
            subsets = sorted({tuple(ids[:k]) for k in range(1, len(ids)+1)} | {(a,) for a in ids})
            for subset in subsets:
                train_x.append(feature(p, subset, branch)); train_p.append(base_by_id[p['match_id']])
                train_y.append(targets[positions[p['match_id']]])
                weights.append(1/(len(train_panel)*len(subsets)))
                training_rows.append({'match_id': p['match_id'], 'read_article_ids': list(subset), 'weight': weights[-1]})
        rows(out/(branch+'-training-subsets.jsonl'), training_rows)
        vx = np.array([feature(p, [c['article_id'] for c in p['candidates']], branch) for p in val_panel])
        vp = np.array([base_by_id[p['match_id']] for p in val_panel])
        vy = np.array([targets[positions[p['match_id']]] for p in val_panel])
        grid, models = [], []
        for alpha in protocol['alpha_grid']:
            model = fit_residual(train_x, train_p, train_y, weights, alpha=alpha)
            prediction = predict_residual(model, vx, vp, np.ones(len(vx)))
            score = metrics(prediction, vy)
            grid.append({'alpha': alpha, **score, 'iterations': model['optimizer']['iterations']})
            models.append(model)
            print('FIT', branch, alpha, score, flush=True)
        winner = min(range(len(grid)), key=lambda i: grid[i]['log_loss'])
        model = models[winner]
        model.update({'branch': branch, 'fit_match_ids': [p['match_id'] for p in train_panel],
                      'validation_match_ids': [p['match_id'] for p in val_panel], 'host_order': host_order,
                      'feature_layout': '11_base_scaled_numeric+2_side_counts+2_side_ages+host_counts+optional_home_away_384',
                      'base_model_sha256': hashlib.sha256((out/'f-base.json').read_bytes()).hexdigest(),
                      'unread_body_allowed': False})
        dump(out/('f-'+branch+'.json'), model)
        selected_models[branch], validation_grids[branch] = model, grid
    # Selection is frozen here, before test metrics or test labels are inspected.
    validation_baseline = metrics(np.array([base_by_id[p['match_id']] for p in val_panel]),
                                  np.array([targets[positions[p['match_id']]] for p in val_panel]))
    body_val = min(g['log_loss'] for g in validation_grids['body'])
    validation_choice = 'body' if body_val < validation_baseline['log_loss'] else 'base'
    dump(out/'selection.json', {'validation_choice': validation_choice, 'validation_baseline': validation_baseline,
                               'branch_grids': validation_grids, 'test_consulted': False, 'online_promotion': False})
    predictions, evaluation = [], {}
    for split, ps in [('validation', val_panel), ('test', test_panel)]:
        b = np.array([base_by_id[p['match_id']] for p in ps]); y = np.array([targets[positions[p['match_id']]] for p in ps])
        pred = {'base': b}
        for branch, model in selected_models.items():
            x = np.array([feature(p, [c['article_id'] for c in p['candidates']], branch) for p in ps])
            pred[branch] = predict_residual(model, x, b, np.ones(len(ps)))
            if branch == 'body' and split == 'test':
                shuffled = x.copy()
                permutation = np.random.default_rng(20261020).permutation(len(ps))
                shuffled[:, semantic_dim_start:] = shuffled[permutation, semantic_dim_start:]
                pred['body_permuted_inference_control'] = predict_residual(model, shuffled, b, np.ones(len(ps)))
                dump(out/'test-body-permutation.json', {'match_ids': [p['match_id'] for p in ps], 'permutation': permutation.tolist(),
                    'purpose': 'descriptive_inference_sensitivity_not_retrained_null_model'})
        evaluation[split] = {k: metrics(v, y) for k, v in pred.items()}
        evaluation[split]['body_vs_base'] = paired_delta(b, pred['body'], y)
        evaluation[split]['body_vs_title'] = paired_delta(pred['title'], pred['body'], y)
        for i, p in enumerate(ps):
            predictions.append({'match_id': p['match_id'], 'split': split,
                'used_article_ids': [c['article_id'] for c in p['candidates']],
                'probabilities': {k: v[i].tolist() for k, v in pred.items()}, 'outcome_index': int(y[i]),
                'loss': {k: float(loss_each(v[i:i+1], y[i:i+1])[0]) for k, v in pred.items()}})
    rows(out/'heldout-predictions.jsonl', predictions)
    # Fixed candidate f transitions: labels kept for holdout diagnostics only.
    transitions = []
    body_model = selected_models['body']
    for p in test_panel:
        mid = p['match_id']; before = base_by_id[mid]; read_ids = []
        for c in p['candidates']:
            aid = c['article_id']; public = corpus.unread_candidate(aid)
            query = {'match_id': mid, 'prediction_before': before.tolist(), 'already_read_ids': list(read_ids),
                     'candidate_article_id': aid, 'title': public['title'],
                     'title_vector_row': corpus.positions[aid], 'days_before_match': c['days_before_match_date'],
                     'source_team': public['source_team'], 'body_available_to_g': False}
            read_ids.append(aid)
            after = predict_residual(body_model, feature(p, read_ids, 'body')[None, :], base_by_id[mid][None, :], [len(read_ids)])[0]
            y = targets[positions[mid]]
            lb, la = float(loss_each(before[None, :], [y])[0]), float(loss_each(after[None, :], [y])[0])
            transitions.append({'query_before_read': query, 'prediction_after': after.tolist(),
                                'used_input_ids': list(read_ids), 'loss_before': lb, 'loss_after': la,
                                'signed_gain': lb-la, 'probability_L1_change': float(abs(after-before).sum()),
                                'cost': None, 'split': 'test_evaluation_only', 'g_training_allowed': False,
                                'provenance': 'historical_reconstruction_current_version_date_assumption',
                                'f_model_sha256': hashlib.sha256((out/'f-body.json').read_bytes()).hexdigest()})
            before = after
    rows(out/'test-fixed-f-transitions.jsonl', transitions)
    zero_checks = []
    for p in test_panel:
        base = base_by_id[p['match_id']]
        empty = predict_residual(body_model, feature(p, [], 'body')[None, :], base[None, :], [0])[0]
        zero_checks.append(bool(np.array_equal(base, empty)))
    # Read saved model and reproduce all held-out body outputs before delivery.
    loaded = json.loads((out/'f-body.json').read_text(encoding='utf-8'))
    rp = predict_residual(loaded,
        np.array([feature(p, [c['article_id'] for c in p['candidates']], 'body') for p in test_panel]),
        np.array([base_by_id[p['match_id']] for p in test_panel]), np.ones(len(test_panel)))
    expected = np.array([p['probabilities']['body'] for p in predictions if p['split'] == 'test'])
    assert np.array_equal(rp, expected) and all(zero_checks)
    report = {'status': 'COMPLETED_RESEARCH_EVALUATION', 'protocol': protocol,
              'split_matches': dict(Counter(p['split'] for p in panel)),
              'panel_articles': len({c['article_id'] for p in panel for c in p['candidates']}),
              'excluded_link_reasons': dict(Counter(r['reason'] for r in rejected)),
              'base_fit_matches': len(base_fit), 'base_C': best_c, 'validation_choice': validation_choice,
              'branch_parameter_counts': {k: m['parameter_count'] for k, m in selected_models.items()},
              'selected_alpha': {k: m['alpha'] for k, m in selected_models.items()},
              'evaluation': evaluation, 'test_fixed_f_transitions': len(transitions),
              'transition_gain_counts': {'positive': sum(t['signed_gain'] > 1e-12 for t in transitions),
                                         'negative': sum(t['signed_gain'] < -1e-12 for t in transitions),
                                         'zero': sum(abs(t['signed_gain']) <= 1e-12 for t in transitions)},
              'checks': {'empty_evidence_exact_base_identity': all(zero_checks),
                         'saved_model_replay_exact': True, 'split_body_overlap_zero': True,
                         'base_fitted_before_all_text_panel': True},
              'new_g_training_rows': 0, 'new_g_trained': False, 'production_modified': False,
              'limitations': ['Historical current body versions; not prospective snapshots.',
                  'Heuristic candidate binding; not every article is a verified match-specific claim.',
                  'Only four clubs contribute text; repeated teams and date dependence reduce effective sample size.',
                  'Numeric base frozen before July 2019 to prevent same-match label fitting; not a strong current market baseline.',
                  'Text outcomes fit a shared neural residual, not causal player effects or guaranteed useful evidence.',
                  'Test transitions are quarantined evaluation labels, not g training data.',
                  'No claim that text f or g is calibrated or production ready.']}
    dump(out/'report.json', report)
    files = sorted(p for p in out.iterdir() if p.is_file())
    dump(out/'manifest.json', {'files': [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
                               'source_files': [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                   for p in [Path(__file__), Path(__file__).resolve().parents[1]/'football_research/acquisition/corpus_text_f.py']]})
    print('COMPLETE', json.dumps({'split': report['split_matches'], 'evaluation': evaluation, 'choice': validation_choice}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
