"""Nested temporal f cross-fitting -> pre-query g -> locked historical policy evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_research.acquisition.corpus_text_f import fit_residual, predict_residual, loss_each
from football_research.acquisition.corpus_rolling_g import (
    subsets, prequery_features, fit_gain, predict_gain, match_weights, choose_action)
from football_research.data_adapters.e_drive_corpus import ExternalFootballCorpus


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def purge(history, targets):
    bodies = {c['body_sha256'] for p in targets for c in p['candidates']}
    # Remove entire training match when a body cluster crosses this boundary.
    return [p for p in history if not bodies.intersection(c['body_sha256'] for c in p['candidates'])]


class Data:
    def __init__(self, root, previous, panel):
        self.corpus = ExternalFootballCorpus(root)
        self.panel = {p['match_id']: p for p in panel}
        self.matches = {m['match_id']: m for m in self.corpus.rows('structured/matches.jsonl')}
        self.positions = {r['match_id']: r['row'] for r in self.corpus.rows('vectors/prematch_numeric.rows.jsonl')}
        self.base = json.loads((previous/'f-base.json').read_text(encoding='utf-8'))
        self.hosts = sorted({a['host'] for a in self.corpus.articles.values()})
        self.num = np.load(root/'vectors/prematch_numeric.npy')
        self.y = np.load(root/'structured/labels/prematch_numeric_targets.npy')
        target_rows = self.corpus.rows('structured/labels/prematch_numeric_targets.rows.jsonl')
        assert all(self.positions[r['match_id']] == r['row'] for r in target_rows)
        self.cache = {}

    def prior(self, mid):
        x = (self.num[self.positions[mid]]-self.base['mean'])/self.base['scale']
        return softmax(x@np.array(self.base['coef']).T+self.base['intercept'])

    def feature(self, mid, read):
        key = (mid, tuple(sorted(read)))
        if key in self.cache:
            return self.cache[key]
        p, m = self.panel[mid], self.matches[mid]
        allowed = {c['article_id'] for c in p['candidates']}
        if len(read) != len(set(read)) or not set(read) <= allowed:
            raise ValueError('Invalid read subset')
        cs = [c for c in p['candidates'] if c['article_id'] in read]
        sides = [[c for c in cs if self.corpus.articles[c['article_id']]['source_team'] == m[side]]
                 for side in ('home_team', 'away_team')]
        v = list((self.num[self.positions[mid]]-self.base['mean'])/self.base['scale'])
        v += [len(s) for s in sides]
        v += [float(np.mean([c['days_before_match_date'] for c in s])) if s else 0. for s in sides]
        v += [sum(self.corpus.articles[c['article_id']]['host'] == h for c in cs) for h in self.hosts]
        for s in sides:
            vs = [self.corpus.read_evidence(c['article_id'], already_read_ids=set(read))['body_vector'] for c in s]
            v.extend(np.mean(vs, axis=0) if vs else np.zeros(384))
        self.cache[key] = np.array(v)
        return self.cache[key]

    def query(self, mid, read, aid, before):
        p, m = self.panel[mid], self.matches[mid]
        c = next(c for c in p['candidates'] if c['article_id'] == aid)
        a = self.corpus.unread_candidate(aid)
        # Only public title vectors, never candidate body or result.
        return prequery_features(probabilities=before, read_count=len(read), menu_count=len(p['candidates']),
            age_days=c['days_before_match_date'], source_side=int(a['source_team'] == m['away_team']),
            host_index=self.hosts.index(self.corpus.articles[aid]['host']), host_count=len(self.hosts),
            title_vector=a['title_vector'],
            read_title_vectors=[self.corpus.unread_candidate(i)['title_vector'] for i in read])

    def fit_f(self, ps, alpha):
        xs, bs, ys, ws = [], [], [], []
        for p in ps:
            mid = p['match_id']; ss = subsets([c['article_id'] for c in p['candidates']])[1:]
            for s in ss:
                xs.append(self.feature(mid, s)); bs.append(self.prior(mid)); ys.append(self.y[self.positions[mid]])
                ws.append(1/(len(ps)*len(ss)))
        return fit_residual(xs, bs, ys, ws, alpha=alpha)

    def predict(self, f, mid, read):
        return predict_residual(f, self.feature(mid, read)[None], self.prior(mid)[None], [len(read)])[0]


def nested_f(data, history, targets, out, name):
    history = purge(history, targets)
    assert max(p['date'] for p in history) < min(p['date'] for p in targets)
    boundary = history[max(1, int(len(history)*.75))]['date']
    inner_val = [p for p in history if p['date'] >= boundary]
    inner_train = purge([p for p in history if p['date'] < boundary], inner_val)
    assert len(inner_train) >= 8 and len(inner_val) >= 3
    grid = []
    for alpha in [.1, 1., 10.]:
        f = data.fit_f(inner_train, alpha)
        probs = np.array([data.predict(f, p['match_id'], [c['article_id'] for c in p['candidates']]) for p in inner_val])
        y = [data.y[data.positions[p['match_id']]] for p in inner_val]
        grid.append({'alpha': alpha, 'log_loss': float(loss_each(probs, y).mean())})
    alpha = min(grid, key=lambda r: r['log_loss'])['alpha']
    f = data.fit_f(history, alpha)
    f.update({'fit_match_ids': [p['match_id'] for p in history], 'fit_max_date': max(p['date'] for p in history),
              'target_min_date': min(p['date'] for p in targets), 'branch': 'body', 'host_order': data.hosts,
              'base_model_sha256': digest(out/'f-base.json'), 'hyperparameter_selection': 'nested_earlier_dates_only',
              'inner_train_ids': [p['match_id'] for p in inner_train],
              'inner_validation_ids': [p['match_id'] for p in inner_val], 'inner_grid': grid})
    path = out/(name+'.json'); dump(path, f)
    print('F_FOLD', name, 'fit', len(history), 'target', len(targets), 'alpha', alpha, flush=True)
    return f, digest(path)


def transitions(data, ps, f, fsha, split):
    rows, vectors, state_probs = [], [], {}
    for p in ps:
        mid = p['match_id']; ids = [c['article_id'] for c in p['candidates']]
        states = subsets(ids)
        probs = {s: data.predict(f, mid, s) for s in states}
        state_probs[mid] = probs
        y = int(data.y[data.positions[mid]])
        for read in states:
            for aid in ids:
                if aid in read:
                    continue
                after_ids = tuple(i for i in ids if i in set(read) | {aid})
                before, after = probs[read], probs[after_ids]
                loss_before, loss_after = -np.log(before[y]), -np.log(after[y])
                vectors.append(data.query(mid, read, aid, before))
                rows.append({'match_id': mid, 'date': p['date'], 'split': split,
                    'query': {'already_read_ids': list(read), 'candidate_id': aid,
                              'probabilities_before': before.tolist(), 'unread_body_visible': False},
                    'label': {'probabilities_after': after.tolist(), 'outcome_index': y,
                              'loss_before': float(loss_before), 'loss_after': float(loss_after),
                              'signed_gain': float(loss_before-loss_after)},
                    'f_model_sha256': fsha, 'f_fit_max_date': f['fit_max_date'],
                    'f_target_in_fit': mid in f['fit_match_ids'],
                    'g_training_allowed': split == 'train', 'propensity': None,
                    'label_origin': 'exhaustive_offline_counterfactual_historical_reconstruction'})
    return rows, np.array(vectors), state_probs


def gain_metrics(model, x, rows):
    y = np.array([r['label']['signed_gain'] for r in rows])
    pred = predict_gain(model, x); w = match_weights([r['match_id'] for r in rows])
    return {'match_weighted_mse': float(w @ ((pred-y)**2)),
            'match_weighted_mae': float(w @ abs(pred-y)), 'mean_predicted_gain': float(w @ pred)}


def evaluate_policy(data, ps, states, g, budget, cost):
    details = []
    for p in ps:
        mid = p['match_id']; ids = [c['article_id'] for c in p['candidates']]; probs = states[mid]
        y = int(data.y[data.positions[mid]])
        lb = float(-np.log(probs[()][y])); chosen, trace = [], []
        for step in range(min(budget, len(ids))):
            read = tuple(i for i in ids if i in chosen)
            remaining = [i for i in ids if i not in chosen]
            x = np.array([data.query(mid, read, a, probs[read]) for a in remaining])
            aid, net = choose_action(g, remaining, x, cost)
            trace.append({'read_ids': list(read), 'candidate_ids': remaining, 'predicted_net_values': net,
                          'selected': aid, 'stop_reason': 'nonpositive_estimated_net_gain' if aid is None else None})
            if aid is None:
                break
            chosen.append(aid)
        final = tuple(i for i in ids if i in chosen)
        k = min(budget, len(ids))
        random_subsets = [s for s in probs if len(s) == k]
        losses = {'no_query': lb, 'fixed_order': float(-np.log(probs[tuple(ids[:k])][y])),
                  'uniform_random_exact': float(np.mean([-np.log(probs[s][y]) for s in random_subsets])),
                  'learned_g': float(-np.log(probs[final][y]))}
        reads = {'no_query': 0, 'fixed_order': k, 'uniform_random_exact': k, 'learned_g': len(chosen)}
        details.append({'match_id': mid, 'budget': budget, 'cost': cost, 'trace': trace,
                        'loss': losses, 'reads': reads,
                        'net_loss_reduction': {s: lb-losses[s]-cost*reads[s] for s in losses}})
    summary = {s: {'mean_log_loss': float(np.mean([r['loss'][s] for r in details])),
                   'mean_reads': float(np.mean([r['reads'][s] for r in details])),
                   'mean_net_loss_reduction': float(np.mean([r['net_loss_reduction'][s] for r in details]))}
               for s in ('no_query', 'fixed_order', 'uniform_random_exact', 'learned_g')}
    return summary, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--previous', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    panel = [json.loads(l) for l in (args.previous/'panel.jsonl').read_text(encoding='utf-8').splitlines()]
    train = [p for p in panel if p['split'] == 'train']; val = [p for p in panel if p['split'] == 'validation']
    test = [p for p in panel if p['split'] == 'test']
    # Freeze date-only folds before target loading. Never split a calendar date.
    dates = sorted({p['date'] for p in train}); first_date = train[24]['date']
    later_dates = [d for d in dates if d >= first_date]
    folds = [[p for p in train if p['date'] in ds] for ds in np.array_split(later_dates, 5)]
    protocol = {'schema': 'footballmind-rolling-f-g-experiment-v1', 'warmup_matches_min': 24,
        'rolling_folds': [[p['match_id'] for p in f] for f in folds],
        'f_alpha_grid': [.1, 1., 10.], 'f_selection': 'nested_75_percent_time_history_then_refit',
        'g_grid': ['zero', 'constant', 'ridge:0.1', 'ridge:1', 'ridge:10', 'ridge:100'],
        'g_selection': 'minimum_validation_match_weighted_gain_MSE',
        'labels': 'all_states_all_remaining_actions_including_negative_and_zero',
        'weights': 'each_match_equal_total_weight_edges_equal_within_match',
        'evaluation_budgets': [1, 2, 3], 'costs_research_loss_units': [0., .001, .005],
        'frozen_test_ids': [p['match_id'] for p in test], 'test_previously_examined': True,
        'g_train_test_overlap_allowed': False, 'prospective': False, 'online_promotion': False,
        'future_value_h_trained': False, 'g_scope': 'one_step_conditional_gain',
        'historical_body_version_unknown': True, 'network': False,
        'panel_source_sha256': digest(args.previous/'panel.jsonl')}
    dump(out/'protocol.json', protocol); write_rows(out/'panel.jsonl', panel)
    shutil.copyfile(args.previous/'f-base.json', out/'f-base.json')
    data = Data(args.root, args.previous, panel)
    all_rows, all_x, fold_summaries = [], [], []
    for i, targets in enumerate(folds):
        history = [p for p in train if p['date'] < min(t['date'] for t in targets)]
        f, sha = nested_f(data, history, targets, out, 'f-fold-'+str(i))
        rs, x, _ = transitions(data, targets, f, sha, 'train')
        all_rows.extend(rs); all_x.append(x)
        fold_summaries.append({'fold': i, 'fit_matches': len(f['fit_match_ids']), 'target_matches': len(targets),
                               'transitions': len(rs), 'f_sha256': sha, 'alpha': f['alpha']})
    xtrain = np.concatenate(all_x); ytrain = np.array([r['label']['signed_gain'] for r in all_rows])
    mids = [r['match_id'] for r in all_rows]
    assert all(not r['f_target_in_fit'] and r['f_fit_max_date'] < r['date'] for r in all_rows)
    f, fsha = nested_f(data, train, val+test, out, 'f-evaluation')
    vrows, vx, vstates = transitions(data, val, f, fsha, 'validation')
    write_rows(out/'g-train-transitions.jsonl', all_rows); np.save(out/'g-train-query-vectors.npy', xtrain)
    write_rows(out/'g-validation-transitions.jsonl', vrows); np.save(out/'g-validation-query-vectors.npy', vx)
    candidates, scores = [], []
    for kind, alpha in [('zero', 1.), ('constant', 1.)]+[('ridge', a) for a in [.1, 1., 10., 100.]]:
        g = fit_gain(xtrain, ytrain, mids, kind=kind, alpha=alpha)
        g.update({'host_order': data.hosts, 'f_training_versions': [s['f_sha256'] for s in fold_summaries],
                  'evaluation_f_sha256': fsha, 'validation_match_ids': [p['match_id'] for p in val]})
        name = kind+'-'+str(alpha); dump(out/('g-'+name+'.json'), g)
        score = {'name': name, 'kind': kind, 'alpha': alpha, **gain_metrics(g, vx, vrows)}
        candidates.append(g); scores.append(score)
        print('G_VALIDATION', score, flush=True)
    winner = min(range(len(scores)), key=lambda i: scores[i]['match_weighted_mse'])
    g = candidates[winner]
    dump(out/'g-selected.json', g)
    dump(out/'selection.json', {'selected': scores[winner], 'grid': scores, 'test_consulted': False,
                               'note': 'Baseline can win; no forced conditional-model promotion'})
    # Test labels and state transitions only evaluated after selected g is saved.
    trows, tx, tstates = transitions(data, test, f, fsha, 'test_evaluation_only')
    write_rows(out/'g-test-transitions.jsonl', trows); np.save(out/'g-test-query-vectors.npy', tx)
    assert not set(mids).intersection(p['match_id'] for p in val+test)
    assert not {c['body_sha256'] for p in train for c in p['candidates']}.intersection(
        c['body_sha256'] for p in val+test for c in p['candidates'])
    loaded = json.loads((out/'g-selected.json').read_text(encoding='utf-8'))
    assert np.array_equal(predict_gain(g, tx), predict_gain(loaded, tx))
    policy_scores, traces = {}, []
    for split, ps, states in [('validation', val, vstates), ('test', test, tstates)]:
        policy_scores[split] = {}
        for cost in protocol['costs_research_loss_units']:
            for budget in protocol['evaluation_budgets']:
                score, detail = evaluate_policy(data, ps, states, loaded, budget, cost)
                policy_scores[split][f'budget={budget},cost={cost}'] = score
                traces.extend({'split': split, **r} for r in detail)
    write_rows(out/'policy-traces.jsonl', traces)
    report = {'status': 'TRAINED_AND_EVALUATED_RESEARCH_ONLY', 'folds': fold_summaries,
        'g_training_matches': len(set(mids)), 'g_training_rows': len(all_rows),
        'g_validation_matches': len(val), 'g_validation_rows': len(vrows),
        'g_test_matches': len(test), 'g_test_rows': len(trows),
        'train_gain_signs': dict(Counter('positive' if y > 1e-12 else 'negative' if y < -1e-12 else 'zero' for y in ytrain)),
        'selected_g': scores[winner], 'validation_grid': scores,
        'test_gain_metrics': {s['name']: gain_metrics(m, tx, trows) for s, m in zip(scores, candidates)},
        'policy_evaluation': policy_scores, 'saved_model_replay_exact': True,
        'training_target_overlap': False, 'production_changed': False, 'gpu_used': False,
        'limitations': ['historical current-version text, not prospective',
                       '46 test matches were examined in earlier f experiment; repeated descriptive holdout',
                       'fold-dependent f changes label mechanism; not a single calibrated deployed f',
                       'most training bags contain one article; ranking evidence limited',
                       'gain is realized loss change, not an observed expected VoI',
                       'one-step g only; no newly trained continuation h',
                       'f remains a research candidate, not established effective on new live matches']}
    dump(out/'report.json', report)
    with (out/'逐动作收益.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['split', 'match', 'read_count', 'article', 'before_loss', 'after_loss', 'direction', 'magnitude'])
        for r in all_rows+vrows+trows:
            d = r['label']['signed_gain']
            writer.writerow([r['split'], r['match_id'], len(r['query']['already_read_ids']), r['query']['candidate_id'],
                             r['label']['loss_before'], r['label']['loss_after'], '损失减少' if d > 0 else '损失增加' if d < 0 else '不变', abs(d)])
    dump(out/'manifest.json', {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'})
    print('COMPLETE', out, 'train', len(set(mids)), len(all_rows), 'selected', scores[winner]['name'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
