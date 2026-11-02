"""Fixed-f local training of full-body state-dependent evidence selection, cost penalty zero."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.train_e_drive_rolling_g import Data, nested_f, transitions, dump, write_rows, digest
from football_research.acquisition.corpus_rolling_g import match_weights
from football_research.acquisition.postread_g_use import build_features, fit_selector, score, select
from football_research.acquisition import convex_text_f

MODES = ['title_local', 'title_state', 'body_only', 'body_state']


class ResearchData(Data):
    def predict(self, f, mid, read):
        if f.get('schema') == convex_text_f.SCHEMA:
            return convex_text_f.predict(f,self.feature(mid,read)[None],self.prior(mid)[None],[len(read)])[0]
        return super().predict(f,mid,read)


class SelectorInputs:
    """No labels and no after prediction exposed to g's feature builder."""
    def __init__(self, data):
        self.data = data
        self.cache = {}
        self.titles = {aid: data.corpus.unread_candidate(aid)['title_vector'] for aid in data.corpus.articles}
        self.bodies = {}

    def vector(self, mode, mid, adopted, aid, before):
        key = (mode, mid, tuple(adopted), aid, tuple(before))
        if key in self.cache:
            return self.cache[key]
        p = self.data.panel[mid]; ids = [c['article_id'] for c in p['candidates']]
        # Candidate pool is already fetched. In full-body mode, all its bodies are g-visible.
        if mode.startswith('body'):
            for article in ids:
                if article not in self.bodies:
                    self.bodies[article] = self.data.corpus.read_evidence(article, already_read_ids=set(ids))['body_vector']
        c = next(c for c in p['candidates'] if c['article_id'] == aid)
        a = self.data.corpus.articles[aid]
        vector = build_features(mode=mode, probabilities=before, candidate_id=aid,
            g_seen_ids=ids, f_adopted_ids=adopted, menu_ids=ids,
            embedding_by_id=self.bodies, title_by_id=self.titles,
            age_days=c['days_before_match_date'],
            source_side=int(a['source_team'] == self.data.matches[mid]['away_team']),
            host_index=self.data.hosts.index(a['host']), host_count=len(self.data.hosts))
        self.cache[key] = vector
        return vector

    def matrix(self, mode, rows):
        return np.array([self.vector(mode, r['match_id'], r['query']['already_read_ids'],
            r['query']['candidate_id'], r['query']['probabilities_before']) for r in rows])


def rollout(inputs, panels, states, model, budget):
    result = []
    for p in panels:
        mid = p['match_id']; ids = [c['article_id'] for c in p['candidates']]
        cap = len(ids) if budget == 'all' else min(budget, len(ids))
        adopted, trace = [], []
        while len(adopted) < cap:
            canonical = tuple(a for a in ids if a in adopted)
            remaining = [a for a in ids if a not in adopted]
            x = np.array([inputs.vector(model['mode'], mid, canonical, a, states[mid][canonical]) for a in remaining])
            chosen, values = select(model, remaining, x)
            trace.append({'g_seen_ids': ids, 'g_body_visible': model['mode'].startswith('body'),
                'f_adopted_ids': list(canonical), 'candidate_ids': remaining, 'estimated_gains': values,
                'selected': chosen, 'stop_reason': 'nonpositive_estimated_gain' if chosen is None else None})
            if chosen is None:
                break
            adopted.append(chosen)
        final = tuple(a for a in ids if a in adopted)
        result.append({'match_id': mid, 'budget': budget, 'selected_article_ids': adopted,
                       'f_adopted_ids': list(final), 'probabilities': states[mid][final].tolist(),
                       'trace': trace, 'stop_reason': 'nonpositive_estimated_gain' if trace and trace[-1]['selected'] is None
                       else 'pool_exhausted' if len(adopted) == len(ids) else 'adoption_cap_reached'})
    return result


def evaluate(data, panels, states, runs, budget):
    details = []
    byid = {r['match_id']: r for r in runs}
    for p in panels:
        mid = p['match_id']; ids = [c['article_id'] for c in p['candidates']]; sp = states[mid]
        y = int(data.y[data.positions[mid]])
        k = len(ids) if budget == 'all' else min(budget, len(ids))
        loss = lambda prob: float(-np.log(max(prob[y], 1e-15)))
        random_sets = [s for s in sp if len(s) == k]
        baseline = {'no_information': loss(sp[()]), 'all_information': loss(sp[tuple(ids)]),
            'fixed_order': loss(sp[tuple(ids[:k])]),
            'random_exact': float(np.mean([loss(sp[s]) for s in random_sets])),
            'g': loss(byid[mid]['probabilities'])}
        n = len(byid[mid]['f_adopted_ids'])
        matched = [s for s in sp if len(s) == n]
        baseline['random_matched_g_count_descriptive'] = float(np.mean([loss(sp[s]) for s in matched]))
        details.append({'match_id': mid, 'loss': baseline, 'candidate_count': len(ids), 'g_adopted_count': n,
                        'g_gain': baseline['no_information']-baseline['g'],
                        'g_vs_fixed': baseline['fixed_order']-baseline['g'],
                        'g_vs_random': baseline['random_exact']-baseline['g'], 'outcome_index': y})
    return {'mean_log_loss': {k: float(np.mean([d['loss'][k] for d in details])) for k in details[0]['loss']},
            'mean_g_adoptions': float(np.mean([d['g_adopted_count'] for d in details])),
            'matches_with_any_g_adoption': sum(d['g_adopted_count'] > 0 for d in details)}, details


def export_transitions(path, rows, panel):
    lookup = {p['match_id']: p for p in panel}
    write_rows(path, [{**{k: v for k, v in r.items() if k != 'query'},
        'selector_state': {'g_seen_ids': [c['article_id'] for c in lookup[r['match_id']]['candidates']],
            'f_adopted_ids': r['query']['already_read_ids'], 'candidate_id': r['query']['candidate_id'],
            'probabilities_before': r['query']['probabilities_before'], 'candidate_body_legal_input': True}}
        for r in rows])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--previous', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fixed-convex-f',type=Path)
    ap.add_argument('--f-boundary-date', help='Optional predeclared YYYY-MM-DD boundary between f and g fitting')
    args = ap.parse_args(); started = time.perf_counter(); out = args.output
    out.mkdir(parents=True, exist_ok=False)
    panel = [json.loads(l) for l in (args.previous/'panel.jsonl').read_text(encoding='utf-8').splitlines()]
    old_train = [p for p in panel if p['split'] == 'train']
    if args.f_boundary_date:
        from datetime import date
        boundary = date.fromisoformat(args.f_boundary_date).isoformat()
    else:
        boundary = old_train[24]['date']
    ftrain = [p for p in old_train if p['date'] < boundary]
    train = [p for p in old_train if p['date'] >= boundary]
    val = [p for p in panel if p['split'] == 'validation']; test = [p for p in panel if p['split'] == 'test']
    groups = [ftrain, train, val, test]
    if any(not group for group in groups):
        raise ValueError('All four temporal groups must be nonempty')
    body_sets = [{c['body_sha256'] for p in group for c in p['candidates']} for group in groups]
    if any(body_sets[i] & body_sets[j] for i in range(4) for j in range(i)):
        raise ValueError('Shared body cluster crosses f/g/validation/test boundary')
    protocol = {'schema': 'footballmind-postread-g-use-experiment-v1', 'cost_penalty': 0.,
        'task': 'g_sees_candidate_body_then_selects_evidence_to_adopt_into_fixed_f',
        'f_fit_ids': [p['match_id'] for p in ftrain], 'g_fit_ids': [p['match_id'] for p in train],
        'f_g_boundary_date': boundary,
        'g_validation_ids': [p['match_id'] for p in val], 'evaluation_ids': [p['match_id'] for p in test],
        'same_fixed_f_for_every_label_and_policy': True, 'modes': MODES,
        'learners': {'ridge': [.01, .1, 1., 10.], 'mlp': [.01, .1, 1.]}, 'hidden': 16,
        'g_hyperparameter_selection': 'validation_final_policy_log_loss_budget_3_then_gain_MSE_tiebreak',
        'stop_threshold': 0., 'evaluation_adoption_caps': [1, 2, 3, 'all'],
        'training_exploration': 'enumerate_all_subsets_and_remaining_actions_no_gain_filter',
        'weights': 'equal_total_weight_per_physical_match', 'pool': 'same_archived_candidates_for_all_policies',
        'g_seen_vs_f_adopted_separate': True, 'encoding': 'frozen_MiniLM_384_chunk_mean_no_encoder_finetuning',
        'outcome_or_after_prediction_g_input': False, 'historical_reconstruction': True,
        'validation_and_test_previously_examined': True, 'online_promotion': False, 'network': False,
        'no_acquisition_cost_savings_claim': True, 'future_h_trained': False}
    if args.fixed_convex_f:
        protocol.update(fixed_f_source=str(args.fixed_convex_f),fixed_f_source_sha256=digest(args.fixed_convex_f),
            fixed_f_role='research_challenger_not_validated_improvement',
            prior_postread_test_already_seen=True)
    dump(out/'protocol.json', protocol); write_rows(out/'panel.jsonl', panel)
    shutil.copyfile(args.previous/'f-base.json', out/'f-base.json')
    data = ResearchData(args.root, args.previous, panel)
    if args.fixed_convex_f:
        f=json.loads(args.fixed_convex_f.read_text(encoding='utf-8'))
        assert f['schema']==convex_text_f.SCHEMA
        assert set(f['fit_match_ids']) == set(protocol['f_fit_ids'])
        assert f['base_model_sha256'] == digest(out/'f-base.json')
        assert f['fit_max_date'] < min(p['date'] for p in train+val+test)
        shutil.copyfile(args.fixed_convex_f,out/'f-body.json'); fsha=digest(out/'f-body.json')
    else:
        f, fsha = nested_f(data, ftrain, train+val+test, out, 'f-body')
    inputs = SelectorInputs(data)
    tr, _, ts = transitions(data, train, f, fsha, 'train')
    vr, _, vs = transitions(data, val, f, fsha, 'validation')
    y = np.array([r['label']['signed_gain'] for r in tr]); vy = np.array([r['label']['signed_gain'] for r in vr])
    mids = [r['match_id'] for r in tr]; vw = match_weights([r['match_id'] for r in vr])
    export_transitions(out/'training-transitions.jsonl', tr, panel)
    export_transitions(out/'validation-transitions.jsonl', vr, panel)
    selected, selection = {}, {}
    for mode in MODES:
        x, xv = inputs.matrix(mode, tr), inputs.matrix(mode, vr)
        np.save(out/(mode+'-train-vectors.npy'), x)
        grids, models = [], []
        for kind, alphas in protocol['learners'].items():
            for alpha in alphas:
                t0 = time.perf_counter()
                model = fit_selector(x, y, mids, kind=kind, alpha=alpha, mode=mode)
                model.update(f_model_sha256=fsha, host_order=data.hosts,
                             validation_match_ids=protocol['g_validation_ids'])
                runs = rollout(inputs, val, vs, model, 3)
                stats, _ = evaluate(data, val, vs, runs, 3)
                pred = score(model, xv)
                item = {'kind': kind, 'alpha': alpha, 'policy_log_loss': stats['mean_log_loss']['g'],
                    'gain_mse': float(vw@((pred-vy)**2)), 'mean_adoptions': stats['mean_g_adoptions'],
                    'parameters': model['parameter_count'], 'seconds': time.perf_counter()-t0}
                grids.append(item); models.append(model)
                dump(out/f'g-{mode}-{kind}-{alpha}.json', model)
                print('TRAINED', mode, item, flush=True)
        winner = min(range(len(grids)), key=lambda i: (grids[i]['policy_log_loss'], grids[i]['gain_mse']))
        selected[mode] = models[winner]; selection[mode] = {'selected': grids[winner], 'grid': grids}
        dump(out/('g-'+mode+'-selected.json'), selected[mode])
    # Every model and policy choice saved before evaluation labels are consulted.
    dump(out/'selection.json', {'modes': selection, 'test_used_for_selection': False,
                               'primary_comparison': 'body_state_vs_title_local_and_matched_baselines'})
    er, _, es = transitions(data, test, f, fsha, 'test_evaluation_only')
    export_transitions(out/'test-transitions.jsonl', er, panel)
    comparisons, traces, detail_rows = {}, [], []
    for split, ps, states, labels in [('validation', val, vs, vr), ('test', test, es, er)]:
        comparisons[split] = {}
        w = match_weights([r['match_id'] for r in labels]); target = np.array([r['label']['signed_gain'] for r in labels])
        for mode, model in selected.items():
            x = inputs.matrix(mode, labels)
            reloaded = json.loads((out/('g-'+mode+'-selected.json')).read_text(encoding='utf-8'))
            assert np.array_equal(score(model, x), score(reloaded, x))
            comparisons[split][mode] = {'gain_mse': float(w@((score(model, x)-target)**2)), 'budgets': {}}
            for budget in protocol['evaluation_adoption_caps']:
                runs = rollout(inputs, ps, states, reloaded, budget)
                stats, details = evaluate(data, ps, states, runs, budget)
                comparisons[split][mode]['budgets'][str(budget)] = stats
                traces.extend({'split': split, 'mode': mode, **r} for r in runs)
                detail_rows.extend({'split': split, 'mode': mode, 'budget': budget, **r} for r in details)
    write_rows(out/'policy-traces.jsonl', traces); write_rows(out/'per-match-comparison.jsonl', detail_rows)
    fids = set(f['fit_match_ids']); tids = set(mids)
    assert not fids & (tids | set(protocol['g_validation_ids']) | set(protocol['evaluation_ids']))
    assert not tids & (set(protocol['g_validation_ids']) | set(protocol['evaluation_ids']))
    assert all(not r['f_target_in_fit'] and r['f_model_sha256'] == fsha for r in tr+vr+er)
    report = {'status': 'TRAINED_LOCAL_FIXED_F_POSTREAD_SELECTORS', 'fixed_f_sha256': fsha,
        'f_fit_matches': len(f['fit_match_ids']), 'g_training_matches': len(tids), 'g_training_transitions': len(tr),
        'validation_matches': len(val), 'validation_transitions': len(vr), 'test_matches': len(test), 'test_transitions': len(er),
        'train_gain_signs': dict(Counter('positive' if v > 1e-12 else 'negative' if v < -1e-12 else 'near_zero' for v in y)),
        'training_matches_with_multiple_candidates': sum(len(p['candidates']) > 1 for p in train),
        'model_selection': selection, 'comparisons': comparisons,
        'duration_seconds': time.perf_counter()-started, 'network_calls': 0, 'paid_api_calls': 0, 'gpu_used': False,
        'cost_penalty': 0., 'production_modified': False, 'saved_g_model_replay_exact': True,
        'limitations': ['historical reconstructed current-version bodies', 'validation and test reused from earlier research',
            f'{len(tids)} independent gain-training matches; transitions are correlated within matches',
            f'f trained on earlier {len(f["fit_match_ids"])}-match text panel',
            'body means may lose negation/conditional/time semantics', 'one-step selector, future h not trained',
            'no automatic online promotion or claim of acquisition savings']}
    dump(out/'report.json', report)
    with (out/'逐场策略对比.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['split','mode','cap','match','candidates','adopted','no_info_loss','all_loss','fixed_loss','random_loss','g_loss','vs_no_info_direction','magnitude'])
        for r in detail_rows:
            gain = r['g_gain']; losses = r['loss']
            writer.writerow([r['split'],r['mode'],r['budget'],r['match_id'],r['candidate_count'],r['g_adopted_count'],
                *[losses[k] for k in ['no_information','all_information','fixed_order','random_exact','g']],
                '损失减少' if gain > 0 else '损失增加' if gain < 0 else '不变',abs(gain)])
    dump(out/'manifest.json', {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'})
    print('COMPLETE', str(out), 'seconds', report['duration_seconds'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
