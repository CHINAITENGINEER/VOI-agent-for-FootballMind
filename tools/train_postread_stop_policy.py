"""Train-only forward holdout stopping calibration for the post-reading selector."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.train_e_drive_postread_g import ResearchData, SelectorInputs, evaluate
from tools.train_e_drive_rolling_g import transitions, purge, dump, write_rows, digest
from football_research.acquisition.postread_g_use import fit_selector, score

THRESHOLDS = [-.05, -.02, -.01, -.005, 0., .005, .01, .02, .05]


def temporal_folds(panels):
    """Keep dates intact, train strictly before each evaluation block, purge bodies."""
    dates = sorted({p['date'] for p in panels})
    if len(dates) < 8:
        raise ValueError('At least eight distinct dates required')
    blocks = np.array_split(np.array(dates), 4)
    folds = []
    for block in blocks[1:]:
        target = [p for p in panels if p['date'] in set(block)]
        history = purge([p for p in panels if p['date'] < min(block)], target)
        if not history or not target:
            raise ValueError('Empty forward fold')
        folds.append((history, target))
    return folds


def choose_threshold(losses):
    """One-SE rule; prefer the unshifted rule, then the more conservative tie."""
    grid = []
    for threshold, values in losses.items():
        a = np.asarray(values, dtype=float)
        if len(a) < 2 or not np.isfinite(a).all():
            raise ValueError('Need finite per-match losses')
        grid.append({'threshold': float(threshold), 'mean_loss': float(a.mean()),
                     'standard_error': float(a.std(ddof=1)/np.sqrt(len(a)))})
    best = min(grid, key=lambda r: (r['mean_loss'], abs(r['threshold'])))
    admissible = [r for r in grid if r['mean_loss'] <= best['mean_loss']+best['standard_error']]
    selected = min(admissible, key=lambda r: (abs(r['threshold']), -r['threshold']))
    return selected, best, grid


def run_policy(inputs, panels, states, model, threshold, cap=3):
    runs = []
    for p in panels:
        mid = p['match_id']; ids = [c['article_id'] for c in p['candidates']]
        adopted, trace = [], []
        while len(adopted) < min(cap, len(ids)):
            state = tuple(a for a in ids if a in adopted)
            remaining = [a for a in ids if a not in adopted]
            x = np.array([inputs.vector('body_state', mid, state, a, states[mid][state]) for a in remaining])
            values = score(model, x); best = int(np.argmax(values))
            selected = remaining[best] if values[best] > threshold else None
            trace.append({'g_seen_ids': ids, 'f_adopted_ids': list(state), 'candidate_ids': remaining,
                          'estimated_gains': values.tolist(), 'stop_threshold': threshold, 'selected': selected})
            if selected is None:
                break
            adopted.append(selected)
        state = tuple(a for a in ids if a in adopted)
        runs.append({'match_id': mid, 'probabilities': states[mid][state].tolist(),
                     'f_adopted_ids': list(state), 'selected_article_ids': adopted, 'trace': trace})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--previous', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); start = time.perf_counter()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    panel = [json.loads(line) for line in (args.previous/'panel.jsonl').read_text(encoding='utf-8').splitlines()]
    old = json.loads((args.previous/'protocol.json').read_text(encoding='utf-8'))
    train = [p for p in panel if p['match_id'] in set(old['g_fit_ids'])]
    folds = temporal_folds(train)
    protocol = {'schema': 'postread-train-only-stop-policy-v1', 'threshold_grid': THRESHOLDS,
        'g_kind': 'ridge', 'g_alpha': 10., 'mode': 'body_state', 'adoption_cap': 3,
        'selection': 'forward_OOF_mean_policy_loss_one_SE_prefer_closest_to_zero',
        'threshold_is_policy_parameter_not_cost_or_calibrated_expected_gain': True,
        'cost_penalty': 0., 'fixed_f_sha256': digest(args.previous/'f-body.json'),
        'g_fit_ids': old['g_fit_ids'], 'validation_ids': old['g_validation_ids'],
        'test_ids': old['evaluation_ids'], 'no_validation_or_test_for_threshold_fit': True,
        'research_periods_previously_examined': True, 'inherited_ridge_alpha_previously_validation_selected': True,
        'production_ready': False, 'network': False,
        'folds': [{'fit_ids': [p['match_id'] for p in a], 'heldout_ids': [p['match_id'] for p in b]} for a,b in folds]}
    dump(out/'protocol.json', protocol)
    data = ResearchData(args.root, args.previous, panel); inputs = SelectorInputs(data)
    f = json.loads((args.previous/'f-body.json').read_text(encoding='utf-8'))
    fsha = protocol['fixed_f_sha256']
    tr, _, states = transitions(data, train, f, fsha, 'g_train')
    losses = {t: [] for t in THRESHOLDS}; oof_rows = []
    for i, (history, target) in enumerate(folds):
        fit_ids = {p['match_id'] for p in history}
        rows = [r for r in tr if r['match_id'] in fit_ids]
        model = fit_selector(inputs.matrix('body_state', rows), [r['label']['signed_gain'] for r in rows],
            [r['match_id'] for r in rows], kind='ridge', alpha=10., mode='body_state')
        dump(out/f'fold-{i+1}-g.json', model)
        for threshold in THRESHOLDS:
            runs = run_policy(inputs, target, states, model, threshold)
            summary, detail = evaluate(data, target, states, runs, 3)
            losses[threshold].extend(d['loss']['g'] for d in detail)
            oof_rows.extend({'fold': i+1, 'threshold': threshold, **d} for d in detail)
        print('FORWARD_FOLD', i+1, 'train', len(history), 'holdout', len(target), flush=True)
    selected, best, grid = choose_threshold(losses)
    write_rows(out/'oof-per-match.jsonl', oof_rows)
    selection = {'selected': selected, 'minimum_mean': best, 'grid': grid,
                 'oof_matches': len(losses[0.]), 'external_validation_test_outcomes_used_for_selection': False,
                 'training_region_oof_outcomes_used_for_selection': True}
    dump(out/'selection.json', selection)  # Frozen before external evaluation.
    model = fit_selector(inputs.matrix('body_state', tr), [r['label']['signed_gain'] for r in tr],
        [r['match_id'] for r in tr], kind='ridge', alpha=10., mode='body_state')
    dump(out/'g-body-state.json', model)
    dump(out/'stop-policy.json', {'schema': 'postread-stop-policy-v1', 'g_sha256': digest(out/'g-body-state.json'),
        'fixed_f_sha256': fsha, 'threshold': selected['threshold'], 'adoption_cap': 3, 'production_ready': False})
    report = {'selection': selection, 'evaluation': {}}
    for name, key in [('validation', 'g_validation_ids'), ('test', 'evaluation_ids')]:
        group = [p for p in panel if p['match_id'] in set(old[key])]
        _, _, sp = transitions(data, group, f, fsha, name)
        summaries = {}
        for policy, threshold in [('zero_threshold', 0.), ('train_calibrated', selected['threshold'])]:
            runs = run_policy(inputs, group, sp, model, threshold)
            summary, detail = evaluate(data, group, sp, runs, 3)
            summaries[policy] = summary
            write_rows(out/f'{name}-{policy}-trace.jsonl', runs)
            write_rows(out/f'{name}-{policy}-per-match.jsonl', detail)
        report['evaluation'][name] = summaries
    report.update(elapsed_seconds=time.perf_counter()-start, network_requests=0,
        conclusion='Exploratory train-only stopping experiment; no production promotion; prior research test reused.')
    dump(out/'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
