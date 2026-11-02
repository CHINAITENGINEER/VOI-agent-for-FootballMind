"""Read-only matched comparisons and initial-state identifiability diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .state import parse_time


def analyze(path):
    path = Path(path)
    report = json.loads((path/'report.json').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in (path/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = {strategy:{r['match_id']:r for r in rows if r['strategy']==strategy and r['budget_money']==.05 and r['seed']==0}
                for strategy in ('base-only','entropy','all-available','risk-mean','risk-conservative','two-step')}
    pairs = []
    for strategy in ('risk-mean','risk-conservative','two-step'):
        for reference in ('base-only','entropy','all-available'):
            a,b = selected[strategy],selected[reference]
            blocks = defaultdict(list)
            for match in sorted(a.keys() & b.keys()):
                time = parse_time(a[match]['kickoff'])
                year,week,_ = time.isocalendar()
                blocks[year,week].append([a[match]['log_loss']-b[match]['log_loss'],
                                        a[match]['cost']['money']-b[match]['cost']['money']])
            values = [np.array(v) for v in blocks.values()]
            sums = np.array([v.sum(axis=0) for v in values])
            counts = np.array([len(v) for v in values])
            rng = np.random.default_rng(1729)
            draws = rng.integers(0,len(values),size=(2000,len(values)))
            samples = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
            pairs.append({'strategy':strategy,'reference':reference,'common_matches':int(counts.sum()),
                'mean_log_loss_difference':float(sums[:,0].sum()/counts.sum()),
                'loss_difference_ci95':np.quantile(samples[:,0],[.025,.975]).tolist(),
                'mean_cost_difference':float(sums[:,1].sum()/counts.sum()),
                'cost_difference_ci95':np.quantile(samples[:,1],[.025,.975]).tolist(),
                'scope':'paired UTC-week bootstrap; same max budget, possibly different realized costs; no multiplicity or fitted-model correction'})
    diag = [json.loads(s) for s in (path/'risk-diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
    # Only mask=0: identity/time/candidate catalogue alone may give no
    # distinguishable match-level context to this deliberately small estimator.
    initial = defaultdict(dict)
    for row in diag:
        if str(row['mask']) == '0':
            initial[row['match_id']][row['action_id']] = row['estimated_gain']
    identifiability = []
    for fold in report['folds']:
        signatures = {json.dumps(initial[m],sort_keys=True) for m in fold['test_match_ids'] if m in initial}
        identifiability.append({'fold':fold['fold'],'matches':len(fold['test_match_ids']),
                               'distinct_initial_value_vectors':len(signatures)})
    return {'status':'OK','comparisons':pairs,'initial_state_identifiability':identifiability,
        'scientific_boundary':'These diagnostics do not prove conditional mean gain is random merely because single-game sign accuracy is low.',
        'stopping_regret_boundary':'Reported regret measures missed feasible continuation after stopping. An all-acquired policy has no continuation and zero by construction; this is not total policy regret.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = analyze(args.run)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'OK','initial_state_identifiability':result['initial_state_identifiability']}))


if __name__ == '__main__':
    main()
