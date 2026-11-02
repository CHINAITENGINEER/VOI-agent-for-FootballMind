"""Offline paired development ablation on the previous frozen rows and folds."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition import scoped_reviewed_f as model
from tools.reconstruct_panel_priors import no_network
from tools.train_reviewed_rich_f import write

SOURCE = ROOT / 'var/reviews/fm-reviewed-rich-f-training-20261014-v1'
GRID = [None, 10., 1., .1]


def metrics(fitted, rows):
    groups = defaultdict(list)
    for r in rows:
        p = model.predict(fitted, r); y = r['outcome']
        groups[r['physical_match_id']].append((float(-np.log(p[y])), float(np.sum((p - np.eye(3)[y]) ** 2))))
    return [{'physical_match_id': k, 'log_loss': float(np.mean([v[0] for v in values])),
             'brier': float(np.mean([v[1] for v in values]))} for k, values in sorted(groups.items())]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    socket.socket.connect = no_network; socket.getaddrinfo = no_network
    read = lambda name: json.loads((SOURCE / name).read_text(encoding='utf-8'))
    rows = read('training-rows.json'); old = read('report.json'); old_plan = read('plan.json')
    dev = [r for r in rows if r['physical_match_id'] in old_plan['development_ids']]
    probe = [r for r in rows if r['physical_match_id'] == old_plan['post_selection_probe_id']]
    modes = ['direct_only', 'typed_context']
    write(out / 'plan.json', {'schema': 'scoped-reviewed-f-experiment-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'rules': model.rule_manifest(),
        'source_hashes': {n: sha256((SOURCE / n).read_bytes()).hexdigest() for n in
                          ['training-rows.json', 'structured-priors.json', 'plan.json', 'report.json']},
        'modes': modes, 'grid': GRID, 'folds': old['folds'],
        'design_informed_by_previous_ticket_probe': True, 'ticket_probe_now_regression_only': True,
        'selection': 'same paired one-SE rule, per family',
        'source_basis': 'same historical development rows; no new independent validation',
        'unknown_information_value': None, 'live_activation': False, 'g_training': False})
    facts = {r['physical_match_id']: r for r in rows}
    dt = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00'))
    report = {'schema': 'scoped-reviewed-f-training-result-v1', 'families': {},
              'development_matches': len({r['physical_match_id'] for r in dev}), 'states': len(dev),
              'validation_matches': old['forward_validation_matches'],
              'validation_team_bias': old['validation_team_bias'], 'new_validation_matches': 0,
              'ticket_regression_is_not_independent': True, 'production_ready': False,
              'network': 0, 'provider_calls': 0, 'g_training': False}
    for mode in modes:
        scores = {lam: [] for lam in GRID}; optimizers = []
        for fold in old['folds']:
            train_ids, valid_ids = fold['training_ids'], fold['validation_ids']
            assert not set(train_ids) & set(valid_ids)
            assert max(dt(facts[i]['result_available_at']) for i in train_ids) < min(dt(facts[i]['decision_at']) for i in valid_ids)
            train = [r for r in dev if r['physical_match_id'] in train_ids]
            valid = [r for r in dev if r['physical_match_id'] in valid_ids]
            for lam in GRID:
                fitted = model.fit(train, lam, mode)
                assert set(fitted['head']['encoder']['fit_match_ids']) == set(train_ids)
                scores[lam].extend(metrics(fitted, valid)); optimizers.append(fitted['head']['optimizer'])
        matrix = np.asarray([[x['log_loss'] for x in scores[lam]] for lam in GRID])
        best = int(np.argmin(matrix.mean(axis=1)))
        se = (matrix - matrix[best]).std(axis=1, ddof=1) / np.sqrt(matrix.shape[1])
        selected = next(i for i in range(len(GRID)) if matrix[i].mean() <= matrix[best].mean() + se[i] + 1e-12)
        candidate = 1 + int(np.argmin(matrix[1:].mean(axis=1)))
        data = {'selected_regularization': GRID[selected], 'candidate_regularization': GRID[candidate],
                'candidates': [{'regularization': lam, 'log_loss': float(matrix[i].mean()),
                    'brier': float(np.mean([x['brier'] for x in scores[lam]])),
                    'paired_se_to_best': float(se[i]), 'per_match': scores[lam]} for i, lam in enumerate(GRID)],
                'fold_optimizers': optimizers, 'models': {}}
        for name, index in [('selected', selected), ('candidate', candidate)]:
            fitted = model.fit(dev, GRID[index], mode)
            path = out / f'{mode}-{name}.json'; model.save(fitted, path)
            loaded = model.load(path)
            assert np.array_equal(model.predict(fitted, dev[-1]), model.predict(loaded, dev[-1]))
            preds = [model.predict(fitted, r).tolist() for r in probe]
            data['models'][name] = {'parameters': int(np.asarray(fitted['head']['coefficients']).size),
                'optimizer': fitted['head']['optimizer'], 'ticket_predictions': preds,
                'ticket_l1_shift': float(np.abs(np.asarray(preds[-1]) - preds[0]).sum())}
        counts = Counter(); audit = []
        for r in rows:
            routed, annotations = model.route(r, mode)
            counts.update(x['lane'] for x in annotations)
            audit.append({'physical_match_id': r['physical_match_id'], 'used_input_ids': r['used_input_ids'],
                          'before_count': len(r['evidence']), 'routed_count': len(routed['evidence']),
                          'annotations': annotations})
        write(out / f'{mode}-routing-audit.json', audit)
        data['state_exposure_lane_counts_not_unique_claims'] = dict(counts)
        report['families'][mode] = data
        print(json.dumps({'mode': mode, 'selected': GRID[selected], 'candidate': GRID[candidate],
                          'log_loss': matrix.mean(axis=1).tolist()}), flush=True)
    write(out / 'report.json', report)


if __name__ == '__main__':
    main()
