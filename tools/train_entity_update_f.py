"""Restore exact provenance, resolve pending evidence, train on frozen folds."""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import socket
import sys
from urllib.parse import urlsplit
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition import reviewed_rich_f as rich
from football_research.acquisition import entity_evidence_updates as temporal
from football_research.acquisition.context_predictor import _digest
from tools.review_f_evidence_panel import MAPPING
from tools.reconstruct_panel_priors import no_network
from tools.train_reviewed_rich_f import write

SOURCE = ROOT / 'var/reviews/fm-reviewed-rich-f-training-20261014-v1'
PANEL = ROOT / 'var/work/fm-f-evidence-training-panel-20261014-01'
GRID = [None, 10., 1., .1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def enrich(rows):
    protocol = read(ROOT / 'var/f-entity-semantic-20261011-v2/protocol.json')
    records = read(protocol['source_bindings']['records']['path'])
    cases = read(protocol['source_bindings']['cases']['path'])
    clocks = {}; fixtures = {}; concepts = {}
    for case in cases:
        mid = case['fixture']['physical_match_id']; fixtures[mid] = case['fixture']
        for d in case['documents']:
            key = mid, d['url']; value = d.get('source_published_at')
            if key in clocks and clocks[key] != value:
                raise ValueError('Ambiguous document publication version')
            clocks[key] = value
    diagnostic = read(ROOT / 'var/reviews/fm-f-evidence-panel-acceptance-20261014-v2/diagnostic.json')
    samples = [json.loads(x) for x in (PANEL / 'samples.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    by_sample = {s['sample_id']: s for s in samples}
    for entry in diagnostic['samples']:
        s = by_sample[entry['sample_ids'][0]]; mid = s['fixture']['physical_match_id']
        clocks[mid, s['url']] = s['original_clocks']['source_published_at']
        for record in entry['semantic_records']:
            records.append(record)
            fixtures[mid] = record['input']['fixture']
            for c in s['review']['claims']:
                concepts[record['input']['input_id'], c['assertion']] = MAPPING[c['claim_id']][0]
    bank = {}
    for rec in records:
        key = rec['input']['input_id']
        if key in bank and bank[key]['response'] != rec['response']:
            raise ValueError('Conflicting frozen extraction version')
        bank[key] = rec
    result = []
    for row in rows:
        mid = row['physical_match_id']; events = {}
        for input_id in row['used_input_ids']:
            rec = bank[input_id]; public = rec['input']
            assert public['fixture']['physical_match_id'] == mid
            for claim in rec['response']['claims']:
                e = rich.evidence_tokens(claim, public['channel'], fixtures[mid], concepts.get((input_id, claim['statement'])))
                event = {'physical_match_id': mid, 'input_id': input_id, 'entity': claim['entity'],
                         'document_url': public['url'], 'source_host': urlsplit(public['url']).hostname,
                         'source_published_at': clocks.get((mid, public['url'])),
                         'historical_available_at': None, 'claim': claim, 'evidence': e}
                event_id = _digest(event); events[event_id] = {**event, 'event_id': event_id}
        assert rich.unique_evidence([e['evidence'] for e in events.values()]) == row['evidence'], mid
        result.append({**row, 'evidence_events': list(events.values())})
    return result


def metrics(model, rows):
    grouped = defaultdict(list)
    for r in rows:
        p = temporal.predict(model, r); y = r['outcome']
        grouped[r['physical_match_id']].append((float(-np.log(p[y])), float(np.sum((p - np.eye(3)[y])**2))))
    return [{'physical_match_id': mid, 'log_loss': float(np.mean([x[0] for x in vals])),
             'brier': float(np.mean([x[1] for x in vals]))} for mid, vals in sorted(grouped.items())]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    socket.socket.connect = no_network; socket.getaddrinfo = no_network
    old_plan, old_report = read(SOURCE / 'plan.json'), read(SOURCE / 'report.json')
    write(out / 'plan.json', {'schema': temporal.VERSION, 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'source_rows_sha256': sha256((SOURCE / 'training-rows.json').read_bytes()).hexdigest(),
        'folds': [{k: fold[k] for k in ('training_ids', 'validation_ids')} for fold in old_report['folds']],
        'grid': GRID, 'update_rule': 'same exact entity/team/facet, same source host, different document, strictly later publication; pending to resolved only',
        'unknown_time': 'retain without superseding', 'cross_source_conflict': 'retain without adjudication',
        'new_independent_validation_matches': 0, 'g_training': False, 'runtime_activation': False})
    rows = enrich(read(SOURCE / 'training-rows.json'))
    write(out / 'event-training-rows.json', rows)
    audit = []; changes = []
    for i, row in enumerate(rows):
        resolved, info = temporal.resolve(row)
        audit.append({'row_index': i, 'physical_match_id': row['physical_match_id'], **info})
        if resolved['evidence'] != row['evidence']:
            changes.append(i)
    write(out / 'event-resolution-audit.json', audit)
    dev = [r for r in rows if r['physical_match_id'] in old_plan['development_ids']]
    scores = {lam: [] for lam in GRID}; optimizers = []
    dt = temporal.timestamp; facts = {r['physical_match_id']: r for r in dev}
    for fold in old_report['folds']:
        train_ids, valid_ids = fold['training_ids'], fold['validation_ids']
        assert not set(train_ids) & set(valid_ids)
        assert max(dt(facts[i]['result_available_at']) for i in train_ids) < min(dt(facts[i]['decision_at']) for i in valid_ids)
        train = [r for r in dev if r['physical_match_id'] in train_ids]
        valid = [r for r in dev if r['physical_match_id'] in valid_ids]
        for lam in GRID:
            m = temporal.fit(train, lam); scores[lam].extend(metrics(m, valid))
            assert set(m['head']['head']['encoder']['fit_match_ids']) == set(train_ids)
            optimizers.append(m['head']['head']['optimizer'])
    matrix = np.array([[r['log_loss'] for r in scores[lam]] for lam in GRID])
    best = int(np.argmin(matrix.mean(axis=1))); se = (matrix - matrix[best]).std(axis=1, ddof=1)/np.sqrt(matrix.shape[1])
    chosen = next(i for i in range(len(GRID)) if matrix[i].mean() <= matrix[best].mean() + se[i] + 1e-12)
    candidate = 1 + int(np.argmin(matrix[1:].mean(axis=1)))
    models = {}
    for name, index in [('selected', chosen), ('candidate', candidate)]:
        model = temporal.fit(dev, GRID[index]); path = out / f'{name}.json'; temporal.save(model, path)
        assert np.array_equal(temporal.predict(model, dev[-1]), temporal.predict(temporal.load(path), dev[-1]))
        models[name] = model
    write(out / 'report.json', {'schema': temporal.VERSION, 'changed_state_indices': changes,
        'changed_physical_matches': sorted({rows[i]['physical_match_id'] for i in changes}),
        'source_event_exposures': sum(len(r['evidence_events']) for r in rows),
        'superseded_event_exposures': sum(e['status'] == 'superseded_pending_assertion' for a in audit for e in a['events']),
        'conflict_groups': sum(len(a['conflicts']) for a in audit),
        'selected_regularization': GRID[chosen], 'candidate_regularization': GRID[candidate],
        'candidates': [{'regularization': lam, 'log_loss': float(matrix[i].mean()),
                       'brier': float(np.mean([r['brier'] for r in scores[lam]])),
                       'paired_se_to_best': float(se[i]), 'per_match': scores[lam]} for i, lam in enumerate(GRID)],
        'optimizers': optimizers, 'production_ready': False, 'network': 0, 'provider_calls': 0, 'g_training': False,
        'limitations': ['source publication order is not historical availability', 'same reused Leeds-heavy development folds',
                       'conflicts retained, not adjudicated; no fuzzy entity aliases', 'not an online runtime change']})
    # Show the exact real updated assertion chain, without outcomes/gain.
    examples = []
    for i in changes:
        r = rows[i]
        examples.append({'physical_match_id': r['physical_match_id'], 'audit': audit[i],
            'prediction': temporal.predict(models['candidate'], r).tolist(),
            'interpretation': 'full_fit_response_diagnostic_not_validation', 'gain': None})
    write(out / 'updated-state-examples.json', examples)
    print(json.dumps({'changed_states': len(changes), 'selected': GRID[chosen], 'candidate': GRID[candidate],
                      'log_loss': matrix.mean(axis=1).tolist()}))


if __name__ == '__main__':
    main()
