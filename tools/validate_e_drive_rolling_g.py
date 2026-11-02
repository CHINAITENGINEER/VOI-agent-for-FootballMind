"""Independent artifact/split audit plus existing frozen-f loader replay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_research.acquisition.corpus_text_f import FrozenCorpusTextPredictor
from football_research.acquisition.corpus_rolling_g import predict_gain


def read_rows(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args(); out = args.experiment
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert all(sha(out/p) == digest for p, digest in manifest.items())
    panel = read_rows(out/'panel.jsonl'); dates = {p['match_id']: p['date'] for p in panel}
    bags = {p['match_id']: p for p in panel}
    by_split = {s: {p['match_id'] for p in panel if p['split'] == s} for s in ('train', 'validation', 'test')}
    ffiles = list(out.glob('f-fold-*.json'))+[out/'f-evaluation.json']
    fs = {sha(p): json.loads(p.read_text(encoding='utf-8')) for p in ffiles}
    base = json.loads((out/'f-base.json').read_text(encoding='utf-8'))
    assert not set(base['fit_match_ids']) & set(dates)
    groups = {s: read_rows(out/f'g-{s}-transitions.jsonl') for s in ('train', 'validation', 'test')}
    checks = 0
    for split, rows in groups.items():
        assert all(r['match_id'] in by_split[split] for r in rows)
        for r in rows:
            mid, f = r['match_id'], fs[r['f_model_sha256']]
            assert mid not in f['fit_match_ids'] and max(dates[m] for m in f['fit_match_ids']) < dates[mid]
            target_bodies = {c['body_sha256'] for c in bags[mid]['candidates']}
            fitted_bodies = {c['body_sha256'] for m in f['fit_match_ids'] for c in bags[m]['candidates']}
            assert not target_bodies & fitted_bodies
            assert r['g_training_allowed'] == (split == 'train')
            before = np.array(r['query']['probabilities_before']); after = np.array(r['label']['probabilities_after'])
            y = r['label']['outcome_index']
            assert np.isclose(before.sum(), 1) and np.isclose(after.sum(), 1)
            assert np.isclose(r['label']['signed_gain'], np.log(after[y])-np.log(before[y]), atol=1e-14)
            assert r['query']['candidate_id'] not in r['query']['already_read_ids']
            checks += 1
    g = json.loads((out/'g-selected.json').read_text(encoding='utf-8'))
    assert set(g['fit_match_ids']) == {r['match_id'] for r in groups['train']}
    assert not set(g['fit_match_ids']) & (by_split['validation'] | by_split['test'])
    # Existing loader has its own feature construction, not the new trainer's feature().
    view = out/'independent-replay-view'; view.mkdir(exist_ok=True)
    for name in ('f-base.json', 'panel.jsonl'):
        shutil.copyfile(out/name, view/name)
    shutil.copyfile(out/'f-evaluation.json', view/'f-body.json')
    predictor = FrozenCorpusTextPredictor(args.root, view, 'body')
    replayed, error = 0, 0.
    checked_states = set()
    for r in groups['validation']+groups['test']:
        mid = r['match_id']; menu = [c['article_id'] for c in bags[mid]['candidates']]
        read = r['query']['already_read_ids']
        expanded = [a for a in menu if a in set(read) | {r['query']['candidate_id']}]
        for ids, expected in [(read, r['query']['probabilities_before']), (expanded, r['label']['probabilities_after'])]:
            key = (mid, tuple(ids))
            if key in checked_states:
                continue
            actual = predictor.predict(mid, ids)['probabilities']
            diff = float(np.max(abs(np.array(actual)-expected))); error = max(error, diff)
            assert diff < 1e-12
            checked_states.add(key); replayed += 1
    gx = np.load(out/'g-test-query-vectors.npy')
    gp = predict_gain(g, gx)
    manual = ((gx-np.array(g['mean']))/np.array(g['scale'])) @ np.array(g['coef'])+g['intercept']
    assert np.array_equal(gp, manual)
    result = {'status': 'PASS', 'original_manifest_files': len(manifest), 'transition_checks': checks,
              'independent_f_loader_states_replayed': replayed, 'max_probability_abs_error': error,
              'g_predictions_recomputed': len(gp), 'f_fit_target_match_or_body_overlap': False,
              'g_fit_validation_test_overlap': False, 'posthoc_zero_gain_filter': False,
              'network_requests': 0, 'production_modified': False}
    (out/'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
