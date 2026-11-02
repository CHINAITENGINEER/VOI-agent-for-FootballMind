"""Frozen old-g/new-f transfer on an explicitly previously studied common pool.

This is a separate diagnostic, never an untouched test result. Old pre-label
state/entity features and cached text representations are reused. Only current
probability/entropy/margin fields are replaced by the frozen new f's outputs.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from football_research.acquisition.postread_semantic_embedding import predict_frozen_head
from football_research.acquisition.sequential_evidence_f import FrozenF
from football_research.acquisition.sequential_postread_g import (
    EvidencePanel, VectorStore, FrozenEvaluator, causal_metadata, run_policy,
    settle_path, summarize_paths, save_json, save_rows, sha_file)

OLD = Path('E:/FootballMind/experiments/postread-clean-expansion-20261025-v1')
SEMANTIC = Path('E:/FootballMind/experiments/postread-qwen-semantic-20261025-v1')


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def transfer_feature(row, probability):
    result = copy.deepcopy(row)
    names = result['state_feature_names']
    p = np.asarray(probability, dtype=float)
    top = np.sort(p)
    values = dict(p_home=p[0], p_draw=p[1], p_away=p[2],
                  entropy_nats=float(-np.sum(p*np.log(p))), top_two_margin=float(top[-1]-top[-2]))
    for name, value in values.items():
        result['state_numeric'][names.index(name)] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--f-model', type=Path, required=True)
    parser.add_argument('--g-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Diagnostic output already exists')
    import torch
    torch.set_num_threads(1)
    old_bundle = joblib.load(SEMANTIC/'training/models.joblib')
    old_vectors = np.load(SEMANTIC/'embeddings.npz', allow_pickle=False)
    new_bundle = joblib.load(args.g_model/'g-models.joblib')
    new_selection = json.loads((args.g_model/'selection-frozen.json').read_text(encoding='utf-8'))
    if new_bundle['embedding_dimension'] != 256 or new_selection['selected'] != new_bundle['selected']:
        raise ValueError('Unexpected new-g representation or unfrozen selection')
    if any(sha_file(path) != expected for path, expected in new_bundle['source_sha256'].items()):
        raise ValueError('New-g source dataset changed')
    frozen = FrozenF.load(args.f_model)
    if new_bundle['f_model_sha256'] != frozen.f_model_sha256:
        raise ValueError('New g/f SHA binding mismatch')
    panel = EvidencePanel(args.data, VectorStore(args.vectors),
                          roles={'f_fit','f_train','f_dev','g_fit','g_dev','validation','test'}, include_outcomes=True)
    if panel.metadata_names != new_bundle['metadata_names']:
        raise ValueError('New-g metadata contract changed')
    features = {r['sample_id']: r for r in rows(OLD/'features.jsonl')}
    samples = rows(OLD/'samples.jsonl')
    old_docs = rows(OLD/'documents.jsonl')
    docs_by_match = {}
    lookup = {}
    for doc in old_docs:
        docs_by_match.setdefault(doc['physical_match_id'], {})[doc['candidate_index']] = doc
    for record in samples:
        if record['discovered_mask'] == (1 << record['candidate_count']) - 1:
            lookup[(record['physical_match_id'], record['acquired_mask'], record['candidate_index'])] = features[record['sample_id']]
    aliases = {}
    for mid, match in panel.matches.items():
        for alias in [mid, *match.get('aliases', [])]:
            if alias in aliases and aliases[alias] != mid:
                raise ValueError('Ambiguous explicit identity alias')
            aliases[alias] = mid
    bindings = []; excluded = []
    for old_mid, documents in docs_by_match.items():
        mid = aliases.get(old_mid)
        if mid not in panel.sequences or mid not in panel.outcomes:
            excluded.append(dict(original_match_id=old_mid, reason='not_current_eligible_sequence_with_result')); continue
        original = panel.sequences[mid]
        order = [documents[i] for i in range(len(documents))]
        ids = ['sha256:'+doc['text_sha256'] for doc in order]
        if not set(ids).issubset(original['document_ids']):
            excluded.append(dict(original_match_id=old_mid, physical_match_id=mid,
                                 reason='old_candidate_body_not_eligible_in_current_protocol')); continue
        positions = [original['document_ids'].index(doc_id) for doc_id in ids]
        rebuilt = dict(original, document_ids=ids, vectors=original['vectors'][positions],
                       document_records=[original['document_records'][i] for i in positions],
                       original_candidate_count=len(ids), capped_candidate_count=len(ids),
                       order_basis='old study candidate order; explicit common-pool transfer diagnostic')
        rebuilt['meta'] = causal_metadata(rebuilt, range(len(ids)))
        panel.sequences[mid] = rebuilt
        bindings.append(dict(old_id=old_mid, physical_match_id=mid, documents=ids))
    evaluator = FrozenEvaluator(frozen, panel)
    selected = new_bundle['selected']; new_head = new_bundle['fitted'][selected['model']]
    paths = []
    for binding in bindings:
        old_mid = binding['old_id']; mid = binding['physical_match_id']; n = len(binding['documents'])
        for budget in (1, 2, 3):
            for strategy in ('none', 'all', 'fixed', 'random', 'new_g'):
                for repeat in range(10 if strategy == 'random' else 1):
                    path = run_policy(evaluator, mid, budget=budget,
                                      strategy='g' if strategy == 'new_g' else strategy,
                                      spec=new_head, target_mean=new_bundle['target_mean'],
                                      target_scale=new_bundle['target_scale'], threshold=selected['threshold'],
                                      seed=20261025+repeat)
                    path = settle_path(path, panel.outcomes[mid])
                    path.update(strategy=strategy, split_role='previously_studied_transfer_diagnostic',
                                original_match_id=old_mid, main_split=panel.matches[mid]['split_role'])
                    paths.append(path)
            adopted = []; decisions = []; before = evaluator.predict(mid, []); score_seconds = 0.0
            stop = 'budget_exhausted'
            for _ in range(min(n,budget)):
                remaining = [i for i in range(n) if i not in adopted]
                if not remaining:
                    stop = 'pool_exhausted'; break
                p = evaluator.predict(mid, adopted)
                mask = sum(1 << i for i in adopted)
                score_started = time.perf_counter()
                ff = [transfer_feature(lookup[(old_mid,mask,i)], p) for i in remaining]
                predicted = predict_frozen_head(old_bundle, ff, old_vectors)
                score_seconds += time.perf_counter() - score_started
                index = int(np.argmax(predicted))
                decisions.append(dict(adopted_indices=list(adopted), p_before=p.tolist(),
                                      scores=[dict(candidate_index=i, predicted_gain=float(score))
                                              for i,score in zip(remaining,predicted)]))
                if predicted[index] <= 0:
                    stop = 'legacy_nonpositive_gain'; break
                adopted.append(remaining[index])
            after = evaluator.predict(mid, adopted)
            y = panel.outcomes[mid]['outcome_class']
            legacy_path = dict(physical_match_id=mid, original_match_id=old_mid,
                              split_role='previously_studied_transfer_diagnostic', main_split=panel.matches[mid]['split_role'],
                              strategy='original_frozen_g_transfer', budget=budget, selected_indices=adopted,
                              selected_ids=[binding['documents'][i] for i in adopted],
                              unselected_ids=[binding['documents'][i] for i in range(n) if i not in adopted],
                              p_before=before.tolist(), p_after=after.tolist(), outcome_class=y,
                              final_log_loss=float(-np.log(after[y])), realized_gain=float(np.log(after[y])-np.log(before[y])),
                              f_updates=len(adopted), stopping_position=len(adopted), stop_reason=stop,
                              candidate_count=n, f_model_sha256=frozen.f_model_sha256,
                              g_feature_and_scoring_seconds=score_seconds,
                              no_duplicate_adoption=len(adopted)==len(set(adopted)),
                              all_strategy_budget_comparable=True, trace=decisions)
            paths.append(settle_path(legacy_path, panel.outcomes[mid]))
    for path in paths:
        mid = path['physical_match_id']
        old_mid = path['original_match_id']
        path['main_fit_role'] = panel.matches[mid].get('fit_role', panel.matches[mid]['split_role'])
        path['old_split'] = docs_by_match[old_mid][0]['split']
        if len(path['selected_indices']) == len(panel.sequences[mid]['document_ids']):
            path['stop_reason'] = 'pool_exhausted'
    strata = {}
    for role in sorted({path['main_fit_role'] for path in paths}):
        strata[role] = summarize_paths([path for path in paths if path['main_fit_role'] == role])
    args.output.mkdir(parents=True)
    save_rows(args.output/'paths.jsonl', paths)
    save_rows(args.output/'bindings.jsonl', bindings)
    save_rows(args.output/'exclusions.jsonl', excluded)
    save_json(args.output/'comparison.json', {'scope':'previously_studied_cross_f_transfer_diagnostic_not_independent_validation',
              'matches':len(bindings),'summary':summarize_paths(paths),'by_main_fit_role':strata,
              'old_g_sha256':sha_file(SEMANTIC/'training/models.joblib'), 'new_g_sha256':sha_file(args.g_model/'g-models.joblib'),
              'new_f_sha256':frozen.f_model_sha256, 'old_g_retrained':False,
              'old_g_input_bridge':'Only p_home/p_draw/p_away/entropy/top2 replaced with actual new-f prefix outputs; original pre-label claims/text/time features retained.',
              'limitations':['Original g was trained/selected on this historical panel; this is not new out-of-sample evidence.',
                             'Prior and current f are supplied by the new frozen experiment, not copied old after outputs.',
                             'Restricted to exact previously reviewed candidate pools eligible under both protocols.',
                             'All-documents is an uncapped reference and may exceed budgets 1 or 2; per-path budget-comparability flags are retained.'],
              'network_requests':0})
    print(json.dumps({'complete':True,'matches':len(bindings),'excluded':len(excluded),'output':str(args.output)}))


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
