"""Train new offline sequence-f research models; preserve every old artifact.

Model selection uses only the final time block inside f_fit.  g_fit, g_dev,
validation and test are never fitted or used for early stopping.  No networking.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from football_research.acquisition.sequential_evidence_f import (
    FrozenF, SequenceEvidenceNetwork, SequenceFConfig, PriorCalibration, checked_prior,
    prefix_vectors, match_balanced_prefix_loss, save_frozen, file_sha256, parameter_counts,
    LegacyValidatedFSubset,
)


DEFAULT_META = ('log1p_days_before_match', 'publication_time_unknown', 'historical_version_unknown', 'source_is_official', 'log1p_body_characters', 'historical_reconstruction')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').split('\n') if x.strip()]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_rows(path, records):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        for value in records:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def no_network(*args, **kwargs):
    raise RuntimeError('Offline training only; network disabled')


def load_vectors(path):
    with np.load(path, allow_pickle=False) as z:
        if {'keys', 'vectors'}.issubset(z.files):
            if z['keys'].ndim != 1 or z['vectors'].ndim != 2 or len(z['keys']) != len(z['vectors']) or len(set(map(str, z['keys']))) != len(z['keys']):
                raise ValueError('Vector keys and rows are not a one-to-one matrix')
            return {str(k): np.asarray(v, dtype=np.float64) for k, v in zip(z['keys'], z['vectors'])}
        return {str(k): np.asarray(z[k], dtype=np.float64) for k in z.files}


def load_data(directory, embeddings, dimension, allow_missing=False, baseline_only=False, label_roles=('f_train', 'f_dev')):
    directory = Path(directory)
    matches = read_rows(directory / 'matches.jsonl')
    label_ids = {r['physical_match_id'] for r in matches if label_roles is None or r.get('fit_role') in label_roles}
    # The physical label file is shared; project only permitted match IDs here.
    # Non-f development/evaluation label values are not consumed until the
    # explicit frozen evaluation stage.
    outcomes = {r['physical_match_id']: r for r in read_rows(directory / 'outcomes.jsonl') if r['physical_match_id'] in label_ids}
    docs = {r['document_id']: r for r in read_rows(directory / 'documents.jsonl')}
    sequences = read_rows(directory / 'sequences.jsonl')
    schema = read_json(directory / 'schema.json')
    declared_names = schema.get('document_meta_names', schema.get('metadata_names'))
    if not isinstance(declared_names, list) or any(not isinstance(name, str) or not name for name in declared_names):
        raise ValueError('Explicit saved metadata column names required')
    names = tuple(declared_names)
    vectors = {} if baseline_only else load_vectors(embeddings)
    match_by = {}
    excluded = []
    for r in matches:
        mid = r['physical_match_id']
        if mid in match_by:
            raise ValueError('Duplicate physical match record: ' + mid)
        outcome = outcomes.get(mid, {}).get('outcome_class')
        if r.get('prior') is None or (mid in label_ids and outcome is None):
            excluded.append({'physical_match_id': mid, 'scope': 'base', 'reason': 'prior_or_outcome_missing'})
            continue
        prior = checked_prior(r['prior'])
        if outcome is not None and (isinstance(outcome, bool) or outcome not in (0, 1, 2)):
            raise ValueError('Invalid outcome class: ' + mid)
        match_by[mid] = dict(r, prior=prior, outcome=int(outcome) if outcome is not None else None)
    frozen_summary = read_json(directory / 'dataset-summary.json')
    boundary_date = frozen_summary.get('f_dev_start_date')
    if not boundary_date:
        raise ValueError('Frozen dataset must declare its f_dev boundary; trainer never invents a split')
    for r in match_by.values():
        r['evaluation_role'] = ('f_train' if r['match_date'] < boundary_date else 'f_dev') if r['split_role'] == 'f_fit' else r['split_role']
        if r.get('fit_role') != r['evaluation_role']:
            raise ValueError('Frozen fit_role disagrees with declared date boundary: ' + r['physical_match_id'])
    if baseline_only:
        return list(match_by.values()), [], names, excluded, boundary_date
    result = []
    seen = set()
    for sequence in sequences:
        mid = sequence['physical_match_id']
        if mid in seen:
            raise ValueError('Multiple primary sequences for physical match: ' + mid)
        seen.add(mid)
        if label_roles is not None and mid in match_by and match_by[mid]['evaluation_role'] not in label_roles:
            # Training needs complete vectors only for f_train/f_dev. The other
            # roles are encoded independently and loaded in frozen evaluation.
            continue
        if not sequence.get('eligible') or mid not in match_by:
            excluded.append({'physical_match_id': mid, 'scope': 'sequence', 'reason': 'sequence_ineligible_or_prior_outcome_missing'})
            continue
        ids = sequence['document_ids']
        if len(ids) != len(set(ids)):
            raise ValueError('Repeated exact document ID in primary sequence: ' + mid)
        metadata = np.asarray(sequence['document_meta'], dtype=float)
        if not ids:
            excluded.append({'physical_match_id': mid, 'scope': 'sequence', 'reason': 'no_document'})
            continue
        if metadata.shape != (len(ids), len(names)) or not np.isfinite(metadata).all():
            raise ValueError('Explicit metadata shape/values invalid: ' + mid)
        missing = [i for i in ids if docs[i]['text_sha256'] not in vectors]
        if missing:
            excluded.append({'physical_match_id': mid, 'scope': 'sequence', 'reason': 'frozen_vector_missing', 'document_ids': missing})
            if not allow_missing:
                raise ValueError(f'{mid}: {len(missing)} frozen vectors missing; do not replace with zeros')
            continue
        matrix = prefix_vectors(np.stack([vectors[docs[i]['text_sha256']] for i in ids]), dimension)
        result.append(dict(match_by[mid], document_ids=ids, vectors=matrix, metadata=metadata,
                           sequence_provenance={k: v for k, v in sequence.items() if k not in ('document_meta', 'document_ids')}))
    if not any(r['evaluation_role'] == 'f_train' for r in result) or not any(r['evaluation_role'] == 'f_dev' for r in result):
        raise ValueError('Both f_train and f_dev need text sequences; cannot use g labels to select f')
    return list(match_by.values()), result, names, excluded, boundary_date


def fit_calibration(rows):
    training = [r for r in rows if r['evaluation_role'] == 'f_train']
    if not training:
        raise ValueError('No train-only structural prior rows')
    prior = torch.tensor(np.stack([r['prior'] for r in training]), dtype=torch.float32)
    outcome = torch.tensor([r['outcome'] for r in training])
    model = PriorCalibration()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=.5, max_iter=100, tolerance_grad=1e-8, line_search_fn='strong_wolfe')
    history = []
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(model(prior), outcome)
        # Small, declared stabilizer anchored at the raw-market identity.
        penalty = 1e-4 * (model.log_scale.square() + model.bias.square().mean())
        objective = loss + penalty
        objective.backward()
        history.append({'evaluation': len(history), 'train_log_loss': float(loss.detach()), 'objective': float(objective.detach())})
        return objective
    optimizer.step(closure)
    model.eval()
    return model.state_dict(), history


def training_ids_sha(rows):
    ids = sorted(r['physical_match_id'] for r in rows if r['evaluation_role'] == 'f_train')
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def training_values_sha(rows):
    values = [{'physical_match_id': r['physical_match_id'], 'prior': r['prior'].tolist(), 'outcome_class': r['outcome']}
              for r in sorted(rows, key=lambda r: r['physical_match_id']) if r['evaluation_role'] == 'f_train']
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode('utf-8')).hexdigest()


def calibration_bins(probabilities, labels, bins=10):
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    result = []
    for i in range(bins):
        lower, upper = i / bins, (i + 1) / bins
        mask = (confidence >= lower) & ((confidence <= upper) if i == bins - 1 else (confidence < upper))
        n = int(mask.sum())
        result.append({'lower': lower, 'upper': upper, 'n': n,
            'mean_confidence': float(confidence[mask].mean()) if n else None,
            'observed_accuracy': float(correct[mask].mean()) if n else None})
    return result, float(sum(r['n'] * abs(r['mean_confidence'] - r['observed_accuracy']) for r in result if r['n']) / len(labels))


def plot_calibration(records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    roles = sorted({r['role'] for r in records})
    fig, axes = plt.subplots(2, max(1, math.ceil(len(roles) / 2)), figsize=(12, 7), squeeze=False)
    for ax, role in zip(axes.flat, roles):
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Reference y=x')
        for record in records:
            if record['role'] != role:
                continue
            occupied = [b for b in record['bins'] if b['n']]
            ax.plot([b['mean_confidence'] for b in occupied], [b['observed_accuracy'] for b in occupied], 'o-', label=record['model'])
        ax.set(title=role, xlabel='Mean top-class probability', ylabel='Observed top-class accuracy', xlim=(.3, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(roles):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output / 'prior-calibration.png', dpi=170)
    plt.close(fig)


def evaluate_frozen_stage(args):
    """The only stage allowed to evaluate held-out labels; weights already frozen."""
    freeze = read_json(args.output / 'selection-freeze.json')
    out = args.output / 'evaluation'
    if out.exists():
        raise FileExistsError('Frozen evaluation already exists; do not repeatedly expose holdout results')
    for path, digest in freeze['source_hashes'].items():
        if file_sha256(path) != digest:
            raise ValueError('Dataset or vectors changed since selection freeze: ' + path)
    for folder, digest in freeze['models'].items():
        if file_sha256(args.output / folder / 'model.pt') != digest:
            raise ValueError('Selected model changed since freeze')
    dimensions = {read_json(args.output / folder / 'model.json')['config']['embedding_dim'] for folder in freeze['models']}
    if len(dimensions) != 1:
        raise ValueError('Selected models use different embedding contracts')
    base_rows, sequences, names, exclusions, boundary_date = load_data(args.data, args.embeddings, next(iter(dimensions)), args.allow_missing_vectors, label_roles=None)
    out.mkdir()
    metrics = []
    prediction_lookup = {}
    seen_shas = set()
    for folder, digest in freeze['models'].items():
        if digest in seen_shas:
            continue
        seen_shas.add(digest)
        frozen = FrozenF.load(args.output / folder)
        if tuple(names) != frozen.config.metadata_names:
            raise ValueError('Evaluation metadata order differs')
        prediction_rows = []
        for role in sorted({r['evaluation_role'] for r in sequences}):
            rr = [r for r in sequences if r['evaluation_role'] == role]
            prefix_means, final_losses, final_brier, final_accuracy = [], [], [], []
            for r in rr:
                trace = frozen.predict_prefixes(r['prior'], r['vectors'], r['metadata'], r['document_ids'])
                losses = [float(-np.log(max(step['probabilities'][r['outcome']], 1e-12))) for step in trace]
                final_p = np.asarray(trace[-1]['probabilities'])
                prefix_means.append(float(np.mean(losses)))
                final_losses.append(losses[-1])
                final_brier.append(float(np.square(final_p - np.eye(3)[r['outcome']]).sum()))
                final_accuracy.append(int(final_p.argmax() == r['outcome']))
                for step in trace:
                    prediction_rows.append({'physical_match_id': r['physical_match_id'], 'split_role': role, 'document_order': r['document_ids'],
                        'outcome_class': r['outcome'], 'prior': r['prior'].tolist(), **step,
                        'log_loss': float(-np.log(max(step['probabilities'][r['outcome']], 1e-12)))})
            metrics.append({'model': folder, 'model_sha256': digest, 'role': role, 'physical_matches': len(rr),
                'documents': sum(len(r['document_ids']) for r in rr), 'prefixes_including_s0': sum(len(r['document_ids']) + 1 for r in rr),
                'match_balanced_prefix_log_loss': float(np.mean(prefix_means)), 'final_log_loss': float(np.mean(final_losses)),
                'final_brier_score': float(np.mean(final_brier)), 'final_accuracy': float(np.mean(final_accuracy)),
                'metric_source': 'exact exported prefix probabilities; float64 aggregation; log floor 1e-12'})
        write_rows(out / (folder + '-prefix-predictions.jsonl'), prediction_rows)
        prediction_lookup[folder] = {(r['physical_match_id'], r['prefix_length']): r for r in prediction_rows}
    baseline = FrozenF.load(args.output / 'calibrated-prior')
    text_baseline_metrics = []
    text_baseline_predictions = []
    for role in sorted({r['evaluation_role'] for r in sequences}):
        rr = [r for r in sequences if r['evaluation_role'] == role]
        for name in ('no_text_market_prior', 'no_text_calibrated_prior'):
            losses, briers, accuracy = [], [], []
            for r in rr:
                pp = r['prior'] if name == 'no_text_market_prior' else baseline.predict(r['prior'], [])
                loss = float(-np.log(max(pp[r['outcome']], 1e-12)))
                losses.append(loss); briers.append(float(np.square(pp - np.eye(3)[r['outcome']]).sum())); accuracy.append(int(pp.argmax() == r['outcome']))
                text_baseline_predictions.append({'physical_match_id': r['physical_match_id'], 'role': role, 'model': name,
                    'probabilities': pp.tolist(), 'outcome_class': r['outcome'], 'log_loss': loss, 'adopted_document_ids': []})
            text_baseline_metrics.append({'role': role, 'model': name, 'physical_matches': len(rr),
                'match_balanced_prefix_log_loss': float(np.mean(losses)), 'final_log_loss': float(np.mean(losses)),
                'final_brier_score': float(np.mean(briers)), 'final_accuracy': float(np.mean(accuracy)), 'scope': 'exact same news-bearing matches as selected models'})
    write_json(out / 'text-subset-baseline-metrics.json', text_baseline_metrics)
    write_rows(out / 'text-subset-baseline-predictions.jsonl', text_baseline_predictions)
    baseline_metrics = []
    calibration = []
    for role in sorted({r['evaluation_role'] for r in base_rows}):
        rr = [r for r in base_rows if r['evaluation_role'] == role]
        pp = np.stack([r['prior'] for r in rr]);yy = np.array([r['outcome'] for r in rr])
        with torch.inference_mode():
            calibrated = baseline.network.base(torch.tensor(pp, dtype=torch.float32)).softmax(-1).numpy()
        for name, probs in [('market_prior', pp), ('calibrated_prior', calibrated)]:
            bins, ece = calibration_bins(probs, yy)
            calibration.append({'role': role, 'model': name, 'bins': bins, 'ece_top_label_10bins': ece})
            baseline_metrics.append({'role': role, 'model': name, 'physical_matches': len(rr),
                'log_loss': float(-np.log(np.maximum(probs[np.arange(len(rr)), yy], 1e-12)).mean()),
                'brier_score': float(np.square(probs - np.eye(3)[yy]).sum(1).mean()), 'accuracy': float((probs.argmax(1) == yy).mean()), 'ece_top_label_10bins': ece})
    write_json(out / 'selected-model-metrics.json', metrics)
    write_json(out / 'structural-baseline-metrics.json', baseline_metrics)
    write_json(out / 'structural-calibration.json', calibration)
    plot_calibration(calibration, out)
    legacy_relative = freeze.get('legacy_reference_path')
    legacy_path = args.output / legacy_relative if legacy_relative else None
    if legacy_path is not None and legacy_path.exists():
        if file_sha256(legacy_path) != freeze.get('legacy_reference_sha256'):
            raise ValueError('Old-f reference differs from selection freeze')
        old_rows = [r for r in read_rows(legacy_path) if r['prefix_length'] > 0 and r['status'] == 'evaluable']
        matches_by_id = {r['physical_match_id']: r for r in base_rows}
        common = []
        for old in old_rows:
            key = (old['physical_match_id'], old['prefix_length'])
            if not all(key in lookup for lookup in prediction_lookup.values()):
                continue
            match = matches_by_id[key[0]]
            probs = {'old_f_v1': old['probabilities'], 'market_prior': match['prior'].tolist(),
                     'calibrated_prior': baseline.predict(match['prior'], []).tolist(),
                     **{name: lookup[key]['probabilities'] for name, lookup in prediction_lookup.items()}}
            for name, p in probs.items():
                common.append({'physical_match_id': key[0], 'prefix_length': key[1], 'role': match['evaluation_role'],
                    'model': name, 'probabilities': p, 'outcome_class': match['outcome'],
                    'log_loss': float(-np.log(max(p[match['outcome']], 1e-12)))})
        common_metrics = []
        for role in sorted({r['role'] for r in common}):
            for name in sorted({r['model'] for r in common}):
                rr = [r for r in common if r['role'] == role and r['model'] == name]
                mids = {r['physical_match_id'] for r in rr}
                common_metrics.append({'role': role, 'model': name, 'physical_matches': len(mids), 'nonempty_common_prefixes': len(rr),
                    'pooled_prefix_log_loss': float(np.mean([r['log_loss'] for r in rr])),
                    'match_balanced_common_prefix_log_loss': float(np.mean([np.mean([r['log_loss'] for r in rr if r['physical_match_id'] == mid]) for mid in mids]))})
        write_rows(out / 'old-f-common-prefixes.jsonl', common)
        write_json(out / 'old-f-common-subset-metrics.json', common_metrics)
    write_json(out / 'report.json', {'status': 'frozen_evaluation_completed_once', 'selection_freeze_sha256': file_sha256(args.output / 'selection-freeze.json'),
        'refitted': False, 'hyperparameters_changed': False, 'model_selection_performed': False, 'network_requests': 0,
        'evaluation_labels_used_only_after_freeze': True, 'source_hashes': freeze['source_hashes']})
    write_json(out / 'manifest.json', {str(p.relative_to(out)): file_sha256(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps({'evaluation_output': str(out), 'metrics': metrics}, ensure_ascii=False))


def prepare_legacy_subset(args):
    """No labels opened: produce old-f predictions only with exact source bindings."""
    from football_research.acquisition.semantic_f import public_input, EXTRACTION_VERSION
    model_path = ROOT / 'var/f-entity-semantic-f-shadow-materialization-20261011-v1/model.json'
    expected = '34ce858f02165c20779a05b6bed20f1c121008ea7fbe389b64022508f0d7b583'
    archive = ROOT / 'var/f-semantic-residual-20261010-v1/assistant-records.json'
    old_data = Path('E:/FootballMind/experiments/postread-clean-expansion-20261025-v1')
    correction_path = Path('E:/FootballMind/experiments/unified-f-postread-g-20261025-v1/review-corrections.jsonl')
    paths = [Path('E:/FootballMind/experiments/unified-f-postread-g-expansion-20261025-v1/semantic-reviews.jsonl'), old_data / 'semantic-reviews.jsonl']
    sources = [model_path, archive, correction_path, old_data / 'documents.jsonl', *paths,
               args.data / 'matches.jsonl', args.data / 'documents.jsonl', args.data / 'sequences.jsonl']
    hashes = {str(p): file_sha256(p) for p in sources}
    records = read_json(archive)
    for path in paths:
        records.extend(read_rows(path))
    existing_ids = {r['input']['input_id'] for r in records}
    current_matches = read_rows(args.data / 'matches.jsonl')
    alias_to_match = {}
    for match in current_matches:
        for alias in [match['physical_match_id'], *match.get('aliases', [])]:
            alias_to_match[alias] = match
    old_documents = {r['semantic_input_id']: r for r in read_rows(old_data / 'documents.jsonl')}
    reconstructed = []
    unbound_corrections = []
    # These corrections contain saved typed claims, but not a complete public
    # input envelope. Reconstruct only if the exact saved input_id is reproduced.
    for correction in read_rows(correction_path):
        sid = correction.get('semantic_input_id')
        if not sid or sid in existing_ids or sid not in old_documents:
            continue
        document = old_documents[sid]
        mid = document['physical_match_id']
        match = alias_to_match.get(mid)
        if match is None:
            unbound_corrections.append({'input_id': sid, 'reason': 'explicit_match_alias_missing'})
            continue
        fixture = {'physical_match_id': mid, 'home_name': match['home_team'], 'away_name': match['away_team'], 'kickoff': correction.get('semantic_fixture_kickoff')}
        public = public_input(fixture, {'url': document['source_url'], 'title': document['title'], 'text': document['text']}, 'body')
        if public['input_id'] != sid:
            unbound_corrections.append({'input_id': sid, 'reason': 'exact_public_input_reconstruction_differs'})
            continue
        records.append({'input': public, 'response': {'schema': EXTRACTION_VERSION, 'input_id': sid, 'claims': correction['claims']}})
        reconstructed.append(sid)
    model = LegacyValidatedFSubset(model_path, records, expected)
    documents = {r['document_id']: r for r in read_rows(args.data / 'documents.jsonl')}
    match_by = {r['physical_match_id']: r for r in current_matches}
    out = args.output / args.legacy_reference_name
    if out.exists():
        raise FileExistsError('Legacy subset reference already exists')
    out.mkdir(parents=True)
    predictions = []
    audit = []
    for sequence in read_rows(args.data / 'sequences.jsonl'):
        if not sequence.get('eligible'):
            continue
        match = match_by[sequence['physical_match_id']]
        docs = [documents[key] for key in sequence['document_ids']]
        trace = model.predict_prefixes(match['prior'], docs, match['physical_match_id'], match.get('aliases', []))
        for row in trace:
            predictions.append({'physical_match_id': match['physical_match_id'], 'fit_role': match['fit_role'], **row})
        audit.append({'physical_match_id': match['physical_match_id'], 'fit_role': match['fit_role'], 'documents': len(docs),
            'nonempty_evaluable_prefixes': sum(r['prefix_length'] > 0 and r['status'] == 'evaluable' for r in trace),
            'full_sequence_evaluable': trace[-1]['status'] == 'evaluable', 'final_reason': trace[-1]['reason']})
    write_rows(out / 'legacy-prefix-predictions.jsonl', predictions)
    write_rows(out / 'binding-audit.jsonl', audit)
    report = {'status': 'old_f_same_prior_prefix_reference_prepared', 'model_path': str(model_path), 'model_sha256': expected,
        'sequence_matches': len(audit), 'matches_with_any_nonempty_evaluable_prefix': sum(r['nonempty_evaluable_prefixes'] > 0 for r in audit),
        'full_sequence_evaluable_matches': sum(r['full_sequence_evaluable'] for r in audit),
        'nonempty_evaluable_prefixes': sum(r['nonempty_evaluable_prefixes'] for r in audit),
        'missing_or_ambiguous_extraction_is_not_zero_gain_or_empty_claims': True,
        'reconstructed_exact_input_ids': reconstructed, 'unbound_corrections': unbound_corrections, 'invalid_semantic_records': model.invalid_records,
        'comparison_scope': 'same market prior, no known-title channel, only exact body+match bound validated archived assertions; not an original-runtime replay',
        'labels_file_opened': False, 'outcome_or_loss_evaluated': False, 'network_requests': 0, 'model_fitted': False,
        'source_hashes': hashes, 'source_files_unchanged': all(file_sha256(p) == h for p, h in hashes.items())}
    write_json(out / 'report.json', report)
    write_json(out / 'manifest.json', {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps(report, ensure_ascii=False, indent=2))


def collate(rows, dimension, meta_dim):
    lengths = torch.tensor([len(r['document_ids']) for r in rows], dtype=torch.long)
    steps = int(lengths.max())
    docs = torch.zeros(len(rows), steps, dimension)
    meta = torch.zeros(len(rows), steps, meta_dim)
    for i, r in enumerate(rows):
        n = len(r['document_ids'])
        docs[i, :n] = torch.tensor(r['vectors'], dtype=torch.float32)
        meta[i, :n] = torch.tensor(r['metadata'], dtype=torch.float32)
    return (torch.tensor(np.stack([r['prior'] for r in rows]), dtype=torch.float32), docs, meta,
            torch.tensor([r['outcome'] for r in rows]), lengths)


def evaluate(network, records, batch_size=64):
    if not records:
        return None
    network.eval()
    total_prefix_loss = []
    final_losses = []
    brier = []
    correct = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            prior, docs, meta, labels, lengths = collate(batch, network.config.embedding_dim, len(network.config.metadata_names))
            logits, _ = network(prior, docs, meta)
            lp = logits.log_softmax(dim=-1)
            for i, (r, n) in enumerate(zip(batch, lengths.tolist())):
                per_prefix = -lp[i, :n + 1, r['outcome']]
                probs = lp[i, n].exp().numpy()
                final_losses.append(float(per_prefix[-1]))
                total_prefix_loss.append(float(per_prefix.mean()))
                brier.append(float(np.sum((probs - np.eye(3)[r['outcome']]) ** 2)))
                correct.append(int(np.argmax(probs) == r['outcome']))
    return {'physical_matches': len(records), 'documents': sum(len(r['document_ids']) for r in records),
            'prefixes_including_s0': sum(len(r['document_ids']) + 1 for r in records),
            'match_balanced_prefix_log_loss': float(np.mean(total_prefix_loss)),
            'final_log_loss': float(np.mean(final_losses)), 'final_brier_score': float(np.mean(brier)), 'final_accuracy': float(np.mean(correct))}


def train_one(config, base_state, records, epochs, patience, lr, weight_decay, batch_size):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = SequenceEvidenceNetwork(config)
    model.base.load_state_dict(base_state)
    model.freeze_base()
    training = [r for r in records if r['evaluation_role'] == 'f_train']
    validation = [r for r in records if r['evaluation_role'] == 'f_dev']
    all_meta = np.concatenate([r['metadata'] for r in training])
    scale = all_meta.std(axis=0)
    scale[scale < 1e-8] = 1.
    model.set_metadata_normalizer(all_meta.mean(axis=0), scale)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    history = []
    best = math.inf
    best_state = None
    best_epoch = None
    rng = np.random.default_rng(config.seed)
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            index = rng.permutation(len(training))
            for start in range(0, len(index), batch_size):
                batch = [training[i] for i in index[start:start + batch_size]]
                prior, docs, meta, labels, lengths = collate(batch, config.embedding_dim, len(config.metadata_names))
                optimizer.zero_grad()
                logits, _ = model(prior, docs, meta)
                loss = match_balanced_prefix_loss(logits, labels, lengths)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        train_metrics = evaluate(model, training)
        validation_metrics = evaluate(model, validation)
        score = validation_metrics['match_balanced_prefix_log_loss']
        history.append({'epoch': epoch, 'train': train_metrics, 'f_dev': validation_metrics})
        if score < best - 1e-7:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def plot_curves(histories, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for name, history in histories.items():
        for ax, split in zip(axes, ('train', 'f_dev')):
            ax.plot([r['epoch'] for r in history], [r[split]['match_balanced_prefix_log_loss'] for r in history], label=name, linewidth=1.2)
            ax.set_title('f training matches' if split == 'train' else 'Chronological f_dev matches')
            ax.set_xlabel('Epoch')
            ax.grid(alpha=.25)
    axes[0].set_ylabel('Match-balanced prefix log loss (lower is better)')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'training-curves.png', dpi=170)
    plt.close(fig)


def learning_curve(chosen, base_state, sequences, args):
    """Text-training-size diagnostic; fixed calibration, shape, seed, and f_dev."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    name, network, metadata, _ = chosen
    training = [r for r in sequences if r['evaluation_role'] == 'f_train']
    validation = [r for r in sequences if r['evaluation_role'] == 'f_dev']
    # IDs only, no outcomes/gains/model errors participate in subset membership.
    ranked = sorted(training, key=lambda r: hashlib.sha256((str(network.config.seed) + ':' + r['physical_match_id']).encode()).hexdigest())
    results = []
    directory = args.output / 'learning-curve'
    directory.mkdir()
    for fraction in (.25, .5, 1.):
        count = max(1, math.ceil(len(ranked) * fraction))
        subset = ranked[:count]
        if fraction == 1.:
            fitted = network
            best_epoch = metadata['best_epoch']
            history = read_json(args.output / name / 'training-history.json')
        else:
            fitted, history, best_epoch = train_one(network.config, base_state, subset + validation,
                args.epochs, args.patience, args.learning_rate, args.weight_decay, args.batch_size)
            save_frozen(fitted, directory / f'text-train-{int(fraction * 100)}pct',
                {'diagnostic_only': True, 'selected_shape_from': name, 'base_calibration_unchanged': True,
                 'subset_rule': 'seeded hash of physical_match_id; no outcome-dependent selection',
                 'training_match_ids': [r['physical_match_id'] for r in subset], 'best_epoch': best_epoch})
        result = {'text_training_fraction': fraction, 'text_training_matches': count,
            'text_training_documents': sum(len(r['document_ids']) for r in subset), 'seed': network.config.seed,
            'configuration': name, 'best_epoch': best_epoch, 'train': evaluate(fitted, subset), 'f_dev': evaluate(fitted, validation),
            'f_dev_match_ids': [r['physical_match_id'] for r in validation], 'base_calibration_uses_full_f_train_structural_rows': True,
            'affects_selected_model': False}
        results.append(result)
        write_json(directory / f'history-{int(fraction * 100)}pct.json', history)
    write_json(directory / 'metrics.json', results)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, label in [('train', 'Training subset'), ('f_dev', 'Same chronological f_dev')]:
        ax.plot([r['text_training_matches'] for r in results], [r[key]['match_balanced_prefix_log_loss'] for r in results], marker='o', label=label)
    ax.set_xlabel('Independent text-training matches (25%, 50%, 100%)')
    ax.set_ylabel('Match-balanced prefix log loss')
    ax.set_title('Learning curve: fixed shape, seed and calibrated prior')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'learning-curves.png', dpi=170)
    plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1/data'))
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--output', type=Path, default=Path('E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1/f-models'))
    parser.add_argument('--prefix-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--seeds', default='11,29,47')
    parser.add_argument('--architectures', default='set_residual_mlp,gru')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--allow-missing-vectors', action='store_true')
    parser.add_argument('--baseline-only', action='store_true', help='Train structural prior calibration into output/baseline without document vectors')
    parser.add_argument('--frozen-base', type=Path, help='Reuse a previously frozen calibration model; f_train ID binding must match')
    parser.add_argument('--evaluate-frozen', action='store_true', help='One held-out evaluation after selection-freeze.json exists; never fit a model')
    parser.add_argument('--prepare-legacy-subset', action='store_true', help='Read archived typed responses, recompute old f on exact bound subset without outcome labels')
    parser.add_argument('--legacy-reference-name', default='legacy-f-v1-reference-v2')
    parser.add_argument('--encoder-provenance', type=Path, help='Frozen encoder model/version and representation contract; auto-detected beside embeddings')
    args = parser.parse_args()
    socket.socket.connect = socket.create_connection = socket.getaddrinfo = no_network
    torch.set_num_threads(args.threads)
    if args.prepare_legacy_subset:
        prepare_legacy_subset(args)
        return
    if args.evaluate_frozen:
        if args.embeddings is None:
            parser.error('--embeddings required for frozen text evaluation')
        evaluate_frozen_stage(args)
        return
    if not args.baseline_only and args.embeddings is None:
        parser.error('--embeddings required for text models')
    if args.encoder_provenance is None and args.embeddings is not None and (args.embeddings.parent / 'encoder-provenance.json').exists():
        args.encoder_provenance = args.embeddings.parent / 'encoder-provenance.json'
    if args.baseline_only:
        args.output = args.output / 'baseline'
    elif args.frozen_base is None and (args.output / 'baseline/calibrated-prior/model.json').exists():
        args.frozen_base = args.output / 'baseline/calibrated-prior'
    if args.output.exists() and any(p.name not in ('baseline', 'legacy-f-v1-reference', 'legacy-f-v1-reference-v2', 'implementation-validation.json') for p in args.output.iterdir()):
        raise FileExistsError('Use a new empty f-model output directory')
    started = time.perf_counter()
    originals = [args.data / name for name in ('matches.jsonl', 'outcomes.jsonl', 'documents.jsonl', 'sequences.jsonl', 'schema.json', 'dataset-summary.json')]
    originals.extend([Path(__file__), ROOT / 'football_research/acquisition/sequential_evidence_f.py',
                      ROOT / 'football_research/acquisition/full_history_sequence_metadata.py'])
    if args.embeddings is not None:
        originals.append(args.embeddings)
    if args.encoder_provenance is not None:
        originals.append(args.encoder_provenance)
    encoder_provenance = {'path': str(args.encoder_provenance), 'sha256': file_sha256(args.encoder_provenance),
                          'contract': read_json(args.encoder_provenance)} if args.encoder_provenance is not None else None
    hashes = {str(p): file_sha256(p) for p in originals}
    base_rows, sequences, names, exclusions, boundary_date = load_data(args.data, args.embeddings, args.prefix_dim, args.allow_missing_vectors, args.baseline_only)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / 'exclusions.jsonl', exclusions)
    write_json(args.output / 'split.json', {'selection_rule': 'preserve dataset-frozen f_train/f_dev; boundary based on last chronological 20% of text-bearing f_fit matches', 'f_dev_start_date': boundary_date,
        'roles': {role: [r['physical_match_id'] for r in base_rows if r['evaluation_role'] == role] for role in sorted({r['evaluation_role'] for r in base_rows})},
        'sequence_roles': {role: [r['physical_match_id'] for r in sequences if r['evaluation_role'] == role] for role in sorted({r['evaluation_role'] for r in sequences})},
        'g_fit_outcomes_used_to_fit_f': False, 'source_hashes': hashes})
    if args.frozen_base is not None:
        frozen_base = FrozenF.load(args.frozen_base)
        if frozen_base.config.architecture != 'calibrated_prior' or frozen_base.manifest['research_metadata']['f_train_ids_sha256'] != training_ids_sha(base_rows):
            raise ValueError('Frozen base training population differs from current split')
        binding = args.frozen_base.parent / 'training-value-binding.json'
        if binding.exists():
            if read_json(binding)['f_train_values_sha256'] != training_values_sha(base_rows):
                raise ValueError('Frozen base priors/outcomes differ from current f_train')
        else:
            previous_hashes = frozen_base.manifest['research_metadata']['source_hashes']
            for name in ('matches.jsonl', 'outcomes.jsonl'):
                path = str(args.data / name)
                if previous_hashes.get(path) != hashes[path]:
                    raise ValueError('Cannot establish unchanged frozen-base training values: ' + name)
        base_state = frozen_base.network.base.state_dict()
        calibration_history = {'reused_frozen_base': str(args.frozen_base), 'f_model_sha256': frozen_base.f_model_sha256, 'refitted': False}
    else:
        base_state, calibration_history = fit_calibration(base_rows)
    write_json(args.output / 'calibration-history.json', calibration_history)
    write_json(args.output / 'training-value-binding.json', {'f_train_values_sha256': training_values_sha(base_rows),
        'f_train_ids_sha256': training_ids_sha(base_rows), 'fields': ['physical_match_id', 'prior', 'outcome_class'],
        'roles': ['f_train'], 'labels_outside_f_train_included': False})
    base_network = SequenceEvidenceNetwork(SequenceFConfig(architecture='calibrated_prior', embedding_dim=args.prefix_dim, metadata_names=names, hidden_dim=args.hidden_dim))
    base_network.base.load_state_dict(base_state)
    save_frozen(base_network, args.output / 'calibrated-prior', {'training_role': 'f_train', 'fit_method': 'LBFGS' if args.frozen_base is None else 'reused frozen base', 'l2_penalty': .0001,
        'training_matches': sum(r['evaluation_role'] == 'f_train' for r in base_rows), 'f_train_ids_sha256': training_ids_sha(base_rows), 'source_hashes': hashes})
    structural_metrics = []
    structural_calibration = []
    structural_predictions = []
    for role in ('f_train', 'f_dev'):
        rr = [r for r in base_rows if r['evaluation_role'] == role and r['outcome'] is not None]
        raw = np.stack([r['prior'] for r in rr])
        with torch.inference_mode():
            calibrated = base_network.base(torch.tensor(raw, dtype=torch.float32)).softmax(-1).numpy()
        yy = np.array([r['outcome'] for r in rr])
        for name, pp in [('market_prior', raw), ('calibrated_prior', calibrated)]:
            bins, ece = calibration_bins(pp, yy)
            structural_calibration.append({'role': role, 'model': name, 'bins': bins, 'ece_top_label_10bins': ece})
            structural_metrics.append({'role': role, 'model': name, 'physical_matches': len(rr), 'log_loss': float(-np.log(np.maximum(pp[np.arange(len(rr)), yy], 1e-12)).mean()),
                'brier_score': float(np.square(pp - np.eye(3)[yy]).sum(1).mean()), 'accuracy': float((pp.argmax(1) == yy).mean()), 'ece_top_label_10bins': ece})
        for i, r in enumerate(rr):
            structural_predictions.append({'physical_match_id': r['physical_match_id'], 'role': role, 'outcome_class': r['outcome'],
                'market_prior': raw[i].tolist(), 'calibrated_prior': calibrated[i].tolist()})
    write_json(args.output / 'structural-baseline-metrics.json', structural_metrics)
    write_json(args.output / 'structural-calibration.json', structural_calibration)
    write_rows(args.output / 'structural-predictions.jsonl', structural_predictions)
    plot_calibration(structural_calibration, args.output)
    if args.baseline_only:
        if any(file_sha256(Path(p)) != digest for p, digest in hashes.items()):
            raise ValueError('Input changed during baseline training')
        report = {'status': 'structural_prior_calibration_trained_not_promoted', 'training_role': 'f_train', 'training_matches': sum(r['evaluation_role'] == 'f_train' for r in base_rows),
            'evaluated_roles': ['f_train', 'f_dev'], 'deferred_roles': ['g_fit', 'g_dev', 'validation', 'test'], 'base_parameters': 4, 'prior_bearing_matches': len(base_rows),
            'f_dev_start_date': boundary_date, 'no_document_vectors_required': True, 'network_requests': 0, 'external_api_calls': 0, 'source_hashes': hashes,
            'elapsed_seconds': time.perf_counter() - started, 'production_ready': False, 'original_inputs_unchanged': True,
            'frozen_base_path': str(args.output / 'calibrated-prior'), 'later_text_models_must_reuse_base': True}
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'manifest.json', {str(p.relative_to(args.output)): file_sha256(p) for p in args.output.rglob('*') if p.is_file() and p.name != 'manifest.json'})
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    raw_text_baseline = SequenceEvidenceNetwork(SequenceFConfig(architecture='calibrated_prior', embedding_dim=args.prefix_dim, metadata_names=names, hidden_dim=args.hidden_dim))
    write_json(args.output / 'text-subset-baselines.json', {
        name: {role: evaluate(network, [r for r in sequences if r['evaluation_role'] == role]) for role in ('f_train', 'f_dev')}
        for name, network in [('market_prior', raw_text_baseline), ('calibrated_prior', base_network)]})
    models = []
    histories = {}
    all_metrics = []
    roles = ['f_train', 'f_dev']
    for architecture in args.architectures.split(','):
        if architecture not in ('set_residual_mlp', 'gru'):
            raise ValueError('Only declared set/gru experiments allowed')
        for seed in map(int, args.seeds.split(',')):
            config = SequenceFConfig(architecture=architecture, embedding_dim=args.prefix_dim, metadata_names=names, hidden_dim=args.hidden_dim, seed=seed)
            network, history, best_epoch = train_one(config, base_state, sequences, args.epochs, args.patience, args.learning_rate, args.weight_decay, args.batch_size)
            name = f'{architecture}-seed{seed}'
            metadata = {'selection_role': 'f_dev', 'selection_metric': 'match_balanced_prefix_log_loss', 'best_epoch': best_epoch,
                'training_role': 'f_train', 'optimizer': 'AdamW', 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
                'optimizer_betas': [.9, .999], 'optimizer_epsilon': 1e-8, 'gradient_clip_norm': 5.,
                'batch_size_matches': args.batch_size, 'maximum_epochs': args.epochs, 'early_stopping_patience': args.patience,
                'early_stopping_minimum_improvement': 1e-7, 'checkpoint_rule': 'lowest f_dev match-balanced prefix log loss; restore checkpoint before export',
                'initial_residual_head': 'zeros; epoch0 equals frozen calibrated prior',
                'loss': 'cross entropy averaged over all prefixes including s0, then equally over physical matches',
                'base_frozen_during_text_training': True, 'metadata_fit_only_f_train': True, 'source_hashes': hashes,
                'encoder_provenance': encoder_provenance,
                'f_train_matches': sum(r['evaluation_role'] == 'f_train' for r in sequences), 'f_dev_matches': sum(r['evaluation_role'] == 'f_dev' for r in sequences)}
            manifest = save_frozen(network, args.output / name, metadata)
            write_json(args.output / name / 'training-history.json', history)
            histories[name] = history
            metrics = {role: evaluate(network, [r for r in sequences if r['evaluation_role'] == role]) for role in roles}
            write_json(args.output / name / 'metrics.json', metrics)
            all_metrics.append({'model': name, 'model_sha256': manifest['f_model_sha256'], 'best_epoch': best_epoch, 'parameters': parameter_counts(network), 'metrics': metrics})
            models.append((name, network, metadata, metrics['f_dev']['match_balanced_prefix_log_loss']))
            print(json.dumps({'finished_model': name, 'best_epoch': best_epoch, 'f_dev_prefix_log_loss': models[-1][-1]}, ensure_ascii=False), flush=True)
    chosen = min(models, key=lambda x: (x[3], x[0]))
    selections = {'selected': chosen}
    for architecture in args.architectures.split(','):
        selections['selected-' + architecture] = min([r for r in models if r[1].config.architecture == architecture], key=lambda x: (x[3], x[0]))
    for folder, (name, network, metadata, _) in selections.items():
        save_frozen(network, args.output / folder, dict(metadata, selected_from=name))
        frozen = FrozenF.load(args.output / folder)
        predictions = []
        for r in sequences:
            if r['evaluation_role'] not in ('f_train', 'f_dev'):
                continue
            trace = frozen.predict_prefixes(r['prior'], r['vectors'], r['metadata'], r['document_ids'])
            for step in trace:
                p = step['probabilities']
                predictions.append({'physical_match_id': r['physical_match_id'], 'split_role': r['evaluation_role'], 'document_order': r['document_ids'],
                    'outcome_class': r['outcome'], 'prior': r['prior'].tolist(), **step, 'log_loss': float(-np.log(max(p[r['outcome']], 1e-12)))})
        write_rows(args.output / folder / 'prefix-predictions.jsonl', predictions)
    learning_curve(chosen, base_state, sequences, args)
    legacy_reference = next((args.output / name / 'legacy-prefix-predictions.jsonl' for name in ('legacy-f-v1-reference-v2', 'legacy-f-v1-reference') if (args.output / name / 'legacy-prefix-predictions.jsonl').exists()), None)
    write_json(args.output / 'selection-freeze.json', {'selected_model': chosen[0], 'selection_uses_only': 'f_dev',
        'models': {folder: file_sha256(args.output / folder / 'model.pt') for folder in selections},
        'source_hashes': hashes, 'held_out_evaluation_completed': False,
        'legacy_reference_path': str(legacy_reference.relative_to(args.output)) if legacy_reference else None,
        'legacy_reference_sha256': file_sha256(legacy_reference) if legacy_reference else None,
        'architecture_comparison_caveat': 'Set control masks history-dependent metadata; GRU retains it. Differences cannot be attributed to recurrent architecture alone.'})
    write_json(args.output / 'model-comparison.json', all_metrics)
    plot_curves(histories, args.output)
    if any(file_sha256(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError('Original input changed during training')
    report = {'status': 'research_models_frozen_pending_separate_evaluation', 'selected_model': chosen[0], 'selection_uses_only': 'chronological f_dev inside f_fit',
        'original_inputs_unchanged': True, 'production_ready': False, 'network_requests': 0, 'external_api_calls': 0,
        'structural_matches': len(base_rows), 'news_matches': len(sequences), 'news_documents_occurrences': sum(len(r['document_ids']) for r in sequences),
        'metadata_names': names, 'prefix_dimension': args.prefix_dim, 'native_vector_prefix_normalized': True,
        'encoder_provenance': encoder_provenance,
        'elapsed_seconds': time.perf_counter() - started, 'torch_threads': args.threads, 'source_hashes': hashes,
        'old_f_v1_comparison': 'not evaluated by this new predictor; only existing reproducible subset may be compared separately',
        'g_training': 'not part of this tool; pair g with exact checkpoint SHA and keep g_fit disjoint from f_fit'}
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'manifest.json', {str(p.relative_to(args.output)): file_sha256(p) for p in args.output.rglob('*') if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
