"""Portable frozen FootballMind research inference and separate settlement."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
import uuid


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class FreezeApplication:
    """No HTTP, training, implicit labels, or fallback predictions.

    The bundle contains fixed development cases and precomputed full-body
    vectors. New text is not encoded by this application.
    """
    def __init__(self, bundle_dir=None):
        self.root = Path(bundle_dir or Path(__file__).resolve().parent).resolve()
        self.manifest = _read(self.root / 'manifest.json')
        self.verify()
        self.config = _read(self.root / 'bundle-config.json')
        sys.path.insert(0, str(self.root / 'src'))
        import fm_defense_core
        if Path(fm_defense_core.__file__).resolve().parent != self.root / 'src/fm_defense_core':
            raise RuntimeError('Another bundle implementation is already loaded')
        import numpy as np
        import torch
        import joblib
        from fm_defense_core.sequential_evidence_f import FrozenF
        from fm_defense_core.sequential_postread_g import FrozenEvaluator, candidate_features, predict_head
        torch.set_num_threads(1)
        self.np = np
        self.FrozenEvaluator = FrozenEvaluator
        self.candidate_features = candidate_features
        self.predict_head = predict_head
        self.f = FrozenF.load(self.root / self.config['f_model_path'])
        self.g = joblib.load(self.root / self.config['g_model_path'])
        if self.g['f_model_sha256'] != self.f.f_model_sha256:
            raise ValueError('g was trained against a different f checkpoint')
        if list(self.f.config.metadata_names) != self.g['metadata_names']:
            raise ValueError('f/g metadata schema mismatch')
        for package, version in self.config['dependency_versions'].items():
            if importlib.metadata.version(package) != version:
                raise RuntimeError(f'Dependency differs: {package}; expected {version}')
        self.cases = {r['case_id']: r for r in _read(self.root / 'data/cases.json')}
        self.vectors = np.load(self.root / 'data/embeddings.npz', allow_pickle=False)
        self.lock = threading.RLock()
        self.completed_runs = {}

    def _path(self, relative):
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Artifact path escapes bundle')
        return p

    def verify(self):
        failed = []
        for relative, expected in self.manifest['files_sha256'].items():
            p = self._path(relative)
            if not p.is_file() or _sha(p) != expected:
                failed.append(relative)
        if failed:
            raise ValueError('Frozen bundle integrity failed: ' + ', '.join(failed))
        return {'status': 'PASS', 'files': len(self.manifest['files_sha256']),
                'bundle_id': self.manifest['bundle_id']}

    def catalog(self):
        return [dict(case_id=r['case_id'], physical_match_id=r['physical_match_id'], **r['match'],
                     candidate_count=len(r['documents']), split_role=r['split_role']) for r in self.cases.values()]

    def get_case(self, case_id):
        r = self.cases[case_id]
        return {'case_id': case_id, 'physical_match_id': r['physical_match_id'], 'match': r['match'],
                'prior': r['prior'], 'prior_provenance': r['prior_provenance'],
                'candidates': [self._candidate(d) for d in r['documents']],
                'defaults': self.config['presentation_default'], 'models': self.config['models'],
                'development_selected_policy': self.config['development_selected_policy'],
                'limitations': self.config['limitations'], 'outcome_available_in_separate_settlement': r['settlement_available']}

    @staticmethod
    def _candidate(d):
        return {k: d.get(k) for k in ['document_id', 'text_sha256', 'title', 'source_url', 'source_domain',
                                      'published_at', 'captured_at', 'source_available_at', 'body_characters',
                                      'historical_version_unknown', 'source_type']}

    def get_evidence(self, case_id, document_id):
        r = self.cases[case_id]
        d = next((d for d in r['documents'] if d['document_id'] == document_id), None)
        if d is None:
            raise KeyError('Document is not in this case')
        p = self._path(d['body_file'])
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != d['text_sha256']:
            raise ValueError('Body SHA differs from frozen semantic vector binding')
        return {**self._candidate(d), 'text': raw.decode('utf-8'), 'body_file': d['body_file'],
                'claims': d.get('existing_claims'), 'target_binding': d.get('target_binding'),
                'claim_note': 'Inherited explicit source assertions; missing claims are UNKNOWN, not invented.'}

    def _panel(self, case):
        from types import SimpleNamespace
        from fm_defense_core.full_history_sequence_metadata import sequence_metadata
        docs = case['documents']; ids = [d['document_id'] for d in docs]
        vv = []
        for d in docs:
            self.get_evidence(case['case_id'], d['document_id'])
            v = self.vectors[d['text_sha256']]
            if v.shape != (1024,) or v.dtype != self.np.float32 or not self.np.isfinite(v).all():
                raise ValueError('Invalid frozen body embedding')
            vv.append(v)
        seq = {'document_ids': ids, 'vectors': self.np.stack(vv),
               'meta': self.np.asarray(sequence_metadata(docs, case['match']['match_date'])),
               'prior': self.np.asarray(case['prior'], dtype=float), 'document_records': docs,
               'match_date': case['match']['match_date'], 'metadata_names': self.g['metadata_names']}
        return SimpleNamespace(sequences={case['physical_match_id']: seq}, metadata_names=self.g['metadata_names'])

    def run_case(self, case_id, model_name=None):
        with self.lock:
            self.verify()
            case = self.cases[case_id]
            name = model_name or self.config['presentation_default']['model_name']
            baseline = name in ('fixed_order', 'all_evidence')
            if name not in self.config['models'] or (not baseline and name not in self.g['fitted']):
                raise ValueError('Unknown frozen g head')
            setting = self.config['models'][name]
            budget = setting['budget']; threshold = setting['threshold']
            panel = self._panel(case); mid = case['physical_match_id']
            evaluator = self.FrozenEvaluator(self.f, panel)
            ids = panel.sequences[mid]['document_ids']; adopted = []; steps = []
            before = evaluator.predict(mid, []).tolist(); stop = 'budget_exhausted'
            started = time.perf_counter()
            limit = len(ids) if name == 'all_evidence' else min(len(ids), budget)
            for index in range(limit):
                remaining = [i for i in range(len(ids)) if i not in adopted]
                p_before = evaluator.predict(mid, adopted).tolist()
                if baseline:
                    scores = None; best = 0; candidate = remaining[0]
                else:
                    xx = self.np.stack([self.candidate_features(evaluator, mid, adopted, i, budget, self.g['embedding_dimension'])[0] for i in remaining])
                    scores = self.predict_head(self.g['fitted'][name], xx, self.g['target_mean'], self.g['target_scale'])
                    if not self.np.isfinite(scores).all():
                        raise ValueError('Nonfinite frozen g score')
                    best = int(self.np.argmax(scores)); candidate = remaining[best]
                row = {'step': index + 1, 'state_before': {'adopted_ids': [ids[i] for i in adopted],
                       'remaining_budget': max(0, budget - len(adopted)), 'remaining_candidates': len(remaining)},
                       'p_before': p_before, 'candidate_scores': [] if baseline else [{'document_id': ids[i], 'score': float(v)} for i, v in zip(remaining, scores)],
                       'threshold': threshold, 'selected_document_id': None, 'p_after': None,
                       'f_executed': False, 'used_input_ids_before': [ids[i] for i in adopted],
                       'used_input_ids_after': None,
                       'adopted_ids': [ids[i] for i in adopted], 'stop_reason': None,
                       'tie_break': 'first candidate in frozen source order',
                       'all_candidate_scores_equal': None if baseline else bool(self.np.max(scores) == self.np.min(scores)),
                       'selection_mode': 'explicit_baseline_no_g_selection' if baseline else 'frozen_g_selection'}
                if not baseline and scores[best] <= threshold:
                    stop = 'predicted_gain_below_frozen_threshold'; row['stop_reason'] = stop; steps.append(row); break
                adopted.append(candidate)
                row['selected_document_id'] = ids[candidate]
                row['p_after'] = evaluator.predict(mid, adopted).tolist()
                row['f_executed'] = True
                row['used_input_ids_after'] = [ids[i] for i in adopted]
                row['adopted_ids'] = [ids[i] for i in adopted]
                steps.append(row)
            if len(adopted) == len(ids):
                stop = 'pool_exhausted'
            result = {'run_id': 'defense-' + uuid.uuid4().hex, 'case_id': case_id, 'physical_match_id': mid,
                      'model_name': name, 'model_role': setting['role'], 'p_before': before,
                      'p_after': evaluator.predict(mid, adopted).tolist(), 'steps': steps,
                      'selected_ids': [ids[i] for i in adopted], 'unselected_ids': [ids[i] for i in range(len(ids)) if i not in adopted],
                      'stop_reason': stop, 'budget': budget, 'threshold': threshold,
                      'body_adoptions': len(adopted), 'candidate_count': len(ids),
                      'final_state_recomputed': True, 'no_new_evidence_at_stop': stop == 'predicted_gain_below_frozen_threshold',
                      'g_readable_ids': list(ids), 'f_used_ids': [ids[i] for i in adopted],
                      'model_bindings': {'f_sha256': self.f.f_model_sha256, 'g_archive_sha256': self.config['g_model_sha256'],
                                         'encoder': self.config['encoder']},
                      'elapsed_seconds': time.perf_counter() - started,
                      'status': 'STOPPED', 'scope': 'historical_frozen_research_inference',
                      'outcomes_used_in_prediction': False, 'training_calls': 0, 'network_requests': 0,
                      'current_head_reads_candidate_body_embedding': False if baseline else self.g['fitted'][name].get('input') == 'full',
                      'g_selection_executed': not baseline, 'same_budget_comparison': name != 'all_evidence' or len(ids) <= budget,
                      'limitations': self.config['limitations']}
            # Copy after inference prevents caller mutation from changing settlement.
            self.completed_runs[result['run_id']] = json.loads(json.dumps(result))
            return result

    def settle_run(self, run):
        self.verify()
        run_id = run if isinstance(run, str) else run.get('run_id')
        if run_id not in self.completed_runs:
            raise ValueError('Only a successfully completed local run can be settled')
        actual = self.completed_runs[run_id]
        labels = _read(self.root / 'settlement/outcomes.json')
        label = labels.get(actual['physical_match_id'])
        if label is None:
            return {'run_id': run_id, 'status': 'outcome_unknown', 'gain': None}
        y = label['outcome_class']
        if isinstance(y, bool) or y not in (0, 1, 2):
            raise ValueError('Invalid outcome class')
        before = -math.log(actual['p_before'][y]); after = -math.log(actual['p_after'][y])
        return {'run_id': run_id, 'status': 'settled_historical_diagnostic', 'outcome_class': y,
                'outcome_source': label.get('source_refs'), 'loss_before': before, 'loss_after': after,
                'realized_gain': before - after, 'gain': before - after,
                'formula': 'loss_before - loss_after', 'prediction_completed_before_settlement': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--command', choices=['verify', 'catalog', 'case', 'run', 'evidence'], default='verify')
    parser.add_argument('--case-id'); parser.add_argument('--model'); parser.add_argument('--document-id')
    parser.add_argument('--settle', action='store_true'); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    def offline(*a, **k):
        raise RuntimeError('Frozen bundle CLI prohibits network access')
    socket.create_connection = socket.socket.connect = socket.getaddrinfo = offline
    app = FreezeApplication(args.bundle)
    case_id = args.case_id or next(iter(app.cases))
    if args.command == 'verify': result = app.verify()
    elif args.command == 'catalog': result = app.catalog()
    elif args.command == 'case': result = app.get_case(case_id)
    elif args.command == 'evidence': result = app.get_evidence(case_id, args.document_id)
    else:
        result = app.run_case(case_id, args.model)
        if args.settle: result = {'prediction': result, 'settlement': app.settle_run(result)}
    value = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x', encoding='utf8') as stream: stream.write(value + '\n')
        print(json.dumps({'output': str(args.output.resolve()), 'sha256': _sha(args.output)}))
    else: print(value)


if __name__ == '__main__':
    main()
