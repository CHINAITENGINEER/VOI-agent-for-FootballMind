"""Local full-body online inference bound to the separately frozen f/g release.

No training, outcome loading, remote inference, downloads or legacy fallback.
The live metadata adaptation changes only historical_reconstruction to zero.
All other metadata are recomputed for the actual adopted prefix.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import threading
from types import SimpleNamespace

import numpy as np
from footballmind.config import (load_config, RELEASE_ID, ENCODER_REVISION, F_SHA256, G_SHA256)

ROOT = Path(__file__).resolve().parents[1]
# Portable compatibility names only. Resolve selected configuration at call time,
# so importing the module cannot override an explicit --config with environment.
RELEASE = ROOT / 'var/releases' / RELEASE_ID
REVISION = ENCODER_REVISION
MODEL_PATH = ROOT / 'var/models/Qwen3-Embedding-0.6B/snapshots' / REVISION
CACHE_ROOT = ROOT / 'var/cache/embeddings/live-postread-qwen-v1'
F_SHA = F_SHA256
G_SHA = G_SHA256
DEFAULT_HEAD = 'fulltext_mlp_seed20261027'
ALLOWED_HEADS = {DEFAULT_HEAD, 'state_ridge', 'fixed_order'}
FORBIDDEN_FIELDS = {'outcome', 'outcome_class', 'y_true', 'realized_gain', 'gain',
                    'p_after', 'prediction_after', 'loss_after', 'loss_before', 'labels'}
_ENCODER_LOCK = threading.RLock()


class BodyEncodingBudgetExceeded(ValueError):
    """Body retained, but fresh full-text encoding exceeds this run's budget."""
    reason = 'body_encoding_token_budget_exceeded'

    def __init__(self, token_count, max_input_tokens):
        self.token_count = int(token_count)
        self.max_input_tokens = int(max_input_tokens)
        super().__init__(self.reason + ': tokens=' + str(self.token_count)
                         + ', max_input_tokens=' + str(self.max_input_tokens)
                         + '; no truncation or model substitution')


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def _no_labels(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_FIELDS:
                raise ValueError('Post-outcome or candidate-after input is forbidden: ' + key)
            _no_labels(child)
    elif isinstance(value, list):
        for child in value:
            _no_labels(child)


def _aware_time(value, field):
    if not isinstance(value, str):
        raise ValueError(field + ' must be an explicit timezone-aware timestamp')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(field + ' is invalid') from exc
    if result.tzinfo is None:
        raise ValueError(field + ' must include timezone')
    return result.astimezone(timezone.utc)


def _load_release(path):
    # Load sklearn's pandas/Arrow native dependencies before Torch on Windows;
    # this affects module initialization only, not frozen model computation.
    import sklearn.preprocessing  # noqa: F401
    source = Path(path) / 'footballmind_defense.py'
    spec = importlib.util.spec_from_file_location('footballmind_live_postread_release', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FreezeApplication(path)


class LivePostreadModel:
    """Encode acquired bodies, score current candidates, predict adopted evidence.

    ``evaluate`` recommends an index but never adopts it. ``predict`` recomputes
    the supplied adopted prefix. Vectors may be a SHA-keyed map or aligned array;
    every body has an exact UTF-8 SHA and every vector is finite unit float32.
    Use ``encode_document`` in live workers to preserve the encoder provenance.
    """

    @classmethod
    def readiness(cls, release_dir=None, model_path=None, *, workspace_config=None):
        workspace = workspace_config or load_config()
        release = Path(release_dir or workspace.release_dir).resolve()
        snapshot = Path(model_path or workspace.encoder_dir).resolve()
        required = [release / 'footballmind_defense.py', release / 'bundle-config.json',
                    release / 'manifest.json', release / 'models/f/model.pt',
                    release / 'models/g/g-models.joblib',
                    snapshot / 'model.safetensors', snapshot / 'config.json',
                    snapshot / 'tokenizer.json', snapshot / 'tokenizer_config.json']
        missing = [str(path) for path in required if not path.is_file()]
        identity_ok = snapshot.name == REVISION
        config_error = None
        try:
            config = json.loads((release / 'bundle-config.json').read_text(encoding='utf-8'))
            if not isinstance(config, dict):
                raise ValueError('Frozen bundle configuration must be an object')
            identity_ok = identity_ok and config.get('f_model_sha256') == F_SHA and config.get('g_model_sha256') == G_SHA
        except (OSError, ValueError) as exc:
            config_error = type(exc).__name__
        from importlib.metadata import version, PackageNotFoundError
        dependencies_missing = []
        for package in ('torch', 'transformers', 'numpy', 'scikit-learn', 'joblib'):
            try:
                version(package)
            except PackageNotFoundError:
                dependencies_missing.append(package)
        return {'ready': not missing and identity_ok and config_error is None and not dependencies_missing,
                'release_dir': str(release), 'model_path': str(snapshot),
                'missing_files': missing, 'identity_matches': identity_ok,
                'config_error': config_error, 'dependencies_missing': dependencies_missing,
                'configuration_fingerprint': workspace.fingerprint(), 'weights_loaded': False,
                'full_digest_validation': 'performed on model load, not readiness',
                'network_requests': 0, 'external_api_required': False}

    def __init__(self, release_dir=None, model_path=None, cache_root=None, head=None, *, workspace_config=None):
        workspace = workspace_config or load_config()
        head = workspace.default_head if head is None else head
        if head not in ALLOWED_HEADS:
            raise ValueError('Unknown or unfrozen post-read head')
        self.release_dir = Path(release_dir or workspace.release_dir).resolve()
        self.model_path = Path(model_path or workspace.encoder_dir).resolve()
        ready = self.readiness(self.release_dir, self.model_path, workspace_config=workspace)
        if not ready['ready']:
            raise FileNotFoundError('Local frozen post-read artifacts unavailable: ' + json.dumps(ready, ensure_ascii=False))
        self.backend = _load_release(self.release_dir)
        self.f = self.backend.f
        self.g = self.backend.g
        if self.f.f_model_sha256 != F_SHA or self.g['f_model_sha256'] != F_SHA:
            raise ValueError('Frozen f/g binding differs from the approved live migration')
        if _sha_file(self.release_dir / self.backend.config['g_model_path']) != G_SHA:
            raise ValueError('Frozen g archive checksum mismatch')
        from fm_defense_core.postread_semantic_embedding import EncoderContract, SemanticCache, LocalSemanticEncoder
        from fm_defense_core.full_history_sequence_metadata import METADATA_NAMES, document_metadata
        self.contract = EncoderContract()
        contracts = json.loads((self.release_dir / 'provenance/encoder-provenance.json').read_text(encoding='utf-8'))
        if asdict(self.contract) != contracts['short_contract']:
            raise ValueError('Encoder implementation contract differs from frozen release')
        self.long_contract = EncoderContract(**contracts['long_contract'])
        self.encoder_class = LocalSemanticEncoder
        self.encoder = None
        self.cache_root = Path(cache_root or workspace.cache_dir)
        self.caches = {c.fingerprint(): SemanticCache(self.cache_root, c) for c in [self.contract, self.long_contract]}
        self.metadata_names = list(METADATA_NAMES)
        if self.metadata_names != list(self.f.config.metadata_names) or self.metadata_names != self.g['metadata_names']:
            raise ValueError('Frozen metadata order mismatch')
        self._document_metadata = document_metadata
        self.head = head
        self.threshold = self.backend.config['models'][head]['threshold']
        self.lock = threading.RLock()
        self._bindings = {'schema': 'footballmind-live-postread-model-v1',
            'release_id': self.backend.config['bundle_id'], 'release_manifest_sha256': _sha_file(self.release_dir / 'manifest.json'),
            'configuration_fingerprint': workspace.fingerprint(),
            'f_model_sha256': F_SHA, 'g_model_sha256': G_SHA, 'g_head': head,
            'threshold': self.threshold, 'encoder': asdict(self.contract),
            'long_encoder': asdict(self.long_contract), 'encoder_native_dimension': 1024,
            'model_prefix_dimension': 256, 'metadata_names': self.metadata_names,
            'g_feature_dimension': 1140 if head == DEFAULT_HEAD else 349 if head == 'state_ridge' else None,
            'metadata_adapter': 'live-postread-causal-metadata-v2-utc-days',
            'metadata_changes': {'historical_reconstruction': {'training': 1.0, 'live': 0.0},
                'publication_age': {'training': 'calendar-day difference using saved match_date/publication date',
                    'live': 'timezone-aware timestamps normalized to UTC calendar days; date-only publication retains its source calendar date',
                    'resolution': 'calendar days, not exact elapsed hours'}},
            'target_mention_rule': 'frozen training rule: whole Unicode tokens, flexible whitespace, longest overlapping explicit alias counted once',
            'live_transfer_calibrated': False, 'training_feature_budgets': [1, 2, 3, 6],
            'cost_scope': 'only explicit not-yet-incurred f incremental cost; acquired body/encoding cost is sunk',
            'legacy_model_fallback': False, 'training_enabled': False, 'outcomes_used': False}

    @property
    def bindings(self):
        return deepcopy(self._bindings)

    def _normalize_document(self, record):
        _no_labels(record)
        if not isinstance(record, dict):
            raise TypeError('Document record must be a dictionary')
        text = record.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Explicit nonempty full text is required')
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if record.get('text_sha256') not in (None, digest):
            raise ValueError('Body SHA does not match the supplied full text')
        identifier = record.get('document_id')
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('Document ID is required')
        doc = deepcopy(record)
        doc.update(text_sha256=digest, body_characters=len(text))
        claims = doc.get('existing_claims')
        if claims is not None and (not isinstance(claims, list) or any(not isinstance(c, dict) for c in claims)):
            raise ValueError('Claims must be an explicit list of records or unknown')
        return doc

    def encode_document(self, record, *, max_input_tokens=None):
        """Actual local Qwen inference or immutable exact-text cache hit."""
        with self.lock, _ENCODER_LOCK:
            if max_input_tokens is not None and (isinstance(max_input_tokens, bool)
                    or not isinstance(max_input_tokens, int) or max_input_tokens < 1):
                raise ValueError('max_input_tokens must be a positive integer or None')
            doc = self._normalize_document(record)
            text = doc['text']
            for contract in (self.contract, self.long_contract):
                cache = self.caches[contract.fingerprint()]
                hit = cache.get(text)
                if hit is not None:
                    vector, metadata = hit
                    self._vector(vector)
                    return self._encoded(doc, vector, metadata, contract, True)
            tokens = None
            if max_input_tokens is not None:
                if self.encoder is not None:
                    tokenizer = self.encoder.tokenizer
                else:
                    tokenizer = getattr(self, '_budget_tokenizer', None)
                    if tokenizer is None:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True,
                            padding_side='left', trust_remote_code=False)
                        self._budget_tokenizer = tokenizer
                tokens = len(tokenizer.encode(text, truncation=False))
                if tokens > max_input_tokens:
                    raise BodyEncodingBudgetExceeded(tokens, max_input_tokens)
            import torch
            threads = torch.get_num_threads()
            try:
                if self.encoder is None:
                    self.encoder = self.encoder_class(self.model_path, self.contract)
                if tokens is None:
                    tokens = len(self.encoder.tokenizer.encode(text, truncation=False))
                contract = self.contract if tokens <= self.contract.max_input_tokens else self.long_contract
                if tokens > contract.max_input_tokens:
                    raise ValueError('Full body exceeds frozen token limit; no truncation or fallback')
                self.encoder.contract = contract
                vector, measured = self.encoder.encode_one(text)
                self._vector(vector)
                measured.update(execution_device='cpu', output_dtype='float32', no_prompt=True,
                                effective_contract=asdict(contract), encoder_model_sha256=contract.model_sha256)
                metadata = self.caches[contract.fingerprint()].put(text, vector, measured)
            finally:
                torch.set_num_threads(threads)
            return self._encoded(doc, vector, metadata, contract, False)

    def _encoded(self, doc, vector, metadata, contract, hit):
        encoding = {**metadata, 'cache_hit': hit, 'text_sha256': doc['text_sha256'],
                    'contract_fingerprint': contract.fingerprint(),
                    'vector_path': str(self.caches[contract.fingerprint()].root / (doc['text_sha256'] + '.npy')),
                    'dimension': 1024, 'prefix_dimension': 256, 'external_api_requests': 0}
        doc['semantic_encoding'] = deepcopy(encoding)
        return {'document': doc, 'vector': vector.copy(), 'encoding': encoding}

    @staticmethod
    def _vector(value):
        v = np.asarray(value)
        if v.dtype != np.float32 or v.shape != (1024,) or not np.isfinite(v).all():
            raise ValueError('Expected finite float32 Qwen vector of dimension 1024')
        if abs(float(np.linalg.norm(v)) - 1.0) > 1e-5:
            raise ValueError('Qwen vector must have unit norm')
        return v

    @staticmethod
    def _binding(doc, fixture):
        if doc.get('target_binding') is not None:
            return doc['target_binding']
        result = {}
        alias_known = []
        for side in ('home', 'away'):
            name = fixture.get(side + '_team')
            aliases = fixture.get(side + '_aliases')
            if aliases is None:
                aliases = [name] if isinstance(name, str) and name else []
            if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
                raise ValueError('Team aliases must be explicitly declared nonempty strings')
            alias_known.append(bool(aliases))
            # Literal mentions only: not team identity proof or a modeled claim.
            spans = set()
            for alias in aliases:
                pattern = r'(?<!\w)' + r'\s+'.join(re.escape(token) for token in alias.split()) + r'(?!\w)'
                for match in re.finditer(pattern, doc['text'], flags=re.I):
                    spans.add(match.span())
            chosen = []
            for start, end in sorted(spans, key=lambda span: (-(span[1] - span[0]), span[0])):
                if not any(start < other_end and end > other_start for other_start, other_end in chosen):
                    chosen.append((start, end))
            result['exact_' + side + '_team_mentions'] = len(chosen)
            owner = doc.get('source_owner_team')
            result['source_target_' + side] = isinstance(owner, str) and isinstance(name, str) and owner.casefold() == name.casefold()
        result['target_aliases_unknown'] = not all(alias_known)
        result['target_aliases_partial'] = not all(fixture.get(s + '_aliases_complete') is True for s in ('home', 'away'))
        result['source_target_unknown'] = not isinstance(doc.get('source_owner_team'), str) or not doc['source_owner_team'].strip()
        return result

    def _evaluator(self, fixture, prior, records, vectors, adopted_indices):
        _no_labels(fixture)
        mid = fixture.get('physical_match_id')
        if not isinstance(mid, str) or not mid:
            raise ValueError('Explicit physical_match_id is required')
        kickoff = _aware_time(fixture.get('kickoff'), 'kickoff')
        if 'prediction_cutoff' in fixture:
            cutoff = _aware_time(fixture['prediction_cutoff'], 'prediction_cutoff')
            if cutoff >= kickoff:
                raise ValueError('Prediction cutoff must precede kickoff')
        docs = [self._normalize_document(d) for d in records]
        ids = [d['document_id'] for d in docs]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate document IDs')
        adopted = list(adopted_indices)
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= len(docs) for i in adopted) or len(set(adopted)) != len(adopted):
            raise ValueError('Invalid or repeated adopted indices')
        if isinstance(vectors, dict):
            vv = [self._vector(vectors[d['text_sha256']]) for d in docs]
        else:
            if len(vectors) != len(docs):
                raise ValueError('Vector/document count differs')
            vv = [self._vector(v) for v in vectors]
        matrix = np.stack(vv) if vv else np.empty((0, 1024), dtype=np.float32)
        for i, doc in enumerate(docs):
            doc['target_binding'] = self._binding(doc, fixture)
            enc = doc.get('semantic_encoding')
            if enc is not None and (enc.get('text_sha256') != doc['text_sha256'] or enc.get('contract_fingerprint') not in self.caches):
                raise ValueError('Document encoder binding mismatch')
            if enc is not None:
                cached = self.caches[enc['contract_fingerprint']].get(doc['text'])
                if cached is None or cached[1]['vector_file_sha256'] != enc.get('vector_file_sha256') or not np.array_equal(cached[0], matrix[i]):
                    raise ValueError('Supplied vector differs from immutable exact-body encoding')
            elif doc['text_sha256'] in self.backend.vectors:
                if not np.array_equal(self.backend.vectors[doc['text_sha256']], matrix[i]):
                    raise ValueError('Supplied vector differs from frozen exact-body vector')
            else:
                raise ValueError('New body must be encoded locally before prediction; vector provenance missing')
        # An explicit static snapshot is necessary because frozen historical
        # document_metadata has historical_reconstruction hardcoded to one.
        # Adopted rows are recomputed at their own prefix; all other rows use
        # exactly the current adopted prefix. No future candidate enters f.
        meta = np.zeros((len(docs), len(self.metadata_names)))
        known = []
        for i in adopted:
            meta[i] = self._live_meta(docs[i], known, kickoff.date().isoformat())
            known.append(docs[i])
        for i in range(len(docs)):
            if i not in adopted:
                meta[i] = self._live_meta(docs[i], known, kickoff.date().isoformat())
        sequence = {'document_ids': ids, 'vectors': matrix, 'meta': meta,
                    'prior': np.asarray(prior, dtype=float), 'metadata_names': self.metadata_names,
                    'match_date': kickoff.date().isoformat()}
        panel = SimpleNamespace(sequences={mid: sequence}, metadata_names=self.metadata_names)
        return self.backend.FrozenEvaluator(self.f, panel), mid, docs, adopted

    def _live_meta(self, doc, adopted, date):
        # Preserve the archived source string. Normalize only the adapter's
        # calculation view, retaining the frozen calendar-day (not hour) feature.
        calculation_doc = dict(doc)
        publication = doc.get('published_at') or doc.get('source_published_at')
        if isinstance(publication, str):
            try:
                parsed = datetime.fromisoformat(publication.replace('Z', '+00:00'))
            except ValueError:
                parsed = None
            if parsed is not None and parsed.tzinfo is not None:
                calculation_doc['published_at'] = parsed.astimezone(timezone.utc).isoformat()
        values = np.asarray(self._document_metadata(calculation_doc, adopted, date), dtype=float)
        values[self.metadata_names.index('historical_reconstruction')] = 0.0
        if values.shape != (len(self.metadata_names),) or not np.isfinite(values).all():
            raise ValueError('Invalid explicit live document metadata')
        return values

    def _prediction(self, evaluator, mid, adopted):
        p = evaluator.predict(mid, adopted)
        return {'probabilities': p.tolist(),
                'used_input_ids': [evaluator.panel.sequences[mid]['document_ids'][i] for i in adopted],
                'hidden_state': evaluator.current_hidden(mid, adopted).tolist(),
                'f_model_sha256': F_SHA, 'f_executed': True,
                'has_adopted_evidence': bool(adopted), 'outcomes_used': False}

    def initial_prediction(self, fixture, prior):
        return self.predict(fixture, prior, [], [], [])

    def predict(self, fixture, prior, documents, vectors, adopted_indices):
        with self.lock:
            evaluator, mid, _, adopted = self._evaluator(fixture, prior, documents, vectors, adopted_indices)
            return self._prediction(evaluator, mid, adopted)

    def evaluate(self, fixture, prior, documents, vectors, adopted_indices, budget=3, cost_loss_units=0.0):
        with self.lock:
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
                raise ValueError('Budget must be a nonnegative integer')
            if isinstance(cost_loss_units, bool) or not np.isfinite(cost_loss_units) or cost_loss_units < 0:
                raise ValueError('Future incremental cost must be finite and nonnegative')
            evaluator, mid, docs, adopted = self._evaluator(fixture, prior, documents, vectors, adopted_indices)
            prediction = self._prediction(evaluator, mid, adopted)
            remaining = [i for i in range(len(docs)) if i not in adopted]
            scored = []
            reason = 'budget_exhausted' if len(adopted) >= budget else 'candidate_pool_exhausted' if not remaining else None
            selected = None
            if reason is None:
                if self.head == 'fixed_order':
                    selected = remaining[0]
                else:
                    xx = np.stack([self.backend.candidate_features(evaluator, mid, adopted, i, budget, self.g['embedding_dimension'])[0] for i in remaining])
                    scores = self.backend.predict_head(self.g['fitted'][self.head], xx, self.g['target_mean'], self.g['target_scale'])
                    if not np.isfinite(scores).all():
                        raise ValueError('Nonfinite frozen g prediction')
                    scored = [{'index': i, 'document_id': docs[i]['document_id'], 'text_sha256': docs[i]['text_sha256'],
                               'score': float(score), 'net_gain': float(score - cost_loss_units)} for i, score in zip(remaining, scores)]
                    winner = int(np.argmax(scores))
                    if scores[winner] - cost_loss_units > self.threshold:
                        selected = remaining[winner]
                    else:
                        reason = 'predicted_gain_below_frozen_threshold'
            return {'prediction': prediction, 'candidate_scores': scored, 'selected_index': selected,
                    'selected_document_id': docs[selected]['document_id'] if selected is not None else None,
                    'stop_reason': reason, 'threshold': self.threshold, 'budget': budget,
                    'cost_loss_units': float(cost_loss_units), 'budget_outside_training_support': budget not in (1, 2, 3, 6),
                    'g_readable_ids': [d['document_id'] for d in docs],
                    'f_used_ids': prediction['used_input_ids'], 'f_calls': evaluator.calls,
                    'g_selection_executed': self.head != 'fixed_order',
                    'candidate_body_embedding_used': self.head == DEFAULT_HEAD,
                    'bindings': self.bindings, 'new_after_prediction': None}
