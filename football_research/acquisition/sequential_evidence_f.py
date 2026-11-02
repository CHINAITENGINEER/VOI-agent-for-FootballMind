"""Research-only, fixed-parameter prediction from an adopted evidence prefix.

No network, semantic provider, online update, or production model registration.
The prior is a three-way [home, draw, away] distribution.  A sequence consists
only of documents already adopted by f; a caller must not pass an unread pool.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCHEMA = 'footballmind-sequential-evidence-f-research-v1'
HISTORY_DEPENDENT_METADATA = (
    'exact_prior_body_duplicate', 'adopted_claims_unknown', 'repeated_claim_count',
    'existing_multistatus_count', 'candidate_introduced_multistatus_count',
    'candidate_expanded_multistatus_count', 'max_similarity_to_adopted', 'mean_similarity_to_adopted',
)


@dataclass(frozen=True)
class SequenceFConfig:
    architecture: str = 'set_residual_mlp'
    embedding_dim: int = 256
    metadata_names: tuple[str, ...] = ()
    hidden_dim: int = 64
    dropout: float = 0.1
    seed: int = 11
    set_masked_metadata_names: tuple[str, ...] = HISTORY_DEPENDENT_METADATA

    def __post_init__(self):
        if self.architecture not in {'set_residual_mlp', 'gru', 'calibrated_prior'}:
            raise ValueError('Unknown architecture')
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError('Positive dimensions required')
        if len(set(self.metadata_names)) != len(self.metadata_names):
            raise ValueError('Duplicate metadata column name')
        if not 0 <= self.dropout < 1:
            raise ValueError('Invalid dropout')


def checked_prior(prior):
    value = np.asarray(prior, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Finite home/draw/away prior required')
    if (value < 0).any() or (value > 1).any() or not np.isclose(value.sum(), 1, atol=1e-6, rtol=0):
        raise ValueError('Prior must be a probability distribution, not odds or percentages')
    return value


def prefix_vectors(vectors, dimension):
    values = np.asarray(vectors, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0, dimension), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < dimension or not np.isfinite(values).all():
        raise ValueError('Finite document vector matrix with sufficient dimension required')
    values = values[:, :dimension].copy()
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('A document vector cannot be a missing/zero vector')
    return values / norms


class PriorCalibration(nn.Module):
    """Positive scalar log-prior scale plus centered class intercepts (4 params)."""

    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, prior):
        # Numerical protection is explicit; it is not target clipping.
        logp = prior.clamp(min=1e-12).log()
        return logp * self.log_scale.clamp(-5, 5).exp() + self.bias - self.bias.mean()


class SequenceEvidenceNetwork(nn.Module):
    def __init__(self, config: SequenceFConfig):
        super().__init__()
        self.config = config
        self.base = PriorCalibration()
        width = config.embedding_dim + len(config.metadata_names) + 3
        h = config.hidden_dim
        self.register_buffer('meta_mean', torch.zeros(len(config.metadata_names)))
        self.register_buffer('meta_scale', torch.ones(len(config.metadata_names)))
        self.register_buffer('meta_input_mask', torch.tensor([
            0. if config.architecture == 'set_residual_mlp' and name in config.set_masked_metadata_names else 1.
            for name in config.metadata_names]))
        if config.architecture == 'set_residual_mlp':
            self.document_encoder = nn.Sequential(nn.Linear(width, h), nn.Tanh(), nn.Dropout(config.dropout), nn.Linear(h, h), nn.Tanh())
            self.residual = nn.Sequential(nn.Linear(h + 4, h), nn.Tanh(), nn.Dropout(config.dropout), nn.Linear(h, 3))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        elif config.architecture == 'gru':
            self.initial_hidden = nn.Linear(3, h)
            self.gru = nn.GRU(width, h, batch_first=True)
            self.dropout = nn.Dropout(config.dropout)
            self.residual = nn.Linear(h, 3)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

    def forward(self, prior, docs, metadata):
        """Return logits and hidden states for s0,...,sT; never pool future tokens."""
        batch, steps, _ = docs.shape
        base = self.base(prior)
        log_prior = prior.clamp(min=1e-12).log()
        if self.config.architecture == 'calibrated_prior':
            return base[:, None, :].expand(batch, steps + 1, 3), torch.zeros(batch, steps + 1, self.config.hidden_dim, device=prior.device, dtype=prior.dtype)
        if steps == 0:
            hidden = torch.tanh(self.initial_hidden(log_prior)) if self.config.architecture == 'gru' else torch.zeros(batch, self.config.hidden_dim, device=prior.device, dtype=prior.dtype)
            return base[:, None, :], hidden[:, None, :]
        # Prefix-dependent relationships are excluded from the unordered set
        # control. Keeping them per-document would leak original visit order.
        normalized = ((metadata - self.meta_mean) / self.meta_scale) * self.meta_input_mask
        inputs = torch.cat([docs, normalized, log_prior[:, None, :].expand(-1, steps, -1)], dim=-1)
        if self.config.architecture == 'set_residual_mlp':
            encoded = self.document_encoder(inputs)
            count = torch.arange(1, steps + 1, device=prior.device, dtype=prior.dtype)[None, :, None]
            pooled = encoded.cumsum(dim=1) / count
            context = torch.cat([pooled, log_prior[:, None, :].expand(-1, steps, -1), count.log1p().expand(batch, -1, -1)], dim=-1)
            residual = self.residual(context)
            hidden = torch.cat([torch.zeros(batch, 1, self.config.hidden_dim, device=prior.device, dtype=prior.dtype), pooled], dim=1)
        else:
            h0 = torch.tanh(self.initial_hidden(log_prior))
            output, _ = self.gru(inputs, h0[None, :, :].contiguous())
            # Subtract the empty-evidence head so s0 is exactly the base model.
            residual = self.residual(self.dropout(output)) - self.residual(h0)[:, None, :]
            hidden = torch.cat([h0[:, None, :], output], dim=1)
        logits = torch.cat([base[:, None, :], base[:, None, :] + residual], dim=1)
        return logits, hidden

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def set_metadata_normalizer(self, mean, scale):
        mean = np.asarray(mean, dtype=float)
        scale = np.asarray(scale, dtype=float)
        shape = (len(self.config.metadata_names),)
        if mean.shape != shape or scale.shape != shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid train-only metadata normalizer')
        with torch.no_grad():
            self.meta_mean.copy_(torch.as_tensor(mean, dtype=self.meta_mean.dtype))
            self.meta_scale.copy_(torch.as_tensor(scale, dtype=self.meta_scale.dtype))


def match_balanced_prefix_loss(logits, outcome, lengths):
    """Each physical match gets weight 1, each of its n+1 prefixes gets 1/(n+1)."""
    if logits.ndim != 3 or logits.shape[0] != len(lengths):
        raise ValueError('Batch shape mismatch')
    labels = outcome[:, None].expand(-1, logits.shape[1])
    losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
    index = torch.arange(logits.shape[1], device=logits.device)[None, :]
    valid = index <= lengths[:, None]
    return ((losses * valid).sum(dim=1) / (lengths + 1)).mean()


def parameter_counts(network):
    return {'total': sum(p.numel() for p in network.parameters()), 'trainable': sum(p.numel() for p in network.parameters() if p.requires_grad), 'base': sum(p.numel() for p in network.base.parameters())}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_frozen(network, directory, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'model.pt').exists() or (directory / 'model.json').exists():
        raise FileExistsError('Refusing to overwrite an existing research model')
    network.eval()
    torch.save(network.state_dict(), directory / 'model.pt')
    report = {'schema': SCHEMA, 'config': asdict(network.config), 'f_model_sha256': file_sha256(directory / 'model.pt'),
              'parameters': parameter_counts(network), 'production_ready': False, 'runtime_enabled': False,
              'software': {'torch': torch.__version__, 'numpy': np.__version__},
              'probability_order': ['home', 'draw', 'away'], 'input_vector_transform': 'not used by calibrated_prior' if network.config.architecture == 'calibrated_prior' else 'take native prefix then L2 normalize each document',
              'metadata_normalization': 'not used by calibrated_prior' if network.config.architecture == 'calibrated_prior' else 'fit only on f_train documents; stored as checkpoint buffers',
              'effective_metadata_names': [] if network.config.architecture == 'calibrated_prior' else [name for name, active in zip(network.config.metadata_names, network.meta_input_mask.tolist()) if active],
              'masked_history_metadata_for_set_control': [name for name, active in zip(network.config.metadata_names, network.meta_input_mask.tolist()) if not active],
              's0': 'calibrated market prior; no text residual', 'parameter_updates_during_predict': False,
              'label_usage': 'training and evaluation only; excluded from prediction arguments', 'research_metadata': metadata or {}}
    (directory / 'model.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


class FrozenF:
    """Frozen offline research interface. `order` names supplied documents in order.

    `order` does not cause sorting or retrieval.  Omit it only when input IDs are
    unavailable, in which case used_input_ids is null rather than invented IDs.
    """

    def __init__(self, network, f_model_sha256=None, manifest=None):
        self.network = network.cpu().eval()
        self.config = network.config
        self.f_model_sha256 = f_model_sha256
        self.manifest = manifest or {}
        for p in self.network.parameters():
            p.requires_grad_(False)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / 'model.json').read_text(encoding='utf-8'))
        if manifest['schema'] != SCHEMA or file_sha256(directory / 'model.pt') != manifest['f_model_sha256']:
            raise ValueError('Frozen f manifest/checkpoint binding differs')
        config = dict(manifest['config'])
        config['metadata_names'] = tuple(config['metadata_names'])
        config['set_masked_metadata_names'] = tuple(config.get('set_masked_metadata_names', HISTORY_DEPENDENT_METADATA))
        network = SequenceEvidenceNetwork(SequenceFConfig(**config))
        state = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
        network.load_state_dict(state, strict=True)
        return cls(network, manifest['f_model_sha256'], manifest)

    def _inputs(self, prior, doc_vectors, doc_meta, order):
        prior = checked_prior(prior)
        vectors = prefix_vectors(doc_vectors, self.config.embedding_dim)
        count = len(vectors)
        width = len(self.config.metadata_names)
        if doc_meta is None:
            if count and width:
                raise ValueError('Explicit document metadata required; no implicit missing-value facts')
            meta = np.zeros((count, width), dtype=float)
        else:
            meta = np.asarray(doc_meta, dtype=float)
            if count == 0 and meta.size == 0:
                meta = np.zeros((0, width), dtype=float)
            if meta.shape != (count, width) or not np.isfinite(meta).all():
                raise ValueError('Metadata shape, column order or values invalid')
        if order is not None:
            if len(order) != count or any(not isinstance(x, str) or not x for x in order) or len(set(order)) != len(order):
                raise ValueError('Document IDs must be explicit, unique and aligned with vector order')
        return torch.tensor(prior, dtype=torch.float32)[None, :], torch.tensor(vectors, dtype=torch.float32)[None, :, :], torch.tensor(meta, dtype=torch.float32)[None, :, :]

    def predict_prefixes(self, prior, doc_vectors, doc_meta=None, order: Sequence[str] | None = None):
        inputs = self._inputs(prior, doc_vectors, doc_meta, order)
        with torch.inference_mode():
            logits, hidden = self.network(*inputs)
            probabilities = logits.softmax(dim=-1)[0].double().numpy()
            states = hidden[0].double().numpy()
        return [{'prefix_length': i, 'probabilities': probabilities[i].tolist(), 'used_input_ids': list(order[:i]) if order is not None else None,
                 'hidden_state': states[i].tolist(), 'f_model_sha256': self.f_model_sha256, 'parameter_updated': False} for i in range(len(probabilities))]

    def predict(self, prior, doc_vectors, doc_meta=None, order=None):
        return np.asarray(self.predict_prefixes(prior, doc_vectors, doc_meta, order)[-1]['probabilities'])

    def hidden_state(self, prior, doc_vectors, doc_meta=None, order=None):
        return np.asarray(self.predict_prefixes(prior, doc_vectors, doc_meta, order)[-1]['hidden_state'])

    def predict_many(self, prior, vector_sequences, meta_sequences=None, orders=None):
        count = len(vector_sequences)
        if meta_sequences is None:
            meta_sequences = [None] * count
        if orders is None:
            orders = [None] * count
        if len(meta_sequences) != count or len(orders) != count:
            raise ValueError('Sequence batch lengths differ')
        return np.asarray([self.predict(prior, v, m, o) for v, m, o in zip(vector_sequences, meta_sequences, orders)])


class LegacyValidatedFSubset:
    """Old frozen coefficients, only where exact archived typed body inputs bind.

    This is a historical coefficient/projection comparison, not a runtime replay.
    No title evidence is silently added, and missing extraction is never treated
    as an empty claim list.  Every provided source response is protocol checked.
    """

    def __init__(self, model_path, semantic_records, expected_sha256):
        from .entity_semantic_f import EntitySemanticF
        from .semantic_f import validate_response
        if file_sha256(model_path) != expected_sha256:
            raise ValueError('Old f checkpoint SHA differs')
        self.model = EntitySemanticF.load(model_path)
        self.f_model_sha256 = expected_sha256
        self.records = {}
        self.invalid_records = []
        for record in semantic_records:
            public = record.get('input', {})
            if public.get('channel') != 'body':
                continue
            try:
                validate_response(public, record['response'])
            except (ValueError, KeyError) as error:
                self.invalid_records.append({'input_id': public.get('input_id'), 'error': type(error).__name__ + ': ' + str(error)})
                continue
            key = (public['fixture']['physical_match_id'], hashlib.sha256(public['text'].encode('utf-8')).hexdigest())
            self.records.setdefault(key, []).append(record)

    def predict_prefixes(self, prior, documents, physical_match_id, aliases=()):
        from collections import Counter
        from .entity_semantic_f import _modeled, predicate_key, _predict, REPRESENTATION_SHA256
        from .semantic_f import FEATURES
        p = checked_prior(prior)
        if (p <= 0).any():
            raise ValueError('Legacy f requires strictly positive prior')
        identities = {physical_match_id, *aliases}
        accumulated = set()
        used = []
        blocked = False
        def projected():
            counts = Counter(':'.join((key[0], key[2], key[3])) for key in accumulated)
            values = [float(counts.get(name, 0)) for name in FEATURES]
            state = {'prior': p.tolist(), 'representation_sha256': REPRESENTATION_SHA256, 'title': [0.] * len(FEATURES),
                'title_entity_counts': [0.] * len(FEATURES), 'novel_body': [float(v > 0) for v in values], 'novel_body_entity_counts': values}
            return _predict(self.model.coefficients, state).tolist()
        rows = [{'prefix_length': 0, 'probabilities': projected(), 'used_input_ids': [], 'used_document_ids': [], 'status': 'evaluable', 'reason': None}]
        for index, document in enumerate(documents):
            matches = [record for mid in identities for record in self.records.get((mid, document['text_sha256']), [])]
            claim_versions = {json.dumps(record['response']['claims'], ensure_ascii=False, sort_keys=True) for record in matches}
            reason = None
            if not matches:
                reason = 'no_exact_body_and_match_bound_validated_typed_response'
            elif len(claim_versions) != 1:
                reason = 'multiple_incompatible_archived_semantic_responses'
            elif blocked:
                reason = 'earlier_prefix_extraction_missing_or_ambiguous'
            if reason:
                blocked = True
                rows.append({'prefix_length': index + 1, 'probabilities': None, 'used_input_ids': None,
                    'used_document_ids': [d['document_id'] for d in documents[:index + 1]], 'status': 'not_evaluable', 'reason': reason})
                continue
            record = sorted(matches, key=lambda x: x['input']['input_id'])[0]
            used.append(record['input']['input_id'])
            accumulated.update(predicate_key(c) for c in record['response']['claims'] if _modeled(c))
            rows.append({'prefix_length': index + 1, 'probabilities': projected(), 'used_input_ids': list(used),
                'used_document_ids': [d['document_id'] for d in documents[:index + 1]], 'status': 'evaluable', 'reason': None,
                'candidate_claim_count': len(record['response']['claims']), 'modeled_distinct_predicates_in_prefix': len(accumulated)})
        for row in rows:
            row.update(f_model_sha256=self.f_model_sha256, comparison_scope='same supplied market prior and adopted full-body prefix; empty title channel',
                historical_reconstruction=True, runtime_replay=False, source_truth_independently_verified=False)
        return rows
