"""Entity-aware semantic f for the bounded FootballMind research corpus.

This is a new, opt-in f representation.  The frozen original semantic f uses
binary ``team:kind:value`` predicates, which necessarily maps two different
players with the same status to the same vector.  This module preserves that
coarse representation for generalisation, and counts distinct source entities
per status. Entity identity is used for deduplication, not learned name weights.
There is no dictionary fitted from unread bodies or validation matches. New
names are represented immediately when actually revealed. Counts refer to
source entities, including collective assertions, NOT verified player counts.
No player importance, position, alias, outcome or market label is hand assigned.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .context_predictor import _digest
from .historical_context_training import _project
from .historical_predictor_selection import _optimize
from .match_conditioned_f import bind_prior
from .news_link_projection import VERSION as NEWS_PROJECTION
from .semantic_f import EXTRACTION_VERSION, FEATURES, public_input, validate_response
from .state import parse_time
from .text_returns import _p

VERSION = 'quoted-entity-count-semantic-match-residual-f-v1'
REPRESENTATION_VERSION = 'distinct-source-entity-predicate-counts-v1'
ENTITY_FEATURE_PREFIX = 'entity:'
CHANNEL_FEATURES = ['coarse:'+x for x in FEATURES] + ['entity_count:'+x for x in FEATURES]
PARAMETERS = {
    'regularizations': [None, 10., 1., .1],
    'warmup_matches': 6,
    'validation_block_matches': 3,
    'minimum_fold_matches': 3,
    'selection_rule': 'simplest_within_one_paired_standard_error_of_best',
    'coarse_aggregation': 'binary_assertion_presence_not_document_frequency',
    'entity_aggregation': 'distinct_source_entity_counts_not_document_frequency_or_verified_player_count',
    'channels': 'known_titles_and_novel_body_predicates',
    'interactions': 'home_away_prior_balance',
    'coarse_features': FEATURES,
    'channel_features': CHANNEL_FEATURES,
    'text_projection': NEWS_PROJECTION,
    'extraction_version': EXTRACTION_VERSION,
    'representation_version': REPRESENTATION_VERSION,
    'entity_identity': 'casefolded_whitespace_normalized_source_entity_exact_match',
    'unknown_entity_policy': 'all_revealed_names_counted_without_training_name_vocabulary',
    'status_disagreement': 'retain_each_source_status_no_last_write_wins_or_truth_imputation',
    'vector_order': 'title_coarse_title_count_body_coarse_body_count_then_prior_interactions',
}
REPRESENTATION_SHA256 = _digest(PARAMETERS)


def normalize_entity(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('A source entity name is required')
    return ' '.join(value.casefold().split())


def predicate_key(claim):
    """Exact source predicate identity; no alias invention or fuzzy matching."""
    return (claim['team'], normalize_entity(claim['entity']), claim['kind'], claim['value'])


def _modeled(claim):
    return (claim['scope'] == 'this_match' and claim['team'] in ('home', 'away')
            and claim['kind'] != 'other' and ':'.join((claim['team'], claim['kind'], claim['value'])) in FEATURES)


def _entity_label(token):
    team, entity, kind, value = token
    return ENTITY_FEATURE_PREFIX + '|'.join((team, entity, kind, value))


class EntityEvidenceBank:
    """Frozen extractor bank with entity-level set projection.

    The original ``EvidenceBank`` remains the validator of the response
    protocol.  This wrapper changes only the representation; it does not
    create a second transport, extraction, or source archive.
    """
    def __init__(self, records):
        from .semantic_f import EvidenceBank as CoarseEvidenceBank
        records = deepcopy(records)
        self._coarse = CoarseEvidenceBank(records)
        self.records = self._coarse.records
        self.sha256 = self._coarse.sha256
        self._record_hashes = {key: _digest(record) for key, record in self.records.items()}
        self.representation_sha256 = REPRESENTATION_SHA256

    @staticmethod
    def _items(checked):
        pieces = {'title': {}, 'body': {}}
        for doc in [*checked['observed_links'], *checked['documents']]:
            if doc.get('title', '').strip():
                item = public_input(checked['fixture'], doc, 'title')
                pieces['title'][item['input_id']] = item
        for doc in checked['documents']:
            item = public_input(checked['fixture'], doc, 'body')
            pieces['body'][item['input_id']] = item
        return pieces

    def project(self, state):
        checked = _project(state, text_projection=NEWS_PROJECTION)
        if checked['prior'] is None:
            raise ValueError('Frozen match prior required')
        pieces = self._items(checked)
        entities = {channel: set() for channel in ('title', 'body')}
        used, ignored = [], []
        for channel, items in pieces.items():
            for key, public in sorted(items.items()):
                record = self.records.get(key)
                if record is None:
                    raise ValueError('SEMANTIC_EXTRACTION_MISSING:' + key)
                if record['input'] != public or _digest(record) != self._record_hashes.get(key):
                    raise ValueError('Frozen semantic extraction was modified')
                claims = validate_response(public, record['response'])
                used.append(key)
                for claim in claims:
                    if _modeled(claim):
                        entities[channel].add(predicate_key(claim))
                    else:
                        ignored.append({'input_id': key, 'claim': claim,
                                        'reason': 'outside_frozen_match_feature_support'})
        title_entities = entities['title']
        novel_entities = entities['body'] - title_entities
        def counts(tokens):
            grouped = Counter(':'.join((t[0], t[2], t[3])) for t in tokens)
            return [float(grouped[f]) for f in FEATURES]
        title_count, novel_count = counts(title_entities), counts(novel_entities)
        availability = {}
        for team, entity, kind, value in title_entities | entities['body']:
            if kind == 'availability':
                availability.setdefault((team, entity), set()).add(value)
        # A body restatement is not new information; subtraction occurs at the
        # full entity-predicate level, rather than only at coarse status level.
        return {
            'physical_match_id': checked['physical_match_id'],
            'decision_at': checked['decision_at'],
            'prior': checked['prior'],
            'title': [float(n > 0) for n in title_count],
            'novel_body': [float(n > 0) for n in novel_count],
            'title_entity_counts': title_count,
            'novel_body_entity_counts': novel_count,
            'title_entity_keys': [_entity_label(t) for t in sorted(title_entities)],
            'novel_body_entity_keys': [_entity_label(t) for t in sorted(novel_entities)],
            'multiple_status_entities': [{'team': team, 'entity': entity, 'statuses': sorted(values)}
                for (team, entity), values in sorted(availability.items()) if len(values) > 1],
            'entity_count_scope': 'distinct_source_entities_not_verified_individual_players',
            'representation_sha256': self.representation_sha256,
            'used_input_ids': used,
            'unmodeled_claims': ignored,
            'extraction_status': 'source_semantics_quote_checked_not_independent_truth',
        }


def vector(state):
    if not isinstance(state, dict) or state.get('prior') is None:
        raise ValueError('Entity semantic state with frozen prior required')
    _p(state['prior'])
    if state.get('representation_sha256') != REPRESENTATION_SHA256:
        raise ValueError('Entity representation binding differs')
    channels = []
    for name in ('title', 'novel_body'):
        coarse = np.asarray(state[name], dtype=float)
        counts = np.asarray(state[name+'_entity_counts'], dtype=float)
        if (coarse.shape != (len(FEATURES),) or counts.shape != coarse.shape
                or not np.all(np.isfinite(counts)) or np.any(counts < 0)
                or np.any(counts != np.floor(counts))
                or not np.array_equal(coarse, (counts > 0).astype(float))):
            raise ValueError('Consistent nonnegative distinct-entity counts required')
        channels.append(np.concatenate([coarse, counts]))
    base = np.concatenate(channels)
    return np.concatenate([base, (state['prior'][0]-state['prior'][2])*base])


def _fit(rows, regularization):
    counts = Counter(r['physical_match_id'] for r in rows)
    holder = type('EntityEncoderHolder', (), {'encoder': {
        'representation_version': REPRESENTATION_VERSION,
        'feature_count': len(vector(rows[0]['state'])),
    }})()
    design = (holder, np.asarray([vector(r['state']) for r in rows]),
              np.asarray([r['outcome'] for r in rows]),
              np.log([r['state']['prior'] for r in rows]),
              np.asarray([1./(len(counts)*counts[r['physical_match_id']]) for r in rows]))
    fitted, info = _optimize(design, regularization)
    if not info['success']:
        raise ValueError('Entity semantic residual optimization failed')
    return fitted.coefficients, info


def _predict(coefficients, state):
    return softmax(np.log(state['prior']) + vector(state) @ np.asarray(coefficients))


class EntitySemanticF:
    def __init__(self, coefficients, metadata):
        self.coefficients, self.metadata = deepcopy((coefficients, metadata))
        expected = 4*len(CHANNEL_FEATURES)
        if np.asarray(coefficients).shape != (expected, 3) or not np.all(np.isfinite(coefficients)):
            raise ValueError('Invalid entity semantic coefficients')
        self.model_version = VERSION + ':' + _digest(self.payload())[:24]

    def payload(self):
        return {'schema': VERSION, 'parameters': PARAMETERS,
                'representation_sha256': REPRESENTATION_SHA256,
                'coefficients': self.coefficients, 'metadata': self.metadata}

    def predict(self, state, *, prior_snapshot, evidence_bank):
        if self.model_version != VERSION + ':' + _digest(self.payload())[:24]:
            raise ValueError('Frozen entity semantic f was modified')
        if evidence_bank.representation_sha256 != REPRESENTATION_SHA256:
            raise ValueError('Entity representation differs from frozen model')
        projected = evidence_bank.project(bind_prior(state, prior_snapshot))
        return {'probabilities': _predict(self.coefficients, projected).tolist(),
                'initial_probabilities': projected['prior'], 'model_version': self.model_version,
                'prior_snapshot_sha256': prior_snapshot['snapshot_sha256'],
                'used_semantic_input_ids': projected['used_input_ids'],
                'semantic_bank_sha256': evidence_bank.sha256,
                'entity_representation_sha256': evidence_bank.representation_sha256,
                'projection': projected, 'evidence_origin': 'historical_reconstruction',
                'gain_model_enabled': False, 'production_ready': False}

    def save(self, path):
        with Path(path).open('x', encoding='utf-8') as handle:
            json.dump({**self.payload(), 'model_version': self.model_version}, handle,
                      ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        if (raw.get('schema') != VERSION or raw.get('parameters') != PARAMETERS
                or raw.get('representation_sha256') != REPRESENTATION_SHA256):
            raise ValueError('Unknown entity semantic predictor protocol')
        model = cls(raw['coefficients'], raw['metadata'])
        if model.model_version != raw.get('model_version'):
            raise ValueError('Entity semantic predictor digest mismatch')
        return model


def fit_entity_semantic_f(rows, *, dataset_binding):
    """A-only purged forward selection, preserving the original split protocol."""
    rows, facts = deepcopy(rows), {}
    for row in rows:
        mid = row['physical_match_id']
        fact = (row['outcome'], row['result_available_at'], row['state']['decision_at'])
        if (type(fact[0]) is not int or fact[0] not in (0, 1, 2)
                or row['state']['physical_match_id'] != mid
                or parse_time(fact[1]) <= parse_time(fact[2])
                or (mid in facts and facts[mid] != fact)):
            raise ValueError('One consistent physical match label/decision required')
        vector(row['state'])
        facts[mid] = fact
    ids = sorted(facts, key=lambda mid: (parse_time(facts[mid][2]), mid))
    if len(ids) < PARAMETERS['warmup_matches'] + PARAMETERS['validation_block_matches']:
        raise ValueError('Insufficient independent matches')
    scores = [[] for _ in PARAMETERS['regularizations']]
    folds = []
    for start in range(PARAMETERS['warmup_matches'], len(ids), PARAMETERS['validation_block_matches']):
        valid = ids[start:start+PARAMETERS['validation_block_matches']]
        boundary = min(parse_time(facts[mid][2]) for mid in valid)
        train = [mid for mid in ids[:start] if parse_time(facts[mid][1]) < boundary]
        if len(train) < PARAMETERS['minimum_fold_matches'] or len({facts[mid][0] for mid in train}) < 2:
            raise ValueError('Insufficient purged forward training fold')
        hypotheses = []
        for index, lam in enumerate(PARAMETERS['regularizations']):
            coefficients, fit = _fit([r for r in rows if r['physical_match_id'] in train], lam)
            losses = [float(np.mean([-np.log(_predict(coefficients, r['state'])[r['outcome']])
                for r in rows if r['physical_match_id'] == mid])) for mid in valid]
            scores[index].extend(losses)
            hypotheses.append({'regularization': lam, 'match_losses': dict(zip(valid, losses)),
                               'coefficients_sha256': _digest(coefficients), 'optimizer': fit})
        folds.append({'training_ids': train, 'validation_ids': valid, 'hypotheses': hypotheses,
                      'first_validation_decision': boundary.isoformat()})
    matrix = np.asarray(scores)
    means = matrix.mean(axis=1)
    best = int(np.argmin(means))
    se = (matrix-matrix[best]).std(axis=1, ddof=1)/np.sqrt(matrix.shape[1])
    selected = next(i for i in range(len(means)) if means[i] <= means[best] + se[i] + 1e-12)
    lam = PARAMETERS['regularizations'][selected]
    coefficients, optimizer = _fit(rows, lam)
    metadata = {'fit_execution_at': datetime.now(timezone.utc).isoformat(),
        'training_match_ids': ids, 'training_rows_sha256': _digest(rows),
        'training_states': len(rows), 'dataset_binding': deepcopy(dataset_binding),
        'optimizer': optimizer, 'representation_sha256': REPRESENTATION_SHA256,
        'selection': {'folds': folds, 'independent_validation_matches': matrix.shape[1],
            'candidates': [{'regularization': value, 'mean_match_log_loss': float(means[i]),
                'paired_standard_error_to_best': float(se[i])}
                for i, value in enumerate(PARAMETERS['regularizations'])],
            'best_mean_index': best, 'selected_regularization': lam,
            'base_only_selected': lam is None, 'selection_rule': PARAMETERS['selection_rule'],
            'standard_error_scope': 'dependent_small_development_sample_not_confidence_bound'},
        'evidence_origin': 'historical_reconstruction', 'production_ready': False,
        'gain_model_enabled': False,
        'effect_sizes': 'learned_from_A_outcomes_not_source_or_LLM_numbers',
        'representation_note': 'Names deduplicate source predicates; coefficients act on counts, not player identities or importance.'}
    return EntitySemanticF(coefficients, metadata)


def fit_diagnostic_candidate(rows, *, selected_model):
    """A-only best nonzero candidate; keep the original selection decision."""
    if selected_model.model_version != VERSION+':'+_digest(selected_model.payload())[:24]:
        raise ValueError('Frozen selected entity model was modified')
    if _digest(rows) != selected_model.metadata['training_rows_sha256']:
        raise ValueError('Diagnostic candidate requires exact original A training rows')
    options = [c for c in selected_model.metadata['selection']['candidates'] if c['regularization'] is not None]
    best = min(options, key=lambda c: c['mean_match_log_loss'])
    coefficients, optimizer = _fit(rows, best['regularization'])
    metadata = deepcopy(selected_model.metadata)
    metadata.update(fit_execution_at=datetime.now(timezone.utc).isoformat(),
        candidate_role='diagnostic_only_A_forward_best_nonzero',
        applied_regularization=best['regularization'], selected_model_version=selected_model.model_version,
        selected_deployment_regularization=selected_model.metadata['selection']['selected_regularization'],
        optimizer=optimizer, production_ready=False, gain_model_enabled=False,
        selection_warning='Original one-SE selection retained; diagnostic is not an accepted improvement.')
    return EntitySemanticF(coefficients, metadata)
