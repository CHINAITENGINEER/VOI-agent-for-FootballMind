"""Versioned role-by-text channels for the existing optional-offset softmax.

Verified entity mentions determine lexical channels, not scores or causal
importance. The same vocabulary/optimizer and fixed support gates are retained;
old model artifacts and their unconditioned encoders are never reinterpreted.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .context_predictor import ContextPredictor, NUMERIC, ORIGINS, PARAMETERS, _digest, _grams, _training
from .fixture_roles import ROLES, VERSION as ROLE_VERSION, role_state
from .optional_prior_predictor import BASE_VERSION, DOMAIN_VERSION, _domain, _domains, _supported
from .state import parse_time
from .visible_state import VERSION as STATE_VERSION

VERSION = 'role-optional-prior-visible-context-v1'
ENCODER_VERSION = 'verified-role-character-tfidf-context-v1'
ROLE_SUPPORT_VERSION = 'actual-prior-domain-role-coverage-v1'
NORMALIZATION = 'one_l2_norm_over_all_role_lexical_blocks_then_original_numeric_and_source_blocks'


def _role_coverage(rows):
    result = {}
    for domain in ('provided_prior', 'missing_prior'):
        result[domain] = {}
        for role in ROLES:
            members = [row for row in rows if _domain(row['state']) == domain
                       and any(_grams(row['state']['role_texts'][role]))]
            facts = {row['physical_match_id']: row['outcome'] for row in members}
            result[domain][role] = {'physical_matches': len(facts), 'physical_match_ids': sorted(facts),
                'class_counts': {str(y): sum(label == y for label in facts.values()) for y in range(3)},
                'sources': sorted({source for row in members for source in row['state']['sources']})}
    return result


def _prepared(records, as_of, evidence_origin):
    # Reuse the original temporal, provider alias, A-only, class and label gates.
    # Rebuild the distinct-prefix key to retain actual role/text differences.
    _, aliases, ready = _training(records, as_of, evidence_origin, allow_missing_prior=True)
    unique, identities = {}, {}
    for original in records:
        state, physical = role_state(original['visible_state']), original['physical_match_id']
        if state['physical_match_id'] != physical:
            raise ValueError('Verified role identity differs from the training physical fixture')
        scope = state['identity_scope']
        fact = {key: deepcopy(scope[key]) for key in ('home_id', 'away_id', 'competition_id', 'kickoff')}
        if physical in identities and identities[physical] != fact:
            raise ValueError('A training physical fixture cannot change its verified home/away roles')
        identities[physical] = fact
        key = _digest([physical, state['prior'], state['role_texts'], state['sources'], state['numeric']])
        row = {'physical_match_id': physical, 'outcome': original['outcome'],
            'result_available_at': parse_time(original['result_available_at']).isoformat(), 'state': state}
        if key not in unique or _digest(row) < _digest(unique[key]):
            unique[key] = row
    rows = [unique[key] for key in sorted(unique)]
    ready = deepcopy(ready)
    ready.update(distinct_prefixes=len(rows), prior_domains=_domains(rows), role_coverage=_role_coverage(rows),
        synthetic_mask_views=0, role_schema=ROLE_VERSION,
        prior_domain_gate='12_actual_text_physical_matches_and_three_classes_per_domain',
        role_gate='coverage_disclosure_not_conditional_calibration_or_utility',
        support_scope='global_fit_gate_plus_actual_prior_domain_with_verified_role_channels')
    if rows and not any(any(_grams(row['state']['texts'])) for row in rows):
        ready['reasons'].append('no_visible_text_vocabulary')
        ready['ready_to_fit'] = False
    return rows, aliases, ready, identities


def training_readiness(records, *, as_of, evidence_origin):
    return _prepared(records, as_of, evidence_origin)[2]


class RoleContextPredictor(ContextPredictor):
    version = VERSION

    def _vector(self, state):
        vocabulary = self.encoder['vocabulary']
        vectors, known = [], 0
        for role in ROLES:
            counts = Counter(_grams(state['role_texts'][role]))
            lexical = np.asarray([(1.+math.log(counts[token])) if counts[token] else 0. for token in vocabulary])
            lexical *= np.asarray(self.encoder['idf'])
            vectors.append(lexical)
            known += len(set(counts).intersection(vocabulary))
        # Role x lexical feature is a genuine independent channel, not a word
        # prepended to flattened text or an entity ID fed to the classifier.
        lexical = np.concatenate(vectors)
        norm = np.linalg.norm(lexical)
        if not norm:
            return np.zeros(len(ROLES)*len(vocabulary)+len(NUMERIC)+len(self.encoder['sources'])), 0
        lexical /= norm
        categories = np.asarray([float(source in state['sources']) for source in self.encoder['sources']])
        if np.linalg.norm(categories):
            categories /= np.linalg.norm(categories)
        return np.concatenate([lexical, np.asarray(state['numeric'])/math.sqrt(len(NUMERIC)), categories]), known

    def fit(self, records, *, as_of, dataset_version, evidence_origin):
        rows, aliases, ready, identities = _prepared(records, as_of, evidence_origin)
        if not ready['ready_to_fit']:
            raise ValueError('Role context predictor not_ready: '+', '.join(ready['reasons']))
        if not isinstance(dataset_version, str) or not dataset_version.strip():
            raise ValueError('Explicit input dataset version required')
        executed = datetime.now(timezone.utc)
        if parse_time(as_of) > executed:
            raise ValueError('Training cutoff cannot follow actual model fitting')
        counts = ready['class_counts']
        base = [counts[str(y)]/ready['physical_matches'] for y in range(3)]
        offsets = deepcopy(rows)
        for row in offsets:
            if row['state']['prior'] is None:
                row['state']['prior'] = list(base)
        # The existing fit routine builds its A-only physical-game vocabulary,
        # calls this subclass's vector, and uses the unchanged fixed optimizer.
        sizes, result = self._fit_rows(offsets)
        self.encoder.update(version=ENCODER_VERSION, role_schema=ROLE_VERSION, role_order=list(ROLES),
                            lexical_normalization=NORMALIZATION)
        self.metadata = {'training_as_of': parse_time(as_of).isoformat(), 'fit_execution_at': executed.isoformat(),
            'dataset_version': dataset_version, 'evidence_origin': evidence_origin, 'visible_state_schema': STATE_VERSION,
            'training_physical_match_ids': sorted(sizes), 'training_match_ids': sorted(aliases),
            'training_aliases': aliases, 'training_identity_scope': identities,
            'readiness': ready, 'training_records_sha256': _digest(rows), 'objective_offsets_sha256': _digest(offsets),
            'empirical_base': {'schema': BASE_VERSION, 'probabilities': base, 'class_counts': deepcopy(counts),
                'physical_matches': ready['physical_matches'], 'training_as_of': parse_time(as_of).isoformat(),
                'basis': 'one_label_per_independent_A_physical_fixture', 'smoothing': None},
            'prior_domains': deepcopy(ready['prior_domains']), 'prior_domain_schema': DOMAIN_VERSION,
            'role_schema': ROLE_VERSION, 'role_coverage': deepcopy(ready['role_coverage']),
            'role_support_schema': ROLE_SUPPORT_VERSION,
            'role_scope': 'verified_entity_mentions_in_original_visible_text_not_full_event_argument_semantics',
            'unattributed_and_ambiguous': 'retained_in_distinct_lexical_channels_not_removed_or_scored_by_rules',
            'unknown_role_behavior': 'explicit_extrapolation_not_known_zero_information_value',
            'match_weighting': 'equal_physical_fixture_then_equal_distinct_actual_role_visible_prefix',
            'encoder_weighting': 'one_union_of_visible_texts_per_training_physical_fixture',
            'offset_scope': 'provided_original_prior_or_A_label_empirical_base_not_current_posterior',
            'input_scope': 'verified_role_by_text_before_only_no_unread_return_current_posterior_identifier_or_cost_features',
            'synthetic_mask_views': 0, 'actual_actor_records_rewritten': False,
            'unsupported_domain_behavior': 'base_only_context_effect_and_numeric_information_value_unknown',
            'label_scope': 'caller_verified_raw_gateway_required_not_proven_by_outcome_integer',
            'calibration_status': 'not_established', 'production_ready': False,
            'optimizer': {'method': 'scipy-L-BFGS-B', 'iterations': int(result.nit), 'objective': float(result.fun)}}
        self.model_version = VERSION+':'+_digest(self._payload())[:24]
        return self

    def role_support(self, state):
        domain = _domain(state)
        roles = [role for role in ROLES if state['role_counts'][role] > 0]
        coverage = self.metadata['role_coverage'][domain]
        unseen = [role for role in roles if coverage[role]['physical_matches'] == 0]
        return {'role_schema': ROLE_VERSION, 'role_identity_scope': deepcopy(state['identity_scope']),
            'role_counts': deepcopy(state['role_counts']),
            'role_support': {'schema': ROLE_SUPPORT_VERSION, 'prior_domain': domain,
                'observed_roles': roles, 'unseen_training_roles': unseen,
                'training_physical_matches_by_role': {role: coverage[role]['physical_matches'] for role in ROLES},
                'calibration_status': 'not_established',
                'scope': 'observed_lexical_role_channels_not_semantic_truth_or_utility'},
            'role_support_status': 'unseen_role_extrapolation' if unseen else 'observed_role_channels_not_calibrated'}

    def predict(self, visible_state, *, physical_match_id):
        self._check()
        state = role_state(visible_state)
        if not isinstance(physical_match_id, str) or not physical_match_id or physical_match_id != state['physical_match_id']:
            raise ValueError('Prediction must use the verified role physical identity')
        if (physical_match_id in self.metadata['training_physical_match_ids']
                or state['provider_match_id'] in self.metadata['training_match_ids']):
            raise ValueError('Training physical fixtures/provider aliases cannot be held-out predictions')
        if parse_time(state['decision_at']) <= parse_time(self.metadata['training_as_of']):
            raise ValueError('Prediction decision must follow the frozen training cutoff')
        domain = _domain(state)
        supported = self.metadata['prior_domains'][domain]['context_supported']
        base = self.metadata['empirical_base']['probabilities'] if domain == 'missing_prior' else state['prior']
        vector, known = self._vector(state)
        reason = ('no_visible_text' if not state['texts'] else 'all_text_out_of_vocabulary' if not known
                  else domain+'_context_unsupported' if not supported else None)
        unknown_sources = sorted(set(state['sources'])-set(self.metadata['prior_domains'][domain]['sources']))
        applied = np.zeros(3) if reason else vector@np.asarray(self.coefficients)
        probabilities = list(base) if reason else softmax(np.log(base)+applied).tolist()
        return {'schema': VERSION, 'model_version': self.model_version, 'match_id': state['provider_match_id'],
            'physical_match_id': physical_match_id, 'probabilities': probabilities,
            'prior_probabilities': state['prior'], 'base_probabilities': list(base),
            'forecast_basis': BASE_VERSION if domain == 'missing_prior' else visible_state['before']['prior_basis'],
            'base_source': {'training_as_of': self.metadata['training_as_of'],
                'physical_matches': self.metadata['empirical_base']['physical_matches']} if domain == 'missing_prior'
                else {'visible_prior_basis': visible_state['before']['prior_basis']},
            'prior_domain': domain, 'prior_domain_context_supported': supported, 'logit_residual': applied.tolist(),
            'prediction_kind': 'empirical_base_fallback' if reason and domain == 'missing_prior'
                else 'provided_prior_fallback' if reason else 'regularized_context_forecast',
            'fallback_reason': reason, 'known_character_features': known, 'unknown_sources': unknown_sources,
            'support_status': domain+'_context_unsupported' if not supported else 'base_only' if reason
                else 'source_extrapolation' if unknown_sources else 'shared_role_feature_projection',
            'context_effect_status': 'UNKNOWN' if not supported else 'not_calibrated',
            'numeric_information_value': None, 'information_value_status': 'UNKNOWN_predictor_is_not_a_value_estimator',
            'evidence_origin': self.metadata['evidence_origin'], 'calibration_status': 'not_established',
            'execution_scope': 'research_prediction_not_live_deployment_authorization', 'production_ready': False,
            **self.role_support(state)}

    def _payload(self):
        return {'schema': VERSION, 'parameters': PARAMETERS, 'metadata': self.metadata,
                'encoder': self.encoder, 'coefficients': self.coefficients}

    def _check(self):
        if not self.model_version or self.model_version != VERSION+':'+_digest(self._payload())[:24]:
            raise ValueError('Frozen role-context predictor content/version mismatch')

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('schema') != VERSION or payload.get('parameters') != PARAMETERS:
            raise ValueError('Unsupported role-context predictor schema or fixed parameters')
        model = cls()
        model.metadata, model.encoder, model.coefficients = [payload[k] for k in ('metadata', 'encoder', 'coefficients')]
        model.model_version = payload.get('model_version')
        model._check()
        enc, meta = model.encoder, model.metadata
        ids, counts = meta['training_physical_match_ids'], meta['readiness']['class_counts']
        features = len(ROLES)*len(enc['vocabulary'])+len(NUMERIC)+len(enc['sources'])
        if (enc.get('version') != ENCODER_VERSION or enc.get('role_schema') != ROLE_VERSION
                or enc.get('role_order') != list(ROLES) or enc.get('lexical_normalization') != NORMALIZATION
                or enc.get('numeric_names') != list(NUMERIC) or not 1 <= len(enc['vocabulary']) <= PARAMETERS['max_features']
                or enc['vocabulary'] != sorted(set(enc['vocabulary'])) or enc['sources'] != sorted(set(enc['sources']))
                or len(enc['idf']) != len(enc['vocabulary']) or any(not math.isfinite(v) or v < 1 for v in enc['idf'])
                or np.asarray(model.coefficients).shape != (features, 3) or not np.all(np.isfinite(model.coefficients))
                or meta.get('evidence_origin') not in ORIGINS or meta.get('visible_state_schema') != STATE_VERSION
                or len(ids) != len(set(ids)) or len(ids) < PARAMETERS['min_physical_matches']
                or set(counts) != {'0', '1', '2'} or any(type(n) is not int or n < 1 for n in counts.values())
                or sum(counts.values()) != len(ids) or meta['readiness']['physical_matches'] != len(ids)
                or meta['empirical_base'] != {'schema': BASE_VERSION, 'probabilities': [counts[str(y)]/len(ids) for y in range(3)],
                    'class_counts': counts, 'physical_matches': len(ids), 'training_as_of': meta['training_as_of'],
                    'basis': 'one_label_per_independent_A_physical_fixture', 'smoothing': None}
                or set(meta['training_aliases']) != set(meta['training_match_ids'])
                or set(meta['training_aliases'].values()) != set(ids) or set(meta['training_identity_scope']) != set(ids)
                or not parse_time(meta['training_as_of']) <= parse_time(meta['fit_execution_at'])
                or meta.get('synthetic_mask_views') != 0 or meta.get('actual_actor_records_rewritten') is not False
                or meta.get('role_schema') != ROLE_VERSION or meta.get('role_support_schema') != ROLE_SUPPORT_VERSION
                or meta.get('prior_domain_schema') != DOMAIN_VERSION
                or set(meta['prior_domains']) != {'provided_prior', 'missing_prior'}
                or meta['prior_domains'] != meta['readiness']['prior_domains']
                or meta['role_coverage'] != meta['readiness']['role_coverage']):
            raise ValueError('Invalid role encoder, empirical base or physical training support')
        for domain, d in meta['prior_domains'].items():
            if (d['physical_match_ids'] != sorted(set(d['physical_match_ids'])) or not set(d['physical_match_ids']) <= set(ids)
                    or d['text_physical_match_ids'] != sorted(set(d['text_physical_match_ids']))
                    or not set(d['text_physical_match_ids']) <= set(d['physical_match_ids'])
                    or d['physical_matches'] != len(d['physical_match_ids']) or d['text_physical_matches'] != len(d['text_physical_match_ids'])
                    or set(d['class_counts']) != {'0', '1', '2'} or set(d['text_class_counts']) != {'0', '1', '2'}
                    or any(type(d['text_class_counts'][str(y)]) is not int or not 0 <= d['text_class_counts'][str(y)] <= d['class_counts'][str(y)] <= counts[str(y)] for y in range(3))
                    or sum(d['class_counts'].values()) != d['physical_matches'] or sum(d['text_class_counts'].values()) != d['text_physical_matches']
                    or type(d['context_supported']) is not bool or d['context_supported'] != _supported(d['text_class_counts'])
                    or d['sources'] != sorted(set(d['sources'])) or not set(d['sources']) <= set(enc['sources'])
                    or set(meta['role_coverage'].get(domain, {})) != set(ROLES)):
                raise ValueError('Invalid actual prior-domain or role support')
            for coverage in meta['role_coverage'][domain].values():
                members, classes = coverage['physical_match_ids'], coverage['class_counts']
                if (members != sorted(set(members)) or not set(members) <= set(d['text_physical_match_ids'])
                        or coverage['physical_matches'] != len(members) or set(classes) != {'0', '1', '2'}
                        or any(type(classes[str(y)]) is not int or not 0 <= classes[str(y)] <= d['text_class_counts'][str(y)] for y in range(3))
                        or sum(classes.values()) != len(members) or coverage['sources'] != sorted(set(coverage['sources']))
                        or not set(coverage['sources']) <= set(d['sources'])):
                    raise ValueError('Invalid independent role-channel coverage')
        return model


__all__ = ['RoleContextPredictor', 'training_readiness', 'VERSION', 'BASE_VERSION', 'ROLE_SUPPORT_VERSION']
