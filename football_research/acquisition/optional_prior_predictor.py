"""Same visible-context softmax with an explicit training-learned missing offset.

An absent market snapshot stays absent in source records. Its model offset is
the empirical label frequency of independent A fixtures, learned before the
frozen training cutoff. Context corrections require observed text support in
the corresponding prior domain; no missing-prior actor views are synthesized.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .context_predictor import (
    ContextPredictor, ENCODER_VERSION, NUMERIC, ORIGINS, PARAMETERS,
    _digest, _grams, _state, _training,
)
from .state import parse_time
from .visible_state import VERSION as STATE_VERSION

VERSION = 'optional-prior-visible-context-v1'
DOMAIN_VERSION = 'actual-prior-domain-text-support-v1'
BASE_VERSION = 'independent-training-fixture-label-frequency-v1'
_DOMAINS = ('provided_prior','missing_prior')


def _domain(state):
    return 'missing_prior' if state['prior'] is None else 'provided_prior'


def _supported(counts):
    return sum(counts.values())>=PARAMETERS['min_physical_matches'] and all(
        counts.get(str(y),0)>=PARAMETERS['min_class_matches'] for y in range(3))


def _domains(rows):
    result = {}
    for domain in _DOMAINS:
        members = {r['physical_match_id']:r['outcome'] for r in rows if _domain(r['state'])==domain}
        with_text = {r['physical_match_id']:r['outcome'] for r in rows if _domain(r['state'])==domain
                     and any(_grams(r['state']['texts']))}
        counts = {str(y):sum(value==y for value in members.values()) for y in range(3)}
        text_counts = {str(y):sum(value==y for value in with_text.values()) for y in range(3)}
        result[domain] = {'physical_matches':len(members),'physical_match_ids':sorted(members),
            'class_counts':counts,'text_physical_matches':len(with_text),
            'text_physical_match_ids':sorted(with_text),'text_class_counts':text_counts,
            'context_supported':_supported(text_counts),
            'sources':sorted({source for row in rows if _domain(row['state'])==domain for source in row['state']['sources']})}
    return result


def _prepared(records, as_of, evidence_origin):
    rows, aliases, ready = _training(records,as_of,evidence_origin,allow_missing_prior=True)
    ready = deepcopy(ready)
    ready.update(prior_domains=_domains(rows),synthetic_mask_views=0,
        prior_domain_gate='12_actual_text_physical_matches_and_three_classes_per_domain',
        support_scope='global_fit_gate_plus_explicit_actual_prior_domain_not_conditional_calibration')
    if rows and not any(any(_grams(r['state']['texts'])) for r in rows):
        ready['reasons'].append('no_visible_text_vocabulary')
        ready['ready_to_fit']=False
    return rows,aliases,ready


def training_readiness(records, *, as_of, evidence_origin):
    """No fitting, offset fabrication or numerical estimate for an unready set."""
    return _prepared(records,as_of,evidence_origin)[2]


class OptionalPriorContextPredictor(ContextPredictor):
    version = VERSION

    def fit(self, records, *, as_of, dataset_version, evidence_origin):
        rows,aliases,ready = _prepared(records,as_of,evidence_origin)
        if not ready['ready_to_fit']:
            raise ValueError('Optional-prior context predictor not_ready: '+', '.join(ready['reasons']))
        if not isinstance(dataset_version,str) or not dataset_version.strip():
            raise ValueError('Explicit input dataset version required')
        executed = datetime.now(timezone.utc)
        if parse_time(as_of)>executed:
            raise ValueError('Training cutoff cannot follow actual model fitting')
        # Count an independent physical fixture once, not once per query,
        # provider alias, trajectory, prefix, headline or prior domain.
        counts = ready['class_counts']
        base = [counts[str(y)]/ready['physical_matches'] for y in range(3)]
        offsets = deepcopy(rows)
        for row in offsets:
            if row['state']['prior'] is None:
                row['state']['prior']=list(base)
        sizes,result = self._fit_rows(offsets)  # Exact existing encoder/optimizer.
        self.metadata = {'training_as_of':parse_time(as_of).isoformat(),'fit_execution_at':executed.isoformat(),
            'dataset_version':dataset_version,'evidence_origin':evidence_origin,'visible_state_schema':STATE_VERSION,
            'training_physical_match_ids':sorted(sizes),'training_match_ids':sorted(aliases),'training_aliases':aliases,
            'readiness':ready,'training_records_sha256':_digest(rows),'objective_offsets_sha256':_digest(offsets),
            'empirical_base':{'schema':BASE_VERSION,'probabilities':base,'class_counts':deepcopy(counts),
                'physical_matches':ready['physical_matches'],'training_as_of':parse_time(as_of).isoformat(),
                'basis':'one_label_per_independent_A_physical_fixture','smoothing':None},
            'prior_domains':deepcopy(ready['prior_domains']),'prior_domain_schema':DOMAIN_VERSION,
            'match_weighting':'equal_physical_fixture_then_equal_distinct_actual_visible_prefix',
            'encoder_weighting':'one_union_of_visible_texts_per_training_physical_fixture',
            'offset_scope':'provided_original_prior_or_A_label_empirical_base_not_current_posterior',
            'input_scope':'before_only_no_candidate_return_current_posterior_identifier_or_cost_features',
            'synthetic_mask_views':0,'actual_actor_records_rewritten':False,
            'unsupported_domain_behavior':'base_only_context_effect_and_numeric_information_value_unknown',
            'label_scope':'caller_verified_raw_gateway_required_not_proven_by_outcome_integer',
            'calibration_status':'not_established','production_ready':False,
            'optimizer':{'method':'scipy-L-BFGS-B','iterations':int(result.nit),'objective':float(result.fun)}}
        self.model_version=VERSION+':'+_digest(self._payload())[:24]
        return self

    def predict(self, visible_state, *, physical_match_id):
        self._check()
        state = _state(visible_state,allow_missing_prior=True)
        if not isinstance(physical_match_id,str) or not physical_match_id:
            raise ValueError('A canonical physical prediction identity is required')
        if (physical_match_id in self.metadata['training_physical_match_ids']
                or state['provider_match_id'] in self.metadata['training_match_ids']):
            raise ValueError('Training physical fixtures/provider aliases cannot be held-out predictions')
        if parse_time(state['decision_at'])<=parse_time(self.metadata['training_as_of']):
            raise ValueError('Prediction decision must follow the frozen training cutoff')
        domain = _domain(state)
        supported = self.metadata['prior_domains'][domain]['context_supported']
        base = self.metadata['empirical_base']['probabilities'] if domain=='missing_prior' else state['prior']
        vector,known = self._vector(state)
        reason = ('no_visible_text' if not state['texts'] else 'all_text_out_of_vocabulary' if not known
                  else domain+'_context_unsupported' if not supported else None)
        unknown_sources = sorted(set(state['sources'])-set(self.metadata['prior_domains'][domain]['sources']))
        learned_residual = vector@np.asarray(self.coefficients)
        applied = np.zeros(3) if reason else learned_residual
        probabilities = list(base) if reason else softmax(np.log(base)+applied).tolist()
        return {'schema':VERSION,'model_version':self.model_version,'match_id':state['provider_match_id'],
            'physical_match_id':physical_match_id,'probabilities':probabilities,
            'prior_probabilities':state['prior'],'base_probabilities':list(base),
            'forecast_basis':BASE_VERSION if domain=='missing_prior' else visible_state['before']['prior_basis'],
            'base_source':{'training_as_of':self.metadata['training_as_of'],
                'physical_matches':self.metadata['empirical_base']['physical_matches']}
                if domain=='missing_prior' else {'visible_prior_basis':visible_state['before']['prior_basis']},
            'prior_domain':domain,'prior_domain_context_supported':supported,
            'logit_residual':applied.tolist(),'prediction_kind':'empirical_base_fallback' if reason and domain=='missing_prior'
                else 'provided_prior_fallback' if reason else 'regularized_context_forecast',
            'fallback_reason':reason,'known_character_features':known,'unknown_sources':unknown_sources,
            'support_status':domain+'_context_unsupported' if not supported else 'base_only' if reason
                else 'source_extrapolation' if unknown_sources else 'shared_feature_projection',
            'context_effect_status':'UNKNOWN' if not supported else 'not_calibrated',
            'numeric_information_value':None,'information_value_status':'UNKNOWN_predictor_is_not_a_value_estimator',
            'evidence_origin':self.metadata['evidence_origin'],'calibration_status':'not_established',
            'execution_scope':'research_prediction_not_live_deployment_authorization','production_ready':False}

    def _payload(self):
        return {'schema':VERSION,'parameters':PARAMETERS,'metadata':self.metadata,
                'encoder':self.encoder,'coefficients':self.coefficients}

    def _check(self):
        if not self.model_version or self.model_version!=VERSION+':'+_digest(self._payload())[:24]:
            raise ValueError('Frozen optional-prior predictor content/version mismatch')

    @classmethod
    def load(cls,path):
        payload=json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('schema')!=VERSION or payload.get('parameters')!=PARAMETERS:
            raise ValueError('Unsupported optional-prior predictor schema or fixed parameters')
        model=cls()
        model.metadata,model.encoder,model.coefficients=[payload[k] for k in ('metadata','encoder','coefficients')]
        model.model_version=payload.get('model_version')
        model._check()
        enc,meta=model.encoder,model.metadata
        features=len(enc['vocabulary'])+len(NUMERIC)+len(enc['sources'])
        ids=meta['training_physical_match_ids']
        counts=meta['readiness']['class_counts']
        base=meta['empirical_base']
        if (enc.get('version')!=ENCODER_VERSION or enc.get('numeric_names')!=list(NUMERIC)
                or not 1<=len(enc['vocabulary'])<=PARAMETERS['max_features']
                or enc['vocabulary']!=sorted(set(enc['vocabulary'])) or enc['sources']!=sorted(set(enc['sources']))
                or len(enc['idf'])!=len(enc['vocabulary']) or any(not math.isfinite(v) or v<1 for v in enc['idf'])
                or np.asarray(model.coefficients).shape!=(features,3) or not np.all(np.isfinite(model.coefficients))
                or meta.get('evidence_origin') not in ORIGINS or meta.get('visible_state_schema')!=STATE_VERSION
                or len(ids)!=len(set(ids)) or len(ids)<PARAMETERS['min_physical_matches']
                or set(counts)!={'0','1','2'} or any(type(n) is not int or n<1 for n in counts.values())
                or sum(counts.values())!=len(ids) or meta['readiness']['physical_matches']!=len(ids)
                or base!={'schema':BASE_VERSION,'probabilities':[counts[str(y)]/len(ids) for y in range(3)],
                    'class_counts':counts,'physical_matches':len(ids),'training_as_of':meta['training_as_of'],
                    'basis':'one_label_per_independent_A_physical_fixture','smoothing':None}
                or set(meta['training_aliases'])!=set(meta['training_match_ids'])
                or set(meta['training_aliases'].values())!=set(ids)
                or not parse_time(meta['training_as_of'])<=parse_time(meta['fit_execution_at'])
                or meta.get('synthetic_mask_views')!=0 or meta.get('actual_actor_records_rewritten') is not False
                or meta.get('prior_domain_schema')!=DOMAIN_VERSION
                or set(meta['prior_domains'])!=set(_DOMAINS)
                or meta['prior_domains']!=meta['readiness']['prior_domains']):
            raise ValueError('Invalid optional-prior encoder, empirical base or training support')
        for domain in _DOMAINS:
            d=meta['prior_domains'][domain]
            if (d['physical_match_ids']!=sorted(set(d['physical_match_ids']))
                    or not set(d['physical_match_ids'])<=set(ids)
                    or d['text_physical_match_ids']!=sorted(set(d['text_physical_match_ids']))
                    or not set(d['text_physical_match_ids'])<=set(d['physical_match_ids'])
                    or d['physical_matches']!=len(d['physical_match_ids'])
                    or d['text_physical_matches']!=len(d['text_physical_match_ids'])
                    or set(d['class_counts'])!={'0','1','2'} or set(d['text_class_counts'])!={'0','1','2'}
                    or any(type(d['text_class_counts'][str(y)]) is not int or
                        not 0<=d['text_class_counts'][str(y)]<=d['class_counts'][str(y)]<=counts[str(y)] for y in range(3))
                    or sum(d['class_counts'].values())!=d['physical_matches']
                    or sum(d['text_class_counts'].values())!=d['text_physical_matches']
                    or type(d['context_supported']) is not bool or d['context_supported']!=_supported(d['text_class_counts'])
                    or d['sources']!=sorted(set(d['sources'])) or not set(d['sources'])<=set(enc['sources'])):
                raise ValueError('Invalid actual prior-domain support; masked views cannot supply matches')
        return model


__all__=['OptionalPriorContextPredictor','training_readiness','VERSION']
