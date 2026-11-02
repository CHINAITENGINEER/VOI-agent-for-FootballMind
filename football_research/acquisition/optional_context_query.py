"""Query-time inputs for a fixed optional-prior predictor's signed gain.

An absent market prior stays absent. The independent A-label base is a separate
forecast input, never a fabricated market observation or a future response.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.special import softmax

from .context_predictor import _state, NUMERIC
from .joint_planner import _cost
from .optional_prior_predictor import VERSION as PREDICTOR_VERSION, BASE_VERSION
from .state import parse_time
from .text_returns import _p

GAIN_VERSION = 'official-optional-context-gain-v1'
FORECAST_FIELDS = ('schema', 'model_version', 'match_id', 'physical_match_id',
    'probabilities', 'base_probabilities', 'prior_probabilities', 'prior_domain',
    'prior_domain_context_supported', 'base_source', 'forecast_basis',
    'context_effect_status', 'support_status', 'fallback_reason', 'evidence_origin')


def validate_forecast(forecast, *, state=None, predictor=None):
    """Validate explicit provenance; an unsupported domain is valid UNKNOWN.

    With a predictor, independently recompute the supplied query-time forecast
    from the same public features, and bind its original A base/support.
    """
    if not isinstance(forecast, dict) or any(k not in forecast for k in FORECAST_FIELDS):
        raise ValueError('The complete optional-prior forecast is required')
    value = {k: deepcopy(forecast[k]) for k in FORECAST_FIELDS}
    if (value['schema'] != PREDICTOR_VERSION
            or not isinstance(value['model_version'], str) or not value['model_version'].startswith(PREDICTOR_VERSION+':')
            or any(not isinstance(value[k], str) or not value[k] for k in ('match_id','physical_match_id'))
            or value['evidence_origin'] not in {'real_observational','constructed_mechanism'}
            or type(value['prior_domain_context_supported']) is not bool):
        raise ValueError('Explicit optional-prior model, fixture, origin and support required')
    p, base = _p(value['probabilities']), _p(value['base_probabilities'])
    missing = value['prior_probabilities'] is None
    domain = 'missing_prior' if missing else 'provided_prior'
    if value['prior_domain'] != domain:
        raise ValueError('Forecast prior domain differs from its actual original prior')
    if missing:
        source = value['base_source']
        if (value['forecast_basis'] != BASE_VERSION or not isinstance(source, dict)
                or set(source) != {'training_as_of','physical_matches'}
                or type(source['physical_matches']) is not int or source['physical_matches'] < 12):
            raise ValueError('Missing prior requires an independent training-label base')
        parse_time(source['training_as_of'])
    elif (_p(value['prior_probabilities']) != base
          or value['forecast_basis'] not in {'observed_market_snapshot','explicit_open_text_initial_prior'}
          or value['base_source'] != {'visible_prior_basis':value['forecast_basis']}):
        raise ValueError('Supplied original prior and forecast base differ')
    supported = value['prior_domain_context_supported']
    if ((not supported and (value['context_effect_status'] != 'UNKNOWN'
            or value['support_status'] != domain+'_context_unsupported' or p != base))
            or supported and value['context_effect_status'] != 'not_calibrated'):
        raise ValueError('Unsupported context effects must remain UNKNOWN and base-only')
    if state is not None:
        if state['provider_match_id'] != value['match_id'] or state['prior'] != value['prior_probabilities']:
            raise ValueError('Forecast identity or original prior differs from the visible state')
        if missing and parse_time(value['base_source']['training_as_of']) >= parse_time(state['decision_at']):
            raise ValueError('The learned base must precede the actual prediction')
    if predictor is not None:
        predictor._check()
        meta = predictor.metadata
        if (predictor.model_version != value['model_version'] or meta['evidence_origin'] != value['evidence_origin']
                or supported != meta['prior_domains'][domain]['context_supported']
                or value['physical_match_id'] in meta['training_physical_match_ids']
                or value['match_id'] in meta['training_match_ids']):
            raise ValueError('Forecast must bind the same held-out optional predictor and actual domain support')
        if missing and (base != meta['empirical_base']['probabilities'] or value['base_source'] != {
                'training_as_of':meta['training_as_of'], 'physical_matches':meta['empirical_base']['physical_matches']}):
            raise ValueError('Forecast base differs from the frozen independent A labels')
        if state is not None:
            if not parse_time(meta['training_as_of']) <= parse_time(meta['fit_execution_at']) <= parse_time(state['decision_at']):
                raise ValueError('Actual predictor fitting must precede the query')
            vector, known = predictor._vector(state)
            reason = ('no_visible_text' if not state['texts'] else 'all_text_out_of_vocabulary' if not known
                      else domain+'_context_unsupported' if not supported else None)
            result = base if reason else softmax(np.log(base)+vector@np.asarray(predictor.coefficients)).tolist()
            if value['fallback_reason'] != reason or any(abs(a-b)>1e-12 for a,b in zip(p,result)):
                raise ValueError('Query forecast differs from the frozen predictor on the actual visible state')
    return value


def query_input(visible, action, prediction):
    """Whitelist only the pre-query public state and observed candidate labels."""
    state = _state(visible, allow_missing_prior=True)
    forecast = validate_forecast(prediction, state=state)
    candidate = visible['candidates'].get(action['action_id'])
    tool = action.get('tool')
    request = ({'tool':tool,'url':action['url']} if tool == 'read_url' else
               {'tool':tool,'query':action['query'],'source':action.get('source','open_web')})
    if (tool not in {'read_url','search_web'} or candidate is None or candidate['request'] != request
            or candidate['information_key'] != action['information_key']
            or not candidate['available'] or candidate['already_attempted']):
        raise ValueError('The action must be an actual available pre-query candidate')
    if candidate['declared_cost'] != _cost(action) or candidate['cost_status'] != action.get('cost_status'):
        raise ValueError('Query cost differs from the actual visible menu')
    before = visible['before']
    links = {row['link_ref']:row for row in before['observed_links']}
    documents = {row['document_index']:row for row in before['documents']}
    parents = []
    for ref in candidate['observed_link_refs']:
        link = links.get(ref)
        if link is None or link['target_url'] != request.get('url'):
            raise ValueError('Candidate labels must come from an observed parent link')
        doc = documents[link['document_index']]
        parents.append({'parent':{k:doc[k] for k in ('document_id','raw_sha256','url','observed_at','available_at')},
            'labels':[{k:a[k] for k in ('anchor_index','text','text_truncated')} for a in link['observed_anchors']]})
    return {'schema':GAIN_VERSION,'state':{k:deepcopy(state[k]) for k in
                ('provider_match_id','decision_at','prediction_cutoff','kickoff_at','prior','texts','sources','numeric')},
        'numeric_names':list(NUMERIC),'prediction_before':list(forecast['probabilities']),
        'forecast':forecast,'prior_missing':state['prior'] is None,
        'ordered_history':[{k:deepcopy(row.get(k)) for k in ('action_id','status','information_key')}
                           for row in before['history']['attempts']],
        'candidate':{'request':request,'declared_cost':deepcopy(candidate['declared_cost']),
                     'cost_status':candidate['cost_status'],'observed_parents':parents}}


__all__ = ['GAIN_VERSION','query_input','validate_forecast']
