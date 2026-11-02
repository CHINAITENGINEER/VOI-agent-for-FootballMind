"""Pre-query role-conditioned gain inputs; original market absence stays absent."""
from copy import deepcopy

import numpy as np
from scipy.special import softmax

from .context_predictor import NUMERIC
from .fixture_roles import (role_state, candidate_role_state, validate_role_state,
                            VERSION as ROLE_VERSION)
from .joint_planner import _cost
from .optional_context_query import FORECAST_FIELDS as BASE_FORECAST_FIELDS
from .role_context_predictor import VERSION as PREDICTOR_VERSION, BASE_VERSION
from .state import parse_time
from .text_returns import _p

GAIN_VERSION = 'official-role-context-gain-v1'
ROLE_FORECAST_FIELDS = ('role_schema', 'role_identity_scope', 'role_counts',
                        'role_support', 'role_support_status')
FORECAST_FIELDS = BASE_FORECAST_FIELDS + ROLE_FORECAST_FIELDS


def validate_forecast(forecast, *, state=None, predictor=None):
    """Require the complete held-out role-f forecast on this original state."""
    if not isinstance(forecast, dict) or any(k not in forecast for k in FORECAST_FIELDS):
        raise ValueError('The complete role-conditioned forecast is required')
    value = {key: deepcopy(forecast[key]) for key in FORECAST_FIELDS}
    if (value['schema'] != PREDICTOR_VERSION or value['role_schema'] != ROLE_VERSION
            or not isinstance(value['model_version'], str)
            or not value['model_version'].startswith(PREDICTOR_VERSION + ':')
            or any(not isinstance(value[key], str) or not value[key] for key in ('match_id', 'physical_match_id'))
            or value['evidence_origin'] not in {'real_observational', 'constructed_mechanism'}
            or type(value['prior_domain_context_supported']) is not bool):
        raise ValueError('Explicit role predictor, fixture, origin and support required')
    p, base = _p(value['probabilities']), _p(value['base_probabilities'])
    missing = value['prior_probabilities'] is None
    domain = 'missing_prior' if missing else 'provided_prior'
    if value['prior_domain'] != domain:
        raise ValueError('Forecast prior domain differs from its original prior')
    if missing:
        source = value['base_source']
        if (value['forecast_basis'] != BASE_VERSION or not isinstance(source, dict)
                or set(source) != {'training_as_of', 'physical_matches'}
                or type(source['physical_matches']) is not int or source['physical_matches'] < 12):
            raise ValueError('Missing prior requires its independent A-label base')
        parse_time(source['training_as_of'])
    elif (_p(value['prior_probabilities']) != base
          or value['forecast_basis'] not in {'observed_market_snapshot', 'explicit_open_text_initial_prior'}
          or value['base_source'] != {'visible_prior_basis': value['forecast_basis']}):
        raise ValueError('Original prior and forecast base differ')
    supported = value['prior_domain_context_supported']
    if ((not supported and (value['context_effect_status'] != 'UNKNOWN'
            or value['support_status'] != domain + '_context_unsupported' or p != base))
            or supported and value['context_effect_status'] != 'not_calibrated'):
        raise ValueError('Unsupported context effects must remain UNKNOWN and base-only')
    if state is not None:
        validate_role_state(state)
        if (state['provider_match_id'] != value['match_id'] or state['prior'] != value['prior_probabilities']
                or state['physical_match_id'] != value['physical_match_id']
                or state['identity_scope'] != value['role_identity_scope']
                or state['role_counts'] != value['role_counts']):
            raise ValueError('Forecast identity, original prior or role binding differs from visible state')
        if missing and parse_time(value['base_source']['training_as_of']) >= parse_time(state['decision_at']):
            raise ValueError('Learned base must precede the actual prediction')
    if predictor is not None:
        predictor._check()
        meta = predictor.metadata
        if (predictor.model_version != value['model_version'] or meta['evidence_origin'] != value['evidence_origin']
                or supported != meta['prior_domains'][domain]['context_supported']
                or value['physical_match_id'] in meta['training_physical_match_ids']
                or value['match_id'] in meta['training_match_ids']):
            raise ValueError('Forecast requires the same held-out role predictor and original domain support')
        if missing and (base != meta['empirical_base']['probabilities'] or value['base_source'] != {
                'training_as_of': meta['training_as_of'], 'physical_matches': meta['empirical_base']['physical_matches']}):
            raise ValueError('Forecast base differs from frozen independent A labels')
        if state is not None:
            if not parse_time(meta['training_as_of']) <= parse_time(meta['fit_execution_at']) <= parse_time(state['decision_at']):
                raise ValueError('Actual predictor fitting must precede the query')
            expected_roles = predictor.role_support(state)
            if any(value[key] != expected_roles[key] for key in ROLE_FORECAST_FIELDS):
                raise ValueError('Role forecast support differs from frozen A coverage')
            vector, known = predictor._vector(state)
            reason = ('no_visible_text' if not state['texts'] else 'all_text_out_of_vocabulary' if not known
                      else domain + '_context_unsupported' if not supported else None)
            expected = base if reason else softmax(np.log(base) + vector @ np.asarray(predictor.coefficients)).tolist()
            if value['fallback_reason'] != reason or any(abs(a-b) > 1e-12 for a,b in zip(p, expected)):
                raise ValueError('Query forecast differs from frozen role predictor on the visible state')
    return value


def query_input(visible, action, prediction):
    """Bind role text exclusively to available state and observed parent labels."""
    state = role_state(visible)
    forecast = validate_forecast(prediction, state=state)
    candidate = visible['candidates'].get(action['action_id'])
    tool = action.get('tool')
    request = ({'tool': tool, 'url': action['url']} if tool == 'read_url' else
               {'tool': tool, 'query': action['query'], 'source': action.get('source', 'open_web')})
    if (tool not in {'read_url', 'search_web'} or candidate is None or candidate['request'] != request
            or candidate['information_key'] != action['information_key']
            or not candidate['available'] or candidate['already_attempted']):
        raise ValueError('Action must be an available actual pre-query candidate')
    if candidate['declared_cost'] != _cost(action) or candidate['cost_status'] != action.get('cost_status'):
        raise ValueError('Query cost differs from the actual visible menu')
    before = visible['before']
    links = {row['link_ref']: row for row in before['observed_links']}
    documents = {row['document_index']: row for row in before['documents']}
    parents = []
    for ref in candidate['observed_link_refs']:
        link = links.get(ref)
        if link is None or link['target_url'] != request.get('url'):
            raise ValueError('Candidate labels require an observed parent link')
        doc = documents[link['document_index']]
        parents.append({'parent': {key: doc[key] for key in ('document_id', 'raw_sha256', 'url', 'observed_at', 'available_at')},
            'labels': [{key: anchor[key] for key in ('anchor_index', 'text', 'text_truncated')}
                       for anchor in link['observed_anchors']]})
    return {'schema': GAIN_VERSION, 'state': deepcopy(state), 'numeric_names': list(NUMERIC),
        'prediction_before': list(forecast['probabilities']), 'forecast': forecast,
        'prior_missing': state['prior'] is None,
        'ordered_history': [{key: deepcopy(row.get(key)) for key in ('action_id', 'status', 'information_key')}
                            for row in before['history']['attempts']],
        'candidate': {'request': request, 'declared_cost': deepcopy(candidate['declared_cost']),
                      'cost_status': candidate['cost_status'], 'observed_parents': parents},
        'candidate_roles': candidate_role_state(visible, action['action_id'])}


__all__ = ['GAIN_VERSION', 'FORECAST_FIELDS', 'query_input', 'validate_forecast']
