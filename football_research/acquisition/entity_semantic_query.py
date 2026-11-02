"""Pre/post query records for the entity-aware semantic f.

This adapter is versioned separately from ``semantic_query``.  It keeps the
same pre-read allowlist and hindsight separation, while exposing the exact
entity-predicate representation to the future g consumer.
"""
from copy import deepcopy
import math

import numpy as np

from .context_predictor import _digest
from .historical_context_training import _project
from .joint_planner import _cost
from .match_conditioned_f import bind_prior
from .news_link_projection import VERSION as NEWS_PROJECTION
from .semantic_f import public_input
from .state import parse_time
from .entity_semantic_f import (FEATURES, CHANNEL_FEATURES, REPRESENTATION_SHA256,
                                VERSION as F_VERSION, vector)

VERSION = 'entity-semantic-fixed-f-query-input-v1'
SUPPORT_VERSION = 'entity-semantic-fixed-f-support-v1'


def _claims(bank, input_ids):
    return [{'input_id': key, 'channel': bank.records[key]['input']['channel'],
             'url': bank.records[key]['input']['url'], 'claim': deepcopy(claim)}
            for key in input_ids for claim in bank.records[key]['response']['claims']]


def _predicate(claim):
    return (claim['team'], ' '.join(claim['entity'].casefold().split()),
            claim['kind'], claim['value'], claim['scope'])


def _feature(claim):
    value = ':'.join((claim['team'], claim['kind'], claim['value']))
    return value if claim['scope'] == 'this_match' and value in FEATURES else None


def _labels(model):
    return CHANNEL_FEATURES


def predictor_support(model, prior):
    if model.model_version != F_VERSION + ':' + _digest(model.payload())[:24]:
        raise ValueError('Frozen entity semantic f was modified')
    labels = _labels(model)
    base_len = len(labels)
    w = np.asarray(model.coefficients)
    if w.shape != (4*base_len, 3):
        raise ValueError('Entity semantic coefficient shape differs from support protocol')
    delta = prior[0] - prior[2]
    title = w[:base_len] + delta*w[2*base_len:3*base_len]
    body = w[base_len:2*base_len] + delta*w[3*base_len:4*base_len]
    t = [labels[i] for i, row in enumerate(title) if float(np.max(row)-np.min(row)) > 1e-14]
    b = [labels[i] for i, row in enumerate(body) if float(np.max(row)-np.min(row)) > 1e-14]
    return {'schema': SUPPORT_VERSION, 'model_version': model.model_version,
        'body_response_status': 'responsive' if b else 'insensitive',
        'title_response_status': 'responsive' if t else 'insensitive',
        'text_response_status': 'responsive' if t or b else 'insensitive',
        'responsive_title_features': t, 'responsive_body_features': b,
        'representation': 'coarse_team_predicates_plus_distinct_source_entity_counts',
        'unseen_entity_policy': 'no_name_vocabulary_revealed_entities_counted',
        'unmodeled_information_value': 'UNKNOWN',
        'scope': 'fixed_entity_f_capability_at_this_prior_not_semantic_truth_or_g_support'}


def query_input(state, action, *, model, prior_snapshot, evidence_bank):
    prediction = model.predict(state, prior_snapshot=prior_snapshot, evidence_bank=evidence_bank)
    checked = _project(bind_prior(state, prior_snapshot), text_projection=NEWS_PROJECTION)
    if action.get('tool') not in {'read_url', 'search_web'}:
        raise ValueError('Semantic queries support public read/search actions only')
    if not isinstance(action.get('action_id'), str) or not action['action_id']:
        raise ValueError('A nonempty action identity is required')
    if type(action.get('available', True)) is not bool:
        raise ValueError('Action availability must be explicit Boolean')
    cost = _cost(action)
    candidate = {k: deepcopy(action[k]) for k in ('action_id', 'tool')}
    candidate.update(cost=cost, cost_status=action['cost_status'], available=action.get('available', True),
                    already_attempted=any(h.get('action_id') == action['action_id'] for h in state.get('history', [])))
    labels, candidate_inputs = [], []
    if action['tool'] == 'read_url':
        links = [link for link in checked['observed_links'] if link['url'] == action.get('url')]
        if not links:
            raise ValueError('Read action requires a previously observed title/URL')
        candidate['url'] = action['url']
        candidate['already_attempted'] |= any(d['url'] == action['url'] for d in checked['documents'])
        for link in links:
            key = public_input(checked['fixture'], link, 'title')['input_id']
            if key not in prediction['used_semantic_input_ids']:
                raise ValueError('Candidate title is absent from the actual visible f input')
            if key not in candidate_inputs:
                labels.append({k: deepcopy(link[k]) for k in ('url', 'title', 'available_at', 'title_basis')})
                candidate_inputs.append(key)
    else:
        if not isinstance(action.get('query'), str) or not action['query'].strip():
            raise ValueError('Search requires its actual public query text')
        candidate['query'] = action['query']
    candidate.update(observed_labels=labels, title_assertions=_claims(evidence_bank, candidate_inputs),
                     unread_body_support='UNKNOWN', unread_body_novelty='UNKNOWN')
    projection = prediction['projection']
    current = {'probabilities': prediction['probabilities'],
        'predictor_support': predictor_support(model, prediction['initial_probabilities']),
        'title_features': projection['title'], 'novel_body_features': projection['novel_body'],
        'title_entity_counts': projection['title_entity_counts'],
        'novel_body_entity_counts': projection['novel_body_entity_counts'],
        'known_input_ids': prediction['used_semantic_input_ids'],
        'known_assertions': _claims(evidence_bank, prediction['used_semantic_input_ids']),
        'history': [{k: deepcopy(h[k]) for k in ('action_id', 'status') if k in h}
                    for h in state.get('history', [])]}
    payload = {'schema': VERSION, 'physical_match_id': checked['physical_match_id'],
        'decision_at': checked['decision_at'], 'prediction_cutoff': checked['prediction_cutoff'],
        'fixture': {k: deepcopy(checked['fixture'][k]) for k in ('physical_match_id','home_name','away_name','kickoff')},
        'information_mode': checked['information_mode'], 'model_version': model.model_version,
        'predictor_payload_sha256': _digest(model.payload()), 'prior_snapshot_sha256': prior_snapshot['snapshot_sha256'],
        'prior_probabilities': prediction['initial_probabilities'], 'representation_sha256': prediction['entity_representation_sha256'],
        'current': current, 'candidate': candidate}
    return {**payload, 'query_sha256': _digest(payload)}


def validate_query(query):
    payload = {k:v for k,v in query.items() if k != 'query_sha256'}
    allowed = {'schema','physical_match_id','decision_at','prediction_cutoff','fixture','information_mode',
               'model_version','predictor_payload_sha256','prior_snapshot_sha256','prior_probabilities',
               'representation_sha256','current','candidate'}
    if set(payload) != allowed or query.get('schema') != VERSION or query.get('query_sha256') != _digest(payload):
        raise ValueError('Pre-query input is modified or contains undeclared fields')
    current_keys = {'probabilities','predictor_support','title_features','novel_body_features',
                    'title_entity_counts','novel_body_entity_counts','known_input_ids','known_assertions','history'}
    candidate_keys = {'action_id','tool','cost','cost_status','available','already_attempted','observed_labels',
                      'title_assertions','unread_body_support','unread_body_novelty'}
    candidate_keys.add('url' if query['candidate'].get('tool') == 'read_url' else 'query')
    support_keys = {'schema','model_version','body_response_status','title_response_status','text_response_status',
                    'responsive_title_features','responsive_body_features','representation',
                    'unseen_entity_policy','unmodeled_information_value','scope'}
    if (set(query['current']) != current_keys or set(query['candidate']) != candidate_keys
            or set(query['current']['predictor_support']) != support_keys
            or set(query['fixture']) != {'physical_match_id','home_name','away_name','kickoff'}):
        raise ValueError('Undeclared nested entity-query fields')
    for row in query['current']['known_assertions'] + query['candidate']['title_assertions']:
        if set(row) != {'input_id','channel','url','claim'} or set(row['claim']) != {'team','entity','kind','value','scope','statement','quotes'}:
            raise ValueError('Undeclared source assertion fields')
    if (any(set(h)-{'action_id','status'} for h in query['current']['history'])
            or any(set(r) != {'url','title','available_at','title_basis'} for r in query['candidate']['observed_labels'])):
        raise ValueError('Only pre-read history and labels are allowed')
    if query['representation_sha256'] != REPRESENTATION_SHA256:
        raise ValueError('Unknown entity representation')
    vector({'prior': query['prior_probabilities'], 'title': query['current']['title_features'],
            'novel_body': query['current']['novel_body_features'],
            'title_entity_counts': query['current']['title_entity_counts'],
            'novel_body_entity_counts': query['current']['novel_body_entity_counts'],
            'representation_sha256': query['representation_sha256']})
    _cost(query['candidate'])


def response_audit(query, after_state, *, model, prior_snapshot, evidence_bank):
    validate_query(query)
    after = model.predict(after_state, prior_snapshot=prior_snapshot, evidence_bank=evidence_bank)
    if (query['model_version'] != after['model_version'] or query['predictor_payload_sha256'] != _digest(model.payload())
            or query['representation_sha256'] != after['entity_representation_sha256']
            or query['prior_snapshot_sha256'] != after['prior_snapshot_sha256']
            or query['physical_match_id'] != after_state['fixture']['physical_match_id']
            or parse_time(query['prediction_cutoff']) != parse_time(after_state['prediction_cutoff'])
            or parse_time(after_state['decision_at']) < parse_time(query['decision_at'])):
        raise ValueError('Return must use the same frozen entity f, fixture, prior and forward clock')
    before_ids, after_ids = set(query['current']['known_input_ids']), set(after['used_semantic_input_ids'])
    if not before_ids <= after_ids:
        raise ValueError('An immediate reveal cannot remove previous visible evidence')
    if _claims(evidence_bank, query['current']['known_input_ids']) != query['current']['known_assertions']:
        raise ValueError('Previously visible semantic assertions changed')
    new_ids = sorted(after_ids-before_ids)
    new_bodies = [i for i in new_ids if evidence_bank.records[i]['input']['channel']=='body']
    if query['candidate']['tool']=='read_url' and any(evidence_bank.records[i]['input']['url'] != query['candidate']['url'] for i in new_bodies):
        raise ValueError('Returned body belongs to a different request')
    old_claims = query['current']['known_assertions']; new_claims = _claims(evidence_bank, new_ids)
    known = {_predicate(x['claim']) for x in old_claims}
    old_features = {_feature(x['claim']) for x in old_claims} - {None}
    novel, repeated, collisions, unmodeled = [], [], [], []
    for row in new_claims:
        claim, feature = row['claim'], _feature(row['claim'])
        if _predicate(claim) in known: repeated.append(row)
        else:
            novel.append(row)
            if feature is not None and feature in old_features: collisions.append(row)
        if feature is None: unmodeled.append(row)
    p, q = query['current']['probabilities'], after['probabilities']
    delta = max(abs(a-b) for a,b in zip(p,q))
    before_projection = {'prior': query['prior_probabilities'], 'title': query['current']['title_features'],
        'novel_body': query['current']['novel_body_features'],
        'title_entity_counts': query['current']['title_entity_counts'],
        'novel_body_entity_counts': query['current']['novel_body_entity_counts'],
        'representation_sha256': query['representation_sha256']}
    feature_changed = not np.array_equal(vector(before_projection), vector(after['projection']))
    if not new_bodies: interpretation='no_new_body_input'
    elif query['current']['predictor_support']['text_response_status']=='insensitive': interpretation='predictor_insensitive_not_information_irrelevant'
    elif collisions and not feature_changed: interpretation='new_predicates_unresolved_by_entity_count_representation'
    elif unmodeled and not feature_changed: interpretation='assertions_outside_frozen_feature_support'
    elif repeated and not novel and not feature_changed: interpretation='same_extracted_predicates_not_proof_of_duplicate_information'
    elif not new_claims: interpretation='no_assertions_extracted_not_proof_of_irrelevance'
    elif delta <= 1e-12: interpretation='zero_entity_f_response_not_proof_of_zero_information_value'
    else: interpretation='entity_f_forecast_changed_not_yet_a_gain'
    payload = {'schema':'entity-semantic-fixed-f-response-audit-v1','query_sha256':query['query_sha256'],
        'physical_match_id':query['physical_match_id'],'model_version':model.model_version,
        'prior_snapshot_sha256':prior_snapshot['snapshot_sha256'],'representation_sha256':after['entity_representation_sha256'],
        'prediction_before':p,'prediction_after':q,'max_probability_change':delta,
        'new_body_input_ids':new_bodies,'new_input_ids':new_ids,'new_predicates':novel,'repeated_predicates':repeated,
        'coarse_feature_overlap':collisions,'unmodeled_assertions':unmodeled,'feature_vector_changed':feature_changed,
        'entity_count_delta': [b-a for a,b in zip(query['current']['novel_body_entity_counts'], after['projection']['novel_body_entity_counts'])],
        'multiple_status_entities_after': after['projection']['multiple_status_entities'],
        'interpretation':interpretation,'information_value':None,'expected_gain_before_read':None,
        'risk_of_harm_before_read':None,'scope':'post_read_source_assertions_not_independent_truth_or_query_estimates'}
    return {**payload,'audit_sha256':_digest(payload)}


def realized_target(query, audit, *, outcome, result_available_at, label_provenance):
    validate_query(query)
    if (audit.get('query_sha256') != query['query_sha256'] or audit.get('audit_sha256') != _digest({k:v for k,v in audit.items() if k!='audit_sha256'})
            or audit.get('model_version') != query['model_version'] or audit.get('prior_snapshot_sha256') != query['prior_snapshot_sha256']
            or audit.get('representation_sha256') != query['representation_sha256']):
        raise ValueError('Post-read audit must bind this exact frozen entity f input')
    if (type(outcome) is not int or outcome not in (0,1,2) or parse_time(result_available_at) <= parse_time(query['fixture']['kickoff'])
            or not label_provenance):
        raise ValueError('A separately sourced after-match result is required')
    gain = math.log(audit['prediction_after'][outcome] / audit['prediction_before'][outcome])
    return {'schema':'entity-semantic-fixed-f-realized-target-v1','query_sha256':query['query_sha256'],
        'audit_sha256':audit['audit_sha256'],'physical_match_id':query['physical_match_id'],
        'model_version':query['model_version'],'representation_sha256':query['representation_sha256'],
        'prior_snapshot_sha256':query['prior_snapshot_sha256'],
        'outcome':outcome,'result_available_at':result_available_at,'label_provenance':deepcopy(label_provenance),
        'realized_gain_nats':gain,'realized_prediction_harm':gain < -1e-12,
        'target':'conditional_frozen_predictor_log_loss_change','representation_interpretation':audit['interpretation'],
        'zero_targets_retained':True,'empirical_expected_gain':None,
        'risk_target_warning':'one_outcome_indicator_not_probability_of_negative_expected_net',
        'research_use':'historical_version_assumption_and_model_conditioning_required','prospective_training_eligible':False}
