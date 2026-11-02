"""Bind a real historical f/g pair to the existing live acquisition runtime.

Only admitted visible text and observed link labels reach the two models.
Historical training availability and actual operational query clocks remain
separate. Small-sample estimates are exposed as such, never as Bayes EVSI.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math

from football_research.acquisition.historical_context_training import HistoricalContextModel
from football_research.acquisition.joint_planner import UNITS, _budget, _cost
from football_research.acquisition.state import parse_time
from .context_voi_policy import ContextVOIPolicy, _digest
from .synthetic_g_teaching_prior import is_synthetic_prior_adapter
from .query_value_selection import predictor_value_blocker, select_query_candidates

VERSION = 'historical-context-research-policy-v1'


class HistoricalContextVOIPolicy(ContextVOIPolicy):
    # Historical/replay inputs require a source publication bound by default.
    # Prospective predictor-only adapters may explicitly opt into a stricter
    # operational rule: an admitted body is usable when its own available_at
    # clock is before the decision, even if the publisher did not expose a
    # publication timestamp.  The unknown timestamp remains unknown.
    allow_unknown_publication_clock = False

    def __init__(self, base_policy, model, *, fixture, artifact, state_generalization=False,
                 synthetic_prior=None):
        if not isinstance(model, HistoricalContextModel) or base_policy.manifest['loss'] != 'log_loss':
            raise ValueError('A frozen historical f/g pair and log-loss cost units are required')
        model._check()
        if json.loads(artifact) != {**model._payload(), 'model_version': model.model_version}:
            raise ValueError('The runtime must use the exact fitted f/g artifact')
        self.fixture = deepcopy(fixture)
        self.physical_match_id = fixture['physical_match_id']
        if synthetic_prior is not None and not is_synthetic_prior_adapter(synthetic_prior):
            raise ValueError('synthetic_prior must be a validated synthetic prior adapter')
        self.synthetic_prior = synthetic_prior
        if self.physical_match_id in model.metadata['A_match_ids'] + model.metadata['B_match_ids']:
            raise ValueError('The live decision fixture cannot train f or g')
        self.base_policy, self.model = base_policy, base_policy.model
        self.predictor = self.value_model = model
        self.predictor_bytes = self.value_bytes = artifact
        self.predictor_sha256 = sha256(artifact).hexdigest()
        self.lambda_cost = deepcopy(base_policy.lambda_cost)
        if type(state_generalization) is not bool:
            raise ValueError('The research state-generalization mode must be explicit')
        self.state_generalization = state_generalization
        self.manifest = {'schema': VERSION, 'base_policy': deepcopy(base_policy.manifest),
            'fixture': self.fixture, 'physical_match_id': self.physical_match_id,
            'predictor_version': model.model_version, 'predictor_sha256': self.predictor_sha256,
            'value_model_version': model.model_version, 'lambda_cost': self.lambda_cost,
            'evidence_origin': model.metadata['evidence_origin'],
            'value_scope': 'empirical_signed_fixed_predictor_gain_not_Bayes_EVSI',
            'prior_scope': 'A_learned_empirical_base_structured_baseline_separate',
            'unknown_policy': 'separately_bounded_exploration', 'production_ready': False}
        if state_generalization:
            self.manifest.update(state_generalization=True,
                decision_evidence_scope='uncalibrated_empirical_mean_with_declared_state_pattern_extrapolation')
        if 'f_selection' in model.metadata:
            self.manifest['predictor_value_gate'] = 'A_selected_base_only_does_not_establish_information_irrelevance_v1'
        if synthetic_prior is not None:
            self.manifest['synthetic_prior'] = deepcopy(synthetic_prior.manifest)
        self.version = VERSION + ':' + _digest(self.manifest)[:24]

    def public_state(self, state):
        self._state(state)
        if state['match_id'] != self.fixture['match_id']:
            raise ValueError('Runtime fixture differs from the declared research case')
        for key in ('home_team_id', 'away_team_id', 'competition_id', 'kickoff'):
            a, b = state['base'][key], self.fixture[key]
            differs = parse_time(a) != parse_time(b) if key == 'kickoff' else a != b
            if differs:
                raise ValueError('Ordered fixture identity changed')
        if state.get('open_text', {}).get('prior_probabilities') is not None:
            raise ValueError('This fitted text chain uses the same A-learned base, not an unseen supplied prior')
        result = deepcopy(state)
        result['open_text'] = {'prior_probabilities': None}
        return result

    def check_start(self, at):
        meta = self.predictor.metadata
        if not parse_time(meta['g_training_as_of']) <= parse_time(meta['fit_execution_at']) <= parse_time(at):
            raise ValueError('Actual f/g fitting must finish before live querying')

    def _visible(self, public, actions=()):
        from football_research.acquisition.visible_state import encode_visible_state
        visible = encode_visible_state(public, actions)
        before = visible['before']
        texts = {row['content_id']: row['text'] for row in before['contents']}
        records = {row['document_index']: row for row in before['documents']}
        audited_inputs = getattr(self, 'text_projection', self.predictor.metadata.get('text_projection')) == 'observed-news-titles-v1'
        documents, skipped = [], []
        for row in before['documents']:
            paragraphs = [texts[r['content_id']] for r in row['content_refs'] if r['role'] == 'body']
            if not paragraphs:
                if audited_inputs:
                    raw = public.get('documents', [])[row['document_index']]
                    skipped.append({'url': row['url'], 'reason': 'no_admitted_body_segments',
                        'reasons': [r['reason'] for r in raw.get('structured_news', {}).get('excluded', [])
                                    if isinstance(r, dict) and isinstance(r.get('reason'), str)]})
                continue
            publication_unknown = row.get('source_published_at') is None
            if publication_unknown and not self.allow_unknown_publication_clock:
                skipped.append({'url': row['url'], 'reason': 'unknown_publication_time'})
                continue
            if publication_unknown:
                try:
                    available_at = parse_time(row['available_at'])
                    decision_at = parse_time(before['decision_at'])
                except (KeyError, TypeError, ValueError):
                    skipped.append({'url': row['url'], 'reason': 'unknown_publication_and_availability_time'})
                    continue
                if available_at > decision_at:
                    skipped.append({'url': row['url'], 'reason': 'body_not_available_before_decision'})
                    continue
            titles = [texts[r['content_id']] for r in row['content_refs'] if r['role'] == 'headline']
            documents.append({'url': row['url'], 'title': '\n'.join(dict.fromkeys(titles)),
                'text': '\n'.join(paragraphs), 'paragraphs': paragraphs,
                'captured_at': row['observed_at'], 'available_at': row['available_at'],
                'source_published_at': row['source_published_at'],
                'source_modified_at': row.get('source_modified_at'), 'raw_sha256': row['raw_sha256']})
            if publication_unknown:
                documents[-1]['publication_time_status'] = 'unknown_operational_available_at'
        links = []
        for link in before['observed_links']:
            parent = records[link['document_index']]
            for anchor in link['observed_anchors']:
                title = anchor['text']
                if isinstance(title, str) and title.strip():
                    links.append({'url': link['target_url'], 'title': title,
                        'parent_url': parent['url'], 'available_at': parent['available_at'],
                        'title_basis': 'observed_parent_link'})
        if audited_inputs:
            from football_research.acquisition.news_link_projection import project_news_links
            links, projection = project_news_links(links)
            visible['news_link_projection'] = projection
        state = {'fixture': deepcopy(self.fixture), 'decision_at': before['decision_at'],
            'prediction_cutoff': before['prediction_cutoff'], 'information_mode': 'operational',
            'prior_probabilities': None, 'documents': documents, 'observed_links': links,
            'history': deepcopy(before['history']['attempts'])}
        return state, visible, skipped

    def predict_public(self, public):
        state, visible, skipped = self._visible(public)
        result = {**self.predictor.predict(state), 'probability_basis': 'historical_text_f_with_A_learned_base',
                'initial_probabilities': list(self.predictor.metadata['empirical_base']),
                'text_projection_audit': {**visible['audit'], 'unmodeled_documents': skipped},
                'modeled_read_documents': len(state['documents'])}
        if 'news_link_projection' in visible:
            selection = self.predictor.metadata.get('g_selection', {})
            result['gain_model_summary'] = {
                'fit_scheme': selection.get('schema', 'fixed_legacy'),
                'constant_baseline_selected': selection.get('constant_baseline_won'),
                'selected_regularization': selection.get('selected_regularization'),
                'selection_independent_matches': len(selection.get('selection_physical_match_ids', [])),
                'conditioning_claim': ('not_learned_constant_gain_baseline' if selection.get('constant_baseline_won')
                    else 'state_candidate_regression_unvalidated'),
                'empirical_decision_benefit_established': False}
            if 'f_selection' in self.predictor.metadata:
                result['gain_model_summary']['predictor_text_response_supported'] = (
                    self.predictor.metadata['f_selection']['base_only_won'] is False)
            result['text_model_input'] = {'schema': 'historical-text-input-v1',
                'modeled_body_documents': len(state['documents']),
                'modeled_body_document_ids': [_digest([d['url'], d['raw_sha256']]) for d in state['documents']],
                'modeled_body_paragraphs': len({p for d in state['documents'] for p in d['paragraphs']}),
                'modeled_link_labels': len({link['title'] for link in state['observed_links']}),
                'excluded_link_labels': visible['news_link_projection']['excluded_labels'],
                'link_projection': visible['news_link_projection'],
                'modeled_publication_times': [{key: d.get(key) for key in
                    ('url', 'source_published_at', 'source_modified_at')} for d in state['documents']],
                'unmodeled_documents': skipped,
                'meaning': 'actual_text_inputs_not_verified_match_facts_or_causal_effects'}
        return result

    def plan_public(self, public, actions, budget, *, candidate_generation_complete=False):
        limits, spent = _budget(budget)
        state, visible, _ = self._visible(public, actions)
        if public['prediction'] != self.predict_public(public):
            raise ValueError('Candidate values require the actual current frozen f prediction')
        candidates = []
        for action in actions:
            cost = _cost(action)
            why = ('action_unavailable' if not action.get('available', True) else
                   'already_attempted' if visible['candidates'][action['action_id']]['already_attempted'] else
                   'physical_budget_exhausted' if any(spent[u]+cost[u] > cap+1e-12 for u, cap in limits.items()) else None)
            value = None
            if not why:
                observed = any(link['url'] == action.get('url') for link in state['observed_links'])
                if action['tool'] == 'read_url' and not observed:
                    estimate = self.value_model._unknown('candidate_title_not_yet_observed')
                elif predictor_value_blocker(public['prediction']):
                    estimate = self.value_model._unknown(predictor_value_blocker(public['prediction']))
                elif self.predictor.metadata.get('f_selection', {}).get('base_only_won') is True:
                    # A predictor that ignores every return has mathematically
                    # zero fixed-f increments. That is an inability of this f,
                    # not evidence that the external information is worthless.
                    estimate = self.value_model._unknown('predictor_text_response_not_supported')
                elif self.state_generalization:
                    estimate = self.value_model.estimate(state, action, state_generalization=True)
                else:
                    estimate = self.value_model.estimate(state, action)
                supported = (estimate['support_status'] in {'supported', 'prototype_supported'}
                             or self.state_generalization and estimate['support_status'] == 'prototype_state_generalization')
                gain = estimate['expected_signed_gain'] if supported else None
                if supported and (type(gain) not in (int, float) or not math.isfinite(gain)):
                    raise ValueError('A supported empirical gain must be finite')
                charge = math.fsum(self.lambda_cost[u]*cost[u] for u in UNITS)
                value = {'schema_version': 'context-empirical-query-value-v1',
                    'candidate_id': action['action_id'], 'status': 'COMPUTED' if supported else 'INELIGIBLE',
                    'reason': estimate['reason'], 'expected_signed_gain': gain, 'gross_evsi': None,
                    'cost': charge, 'net_voi': gain-charge if supported else None,
                    'predictor_version': self.predictor.model_version,
                    'value_model_version': self.value_model.model_version, 'estimate': estimate,
                    'calibration_status': 'not_established'}
                if getattr(self, 'synthetic_prior', None) is not None:
                    prior_features = action.get('synthetic_prior_features')
                    if prior_features is None:
                        prior_features = action.get('g_prior_features')
                    prior_candidate = (dict(prior_features)
                                       if isinstance(prior_features, dict) else {})
                    prior_candidate['candidate_id'] = action['action_id']
                    value = self.synthetic_prior.attach(prior_candidate, value)
            candidates.append({**deepcopy(action), 'cost': cost, 'feasible': why is None,
                               'reason': why or value['reason'], 'value': value})
        selection = select_query_candidates(candidates,
            candidate_generation_complete=candidate_generation_complete)
        result = {'schema': VERSION, 'policy_version': self.version, 'model_version': self.predictor.model_version,
            'match_id': public['state']['match_id'], 'state_digest': _digest(public),
            'prediction': deepcopy(public['prediction']), 'candidates': candidates, 'selection': selection,
            'physical_budget': {'limits': limits, 'spent': spent},
            'candidate_generation_complete': candidate_generation_complete,
            'selection_scope': 'current_feasible_supported_one_step_menu',
            'calibration_status': 'not_established', 'numeric_voi_training_eligible': False}
        result['decision_digest'] = _digest(result)
        return result
