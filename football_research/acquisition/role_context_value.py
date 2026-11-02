"""The existing signed ridge gain, conditioned on verified fixture text roles.

The shared B-only vocabulary is crossed with five roles on both already-read
state and already-observed candidate labels. The same fixed bilinear sketch,
ridge penalty, independent B support gates and before-cost target are retained.
"""
from collections import Counter
from datetime import datetime, timezone
import math

import numpy as np

from .context_value import PARAMETERS, UNITS, _interaction
from .fixture_roles import ROLES, validate_candidate_role_state
from .optional_context_value import (OptionalContextValueModel, _public as _optional_public,
                                     _bundles as _optional_bundles)
from .role_context_predictor import RoleContextPredictor
from .role_context_query import GAIN_VERSION, query_input, validate_forecast
from .context_predictor import _grams

VERSION = 'official-role-context-signed-ridge-value-v1'
SOURCE_VERSION = 'raw-official-role-context-gain-inputs-v1'


def _public(inputs, *, predictor=None):
    context = _optional_public(inputs, predictor=predictor, gain_version=GAIN_VERSION,
                               forecast_validator=validate_forecast)
    roles = inputs['candidate_roles']
    validate_candidate_role_state(roles)
    state = inputs['state']
    if (roles['physical_match_id'] != state['physical_match_id']
            or roles['identity_scope'] != state['identity_scope']
            or roles['role_proof']['visible_state'] != state['role_proof']['visible_state']):
        raise ValueError('Candidate roles must bind the same original pre-query fixture state')
    visible = state['role_proof']['visible_state']
    candidate = visible['candidates'].get(roles['action_id'])
    if candidate is None:
        raise ValueError('Candidate role proof lacks its actual menu entry')
    request = candidate['request']
    action = {**request, 'action_id': roles['action_id'], 'information_key': candidate['information_key'],
              'cost': candidate['declared_cost'], 'cost_status': candidate['cost_status']}
    rebuilt = query_input(visible, action, inputs['forecast'])
    if any(inputs[key] != rebuilt[key] for key in ('state', 'candidate', 'candidate_roles', 'ordered_history')):
        raise ValueError('Role query state, candidate or history differs from original visible proof')
    context['role_texts'] = state['role_texts']
    context['candidate_role_texts'] = roles['role_texts']
    # The parent's label list also includes a search request string. It remains
    # part of request validation but is not read as an observed article label.
    context['labels'] = roles['texts']
    context['mask'][1] = bool(roles['texts'])
    context['mask'] += [bool(state['role_texts'][role]) for role in ROLES]
    context['mask'] += [bool(roles['role_texts'][role]) for role in ROLES]
    return context


def _raw_builder(*args, **kwargs):
    from football_research.data_adapters.role_context_gain import build_role_context_gain_from_runs
    return build_role_context_gain_from_runs(*args, **kwargs)


def _bundles(paths, *, predictor_path, as_of):
    return _optional_bundles(paths, predictor_path=predictor_path, as_of=as_of,
        gain_version=GAIN_VERSION, predictor_type=RoleContextPredictor, public_validator=_public,
        source_version=SOURCE_VERSION, raw_builder=_raw_builder,
        after_support_statuses=('base_only','shared_role_feature_projection','source_extrapolation'))


def training_readiness(gain_dirs, *, predictor_path=None, as_of):
    return _bundles(gain_dirs, predictor_path=predictor_path, as_of=as_of)[4]


class RoleContextValueModel(OptionalContextValueModel):
    schema_version = VERSION
    predictor_type = RoleContextPredictor
    readiness = staticmethod(training_readiness)
    public_validator = staticmethod(_public)
    _training_rows = staticmethod(_bundles)
    lexical_blocks = 2 * len(ROLES)
    mask_size = 3 + 2 * len(ROLES)
    encoding_description = 'shared_B_char_3_5_TFIDF_cross_five_fixture_roles_for_state_and_observed_candidate_labels'
    support_description = 'same_actual_prior_domain_tool_source_and_role_presence_with_role_text_neighborhood_not_calibration'

    @staticmethod
    def _fit_clock():
        return datetime.now(timezone.utc)

    def _role_lexical(self, groups):
        blocks = []
        for role in ROLES:
            counts = Counter(_grams(groups[role]))
            block = np.asarray([(1. + math.log(counts[token])) if counts[token] else 0.
                                for token in self.encoder['vocabulary']])
            blocks.append(block * np.asarray(self.encoder['idf']))
        vector = np.concatenate(blocks)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def _vector(self, context):
        a = self._role_lexical(context['role_texts'])
        b = self._role_lexical(context['candidate_role_texts'])
        numeric = np.asarray(context['numeric']) / math.sqrt(len(context['numeric']))
        channel = np.asarray([float(context['channel'] == key) for key in self.encoder['channels']])
        sources = np.asarray([float(key in context['sources']) for key in self.encoder['sources']])
        if np.linalg.norm(sources):
            sources /= np.linalg.norm(sources)
        state = np.concatenate([a, numeric[:-len(UNITS)], sources])
        candidate = np.concatenate([b, channel, numeric[-len(UNITS):]])
        interactions = _interaction(state, candidate)
        return np.concatenate([[1.], a, b, numeric, channel, sources, interactions]), np.concatenate([a, b])


__all__ = ['RoleContextValueModel', 'training_readiness', 'VERSION', 'GAIN_VERSION', 'SOURCE_VERSION', 'PARAMETERS']
