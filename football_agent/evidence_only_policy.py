"""Market-free evidence acquisition without an invented predictive prior.

The existing runner owns finite exploration, source I/O and cost accounting.
This adapter provides no outcome predictor or numerical information value.
Its identity must already have passed the independent raw identity gateway.
"""
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256

from football_research.acquisition.joint_planner import _budget, _cost
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.state import GROUP_FIELDS, parse_time, sanitize_state
from football_research.acquisition.value import _canonical
from football_spider.storage.raw_archive import digest as identity_digest


VERSION = 'market-free-evidence-only-policy-v1'
IDENTITY_VERSION = 'official-fixture-evidence-identity-v1'


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


class EvidenceOnlyPolicy(LearnedJointVOIPolicy):
    """A validated official identity and explicit unavailable forecast protocol."""

    def __init__(self, identity):
        if (not isinstance(identity, Mapping) or identity.get('schema') != IDENTITY_VERSION
                or identity.get('status') != 'CONFIRMED'
                or identity.get('binding_sha256') != identity_digest(
                    {key: value for key, value in identity.items() if key != 'binding_sha256'})):
            raise ValueError('Evidence-only acquisition requires an unchanged confirmed official identity')
        native = identity['canonical']
        required = ('source_event_id', 'home_id', 'away_id', 'home_name', 'away_name', 'competition', 'kickoff')
        if (any(not isinstance(native.get(key), str) or not native[key].strip() for key in required)
                or identity.get('match_id') != native['source_event_id']
                or native['home_id'] == native['away_id']):
            raise ValueError('Evidence-only identity requires distinct canonical teams and one physical fixture')
        verified = parse_time(identity['verified_at'])
        cutoff = parse_time(identity['prediction_cutoff'])
        kickoff = parse_time(native['kickoff'])
        if not verified < cutoff < kickoff:
            raise ValueError('Evidence-only identity must be verified before its prematch cutoff')
        aliases = identity['aliases']
        for side in ('home', 'away'):
            if (not isinstance(aliases.get(side), list) or not aliases[side]
                    or any(not isinstance(value, str) or not value.strip() for value in aliases[side])
                    or native[side+'_name'] not in aliases[side]):
                raise ValueError('Evidence-only aliases must preserve verified canonical team names')
        self.identity = deepcopy(dict(identity))
        self.physical_match_id = native['source_event_id']
        self.cutoff = cutoff.isoformat()
        self.initial_state = sanitize_state({'match_id': self.physical_match_id,
            'prediction_cutoff': self.cutoff, 'information_mode': 'operational',
            'base': {'competition_id': native['competition'], 'home_team_id': native['home_id'],
                'away_team_id': native['away_id'], 'kickoff': kickoff.isoformat()},
            'groups': {group: {'status': 'not_acquired'} for group in GROUP_FIELDS}})
        self.initial_state.update(action_history=[], prior_status='prior_unavailable', analysis_mode='evidence_only')
        self.initial = deepcopy(self.initial_state)
        self.evidence_match = {'match_id': self.physical_match_id, 'home_team': native['home_name'],
            'away_team': native['away_name'], 'home_aliases': deepcopy(aliases['home']),
            'away_aliases': deepcopy(aliases['away']), 'kickoff_at': kickoff.isoformat(),
            'prediction_cutoff': self.cutoff}
        self.manifest = {'schema': VERSION, 'identity': self.identity,
            'identity_sha256': identity['binding_sha256'], 'physical_match_id': self.physical_match_id,
            'analysis_mode': 'evidence_only', 'prior_status': 'prior_unavailable',
            'probabilities': None, 'prediction_model': None, 'value_model': None,
            'value_scope': 'unknown_without_a_predictive_prior',
            'unknown_policy': 'separately_bounded_exploration', 'numeric_voi_training_eligible': False}
        self.version = VERSION+':'+_digest(self.manifest)[:24]
        # Compatibility protocol for the existing runner, not a fitted model.
        self.model = self
        self.model_version = None
        self.probabilities = None

    @classmethod
    def from_manifest(cls, manifest):
        if not isinstance(manifest, Mapping) or manifest.get('schema') != VERSION:
            raise ValueError('Unsupported evidence-only policy manifest')
        policy = cls(manifest['identity'])
        if policy.manifest != manifest:
            raise ValueError('Evidence-only policy manifest differs from its official identity')
        return policy

    def _state(self, state):
        clean = sanitize_state(state)
        if clean != sanitize_state(self.initial_state):
            raise ValueError('Evidence-only acquisition cannot alter identity or introduce numeric feature observations')
        if (state.get('prior_status', 'prior_unavailable') != 'prior_unavailable'
                or state.get('analysis_mode', 'evidence_only') != 'evidence_only'
                or state.get('open_text') or state.get('probabilities') is not None):
            raise ValueError('Evidence-only acquisition cannot introduce a probability prior')
        history = state.get('action_history', [])
        if (not isinstance(history, list) or any(not isinstance(row, dict) or set(row) != {'action_id'}
                or not isinstance(row['action_id'], str) or not row['action_id'] for row in history)):
            raise ValueError('Evidence-only action history must contain actual request identifiers only')
        return clean

    def public_state(self, state):
        clean = self._state(state)
        return {**clean, 'action_history': deepcopy(state.get('action_history', [])),
            'prior_status': 'prior_unavailable', 'analysis_mode': 'evidence_only'}

    def check_start(self, at):
        if parse_time(at) < parse_time(self.identity['verified_at']):
            raise ValueError('Official identity must be verified before evidence acquisition begins')

    def predict_partial(self, state):
        self._state(state)
        return {'probabilities': None, 'model_version': None, 'prior_status': 'prior_unavailable',
            'probability_basis': 'unavailable_no_market_prior', 'analysis_mode': 'evidence_only',
            'calibration_status': 'not_applicable', 'numeric_confidence': None}

    def predict_public(self, public):
        return self.predict_partial(public['state'])

    def plan(self, state, actions, budget, *, candidate_generation_complete=False):
        prediction = self.predict_partial(state)
        limits, spent = _budget(budget)
        attempted = {row['action_id'] for row in state.get('action_history', [])}
        candidates = []
        for action in actions:
            if action.get('group_ids') or action.get('tool') not in {'read_url', 'search_web'}:
                raise ValueError('Evidence-only acquisition supports public web actions only')
            cost = _cost(action)
            reason = ('action_unavailable' if not action.get('available', True) else
                'already_attempted' if action['action_id'] in attempted else
                'physical_budget_exhausted' if any(spent[u]+cost[u] > cap+1e-12 for u, cap in limits.items()) else None)
            value = {'schema_version': 'prior-unavailable-query-value-v1', 'status': 'INELIGIBLE',
                'reason': 'prior_unavailable', 'candidate_id': action['action_id'],
                'expected_gain': None, 'expected_signed_gain': None, 'gross_evsi': None,
                'net_voi': None, 'cost': None, 'loss': None, 'numeric_confidence': None}
            candidates.append({**deepcopy(action), 'cost': cost, 'feasible': reason is None,
                'reason': reason or 'prior_unavailable', 'value': value})
        exhausted = (not any(c['feasible'] for c in candidates)
            and any(c['reason'] == 'physical_budget_exhausted' for c in candidates))
        selection = {'decision': 'halt' if exhausted else 'defer',
            'reason': 'physical_budget_exhausted' if exhausted else 'defer_prior_unavailable',
            'candidate_id': None, 'stop_certificate': None}
        result = {'schema': VERSION, 'policy_version': self.version, 'model_version': None,
            'match_id': state['match_id'], 'state_digest': _digest(state), 'prediction': prediction,
            'candidates': candidates, 'selection': selection, 'physical_budget': {'limits': limits, 'spent': spent},
            'candidate_generation_complete': candidate_generation_complete,
            'prior_status': 'prior_unavailable', 'numeric_voi_training_eligible': False}
        result['decision_digest'] = _digest(result)
        return result

    def plan_public(self, public, actions, budget, *, candidate_generation_complete=False):
        if public['prediction'] != self.predict_public(public):
            raise ValueError('Evidence-only public forecast must remain explicitly unavailable')
        result = self.plan(public['state'], actions, budget,
            candidate_generation_complete=candidate_generation_complete)
        result['state_digest'] = _digest(public)
        result['decision_digest'] = _digest({k: v for k, v in result.items() if k != 'decision_digest'})
        return result

    def expired_plan(self, public):
        result = self.plan_public(public, [], public['budget'])
        result['selection'].update(decision='halt', reason='forecast_cutoff_elapsed')
        result['decision_digest'] = _digest({k: v for k, v in result.items() if k != 'decision_digest'})
        return result


__all__ = ['VERSION', 'IDENTITY_VERSION', 'EvidenceOnlyPolicy']
