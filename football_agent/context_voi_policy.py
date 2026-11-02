"""Visible-context prediction and empirical query gain in the existing runner.

The fixed outcome predictor and the query-gain regressor have different jobs.
Without a trained regressor, values remain unknown and only the separately
budgeted exploration policy can select a request. No return is enumerated.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from football_research.acquisition.context_predictor import ContextPredictor
from football_research.acquisition.context_gain import _query_input
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.joint_planner import UNITS, _budget, _cost
from football_research.acquisition.state import parse_time
from football_research.acquisition.value import _canonical
from .query_value_selection import predictor_value_blocker, select_query_candidates

VERSION = 'visible-context-empirical-query-policy-v1'


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def predictor_bytes(predictor):
    predictor._check()
    with TemporaryDirectory(prefix='fm-context-') as temporary:
        path = Path(temporary)/'f.json'
        predictor.save(path)
        return path.read_bytes()


class ContextVOIPolicy(LearnedJointVOIPolicy):
    """Public-state adapter; existing runtime still owns transport and budgets."""
    def __init__(self, base_policy, predictor, *, physical_match_id, value_model=None, predictor_artifact=None,
                 value_artifact=None, synthetic_prior=None):
        if (base_policy.manifest.get('schema')!='prospective-market-prior-policy-v1'
                or base_policy.manifest.get('loss') != 'log_loss'):
            raise ValueError('Context signed log-loss policy requires a frozen market initialization with log_loss')
        if not isinstance(predictor,ContextPredictor) or not physical_match_id:
            raise ValueError('A frozen context predictor and canonical fixture are required')
        self.base_policy,self.model,self.predictor=base_policy,base_policy.model,predictor
        self.physical_match_id,self.value_model=physical_match_id,value_model
        if synthetic_prior is not None:
            from .g_teacher_prior import SyntheticGTeacherPriorAdapter
            from .synthetic_g_teaching_prior import SyntheticGTeachingPriorAdapter
            if not isinstance(synthetic_prior, (SyntheticGTeacherPriorAdapter, SyntheticGTeachingPriorAdapter)):
                raise ValueError('synthetic_prior must be a validated synthetic prior adapter')
        self.synthetic_prior = synthetic_prior
        self.value_bytes = value_artifact
        if value_artifact is not None and (value_model is None or json.loads(value_artifact) !=
                {**value_model._payload(), 'model_version': value_model.model_version}):
            raise ValueError('Value artifact bytes do not represent the frozen model')
        self.predictor_bytes = predictor_bytes(predictor) if predictor_artifact is None else predictor_artifact
        if json.loads(self.predictor_bytes) != {**predictor._payload(), 'model_version':predictor.model_version}:
            raise ValueError('Predictor artifact bytes do not represent the frozen model')
        self.predictor_sha256=sha256(self.predictor_bytes).hexdigest()
        self.lambda_cost=deepcopy(base_policy.lambda_cost)
        if physical_match_id in predictor.metadata['training_physical_match_ids']:
            raise ValueError('The held-out fixture was used to train the outcome predictor')
        if value_model is not None:
            meta=value_model.metadata
            if (meta['predictor_sha256']!=self.predictor_sha256
                    or meta['predictor_model_version']!=predictor.model_version
                    or meta['evidence_origin']!=predictor.metadata['evidence_origin']
                    or physical_match_id in meta['training_physical_match_ids']):
                raise ValueError('The gain model must bind the same f and an independent decision fixture')
        self.manifest={'schema':VERSION,'base_policy':deepcopy(base_policy.manifest),
            'physical_match_id':physical_match_id,'predictor_version':predictor.model_version,
            'predictor_sha256':self.predictor_sha256,
            'value_model_version':value_model.model_version if value_model else None,
            'lambda_cost':self.lambda_cost,'evidence_origin':predictor.metadata['evidence_origin'],
            'value_scope':'expected_logged_fixed_predictor_signed_log_loss_reduction',
            'support_scope':'empirical_context_support_not_bayes_evsi_or_coverage_guarantee',
            'unknown_policy':'separately_bounded_exploration','production_ready':False}
        if getattr(self, 'synthetic_prior', None) is not None:
            # The prior is an explicitly separate diagnostic lane.  It is
            # included in the frozen manifest for auditability, but never
            # changes the empirical support gate below.
            self.manifest['synthetic_prior'] = deepcopy(self.synthetic_prior.manifest)
        self.version=VERSION+':'+_digest(self.manifest)[:24]

    def _state(self,state):
        return self.base_policy._state(state)

    def public_state(self,state):
        self.base_policy._state(state)
        value=deepcopy(state)
        prior=list(self.model.probabilities)
        if state.get('open_text',{}).get('prior_probabilities',prior)!=prior:
            raise ValueError('Visible original market prior changed')
        value['open_text']={'prior_probabilities':prior}
        return value

    def check_start(self,at):
        at=parse_time(at)
        if not parse_time(self.model.frozen_at)<=at:
            raise ValueError('The market snapshot was not frozen before execution')
        for model in (self.predictor,self.value_model):
            if model is not None and not (parse_time(model.metadata['training_as_of'])
                    <=parse_time(model.metadata['fit_execution_at'])<=at):
                raise ValueError('Actual model fitting must precede the first query')

    def predict_public(self,public):
        from football_research.acquisition.visible_state import encode_visible_state
        visible=encode_visible_state(public)
        result=self.predictor.predict(visible,physical_match_id=self.physical_match_id)
        return {**result,'initial_model_version':self.model.model_version,
                'initial_probabilities':list(self.model.probabilities),
                'probability_basis':'frozen_visible_context_with_observed_market_prior'}

    def expired_plan(self, public):
        result = {'schema': VERSION, 'policy_version': self.version,
            'model_version': self.predictor.model_version, 'match_id': public['state']['match_id'],
            'state_digest': _digest(public), 'prediction': deepcopy(public['prediction']),
            'candidates': [], 'selection': {'decision': 'halt', 'reason': 'forecast_cutoff_elapsed',
                'candidate_id': None, 'stop_certificate': None},
            'candidate_generation_complete': False, 'numeric_voi_training_eligible': False}
        result['decision_digest'] = _digest(result)
        return result

    def plan_public(self,public,actions,budget,*,candidate_generation_complete=False):
        from football_research.acquisition.visible_state import encode_visible_state
        limits,spent=_budget(budget)
        if any(a.get('group_ids') or a.get('tool') not in {'read_url', 'search_web'} for a in actions):
            raise ValueError('Context query values only support public web information actions')
        visible=encode_visible_state(public,actions)
        prediction=public['prediction']
        if prediction!=self.predict_public(public):
            raise ValueError('Query values require the actual current f prediction')
        candidates=[]
        for action in actions:
            cost=_cost(action)
            why=('action_unavailable' if not action.get('available',True) else
                 'already_attempted' if visible['candidates'][action['action_id']]['already_attempted'] else
                 'physical_budget_exhausted' if any(spent[u]+cost[u]>cap+1e-12 for u,cap in limits.items()) else None)
            value=None
            if not why:
                estimated={'support_status':'UNKNOWN','expected_signed_gain':None,
                           'reason':'context_gain_model_not_fitted'}
                blocker = predictor_value_blocker(prediction)
                if blocker:
                    estimated['reason'] = blocker
                elif self.value_model is not None:
                    query=_query_input(visible,action,prediction['probabilities'])
                    estimated=self.value_model.predict(query,physical_match_id=self.physical_match_id,
                        predictor_sha256=self.predictor_sha256)
                supported=estimated['support_status']=='supported'
                gross=estimated['expected_signed_gain'] if supported else None
                if supported and (type(gross) not in (int,float) or not math.isfinite(gross)):
                    raise ValueError('Supported empirical gain must be finite')
                loss_cost=math.fsum(self.lambda_cost[u]*cost[u] for u in UNITS)
                value={'schema_version':'context-empirical-query-value-v1','candidate_id':action['action_id'],
                    'status':'COMPUTED' if supported else 'INELIGIBLE','reason':estimated['reason'],
                    'expected_signed_gain':gross,'gross_evsi':None,'cost':loss_cost,
                    'net_voi':gross-loss_cost if supported else None,'predictor_version':self.predictor.model_version,
                    'value_model_version':self.value_model.model_version if self.value_model else None,
                    'estimate':estimated,'calibration_status':'not_established'}
                if getattr(self, 'synthetic_prior', None) is not None:
                    # Feature values must be supplied by the candidate
                    # generator.  The adapter intentionally does not infer
                    # injury/tactics/etc. from an action id or from the
                    # returned document.  Missing features become an
                    # explicit unavailable prior and leave the existing
                    # UNKNOWN empirical gate untouched.
                    prior_features = action.get('synthetic_prior_features')
                    if prior_features is None:
                        prior_features = action.get('g_prior_features')
                    prior_candidate = (dict(prior_features) if isinstance(prior_features, dict) else {})
                    prior_candidate['candidate_id'] = action['action_id']
                    value = self.synthetic_prior.attach(prior_candidate, value)
            candidates.append({**deepcopy(action),'feasible':why is None,'cost':cost,
                'reason':why or value['reason'],'value':value})
        selection = select_query_candidates(candidates,
            candidate_generation_complete=candidate_generation_complete)
        result={'schema':VERSION,'policy_version':self.version,'model_version':self.predictor.model_version,
            'match_id':public['state']['match_id'],'state_digest':_digest(public),'prediction':prediction,
            'candidates':candidates,'selection':selection,'physical_budget':{'limits':limits,'spent':spent},
            'candidate_generation_complete':candidate_generation_complete,
            'selection_scope':'current_feasible_supported_one_step_menu',
            'calibration_status':'not_established','numeric_voi_training_eligible':False}
        result['decision_digest']=_digest(result)
        return result
