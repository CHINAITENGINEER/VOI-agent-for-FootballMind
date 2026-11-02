"""Pre-query independent C registration for a frozen context predictor and g.

This supervisor-only contract binds existing model bytes and acquisition
settings. It does not train, acquire evidence, certify a causal comparison or
promote synthetic models to real observations. Callers verify fixture raw
archives first and enforce the same policy specification at actual execution.
"""
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path

from .context_gain import _digest, _inventory, _write
from .context_predictor import ContextPredictor
from .context_value import ContextValueModel
from .joint_planner import UNITS
from .state import parse_time

REGISTRATION = 'context-confirmation-registration-v1'
POLICY_SPEC = 'context-confirmation-policy-v1'
_ORIGINS = {'real_observational', 'constructed_mechanism'}
_POLICY_FIELDS = {'schema', 'seed', 'requests_per_match', 'source_profile_version',
    'news_projection_version', 'evidence_version', 'candidate_discovery_version',
    'news_provider', 'route', 'news_route_overrides', 'allowed_hosts', 'seed_urls',
    'loss', 'lambda_cost', 'web_cost', 'unknown_action_limit'}


def _policy(spec):
    """Validate explicit present-run settings without guessing defaults."""
    if not isinstance(spec, dict) or set(spec) != _POLICY_FIELDS or spec['schema'] != POLICY_SPEC:
        raise ValueError('An explicit complete confirmation policy specification is required')
    if (type(spec['seed']) is not int or type(spec['requests_per_match']) is not int
            or not 1 <= spec['requests_per_match'] <= 4
            or type(spec['unknown_action_limit']) is not int
            or spec['unknown_action_limit'] != spec['requests_per_match']):
        raise ValueError('Confirmation seed and physical/exploration request limits must be fixed')
    for key, versions in (('source_profile_version', range(1, 5)),
                          ('news_projection_version', range(1, 6)),
                          ('candidate_discovery_version', range(1, 11))):
        if type(spec[key]) is not int or spec[key] not in versions:
            raise ValueError('Confirmation source and candidate versions must be explicit')
    if (spec['loss'] != 'log_loss' or spec['route'] not in {'direct', 'environment'}
            or spec['news_provider'] not in {None, 'bing_rss', 'bing_news_rss'}
            or spec['evidence_version'] is not None and not isinstance(spec['evidence_version'], str)):
        raise ValueError('Unsupported confirmation loss or transport settings')
    for key in ('allowed_hosts', 'seed_urls'):
        values = spec[key]
        if (not isinstance(values, list) or any(not isinstance(v, str) or not v or v != v.strip() for v in values)
                or values != sorted(set(values))):
            raise ValueError('Confirmation source capabilities must be explicit sorted unique strings')
    overrides = spec['news_route_overrides']
    if overrides is not None and (not isinstance(overrides, dict)
            or any(not isinstance(k, str) or not k or k != k.strip()
                   or v not in {'environment', 'direct', 'system_tls'}
                   for k, v in overrides.items())):
        raise ValueError('Confirmation news route overrides must be explicit registered transport names')
    weights = spec['lambda_cost']
    if (not isinstance(weights, dict) or set(weights) != set(UNITS)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights.values())):
        raise ValueError('Confirmation must declare all four finite nonnegative loss-cost weights')
    if (not isinstance(spec['web_cost'], dict) or set(spec['web_cost']) != {'money', 'requests'}
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in spec['web_cost'].values())
            or spec['web_cost'] != {'money': .01, 'requests': 1.}):
        raise ValueError('Confirmation web costs must match the fixed acquisition contract')
    _digest(spec)  # Reject non-JSON/nonfinite values before writing a registration.
    return spec


def _registration(plan):
    """Content check reusable by the existing cross-phase inventory dispatcher."""
    from football_research.data_adapters.prospective_learning import SCOPE
    if (not isinstance(plan, Mapping) or plan.get('schema') != REGISTRATION
            or plan.get('purpose') != 'confirmation' or plan.get('inventory_scope') != SCOPE
            or not isinstance(plan.get('matches'), list) or not plan['matches']
            or not isinstance(plan.get('prior_inventory'), list)
            or not isinstance(plan.get('artifacts'), dict)
            or set(plan['artifacts']) != {'text_model', 'text_risk_model'}
            or plan.get('evidence_origin') not in _ORIGINS
            or plan.get('registration_id') != REGISTRATION+':'+_digest(
                {k:v for k,v in plan.items() if k != 'registration_id'})[:24]):
        raise ValueError('Context confirmation registration content differs')
    if plan.get('policy_spec_sha256') != _digest(_policy(plan['policy_spec'])):
        raise ValueError('Confirmation policy specification changed after registration')
    parse_time(plan['created_at'])
    return plan


def _artifact(path, model, at):
    path = Path(path).resolve()
    meta = model.metadata
    if (meta.get('evidence_origin') not in _ORIGINS or not parse_time(meta['training_as_of'])
            <= parse_time(meta['fit_execution_at']) <= at):
        raise ValueError('Actual model fitting must precede confirmation registration')
    return {'path':str(path), 'sha256':sha256(path.read_bytes()).hexdigest(),
        'model_version':model.model_version, 'training_as_of':meta['training_as_of'],
        'fit_execution_at':meta['fit_execution_at'], 'frozen_at':at.isoformat()}


def _admit(initialization_plan, identity_bundle, previous_plans, predictor_path,
           value_model_path, policy_spec, created):
    from football_research.data_adapters.prospective_learning import (
        _bound_matches, _training_events, _merge_inventory, _own_inventory)
    spec = _policy(policy_spec)
    f = ContextPredictor.load(predictor_path)
    g = ContextValueModel.load(value_model_path, predictor_path=predictor_path)
    artifacts = {'text_model':_artifact(predictor_path, f, created),
                 'text_risk_model':_artifact(value_model_path, g, created)}
    if f.metadata['evidence_origin'] != g.metadata['evidence_origin']:
        raise ValueError('Confirmation predictor and value model must share one explicit evidence origin')
    if any(manifest.get('loss') != spec['loss'] or manifest.get('lambda_cost') != spec['lambda_cost']
           for manifest in initialization_plan.get('selected', [])):
        raise ValueError('Confirmation policy cost/loss differs from frozen market initializations')
    matches = _bound_matches(initialization_plan, identity_bundle, created)
    prior = _inventory(previous_plans, created)
    current = {row['source_event_id'] for row in matches}
    if current & {row['source_event_id'] for row in prior}:
        raise ValueError('A confirmation physical fixture already appears in a prior A/B/C cohort')
    f_events = _training_events(f, prior, 'f_train', current)
    if f_events != set(f.metadata['training_physical_match_ids']):
        raise ValueError('Frozen predictor aliases differ from registered physical training fixtures')
    mapping = {mid:row['source_event_id'] for row in prior for mid in row['match_ids']}
    if any(mapping.get(mid) != event for mid,event in f.metadata['training_aliases'].items()):
        raise ValueError('Frozen predictor provider aliases disagree with the physical inventory')
    g_ids = g.metadata['training_physical_match_ids']
    if (not isinstance(g_ids, list) or len(set(g_ids)) != len(g_ids)
            or set(g.metadata['predictor_training_physical_match_ids']) != f_events
            or {row['physical_match_id'] for row in g.support_records} != set(g_ids)):
        raise ValueError('Frozen value model training physical identity metadata differs')
    g_events = set(g_ids)
    if f_events & g_events or current & g_events:
        raise ValueError('Predictor A, value model B and confirmation C must be physically disjoint')
    inventory = {row['source_event_id']:row for row in prior}
    f_binding = {key:artifacts['text_model'][key] for key in ('sha256', 'model_version')}
    for event in g_events:
        row = inventory.get(event)
        if row is None or row['purpose'] != 'calibration':
            raise ValueError('Every value-model training fixture must belong to registered B calibration')
        if row['model_bindings'].get('text_model') != f_binding:
            raise ValueError('Calibration data were collected with a different frozen predictor')
        if not (parse_time(f.metadata['fit_execution_at']) <= parse_time(row['first_registered_at'])
                < parse_time(g.metadata['training_as_of'])):
            raise ValueError('Calibration registration must follow fitted f and precede the g training cutoff')
    own = {'purpose':'confirmation', 'created_at':created.isoformat(), 'matches':matches, 'artifacts':artifacts}
    _merge_inventory(prior+_own_inventory(own))
    return f, g, artifacts, matches, prior


def prepare_context_confirmation(initialization_plan, *, identity_bundle, previous_plans,
                                 predictor_path, value_model_path, policy_spec, output, created_at=None):
    """Register frozen f/g and fixed C acquisition settings before any query.

    ``created_at`` is the caller's actual registration clock, not an inferred
    historical fit date; explicit clocks also enable constructed offline tests.
    The evidence origin stays attached to this registration and both artifacts.
    """
    from football_research.data_adapters.prospective_learning import SCOPE
    at = parse_time(created_at or datetime.now(timezone.utc))
    f, g, artifacts, matches, prior = _admit(initialization_plan, identity_bundle, previous_plans,
        predictor_path, value_model_path, policy_spec, at)
    plan = {'schema':REGISTRATION, 'purpose':'confirmation', 'created_at':at.isoformat(),
        'matches':matches, 'prior_inventory':prior, 'artifacts':artifacts, 'inventory_scope':SCOPE,
        'initialization_plan_sha256':_digest(initialization_plan), 'identity_bundle_sha256':_digest(identity_bundle),
        'policy_spec':deepcopy(policy_spec), 'policy_spec_sha256':_digest(policy_spec),
        'evidence_origin':f.metadata['evidence_origin'], 'supervisor_only':True,
        'independent_effect_confirmed':False, 'production_ready':False,
        'scope':'prequery_fixed_f_g_independent_C_acquisition_not_preregistered_comparative_effect_trial'}
    plan['registration_id'] = REGISTRATION+':'+_digest(plan)[:24]
    _registration(plan)
    _write(output, plan)
    return plan


def validate_context_confirmation(context, *, predictor_path, value_model_path, as_of, policy_spec=None):
    """Load exact local f/g bytes and check the saved pre-query C contract.

    Paths may point to archived run-local copies. Collector/replay callers pass
    their actual policy specification to require exact equality to registration.
    """
    plan = _registration(context['registration'])
    created, now = parse_time(plan['created_at']), parse_time(as_of)
    if not created < now:
        raise ValueError('Confirmation registration must strictly precede actual acquisition')
    if (plan['initialization_plan_sha256'] != _digest(context['initialization_plan'])
            or plan['identity_bundle_sha256'] != _digest(context['identity_bundle'])):
        raise ValueError('Confirmation initialization or identity bundle changed')
    if policy_spec is not None and _policy(policy_spec) != plan['policy_spec']:
        raise ValueError('Actual confirmation policy differs from the registered settings')
    f, g, artifacts, matches, prior = _admit(context['initialization_plan'], context['identity_bundle'],
        context['previous_plans'], predictor_path, value_model_path, plan['policy_spec'], created)
    for key, spec in artifacts.items():
        registered = plan['artifacts'][key]
        if not Path(registered['path']).is_absolute():
            raise ValueError('Registered model artifact path must be absolute')
        spec['path'] = registered['path']
    if (artifacts != plan['artifacts'] or matches != plan['matches'] or prior != plan['prior_inventory']
            or f.metadata['evidence_origin'] != plan['evidence_origin']):
        raise ValueError('Frozen confirmation models, fixture identities or phase inventory differ')
    if any(now >= parse_time(row['prediction_cutoff']) for row in matches):
        raise ValueError('Confirmation prematch acquisition horizon has expired')
    return {'predictor':f, 'value_model':g, 'binding':{'registration_id':plan['registration_id'],
        'registration_sha256':_digest(plan), 'policy_spec_sha256':plan['policy_spec_sha256']},
        'matches':{row['match_id']:row for row in matches}}


__all__ = ['REGISTRATION', 'POLICY_SPEC', 'prepare_context_confirmation', 'validate_context_confirmation']
