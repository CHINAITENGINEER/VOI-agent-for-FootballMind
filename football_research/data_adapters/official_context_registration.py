"""Pre-query B/C admission for official identities without a market prior."""
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

from football_research.acquisition.joint_planner import UNITS
from football_research.acquisition.official_context_models import load_official_predictor, load_official_value
from football_research.acquisition.state import parse_time
from football_spider.storage.raw_archive import digest
from .official_fixture import verify_official_fixture
from .prospective_learning import FACTS, SCOPE, _previous_inventory, _training_events, _own_inventory, _merge_inventory

CALIBRATION = 'official-context-calibration-registration-v1'
CONFIRMATION = 'official-context-confirmation-registration-v1'
SCHEMAS = {'calibration': CALIBRATION, 'confirmation': CONFIRMATION}
ORIGINS = {'real_observational', 'constructed_mechanism'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')


def _settings(value):
    fields = {'requests_per_match', 'seed', 'route', 'news_route_overrides',
              'news_projection_version', 'lambda_cost'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('Official context acquisition needs all explicit settings')
    if (type(value['requests_per_match']) is not int or not 1 <= value['requests_per_match'] <= 4
            or type(value['seed']) is not int or value['route'] not in {'direct', 'environment'}
            or type(value['news_projection_version']) is not int or value['news_projection_version'] not in {5, 6}):
        raise ValueError('Unsupported official context budget, seed, route or projection')
    weights, overrides = value['lambda_cost'], value['news_route_overrides']
    if (not isinstance(weights, dict) or set(weights) != set(UNITS)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights.values())):
        raise ValueError('All four finite nonnegative loss-cost weights must be fixed')
    if overrides is not None and (not isinstance(overrides, dict) or any(
            not isinstance(k, str) or not k or k != k.strip() or v not in {'direct', 'environment', 'system_tls'}
            for k, v in overrides.items())):
        raise ValueError('Official context routes must use registered explicit transports')
    digest(value)
    return deepcopy(value)


def _execution_spec(settings):
    requests = settings['requests_per_match']
    return {'source_profile_version': 5, 'candidate_discovery_version': 11,
        'evidence_version': 'match-evidence-extraction-v5', 'news_provider': 'bing_rss',
        'extraction_version': 3, 'max_steps': 16, 'max_exploration_actions': requests,
        'limits': {'money': .01*requests, 'requests': requests},
        'exploration_limits': {'money': .01*requests, 'requests': requests},
        'exploration': 'uniform_inquiry_unknown_only', 'loss': 'log_loss',
        'web_cost': {'money': .01, 'requests': 1.}}


def _registration(plan):
    """Pure content validation for the shared physical-exposure inventory."""
    if (not isinstance(plan, Mapping) or plan.get('purpose') not in SCHEMAS
            or plan.get('schema') != SCHEMAS[plan['purpose']] or plan.get('inventory_scope') != SCOPE
            or not isinstance(plan.get('matches'), list) or len(plan['matches']) != 1
            or not isinstance(plan.get('prior_inventory'), list) or plan.get('evidence_origin') not in ORIGINS
            or plan.get('prior_status') != 'prior_unavailable'
            or set(plan.get('artifacts', {})) != ({'text_model'} if plan['purpose']=='calibration'
                                                else {'text_model', 'text_risk_model'})
            or plan.get('registration_id') != plan['schema']+':'+digest(
                {k:v for k,v in plan.items() if k!='registration_id'})[:24]):
        raise ValueError('Official context registration changed or has an unsupported phase')
    settings = _settings(plan['settings'])
    if plan.get('execution_spec') != _execution_spec(settings):
        raise ValueError('Official context execution settings differ from their registration')
    parse_time(plan['created_at'])
    return plan


def _artifact(path, model, at):
    meta = model.metadata
    if (meta.get('evidence_origin') not in ORIGINS or not parse_time(meta['training_as_of'])
            <= parse_time(meta['fit_execution_at']) <= at):
        raise ValueError('The actual fitted model must precede official context registration')
    return {'path': str(Path(path).resolve()), 'sha256': _sha(path), 'model_version': model.model_version,
        'training_as_of': meta['training_as_of'], 'fit_execution_at': meta['fit_execution_at'],
        'frozen_at': at.isoformat(), 'evidence_origin': meta['evidence_origin']}


def _admit(binding, *, identity_archive, previous_plans, predictor_path, value_model_path, purpose, at):
    verify_official_fixture(binding, archive_root=identity_archive, as_of=at)
    if at >= parse_time(binding['prediction_cutoff']):
        raise ValueError('Official context registration requires an unexpired prematch identity')
    if purpose not in SCHEMAS or (purpose=='confirmation') != (value_model_path is not None):
        raise ValueError('Calibration cannot use g; confirmation requires its frozen g')
    f = load_official_predictor(predictor_path)
    artifacts = {'text_model': _artifact(predictor_path, f, at)}
    canonical = binding['canonical']
    event = canonical['source_event_id']
    prior = _previous_inventory(previous_plans, at)
    if event in {r['source_event_id'] for r in prior}:
        raise ValueError('Official physical fixture already appears in prior A/B/C registration')
    f_events = _training_events(f, prior, 'f_train', {event})
    aliases = {mid:r['source_event_id'] for r in prior for mid in r['match_ids']}
    if (f_events != set(f.metadata['training_physical_match_ids'])
            or any(aliases.get(mid) != physical for mid, physical in f.metadata['training_aliases'].items())):
        raise ValueError('Predictor training aliases differ from registered physical A fixtures')
    g = None
    if purpose == 'confirmation':
        g = load_official_value(value_model_path, predictor_path=predictor_path)
        artifacts['text_risk_model'] = _artifact(value_model_path, g, at)
        g_events = set(g.metadata['training_physical_match_ids'])
        f_binding = {k:artifacts['text_model'][k] for k in ('sha256', 'model_version')}
        if (g.metadata['evidence_origin'] != f.metadata['evidence_origin']
                or g.metadata['predictor_sha256'] != f_binding['sha256']
                or g.metadata['predictor_model_version'] != f.model_version
                or set(g.metadata['predictor_training_physical_match_ids']) != f_events
                or {r['physical_match_id'] for r in g.support_records} != g_events
                or f_events & g_events or event in g_events):
            raise ValueError('Official A/B/C models must share one frozen f and disjoint physical fixtures')
        by_event = {r['source_event_id']: r for r in prior}
        for physical in g_events:
            row = by_event.get(physical)
            if (row is None or row['purpose'] != 'calibration'
                    or row['model_bindings'].get('text_model') != f_binding
                    or not parse_time(f.metadata['fit_execution_at']) <= parse_time(row['first_registered_at'])
                    < parse_time(g.metadata['training_as_of'])):
                raise ValueError('Value-model training needs registered B collected with this fitted f')
        if (set(g.metadata['training_aliases']) != set(g.metadata['training_match_ids'])
                or set(g.metadata['training_aliases'].values()) != g_events
                or any(aliases.get(mid) != physical for mid, physical in g.metadata['training_aliases'].items())):
            raise ValueError('Value-model training aliases differ from registered physical B fixtures')
    matches = [{'match_id': binding['match_id'], 'source_event_id': event,
        'fixture_facts': {k:canonical[k] for k in FACTS}, 'kickoff_at': canonical['kickoff'],
        'prediction_cutoff': binding['prediction_cutoff'], 'identity_binding_sha256': binding['binding_sha256']}]
    _merge_inventory(prior+_own_inventory({'purpose':purpose, 'created_at':at.isoformat(),
                                         'matches':matches, 'artifacts':artifacts}))
    return f, g, artifacts, matches, prior


def prepare_official_context_registration(identity_path, *, identity_archive, previous_plan_paths,
        predictor_path, output, purpose, settings, value_model_path=None, created_at=None):
    """Freeze an existing f (B), or matching f/g (C), before any acquisition."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Official context registration requires a new output file')
    now = datetime.now(timezone.utc)
    at = parse_time(now if created_at is None else created_at)
    if at > now:
        raise ValueError('Official context registration cannot be backfilled from a future clock')
    if not isinstance(previous_plan_paths, (list, tuple)):
        raise ValueError('Explicit previous registration paths are required')
    identity = Path(identity_path).resolve()
    raw = Path(identity_archive).resolve()
    if output.is_relative_to(raw):
        raise ValueError('Registration output cannot modify the original identity archive')
    settings = _settings(settings)
    previous_paths = [Path(p).resolve() for p in previous_plan_paths]
    binding, previous = _read(identity), [_read(p) for p in previous_paths]
    f, _, artifacts, matches, prior = _admit(binding, identity_archive=raw, previous_plans=previous,
        predictor_path=predictor_path, value_model_path=value_model_path, purpose=purpose, at=at)
    from football_agent.match_source_profiles import resolve_verified_profile
    profile = resolve_verified_profile(binding, version=5)
    # Resolve the existing capability/route contract now, before registration.
    from football_agent.news_discovery import NewsDiscoveryTools
    from football_spider.storage.raw_archive import RawArchive
    NewsDiscoveryTools(RawArchive(raw), profile['allowed_hosts'], provider='bing_rss',
        default_route=settings['route'], route_overrides=settings['news_route_overrides'], extraction_version=3,
        index_urls=[e['url'] for e in profile['entries'] if e['kind'] in {'news_index', 'club_home'}],
        projection_version=settings['news_projection_version'])
    plan = {'schema':SCHEMAS[purpose], 'purpose':purpose, 'created_at':at.isoformat(),
        'artifacts':artifacts, 'matches':matches, 'prior_inventory':prior, 'inventory_scope':SCOPE,
        'identity':binding, 'sources':{'identity':{'path':str(identity), 'sha256':_sha(identity), 'archive':str(raw)},
            'previous_plans':[{'path':str(p), 'sha256':_sha(p)} for p in previous_paths]},
        'settings':settings, 'execution_spec':_execution_spec(settings), 'source_profile':profile,
        'evidence_origin':f.metadata['evidence_origin'], 'prior_status':'prior_unavailable',
        'numeric_training_eligible':False, 'production_ready':False, 'independent_effect_confirmed':False,
        'scope':'prequery_official_B_C_without_market_prior_or_imputed_actor_observations'}
    plan['registration_id'] = plan['schema']+':'+digest(plan)[:24]
    _registration(plan)
    _write(output, plan)
    return plan


def validate_official_context_registration(plan, *, identity_archive, previous_plans, predictor_path,
                                           value_model_path=None, as_of):
    """Revalidate saved registration using original or run-local source copies."""
    _registration(plan)
    at, now = parse_time(plan['created_at']), parse_time(as_of)
    if not at < now < parse_time(plan['identity']['prediction_cutoff']):
        raise ValueError('Registration must precede actual prematch acquisition')
    f, g, artifacts, matches, prior = _admit(plan['identity'], identity_archive=identity_archive,
        previous_plans=previous_plans, predictor_path=predictor_path, value_model_path=value_model_path,
        purpose=plan['purpose'], at=at)
    for key, artifact in artifacts.items():
        saved = plan['artifacts'][key]
        if not Path(saved['path']).is_absolute():
            raise ValueError('Registered artifact path must be absolute')
        artifact['path'] = saved['path']
    from football_agent.match_source_profiles import resolve_verified_profile
    if (artifacts != plan['artifacts'] or matches != plan['matches'] or prior != plan['prior_inventory']
            or f.metadata['evidence_origin'] != plan['evidence_origin']
            or resolve_verified_profile(plan['identity'], version=5) != plan['source_profile']):
        raise ValueError('Official context models, exposure inventory or source profile differ')
    return {'identity':plan['identity'], 'registration':plan, 'predictor':f, 'value_model':g}


__all__ = ['CALIBRATION', 'CONFIRMATION', 'prepare_official_context_registration',
           'validate_official_context_registration']
