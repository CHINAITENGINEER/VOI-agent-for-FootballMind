"""Pre-query B cohort validation for the visible-context runtime.

Registrations and model snapshots are supervisor metadata. They never enter
the Agent's visible state or become a substitute for actual source returns.
"""
from hashlib import sha256
import json
from pathlib import Path

from football_research.acquisition.context_gain import _registration, _bind_model, _inventory, _digest
from football_research.acquisition.context_predictor import ContextPredictor
from football_research.acquisition.state import parse_time


def confirmation_policy_spec(*, seed, requests_per_match, source_profile_version,
                             news_projection_version, evidence_version, candidate_discovery_version,
                             news_provider, route, news_route_overrides, allowed_hosts, seed_urls, base_policy):
    """Freeze actual collector choices, including costs, before C tool calls."""
    from copy import deepcopy
    return {'schema': 'context-confirmation-policy-v1', 'seed': seed,
        'requests_per_match': requests_per_match, 'unknown_action_limit': requests_per_match,
        'source_profile_version': source_profile_version, 'news_projection_version': news_projection_version,
        'evidence_version': evidence_version,
        'candidate_discovery_version': (candidate_discovery_version if candidate_discovery_version is not None
            else {1: 2, 2: 3, 3: 4, 4: 10}[source_profile_version]),
        'news_provider': news_provider, 'route': route, 'news_route_overrides': deepcopy(news_route_overrides),
        'allowed_hosts': sorted(set(allowed_hosts)), 'seed_urls': sorted(set(seed_urls)),
        'loss': base_policy.manifest['loss'], 'lambda_cost': deepcopy(base_policy.lambda_cost),
        'web_cost': {'money': .01, 'requests': 1.}}


def validate_context_collection(context, *, predictor_path, as_of):
    from .prospective_learning import _bound_matches, _training_events, _merge_inventory, _own_inventory
    plan = _registration(context['registration'])
    model = ContextPredictor.load(predictor_path)
    model_sha = sha256(Path(predictor_path).read_bytes()).hexdigest()
    _bind_model(plan, model, model_sha, parse_time(as_of), model.metadata['evidence_origin'])
    if (plan['initialization_plan_sha256'] != _digest(context['initialization_plan']) or
            plan['identity_bundle_sha256'] != _digest(context['identity_bundle'])):
        raise ValueError('Context registration initialization or identity bundle changed')
    created = parse_time(plan['created_at'])
    matches = _bound_matches(context['initialization_plan'], context['identity_bundle'], created)
    if matches != plan['matches']:
        raise ValueError('Context registration physical matches changed')
    prior = _inventory(context['previous_plans'], created)
    if prior != plan['prior_inventory']:
        raise ValueError('Context registration phase inventory differs')
    events = _training_events(model, prior, 'f_train', {r['source_event_id'] for r in matches})
    if events != set(model.metadata['training_physical_match_ids']):
        raise ValueError('Context predictor training physical identities differ')
    _merge_inventory(prior + _own_inventory(plan))
    return {'predictor': model, 'binding': {'registration_id': plan['registration_id'],
        'registration_sha256': _digest(plan)}, 'matches': {r['match_id']: r for r in matches}}


def verify_context_run(folder, config):
    folder = Path(folder)
    metadata = config['supervisor_metadata']
    if config['schema'] != 'learned-voi-open-execution-v4':
        raise ValueError('Context calibration requires the actual visible-context runtime')
    raw = (folder/'context-calibration.json').read_bytes()
    if sha256(raw).hexdigest() != metadata['input_bindings'].get('context_calibration_sha256'):
        raise ValueError('Context calibration snapshot differs from the pre-query binding')
    context = json.loads(raw)
    checked = validate_context_collection(context, predictor_path=folder/'context-predictor.json',
                                          as_of=config['created_at'])
    mid = config['initial_state']['match_id']
    if (checked['binding'] != metadata.get('context_calibration') or
            context['identity_bundle']['bindings'].get(mid) != metadata.get('verified_fixture_identity') or
            checked['matches'][mid]['source_event_id'] != config['policy_manifest']['physical_match_id'] or
            config['policy_manifest']['base_policy'] not in context['initialization_plan']['selected']):
        raise ValueError('Context runtime does not match its pre-query physical fixture registration')
    if 'context_value_model' in config:
        raise ValueError('B calibration collection must preserve the registered exploration policy')
    return checked['binding']


def verify_context_confirmation_run(folder, config):
    """Bind a saved C run to f/g, source capabilities and pre-query policy."""
    from football_research.acquisition.context_confirmation import validate_context_confirmation
    from football_agent.evidence_candidates import EvidenceDocumentCandidates
    from football_agent.match_source_profiles import resolve_verified_profile
    from football_agent.inquiry_exploration import UniformInquiryExploration
    from football_spider.storage.raw_archive import RawArchive
    folder = Path(folder)
    metadata = config['supervisor_metadata']
    if (config['schema'] != 'learned-voi-open-execution-v4' or 'context_value_model' not in config
            or 'context_calibration' in metadata or 'learning_cohort' in metadata):
        raise ValueError('Context confirmation requires its independent f/g runtime')
    raw = (folder/'context-confirmation.json').read_bytes()
    if sha256(raw).hexdigest() != metadata['input_bindings'].get('context_confirmation_sha256'):
        raise ValueError('Context confirmation snapshot changed')
    context = json.loads(raw)
    spec = context['registration']['policy_spec']
    checked = validate_context_confirmation(context, predictor_path=folder/'context-predictor.json',
        value_model_path=folder/'context-value-model.json', as_of=config['created_at'], policy_spec=spec)
    mid = config['initial_state']['match_id']
    manifests = context['initialization_plan']['selected']
    index = next(i for i, manifest in enumerate(manifests)
                 if manifest['market_initialization']['fixture']['match_id'] == mid)
    manifest = manifests[index]
    binding = context['identity_bundle']['bindings'][mid]
    if (metadata.get('context_confirmation') != checked['binding']
            or metadata.get('verified_fixture_identity') != binding
            or config['policy_manifest']['physical_match_id'] != checked['matches'][mid]['source_event_id']
            or config['policy_manifest']['base_policy'] != manifest
            or config['policy_manifest']['lambda_cost'] != spec['lambda_cost']
            or manifest['loss'] != spec['loss']):
        raise ValueError('Context confirmation fixture or policy differs from registration')
    caps = {'money': .01*spec['requests_per_match'], 'requests': spec['requests_per_match']}
    if (config['limits'] != caps or config['exploration_limits'] != caps
            or config['max_exploration_actions'] != spec['unknown_action_limit']
            or config['max_steps'] != 16 or config['candidate_generation_complete'] is not False
            or config['inquiry_exploration'] != UniformInquiryExploration(spec['seed']+index).configuration()):
        raise ValueError('Context confirmation acquisition budget or sampler differs')
    profile = resolve_verified_profile(binding, version=spec['source_profile_version'])
    hosts = sorted(set(spec['allowed_hosts']+profile['allowed_hosts']))
    seeds = sorted(set(spec['seed_urls']+profile['seed_urls']))
    fixture = manifest['market_initialization']['fixture']
    identity = {'match_id': mid, 'home_team': fixture['home_team_name'],
        'away_team': fixture['away_team_name'], 'kickoff_at': fixture['kickoff'],
        'prediction_cutoff': manifest['market_initialization']['prediction_cutoff'],
        'home_aliases': binding['aliases']['home'], 'away_aliases': binding['aliases']['away']}
    generator = EvidenceDocumentCandidates(match_terms=[binding['canonical'][s+'_name'] for s in ('home','away')],
        seed_urls=seeds, allowed_hosts=hosts, numeric_actions=[], web_cost=spec['web_cost'],
        match=identity, profile=profile, discovery_version=spec['candidate_discovery_version'])
    if metadata['generator'] != generator.configuration():
        raise ValueError('Context confirmation candidate capabilities differ')
    if spec['news_provider']:
        from football_agent.news_discovery import NewsDiscoveryTools
        transport = NewsDiscoveryTools(RawArchive(folder/'source-archive'), hosts,
            provider=spec['news_provider'], default_route=spec['route'], extraction_version=3,
            route_overrides=spec['news_route_overrides'], projection_version=spec['news_projection_version'],
            index_urls=[entry['url'] for entry in profile['entries'] if entry['kind'] in {'news_index','club_home'}])
    else:
        from football_agent.inquiry_transport import RoutedOpenWebTools
        transport = RoutedOpenWebTools(RawArchive(folder/'source-archive'), hosts,
            route=spec['route'], extraction_version=3)
    if metadata['transport'] != transport.configuration():
        raise ValueError('Context confirmation transport differs')
    from football_agent.match_evidence import VERSION as DEFAULT_EVIDENCE
    if config['evidence_projection'] != {'match': identity, 'version': spec['evidence_version'] or DEFAULT_EVIDENCE}:
        raise ValueError('Context confirmation evidence projection differs')
    return checked['binding']
