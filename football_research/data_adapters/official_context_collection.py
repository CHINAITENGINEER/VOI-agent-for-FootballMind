"""Single-execution official B/C acquisition with archived f/g and no prior."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
from uuid import uuid4

from filelock import FileLock

from football_agent.learned_voi_runtime import digest
from football_research.acquisition.state import parse_time
from football_spider.storage.raw_archive import RawArchive
from .official_context_registration import (
    _read, _sha, _write, _registration, validate_official_context_registration,
)

CLAIM = 'official-context-execution-claim-v1'
RUNTIME = 'learned-voi-open-execution-v6'


def _components(checked, folder):
    from football_agent.official_context_policy import OfficialContextVOIPolicy
    from football_agent.evidence_candidates import EvidenceDocumentCandidates
    from football_agent.news_discovery import NewsDiscoveryTools
    plan, binding = checked['registration'], checked['identity']
    settings, profile = plan['settings'], plan['source_profile']
    from football_research.acquisition.official_context_models import is_role_predictor
    extra = {}
    if is_role_predictor(checked['predictor']):
        from .fixture_role_identity import resolve_fixture_role_identity
        extra['role_identity'] = resolve_fixture_role_identity(binding,
            archive_root=checked['identity_archive'], as_of=plan['created_at'])
    policy = OfficialContextVOIPolicy(binding, checked['predictor'], value_model=checked['value_model'],
        predictor_artifact=checked['predictor_bytes'], value_artifact=checked['value_bytes'],
        lambda_cost=settings['lambda_cost'], **extra)
    generator = EvidenceDocumentCandidates(
        match_terms=[binding['canonical'][side+'_name'] for side in ('home', 'away')],
        seed_urls=profile['seed_urls'], allowed_hosts=profile['allowed_hosts'], numeric_actions=[],
        web_cost=plan['execution_spec']['web_cost'], match=policy.evidence_match,
        profile=profile, discovery_version=11)
    tools = NewsDiscoveryTools(RawArchive(Path(folder)/'source-archive'), profile['allowed_hosts'],
        provider='bing_rss', default_route=settings['route'], route_overrides=settings['news_route_overrides'],
        extraction_version=3, index_urls=[entry['url'] for entry in profile['entries']
            if entry['kind'] in {'news_index', 'club_home'}], projection_version=settings['news_projection_version'])
    return policy, generator, tools


def _checked_models(plan, *, identity_archive, previous_plans, predictor_path, value_model_path, at):
    result = validate_official_context_registration(plan, identity_archive=identity_archive,
        previous_plans=previous_plans, predictor_path=predictor_path, value_model_path=value_model_path, as_of=at)
    result.update(predictor_bytes=Path(predictor_path).read_bytes(),
                  value_bytes=None if value_model_path is None else Path(value_model_path).read_bytes(),
                  identity_archive=str(Path(identity_archive).resolve()))
    # Bind bytes again after deserializing, before building an execution policy.
    if (_sha(predictor_path) != plan['artifacts']['text_model']['sha256'] or value_model_path is not None
            and _sha(value_model_path) != plan['artifacts']['text_risk_model']['sha256']):
        raise ValueError('Registered model bytes changed during validation')
    return result


def verify_official_context_run(folder, config):
    """Verify the original v6 supervisor contract using run-local copies only.

    Returns context, identity, registration and loaded predictor/value_model.
    Actual response projection and replay remain the existing runner's job.
    """
    from football_agent.inquiry_exploration import UniformInquiryExploration
    from football_agent.learned_voi_web import ArchivedAndWebExecutor
    folder = Path(folder)
    if config.get('schema') != RUNTIME:
        raise ValueError('Official context acquisition requires the explicit v6 runtime')
    metadata = config['supervisor_metadata']
    context = metadata['official_context_collection']
    plan = _registration(context['registration'])
    settings, claim = context['settings'], context['execution_claim']
    if (settings != plan['settings'] or _read(folder/'registration.json') != plan
            or _read(folder/'identity.json') != plan['identity']
            or metadata['verified_fixture_identity'] != plan['identity']
            or _sha(folder/'identity.json') != plan['sources']['identity']['sha256']
            or any(key in metadata for key in ('learning_cohort', 'context_calibration',
                                              'context_confirmation', 'evidence_collection'))):
        raise ValueError('Official context run identity, registration or settings differ')
    sources = plan['sources']['previous_plans']
    previous = []
    paths = sorted((folder/'registration-inputs').glob('previous-*.json'))
    if len(paths) != len(sources):
        raise ValueError('Run must retain all prior registration source copies')
    for index, source in enumerate(sources):
        path = folder/'registration-inputs'/f'previous-{index:03d}.json'
        if _sha(path) != source['sha256']:
            raise ValueError('Previous physical-registration source bytes changed')
        previous.append(_read(path))
    if previous != context['previous_plans']:
        raise ValueError('Previous physical-registration metadata differs from its archived bytes')
    model_paths = {}
    for key, config_key, filename in (
            ('text_model', 'context_predictor', 'context-predictor.json'),
            ('text_risk_model', 'context_value_model', 'context-value-model.json')):
        if key not in plan['artifacts']:
            if config_key in config or (folder/filename).exists():
                raise ValueError('Calibration cannot introduce a value model after registration')
            continue
        artifact, spec = plan['artifacts'][key], config[config_key]
        if (spec != {'file':filename, 'sha256':artifact['sha256'], 'model_version':artifact['model_version'],
                     'evidence_origin':artifact['evidence_origin']} or _sha(folder/filename) != artifact['sha256']):
            raise ValueError('Run model differs from its exact frozen official-context artifact')
        model_paths[key] = folder/filename
    checked = _checked_models(plan, identity_archive=folder/'identity-archive', previous_plans=previous,
        predictor_path=model_paths['text_model'], value_model_path=model_paths.get('text_risk_model'),
        at=config['created_at'])
    policy, generator, tools = _components(checked, folder)
    spec = plan['execution_spec']
    if (config['policy_manifest'] != policy.manifest or config['policy_version'] != policy.version
            or config['initial_state'] != policy.public_state(policy.initial_state)
            or config['evidence_projection'] != {'match':policy.evidence_match, 'version':spec['evidence_version']}
            or metadata['generator'] != generator.configuration() or metadata['transport'] != tools.configuration()
            or config['generator_version'] != generator.version+':'+digest(generator.configuration())
            or config['provider_version'] != ArchivedAndWebExecutor.version+':'+digest(tools.configuration())):
        raise ValueError('Official context policy or source capabilities differ from their frozen contract')
    if (set(claim) != {'schema', 'registration_id', 'registration_sha256', 'registration_file_sha256',
                       'source_event_id', 'run_id', 'output', 'claimed_at', 'settings'}
            or claim['schema'] != CLAIM or claim['registration_id'] != plan['registration_id']
            or claim['registration_sha256'] != digest(plan)
            or claim['registration_file_sha256'] != _sha(folder/'registration.json')
            or claim['source_event_id'] != plan['identity']['match_id'] or claim['run_id'] != config['run_id']
            or not Path(claim['output']).is_absolute() or claim['settings'] != settings
            or not parse_time(plan['created_at']) < parse_time(claim['claimed_at']) <= parse_time(config['created_at'])):
        raise ValueError('Official context execution differs from its single pre-query claim')
    if (config['limits'] != spec['limits'] or config['exploration_limits'] != spec['exploration_limits']
            or config['max_steps'] != spec['max_steps']
            or config['max_exploration_actions'] != spec['max_exploration_actions']
            or config['inquiry_exploration'] != UniformInquiryExploration(settings['seed']).configuration()):
        raise ValueError('Official context physical or exploratory budget changed')
    result = {key:checked[key] for key in ('identity', 'registration', 'predictor', 'value_model')} | {'context':context}
    if policy.role_context_mode:
        result['role_identity'] = deepcopy(policy.role_identity)
    return result


def collect_official_context(registration_path, *, output):
    """Execute one frozen B/C registration once, including interrupted attempts."""
    from football_agent.learned_voi_runtime import LearnedVOIRuntime
    from football_agent.learned_voi_web import ArchivedAndWebExecutor
    from football_agent.match_analysis_report import write_analysis_report
    registration_path, output = Path(registration_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Official context runs are replayed, never reacquired')
    plan = _registration(_read(registration_path))
    source = plan['sources']['identity']
    if _sha(source['path']) != source['sha256'] or _read(source['path']) != plan['identity']:
        raise ValueError('Original official identity changed after registration')
    previous = []
    for reference in plan['sources']['previous_plans']:
        if _sha(reference['path']) != reference['sha256']:
            raise ValueError('Previous registration changed after official context admission')
        previous.append(_read(reference['path']))
    now = datetime.now(timezone.utc)
    checked = _checked_models(plan, identity_archive=source['archive'], previous_plans=previous,
        predictor_path=plan['artifacts']['text_model']['path'],
        value_model_path=plan['artifacts'].get('text_risk_model', {}).get('path'), at=now)
    binding, settings = checked['identity'], plan['settings']
    policy, generator, tools = _components(checked, output)
    profile = plan['source_profile']
    row = {'match_id':binding['match_id'], 'prediction_cutoff':binding['prediction_cutoff'],
        'features_available_at':binding['verified_at'], 'information_mode':'operational', **policy.initial_state['base']}
    executor = ArchivedAndWebExecutor(row, [], tools, seed_urls=profile['seed_urls'])
    run_id = 'official-context-'+uuid4().hex
    executor.run_id = run_id
    claim_path = registration_path.with_name(registration_path.name+'.acquisition.json')
    claim = {'schema':CLAIM, 'registration_id':plan['registration_id'], 'registration_sha256':digest(plan),
        'registration_file_sha256':_sha(registration_path), 'source_event_id':binding['match_id'],
        'run_id':run_id, 'output':str(output), 'claimed_at':now.isoformat(), 'settings':settings}
    with FileLock(str(claim_path)+'.lock', timeout=0):
        _write(claim_path, claim)
    metadata = {'verified_fixture_identity':binding, 'generator':generator.configuration(),
        'transport':tools.configuration(), 'official_context_collection':{'registration':plan,
            'previous_plans':previous, 'settings':settings, 'execution_claim':claim},
        'initialization':{'status':'prior_unavailable', 'probabilities':None, 'numeric_voi':None}}

    def copy_inputs():
        if not (output/'identity-archive').exists():
            shutil.copytree(source['archive'], output/'identity-archive')
        pairs = [(registration_path, output/'registration.json'), (Path(source['path']), output/'identity.json')]
        pairs += [(Path(reference['path']), output/'registration-inputs'/f'previous-{i:03d}.json')
                  for i, reference in enumerate(plan['sources']['previous_plans'])]
        for original, local in pairs:
            data = original.read_bytes()
            if local.exists():
                if local.read_bytes() != data:
                    raise ValueError('Original registered acquisition inputs changed during execution')
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                with local.open('xb') as stream:
                    stream.write(data)
        from .official_fixture import verify_official_fixture
        verify_official_fixture(binding, archive_root=output/'identity-archive', as_of=now)
        # The runner has persisted its exact model bytes and configuration
        # before calling execute. Validate all local source copies before I/O.
        verify_official_context_run(output, _read(output/'configuration.json'))

    def execute(action, public):
        copy_inputs()
        return executor(action, public)

    spec = plan['execution_spec']
    result = LearnedVOIRuntime(policy, seed=settings['seed'], exploration_limits=spec['exploration_limits'],
        max_exploration_actions=spec['max_exploration_actions'], max_steps=spec['max_steps']).run(
            policy.initial_state, run_id=run_id, limits=spec['limits'], propose=generator, execute=execute,
            output=output, provider_version=executor.version+':'+digest(tools.configuration()),
            generator_version=generator.version+':'+digest(generator.configuration()), run_metadata=metadata,
            evidence_match=policy.evidence_match, evidence_version=spec['evidence_version'])
    copy_inputs()
    verify_official_context_run(output, _read(output/'configuration.json'))
    write_analysis_report(result, match=policy.evidence_match, output=output)
    return {'status':'collected_official_context', 'purpose':plan['purpose'], 'run_directory':str(output),
        'run_id':run_id, 'match_id':binding['match_id'], 'source_event_id':binding['match_id'],
        'evidence_origin':plan['evidence_origin'], 'http_requests':result['measured_resources']['requests'],
        'documents':len(result['documents']), 'prior_probabilities':None,
        'final_probabilities':result['final_probabilities'], 'numeric_information_value_is_calibrated':False,
        'terminal_selection':result['terminal_selection'], 'independent_labels':0}


__all__ = ['collect_official_context', 'verify_official_context_run']
