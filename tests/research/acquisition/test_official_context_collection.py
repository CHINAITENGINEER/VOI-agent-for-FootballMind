"""Constructed official B/C contracts, with all network access forbidden."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket
from zoneinfo import ZoneInfo

import pytest

from football_research.data_adapters import official_context_registration as registration
from football_research.data_adapters import official_context_collection as collection
from football_research.data_adapters.prospective_learning import _previous_inventory
from football_research.data_adapters.official_fixture import prepare_official_fixture
from football_spider.storage.raw_archive import digest
from test_context_gain import exposure
from test_optional_prior_predictor import fit, rows
from test_official_fixture import Transport, EVENT
from test_finland_stage_fixture import schedule
from test_evidence_collection import transport

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Official context tests cannot use the network')
    for owner, key in ((socket, 'getaddrinfo'), (socket, 'create_connection'),
                       (socket.socket, 'connect'), (socket.socket, 'connect_ex')):
        monkeypatch.setattr(owner, key, blocked)


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    f = fit(rows(), 'constructed-official-context-collection')
    path = tmp_path_factory.mktemp('official-context-f')/'f.json'
    f.save(path)
    return f, path


def settings(requests=2):
    return {'requests_per_match': requests, 'seed': 198, 'route': 'direct',
        'news_route_overrides': None, 'news_projection_version': 6,
        'lambda_cost': {'money': 1., 'requests': 0., 'latency_ms': 0., 'tokens': 0.}}


def registration_options(root, model_path, previous_plans, *, requests=2):
    at = datetime.now(timezone.utc)
    kickoff = (at+timedelta(hours=6)).astimezone(ZoneInfo('Europe/Helsinki'))
    prepared = prepare_official_fixture(EVENT, output=root/'prepare', clock=lambda: at,
        transport=Transport(schedule(date=kickoff.strftime('%Y%m%d'), clock=kickoff.strftime('%H:%M'))))
    paths = []
    for i, value in enumerate(previous_plans):
        path = root/f'previous-{i}.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        paths.append(path)
    return {'identity_path': prepared['identity_path'], 'identity_archive': prepared['identity_archive'],
        'previous_plan_paths': paths, 'predictor_path': model_path, 'output': root/'registration.json',
        'purpose': 'calibration', 'settings': settings(requests)}


def registered_inputs(root, model_path, previous_plans, *, purpose='calibration', value_model_path=None, requests=2):
    options = registration_options(root, model_path, previous_plans, requests=requests)
    options.update(purpose=purpose, value_model_path=value_model_path)
    registration.prepare_official_context_registration(**options)
    return options['output']


def test_official_B_registration_binds_actual_f_identity_and_physical_exposure(tmp_path, model):
    path = registered_inputs(tmp_path, model[1], [exposure(model[0])])
    plan = read(path)
    assert plan['schema'] == registration.CALIBRATION
    assert plan['purpose'] == 'calibration' and plan['prior_status'] == 'prior_unavailable'
    assert plan['evidence_origin'] == 'constructed_mechanism'
    assert set(plan['artifacts']) == {'text_model'}
    assert plan['artifacts']['text_model']['sha256'] == sha256(model[1].read_bytes()).hexdigest()
    assert plan['execution_spec']['max_exploration_actions'] == 2
    assert plan['execution_spec']['limits'] == {'money': .02, 'requests': 2}
    assert not plan['numeric_training_eligible'] and not plan['production_ready']
    assert 'initialization_manifest' not in json.dumps(plan)
    inventory = _previous_inventory([exposure(model[0]), plan], datetime.now(timezone.utc))
    row = next(r for r in inventory if r['source_event_id'] == EVENT)
    assert row['purpose'] == 'calibration'
    assert row['model_bindings']['text_model']['sha256'] == plan['artifacts']['text_model']['sha256']
    options = registration_options(tmp_path/'repeat', model[1], [exposure(model[0]), plan])
    with pytest.raises(ValueError, match='prior A/B/C'):
        registration.prepare_official_context_registration(**options)


@pytest.mark.parametrize('change', ['no_A', 'swapped_alias', 'g_for_B', 'missing_g_for_C',
                                   'future_clock', 'before_fit', 'raw', 'settings'])
def test_registration_rejects_unbound_training_phases_clocks_and_raw(tmp_path, model, change):
    old = exposure(model[0])
    if change == 'swapped_alias':
        old['matches'][0]['source_event_id'], old['matches'][1]['source_event_id'] = (
            old['matches'][1]['source_event_id'], old['matches'][0]['source_event_id'])
        old['cohort_id'] = old['schema']+':'+digest({k:v for k,v in old.items() if k!='cohort_id'})[:24]
    options = registration_options(tmp_path, model[1], [] if change=='no_A' else [old])
    if change=='g_for_B':
        options['value_model_path'] = model[1]
    elif change=='missing_g_for_C':
        options['purpose'] = 'confirmation'
    elif change=='future_clock':
        options['created_at'] = datetime.now(timezone.utc)+timedelta(hours=1)
    elif change=='before_fit':
        options['created_at'] = '2025-01-14T00:00:00Z'
    elif change=='raw':
        path = next(Path(options['identity_archive']).rglob('*.observation.json'))
        raw = read(path); raw['params'] = {'group':'1'}
        path.write_text(json.dumps(raw), encoding='utf-8')
    elif change=='settings':
        options['settings']['requests_per_match'] = True
    with pytest.raises(ValueError):
        registration.prepare_official_context_registration(**options)
    assert not options['output'].exists()


def test_validation_uses_archived_f_and_identity_without_original_model_path(tmp_path, model):
    original = tmp_path/'f-original.json'
    original.write_bytes(model[1].read_bytes())
    path = registered_inputs(tmp_path, original, [exposure(model[0])])
    plan = read(path)
    local = tmp_path/'f-local.json'
    local.write_bytes(original.read_bytes())
    original.unlink()
    checked = registration.validate_official_context_registration(plan,
        identity_archive=plan['sources']['identity']['archive'], previous_plans=[exposure(model[0])],
        predictor_path=local, as_of=datetime.now(timezone.utc))
    assert checked['predictor'].model_version == model[0].model_version
    assert checked['value_model'] is None
    with pytest.raises(FileExistsError):
        registration.prepare_official_context_registration(
            plan['sources']['identity']['path'], identity_archive=plan['sources']['identity']['archive'],
            previous_plan_paths=[], predictor_path=local, output=path,
            purpose='calibration', settings=plan['settings'])


def test_formal_B_collector_no_market_prior_single_execution_and_exact_replay(tmp_path, model, monkeypatch):
    from football_agent.learned_voi_replay import replay_saved_run
    path = registered_inputs(tmp_path, model[1], [exposure(model[0])], requests=4)
    calls = transport(monkeypatch)
    report = collection.collect_official_context(path, output=tmp_path/'run')
    config, run = read(tmp_path/'run/configuration.json'), read(tmp_path/'run/run.json')
    checked = collection.verify_official_context_run(tmp_path/'run', config)
    assert report['purpose']=='calibration' and report['http_requests']==len(calls)==4
    assert checked['registration']==read(path) and checked['value_model'] is None
    assert config['schema']=='learned-voi-open-execution-v6'
    assert (tmp_path/'run/context-predictor.json').read_bytes()==model[1].read_bytes()
    assert not (tmp_path/'run/context-value-model.json').exists()
    assert not (tmp_path/'run/prior-archive').exists()
    assert report['prior_probabilities'] is None and len(run['final_probabilities'])==3
    checked_replay = replay_saved_run(tmp_path/'run', output=tmp_path/'replay', project_root=ROOT)
    assert checked_replay['identical_replay'] and checked_replay['replay_network_requests']==0
    assert len(calls)==4
    for output in (tmp_path/'run', tmp_path/'another-run'):
        with pytest.raises(FileExistsError):
            collection.collect_official_context(path, output=output)
    assert len(calls)==4


def test_unreconciled_first_response_keeps_claim_and_cannot_repeat_requests(tmp_path, model, monkeypatch):
    path = registered_inputs(tmp_path, model[1], [exposure(model[0])])
    calls = transport(monkeypatch, interrupt=True)
    report = collection.collect_official_context(path, output=tmp_path/'run')
    assert len(calls)==1 and report['terminal_selection']['reason']=='execution_reconciliation_required'
    with pytest.raises(FileExistsError):
        collection.collect_official_context(path, output=tmp_path/'retry')
    assert len(calls)==1


@pytest.mark.parametrize('change', ['budget', 'raw_identity', 'registration', 'model', 'previous'])
def test_run_verifier_rejects_modified_budget_identity_model_and_phase_inputs(tmp_path, model, monkeypatch, change):
    path = registered_inputs(tmp_path, model[1], [exposure(model[0])], requests=1)
    calls = transport(monkeypatch)
    collection.collect_official_context(path, output=tmp_path/'run')
    config = read(tmp_path/'run/configuration.json')
    if change=='budget':
        config['limits']['requests'] = 4
    elif change=='registration':
        config['supervisor_metadata']['official_context_collection']['registration']['purpose'] = 'confirmation'
    else:
        changed_path = (next((tmp_path/'run/identity-archive').rglob('*.observation.json'))
            if change=='raw_identity' else tmp_path/'run/context-predictor.json'
            if change=='model' else tmp_path/'run/registration-inputs/previous-000.json')
        changed_path.write_text('{}', encoding='utf-8')
    with pytest.raises((ValueError, KeyError)):
        collection.verify_official_context_run(tmp_path/'run', config)
    assert len(calls)==1
