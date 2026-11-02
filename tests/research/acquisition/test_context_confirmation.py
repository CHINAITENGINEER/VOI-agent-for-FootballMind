"""Constructed f/g C admission checks; no real fitting or efficacy claim."""
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from football_research.acquisition.context_confirmation import (
    REGISTRATION, prepare_context_confirmation, validate_context_confirmation,
)
from football_research.acquisition.context_gain import _digest, _inventory
from football_research.data_adapters.context_collection import confirmation_policy_spec
from football_research.data_adapters.prospective_learning import _merge_inventory, _own_inventory
from test_context_gain import exposure, registration
from test_context_value import fitted
from test_context_value_policy import inputs_with_loss_conversion, AT


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Confirmation registration must never acquire remote information')
    for owner, name in ((socket, 'getaddrinfo'), (socket, 'create_connection'),
                        (socket.socket, 'connect'), (socket.socket, 'connect_ex')):
        monkeypatch.setattr(owner, name, forbidden)


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp('constructed-confirmation-fg')
    f, fpath, g = fitted(root)
    gpath = root/'g.json'
    g.save(gpath)
    return f, fpath, g, gpath


def prior_plans(models):
    f, fpath, g, _ = models
    a = exposure(f)
    facts = a['matches'][0]['fixture_facts']
    b = registration((f, fpath, sha256(fpath.read_bytes()).hexdigest()), [
        {'match_id':'provider:'+physical, 'source_event_id':physical, 'fixture_facts':deepcopy(facts)}
        for physical in g.metadata['training_physical_match_ids']])
    b['registration_id'] = b['schema']+':'+_digest({k:v for k,v in b.items() if k!='registration_id'})[:24]
    return [a,b]


def setup(root, models):
    inputs, base = inputs_with_loss_conversion(root/'source', 1.)
    policy = confirmation_policy_spec(seed=197, requests_per_match=3, source_profile_version=1,
        news_projection_version=1, evidence_version=None, candidate_discovery_version=2,
        news_provider='bing_rss', route='environment', news_route_overrides=None,
        allowed_hosts=['www.uefa.com'], seed_urls=[], base_policy=base)
    previous = prior_plans(models)
    return {'initialization_plan':inputs['plan'],
        'identity_bundle':{'bindings':{inputs['binding']['match_id']:inputs['binding']}},
        'previous_plans':previous, 'predictor_path':models[1], 'value_model_path':models[3],
        'policy_spec':policy, 'created_at':AT, 'output':root/'registration.json'}


def prepare(options):
    initial = options['initialization_plan']
    return prepare_context_confirmation(initial, **{k:v for k,v in options.items() if k!='initialization_plan'})


def context(options, plan):
    return {key:deepcopy(options[key]) for key in ('initialization_plan','identity_bundle','previous_plans')} | {
        'registration':plan}


def validate(options, plan, **kwargs):
    return validate_context_confirmation(context(options, plan), predictor_path=options['predictor_path'],
        value_model_path=options['value_model_path'], as_of=kwargs.pop('as_of',AT+timedelta(seconds=1)),
        policy_spec=kwargs.pop('policy_spec',options['policy_spec']), **kwargs)


def resign(plan):
    key = 'cohort_id' if 'cohort_id' in plan else 'registration_id'
    plan[key] = plan['schema']+':'+_digest({k:v for k,v in plan.items() if k!=key})[:24]


def save_changed_model(path, original, changes):
    payload = json.loads(Path(original).read_text(encoding='utf-8'))
    changes(payload)
    payload['model_version'] = payload['schema']+':'+_digest({k:v for k,v in payload.items() if k!='model_version'})[:24]
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)+'\n',encoding='utf-8')
    return path


def test_registers_exact_existing_f_g_and_settings_without_claiming_effect(models, tmp_path):
    options = setup(tmp_path, models)
    original = deepcopy(options['initialization_plan'])
    plan = prepare(options)
    assert plan['schema'] == REGISTRATION and plan['purpose'] == 'confirmation'
    assert plan['evidence_origin'] == 'constructed_mechanism'
    assert plan['supervisor_only'] and not plan['production_ready'] and not plan['independent_effect_confirmed']
    assert plan['artifacts']['text_model']['sha256'] == sha256(models[1].read_bytes()).hexdigest()
    assert plan['artifacts']['text_risk_model']['sha256'] == sha256(models[3].read_bytes()).hexdigest()
    checked = validate(options, plan)
    assert checked['predictor'].model_version == models[0].model_version
    assert checked['value_model'].model_version == models[2].model_version
    assert checked['binding'] == {'registration_id':plan['registration_id'],
        'registration_sha256':_digest(plan), 'policy_spec_sha256':_digest(options['policy_spec'])}
    assert checked['matches']['spftll:match:101']['source_event_id'] == 'uefa:match:2050993'
    assert options['initialization_plan'] == original
    with pytest.raises(FileExistsError):
        prepare(options)


def test_run_local_model_copies_validate_without_original_paths(models, tmp_path):
    options = setup(tmp_path, models)
    original_f, original_g = tmp_path/'registered-f.json', tmp_path/'registered-g.json'
    original_f.write_bytes(models[1].read_bytes())
    original_g.write_bytes(models[3].read_bytes())
    options.update(predictor_path=original_f, value_model_path=original_g)
    plan = prepare(options)
    local_f, local_g = tmp_path/'local-f.json', tmp_path/'local-g.json'
    local_f.write_bytes(original_f.read_bytes())
    local_g.write_bytes(original_g.read_bytes())
    original_f.unlink()
    original_g.unlink()
    options.update(predictor_path=local_f, value_model_path=local_g)
    assert validate(options, plan)['value_model'].model_version == models[2].model_version


@pytest.mark.parametrize('field,value', [('seed',199), ('requests_per_match',2),
    ('candidate_discovery_version',3), ('route','direct'), ('lambda_cost',{'money':2.,'requests':0.,'latency_ms':0.,'tokens':0.})])
def test_actual_policy_cannot_silently_change_registered_choices(models, tmp_path, field, value):
    options = setup(tmp_path, models)
    plan = prepare(options)
    actual = deepcopy(options['policy_spec'])
    actual[field] = value
    if field == 'requests_per_match': actual['unknown_action_limit'] = value
    with pytest.raises(ValueError, match='registered settings'):
        validate(options, plan, policy_spec=actual)


@pytest.mark.parametrize('field,value', [('seed',True), ('unknown_action_limit',0),
    ('allowed_hosts',['www.uefa.com','www.uefa.com']), ('lambda_cost',{'money':float('nan')}),
    ('web_cost',{'money':.01,'requests':True}), ('candidate_discovery_version',None)])
def test_incomplete_or_ambiguous_policy_rejected_without_output(models, tmp_path, field, value):
    options = setup(tmp_path, models)
    options['policy_spec'][field] = value
    with pytest.raises(ValueError):
        prepare(options)
    assert not options['output'].exists()


def test_policy_loss_conversion_must_match_every_frozen_market_prior(models, tmp_path):
    options = setup(tmp_path, models)
    options['policy_spec']['lambda_cost']['money'] = 9.
    with pytest.raises(ValueError, match='cost/loss'):
        prepare(options)
    assert not options['output'].exists()


@pytest.mark.parametrize('route', ['environment', 'direct', 'system_tls'])
def test_source_route_overrides_bind_the_existing_three_transport_modes(models, tmp_path, route):
    options = setup(tmp_path, models)
    options['policy_spec']['news_route_overrides'] = {'www.uefa.com':route}
    plan = prepare(options)
    # Source-specific system TLS endpoint admission stays in the actual tools;
    # registering a route never performs a request or grants a new capability.
    assert validate(options, plan)['binding']['policy_spec_sha256'] == _digest(options['policy_spec'])


@pytest.mark.parametrize('event', ['official:0','B:0'])
def test_provider_alias_or_postponement_cannot_relabel_an_A_or_B_fixture_as_C(models, tmp_path, event):
    options = setup(tmp_path, models)
    fixture = options['identity_bundle']['bindings']['spftll:match:101']
    fixture['canonical']['source_event_id'] = event
    fixture['binding_sha256'] = _digest({k:v for k,v in fixture.items() if k!='binding_sha256'})
    with pytest.raises(ValueError, match='prior A/B/C'):
        prepare(options)
    assert not options['output'].exists()


@pytest.mark.parametrize('change', ['missing_B','wrong_B_f','late_B','swapped_A_aliases'])
def test_existing_physical_inventory_must_explain_both_training_phases(models, tmp_path, change):
    options = setup(tmp_path, models)
    a, b = options['previous_plans']
    if change == 'missing_B':
        b['matches'].pop()
    elif change == 'wrong_B_f':
        b['artifacts']['text_model']['sha256'] = 'f'*64
    elif change == 'late_B':
        b['created_at'] = models[2].metadata['training_as_of']
    else:
        a['matches'][0]['source_event_id'], a['matches'][1]['source_event_id'] = (
            a['matches'][1]['source_event_id'], a['matches'][0]['source_event_id'])
    resign(a)
    resign(b)
    with pytest.raises(ValueError):
        prepare(options)
    assert not options['output'].exists()


def test_registered_C_is_visible_to_inventory_and_cannot_be_reexposed_as_A_or_B(models, tmp_path):
    options = setup(tmp_path, models)
    plan = prepare(options)
    rows = _inventory(options['previous_plans']+[plan], AT+timedelta(seconds=1))
    c = next(r for r in rows if r['source_event_id']=='uefa:match:2050993')
    assert c['purpose'] == 'confirmation' and set(c['model_bindings']) == {'text_model','text_risk_model'}
    for purpose in ('f_train','calibration'):
        other = deepcopy(plan)
        other.update(purpose=purpose, artifacts={} if purpose=='f_train' else {'text_model':plan['artifacts']['text_model']})
        with pytest.raises(ValueError, match='cannot change cohort purpose'):
            _merge_inventory(rows+_own_inventory(other))
    options['previous_plans'].append(plan)
    options.update(created_at=AT+timedelta(seconds=1), output=tmp_path/'reregistered.json')
    with pytest.raises(ValueError, match='prior A/B/C'):
        prepare(options)
    assert not options['output'].exists()


def test_actual_fitting_registration_and_prediction_deadline_are_ordered(models, tmp_path):
    options = setup(tmp_path, models)
    plan = prepare(options)
    with pytest.raises(ValueError, match='strictly precede'):
        validate(options, plan, as_of=AT)
    cutoff = plan['matches'][0]['prediction_cutoff']
    with pytest.raises(ValueError, match='horizon has expired'):
        validate(options, plan, as_of=cutoff)
    options['created_at'] = '2027-02-01T00:00:00Z'
    options['output'] = tmp_path/'before-g-fit.json'
    with pytest.raises(ValueError, match='Actual model fitting'):
        prepare(options)
    assert not options['output'].exists()


def test_identical_predictor_payload_with_changed_bytes_is_not_gs_frozen_f(models, tmp_path):
    options = setup(tmp_path, models)
    path = tmp_path/'same-payload-different-bytes.json'
    path.write_bytes(models[1].read_bytes()+b'\n')
    options['predictor_path'] = path
    with pytest.raises(ValueError, match='another f artifact'):
        prepare(options)
    assert not options['output'].exists()


def test_changed_g_and_origin_cannot_pass_a_frozen_registration(models, tmp_path):
    options = setup(tmp_path, models)
    plan = prepare(options)
    path = save_changed_model(tmp_path/'different-g.json', models[3],
                              lambda p:p['coefficients'].__setitem__(0,p['coefficients'][0]+.1))
    options['value_model_path'] = path
    with pytest.raises(ValueError, match='models, fixture identities or phase inventory'):
        validate(options, plan)
    options['value_model_path'] = save_changed_model(tmp_path/'wrong-origin.json', models[3],
        lambda p:p['metadata'].__setitem__('evidence_origin','real_observational'))
    with pytest.raises(ValueError, match='another f artifact or origin'):
        prepare(options)


def test_saved_context_or_inventory_cannot_change_without_detection(models, tmp_path):
    options = setup(tmp_path, models)
    plan = prepare(options)
    changed = deepcopy(plan)
    changed['policy_spec']['seed'] += 1
    with pytest.raises(ValueError, match='registration content differs'):
        validate(options, changed)
    options['identity_bundle']['bindings']['spftll:match:101']['aliases']['home'].append('invented')
    with pytest.raises(ValueError, match='identity bundle changed'):
        validate(options, plan)
