"""Public C collector with fitted f/g and actual archived constructed returns.

No policy/predictor method is replaced. February 2027 is an explicit offline
test clock; zero cost conversion exercises positive-g selection, not efficacy.
"""
from copy import deepcopy
from datetime import datetime, timedelta
from hashlib import sha256
import json
import socket

import pytest

from football_agent.learned_voi_replay import replay_saved_run
from football_research.acquisition.context_confirmation import prepare_context_confirmation
from football_research.data_adapters.context_collection import confirmation_policy_spec, verify_context_confirmation_run
from football_research.data_adapters.prospective_text import collect_prepared_batch
from test_context_gain import exposure, registration, _digest
from test_context_value import fitted, CONTEXTS
from test_context_value_policy import inputs_with_loss_conversion, AT


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a, **k):
        pytest.fail('Constructed C validation cannot perform network calls')
    for owner, name in ((socket, 'getaddrinfo'), (socket.socket, 'connect'), (socket.socket, 'connect_ex')):
        monkeypatch.setattr(owner, name, blocked)


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    return fitted(tmp_path_factory.mktemp('registered-context-C'))


def setup(root, models):
    f, fpath, g = models
    inputs, base = inputs_with_loss_conversion(root/'source', 0.)
    inputs['body'] = inputs['body'].replace(
        b'Croatia may rotate their starting lineup against Spain. Spain will assess player fitness before the match.',
        CONTEXTS['positive'].encode())
    gpath = root/'g.json'
    # Non-default formatting is intentional: C must persist registered bytes.
    gpath.write_text(json.dumps({**g._payload(), 'model_version':g.model_version}, indent=4)+'\n', encoding='utf-8')
    a = exposure(f)
    b = registration((f,fpath,sha256(fpath.read_bytes()).hexdigest()),
        [{'match_id':'provider:'+event, 'source_event_id':event,
          'fixture_facts':deepcopy(a['matches'][0]['fixture_facts'])}
         for event in g.metadata['training_physical_match_ids']])
    b['registration_id'] = b['schema']+':'+_digest({k:v for k,v in b.items() if k!='registration_id'})[:24]
    previous = []
    for i, plan in enumerate([a,b]):
        path = root/f'previous-{i}.json'
        path.write_text(json.dumps(plan), encoding='utf-8')
        previous.append(path)
    options = {'allowed_hosts':['www.uefa.com'], 'requests_per_match':3, 'seed':180,
        'seed_urls':['https://www.uefa.com/news/unread-follow-up/'], 'route':'environment',
        'source_profile_version':1, 'news_projection_version':1, 'news_provider':'bing_rss',
        'evidence_version':None, 'candidate_discovery_version':None, 'news_route_overrides':None}
    spec = confirmation_policy_spec(**options, base_policy=base)
    cpath = root/'C.json'
    prepare_context_confirmation(inputs['plan'], identity_bundle={'bindings':{inputs['binding']['match_id']:inputs['binding']}},
        previous_plans=[a,b], predictor_path=fpath, value_model_path=gpath, policy_spec=spec,
        output=cpath, created_at=AT+timedelta(milliseconds=100))
    return inputs, options, previous, cpath, fpath, gpath


def collect(root, models, monkeypatch, *, drift=None):
    import football_agent.inquiry_tools as web
    import football_agent.learned_voi_runtime as runtime
    import football_research.data_adapters.prospective_text as collector
    from football_agent.news_discovery import NewsDiscoveryTools
    inputs, options, previous, cpath, fpath, gpath = setup(root, models)
    tick = [AT+timedelta(milliseconds=200)]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            tick[0] += timedelta(milliseconds=10)
            return tick[0].astimezone(tz) if tz else tick[0].replace(tzinfo=None)
    calls = []
    def archived_return(self, prepared, **kwargs):
        calls.append(deepcopy(prepared))
        now = Clock.now(AT.tzinfo)
        body = b'<rss><channel></channel></rss>' if prepared['tool']=='search_web' else inputs['body']
        obs = self.archive.save(prepared['source'], prepared['endpoint'], prepared['params'], body,
            status=200, observed_at=now, context={'requested_at':now.isoformat()})
        return {'document':self.project(obs,prepared), 'observation':obs.model_dump(mode='json'),
            'http_attempts':1, 'money':None, 'money_status':'not_measured', 'latency_seconds':.01}
    if drift:
        options.update(drift)
    with monkeypatch.context() as mp:
        for module in (web,runtime,collector):
            mp.setattr(module,'datetime',Clock)
        mp.setattr(NewsDiscoveryTools,'execute',archived_return)
        report = collect_prepared_batch(inputs['root']/'plan.json', output=root/'batch',
            archive_root=inputs['root']/'market', identity_bundle=inputs['root']/'bindings.json',
            identity_archive=inputs['root']/'identity', context_confirmation_path=cpath,
            context_predictor_path=fpath, context_value_model_path=gpath,
            previous_learning_plan_paths=previous, **options)
    return root/'batch/match-101', calls, report, fpath, gpath


def test_registered_f_g_public_collector_selects_updates_and_exactly_replays(models, tmp_path, monkeypatch):
    folder, calls, report, fpath, gpath = collect(tmp_path,models,monkeypatch)
    run, config = read(folder/'run.json'), read(folder/'configuration.json')
    assert len(calls)==report['http_requests']==3
    assert verify_context_confirmation_run(folder,config)==config['supervisor_metadata']['context_confirmation']
    selected = [c for c in run['cycles'] if c['selected_action']]
    assert selected[0]['selection_basis']=='bounded_unknown_information_exploration'
    assert any(c['selection_basis']=='computed_positive_empirical_gain' for c in selected[1:])
    assert any(c['update']['probability_changed'] for c in selected)
    assert all('context_confirmation' not in c['public_before'] for c in run['cycles'])
    assert (folder/'context-predictor.json').read_bytes()==fpath.read_bytes()
    assert (folder/'context-value-model.json').read_bytes()==gpath.read_bytes()
    assert not run['context_response_learning']['real_world_benefit_proven']
    replay = replay_saved_run(folder,output=tmp_path/'replay',project_root=tmp_path)
    assert replay['identical_replay'] and replay['replay_network_requests']==0
    from football_research.data_adapters.prospective_learning import import_training_exposure
    with pytest.raises(ValueError,match='confirmation exposure'):
        import_training_exposure([folder],created_at=AT+timedelta(hours=1))


@pytest.mark.parametrize('drift',[{'requests_per_match':2},{'seed':181},{'source_profile_version':2}])
def test_confirmation_configuration_drift_rejected_before_requests(models,tmp_path,monkeypatch,drift):
    with pytest.raises(ValueError,match='policy differs'):
        collect(tmp_path,models,monkeypatch,drift=drift)
    assert not (tmp_path/'batch').exists()


def test_confirmation_raw_model_change_or_runtime_budget_breaks_verification(models,tmp_path,monkeypatch):
    folder, *_ = collect(tmp_path,models,monkeypatch)
    config = read(folder/'configuration.json')
    config['limits']['requests']=4
    with pytest.raises(ValueError,match='budget or sampler'):
        verify_context_confirmation_run(folder,config)
    config = read(folder/'configuration.json')
    (folder/'context-value-model.json').write_bytes((folder/'context-value-model.json').read_bytes()+b' ')
    with pytest.raises(ValueError,match='models|changed'):
        verify_context_confirmation_run(folder,config)


def test_confirmation_inventory_prevents_later_A_or_B_reuse(models,tmp_path):
    from football_research.data_adapters.prospective_learning import _previous_inventory, prepare_learning_plan
    from football_research.acquisition.context_gain import prepare_context_calibration
    inputs, _, previous, cpath, fpath, _ = setup(tmp_path,models)
    plans = [read(p) for p in previous]+[read(cpath)]
    at = AT+timedelta(milliseconds=300)
    rows = _previous_inventory(plans,at)
    assert next(r for r in rows if r['source_event_id']=='uefa:match:2050993')['purpose']=='confirmation'
    bundle = {'bindings':{inputs['binding']['match_id']:inputs['binding']}}
    with pytest.raises(ValueError,match='cohort purpose'):
        prepare_learning_plan(inputs['plan'],identity_bundle=bundle,purpose='f_train',created_at=at,previous_plans=plans)
    with pytest.raises(ValueError,match='cohort purpose'):
        prepare_context_calibration(inputs['plan'],identity_bundle=bundle,previous_plans=plans,
            predictor_path=fpath,output=tmp_path/'illegal-B.json',created_at=at)
