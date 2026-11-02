"""Real fitted g through policy/runtime, using simulated February 2027 sources.

The .13 loss conversion and zero-cost runtime conversion are explicit mechanism
boundaries. They are not fitted football utility weights or efficacy estimates.
No prediction method is stubbed and every transport is offline constructed.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import socket

import pytest

from football_agent.context_voi_policy import ContextVOIPolicy
from football_agent.learned_voi_replay import replay_saved_run
from football_agent.learned_voi_web import _web_action
from football_research.acquisition.context_gain import _query_input
from football_research.acquisition.visible_state import encode_visible_state
from football_research.data_adapters.prospective_text import MarketPriorPolicy, _save
from football_research.data_adapters.verified_fixtures import verify_fixture_identity
from test_context_value import fitted, CONTEXTS
from test_prospective_training_integration import _source_inputs
from test_verified_collection import _collect


AT = datetime(2027, 2, 10, 10, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Constructed g-policy integration must use zero network')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    monkeypatch.setattr(socket, 'getaddrinfo', blocked)


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    return fitted(tmp_path_factory.mktemp('fitted-context-g-policy'))


def inputs_with_loss_conversion(root, money):
    inputs = _source_inputs(root, at=AT, local_id=101, event_id=2050993)
    saved = inputs['plan']['selected'][0]['market_initialization']
    base = MarketPriorPolicy(saved['fixture'], saved['quote'], frozen_at=saved['frozen_at'],
        prediction_cutoff=saved['prediction_cutoff'], lambda_cost={'money': money})
    inputs['plan']['selected'] = [base.manifest]
    inputs['binding'] = verify_fixture_identity(base.manifest, archive_root=root/'identity',
        observation_ids=[r['observation_id'] for r in inputs['binding']['observations']],
        aliases=inputs['binding']['aliases'], as_of=AT)
    assert inputs['binding']['status'] == 'CONFIRMED'
    _save(root/'plan.json', inputs['plan'])
    _save(root/'bindings.json', {'bindings': {inputs['binding']['match_id']: inputs['binding']}})
    return inputs, base


def policy_public(models, tmp_path, *, context='positive', money=.13, synthetic_prior=None):
    f, fpath, g = models
    inputs, base = inputs_with_loss_conversion(tmp_path/'source', money)
    policy = ContextVOIPolicy(base, f, physical_match_id='uefa:match:2050993',
        value_model=g, predictor_artifact=fpath.read_bytes(), synthetic_prior=synthetic_prior)
    text = CONTEXTS[context]
    doc = {'document_id': 'known-context', 'raw_sha256': 'a'*64,
        'url': 'https://www.uefa.com/news/already-read/', 'status': 'OK',
        'kind': 'article', 'text': text, 'links': [], 'source_published_at': None,
        'observed_at': AT.isoformat(), 'available_at': AT.isoformat(),
        'structured_news': {'version': 'structured-news-discovery-v3', 'role': 'article',
            'segments': [{'text': text, 'start': 0, 'end': len(text),
                'content_source': 'json_ld_article_body', 'json_pointer': 'script[0].articleBody'}]}}
    public = {'state': policy.public_state(base.model.initial), 'documents': [doc], 'attempts': [],
        'decision_at': (AT+timedelta(seconds=1)).isoformat(),
        'budget': {'limits': {'money': 1., 'requests': 10.}, 'spent': {}},
        'prediction': base.model.predict_partial(base.model.initial)}
    public['prediction'] = policy.predict_public(public)
    action = _web_action('read_url', 'https://www.uefa.com/news/actual/', cost={'money': .01, 'requests': 1.})
    return policy, public, action


def test_fitted_g_values_known_context_and_deducts_cost_once(models, tmp_path):
    observed = {}
    for context in ('positive', 'negative'):
        policy, public, action = policy_public(models, tmp_path/context, context=context)
        decision = policy.plan_public(public, [action], public['budget'])
        value = decision['candidates'][0]['value']
        direct = models[2].predict(_query_input(encode_visible_state(public, [action]), action,
            public['prediction']['probabilities']), physical_match_id=policy.physical_match_id,
            predictor_sha256=policy.predictor_sha256)
        assert value['status'] == 'COMPUTED'
        assert value['expected_signed_gain'] == direct['expected_signed_gain']
        assert value['cost'] == pytest.approx(.0013)
        assert value['net_voi'] == pytest.approx(direct['expected_signed_gain']-.0013)
        assert value['estimate']['cost_deducted'] is False
        assert value['gross_evsi'] is None
        assert value['value_model_version'] == models[2].model_version
        observed[context] = decision
    assert observed['positive']['candidates'][0]['value']['net_voi'] > 0
    assert observed['positive']['selection']['decision'] == 'acquire'
    assert observed['negative']['candidates'][0]['value']['net_voi'] < 0
    assert observed['negative']['selection']['decision'] == 'defer'
    assert observed['negative']['selection']['reason'] == 'candidate_generation_incomplete'


def test_known_nonpositive_complete_menu_stops_without_certificate(models, tmp_path):
    policy, public, action = policy_public(models, tmp_path, context='negative')
    decision = policy.plan_public(public, [action], public['budget'], candidate_generation_complete=True)
    assert decision['candidates'][0]['value']['status'] == 'COMPUTED'
    assert decision['candidates'][0]['value']['net_voi'] < 0
    assert decision['selection'] == {'decision': 'stop',
        'reason': 'declared_menu_nonpositive_empirical_mean', 'candidate_id': None, 'stop_certificate': None}


def test_unseen_query_stays_unknown_none_and_never_claims_mean_stop(models, tmp_path):
    policy, public, _ = policy_public(models, tmp_path)
    action = _web_action('read_url', 'https://unseen.example/article/', cost={'money': .01, 'requests': 1.})
    decision = policy.plan_public(public, [action], public['budget'], candidate_generation_complete=True)
    value = decision['candidates'][0]['value']
    assert value['status'] == 'INELIGIBLE'
    assert value['expected_signed_gain'] is value['net_voi'] is None
    assert decision['selection']['decision'] == 'defer'
    assert decision['selection']['reason'] == 'context_gain_support_unknown'
    assert decision['selection']['stop_certificate'] is None


def test_synthetic_prior_is_attached_without_overriding_unknown_gate(models, tmp_path):
    from football_agent.g_teacher_prior import SyntheticGTeacherPriorAdapter
    adapter = SyntheticGTeacherPriorAdapter.from_path(
        'var/submission/p9-g-training-bridge-20261010/g-training-bridge.json')
    policy, public, action = policy_public(models, tmp_path, synthetic_prior=adapter)
    action = _web_action('read_url', 'https://unseen.example/lineup/', cost={'money': .01, 'requests': 1.})
    action['synthetic_prior_features'] = {
        'information_kind': 'lineup_injury', 'uncertainty': .864,
        'decision_sensitivity': .9, 'source_reliability': .82,
        'specificity': .88, 'redundancy': .1, 'misdirection_risk': .2,
        'query_cost': .024}
    decision = policy.plan_public(public, [action], public['budget'], candidate_generation_complete=True)
    value = decision['candidates'][0]['value']
    assert value['status'] == 'INELIGIBLE'
    assert value['expected_signed_gain'] is value['net_voi'] is None
    prior = value['synthetic_prior']
    assert prior['synthetic_prior']['prior_net_value'] == pytest.approx(.0635774320054664)
    assert prior['effective']['support_status'] == 'UNKNOWN'
    assert prior['effective']['selection_eligible'] is False
    assert decision['selection']['decision'] == 'defer'
    assert decision['selection']['reason'] == 'context_gain_support_unknown'


def test_synthetic_teaching_prior_is_attached_without_overriding_unknown_gate(models, tmp_path):
    from football_agent.synthetic_g_teaching_prior import SyntheticGTeachingPriorAdapter
    adapter = SyntheticGTeachingPriorAdapter.from_path(
        'var/submission/p9-g-synthetic-teaching-20261010/model.json')
    policy, public, action = policy_public(models, tmp_path, synthetic_prior=adapter)
    action = _web_action('read_url', 'https://unseen.example/lineup/', cost={'money': .01, 'requests': 1.})
    action['synthetic_prior_features'] = {
        'schema': 'footballmind-g-teaching-features-v1',
        'action': 'lineup', 'scenario': 'rotation_sensitive',
        'entropy': .8974526644565144, 'max_probability': .534700471737682,
        'time_to_kickoff_hours': 24., 'source_reliability': .94,
        'event_relevance': .96, 'tactical_sensitivity': .5, 'cost_units': .018,
    }
    decision = policy.plan_public(public, [action], public['budget'], candidate_generation_complete=True)
    value = decision['candidates'][0]['value']
    assert value['status'] == 'INELIGIBLE'
    assert value['expected_signed_gain'] is value['net_voi'] is None
    assert value['synthetic_prior']['synthetic_prior']['support_status'] == 'synthetic_teaching_only'
    assert value['synthetic_prior']['synthetic_prior']['prior_net_value'] == pytest.approx(.038063264785723375)
    assert value['synthetic_prior']['effective']['support_status'] == 'UNKNOWN'
    assert decision['selection']['decision'] == 'defer'
    assert decision['selection']['reason'] == 'context_gain_support_unknown'


def test_actual_unknown_then_fitted_g_selection_snapshots_and_replays(models, tmp_path, monkeypatch):
    import football_agent.inquiry_tools as web_tools
    import football_agent.learned_voi_runtime as runtime
    import football_research.data_adapters.prospective_text as collector
    import test_verified_collection as fixtures
    f, fpath, g = models
    inputs, _ = inputs_with_loss_conversion(tmp_path/'source', 0.)
    inputs['body'] = inputs['body'].replace(
        b'Croatia may rotate their starting lineup against Spain. Spain will assess player fitness before the match.',
        CONTEXTS['positive'].encode())
    tick = [AT+timedelta(milliseconds=200)]
    class SimulatedFutureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            tick[0] += timedelta(milliseconds=10)
            return tick[0].astimezone(tz) if tz else tick[0].replace(tzinfo=None)
    original_init = runtime.LearnedVOIRuntime.__init__
    def with_constructed_g(self, policy, **kwargs):
        assert not isinstance(policy, ContextVOIPolicy)
        policy = ContextVOIPolicy(policy, f, physical_match_id='uefa:match:2050993',
            value_model=g, predictor_artifact=fpath.read_bytes())
        # Two unknown attempts cover either search/read ordering without choosing
        # a sampler seed based on the realized response or the model estimate.
        kwargs.update(max_exploration_actions=2)
        original_init(self, policy, **kwargs)
    original_collect = fixtures.collect_prepared_batch
    def with_three_attempts(*args, **kwargs):
        kwargs.update(requests_per_match=3,
            seed_urls=['https://www.uefa.com/news/unread-follow-up/'])
        return original_collect(*args, **kwargs)
    with monkeypatch.context() as mp:
        for module in (web_tools, runtime, collector, fixtures):
            mp.setattr(module, 'datetime', SimulatedFutureClock)
        mp.setattr(runtime.LearnedVOIRuntime, '__init__', with_constructed_g)
        mp.setattr(fixtures, 'collect_prepared_batch', with_three_attempts)
        folder, calls, report = _collect(inputs, mp)
    config = json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
    run = json.loads((folder/'run.json').read_text(encoding='utf-8'))
    chosen = [c for c in run['cycles'] if c['selected_action']]
    assert len(chosen) == len(calls) == report['http_requests'] == 3, [c['strict_decision'] for c in run['cycles']]
    assert chosen[0]['selection_basis'] == 'bounded_unknown_information_exploration'
    assert any(c['selection_basis'] == 'computed_positive_empirical_gain' for c in chosen[1:])
    assert 1 <= run['exploration_actions'] <= 2
    assert any(c['update']['probability_changed'] for c in chosen)
    for cycle in chosen:
        if cycle['selection_basis'] != 'computed_positive_empirical_gain':
            continue
        action = cycle['selected_action']
        public = cycle['public_before']
        value = next(c['value'] for c in cycle['strict_decision']['candidates']
            if c['action_id'] == action['action_id'])
        assert public['documents']
        direct = g.predict(_query_input(encode_visible_state(public, cycle['generated_actions']),
            action, public['prediction']['probabilities']), physical_match_id='uefa:match:2050993',
            predictor_sha256=sha256(fpath.read_bytes()).hexdigest())
        assert value['expected_signed_gain'] == value['net_voi'] == direct['expected_signed_gain']
        assert value['net_voi'] > 0 and value['cost'] == 0
        assert cycle['exploration_record'] is None
    assert config['schema'] == 'learned-voi-open-execution-v4'
    assert 'context_calibration' not in config['supervisor_metadata']
    assert (folder/'context-predictor.json').read_bytes() == fpath.read_bytes()
    assert config['context_value_model']['sha256'] == sha256((folder/'context-value-model.json').read_bytes()).hexdigest()
    assert config['context_value_model']['model_version'] == g.model_version
    assert run['context_response_learning']['value_model_version'] == g.model_version
    assert run['context_response_learning']['evidence_origin'] == 'constructed_mechanism'
    assert run['context_response_learning']['real_world_benefit_proven'] is False
    assert datetime.fromisoformat(g.metadata['fit_execution_at']) < datetime.fromisoformat(config['created_at'])
    assert config['created_at'].startswith('2027-02-10')
    replay = replay_saved_run(folder, output=tmp_path/'replay', project_root=tmp_path)
    assert replay['identical_replay'] and replay['replay_network_requests'] == 0
