"""Constructed optional-prior runtime contracts on a simulated February 2027 clock.

Models are genuinely fitted from explicitly constructed fixtures. Cost weights
are test mechanism boundaries, not estimated football utility or effect claims.
"""
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json
import socket

import pytest

from football_agent.official_context_policy import OfficialContextVOIPolicy, VERSION
from football_agent.learned_voi_runtime import LearnedVOIRuntime, OFFICIAL_CONTEXT_VERSION
from football_agent.learned_voi_web import MatchDocumentCandidates, _web_action
from football_research.acquisition.visible_state import encode_visible_state
from football_research.acquisition.optional_context_query import query_input
from football_spider.storage.raw_archive import digest as identity_digest
from test_evidence_only_runtime import AT, identity, receipt
from test_context_value import CONTEXTS
from test_optional_context_value import fitted
from test_optional_prior_predictor import fit, rows


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail('Constructed official runtime tests cannot use the network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(socket, 'getaddrinfo', denied)


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    return fitted(tmp_path_factory.mktemp('constructed-official-f-g'))


def policy(models, *, use_g=True, money=0., binding=None):
    f, path, g = models
    return OfficialContextVOIPolicy(binding or identity(), f, value_model=g if use_g else None,
        predictor_artifact=path.read_bytes(), lambda_cost={'money': money})


def document(*, text=None, seconds=2, link=True):
    text = text or CONTEXTS['positive']
    return {'document_id': 'constructed-read-body', 'raw_sha256': 'a'*64,
        'url': 'https://www.uefa.com/news/already-read/', 'status': 'OK', 'kind': 'article',
        'text': text, 'links': [{'url': 'https://www.uefa.com/news/actual/', 'title': 'HJK observed follow-up'}] if link else [],
        'observed_at': (AT+timedelta(seconds=seconds)).isoformat(),
        'available_at': (AT+timedelta(seconds=seconds)).isoformat(), 'source_published_at': None,
        'structured_news': {'version': 'structured-news-discovery-v3', 'role': 'article',
            'segments': [{'text': text, 'start': 0, 'end': len(text),
                'content_source': 'json_ld_article_body', 'json_pointer': 'script[0].articleBody'}]}}


def public(model, *, docs=None):
    result = {'state': model.public_state(model.initial_state), 'documents': docs or [], 'attempts': [],
        'decision_at': (AT+timedelta(seconds=3)).isoformat(),
        'prediction': model.model.predict_partial(model.initial_state),
        'budget': {'limits': {'money': 1., 'requests': 10.}, 'spent': {}}}
    result['prediction'] = model.predict_public(result)
    return result


def execute_run(tmp_path, model, *, execute=None, times=None, allowance=2, caps=None, exploration=None, seed_urls=None):
    generator = MatchDocumentCandidates(match_terms=['HJK', 'VPS'], seed_urls=seed_urls or [],
        allowed_hosts=['www.uefa.com'], numeric_actions=[], web_cost={'money': .01, 'requests': 1})
    seen = []
    def observed(action, before):
        seen.append((deepcopy(action), deepcopy(before)))
        return execute(action, before) if execute else receipt(doc=document(seconds=len(seen)*2))
    ticks = iter(times or [AT+timedelta(seconds=i) for i in range(30)])
    caps = caps or {'money': .02, 'requests': 2}
    runner = LearnedVOIRuntime(model, seed=201, exploration_limits=exploration or caps,
        max_exploration_actions=allowance, max_steps=8)
    result = runner.run(model.initial_state, run_id='constructed-official-context', limits=caps,
        propose=generator, execute=observed, output=tmp_path/'run', provider_version='constructed-transport',
        generator_version=generator.version, clock=lambda: next(ticks), evidence_match=model.evidence_match,
        evidence_version='match-evidence-extraction-v5')
    return result, seen


def test_actual_missing_prior_forecasts_and_fitted_gain_are_distinct_from_prior(models):
    model = policy(models)
    known = public(model, docs=[document(link=False)])
    empty = public(model)
    assert empty['prediction']['probabilities'] == [7/12, 3/12, 2/12]
    assert known['prediction']['probabilities'] != empty['prediction']['probabilities']
    assert known['prediction']['prior_probabilities'] is None
    assert known['prediction']['base_probabilities'] == empty['prediction']['probabilities']
    visible = encode_visible_state(known)
    assert visible['before']['prior_probabilities'] is None
    assert 'open_text' not in known['state']
    action = _web_action('read_url', 'https://www.uefa.com/news/actual/', cost={'money': .01, 'requests': 1})
    decision = model.plan_public(known, [action], known['budget'])
    value = decision['candidates'][0]['value']
    expected = models[2].predict(query_input(encode_visible_state(known, [action]), action, known['prediction']),
        physical_match_id=model.physical_match_id, predictor_sha256=model.predictor_sha256)
    assert value['status'] == 'COMPUTED' and value['expected_signed_gain'] == expected['expected_signed_gain']
    assert value['net_voi'] == value['expected_signed_gain'] and value['cost'] == 0
    assert value['net_voi'] > 0 and decision['selection']['decision'] == 'acquire'


def test_cost_is_deducted_once_and_mean_stop_requires_complete_supported_menu(models):
    model = policy(models, money=100.)
    known = public(model, docs=[document(link=False)])
    action = _web_action('read_url', 'https://www.uefa.com/news/actual/', cost={'money': .01, 'requests': 1})
    decision = model.plan_public(known, [action], known['budget'])
    value = decision['candidates'][0]['value']
    assert value['status'] == 'COMPUTED' and value['cost'] == 1.
    assert value['net_voi'] == pytest.approx(value['expected_signed_gain']-1.)
    assert value['net_voi'] < 0 and decision['selection']['reason'] == 'candidate_generation_incomplete'
    complete = model.plan_public(known, [action], known['budget'], candidate_generation_complete=True)
    assert complete['selection']['decision'] == 'stop' and complete['selection']['stop_certificate'] is None
    action['url'] = 'https://unseen.example/unknown/'
    unknown = model.plan_public(known, [action], known['budget'], candidate_generation_complete=True)
    assert unknown['selection']['decision'] == 'defer'
    assert unknown['candidates'][0]['value']['net_voi'] is None


def test_unsupported_missing_domain_keeps_numeric_base_and_unknown_query_value():
    provided = fit(rows('provided', 'provided-A'))
    model = OfficialContextVOIPolicy(identity(), provided)
    before = public(model, docs=[document()])
    assert before['prediction']['prior_domain_context_supported'] is False
    assert before['prediction']['probabilities'] == before['prediction']['base_probabilities']
    action = _web_action('read_url', 'https://www.uefa.com/news/actual/', cost={'money': .01, 'requests': 1})
    decision = model.plan_public(before, [action], before['budget'], candidate_generation_complete=True)
    value = decision['candidates'][0]['value']
    assert value['expected_signed_gain'] is value['net_voi'] is None
    assert value['reason'] == 'missing_prior_context_domain_unsupported'
    assert decision['selection']['decision'] == 'defer'


def test_unknown_read_then_real_g_selection_updates_f_and_preserves_original_prior(models, tmp_path):
    # These initial observed request locations have no parent labels, matching
    # the actual constructed g support. Reading a labelled newly found link is
    # a different support mask and must correctly remain UNKNOWN.
    result, seen = execute_run(tmp_path, policy(models), allowance=1,
        seed_urls=['https://www.uefa.com/news/actual/', 'https://www.uefa.com/news/other/'],
        execute=lambda *_: receipt(doc=document(link=False)))
    assert result['schema'] == OFFICIAL_CONTEXT_VERSION and len(seen) == 2
    chosen = [c for c in result['cycles'] if c['selected_action']]
    assert [c['selection_basis'] for c in chosen] == [
        'bounded_unknown_information_exploration', 'computed_positive_empirical_gain']
    assert chosen[0]['prediction_after'] != chosen[0]['prediction_before']
    assert chosen[0]['public_after']['decision_at'] == chosen[0]['returned_at']
    assert chosen[1]['public_before']['documents'] == chosen[0]['public_after']['documents']
    assert result['initial_probabilities'] == [7/12, 3/12, 2/12]
    assert result['final_probabilities'] == result['final_prediction']['probabilities']
    assert result['prior_probabilities'] is None and result['base_probabilities'] == [7/12, 3/12, 2/12]
    assert result['exploration_actions'] == 1 and result['reserved_cost']['requests'] == 2
    for cycle in chosen:
        for snapshot in (cycle['public_before'], cycle['public_after']):
            assert snapshot['prediction']['prior_probabilities'] is None
            assert encode_visible_state(snapshot)['before']['prior_probabilities'] is None
    config = json.loads((tmp_path/'run/configuration.json').read_text(encoding='utf-8'))
    assert config['policy_manifest']['schema'] == VERSION
    assert (tmp_path/'run/context-predictor.json').read_bytes() == models[1].read_bytes()
    assert config['context_value_model']['sha256'] == sha256((tmp_path/'run/context-value-model.json').read_bytes()).hexdigest()
    assert config['context_value_model']['sha256'] == config['policy_manifest']['value_model_sha256']


def test_no_g_has_finite_unknown_exploration_without_invented_value(models, tmp_path):
    result, seen = execute_run(tmp_path, policy(models, use_g=False), allowance=1)
    assert len(seen) == result['exploration_actions'] == 1
    assert result['terminal_selection']['decision'] == 'defer'
    for cycle in result['cycles']:
        for candidate in cycle['strict_decision']['candidates']:
            assert candidate['value']['net_voi'] is None
    no_reads, seen = execute_run(tmp_path/'zero', policy(models, use_g=False), allowance=0)
    assert not seen and no_reads['initial_probabilities'] == no_reads['final_probabilities']


@pytest.mark.parametrize('case', ['failed', 'overrun', 'future_body', 'late_return', 'backwards_return'])
def test_actual_attempt_cost_and_time_boundaries_preserve_last_valid_forecast(models, tmp_path, case):
    times = None
    execute = lambda *_: receipt(status='FAILED')
    exploration = None
    if case == 'overrun':
        execute = lambda *_: receipt(status='FAILED', latency=3)
        exploration = {'money': .02, 'requests': 2, 'latency_ms': 1}
    elif case == 'future_body':
        execute = lambda *_: receipt(doc=document(seconds=3))
    elif case == 'late_return':
        times = [AT, AT+timedelta(hours=2, seconds=-1), AT+timedelta(hours=2, seconds=1), AT+timedelta(hours=2, seconds=2)]
        execute = lambda *_: receipt(doc=document())
    elif case == 'backwards_return':
        times = [AT, AT+timedelta(seconds=2), AT+timedelta(seconds=1), AT+timedelta(seconds=3)]
        execute = lambda *_: receipt(doc=document())
    result, seen = execute_run(tmp_path, policy(models), execute=execute, times=times, exploration=exploration)
    assert len(seen) == 1 and result['reserved_cost']['requests'] == 1
    assert result['documents'] == []
    assert result['initial_probabilities'] == result['final_probabilities'] == [7/12, 3/12, 2/12]
    if case == 'backwards_return':
        assert result['attempts'] == [] and result['final_state']['action_history'] == []
        assert result['terminal_selection']['reason'] == 'receipt_reconciliation_required'
    elif case == 'late_return':
        assert result['terminal_selection']['reason'] == 'forecast_cutoff_elapsed'
        assert result['cycles'][0]['admission']['reason'] == 'after_information_cutoff'
    elif case == 'overrun':
        assert result['terminal_selection']['reason'] == 'actual_resource_budget_overrun'
        assert result['exploration_reserved_cost']['latency_ms'] == 3
    else:
        assert len(result['attempts']) == 1
        assert result['cycles'][0]['public_after']['attempts'] == result['attempts']


def test_fitting_clock_and_heldout_fixture_and_prior_injection_are_rejected(models):
    early_identity = identity()
    early_identity['verified_at'] = '2027-01-01T00:00:00+00:00'
    early_identity['binding_sha256'] = identity_digest({k:v for k,v in early_identity.items() if k != 'binding_sha256'})
    model = policy(models, binding=early_identity)
    with pytest.raises(ValueError, match='fitting'):
        model.check_start(models[2].metadata['fit_execution_at'])
    bad = deepcopy(model.initial_state)
    bad['open_text'] = {'prior_probabilities': [1/3]*3}
    with pytest.raises(ValueError, match='prior'):
        model.public_state(bad)
    for physical in [models[0].metadata['training_physical_match_ids'][0], models[2].metadata['training_physical_match_ids'][0]]:
        binding = identity()
        binding['match_id'] = binding['canonical']['source_event_id'] = physical
        binding['binding_sha256'] = identity_digest({k:v for k,v in binding.items() if k != 'binding_sha256'})
        with pytest.raises(ValueError, match='fixture|independent'):
            policy(models, binding=binding)


def test_first_decision_expired_keeps_prestart_f_and_never_requests(models, tmp_path):
    result, seen = execute_run(tmp_path, policy(models), times=[AT,
        AT+timedelta(hours=2, seconds=1), AT+timedelta(hours=2, seconds=2)])
    assert not seen and result['terminal_selection']['reason'] == 'forecast_cutoff_elapsed'
    assert result['initial_probabilities'] == result['final_probabilities'] == [7/12, 3/12, 2/12]
    assert result['prior_probabilities'] is None
