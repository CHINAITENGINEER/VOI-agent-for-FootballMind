"""Prior-unavailable execution with constructed receipts and no network."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json

import pytest

from football_agent.evidence_only_policy import EvidenceOnlyPolicy, VERSION
from football_agent.learned_voi_runtime import LearnedVOIRuntime, EVIDENCE_ONLY_VERSION
from football_agent.learned_voi_web import MatchDocumentCandidates, _web_action
from football_research.acquisition.value import _canonical


AT = datetime(2027, 2, 10, 10, tzinfo=timezone.utc)
ARTICLE = 'https://www.veikkausliiga.com/news/HJK-observed-match-preview/'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def blocked(*args, **kwargs):
        pytest.fail('Prior-unavailable runtime tests cannot use the network')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    monkeypatch.setattr(socket, 'getaddrinfo', blocked)


def identity():
    value = {'schema': 'official-fixture-evidence-identity-v1', 'status': 'CONFIRMED',
        'match_id': 'veikkausliiga:2027:match:123',
        'canonical': {'source_event_id': 'veikkausliiga:2027:match:123', 'home_id': 'native:HJK',
            'away_id': 'native:VPS', 'home_name': 'HJK', 'away_name': 'VPS',
            'competition': 'Veikkausliiga', 'category': 'senior_men_club', 'season': '2027',
            'stage': 'league', 'kickoff': (AT+timedelta(hours=3)).isoformat(),
            'source_url': 'https://www.veikkausliiga.com/fixtures/123/',
            'score_period': 'unknown', 'settlement_rule': 'unknown'},
        'aliases': {'home': ['HJK'], 'away': ['VPS']}, 'observations': [],
        'verified_at': (AT-timedelta(seconds=1)).isoformat(),
        'prediction_cutoff': (AT+timedelta(hours=2)).isoformat(),
        'observed_generic_entry_urls': ['https://www.veikkausliiga.com/']}
    value['binding_sha256'] = sha256(_canonical(value).encode()).hexdigest()
    return value


def document(*, kind='index', seconds=2):
    text = '' if kind == 'index' else 'HJK may rotate their starting lineup against VPS.'
    return {'document_id': 'constructed-'+kind, 'status': 'OK', 'kind': kind,
        'url': 'https://www.veikkausliiga.com/' if kind == 'index' else ARTICLE,
        'raw_sha256': 'a'*64, 'text': text,
        'observed_at': (AT+timedelta(seconds=seconds)).isoformat(),
        'available_at': (AT+timedelta(seconds=seconds)).isoformat(), 'source_published_at': None,
        'links': [{'url': ARTICLE, 'title': 'HJK match preview'}] if kind == 'index' else [],
        'structured_news': {'version': 'structured-news-discovery-v3', 'role': kind,
            'segments': [] if kind == 'index' else [{'text': text, 'start': 0, 'end': len(text),
                'content_source': 'json_ld_article_body', 'json_pointer': 'script[0].articleBody'}]}}


def receipt(*, status='OK', doc=None, requests=1, latency=0):
    return {'status': status, 'document': doc,
        'actual_cost': {'money': None, 'requests': requests, 'latency_ms': latency, 'tokens': 0},
        'receipt': {'hidden_outcome': 'SUPERVISOR_ONLY'}}


def execute_run(tmp_path, *, propose=None, execute=None, times=None, allowance=4,
                limits=None, exploration_limits=None, max_steps=8):
    policy = EvidenceOnlyPolicy(identity())
    generator = MatchDocumentCandidates(match_terms=['HJK', 'VPS'], seed_urls=[],
        allowed_hosts=['www.veikkausliiga.com'], numeric_actions=[], web_cost={'money': .01, 'requests': 1})
    seen = []
    def observed(action, public):
        seen.append((deepcopy(action), deepcopy(public)))
        if execute:
            return execute(action, public)
        return receipt(doc=document(kind='index' if len(seen) == 1 else 'article', seconds=len(seen)*2))
    clock = iter(times or [AT+timedelta(seconds=i) for i in range(30)])
    caps = limits or {'money': .04, 'requests': 4}
    runtime = LearnedVOIRuntime(policy, seed=180, exploration_limits=exploration_limits or caps,
        max_exploration_actions=allowance, max_steps=max_steps)
    result = runtime.run(policy.initial_state, run_id='evidence-only-constructed', limits=caps,
        propose=propose or generator, execute=observed, output=tmp_path/'run',
        provider_version='constructed-archived-transport', generator_version=generator.version,
        clock=lambda: next(clock), evidence_match=policy.evidence_match,
        evidence_version='match-evidence-extraction-v5')
    return result, seen, policy


def test_existing_discovery_reads_actual_article_without_any_probabilities(tmp_path):
    result, seen, policy = execute_run(tmp_path)
    assert result['schema'] == EVIDENCE_ONLY_VERSION
    assert result['analysis_mode'] == 'evidence_only' and result['prior_status'] == 'prior_unavailable'
    assert result['initial_probabilities'] is result['final_probabilities'] is None
    assert result['expected_gain'] is result['net_voi'] is result['numeric_confidence'] is None
    assert [a['tool'] for a, _ in seen] == ['search_web', 'read_url']
    assert seen[1][0]['url'] == ARTICLE
    assert seen[0][1]['documents'] == []
    assert seen[1][1]['documents'][0]['links'][0]['url'] == ARTICLE
    assert result['evidence']['evidence']
    assert 'SUPERVISOR_ONLY' not in json.dumps([p for _, p in seen])
    for cycle in result['cycles']:
        assert cycle['prediction_before'] is cycle['prediction_after'] is None
        assert cycle['public_before']['prediction']['probabilities'] is None
        for candidate in cycle['strict_decision']['candidates']:
            assert candidate['value']['expected_gain'] is candidate['value']['net_voi'] is None
        if cycle['selected_action']:
            assert cycle['selection_basis'] == 'bounded_unknown_information_exploration'
            assert cycle['public_after']['prediction']['probabilities'] is None
            assert cycle['public_after']['decision_at'] == cycle['returned_at']
            assert cycle['update']['probability_changed'] is False
    assert result['terminal_selection']['reason'] == 'defer_prior_unavailable'
    config = json.loads((tmp_path/'run/configuration.json').read_text(encoding='utf-8'))
    assert config['policy_manifest']['schema'] == VERSION
    assert EvidenceOnlyPolicy.from_manifest(config['policy_manifest']).manifest == policy.manifest
    assert not list((tmp_path/'run').glob('*model.json'))
    from football_agent.match_analysis_report import build_analysis_report
    report = build_analysis_report(result, match=policy.evidence_match)
    assert report['status'] == 'completed_with_gaps'
    assert report['probabilities']['initial'] is report['probabilities']['final'] is None
    assert '缺少可用市场先验' in report['analysis_note_zh']


@pytest.mark.parametrize('status', ['FAILED', 'NO_DATA'])
def test_failure_attempts_and_request_costs_survive_without_numeric_updates(tmp_path, status):
    result, seen, _ = execute_run(tmp_path, execute=lambda *_: receipt(status=status))
    assert len(seen) == 1 and result['reserved_cost']['requests'] == 1
    assert result['attempts'][0]['status'] == status and result['documents'] == []
    post = result['cycles'][0]['public_after']
    assert post['attempts'] == result['attempts'] and post['prediction']['probabilities'] is None
    assert post['budget']['spent']['requests'] == 1


def many_actions(_):
    return [_web_action('read_url', f'https://www.veikkausliiga.com/news/observed-{i}/',
        cost={'money': .01, 'requests': 1}) for i in range(6)]


def test_four_request_budget_and_separate_exploration_allowance_are_finite(tmp_path):
    result, seen, _ = execute_run(tmp_path, propose=many_actions, execute=lambda *_: receipt(status='FAILED'))
    assert len(seen) == result['exploration_actions'] == result['reserved_cost']['requests'] == 4
    assert result['terminal_selection']['reason'] == 'physical_budget_exhausted'
    assert result['terminal_selection']['stop_certificate'] is None
    no_budget, seen, _ = execute_run(tmp_path/'zero', allowance=0)
    assert not seen and no_budget['terminal_selection']['reason'] == 'defer_prior_unavailable'


def test_unmeasured_execution_keeps_reservation_and_never_retries(tmp_path):
    def fail(*args):
        raise OSError('Constructed uncertain transport result')
    result, seen, _ = execute_run(tmp_path, execute=fail)
    assert len(seen) == 1 and result['reserved_cost']['requests'] == 1
    assert result['terminal_selection']['reason'] == 'execution_reconciliation_required'
    assert result['initial_probabilities'] is result['final_probabilities'] is None


def test_actual_exploration_only_overrun_retains_honest_post_and_stops(tmp_path):
    result, seen, _ = execute_run(tmp_path, propose=many_actions,
        execute=lambda *_: receipt(status='FAILED', latency=3),
        exploration_limits={'requests': 4, 'money': .04, 'latency_ms': 1})
    assert len(seen) == 1 and result['exploration_reserved_cost']['latency_ms'] == 3
    assert result['terminal_selection']['reason'] == 'actual_resource_budget_overrun'
    assert result['cycles'][0]['public_after']['budget']['spent']['latency_ms'] == 3
    assert result['cycles'][0]['public_after']['prediction']['probabilities'] is None


@pytest.mark.parametrize('case', ['future_body', 'late_return', 'return_clock_backwards'])
def test_evidence_availability_and_actual_return_boundaries(tmp_path, case):
    times = None
    doc = document(seconds=3 if case == 'future_body' else 2)
    if case == 'late_return':
        cutoff = AT+timedelta(hours=2)
        times = [AT, cutoff-timedelta(seconds=1), cutoff+timedelta(seconds=1), cutoff+timedelta(seconds=2)]
    elif case == 'return_clock_backwards':
        times = [AT, AT+timedelta(seconds=2), AT+timedelta(seconds=1), AT+timedelta(seconds=3)]
    result, seen, _ = execute_run(tmp_path, execute=lambda *_: receipt(doc=doc), times=times)
    assert len(seen) == 1 and result['documents'] == []
    assert result['initial_probabilities'] is result['final_probabilities'] is None
    if case == 'future_body':
        assert result['cycles'][0]['admission']['reason'] == 'after_actual_return_clock'
    elif case == 'late_return':
        assert result['terminal_selection']['reason'] == 'forecast_cutoff_elapsed'
        assert result['cycles'][0]['public_after']['documents'] == []
    else:
        assert result['terminal_selection']['reason'] == 'receipt_reconciliation_required'


def test_state_cannot_admit_uniform_prior_or_numeric_acquisition(tmp_path):
    policy = EvidenceOnlyPolicy(identity())
    changed = deepcopy(policy.initial_state)
    changed['probabilities'] = [1/3]*3
    with pytest.raises(ValueError, match='prior'):
        policy.public_state(changed)
    with pytest.raises(ValueError, match='public web'):
        policy.plan(policy.initial_state, [{'action_id': 'elo', 'tool': 'numeric', 'group_ids': ['elo']}],
            {'limits': {'money': .04, 'requests': 4}, 'spent': {}})


@pytest.mark.parametrize('offsets,reason', [
    ([0, 1, 3, 2, 4, 5, 6], 'Decision clock cannot precede'),
    ([0, 1, 2, 3, 4, 5, 3], 'Finished clock cannot precede'),
])
def test_evidence_only_clocks_cannot_rewind_after_visible_returns(tmp_path, offsets, reason):
    with pytest.raises(ValueError, match=reason):
        execute_run(tmp_path, times=[AT+timedelta(seconds=i) for i in offsets])
