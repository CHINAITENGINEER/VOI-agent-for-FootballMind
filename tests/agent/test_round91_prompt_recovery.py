'''Constructed offline checks; no model calls or external sources.'''
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

import pytest

from football_agent.audit import atomic_write
from football_agent.inquiry_choice import InformationChoiceProtocol
from football_agent.inquiry_choice_prompt import ConciseChoiceProvider


@pytest.fixture
def repair(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('round91_fixture',
        root / 'var/parallel/agent/round91/run_compact_choice.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'HERE', tmp_path)
    monkeypatch.setattr(module, 'LIVE', tmp_path / 'live')
    monkeypatch.setattr(module, 'SOURCES', [])
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat()
    prior = {'task': {'task_id': 'constructed-recovery', 'objective': 'Fixture only.',
        'deadline': deadline, 'initial_facts': [],
        'seed_urls': [c['url'] for c in module.original.CATALOG]},
        'history': {'prior_cli_attempts': 4}, 'limits': {'max_requests': 3},
        'campaign_cli_attempt_cap': 9, 'campaign_source_request_cap': 3}
    parent = {'budget': {'llm_calls': 1, 'http_attempts': 0}}
    evidence = {'run_sha256': 'constructed-parent-hash'}
    monkeypatch.setattr(module, 'parent_record', lambda: (prior, parent, evidence))
    flow = module.workflow()
    def no_calls(*args, **kwargs):
        raise AssertionError('Offline recovery checks must not call a provider or source')
    monkeypatch.setattr(flow.provider, 'decide', no_calls)
    monkeypatch.setattr(flow.tools, 'execute', no_calls)
    return module, flow, prior, parent, evidence


def test_registration_reuses_identity_deadline_and_remaining_budget(repair):
    module, flow, prior, parent, _ = repair
    first = module.register(flow)
    assert module.register(flow) == first
    assert first['task']['deadline'] == prior['task']['deadline']
    assert first['limits'] == {'max_requests': 3, 'max_llm_calls': 4}
    assert first['prior_campaign_attempts'] + first['limits']['max_llm_calls'] == 9
    assert first['budget_refill'] is False
    assert len(list((module.LIVE / 'runs').glob('*/run.json'))) == 1
    assert flow.get(first['run_id'])['budget']['llm_calls'] == 0


def test_restart_after_created_run_recovers_without_duplicate_task(repair):
    module, flow, *_ = repair
    first = module.register(flow)
    interrupted = {**first, 'run_id': None}
    atomic_write(module.LIVE / 'registration.json', interrupted)
    recovered = module.register(flow)
    assert recovered['run_id'] == first['run_id']
    assert recovered['limits'] == first['limits']
    assert len(list((module.LIVE / 'runs').glob('*/run.json'))) == 1


def test_runtime_configuration_change_requires_explicit_new_registration(repair, monkeypatch):
    module, flow, *_ = repair
    first = module.register(flow)
    config = flow.configuration()
    monkeypatch.setattr(flow, 'configuration', lambda: {**config, 'unexpected_change': True})
    with pytest.raises(ValueError, match='configuration changed'):
        module.register(flow)
    assert module.read(module.LIVE / 'registration.json') == first


def test_changed_parent_failure_record_is_not_silently_forgotten(repair):
    module, flow, _, _, evidence = repair
    first = deepcopy(module.register(flow))
    evidence['run_sha256'] = 'different-fixture-parent'
    with pytest.raises(ValueError, match='failure record changed'):
        module.register(flow)
    assert module.read(module.LIVE / 'registration.json') == first


@pytest.mark.parametrize('kind', ['expired', 'spent'])
def test_expired_or_exhausted_campaign_gets_no_new_registration(repair, kind):
    module, flow, prior, parent, _ = repair
    if kind == 'expired':
        prior['task']['deadline'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    else:
        parent['budget']['llm_calls'] = 4
    with pytest.raises(ValueError):
        module.register(flow)
    assert not (module.LIVE / 'registration.json').exists()


def test_concise_provider_keeps_full_schema_and_unknown_evidence_is_rejected(tmp_path):
    provider = ConciseChoiceProvider(tmp_path)
    assert provider.protocol.schema is InformationChoiceProtocol.schema
    assert provider.configuration()['prompt_sha256']
    response = {'questions': [{'question_id': 'q', 'question': 'Fixture unknown',
        'decision_relevance': 'Locate match', 'answer_could_change': 'Next source',
        'based_on_documents': []}], 'claim_updates': [], 'candidates': [],
        'hypotheses': [], 'context_bindings': [],
        'decision': {'kind': 'defer', 'candidate_id': None, 'reason': 'Unresolved', 'revisit_at': None},
        'choice_review': {'alternatives': [], 'tradeoff': 'Preserve unknowns', 'missing_routes': 'Open'}}
    public = {'documents': [], 'questions': [], 'hypotheses': [], 'claims': [],
        'tools': {'allowed_hosts': ['example.com']}, 'task': {'seed_urls': []}}
    InformationChoiceProtocol().validate(response, public)
    false_claim = deepcopy(response)
    false_claim['claim_updates'] = [{'claim_id': 'false', 'question_id': 'q',
        'text': 'Unobserved claim', 'status': 'source_claim',
        'citations': [{'document_id': 'never-acquired', 'quote': 'Invented quotation'}],
        'evidence': {'kind': 'other', 'subject': 'Team', 'scope': 'general', 'scope_key': 'unknown'},
        'hypothesis_links': []}]
    with pytest.raises(ValueError, match='unavailable evidence'):
        InformationChoiceProtocol().validate(false_claim, public)


@pytest.fixture
def core(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('round91_core_fixture',
        root / 'var/parallel/agent/round91/run_core_loop.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'HERE', tmp_path)
    monkeypatch.setattr(module, 'LIVE', tmp_path / 'core-loop')
    monkeypatch.setattr(module, 'SOURCES', [])
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat()
    first = {'run_id': 'original', 'task': {'task_id': 'constructed-core',
        'objective': 'Fixture only', 'deadline': deadline, 'initial_facts': [],
        'seed_urls': [c['url'] for c in module.original.CATALOG]},
        'history': {'prior_cli_attempts': 4}, 'campaign_cli_attempt_cap': 9,
        'campaign_source_request_cap': 3}
    for folder, registration in [('live-choice', first), ('compact-choice', {'run_id': 'compact'})]:
        atomic_write(tmp_path / folder / 'registration.json', registration)
        atomic_write(tmp_path / folder / 'runs' / registration['run_id'] / 'run.json',
            {'reason': 'PROVIDER_FAILED_OR_RESPONSE_UNKNOWN', 'documents': [], 'receipts': [],
             'budget': {'llm_calls': 1}})
    flow = module.workflow()
    def no_calls(*args, **kwargs):
        raise AssertionError('Offline registration test made external call')
    monkeypatch.setattr(flow.provider, 'decide', no_calls)
    monkeypatch.setattr(flow.tools, 'execute', no_calls)
    return module, flow, first


def test_core_probe_retains_both_failed_charges_and_same_deadline(core):
    module, flow, first = core
    reg = module.register(flow)
    assert reg['prior_campaign_attempts'] == 6
    assert reg['limits']['max_llm_calls'] == 3
    assert reg['task']['deadline'] == first['task']['deadline']
    assert module.register(flow) == reg
    assert len(list((module.LIVE / 'runs').glob('*/run.json'))) == 1
    assert reg['budget_refill'] is False


def test_core_probe_recovers_created_run_without_refill(core):
    module, flow, _ = core
    reg = module.register(flow)
    atomic_write(module.LIVE / 'registration.json', {**reg, 'run_id': None})
    assert module.register(flow)['run_id'] == reg['run_id']
    assert len(list((module.LIVE / 'runs').glob('*/run.json'))) == 1


def test_core_probe_does_not_extend_expired_deadline(core):
    module, flow, first = core
    first['task']['deadline'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    atomic_write(module.HERE / 'live-choice/registration.json', first)
    with pytest.raises(ValueError, match='deadline expired'):
        module.register(flow)
    assert not (module.LIVE / 'registration.json').exists()
