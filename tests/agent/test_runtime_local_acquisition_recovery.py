"""Supplemental durable boundaries; fake providers and unchanged raw fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
from types import FunctionType

from jsonschema import validate
import pytest

from football_agent import experience
from football_agent import match_return_assessment as review
from football_agent import runtime_exploration as exploration
from football_agent import runtime_source_claims as claims
from football_agent.acquisition_replay import verify_run

E = runpy.run_path(str(Path(__file__).with_name('test_runtime_local_acquisition_entry.py')))


def _raw_hashes(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (folder/'archive').rglob('*.body.gz')}


def test_failed_local_read_keeps_spent_budget_and_recovery_deduplicates_changed_limit(tmp_path, monkeypatch):
    service, run, calls, seen = E['fixture_runtime'](tmp_path, http_calls=1, page_limit=2,
        options={'max_actions': 2, 'max_declared_cost': .2})
    resumed = None
    try:
        for _ in range(5): service.step(run)
        queued = service.get(run)['queued_action']
        assert queued['local_query']['record_kind'] == 'odds'
        assert queued['local_query']['offset'] == 0 and queued['local_query']['limit'] == 2
        folder = service.folder(run)
        before = _raw_hashes(folder)

        def fail_query(*args, **kwargs):
            raise OSError('SYNTHETIC_TRANSIENT_LOCAL_IO')

        with monkeypatch.context() as transient:
            transient.setattr(experience, 'project_query', fail_query)
            failed = service.step(run)
        assert failed['status'] == 'paused' and failed['pending'] is None
        assert failed['local_usage']['reads'] == 2 and len(calls) == 1
        assert len(failed['documents']) == 2
        failed_audit = verify_run(folder)
        assert failed_audit['failed_local_reads'] == 1
        assert failed_audit['local_documents_reconstructed'] == 1
        assert failed_audit['bounded_exploration']['reserved_actions'] == 1
        assert failed_audit['bounded_exploration']['reserved_declared_cost'] == .05
        assert _raw_hashes(folder) == before

        original_factory = service.proposer_factory
        class ChangedPageSize:
            def __init__(self, inner): self.inner = inner
            def propose(self, public):
                result = self.inner.propose(public)
                for candidate in result['candidates']:
                    if candidate['action'] == 'query_acquired':
                        candidate['local_query']['limit'] = 3
                return result

        service.close()
        resumed = experience.ExperienceService(tmp_path, mode='test_fixture',
            selection_policy=service.selection_policy, enable_return_assessment=True,
            tools_factory=service.tools_factory,
            proposer_factory=lambda folder, effort: ChangedPageSize(original_factory(folder, effort)))
        assert resumed.recover() == []  # The failed receipt is complete, not pending.
        resumed._drive(run, True)
        state = resumed.get(run)
        assert state['last_error'] is None and state['turns'][-1]['action'] == 'defer'
        assert state['local_usage']['reads'] == 2 and len(calls) == 1
        assert state['spent'] == {'http_calls': 1, 'llm_calls': 4}
        models = [op for op in state['operations'] if op['kind'] == 'model']
        proposal = json.loads((folder/'operations'/models[-1]['operation_id']/'proposal.json').read_text(encoding='utf-8'))
        retry = next(row['response'] for row in proposal['candidates'] if row['response']['action'] == 'query_acquired')
        assert retry['local_query']['offset'] == 0 and retry['local_query']['limit'] == 3
        selection = (folder/'operations'/models[-1]['operation_id']/'policy-output.json').read_text(encoding='utf-8')
        assert 'normalized_action_already_accepted' in selection
        report = verify_run(folder)
        assert report['linked_local_reads'] == 2 and report['failed_local_reads'] == 1
        assert report['bounded_exploration']['reserved_actions'] == 1
        assert report['bounded_exploration']['reserved_declared_cost'] == .05
        assert report['bounded_exploration']['remaining_actions'] == 1
        assert report['bounded_exploration']['remaining_declared_cost'] == .15
        assert _raw_hashes(folder) == before
    finally:
        if resumed is not None: resumed.close()
        service.close()


@pytest.mark.parametrize('return_assessment', [False, True])
def test_default_single_candidate_provider_receives_local_protocol_and_replays(tmp_path, monkeypatch, return_assessment):
    captured, underlying = [], {}

    class FakeExperienceProvider:
        def __init__(self, workspace, *, protocol, effort, timeout_seconds):
            self.protocol = protocol
            self.proposer = underlying['factory'](workspace, effort)
            captured.append({'version': protocol.version, 'schema': deepcopy(protocol.schema),
                             'prompt': protocol.prompt, 'timeout_seconds': timeout_seconds})

        def decide(self, public):
            result = self.proposer.propose(public)
            response = next((row for row in result['candidates'] if row['action'] == 'query_acquired'),
                            result['candidates'][0])
            validate(response, self.protocol.schema)
            return {'response': response, 'test_double': True}

    def single_service(root, **kwargs):
        underlying['factory'] = kwargs.pop('proposer_factory')
        kwargs['enable_return_assessment'] = return_assessment
        return experience.ExperienceService(root, **kwargs)

    def response(public, action):
        value = E['H']['match_decision'](public, action)
        value[claims.FIELD] = []
        value['local_query'] = None
        if return_assessment:
            value[review.FIELD] = E['R']['actual_review'](public)
            if value[review.FIELD]:
                for citation in value[review.FIELD]['citations']:
                    citation['quote'] = citation['quote'][:400]
        return value

    original = E['fixture_runtime']
    namespace = dict(original.__globals__, ExperienceService=single_service, _response=response)
    fixture = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    fixture.__kwdefaults__ = original.__kwdefaults__
    monkeypatch.setattr(experience, 'ExperienceProvider', FakeExperienceProvider)
    service, run, calls, seen = fixture(tmp_path, http_calls=1)
    try:
        assert service.provider_factory is None and service.proposer_factory is None
        service._drive(run, True)
        state = service.get(run)
        assert state['last_error'] is None
        assert state['config']['proposal_mode'] == 'legacy_single_candidate'
        assert [turn['action'] for turn in state['turns']] == ['read_url', 'query_acquired', 'query_acquired', 'defer']
        assert state['local_usage']['reads'] == 2 and len(calls) == 1
        assert state['spent'] == {'http_calls': 1, 'llm_calls': 4}
        assert len(captured) == len(seen) == 4
        assert state['user_context'] == [] and state['question_to_user'] is None
        for protocol in captured:
            assert protocol['version'].endswith('-resolved-local-v1')
            assert 'query_acquired' in protocol['schema']['properties']['action']['enum']
            assert 'local_query' in protocol['schema']['properties']
            assert 'OPTIONAL RESOLVED-MATCH LOCAL INFORMATION ACQUISITION' in protocol['prompt']
            assert 'Never manufacture a need for odds' in protocol['prompt']
            assert 'No read has proven positive value' in protocol['prompt']
            assert (review.FIELD in protocol['schema']['properties']) is return_assessment
        report = verify_run(service.folder(run), include_action_value_ledger=True)
        assert report['accepted_decisions_verified'] == 4
        assert report['linked_local_reads'] == report['local_documents_reconstructed'] == 2
        assert report['bounded_exploration']['reserved_actions'] == 1
        assert report['bounded_exploration']['reserved_declared_cost'] == .05
        assert not report['numeric_voi_calibrated'] and report['real_calls_made_by_replay'] == 0
        selected = [entry for entry in report['action_value_ledger']['entries']
                    if entry['selection_basis'] == exploration.BASIS]
        assert len(selected) == 1 and len(selected[0]['candidate_value_assessments']) == 1
    finally: service.close()
