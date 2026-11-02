"""Archived fixture discovery followed by empirical selection in one durable run.

All transport and calibration observations are synthetic. The fixture response
passes through the real archive, normalizer and ExperienceWebTools boundaries.
"""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import runpy

import pytest

from football_agent.acquisition_replay import verify_run
from football_agent.audit import atomic_write, stored_digest
from football_agent.experience import ExperienceService
from football_agent.query_value_contract import VALUE_UNMEASURED
from football_agent.structured_sources import registered_request


_ROOT = Path(__file__).parent
_runtime = runpy.run_path(str(_ROOT / 'test_state_bound_runtime_policy.py'))
_structured = runpy.run_path(str(_ROOT / 'test_structured_sources.py'))
package, policy, match_decision, actual_review, CountingTools = (
    _runtime[key] for key in ('package', 'policy', 'match_decision', 'actual_review', 'CountingTools'))
fixture_row, recent_payload, make_tools = (
    _structured[key] for key in ('fixture_row', 'recent_payload', 'make_tools'))

MATCH = '2026-10-03 构造甲队 vs 构造乙队'


def _row(**changes):
    return fixture_row(**{'matchTime': '2026-10-03T18:00:00', **changes})


def _configuration():
    return package((('a', 1, -.5), ('b', 1, .9), ('c', 2, -.5)))


def _review(public):
    result = actual_review(public)
    if result:
        for citation in result['citations']:
            citation['quote'] = citation['quote'][:400]
    return result


def _service(root, *, payload=None, config=None, limits=None, offer_second_catalogue=False):
    payload = recent_payload([_row()]) if payload is None else payload
    calls, seen, search = [], [], CountingTools()

    class Tools:
        def __init__(self, archive):
            self.archive = archive

        def execute(self, action, **kwargs):
            calls.append(deepcopy(action))
            if action['tool'] == 'read_url':
                assert registered_request(action['url'])['kind'] == 'recent'
                # DNS/session are injected; normalization and RawArchive are real.
                return make_tools(self.archive, [], {'recent': payload}).execute(action, **kwargs)
            return search.execute(action, **kwargs)

    class Proposer:
        def propose(self, public):
            seen.append(deepcopy(public))
            count = len(public['documents'])
            names = ('unrelated',) if count == 0 else ('a', 'b') if count == 1 else ('c',)
            responses = []
            for name in names:
                response = match_decision(public)
                response['query'] = public['match_input'] + ' source ' + name
                response['return_assessment'] = _review(public)
                responses.append(response)
            if count == 0 or count == 1 and offer_second_catalogue:
                source = match_decision(public, 'read_url')
                source['url'] = public['source_catalog']['entries'][0 if count == 0 else 1]['url']
                source['return_assessment'] = _review(public)
                responses.append(source)
            # These wrong world/identity guesses must not control the resolver.
            for response in responses:
                response['match_identity'] = {'status': 'identified', 'label': 'spftll:999999'}
            return {'candidates': responses, 'recommended_index': 0, 'test_double': True}

    service = ExperienceService(root, selection_policy=policy(config or _configuration()),
        proposer_factory=lambda folder, effort: Proposer(), tools_factory=lambda folder: Tools(folder),
        mode='test_fixture', enable_return_assessment=True)
    values = {'match': MATCH}
    if limits:
        values['limits'] = limits
    run = service.create(values)['run_id']
    return service, run, calls, seen


def run_scenario(root):
    """Return an executed, open service and artefacts for the acceptance harness."""
    service, run, calls, seen = _service(root)
    service._drive(run, True)
    return service, run, calls, seen


def _choices(service, run):
    state = service.get(run)
    return [json.loads((service.folder(run) / 'operations' / trace['operation_id'] /
                       'policy-output.json').read_text(encoding='utf-8'))
            for trace in state['selection_trace']]


def test_match_text_entry_discovers_identity_then_selects_information_and_stops(tmp_path):
    service, run, calls, seen = run_scenario(tmp_path)
    try:
        state = service.get(run)
        assert state['last_error'] is None
        assert [turn['action'] for turn in state['turns']] == ['read_url', 'search_web', 'stop']
        assert [call['tool'] for call in calls] == ['read_url', 'search_web']
        assert calls[1]['query'] == MATCH + ' source b'
        assert state['spent'] == {'http_calls': 2, 'llm_calls': 3}
        assert state['question_to_user'] is None and state['user_context'] == []
        assert state['config']['match_input'] == MATCH and state['config']['seed_urls'] == []
        assert [len(item['documents']) for item in seen] == [0, 1, 2]
        assert [item['identity_source_attempts'] for item in seen] == [0, 1, 1]
        choices = _choices(service, run)
        first = choices[0]['diagnostics']
        assert first['selection_basis'] == 'identity_prerequisite_discovery'
        assert first['selected_value']['value_status'] == VALUE_UNMEASURED
        assert first['selected_value']['expected_value'] is None
        assert first['selected_value']['net_value'] is None
        assert first['query_value_decision'] is None
        assert first['state_bound_voi']['intervals'] == {}
        assert first['state_bound_voi']['target_key'] is None
        for choice in choices[1:]:
            diag = choice['diagnostics']['state_bound_voi']
            assert diag['target_key'] == 'spftll:1001'
            assert diag['target_identity_basis'] == 'source_reported'
            assert diag['fixture_identity']['independently_verified_world_fact'] is False
            assert diag['numeric_voi_calibrated'] is False
        assert seen[1]['fixture_identity']['resolution']['target_key'] == 'spftll:1001'
        # The independent resolver ignored the proposer's wrong identity label.
        assert state['match_identity']['label'] == 'spftll:999999'
        assert len(list((service.folder(run) / 'archive').rglob('*.body.gz'))) == 1
        replay = verify_run(service.folder(run))
        assert replay['accepted_decisions_verified'] == replay['fixture_identity_snapshots_verified'] == 3
        assert replay['linked_http_receipts'] == 2
        assert replay['platform_documents_reconstructed'] == 1
        assert replay['real_calls_made_by_replay'] == 0
        assert state['numeric_voi'] is None
    finally:
        service.close()


def test_source_resolved_target_excludes_same_match_from_empirical_support(tmp_path):
    service, run, _, _ = run_scenario(tmp_path)
    try:
        second = _choices(service, run)[1]['diagnostics']['state_bound_voi']
        rows = {item['candidate_id']: item for item in second['intervals'].values()}
        assert rows['a']['dependency_count'] == 59
        assert rows['b']['dependency_count'] == 60
    finally:
        service.close()


def test_only_same_target_positive_history_stays_unmeasured_after_identity(tmp_path):
    config = _configuration()
    for item in config['history']:
        if item['link']['candidate_id'] == 'b':
            item['link']['target_key'] = 'spftll:1001'
    service, run, calls, _ = _service(tmp_path, config=config)
    try:
        service._drive(run, True)
        state = service.get(run)
        assert [turn['action'] for turn in state['turns']] == ['read_url', 'defer']
        assert len(calls) == 1 and state['spent']['http_calls'] == 1
        last = _choices(service, run)[-1]['diagnostics']
        assert last['selection_basis'] == 'open_world_value_withheld'
        by_route = {item['candidate_id']: item for item in last['state_bound_voi']['intervals'].values()}
        assert by_route['b']['dependency_count'] == 0 and by_route['b']['mean_value'] is None
    finally:
        service.close()


@pytest.mark.parametrize('case,payload', [
    ('ambiguous', recent_payload([_row(), _row(matchId=1002)])),
    ('partial', recent_payload([_row(), _row(matchId=1002, homeScore=-1)])),
    ('duplicate_raw', recent_payload([_row(), _row()])),
    ('no_match', recent_payload([_row(home='其他队伍')])),
    ('empty', recent_payload([])),
    ('truncated', recent_payload([_row(matchId=1001 + i) for i in range(80)])),
])
def test_unresolved_source_identity_defers_after_single_prerequisite(tmp_path, case, payload):
    service, run, calls, seen = _service(tmp_path, payload=payload, offer_second_catalogue=True)
    try:
        service._drive(run, True)
        state = service.get(run)
        assert state['last_error'] is None
        assert [turn['action'] for turn in state['turns']] == ['read_url', 'defer']
        assert len(calls) == 1 and state['spent'] == {'http_calls': 1, 'llm_calls': 2}
        assert seen[-1]['identity_source_attempts'] == 1
        diag = _choices(service, run)[-1]['diagnostics']['state_bound_voi']
        assert diag['target_key'] is None and not diag['identity_prerequisite_selected']
        assert not diag['intervals'] and diag['blocked_reason'] == 'canonical_target_unresolved'
        if case == 'duplicate_raw':
            assert any(error['reason'] == 'duplicate_raw_source_ids'
                       for error in seen[-1]['fixture_identity']['errors'])
        assert verify_run(service.folder(run))['fixture_identity_snapshots_verified'] == 2
        assert state['question_to_user'] is None
    finally:
        service.close()


def test_identity_source_consumes_existing_http_budget_before_numeric_acquisition(tmp_path):
    service, run, calls, seen = _service(tmp_path, limits={'http_calls': 1})
    try:
        service._drive(run, True)
        state = service.get(run)
        assert state['last_error'] is None
        assert [turn['action'] for turn in state['turns']] == ['read_url', 'defer']
        assert state['spent'] == {'http_calls': 1, 'llm_calls': 2}
        assert len(calls) == 1 and seen[-1]['remaining']['http_calls'] == 0
        assert _choices(service, run)[-1]['diagnostics']['selection_basis'] == 'resource_budget_gate'
        assert verify_run(service.folder(run))['linked_http_receipts'] == 1
    finally:
        service.close()


def test_no_identity_bootstrap_when_remaining_request_slot_is_already_spent(tmp_path):
    service, run, calls, _ = _service(tmp_path, limits={'http_calls': 1})
    try:
        service.mutate(run, lambda state: state['spent'].update(http_calls=1))
        service.step(run)
        state = service.get(run)
        assert state['turns'][-1]['action'] == 'defer'
        assert not calls and state['question_to_user'] is None
        diag = _choices(service, run)[0]['diagnostics']['state_bound_voi']
        assert diag['identity_prerequisite_selected'] is False
    finally:
        service.close()


def test_replay_rejects_identity_wrapper_tampering_even_with_rehashed_input(tmp_path):
    service, run, _, _ = run_scenario(tmp_path)
    try:
        state = service.get(run)
        operation_id = state['selection_trace'][1]['operation_id']
        path = service.folder(run) / 'operations' / operation_id / 'input.json'
        public = json.loads(path.read_text(encoding='utf-8'))
        public['fixture_identity']['resolution']['target_key'] = 'spftll:900001'
        atomic_write(path, public)
        def update_digest(current):
            next(item for item in current['operations'] if item['operation_id'] == operation_id)['input_digest'] = stored_digest(public)
        service.mutate(run, update_digest)
        with pytest.raises(ValueError):
            verify_run(service.folder(run))
    finally:
        service.close()


@pytest.mark.parametrize('field,direction', [
    ('reference_utc', 'changed_creation'),
    ('snapshot_as_of', 'before_creation'),
    ('snapshot_as_of', 'after_model_reservation'),
])
def test_replay_rejects_identity_time_rebinding_even_with_rehashed_input(tmp_path, field, direction):
    service, run, _, _ = run_scenario(tmp_path)
    try:
        state = service.get(run)
        operation_id = state['selection_trace'][1]['operation_id']
        operation = next(item for item in state['operations'] if item['operation_id'] == operation_id)
        path = service.folder(run) / 'operations' / operation_id / 'input.json'
        public = json.loads(path.read_text(encoding='utf-8'))
        if direction == 'after_model_reservation':
            replacement = datetime.fromisoformat(operation['started_at']) + timedelta(seconds=1)
        else:
            replacement = datetime.fromisoformat(state['created_at']) - timedelta(seconds=1)
        public[field] = replacement.isoformat()
        atomic_write(path, public)
        def update_digest(current):
            next(item for item in current['operations'] if item['operation_id'] == operation_id)['input_digest'] = stored_digest(public)
        service.mutate(run, update_digest)
        with pytest.raises(ValueError, match='REPLAY_IDENTITY_CLOCK_MISMATCH'):
            verify_run(service.folder(run))
    finally:
        service.close()


def test_replay_rejects_removed_public_source_document_even_with_rehashed_input(tmp_path):
    service, run, _, _ = run_scenario(tmp_path)
    try:
        state = service.get(run)
        operation_id = state['selection_trace'][1]['operation_id']
        path = service.folder(run) / 'operations' / operation_id / 'input.json'
        public = json.loads(path.read_text(encoding='utf-8'))
        public['documents'] = []
        atomic_write(path, public)
        def update_digest(current):
            next(item for item in current['operations'] if item['operation_id'] == operation_id)['input_digest'] = stored_digest(public)
        service.mutate(run, update_digest)
        with pytest.raises(ValueError, match='REPLAY_IDENTITY_DOCUMENT_SET_MISMATCH'):
            verify_run(service.folder(run))
    finally:
        service.close()
