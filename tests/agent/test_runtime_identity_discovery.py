"""Identity allowance remains bounded and never becomes a value estimate."""
from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from football_agent.acquisition_boundary import CandidateBatch
from football_agent.runtime_fixture_identity import resolve_match_identity
from football_agent.runtime_identity_discovery import discovery_candidate, identity_source_attempts
from football_agent.structured_sources import source_catalog, detail_links

H = runpy.run_path(str(Path(__file__).with_name('test_match_experience.py')))


def sample():
    p = {'task_id': 'fixture', 'task_type': 'match_analysis',
         'match_input': '2026-10-03 甲队 vs 乙队', 'documents': [],
         'snapshot_as_of': '2026-10-02T00:00:00Z', 'identity_source_attempts': 0,
         'remaining': {'http_calls': 2}, 'final_decision_only': False,
         'source_catalog': source_catalog()}
    responses = []
    for item in p['source_catalog']['entries']:
        response = H['match_decision'](p, 'read_url')
        response['url'] = item['url']
        responses.append(response)
    return p, responses


def choose(p, responses, identity=None):
    b = CandidateBatch.build(p, responses, recommended_index=0, origin='test_fixture')
    return discovery_candidate(p, b, identity or resolve_match_identity(p)), b


def test_prerequisite_selects_a_proposed_catalogue_action_without_llm_utility():
    p, responses = sample()
    a, _ = choose(p, responses)
    b, _ = choose(p, list(reversed(responses)))
    assert a.response['url'] == b.response['url'] == min(r['url'] for r in responses)
    assert identity_source_attempts(responses) == 2


@pytest.mark.parametrize('key,value', [
    ('identity_source_attempts', None), ('identity_source_attempts', True),
    ('identity_source_attempts', -1), ('identity_source_attempts', 1),
    ('identity_source_attempts', 2), ('identity_source_attempts', 0.0),
    ('final_decision_only', True), ('remaining', {'http_calls': 0}),
    ('source_catalog', {}), ('seconds_to_deadline', 0),
    ('match_input', '甲队 vs 乙队'), ('match_input', 'spftll:42'),
])
def test_missing_counter_unsupported_identity_or_resource_gate_does_not_discover(key, value):
    p, responses = sample()
    p[key] = value
    assert choose(p, responses)[0] is None


@pytest.mark.parametrize('status', ['ambiguous', 'conflicting', 'incomplete', 'resolved'])
def test_evidence_problems_do_not_start_another_source_sequence(status):
    p, responses = sample()
    identity = resolve_match_identity(p)
    identity['status'] = status
    assert choose(p, responses, identity)[0] is None


def test_unverified_documents_are_not_silently_replaced_by_new_acquisition():
    p, responses = sample()
    identity = resolve_match_identity(p)
    identity['issues'].append({'reason': 'document_not_archive_verified'})
    assert choose(p, responses, identity)[0] is None


def test_failed_accepted_attempt_and_old_turns_consume_cap():
    p, responses = sample()
    turns = [responses[0], {'action': 'stop'}, {'action': 'defer'}]
    p['identity_source_attempts'] = identity_source_attempts(turns)
    assert p['identity_source_attempts'] == 1
    assert choose(p, responses)[0] is None


def test_prerequisite_never_invents_candidate_or_accepts_detail_or_arbitrary_site():
    p, responses = sample()
    other = deepcopy(responses[0])
    for url in ('https://example.org/fixtures', detail_links('1001')[0]['url']):
        other['url'] = url
        assert choose(p, [other])[0] is None
        assert identity_source_attempts([other]) == 0
