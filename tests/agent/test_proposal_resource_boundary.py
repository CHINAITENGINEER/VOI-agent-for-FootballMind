"""Unfunded proposals may be compared; selected actions still need resources."""
from pathlib import Path
import runpy

import pytest

from football_agent.experience import ExperienceService, public_state, validate_decision

helpers = runpy.run_path(str(Path(__file__).with_name('test_match_experience.py')))


@pytest.fixture
def exhausted(tmp_path):
    service = ExperienceService(tmp_path, mode='test_fixture')
    try:
        state = service.create({'match': '2026-10-03 甲队 vs 乙队'})
        public = public_state(state)
        public['remaining']['http_calls'] = 0
        public['final_decision_only'] = True
        yield public
    finally:
        service.close()


def test_zero_budget_proposals_do_not_change_budget_and_are_not_executable(exhausted):
    response = helpers['match_decision'](exhausted)
    validate_decision(response, exhausted, for_proposal=True)
    assert exhausted['remaining']['http_calls'] == 0 and exhausted['final_decision_only']
    with pytest.raises(ValueError, match='FINAL_DECISION_REQUIRED'):
        validate_decision(response, exhausted)
    exhausted['final_decision_only'] = False
    with pytest.raises(ValueError, match='HTTP_BUDGET_EXHAUSTED'):
        validate_decision(response, exhausted)


@pytest.mark.parametrize('bad', ['url', 'question', 'information_test', 'repeat'])
def test_proposal_mode_preserves_nonresource_execution_contract(exhausted, bad):
    from football_agent.experience import action_key
    response = helpers['match_decision'](exhausted, 'read_url' if bad == 'url' else 'search_web')
    if bad == 'url':
        response['url'] = 'https://unobserved.example.org/fixtures'
        code = 'URL_NOT_OBSERVED'
    elif bad == 'question':
        response['active_question_id'] = 'missing-question'
        code = 'SOURCE_ACTION_REQUIRES_OPEN_ANALYSIS_QUESTION'
    elif bad == 'information_test':
        response['useful_if'] = ' '
        code = 'INFORMATION_TEST_REQUIRED'
    else:
        exhausted['do_not_repeat'] = [action_key(response)]
        code = 'REPEATED_ACTION'
    with pytest.raises(ValueError, match=code):
        validate_decision(response, exhausted, for_proposal=True)
