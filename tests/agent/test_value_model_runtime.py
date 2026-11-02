'''Value-model wiring checks with constructed providers and sources only.'''
from copy import deepcopy
import json
from pathlib import Path
import runpy

from football_agent.acquisition_replay import verify_run
from football_agent.experience import ExperienceService
from football_agent.value_model_adapter import VERSION, CandidateValueModel
from football_agent.value_selection_policy import ValueAwareSelectionPolicy

helpers = runpy.run_path(str(Path(__file__).with_name('test_match_experience.py')))
match_decision, MatchTools = helpers['match_decision'], helpers['MatchTools']


def test_value_model_reconsiders_after_source_return_without_user_research_question(tmp_path):
    # Declared fixture values exercise plumbing, not semantic information gain.
    model = CandidateValueModel({
        'version': VERSION,
        'rules': [{'rule_id': 'initial-source', 'action': 'search_web',
                   'query_contains': 'fixture source B', 'when': {'documents': []},
                   'expected_value': 0.40, 'cost': 0.05}],
    })
    policy = ValueAwareSelectionPolicy(model.assess, model_descriptor=model.configuration())
    seen = []

    class Proposer:
        def propose(self, public):
            seen.append(deepcopy(public))
            a = match_decision(public)
            a['query'] = 'fixture source A ' + str(len(public['documents']))
            b = deepcopy(a)
            b['query'] = 'fixture source B ' + str(len(public['documents']))
            return {'candidates': [a, b], 'recommended_index': 0, 'test_double': True}

    service = ExperienceService(tmp_path, proposer_factory=lambda f, e: Proposer(),
                                selection_policy=policy, tools_factory=lambda f: MatchTools(),
                                mode='test_fixture')
    try:
        run = service.create({'match': '构造甲队 vs 构造乙队'})['run_id']
        service._drive(run, True)
        state = service.get(run)
        assert state['status'] == 'deferred', state['last_error']
        assert [turn['action'] for turn in state['turns']] == ['search_web', 'defer']
        assert state['turns'][0]['query'] == 'fixture source B 0'
        assert state['spent'] == {'http_calls': 1, 'llm_calls': 2}
        assert state['question_to_user'] is None and state['user_context'] == []
        assert [len(public['documents']) for public in seen] == [0, 1]
        assert all('match_context' not in public for public in seen)
        assert seen[1]['documents'][0]['selection_operation_id']
        choices = []
        for trace in state['selection_trace']:
            folder = service.folder(run) / 'operations' / trace['operation_id']
            choices.append(json.loads((folder / 'policy-output.json').read_text(encoding='utf-8')))
        assert choices[0]['candidate_id'] and choices[1]['terminal_action'] == 'defer'
        report = verify_run(service.folder(run))
        assert report['accepted_decisions_verified'] == 2
        assert report['linked_http_receipts'] == 1 and report['real_calls_made_by_replay'] == 0
        assert state['numeric_voi'] is None
    finally:
        service.close()
