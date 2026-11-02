from copy import deepcopy
import json
import threading
import pytest

from football_agent.experience import ExperienceService, MatchAnalysisProtocol, MatchLocalAnalysisProtocol, public_state, validate_decision, now
from football_agent.experience_sources import project_body

URL = 'https://example.org/fixture'

def match_decision(public, action='search_web'):
    return {'task_id': public['task_id'], 'action': action,
            'url': URL if action == 'read_url' else None,
            'query': '测试甲队 测试乙队 官方赛程' if action == 'search_web' else None,
            'engine': 'bing' if action == 'search_web' else None,
            'question_to_user': '请选择这场比赛的日期。' if action == 'ask_user' else None,
            'unknown_to_resolve': '这场比赛的正式阵容是否已发布',
            'useful_if': '具体比赛与正式首发可以对应', 'why_worth_cost': '阵容证据可以修正对比赛实力的分析',
            'previous_return': 'partial' if public['documents'] else 'none',
            'assessment': {'status': 'unresolved', 'answer': '构造任务，尚待核实。', 'claims': [], 'remaining_unknowns': ['阵容']},
            'alternatives': [], 'reason': '针对比赛自主确定需要验证的信息',
            'match_identity': {'status': 'ambiguous' if action == 'ask_user' else 'unresolved', 'label': public['match_input']},
            'analysis_plan': [{'id': 'fixture', 'question': '确认具体比赛与阵容对应关系',
                               'decision_relevance': '避免将其他比赛的阵容用于本场实力分析', 'status': 'open'}],
            'active_question_id': 'fixture' if action in {'search_web', 'read_url'} else None,
            'clarification_field': 'date' if action == 'ask_user' else 'none'}

class Provider:
    def __init__(self, fn): self.fn = fn
    def decide(self, public): return {'response': self.fn(public), 'test_double': True}

class MatchTools:
    def execute(self, action, *, run_id, reserve_request, cancelled):
        search = action['tool'] == 'search_web'
        url = 'https://example.org/search' if search else URL
        request_id = reserve_request(url)
        if not request_id: return {'document': None, 'requests': []}
        body = (b'<rss><channel><item><title>Two fixtures, date unresolved</title><link>https://example.org/fixture</link></item></channel></rss>'
                if search else b'<article>Squad list only &lt;img src=x onerror=alert(1)&gt;</article>')
        document = project_body(body, url, 200)
        document.update(document_id='search-fixture' if search else 'match-page', observed_at=now().isoformat(), source_published_at=None)
        return {'document': document, 'requests': [{'request_id': request_id, 'http_status': 200, 'error': None}]}

@pytest.fixture
def service(tmp_path):
    svc = ExperienceService(tmp_path, provider_factory=lambda f,e: Provider(lambda p: match_decision(p)),
                            tools_factory=lambda f: MatchTools(), mode='test_fixture')
    yield svc
    svc.close()

def create(service, **values):
    return service.create({'match': '测试甲队 vs 测试乙队', **values})['run_id']

def test_user_supplies_match_only_system_owns_objective(service):
    run = create(service)
    state = service.get(run)
    assert state['config']['task_type'] == 'match_analysis'
    assert state['config']['seed_urls'] == []
    assert state['analysis_plan'] == [] and state['user_context'] == []
    assert state['numeric_voi'] is None and state['selection_method'] == 'LLM_QUALITATIVE_BASELINE'
    public = public_state(state)
    assert public['analysis_objective'] and public['match_input'] == '测试甲队 vs 测试乙队'
    assert public['autonomous_match_state']['contract']['no_user_scope_question'] is True
    assert public['autonomous_match_state']['policy']['mode'] == 'conditional_semantic_voi_untrained_g'
    assert any(item['information_need'] == 'rotation_lineup'
               for item in public['autonomous_match_state']['candidate_catalog'])
    assert 'MATCH-FIRST TASK CONTRACT' in MatchAnalysisProtocol.prompt

@pytest.mark.parametrize('extra', [{'question': '先查伤停'}, {'seed_urls': [URL]}, {'match': {}}, {'match': '  '}])
def test_match_input_does_not_require_or_accept_research_instructions(service, extra):
    with pytest.raises(ValueError): create(service, **extra)

def test_automatic_search_read_replan_stop_without_user_question(service):
    seen = []
    def choose(public):
        seen.append(deepcopy(public))
        count = len(public['documents'])
        response = match_decision(public, ['search_web', 'read_url', 'stop'][min(count, 2)])
        if count:
            response['match_identity']['status'] = 'identified'
            response['analysis_plan'] = [{'id': 'lineup', 'question': '这份名单能否确定本场首发',
                                          'decision_relevance': '区分报名名单与实际阵容，避免错误推断轮换', 'status': 'open'}]
            response['active_question_id'] = 'lineup' if count == 1 else None
        if count == 2:
            response['assessment'] = {'status': 'partial', 'answer': '名单不能证明首发，比赛强弱仍需条件性判断。',
                                      'claims': [{'document_id': 'match-page', 'quote': 'Squad list only', 'statement': '仅有报名名单'}],
                                      'remaining_unknowns': ['本场正式首发']}
        return response
    service.provider_factory = lambda f,e: Provider(choose)
    run = create(service); service._drive(run, True); state = service.get(run)
    assert state['status'] == 'deferred'
    assert [t['action'] for t in state['turns']] == ['search_web', 'read_url', 'stop']
    assert state['spent'] == {'llm_calls': 3, 'http_calls': 2}
    assert state['user_context'] == []
    assert [len(p['documents']) for p in seen] == [0, 1, 2]
    assert seen[-1]['previous_analysis_plan'][0]['id'] == 'lineup'
    assert state['analysis_plan'][0]['question'] == '这份名单能否确定本场首发'

def test_must_attempt_fixture_resolution_before_asking(service):
    service.provider_factory = lambda f,e: Provider(lambda p: match_decision(p, 'ask_user'))
    run = create(service); service._drive(run, True)
    state = service.get(run)
    assert state['status'] == 'paused'
    assert state['last_validation_error'] == 'TRY_IDENTIFYING_MATCH_BEFORE_ASKING_USER'

def test_ambiguous_fixture_can_request_identity_after_search(service):
    run = create(service); service.step(run); service.step(run)
    service.provider_factory = lambda f,e: Provider(lambda p: match_decision(p, 'ask_user'))
    service.step(run); before = service.get(run)
    assert before['status'] == 'needs_input'
    service.answer(run, '日期是 2026-10-01，构造测试赛事')
    after = service.get(run)
    assert before['spent'] == after['spent'] and before['deadline'] == after['deadline']

@pytest.mark.parametrize('field,value,error', [
    ('active_question_id', 'not-listed', 'SOURCE_ACTION_REQUIRES_OPEN_ANALYSIS_QUESTION'),
    ('clarification_field', 'competition', 'NO_CLARIFICATION_FIELD_WITHOUT_QUESTION'),
])
def test_every_source_action_maps_to_self_generated_open_question(service,field,value,error):
    public = public_state(service.get(create(service)))
    response = match_decision(public); response[field] = value
    with pytest.raises(ValueError,match=error): validate_decision(response,public)

def test_reject_duplicate_questions_and_already_resolved_target(service):
    public = public_state(service.get(create(service)))
    response = match_decision(public); response['analysis_plan'] *= 2
    with pytest.raises(ValueError,match='DUPLICATE'): validate_decision(response, public)
    response = match_decision(public); response['analysis_plan'][0]['status'] = 'supported'
    with pytest.raises(ValueError,match='OPEN_ANALYSIS_QUESTION'): validate_decision(response, public)

def test_do_not_ask_user_to_set_analysis_scope(service):
    run = create(service); service.step(run); service.step(run)
    public = public_state(service.get(run))
    response = match_decision(public, 'ask_user'); response['question_to_user'] = '你希望重点分析哪些方面？'
    with pytest.raises(ValueError,match='DO_NOT_DELEGATE'): validate_decision(response, public)

def test_default_match_run_selects_match_protocol(service, monkeypatch):
    captured = []
    class CapturedProvider:
        def __init__(self, folder, **kwargs): captured.append(kwargs['protocol'])
        def decide(self, p): return {'response': match_decision(p, 'defer'), 'test_double': True}
    monkeypatch.setattr('football_agent.experience.ExperienceProvider', CapturedProvider)
    service.provider_factory = None
    run = create(service); service.step(run)
    assert captured == [MatchAnalysisProtocol]
    assert service.get(run)['status'] == 'deferred'

def test_legacy_record_remains_readable_without_match_migration(service):
    state = service.create({'question': '旧版核验任务保留'})
    assert public_state(state)['task_type'] == 'legacy_inquiry'
    assert service.export(state['run_id'])['state']['config']['question'] == '旧版核验任务保留'

def test_state_reader_and_writer_share_lock(service, monkeypatch):
    run = create(service)
    reading, release, writer_started, written = (threading.Event() for _ in range(4))
    original = service._read_state
    def controlled(folder):
        if threading.current_thread().name == 'state-reader':
            reading.set()
            assert release.wait(3)
        return original(folder)
    monkeypatch.setattr(service, '_read_state', controlled)
    def write():
        writer_started.set()
        service.mutate(run, lambda s: s.update(last_error='test marker'))
        written.set()
    reader = threading.Thread(target=lambda: service.get(run), name='state-reader')
    writer = threading.Thread(target=write, name='state-writer')
    reader.start()
    try:
        assert reading.wait(3)
        writer.start()
        assert writer_started.wait(3)
        assert not written.wait(0.1)
    finally:
        release.set(); reader.join(3); writer.join(3)
    assert written.is_set() and service.get(run)['last_error']=='test marker'
