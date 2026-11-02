from football_agent.episode_persistence import EpisodePersistence
from football_agent.experience import ExperienceService, now
from football_agent.experience_sources import project_body


URL = 'https://example.org/episode'


def _decision(public, action):
    return {
        'task_id': public['task_id'], 'action': action,
        'url': URL if action == 'read_url' else None,
        'query': None, 'engine': None, 'question_to_user': None,
        'unknown_to_resolve': '是否有首发变化',
        'useful_if': '页面有明确阵容信息', 'why_worth_cost': '可减少未知',
        'previous_return': 'none' if not public['documents'] else 'partial',
        'assessment': {'status': 'unresolved', 'answer': '待核实', 'claims': [],
                       'remaining_unknowns': ['首发']},
        'alternatives': [], 'reason': '按当前未知选择信息',
    }


class Provider:
    def decide(self, public):
        return {'response': _decision(public, 'read_url' if not public['documents'] else 'stop'),
                'test_double': True}


class Tools:
    def execute(self, action, *, run_id, reserve_request, cancelled):
        request_id = reserve_request(URL)
        document = project_body(b'<article>lineup evidence</article>', URL, 200)
        document.update(document_id='episode-doc', observed_at=now().isoformat(),
                        source_published_at=None)
        return {'document': document,
                'requests': [{'request_id': request_id, 'http_status': 200, 'error': None}]}


def test_experience_service_can_opt_in_to_the_same_episode_journal(tmp_path):
    service = ExperienceService(
        tmp_path,
        provider_factory=lambda folder, effort: Provider(),
        tools_factory=lambda folder: Tools(),
        episode_store_factory=EpisodePersistence,
    )
    try:
        run_id = service.create({'question': '核实首发信息', 'seed_urls': [URL]})['run_id']
        service._drive(run_id, True)
        store = EpisodePersistence(service.folder(run_id))
        rows = store.records()
        assert len(rows) == 2
        assert rows[0]['action_id'] == 'read_url'
        assert rows[0]['admission_status'] == 'admitted'
        assert rows[0]['acquisition_id']
        assert rows[1]['action_id'] == 'stop'
        assert rows[1]['acquisition_id'] is None
        assert rows[0]['prediction_before'] is None
        assert rows[0]['prediction_after'] is None
        assert store.validate()['status'] == 'PARTIAL'
        assert store.validate()['numeric_voi_training_eligible'] is False
    finally:
        service.close()


def test_episode_journal_preserves_rejected_late_return_without_actor_leak(tmp_path):
    holder = {}

    class LateTools:
        def execute(self, action, *, run_id, reserve_request, cancelled):
            request_id = reserve_request(URL)
            holder['service'].cancel(run_id)
            document = project_body(b'<article>late evidence</article>', URL, 200)
            document.update(document_id='late-episode-doc', observed_at=now().isoformat(),
                            source_published_at=None)
            return {'document': document,
                    'requests': [{'request_id': request_id, 'http_status': 200, 'error': None}]}

    service = ExperienceService(
        tmp_path,
        provider_factory=lambda folder, effort: Provider(),
        tools_factory=lambda folder: LateTools(),
        episode_store_factory=EpisodePersistence,
    )
    holder['service'] = service
    try:
        run_id = service.create({'question': '核实首发信息', 'seed_urls': [URL]})['run_id']
        service._drive(run_id, True)
        state = service.get(run_id)
        assert state['documents'] == []
        rows = EpisodePersistence(service.folder(run_id)).records()
        assert len(rows) == 1
        assert rows[0]['admission_status'] == 'rejected'
        assert rows[0]['admission_binding']['reason'] == 'cancelled'
        assert rows[0]['acquisition_id']
    finally:
        service.close()


def test_provider_exception_leaves_an_unresolved_acquisition_in_the_journal(tmp_path):
    class FailingTools:
        def execute(self, action, *, run_id, reserve_request, cancelled):
            reserve_request(URL)
            raise RuntimeError('transport interrupted')

    service = ExperienceService(
        tmp_path,
        provider_factory=lambda folder, effort: Provider(),
        tools_factory=lambda folder: FailingTools(),
        episode_store_factory=EpisodePersistence,
    )
    try:
        run_id = service.create({'question': '核实首发信息', 'seed_urls': [URL]})['run_id']
        service._drive(run_id, True)
        state = service.get(run_id)
        assert state['status'] == 'needs_reconciliation'
        assert state['pending']
        rows = EpisodePersistence(service.folder(run_id)).records()
        assert len(rows) == 1
        assert rows[0]['acquisition_id']
        assert rows[0]['admission_status'] == 'unknown'
        assert rows[0]['admission_binding']['reason'] == 'provider_exception'
        assert EpisodePersistence(service.folder(run_id)).validate()['eligible_for_replay'] is False
    finally:
        service.close()

