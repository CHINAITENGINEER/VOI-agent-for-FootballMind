from datetime import timedelta
import json

from football_agent.action_value_ledger import build_action_value_ledger
from football_agent.acquisition_replay import verify_run
from football_agent.experience import ExperienceService, now
from football_agent.experience_sources import project_body


URL = 'https://example.org/late'


def _decision(public, action):
    return {
        'task_id': public['task_id'], 'action': action,
        'url': URL if action == 'read_url' else None,
        'query': None, 'engine': None, 'question_to_user': None,
        'unknown_to_resolve': '页面是否有首发信息',
        'useful_if': '页面正文列出首发', 'why_worth_cost': '可减少未知',
        'previous_return': 'none' if not public['documents'] else 'partial',
        'assessment': {'status': 'unresolved', 'answer': '待核实', 'claims': [],
                       'remaining_unknowns': ['首发']},
        'alternatives': [], 'reason': '读取来源',
    }


class Provider:
    def decide(self, public):
        return {'response': _decision(public, 'read_url' if not public['documents'] else 'stop'),
                'test_double': True}


def test_late_return_is_archived_but_not_admitted(tmp_path):
    holder = {}

    class LateTools:
        def execute(self, action, *, run_id, reserve_request, cancelled):
            request_id = reserve_request(action['url'])
            # The request has already spent its resource. Simulate the return
            # arriving after task cancellation, before actor state is applied.
            holder['service'].cancel(run_id)
            document = project_body(b'<article>late source</article>', URL, 200)
            document.update(document_id='late-doc', observed_at=now().isoformat(),
                            source_published_at=None)
            return {'document': document,
                    'requests': [{'request_id': request_id, 'http_status': 200, 'error': None}]}

    service = ExperienceService(tmp_path, provider_factory=lambda f, e: Provider(),
                                tools_factory=lambda f: LateTools())
    holder['service'] = service
    try:
        run = service.create({'question': '核实来源', 'seed_urls': [URL]} )['run_id']
        service._drive(run, True)
        state = service.get(run)
        assert state['status'] == 'cancelled'
        assert state['spent']['http_calls'] == 1
        assert state['documents'] == []
        assert state['queued_action'] is None and state['pending'] is None
        record = state['acquisition_admissions'][0]
        assert record['status'] == 'rejected' and record['reason'] == 'cancelled'
        result_path = next((service.folder(run) / 'acquisitions').glob('*/result.json'))
        admission_path = result_path.parent / 'admission.json'
        assert json.loads(result_path.read_text(encoding='utf-8'))['document']['document_id'] == 'late-doc'
        assert json.loads(admission_path.read_text(encoding='utf-8')) == record
        assert service.get(run)['spent'] == {'http_calls': 1, 'llm_calls': 1}
        audit = verify_run(service.folder(run), include_action_value_ledger=True)
        assert audit['real_calls_made_by_replay'] == 0
        ledger = build_action_value_ledger(service.folder(run))
        row = ledger['entries'][0]
        assert row['acquisition_status'] == 'rejected_late_return'
        assert row['resource_cost']['http_calls'] == 1
    finally:
        service.close()


def test_deadline_rejection_does_not_leak_public_document(tmp_path):
    holder = {}

    class LateTools:
        def execute(self, action, *, run_id, reserve_request, cancelled):
            request_id = reserve_request(action['url'])
            holder['service'].mutate(run_id, lambda state: state.update(
                deadline=(now() - timedelta(seconds=1)).isoformat()))
            document = project_body(b'<article>too late</article>', URL, 200)
            document.update(document_id='deadline-doc', observed_at=now().isoformat(),
                            source_published_at=None)
            return {'document': document,
                    'requests': [{'request_id': request_id, 'http_status': 200, 'error': None}]}

    service = ExperienceService(tmp_path, provider_factory=lambda f, e: Provider(),
                                tools_factory=lambda f: LateTools())
    holder['service'] = service
    try:
        run = service.create({'question': '核实来源', 'seed_urls': [URL]})['run_id']
        service._drive(run, True)
        state = service.get(run)
        assert state['status'] == 'deadline_expired'
        assert state['documents'] == []
        assert state['acquisition_admissions'][0]['reason'] == 'deadline_expired'
        assert verify_run(service.folder(run))['action_value_ledger_entries'] == 1
    finally:
        service.close()
