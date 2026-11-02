'''Bridge an unmeasured context decision to the existing bounded inquiry.

No invented value estimate, new crawler, automatic forecast update or budget
conversion is introduced. The parent and child retain separate audit records.
'''
from copy import deepcopy
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .audit import RunStore, atomic_write, digest, stored_digest
from .autonomous_context_loop import LoopResult
from .contextual_adaptive_voi import normalize_context
from .inquiry import InquiryWorkflow
from .inquiry_contract import utc
from .inquiry_selection import LearningExplorationSelector
from .inquiry_tools import public_url
from .query_value_contract import DEFER, VALUE_UNMEASURED, WITHHELD_OPEN_WORLD_VALUE_UNKNOWN
from football_research.evaluation.policy_experiment_protocol import audit_parameter_freeze, manifest_hash

VERSION = 'context-inquiry-handoff-v1'


def handoff_configuration(*, workflow, context_keys, seed_urls, max_requests, max_llm_calls):
    if not isinstance(workflow, InquiryWorkflow) or not isinstance(workflow.selector, LearningExplorationSelector):
        raise ValueError('An existing InquiryWorkflow with explicit learning exploration is required')
    if type(max_requests) is not int or not 1 <= max_requests <= 6:
        raise ValueError('Request budget must be 1..6')
    if type(max_llm_calls) is not int or not 1 <= max_llm_calls <= 8:
        raise ValueError('LLM budget must be 1..8')
    if workflow.selector.max_exploration_requests > max_requests:
        raise ValueError('Exploration cap must fit the registered HTTP budget')
    if isinstance(context_keys, (str, bytes)) or isinstance(seed_urls, (str, bytes)):
        raise ValueError('Context keys and seed URLs must be sequences')
    keys = tuple(context_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Context projection must contain unique explicit keys')
    normalize_context({key: None for key in keys})
    urls = [public_url(url, workflow.tools.allowed_hosts) for url in seed_urls]
    if not urls or len(set(urls)) != len(urls):
        raise ValueError('Explicit unique public source seeds are required')
    return {
        'version': VERSION, 'workflow_configuration_hash': digest(workflow.configuration()),
        'context_keys': list(keys), 'seed_urls': urls,
        'max_requests': max_requests, 'max_llm_calls': max_llm_calls,
        'max_exploration_requests': workflow.selector.max_exploration_requests,
        'admission': 'zero-query parent with unmeasured-value DEFER only',
        'deadline': 'original_parent_prediction_cutoff',
        'cost_scope': 'separate HTTP and LLM count caps; no conversion from parent risk units',
        'prediction_update': 'none_open_text_remains_unverified',
    }


class ContextInquiryHandoff:
    '''Persist one child inquiry per parent identity; reuse its existing budgets.

    The zero-query condition prevents prior logical-clock replay returns being
    promoted to newly observed operational evidence. The caller must separately
    declare when the projected initial context became available. These checks
    validate declarations, not the external truth of their timestamps.
    '''

    def __init__(self, root, workflow, *, context_keys, seed_urls, max_requests, max_llm_calls):
        self.root, self.workflow = Path(root).resolve(), workflow
        self._config = handoff_configuration(workflow=workflow, context_keys=context_keys,
            seed_urls=seed_urls, max_requests=max_requests, max_llm_calls=max_llm_calls)
        self._locks = RunStore(self.root)

    def configuration(self):
        return deepcopy(self._config)

    def _write(self, path, body):
        atomic_write(path, {**body, 'record_hash': digest(body)})

    def _read(self, path):
        receipt = json.loads(path.read_text(encoding='utf-8'))
        body = {key: value for key, value in receipt.items() if key != 'record_hash'}
        if stored_digest(body) != receipt.get('record_hash'):
            raise ValueError('Handoff receipt hash mismatch')
        return body

    def _prepare(self, result, context_available_at, freeze_manifest):
        if not isinstance(result, LoopResult) or result.final_action != DEFER or not result.decisions:
            raise ValueError('Only an explicit unmeasured-value DEFER may start learning exploration')
        if result.selected_candidates or result.execution_entries or result.declared_cost_spent:
            raise ValueError('This bridge admits initial-context deferrals without prior replay queries')
        last = result.decisions[-1]
        decision = last['decision']
        if (decision['action'] != DEFER or
                decision.get('stop_certificate', {}).get('kind') != WITHHELD_OPEN_WORLD_VALUE_UNKNOWN):
            raise ValueError('Budget, cutoff or step-limit deferrals do not authorize more acquisition')
        board = last['plan']['board']
        if board['target_key'] != result.target_key or last['context']['values'] != result.final_context:
            raise ValueError('Parent identity and final public context must agree')
        unknown = {row['candidate_id'] for row in decision['assessments'] if row['value_status'] == VALUE_UNMEASURED}
        candidates = [row for row in board['candidates'] if row['candidate_id'] in unknown]
        if not candidates:
            raise ValueError('No explicitly unmeasured candidate is available for a handoff')
        now, cutoff, available = utc(self.workflow.clock()), utc(board['prediction_cutoff']), utc(context_available_at)
        if available > now or available > cutoff:
            raise ValueError('Initial context availability follows the current clock or parent cutoff')
        config = handoff_configuration(workflow=self.workflow, context_keys=self._config['context_keys'],
            seed_urls=self._config['seed_urls'], max_requests=self._config['max_requests'],
            max_llm_calls=self._config['max_llm_calls'])
        if config != self._config:
            raise ValueError('Actual inquiry configuration changed after bridge creation')
        freeze = deepcopy(freeze_manifest)
        audit = audit_parameter_freeze(freeze, earliest_test_prediction_cutoff=cutoff.isoformat())
        if audit['status'] != 'PASS' or utc(freeze['frozen_at']) > now:
            raise ValueError('Exploration configuration must be frozen before execution and cutoff')
        if manifest_hash(freeze['parameters'].get('context_inquiry_handoff')) != manifest_hash(config):
            raise ValueError('Actual handoff differs from the frozen configuration')
        thresholds = {key: config[key] for key in ('max_requests', 'max_llm_calls', 'max_exploration_requests')}
        if freeze['thresholds'] != thresholds:
            raise ValueError('Exploration budget thresholds differ from the frozen contract')
        projection = normalize_context({key: result.final_context.get(key) for key in config['context_keys']})
        identity = str(uuid5(NAMESPACE_URL, VERSION + ':' + result.target_key + ':' + result.episode_id))
        task = {
            'task_id': identity, 'deadline': cutoff.isoformat(), 'seed_urls': config['seed_urls'],
            'objective': '针对当前比赛尚未估值的信息，自主提出可改变下一步信息获取的问题并在登记预算内探索。来源文本保持待核验；不生成数值收益或比赛预测。',
            'initial_facts': [
                '目标标识（不是比赛事实）：' + result.target_key,
                '调用方声明的初始公开上下文：' + json.dumps(projection, ensure_ascii=False, sort_keys=True),
                '上下文可用时间（声明）：' + available.isoformat(),
                '尚未估值的输入（不是已观察事实）：' + json.dumps([
                    {'candidate_id': row['candidate_id'], 'state_variable': row['state_variable']}
                    for row in candidates], ensure_ascii=False, sort_keys=True),
                '问题及查询路径由当前证据生成；探索概率只条件于本次生成的候选，不代表信息有用的概率。',
            ],
        }
        binding = {
            'schema_version': VERSION, 'parent_episode_id': result.episode_id,
            'parent_target_key': result.target_key, 'parent_result_hash': digest(result.as_dict()),
            'context_available_at': available.isoformat(), 'parameter_hash': freeze['parameter_hash'],
            'configuration_hash': digest(config), 'task_hash': digest(task),
            'inquiry_store': str(self.workflow.store.root),
        }
        return identity, task, binding, now, cutoff

    def start(self, *, result, context_available_at, freeze_manifest):
        identity, task, binding, now, cutoff = self._prepare(result, context_available_at, freeze_manifest)
        path = self.root / (identity + '.json')
        with self._locks.lock(identity):
            receipt = self._read(path) if path.is_file() else None
            if receipt and receipt['binding'] != binding:
                raise ValueError('Parent handoff already exists with different immutable inputs')
            if receipt and receipt.get('run_id'):
                run = self.workflow.get(receipt['run_id'])
                self._validate_child(run, task)
                return {**receipt, 'run': run, 'reused': True}
            if now >= cutoff:
                raise ValueError('Parent prediction cutoff passed; no new inquiry may start')
            # Persist intent before creating a child. Workflow.start performs no
            # provider/tool call; an interrupted save can be reconciled locally.
            if receipt is None:
                receipt = {'handoff_id': identity, 'binding': binding, 'state': 'STARTING',
                           'run_id': None, 'created_at': now.isoformat()}
                self._write(path, receipt)
            matches = []
            for candidate in self.workflow.store.root.glob('*/run.json'):
                row = json.loads(candidate.read_text(encoding='utf-8'))
                if row.get('task', {}).get('task_id') == identity:
                    matches.append(self.workflow.get(row['run_id']))
            if len(matches) > 1:
                raise ValueError('Multiple child inquiries require reconciliation before execution')
            run = matches[0] if matches else self.workflow.start(task,
                max_requests=self._config['max_requests'], max_llm_calls=self._config['max_llm_calls'])
            self._validate_child(run, task)
            receipt.update(run_id=run['run_id'], state='BOUND')
            self._write(path, receipt)
            return {**receipt, 'run': run, 'reused': bool(matches)}

    def _validate_child(self, run, task):
        if digest(run['task']) != digest(task) or digest(run['configuration']) != self._config['workflow_configuration_hash']:
            raise ValueError('Existing child task or configuration differs from the frozen handoff')
        if any(run['budget'][key] != self._config[key] for key in ('max_requests', 'max_llm_calls')):
            raise ValueError('Child budgets differ from the registered limits')

    def run(self, *, result, context_available_at, freeze_manifest):
        started = self.start(result=result, context_available_at=context_available_at, freeze_manifest=freeze_manifest)
        identity = started['handoff_id']
        with self._locks.lock(identity):
            run = self.workflow.get(started['run_id'])
            if run['state'] not in {'STOPPED', 'DEFERRED', 'WAITING'}:
                run = self.workflow.run(run['run_id'])
            proof = self.workflow.store.verify(run['run_id'])
            receipt = self._read(self.root / (identity + '.json'))
            receipt.update(state='OBSERVED', run_state=run['state'], run_head_hash=proof['head_hash'],
                           http_attempts=run['budget']['http_attempts'], llm_calls=run['budget']['llm_calls'])
            self._write(self.root / (identity + '.json'), receipt)
            return {**receipt, 'run': run, 'reused': started['reused'],
                    'parent_context_updated': False, 'prediction_updated': False,
                    'expected_information_value': None, 'independent_confirmation': False,
                    'cost_scope': self._config['cost_scope']}
