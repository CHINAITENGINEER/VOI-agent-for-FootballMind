'''New-episode handoff of verified actor evidence and question-scoped observations.

No evaluator labels or revised historical choices are supplied to the actor.
A new task is explicit; parent attempts remain in the campaign cost ledger.
'''
from copy import deepcopy
from pathlib import Path

from .audit import RunStore,digest
from .candidate_grounded_navigation import CandidateGroundedWorkflow
from .navigation_response_repair_verify import verify_repaired_navigation


def repaired_actor_handoff(root,run_id,expected_head,repair_registration):
    proof=verify_repaired_navigation(root,run_id,repair_registration)
    store=RunStore(Path(root)/'runs')
    run=store.load(run_id)
    if proof['audit']['head_hash']!=expected_head:
        raise ValueError('Parent head changed')
    if run['state'] not in {'STOPPED','DEFERRED'} or run['pending'] or not run['decisions']:
        raise ValueError('Parent must have a confirmed terminal state')
    docs=[store.get_blob(run_id,run['documents'][key]) for key in run['document_order']]
    observations=[]
    for event in run['events']:
        if event['type']!='navigation_decision':continue
        choice=event['data']['decision']['choice']
        assessment=choice['return_assessment']
        if assessment is None:continue
        public=store.get_blob(run_id,event['data']['input_ref'])
        observations.append({'assessment':deepcopy(assessment),
            'claims':[deepcopy(choice['claims'][i]) for i in assessment['evidence_claim_indices']],
            'scope':'this_return_for_this_requirement_only_not_source_wide_utility',
            'prospective_information_test':deepcopy(public['last_action_result']['information_test']),
            'origin':{'parent_run_id':run_id,'decision_event_hash':event['event_hash']}})
    return {'version':'verified-actor-memory-handoff-v1','parent_root':str(Path(root).resolve()),
        'parent_run_id':run_id,'parent_head_hash':expected_head,
        'parent_completed_at':run['events'][-1]['at'],'documents':docs,
        'document_digests':{d['document_id']:digest(d) for d in docs},
        'opened':deepcopy(run['opened']),'claims':deepcopy(run['claims']),
        'last_choice':deepcopy(run['decisions'][-1]['choice']),
        'parent_task':deepcopy(run['task']),'parent_task_contract':deepcopy(run['target_contract']),
        'actor_return_assessments':observations,'parent_budget':deepcopy(run['budget']),
        'scope':'Earlier actor proposals, raw evidence and its scoped assessments; no evaluator labels or verified fixture truth. Parent budget remains final.'}


class GroundedMemoryWorkflow(CandidateGroundedWorkflow):
    version='grounded-memory-navigation-workflow-v1'

    def configuration(self):
        result=super().configuration()
        result['inherited_feedback']='verified_actor_scoped_assessments_v1'
        return result

    def start(self,task,handoff,contract,**limits):
        if handoff['version']!='verified-actor-memory-handoff-v1':
            raise ValueError('Verified actor-memory handoff required')
        if task['task_id']==handoff['parent_task']['task_id']:
            raise ValueError('New episode requires a distinct task identity')
        return super().start(task,handoff,contract,**limits)

    def public(self,run,*,now=None):
        result=super().public(run,now=now)
        handoff=self.store.get_blob(run['run_id'],run['handoff_ref'])
        if handoff['version']!='verified-actor-memory-handoff-v1':
            raise ValueError('Unexpected actor-memory handoff')
        inherited=deepcopy(handoff['actor_return_assessments'])
        result['prior_information_tests']=inherited+result['prior_information_tests']
        result['episode_boundary']={'parent_run_id':handoff['parent_run_id'],
            'parent_budget_final':deepcopy(handoff['parent_budget']),
            'inherited_assessment_count':len(inherited),
            'new_task_budget_is_separate':True,
            'reference_time_is_this_task_contract':True,
            'prior_assessments_are_not_current_world_truth':True}
        return result
