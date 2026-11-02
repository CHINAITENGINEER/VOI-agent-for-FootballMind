'''Explicit single repair of an invalid returned decision, with no cost refund.'''
from copy import deepcopy

from .audit import digest
from .inquiry_contract import utc
from .target_navigation import TargetNavigationWorkflow


def validation_feedback(flow,run,returned,public,reply):
    errors=[]
    try:flow.validate(reply['response'],returned['snapshot'],public)
    except Exception as exc:errors.append({'type':type(exc).__name__,'detail':str(exc)[:500]})
    else:raise ValueError('Valid response has no validation repair to request')
    if public['last_action_result'] and reply['response'].get('return_assessment') is None:
        errors.append({'type':'MissingReturnAssessment',
            'detail':'Assess last_action_result against its earlier information_test. Metadata remains unread.'})
    return {'version':'response-validation-feedback-v1','attempts_at_repair':run['budget']['llm_calls'],
        'rejected_input_ref':returned['data']['input_ref'],'rejected_output_ref':returned['data']['output_ref'],
        'errors':errors,'rejected_response':deepcopy(reply['response']),
        'rejected_action_executed':False,'budget_refunded':False,
        'instruction':'The previous returned JSON failed validation. Reconsider all available choices under the current remaining budget. Return fresh schema JSON. Support indices refer to claims in THIS response. Do not merely repeat the rejected action; reading, querying and deferring remain your choices.'}


class RepairableTargetNavigationWorkflow(TargetNavigationWorkflow):
    version = 'target-information-navigation-response-repair-v1'

    def configuration(self):
        result=super().configuration()
        result['response_repair_policy']='one_explicit_repair_original_budget_no_source_action_replayed'
        return result

    def public(self,run,*,now=None):
        result=super().public(run,now=now)
        feedback=run.get('response_validation_feedback')
        result['response_validation_feedback']=(deepcopy(feedback) if feedback
            and run['budget']['llm_calls']==feedback['attempts_at_repair'] else None)
        return result

    def reopen(self,run_id,registration):
        with self.store.lock(run_id):
            run=self.store.load(run_id)
            if run['state']!='DEFERRED' or run['reason']!='INVALID_NAVIGATION_RESPONSE' or run['pending']:
                raise ValueError('Only a known rejected response may be repaired')
            if any(e['type']=='navigation_response_repair_admitted' for e in run['events']):
                raise ValueError('One explicit repair only')
            if run['budget']['max_llm_calls']-run['budget']['llm_calls']<2:
                raise ValueError('Retain capacity for both a new choice and reconsideration')
            if utc(self.clock())>=utc(run['task']['deadline']):raise ValueError('Original deadline expired')
            if registration['run_id']!=run_id or registration['parent_head_hash']!=run['events'][-1]['event_hash']:
                raise ValueError('Repair registration must bind actual rejected prefix')
            if registration['configuration']!=self.configuration():raise ValueError('Repair configuration differs')
            original=deepcopy(self.configuration())
            original['version']=TargetNavigationWorkflow.version
            original.pop('response_repair_policy')
            if original!=run['configuration']:raise ValueError('Model/protocol/tools changed during repair')
            returned=next(e for e in reversed(run['events']) if e['type']=='navigation_reasoning_returned')
            public=self.store.get_blob(run_id,returned['data']['input_ref'])
            reply=self.store.get_blob(run_id,returned['data']['output_ref'])
            feedback=validation_feedback(self,run,returned,public,reply)
            old_head=run['events'][-1]['event_hash']
            run.update(state='READY',reason='VALIDATION_FEEDBACK',configuration=self.configuration(),
                       response_validation_feedback=feedback)
            self._event(run,'navigation_response_repair_admitted',{'old_head_hash':old_head,
                'registration_digest':digest(registration),'configuration':self.configuration(),'feedback':feedback})
            return deepcopy(run)
