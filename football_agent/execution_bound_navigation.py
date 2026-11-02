# The executor owns request identity. Raw model choices contain no task hash.
from copy import deepcopy
from uuid import uuid4
from jsonschema import Draft202012Validator
from .audit import atomic_write,digest
from .candidate_grounded_navigation import SCHEMA as GROUNDED_SCHEMA,PROMPT as GROUNDED_PROMPT
from .grounded_navigation_episode import GroundedMemoryWorkflow
from .inquiry_provider import CodexInquiryProvider
from .target_navigation import validate_contract

SCHEMA=deepcopy(GROUNDED_SCHEMA)
SCHEMA['properties'].pop('task_contract_digest')
SCHEMA['required'].remove('task_contract_digest')
PROMPT=GROUNDED_PROMPT.replace('Echo task_contract_digest. ','')+chr(10)+    'Request identity is bound by the executor. Return only decision fields in the schema. Do not output task_contract_digest or any request identity. The fixed task_contract still governs every decision.'


class ExecutionBoundProtocol:
    version='execution-bound-grounded-navigation-v1'
    schema,prompt=SCHEMA,PROMPT


def model_public(public):
    result=deepcopy(public)
    result.pop('task_contract_digest',None)
    # Old decisions are observations, not examples for copying transport fields.
    for key in ('last_choice','inherited_information_need'):
        if isinstance(result.get(key),dict):result[key].pop('task_contract_digest',None)
    result.pop('response_validation_feedback',None)
    return result


def request_record(public,request_id):
    validate_contract(public['task_contract'],public['task'])
    context=public['execution_context']
    if set(context)!={'run_id','attempt'} or not isinstance(context['run_id'],str) or not context['run_id'] or type(context['attempt']) is not int or context['attempt']<1:
        raise ValueError('Explicit execution context required')
    return {'version':'executor-request-binding-v1','request_id':request_id,
        'task_id':public['task']['task_id'],'task_contract_digest':digest(public['task_contract']),
        'public_input_digest':digest(public),'execution_context':deepcopy(context),'deadline':public['task']['deadline'],
        'protocol':ExecutionBoundProtocol.version}


def verify_response_binding(reply,public,request):
    expected=request_record(public,request['request_id'])
    if request!=expected:raise ValueError('Request task or visible state changed')
    binding=reply['execution_binding']
    if binding!={**expected,'response_digest':digest(reply['response']),'provider_receipt_digest':digest(reply['receipt'])}:
        raise ValueError('Response is not bound to the captured request')
    return True


class ExecutionBoundCodexProvider(CodexInquiryProvider):
    version='execution-bound-codex-provider-v1'

    def __init__(self,workspace,**kwargs):
        super().__init__(workspace,protocol=ExecutionBoundProtocol(),**kwargs)

    def configuration(self):
        result=super().configuration();result['request_identity']='executor_owned_raw_response_unchanged'
        return result

    def decide(self,public):
        captured=deepcopy(public)
        request=request_record(captured,str(uuid4()))
        folder=self.workspace/'bindings'/request['request_id']
        atomic_write(folder/'request.json',request)
        reply=super().decide(captured)
        if captured!=public:raise ValueError('Provider input mutated during invocation')
        result={**reply,'execution_binding':{**request,'response_digest':digest(reply['response']),
            'provider_receipt_digest':digest(reply['receipt'])}}
        verify_response_binding(result,public,request)
        atomic_write(folder/'binding.json',result['execution_binding'])
        return result


class ExecutionBoundWorkflow(GroundedMemoryWorkflow):
    version='execution-bound-grounded-workflow-v1'

    def configuration(self):
        result=super().configuration()
        result.update(protocol=ExecutionBoundProtocol.version,protocol_digest=digest({'schema':SCHEMA,'prompt':PROMPT}),
            request_identity='executor_owned_raw_response_unchanged')
        return result

    def public(self,run,*,now=None):
        result=model_public(super().public(run,now=now))
        result['execution_context']={'run_id':run['run_id'],'attempt':run['budget']['llm_calls']+1}
        return result

    def validate(self,choice,run,public):
        event=run['events'][-1]
        if event['type']!='navigation_reasoning_returned':raise ValueError('Returned call record required')
        if self.store.get_blob(run['run_id'],event['data']['input_ref'])!=public:
            raise ValueError('Response input differs from captured executor context')
        reply=self.store.get_blob(run['run_id'],event['data']['output_ref'])
        binding=reply['execution_binding']
        request=request_record(public,binding['request_id'])
        verify_response_binding(reply,public,request)
        if reply['response']!=choice or public['execution_context']!={'run_id':run['run_id'],'attempt':run['budget']['llm_calls']}:
            raise ValueError('Response belongs to another run or attempt')
        for prior in run['events'][:-1]:
            if prior['type']=='navigation_reasoning_returned':
                previous=self.store.get_blob(run['run_id'],prior['data']['output_ref'])
                if previous.get('execution_binding',{}).get('request_id')==binding['request_id']:
                    raise ValueError('Provider request identity reused')
        return self.validate_choice(choice,run,public)

    def validate_choice(self,choice,run,public):
        validate_contract(run['target_contract'],run['task'])
        if digest(run['target_contract'])!=run['target_contract_digest'] or public['task_contract']!=run['target_contract'] or public['task']!=run['task']:
            raise ValueError('Executor task context changed')
        Draft202012Validator(SCHEMA).validate(choice)
        # Compatibility projection is local validation input only, never a model response.
        projected={**deepcopy(choice),'task_contract_digest':run['target_contract_digest']}
        return super().validate(projected,run,public)
