'''Human-trial provider: stream completion is independent of CLI shutdown.'''
from __future__ import annotations
import hashlib
import json
import subprocess
import time
from uuid import uuid4
from jsonschema import validate
from .audit import atomic_write
from .inquiry_provider import CodexInquiryProvider, provider_schema
from .inquiry_provider_diagnostics import record_process_diagnostic
from .experience_runner import run_streamed
from .provider_contract import build_receipt

class ExperienceProvider(CodexInquiryProvider):
    version='experience-streamed-provider-v1'
    def __init__(self, workspace, *, protocol, effort='low', timeout_seconds=60, runner=None):
        if effort not in {'low','medium','high'}: raise ValueError('EFFORT_PROFILE')
        self.effort = effort
        self.last_attempt_folder = None
        super().__init__(workspace, timeout_seconds=timeout_seconds, runner=runner or run_streamed, protocol=protocol)

    def configuration(self):
        return {**super().configuration(), 'reasoning_effort_requested':self.effort,
                'settings_scope':'this_process_only','model_switch':False,
                'completion_basis':'validated_single_turn_completed_stream'}

    def decide(self, public):
        if not self.executable:raise RuntimeError('CODEX_CLI_NOT_CONFIGURED')
        folder=self.workspace/str(uuid4());folder.mkdir(parents=True)
        self.last_attempt_folder = folder
        schema,output=folder/'response.schema.json',folder/'response.json'
        atomic_write(schema,provider_schema(self.protocol.schema))
        atomic_write(folder/'input.json',public)
        command=[self.executable,'exec','--ephemeral','--ignore-user-config','--skip-git-repo-check',
            '--sandbox','read-only','--json','--color','never','--cd',str(folder),
            '--output-schema',str(schema),'--output-last-message',str(output),
            '-c','web_search='+chr(34)+'disabled'+chr(34),'-c','project_doc_max_bytes=0',
            '-c','approval_policy='+chr(34)+'never'+chr(34)]
        for feature in ('shell_tool','apps','multi_agent','browser_use','computer_use','image_generation',
                        'plugins','hooks','goals','workspace_dependencies','skill_search'):
            command.extend(['--disable',feature])
        command.extend(['--enable','code_mode_host','-c','model_reasoning_effort='+chr(34)+self.effort+chr(34),'-'])
        prompt=self.protocol.prompt+chr(10)+'PUBLIC_STATE_JSON:'+chr(10)+json.dumps(public,ensure_ascii=False,allow_nan=False)
        started=time.monotonic()
        try:
            process=self._runner(command,input=prompt,capture_output=True,text=True,encoding='utf-8',
                timeout=self.timeout_seconds,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except (subprocess.TimeoutExpired,OSError) as exc:
            record_process_diagnostic(folder,stdout=getattr(exc,'stdout',None),stderr=getattr(exc,'stderr',None),
                category='timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'runner_os_error',
                elapsed_seconds=time.monotonic()-started,timeout_seconds=self.timeout_seconds,exception_type=type(exc).__name__)
            raise
        record_process_diagnostic(folder,stdout=process.stdout,stderr=process.stderr,category='stream_process_observed',
            elapsed_seconds=time.monotonic()-started,timeout_seconds=self.timeout_seconds,returncode=process.returncode)
        events=[]
        for line in process.stdout.splitlines():
            try:event=json.loads(line)
            except ValueError:raise RuntimeError('PROVIDER_NON_JSON_EVENT') from None
            if not isinstance(event,dict) or not isinstance(event.get('type'),str):raise RuntimeError('PROVIDER_EVENT_SHAPE')
            if event.get('item') is not None and not isinstance(event['item'],dict):raise RuntimeError('PROVIDER_ITEM_SHAPE')
            events.append(event)
        unexpected=[e['item']['type'] for e in events if isinstance(e.get('item'),dict)
                    and e['item'].get('type') not in {None,'agent_message','reasoning','error'}]
        completed=[i for i,e in enumerate(events) if e['type']=='turn.completed']
        messages=[(i,e['item'].get('text')) for i,e in enumerate(events)
                  if e['type']=='item.completed' and e.get('item',{}).get('type')=='agent_message']
        cleanup=bool(getattr(process,'cleanup_after_completed',False))
        atomic_write(folder/'process.json',{'returncode':process.returncode,'cleanup_after_completed':cleanup,
            'event_types':[e['type'] for e in events],'unexpected_item_types':unexpected})
        if unexpected:raise RuntimeError('PROVIDER_USED_UNDECLARED_TOOL')
        if (len(completed)!=1 or not messages or sum(e['type']=='turn.started' for e in events)!=1
            or any(e['type']=='turn.failed' for e in events) or messages[-1][0]>=completed[0]
            or process.returncode and not cleanup):raise RuntimeError('PROVIDER_TURN_FAILED')
        raw=messages[-1][1]
        if not isinstance(raw,str):raise RuntimeError('PROVIDER_MESSAGE_TEXT')
        response=json.loads(raw)
        validate(response,self.protocol.schema)
        if output.exists() and json.loads(output.read_text(encoding='utf-8'))!=response:
            raise RuntimeError('PROVIDER_FILE_STREAM_MISMATCH')
        stream_file=folder/'stream-response.json'
        stream_file.write_bytes(raw.encode('utf-8'))
        stream_bytes = stream_file.read_bytes()
        receipt={'provider':'codex-cli','version':self.version,'real_provider_call':True,
            'requests':1,'requests_unit':'CLI_turn_provider_internal_requests_unknown',
            'usage':[e.get('usage') for e in events if e['type']=='turn.completed'],
            'latency_seconds':time.monotonic()-started,
            'completed_event_seconds':getattr(process,'completed_event_seconds',None),
            'returncode':process.returncode,'process_exited_normally':process.returncode==0 and not cleanup,
            'cleanup_after_completed':cleanup,'response_origin':'exact_agent_message_text',
            'response_sha256':hashlib.sha256(stream_bytes).hexdigest(),
            'stdout_sha256':hashlib.sha256(process.stdout.encode()).hexdigest(),
            'model_requested':'cli_default_no_override','response_model':None,'tool_events':[],
            'actual_money':None,'money_status':'not_reported'}
        receipt.update(build_receipt(
            provider_version=self.version, protocol=self.protocol, public=public,
            prompt=prompt, schema_bytes=schema.read_bytes(), stream_bytes=stream_bytes,
            output_bytes=output.read_bytes() if output.exists() else None,
            stdout_bytes=process.stdout.encode('utf-8'),
            events=events, command=command,
            disabled_features=('shell_tool','apps','multi_agent','browser_use','computer_use',
                               'image_generation','plugins','hooks','goals',
                               'workspace_dependencies','skill_search'),
            enabled_features=('code_mode_host',), cleanup_after_completed=cleanup,
            returncode=process.returncode, elapsed_seconds=receipt['latency_seconds']))
        atomic_write(folder/'receipt.json',receipt)
        return {'response':response,'receipt':receipt,'profile':{'reasoning_effort_requested':self.effort,
                    'effective_effort_confirmed':False}}
