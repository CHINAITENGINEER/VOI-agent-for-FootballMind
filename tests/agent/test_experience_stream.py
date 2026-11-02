import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest
from football_agent.experience_provider import ExperienceProvider
from football_agent.experience_runner import run_streamed

class Protocol:
    version='test-only'
    prompt='Return value.'
    schema={'type':'object','properties':{'value':{'type':'string'}},'required':['value'],'additionalProperties':False}

def events():
    return [{'type':'turn.started'},{'type':'item.completed','item':{'type':'agent_message','text':json.dumps({'value':'unchanged'})}},
            {'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}]

def run_value(values,**kwargs):
    return SimpleNamespace(returncode=kwargs.get('returncode',0),stdout=chr(10).join(json.dumps(e) for e in values),stderr='',
        cleanup_after_completed=kwargs.get('cleanup',False),completed_event_seconds=0.1)

def test_completed_stream_preserves_nonzero_cleanup_returncode(tmp_path):
    provider=ExperienceProvider(tmp_path,protocol=Protocol,runner=lambda *a,**k:run_value(events(),returncode=1,cleanup=True))
    result=provider.decide({'task':'fixture'})
    assert result['receipt']['returncode']==1 and result['receipt']['cleanup_after_completed']
    assert result['receipt']['process_exited_normally'] is False
    assert result['response']=={'value':'unchanged'}
    assert next(tmp_path.glob('*/stream-response.json')).read_text(encoding='utf-8')==events()[1]['item']['text']

@pytest.mark.parametrize('mutation', ['incomplete','tool','double_turn','failed','wrong_order','nonzero','mismatch'])
def test_invalid_stream_not_accepted(tmp_path,mutation):
    values=events()
    if mutation=='incomplete':values.pop()
    if mutation=='tool':values.insert(1,{'type':'item.completed','item':{'type':'web_search','text':'hidden'}})
    if mutation=='double_turn':values.append({'type':'turn.completed'})
    if mutation=='failed':values.insert(1,{'type':'turn.failed'})
    if mutation=='wrong_order':values=[values[0],values[2],values[1]]
    def runner(command,**kwargs):
        if mutation=='mismatch':Path(command[command.index('--output-last-message')+1]).write_text(json.dumps({'value':'different'}),encoding='utf-8')
        return run_value(values,returncode=1 if mutation=='nonzero' else 0)
    with pytest.raises(RuntimeError):ExperienceProvider(tmp_path,protocol=Protocol,runner=runner).decide({'task':'fixture'})

def test_runner_stops_waiting_after_completed_event():
    payload=repr(chr(10).join(json.dumps(e) for e in events()))
    script='import time; print('+payload+',flush=True); time.sleep(20)'
    started=time.monotonic()
    result=run_streamed([sys.executable,'-u','-c',script],input='',timeout=5)
    assert time.monotonic()-started<5 and result.cleanup_after_completed
    assert result.returncode!=0 and 'turn.completed' in result.stdout

def test_runner_timeout_without_completed_is_not_success():
    with pytest.raises(subprocess.TimeoutExpired):
        run_streamed([sys.executable,'-u','-c','import time; print(123,flush=True); time.sleep(20)'],input='',timeout=0.2)
