'''Bounded CLI reader; completed output and process exit are distinct facts.'''
import json
import os
import signal
import subprocess
import threading
import time
from types import SimpleNamespace

def run_streamed(command, *, input, timeout, encoding='utf-8', creationflags=0, **ignored):
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        creationflags=creationflags,start_new_session=os.name!='nt')
    started=time.monotonic()
    out,err=bytearray(),bytearray()
    completed=threading.Event();overflow=threading.Event()
    observed_at=[]
    def read(stream,target,parse):
        try:
            for line in iter(stream.readline,b''):
                if len(target)+len(line)>1024*1024:
                    overflow.set();break
                target.extend(line)
                if parse:
                    try:
                        event=json.loads(line)
                        if isinstance(event,dict) and event.get('type')=='turn.completed':
                            observed_at.append(time.monotonic()-started);completed.set()
                    except (ValueError,UnicodeError):pass
        except (OSError,ValueError):pass
    readers=[threading.Thread(target=read,args=(process.stdout,out,True),daemon=True),
             threading.Thread(target=read,args=(process.stderr,err,False),daemon=True)]
    for reader in readers:reader.start()
    cleaned=False
    def cleanup():
        nonlocal cleaned
        if process.poll() is not None:return
        cleaned=True
        if os.name=='nt':
            try:subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=4,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except (OSError,subprocess.TimeoutExpired):process.kill()
        else:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        try:process.wait(timeout=4)
        except subprocess.TimeoutExpired:process.kill()
    timed_out=False
    try:
        process.stdin.write(input.encode(encoding));process.stdin.close()
        while process.poll() is None:
            elapsed=time.monotonic()-started
            if overflow.is_set():break
            if completed.is_set():
                try:process.wait(timeout=0.8)
                except subprocess.TimeoutExpired:cleanup()
                break
            if elapsed>=timeout:timed_out=True;break
            completed.wait(min(0.05,timeout-elapsed))
        if timed_out or overflow.is_set():cleanup()
    finally:
        if process.poll() is None:cleanup()
        for reader in readers:reader.join(timeout=1)
    stdout,stderr=bytes(out),bytes(err)
    if timed_out:raise subprocess.TimeoutExpired(command,timeout,output=stdout,stderr=stderr)
    if overflow.is_set():raise RuntimeError('PROVIDER_OUTPUT_LIMIT')
    return SimpleNamespace(args=command,returncode=process.returncode,
        stdout=stdout.decode(encoding),stderr=stderr.decode(encoding,errors='replace'),
        cleanup_after_completed=cleaned and completed.is_set(),
        completed_event_seconds=observed_at[0] if observed_at else None)
