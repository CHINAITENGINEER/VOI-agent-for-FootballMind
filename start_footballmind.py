'''Start or reopen the local human trial without changing global settings.'''
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler
import webbrowser

ROOT=Path(__file__).resolve().parent

def health(port):
    try:
        with build_opener(ProxyHandler({})).open(f'http://127.0.0.1:{port}/api/health',timeout=1) as r:
            data=json.loads(r.read())
        return data if data.get('application')=='FootballMind Human Trial' else None
    except (OSError,ValueError):return None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8770)
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    missing=[m for m in ['requests','bs4','pydantic','filelock','jsonschema'] if importlib.util.find_spec(m) is None]
    if missing:
        print('Missing dependencies: '+', '.join(missing))
        print('Install: python -m pip install -r football_agent/requirements-inquiry.txt')
        return 1
    if not shutil.which('codex'):
        print('Codex CLI was not found on PATH. Use the Python environment of this workspace.')
        return 1
    for port in range(args.port,args.port+6):
        existing=health(port)
        if existing:
            expected=(ROOT/'var'/'experience').resolve()
            if Path(existing.get('data_directory','')).resolve()!=expected:continue
            print(f'Reusing FootballMind: http://127.0.0.1:{port}/')
            if not args.no_browser:webbrowser.open(f'http://127.0.0.1:{port}/')
            return 0
        try:
            with socket.socket() as check:check.bind(('127.0.0.1',port))
        except OSError:continue
        output=ROOT/'var'/'experience'/'server'
        output.mkdir(parents=True,exist_ok=True)
        log=output/('server-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.log')
        with log.open('ab') as stream:
            process=subprocess.Popen([sys.executable,'-u','-m','demo.experience_server','--port',str(port)],
                cwd=ROOT,stdout=stream,stderr=stream,stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(50):
            data=health(port)
            if data:
                record={'pid':process.pid,'port':port,'url':f'http://127.0.0.1:{port}/','log':str(log),'data_directory':data['data_directory']}
                (output/'last-start.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps(record,ensure_ascii=False,indent=2))
                if not args.no_browser:webbrowser.open(record['url'])
                return 0
            if process.poll() is not None:break
            time.sleep(0.2)
        print('Startup did not complete. See '+str(log))
        return 1
    print('Ports are occupied. Try: python start_footballmind.py --port 8780')
    return 1

if __name__=='__main__':raise SystemExit(main())
