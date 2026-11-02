'''Loopback human-trial server: python -m demo.experience_server --open.'''
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import threading
from urllib.parse import urlsplit
import webbrowser
from football_agent.experience import ExperienceService, VERSION

STATIC = Path(__file__).parent / 'static' / 'experience'
ROOT = Path(__file__).resolve().parents[1]
PATH = re.compile(r'^/api/runs/([a-f0-9]{32})(?:/(action|answer|feedback|export))?$')

class ExperienceServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, service):
        if address[0] != '127.0.0.1': raise ValueError('LOCAL_HOST_REQUIRED')
        self.service = service
        super().__init__(address, ExperienceHandler)

class ExperienceHandler(BaseHTTPRequestHandler):
    server_version = 'FootballMindHumanTrial/1'
    def log_message(self, fmt, *args): return
    def headers_valid(self):
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in hosts: return False
        origin = self.headers.get('Origin')
        return origin is None or origin in {f'http://{h}' for h in hosts}
    def send(self, status, payload, content_type='application/json; charset=utf-8', download=False):
        if not isinstance(payload, bytes): payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        own, no = chr(39)+'self'+chr(39), chr(39)+'none'+chr(39)
        self.send_header('Content-Security-Policy', f'default-src {own}; connect-src {own}; script-src {own}; style-src {own}; img-src {own} data:; frame-ancestors {no}; base-uri {no}; form-action {own}')
        if download: self.send_header('Content-Disposition', 'attachment; filename=footballmind-run.json')
        self.end_headers()
        self.wfile.write(payload)
    def json_body(self):
        if self.headers.get_content_type() != 'application/json': raise ValueError('JSON_CONTENT_TYPE_REQUIRED')
        if self.headers.get('Transfer-Encoding'): raise ValueError('TRANSFER_ENCODING_UNSUPPORTED')
        size = int(self.headers.get('Content-Length', '-1'))
        if not 0 < size <= 65536: raise ValueError('BODY_SIZE')
        value = json.loads(self.rfile.read(size).decode('utf-8'))
        if not isinstance(value, dict): raise ValueError('JSON_OBJECT_REQUIRED')
        return value
    def dispatch(self, method):
        if not self.headers_valid(): return self.send(403, {'error':'LOCAL_ORIGIN_REQUIRED'})
        path = urlsplit(self.path).path
        svc = self.server.service
        if method == 'GET' and path in {'/', '/app.js', '/style.css'}:
            name = {'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[path]
            media = {'index.html':'text/html','app.js':'text/javascript','style.css':'text/css'}[name]
            return self.send(200, (STATIC/name).read_bytes(), media+'; charset=utf-8')
        if method == 'GET' and path == '/api/health':
            return self.send(200, {'application':'FootballMind Human Trial','version':VERSION,
                'cli_found':bool(shutil.which('codex')),'mode':svc.mode,
                'data_directory':str(svc.root),'numeric_voi_calibrated':False})
        if path == '/api/runs':
            if method == 'GET': return self.send(200, {'runs':svc.list_runs()})
            if method == 'POST': return self.send(201, svc.create(self.json_body()))
        match = PATH.fullmatch(path)
        if not match: return self.send(404, {'error':'NOT_FOUND'})
        run_id, operation = match.groups()
        if method == 'GET' and operation is None: return self.send(200, svc.get(run_id))
        if method == 'GET' and operation == 'export': return self.send(200, svc.export(run_id), download=True)
        if method != 'POST': return self.send(405, {'error':'METHOD_NOT_ALLOWED'})
        body = self.json_body()
        if operation == 'action':
            if set(body) != {'action'} or body['action'] not in {'auto','step','cancel'}: raise ValueError('ACTION_INVALID')
            if body['action'] == 'cancel': return self.send(200, svc.cancel(run_id))
            return self.send(202, svc.launch(run_id, automatic=body['action']=='auto'))
        if operation == 'answer':
            if set(body) != {'text'}: raise ValueError('ANSWER_FIELDS')
            return self.send(200, svc.answer(run_id, body['text']))
        if operation == 'feedback': return self.send(201, svc.feedback(run_id, body))
        return self.send(404, {'error':'NOT_FOUND'})
    def handle_method(self, method):
        try: self.dispatch(method)
        except (ValueError, TypeError, AttributeError) as exc: self.send(400, {'error':str(exc)[:180]})
        except FileNotFoundError: self.send(404, {'error':'RUN_OR_FILE_NOT_FOUND'})
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as exc: self.send(500, {'error':type(exc).__name__})
    def do_GET(self): self.handle_method('GET')
    def do_POST(self): self.handle_method('POST')

def main():
    parser = argparse.ArgumentParser(description='FootballMind local human-trial service')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--root', default=str(ROOT/'var'/'experience'))
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    service = ExperienceService(args.root)
    try: server = ExperienceServer(('127.0.0.1',args.port),service)
    except OSError:
        service.close()
        raise SystemExit('Port is occupied. Open the existing service or choose --port 8771.')
    url = f'http://127.0.0.1:{server.server_port}/'
    service.recover()
    print(f'FootballMind Human Trial: {url}', flush=True)
    print(f'Data: {service.root}', flush=True)
    if args.open: threading.Timer(0.4, lambda:webbrowser.open(url)).start()
    try: server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        service.close()

if __name__ == '__main__': main()
