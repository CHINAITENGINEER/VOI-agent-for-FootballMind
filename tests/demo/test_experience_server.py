import json
import threading
import pytest
import requests
from demo.experience_server import ExperienceServer
from football_agent.experience import ExperienceService

@pytest.fixture
def server(tmp_path):
    svc=ExperienceService(tmp_path)
    http=ExperienceServer(('127.0.0.1',0),svc)
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{http.server_port}',svc
    http.shutdown();http.server_close();svc.close();thread.join()

def test_create_retrieve_feedback_and_export(server):
    url,svc=server
    r=requests.post(url+'/api/runs',json={'question':'请核实本场轮换证据'})
    assert r.status_code==201
    run=r.json()['run_id'];path=url+'/api/runs/'+run
    assert requests.get(path).json()['mode']=='live'
    assert requests.post(path+'/feedback',json={'category':'ui','comment':'测试反馈记录'}).status_code==201
    exported=requests.get(path+'/export')
    assert 'attachment' in exported.headers['Content-Disposition'] and len(exported.json()['feedback'])==1
    assert requests.post(path+'/action',json={'action':'cancel'}).json()['status']=='cancelled'

def test_match_only_api_creates_system_analysis_objective(server):
    url,svc=server
    r=requests.post(url+'/api/runs',json={'match':'测试甲队 vs 测试乙队，构造赛事'})
    assert r.status_code==201
    state=r.json()
    assert state['config']['task_type']=='match_analysis'
    assert state['config']['seed_urls']==[] and state['config']['analysis_objective']
    assert state['analysis_plan']==[] and state['numeric_voi'] is None
    assert requests.post(url+'/api/runs',json={'match':'测试甲队 vs 测试乙队','question':'请先查天气'}).status_code==400
    html=requests.get(url+'/').text
    selector=lambda name: 'id='+chr(39)+name+chr(39)
    assert selector('match') in html and selector('question') not in html and selector('sources') not in html

def test_local_origin_and_host_validation(server):
    url,_=server
    assert requests.post(url+'/api/runs',json={'question':'a test'},headers={'Origin':'https://evil.example'}).status_code==403
    assert requests.get(url+'/api/health',headers={'Host':'evil.example'}).status_code==403
    assert requests.post(url+'/api/runs',data='not json').status_code==400

def test_static_and_path_boundary(server):
    url,_=server
    r=requests.get(url+'/');assert r.status_code==200 and 'Content-Security-Policy' in r.headers
    assert requests.get(url+'/app.js').status_code==200
    assert requests.get(url+'/api/runs/not-a-run').status_code==404
    assert requests.get(url+'/CURRENT_STATUS.md').status_code==404
    assert requests.post(url+'/api/runs',json={'question':[]}).status_code==400
