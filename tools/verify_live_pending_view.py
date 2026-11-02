"""Offline browser checks using one real terminal run plus explicit UI fixtures."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo.server import create_server
from demo.live_coverage import summarize
from playwright.sync_api import sync_playwright,expect

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'var/parallel/agent/round2/runs/live-g/runs/lg-cd08c6f2688d40f8abc94dbda20d197f/run.json'
OUTPUT=ROOT/'var/submission/footballmind-defense-20261025-v1/ui-verification/live-pending-v2'

def pending_fixture(original, kind, index):
    """Consistent synthetic pre-f state; never presented as a historical snapshot."""
    first=original['steps'][0]
    decision_index=next(i for i,d in enumerate(original['decisions']) if d.get('action_id')==first['action_id'])
    attempt_index=next(i for i,a in enumerate(original['attempts']) if a.get('purpose')=='g_selected_body' and a.get('action',{}).get('url')==first['url'])
    initial=next(a for a in original['semantic_attempts'] if a['input_id']==first['input_id'] and a['kind']=='initial')
    start=datetime.fromisoformat(initial['requested_at'] if kind=='initial' else initial['completed_at'])
    if kind!='initial':start+=timedelta(milliseconds=20)
    snapshot=start+timedelta(seconds=2 if kind=='initial' else 15)
    input_id=first['input_id'];action_id=first['action_id']
    semantic=[]
    if kind!='initial':
        semantic.append(dict(action_id=action_id,input_id=input_id,kind='initial',
            requested_at=initial['requested_at'],completed_at=initial['completed_at'],
            status='FAILED' if kind=='validation_repair' else 'OK',
            ui_fixture_only=True))
    semantic.append(dict(action_id=action_id,input_id=input_id,kind=kind,status='PENDING',
        requested_at=start.isoformat(),completed_at=None,ui_fixture_only=True))
    attempts=deepcopy(original['attempts'][:attempt_index+1])
    document_ids={a['response']['document']['document_id'] for a in attempts}
    documents=[deepcopy(d) for d in original['documents'] if d['document_id'] in document_ids]
    fixture=deepcopy(original)
    fixture.update(state='RUNNING',revision=2000+index,reason=None,steps=[],
        current=deepcopy(original['before']),
        # Exact original initial-f output clock was not persisted separately.
        forecast_at=None,updated_at=snapshot.isoformat(),
        decisions=deepcopy(original['decisions'][:decision_index+1]),
        attempts=attempts,documents=documents,body_rejections=[],evidence_events=[],
        used_input_ids=[],gain=None,outcome=None,pending=None,
        active_body=dict(action_id=action_id,input_id=input_id,
            document_id=attempts[-1]['response']['document']['document_id'],status='awaiting_semantics'),
        semantic_attempts=semantic,
        counts=dict(source=len(attempts),provider=len(semantic),
            body_attempts=sum(a['purpose']=='g_selected_body' for a in attempts)),
        stage='UI 测试快照（非实际处理中记录）：正文已取得，等待'+{'initial':'抽取','validation_repair':'协议修复','claim_review':'复核'}[kind],
        ui_fixture=dict(synthetic=True,stage=kind,clock=snapshot.isoformat(),
            forecast_clock_unknown=True,baseline_probability_source='original.before',
            no_model_or_provider_execution=True))
    fixture['menus']=deepcopy(fixture['decisions'][-1].get('candidates',[]))
    fixture['body_pipeline']=dict(schema='footballmind-live-body-pipeline-v1',
        body_documents=sum(d.get('status')=='OK' and d.get('structured_news',{}).get('role')=='article' and bool(d.get('text','').strip()) for d in documents),
        body_attempts=fixture['counts']['body_attempts'],semantic_requests=len(semantic),
        claim_review_requests=int(kind=='claim_review'),f_body_updates=0,changed_updates=0,
        audited_updates=0,known_modeled_claims=0,modeled_claims=0,semantic_rejected_bodies=0,
        review_rejected_claims=0,initialization_is_body_update=False,training_performed=False,
        market_difference_l1=sum(abs(a-b) for a,b in zip(fixture['prior'],fixture['current'])))
    assert fixture['current']==fixture['before'] and not fixture['steps'] and not fixture['used_input_ids']
    assert fixture['body_pipeline']['body_documents']==1 and fixture['body_pipeline']['market_difference_l1']<1e-12
    assert all(datetime.fromisoformat(a['handoff_at'])<=snapshot for a in attempts)
    assert all(datetime.fromisoformat(d['decision_at'])<=snapshot for d in fixture['decisions'])
    return fixture

class NoAPI:
    def get(self,*args):raise AssertionError('Browser request escaped local fixture')
    post=get

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    original=json.loads(SOURCE.read_text(encoding='utf8'));state={'run':original}
    match=original['match']
    catalog=dict(schedule_ui_contract='footballmind-daily-schedule-ui-v1',server_time=original['updated_at'],
        status='ready',provider_configured=True,refreshing=False,matches=[match],runs=[],coverage=summarize([match]),sources={})
    server=create_server('127.0.0.1',0,NoAPI());worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    origin=f'http://127.0.0.1:{server.server_port}';errors=[];external=[];posts=[];checks=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1500,'height':1050})
            page.on('pageerror',lambda error:errors.append(str(error)))
            def intercept(route):
                target=urlsplit(route.request.url)
                if target.netloc!=urlsplit(origin).netloc:external.append(route.request.url);route.abort();return
                if not target.path.startswith('/api/'):route.continue_();return
                if route.request.method!='GET':posts.append(target.path);raise AssertionError('No mutation in read-only view test')
                result=catalog if target.path.endswith('/catalog') else state['run']
                route.fulfill(status=200,content_type='application/json',body=json.dumps(result,ensure_ascii=False))
            page.route('**/*',intercept)
            page.goto(origin+'/live.html?run='+original['run_id']);expect(page.locator('#evidence-count')).to_have_text('4 次更新')
            page.screenshot(path=str(OUTPUT/'gais-real-terminal.png'),full_page=True);checks.append('Real completed GAIS record displays all four f updates')
            for index,kind in enumerate(('initial','validation_repair','claim_review')):
                fixture=pending_fixture(original,kind,index);state['run']=fixture
                catalog['server_time']=fixture['updated_at']
                page.clock.set_fixed_time(datetime.fromisoformat(fixture['updated_at']))
                page.reload();expect(page.locator('#round-replay')).to_contain_text('等待实际返回')
                expect(page.locator('#round-replay')).not_to_contain_text('未记录可提交的 f 更新')
                expect(page.locator('#round-replay')).to_contain_text('已用')
                expect(page.locator('#evidence-count')).to_have_text('0 次更新')
                expect(page.locator('#body-pipeline')).to_contain_text('已取得 1 篇')
                expect(page.locator('#body-pipeline')).to_contain_text('L1 差异 0.000000')
                expect(page.locator('#body-pipeline')).to_contain_text('f 实际采用 0 篇')
                expect(page.locator('#forecast')).to_contain_text('尚无正文更新')
                page.locator('#round-replay').evaluate("node => {const note=document.createElement('p'); note.textContent='UI 合成测试快照：无真实请求，f 尚未执行；不是历史处理中证据。'; note.style.cssText='padding:16px;background:#fff4dd;color:#603e00;font-weight:bold'; node.prepend(note);}")
                page.locator('#round-replay').screenshot(path=str(OUTPUT/('ui-fixture-'+kind+'-replay.png')))
                page.screenshot(path=str(OUTPUT/('ui-fixture-'+kind+'.png')),full_page=True)
                (OUTPUT/('ui-fixture-'+kind+'.json')).write_text(json.dumps(fixture,ensure_ascii=False,indent=2),encoding='utf8')
                checks.append(dict(kind=kind,current=fixture['current'],forecast_at=fixture['forecast_at'],
                    counts=fixture['counts'],body_pipeline=fixture['body_pipeline'],
                    rendered_wait=page.locator('#round-replay').inner_text(),
                    status='PASS: internally consistent pending state; no f after or gain'))
            assert not errors and not external and not posts,(errors,external,posts)
            browser.close()
    finally:server.shutdown();server.server_close();worker.join(timeout=3)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_sha
    result=dict(status='PASS',real_source=str(SOURCE),source_sha256=source_sha,source_unchanged=True,
        supersedes='live-pending/verification.json (pending fixtures retained terminal current/counters; replaced by this receipt)',
        real_terminal_steps=4,synthetic_ui_states=3,checks=checks,
        browser_errors=errors,external_requests=external,post_requests=posts,new_model_calls=0,
        note='Pending screenshots are explicitly derived UI fixtures, not real pending provider observations.')
    (OUTPUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
