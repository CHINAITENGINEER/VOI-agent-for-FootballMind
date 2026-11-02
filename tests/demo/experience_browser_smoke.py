'''Browser acceptance uses labeled fixtures, never counted as real acquisition.'''
import argparse
import json
from pathlib import Path
import runpy
import sys
import threading
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from playwright.sync_api import sync_playwright, expect
from demo.experience_server import ExperienceServer
from football_agent.experience import ExperienceService
helpers=runpy.run_path(str(ROOT/'tests'/'agent'/'test_match_experience.py'))
Provider,Tools,decision,URL=(helpers[k] for k in ('Provider','MatchTools','match_decision','URL'))

def verify(output):
    output.mkdir(parents=True,exist_ok=True)
    def choose(p):
        if not p['documents']:return decision(p,'search_web')
        if not p['user_context']:return decision(p,'ask_user')
        if len(p['documents'])==1:return decision(p,'read_url')
        r=decision(p,'stop')
        r['assessment']={'status':'supported','answer':'构造测试：名单已读取，只验证交互，不代表真实比赛。',
            'claims':[{'statement':'构造正文已收到','document_id':'match-page','quote':'Squad list only'}],
            'remaining_unknowns':[]}
        return r
    svc=ExperienceService(output/'fixture-data',provider_factory=lambda f,e:Provider(choose),
        tools_factory=lambda f:Tools(),mode='test_fixture')
    server=ExperienceServer(('127.0.0.1',0),svc)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    checks=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            page=browser.new_page(viewport={'width':1440,'height':1080})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            submissions=[]
            page.on('request',lambda r:submissions.append(r.post_data_json) if r.method=='POST' and r.url.endswith('/api/runs') else None)
            page.goto(f'http://127.0.0.1:{server.server_port}/')
            assert page.locator('#question').count()==0 and page.locator('#sources').count()==0
            page.fill('#match','测试甲队 vs 测试乙队，构造比赛')
            page.click('#start')
            expect(page.locator('#answer-form')).to_be_visible(timeout=10000)
            expect(page.locator('#mode-badge')).to_contain_text('构造测试')
            assert len(submissions)==1 and submissions[0]['match']=='测试甲队 vs 测试乙队，构造比赛'
            assert 'question' not in submissions[0] and 'seed_urls' not in submissions[0]
            expect(page.locator('#analysis-plan')).to_contain_text('确认具体比赛与阵容对应关系')
            checks.append('user submits match only; system-generated analysis plan is visible')
            checks.append('agent attempts source discovery before fixture identity clarification; test mode explicit')
            page.fill('#answer','测试赛事，测试日期与对手')
            page.locator('#answer-form button[type=submit]').click()
            expect(page.locator('#status')).to_have_text('已有证据支持，已停止',timeout=15000)
            expect(page.locator('#http-used')).to_have_text('2 / 8')
            expect(page.locator('#model-used')).to_have_text('4 / 5')
            expect(page.locator('#claims')).to_contain_text('Squad list only')
            checks.append('discover -> identify -> read -> reassess -> stop; durable budgets')
            assert page.locator('#documents img').count()==0
            checks.append('source HTML rendered as text, no injected image')
            page.screenshot(path=str(output/'desktop-completed.png'),full_page=True)
            page.locator('#timeline .event button').first.click()
            page.fill('#comment','构造反馈：建议继续核验比赛首发')
            page.locator('#feedback-form button[type=submit]').click()
            expect(page.locator('#feedback-status')).to_contain_text('已保存',timeout=10000)
            with page.expect_download() as dl:page.click('#export')
            dl.value.save_as(str(output/'browser-export.json'))
            export=json.loads((output/'browser-export.json').read_text(encoding='utf-8'))
            assert export['state']['mode']=='test_fixture' and len(export['feedback'])==1
            assert export['feedback'][0]['event_sequence']==1
            checks.append('feedback bound to event and downloadable with full trace')
            page.reload()
            expect(page.locator('#status')).to_have_text('已有证据支持，已停止',timeout=10000)
            expect(page.locator('#model-used')).to_have_text('4 / 5')
            checks.append('refresh restores same run without further calls')
            page.set_viewport_size({'width':390,'height':844})
            page.evaluate('window.scrollTo(0,0)')
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
            page.screenshot(path=str(output/'mobile.png'),full_page=True)
            checks.append('390px mobile width has no horizontal overflow')
            assert not errors,errors
            checks.append('no browser JavaScript errors')
            browser.close()
        result={'kind':'LABELED_FIXTURE_BROWSER_ACCEPTANCE','checks':checks,'real_model_calls':0,'real_source_requests':0}
        (output/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
    finally:server.shutdown();server.server_close();svc.close();thread.join()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='var/parallel/agent/round114/browser')
    args=parser.parse_args();print(json.dumps(verify(Path(args.output)),ensure_ascii=False))
