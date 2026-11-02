"""GET-only browser projection of actual saved postread Application runs.

No acquisition, inference, API provider, or mutation of source run records.
Only the catalogue wrapper is a display fixture; run JSON remains unchanged.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [p.resolve() for p in args.runs]
    input_hashes = {str(p): sha(p) for p in paths}
    records = [json.loads(p.read_text(encoding='utf8')) for p in paths]
    assert all(r['pipeline_mode'] == 'postread_fulltext_v1' for r in records)
    assert all(r['state'] in {'STOPPED', 'FAILED', 'INTERRUPTED', 'CANCELLED'} for r in records)

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(ROOT/'demo/static')))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    state = {'record': records[0]}
    errors, unexpected, requests, checks = [], [], [], []

    def route(request):
        parsed = urlsplit(request.request.url)
        if parsed.netloc != urlsplit(origin).netloc or request.request.method != 'GET':
            unexpected.append({'url': request.request.url, 'method': request.request.method})
            return request.abort()
        path = parsed.path
        if not path.startswith('/api/'):
            return request.continue_()
        requests.append(path)
        r = state['record']
        if path == '/api/live-g/catalog':
            value = dict(schedule_ui_contract='footballmind-daily-schedule-ui-v1',
                pipeline_mode=r['pipeline_mode'], encoder_ready=True, provider_configured=False,
                server_time=r['updated_at'], updated_at=r['updated_at'], status='ready',
                refreshing=False, matches=[r['match']], sources={}, runs=[],
                max_reads=20, default_model_name='fulltext_mlp_seed20261027',
                model_choices=[dict(id='fulltext_mlp_seed20261027', label='冻结全文 g · MLP seed27'),
                    dict(id='state_ridge', label='冻结状态 Ridge 对照'),
                    dict(id='fixed_order', label='固定顺序 · 非 g 对照')],
                coverage={}, limitations=['浏览器验收：仅回放真实已保存的 Application 结果，不启动采集或模型。'])
        elif path == '/api/live-g/runs/'+r['run_id']:
            value = r
        else:
            unexpected.append({'path': path})
            return request.abort()
        request.fulfill(status=200, content_type='application/json', body=json.dumps(value, ensure_ascii=False))

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.route('**/*', route)
            for index, r in enumerate(records):
                state['record'] = r
                page.goto(origin+'/live.html?run='+r['run_id'])
                expect(page.locator('#usage-state')).to_have_text('本轮已结束' if r['state']=='STOPPED' else {'FAILED':'本轮失败','INTERRUPTED':'运行中断','CANCELLED':'已停止'}[r['state']])
                page.evaluate("document.querySelector('#pipeline-description').prepend('【实际 Application 已保存记录；此浏览器仅回放，未采集或推理】 ')")
                expected_steps = [s for s in r.get('steps', []) if s.get('f_executed') is True]
                expected_encoded = sum(x.get('status') == 'encoded' for x in r.get('candidate_pool', []))
                expect(page.locator('#evidence-count')).to_have_text(f'{len(expected_steps)} 次更新')
                expect(page.locator('#postread-pool-count')).to_contain_text(f'已编码 {expected_encoded} 篇')
                assert '语义校验失败' not in page.locator('#body-pipeline').inner_text()
                current = r.get('current')
                if current:
                    assert page.locator('#probabilities .number').all_inner_texts() == [f'{x*100:.2f}%' for x in current]
                comparison = page.locator('#market-probabilities .number').all_inner_texts()
                for round_index, decision in enumerate(r.get('decisions', [])):
                    page.select_option('#replay-select', str(round_index))
                    text = page.locator('#round-replay').inner_text()
                    if decision.get('action_id') is None:
                        assert 'STOP 的 after 不存在' in text
                    if decision.get('g_selection_executed') is False:
                        assert '固定顺序选择' in text
                    assert page.locator('#market-probabilities .number').all_inner_texts() == comparison
                if r.get('decisions'):
                    page.select_option('#replay-select', '0')
                    rows = r['decisions'][0].get('candidates', [])
                    expected = sorted((float(c['score']) for c in rows if isinstance(c.get('score'), (int, float))), reverse=True)
                    displayed = page.locator('#scores .candidate-value strong').all_inner_texts()
                    assert displayed == [('−' if x<0 else '')+f'{abs(x):.6f}' for x in expected]
                if r.get('candidate_pool'):
                    page.locator('#postread-pool > details').first.evaluate('(e)=>e.open=true')
                    body_details = page.locator('#postread-pool > details').first.locator('details').first
                    if body_details.count():
                        body_details.evaluate('(e)=>e.open=true')
                    first = r['candidate_pool'][0]
                    if first.get('text_sha256'):
                        expect(page.locator('#postread-pool > details').first).to_contain_text(first['text_sha256'])
                page.locator('#trace-card').scroll_into_view_if_needed() if page.locator('#trace-card').count() else page.locator('#stage').scroll_into_view_if_needed()
                page.screenshot(path=str(args.output/f'actual-application-{index+1}-desktop.png'))
                page.set_viewport_size({'width': 390, 'height': 844})
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
                page.screenshot(path=str(args.output/f'actual-application-{index+1}-mobile.png'), full_page=True)
                page.set_viewport_size({'width': 1440, 'height': 1000})
                checks.append(dict(path=str(paths[index]), run_id=r['run_id'], head=r.get('model_name'), offline_replay=r.get('offline_replay', False),
                    candidates=len(r.get('candidate_pool', [])), encoded=expected_encoded,
                    actual_f_adoptions=len(expected_steps), decisions=len(r.get('decisions', [])),
                    stop_reason=r.get('reason'), initial=r.get('before'), current=current,
                    market_comparison= comparison, verified=True))
            browser.close()
    finally:
        server.shutdown();server.server_close();worker.join(timeout=3)
    assert not errors and not unexpected, (errors, unexpected)
    assert input_hashes == {str(p): sha(p) for p in paths}
    result = dict(status='passed', scope='actual_application_records_read_only_browser_projection',
        input_hashes=input_hashes, checks=checks, external_requests=0, provider_requests=0,
        model_inference=0, source_bytes_unchanged=True, page_errors=errors,
        unexpected_requests=unexpected, local_get_requests=len(requests),
        catalogue_wrapper='explicit display fixture; actual run responses unchanged')
    (args.output/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'status':'passed','output':str(args.output),'runs':len(checks)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
