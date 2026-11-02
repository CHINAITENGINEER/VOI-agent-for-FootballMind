"""Observe the actual loopback page; never launch or modify an experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--output-name',default='delivery-v2');args=p.parse_args()
    if not re.fullmatch(r'delivery-[a-zA-Z0-9_-]+',args.output_name):raise ValueError('Use a plain delivery directory name')
    out=ROOT/'var/work/fm-live-body-chain-repair-20261025'/args.output_name
    out.mkdir(exist_ok=True)
    posts=[];errors=[];external=[];gets=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':1760,'height':1240},device_scale_factor=1)
        def intercept(route):
            req=route.request
            if not req.url.startswith('http://127.0.0.1:8766/'):
                external.append(req.url);route.abort();return
            if req.method!='GET':
                posts.append({'method':req.method,'url':req.url});route.abort();return
            gets.append(req.url);route.continue_()
        context.route('**/*',intercept)
        page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:8766/live.html?run='+args.run_id,wait_until='networkidle')
        page.locator('#model-output-status').filter(has_text='正文更新').wait_for(timeout=30000)
        page.locator('#market-probabilities').scroll_into_view_if_needed()
        page.screenshot(path=str(out/'live-page-actual.png'),full_page=False)
        text=page.locator('#body-pipeline').inner_text()
        market=page.locator('#market-probabilities').inner_text()
        prediction=page.locator('#probabilities').inner_text()
        market_numbers=page.locator('#market-probabilities .number').all_inner_texts()
        f_numbers=page.locator('#probabilities .number').all_inner_texts()
        runtime=page.evaluate('catalog.runtime_bindings')
        page.locator('#evidence').screenshot(path=str(out/'live-body-evidence.png'))
        browser.close()
    report={'run_id':args.run_id,'checked_at':datetime.now(timezone.utc).isoformat(),
        'normal_page':True,'api_reads':gets,'mutation_requests':posts,'external_browser_requests':external,
        'javascript_errors':errors,'body_pipeline_text':text,'market_text':market,'f_text':prediction,
        'runtime_bindings':runtime,'market_display_values':market_numbers,'f_display_values':f_numbers,
        'probabilities_visibly_different':market_numbers!=f_numbers,
        'note':'Local catalog GET may trigger the server normal background refresh; not counted as zero source HTTP.'}
    assert not errors and not posts and not external and not runtime['reload_required']
    assert len(market_numbers)==len(f_numbers)==3 and market_numbers!=f_numbers
    (out/'browser-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'body_pipeline':text,'runtime_pid':runtime['process_id'],'errors':errors,'output':str(out)},ensure_ascii=False))


if __name__=='__main__':main()
