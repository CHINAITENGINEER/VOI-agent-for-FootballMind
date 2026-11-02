"""Real HTTP/browser acceptance for research g; failure injection never invents data.

Run against python -m demo, e.g. python -X utf8 tests/demo/research_g_browser.py
--url http://127.0.0.1:8766 --output var/work/fm-research-g-frontend-20261019-v1/browser
"""
import argparse
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

from playwright.sync_api import expect, sync_playwright


def verify(url, output):
    output.mkdir(parents=True, exist_ok=True)
    checks, errors, screenshots = [], [], []

    def api(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        with urlopen(Request(url + path, data=data, headers={"Content-Type": "application/json"}), timeout=60) as response:
            return json.load(response)

    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)

        def fresh():
            context = browser.new_context(viewport={"width": 1440, "height": 1080})
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            posts = []
            page.on("request", lambda req: posts.append({"url": req.url, "body": req.post_data_json}) if req.method == "POST" else None)
            return context, page, posts

        def ready(page):
            page.goto(url)
            expect(page.locator(".match-card")).to_have_count(5)
            expect(page.locator("#run-agent")).to_be_enabled()

        def view(page):
            run_id = page.locator("#research-run-meta").inner_text().split(" · ")[0]
            return api(f"/api/research-g/runs/{run_id}/view")

        def shot(page, name, locator=None):
            path = output / (name + ".png")
            if locator:
                page.locator(locator).screenshot(path=str(path))
            else:
                page.evaluate('() => window.scrollTo(0,0)')
                page.screenshot(path=str(path), full_page=True)
            screenshots.append(str(path))

        def prepare(page, match="understat:28893"):
            ready(page)
            page.locator(f'.match-card[data-match-id="{match}"]').click()
            page.locator("#research-options > summary").click()
            page.locator("#research-step-mode").check()
            page.locator("#run-agent").click()
            expect(page.locator("#research-next")).to_be_enabled()
            assert view(page)["revision"] == 0

        def stopped(page):
            expect(page.locator("#agent-status")).to_have_attribute("data-state", "STOPPED", timeout=20000)
            expect(page.locator("#download-research-report")).to_be_enabled()

        ctx, page, posts = fresh()
        ready(page)
        assert not posts
        for match in api("/api/research-g/catalog")["matches"]:
            page.locator(f'.match-card[data-match-id="{match["match_id"]}"]').click()
            expect(page.locator("#home-name")).to_have_text(match["home"])
            expect(page.locator("#prob-home")).to_have_text("—")
        assert not posts
        checks.append("All five historical matches selectable; selection performs no POST or fabricated probability")
        page.locator('.match-card[data-match-id="understat:28893"]').click()
        shot(page, "01-before-start")

        def creation_gate(route):
            # Preserve the real response, but inspect duplicate protection while
            # the create request is unresolved in the browser.
            response = route.fetch()
            assert page.locator("#run-agent").is_disabled()
            assert page.locator("#workspace-mode").is_disabled()
            page.evaluate("() => { document.querySelector('#run-agent').click(); document.querySelector('#research-next').click(); }")
            route.fulfill(response=response)

        page.route("**/api/research-g/runs", creation_gate)
        page.evaluate("() => {const b=document.querySelector('#run-agent');b.click();b.click();}")
        stopped(page)
        page.unroute("**/api/research-g/runs", creation_gate)
        assert len(posts) == 2, posts
        assert posts[0]["body"] == {"match_id": "understat:28893", "max_reads": 3, "cost_loss_units": .001, "strategy": "g_two_step"}
        assert posts[1]["body"] == {"action": "run", "expected_revision": 0}
        auto = view(page)
        save("automatic-run.json", auto)
        expect(page.locator("#prediction-explain")).to_contain_text("f 已重算，概率未变")
        expect(page.locator("#g-status")).to_have_text("已训练 g · 研究模式可运行")
        assert auto["f"]["consumed_body_input_ids"] and auto["f"]["modeled_evidence_count"] == 0
        page.locator("#toggle-prediction").click()
        shot(page, "02-auto-stopped-before-after")
        shot(page, "03-terminal-report", "#research-report")
        count = len(posts)
        page.evaluate("() => {for(const id of ['run-agent','research-next','research-stop'])document.getElementById(id).click();}")
        assert len(posts) == count
        for control in ["run-agent", "research-next", "research-stop"]:
            expect(page.locator("#" + control)).to_be_disabled()
        page.reload()
        stopped(page)
        assert len(posts) == count and view(page)["run_id"] == auto["run_id"]
        # Local storage recovery also works with no query string.
        page.goto(url)
        stopped(page)
        assert len(posts) == count and view(page)["run_id"] == auto["run_id"]
        checks.append("Actual automatic run: one create + one run despite repeated clicks; terminal blocks actions; URL/local-storage reload is GET-only")
        with page.expect_download() as download:
            page.locator("#download-research-report").click()
        download.value.save_as(str(output / "downloaded-report.json"))
        report = json.loads((output / "downloaded-report.json").read_text(encoding="utf-8"))
        assert report["view"]["run_id"] == auto["run_id"] and report["evaluation"]["loss_reduction"] == 0
        checks.append("Downloaded report preserves real view, trace and terminal evaluation")
        page.locator('#tab-button-evidence').click()
        page.locator('.evidence-details > summary').first.click()
        shot(page, "04-disclosed-body-and-inputs")
        expect(page.locator('.research-body').first).to_have_text(auto['steps'][0]['body']['text'])
        # Check all supported sizes with real expanded content.
        overflow = []
        for width in (1920, 1440, 1280, 1100, 768, 760, 390, 320):
            page.set_viewport_size({"width": width, "height": 1000})
            size = page.evaluate("() => ({viewport:innerWidth, width:document.documentElement.scrollWidth})")
            assert size["width"] <= size["viewport"], size
            overflow.append(size)
        page.set_viewport_size({"width": 390, "height": 844})
        shot(page, "05-mobile-disclosed-body")
        page.locator('#tab-button-overview').click()
        page.locator('#toggle-prediction').click()
        shot(page, "06-mobile-completed")
        save("responsive-widths.json", overflow)
        checks.append("Expanded body and report have no horizontal page overflow at 1920/1440/1280/1100/768/760/390/320px")
        ctx.close()

        ctx, page, posts = fresh()
        prepare(page, "understat:28929")
        initial = view(page)
        assert initial["prediction"]["after"] is None and initial["steps"] == []
        page.locator('#toggle-prediction').click()
        expect(page.locator('#after-value')).to_have_text('尚未执行正文更新')
        page.locator('#candidate-actions .research-candidate > summary').first.click()
        shot(page, "07-step-ready-candidate-scores")
        save("step-initial.json", initial)
        count = len(posts)
        page.reload()
        expect(page.locator('#research-next')).to_be_enabled()
        assert len(posts) == count and view(page)["run_id"] == initial["run_id"]
        page.locator('#research-next').click()
        expect(page.locator('#research-run-meta')).to_contain_text('revision 1')
        expect(page.locator('#research-next')).to_be_enabled()
        one = view(page)
        assert len(one['steps']) == 1 and not one['lifecycle']['is_terminal']
        save("step-after-one.json", one)
        page.locator('#toggle-prediction').click()
        page.locator('#timeline .research-step > summary').first.click()
        shot(page, "08-step-one-f-recomputed")
        page.locator('#research-next').click()
        stopped(page)
        final = view(page)
        assert len(final['steps']) == 2
        save("step-completed.json", final)
        shot(page, "09-step-stopped")
        checks.append("Actual two-step Brentford run: start reveals no body; READY refresh does not advance; two next calls select automatically and stop")
        ctx.close()

        # Immediate STOP at creation must retain after=null.
        ctx, page, posts = fresh()
        ready(page)
        page.locator('.match-card[data-match-id="understat:28913"]').click()
        page.locator('#run-agent').click()
        stopped(page)
        assert len(posts) == 1 and view(page)['prediction']['after'] is None
        expect(page.locator('#prediction-explain')).to_contain_text('尚未执行正文更新')
        shot(page, '10-immediate-stop-no-body')
        # Strategy comparison creates independent IDs and preserves old refs.
        old = view(page)['run_id']
        page.locator('#research-new').click()
        page.reload()
        expect(page.locator('#run-agent')).to_be_enabled()
        assert len(posts) == 1
        page.locator('#research-options > summary').click()
        page.locator('#research-strategy').select_option('fixed_order')
        page.locator('#run-agent').click()
        stopped(page)
        different = view(page)
        assert different['run_id'] != old and different['g']['strategy'] == 'fixed_order'
        expect(page.locator('#research-scope')).to_contain_text('固定顺序对照')
        expect(page.locator('#research-evaluation')).to_contain_text('损失降低 ')
        assert '降低 +' not in page.locator('#research-evaluation').inner_text()
        assert page.locator(f'#research-history option[value="{old}"]').count() == 1
        page.reload()
        stopped(page)
        expect(page.locator('#research-strategy')).to_have_value('fixed_order')
        # A real losing comparison is retained and uses the correct loss sign.
        negative = api('/api/research-g/runs',{'match_id':'understat:28893','strategy':'fixed_order'})
        negative = api(f'/api/research-g/runs/{negative["run_id"]}/action',{'action':'run','expected_revision':negative['revision']})
        page.goto(url+'/?research_run='+negative['run_id'])
        stopped(page)
        expect(page.locator('#research-evaluation')).to_contain_text('损失增加 0.013911')
        expect(page.locator('#research-scope')).to_contain_text('固定顺序对照')
        shot(page,'15-fixed-order-negative-result','#research-report')
        checks.append("Immediate stop keeps after null; new strategy gets independent run ID; READY selection and strategy controls recover accurately")
        ctx.close()

        ctx, page, posts = fresh()
        prepare(page)
        stale = view(page)
        api(f'/api/research-g/runs/{stale["run_id"]}/action', {'action':'next','expected_revision':0})
        count = len(posts)
        page.locator('#research-next').click()
        stopped(page)
        expect(page.locator('#notice')).to_contain_text('STALE_REVISION')
        assert len(posts) == count + 1 and view(page)['revision'] == 1
        shot(page, '11-stale-revision-recovered')
        checks.append("Real stale revision HTTP400 causes GET reconciliation and no automatic retry")
        ctx.close()

        ctx, page, posts = fresh()
        prepare(page)
        action_path = '**/api/research-g/runs/*/action'
        def fail_action(route):
            route.fulfill(status=500, content_type='application/json', body=json.dumps({'error':'ACCEPTANCE_INJECTED_FAILURE'}))
        page.route(action_path, fail_action)
        count = len(posts)
        page.locator('#research-next').click()
        expect(page.locator('#notice')).to_contain_text('未自动重复推进')
        assert len(posts) == count + 1 and view(page)['prediction']['after'] is None
        expect(page.locator('#prediction-explain')).to_contain_text('尚未执行正文更新')
        shot(page, '12-injected-request-failure-real-unchanged-state')
        checks.append("Injected action failure does not invent after/body or retry; actual server remains revision 0")
        # If reconciliation also fails, disable mutation until successful GET.
        page.route('**/api/research-g/runs/*/view', lambda r:r.fulfill(status=503,content_type='application/json',body='{"error":"ACCEPTANCE_GET_FAILURE"}'))
        page.locator('#research-next').click()
        expect(page.locator('#notice')).to_contain_text('推进已锁定')
        expect(page.locator('#research-next')).to_be_disabled()
        expect(page.locator('#agent-status')).to_have_text('状态待核对')
        page.unroute('**/api/research-g/runs/*/view')
        page.unroute(action_path, fail_action)
        page.locator('#research-refresh').click()
        expect(page.locator('#research-next')).to_be_enabled()
        assert view(page)['prediction']['after'] is None
        checks.append("Uncertain action + failed reconciliation locks advancement; readonly recovery unlocks it")
        # Real commit followed by response loss: recover committed state exactly.
        def lost_response(route):
            response = route.fetch()
            assert response.status == 200
            route.abort('failed')
        page.route(action_path, lost_response)
        count = len(posts)
        page.locator('#research-next').click()
        stopped(page)
        expect(page.locator('#notice')).to_contain_text('未自动重复推进')
        assert len(posts) == count + 1 and len(view(page)['steps']) == 1
        checks.append("Lost response after actual commit recovers the committed step without executing twice")
        ctx.close()

        ctx, page, posts = fresh()
        ready(page)
        page.route('**/api/research-g/runs',lambda r:r.fulfill(status=500,content_type='application/json',body='{"error":"ACCEPTANCE_CREATE_FAILURE"}'))
        page.locator('#run-agent').click()
        expect(page.locator('#notice')).to_contain_text('创建结果未确认')
        expect(page.locator('#prob-home')).to_have_text('—')
        assert len(posts) == 1
        checks.append("Create failure remains unknown/empty, with no automatic create retry")
        ctx.close()

        ctx, page, posts = fresh()
        prepare(page)
        page.locator('#research-stop').click()
        stopped(page)
        assert view(page)['lifecycle']['reason'] == 'user_stop' and view(page)['prediction']['after'] is None
        checks.append("Manual stop produces a real terminal user_stop without fabricating a body update")
        page.route('**/api/research-g/runs/*/report',lambda r:r.fulfill(status=503,content_type='application/json',body='{"error":"ACCEPTANCE_REPORT_FAILURE"}'))
        page.locator('#research-refresh').click()
        expect(page.locator('#research-evaluation')).to_contain_text('报告读取失败')
        expect(page.locator('#download-research-report')).to_be_disabled()
        count = len(posts)
        page.unroute('**/api/research-g/runs/*/report')
        page.locator('#research-refresh').click()
        stopped(page)
        assert len(posts) == count
        checks.append("Report failures are distinct; readonly retry downloads actual terminal evaluation")
        ctx.close()

        for label, status, body in [('wrong-schema',200,'{"schema_version":"wrong","matches":[]}'),('wrong-service',404,'{"error":"route missing"}')]:
            ctx, page, posts = fresh()
            def reject_catalog(route):
                route.fulfill(status=status,content_type='application/json',body=body)
            page.route('**/api/research-g/catalog',reject_catalog)
            page.goto(url)
            expect(page.locator('#notice')).to_contain_text('python -m demo')
            expect(page.locator('#run-agent')).to_be_disabled()
            expect(page.locator('.match-card')).to_have_count(0)
            assert not posts
            if label=='wrong-service':shot(page,'13-wrong-service-no-fallback')
            page.unroute('**/api/research-g/catalog',reject_catalog)
            page.locator('#retry-detail').click()
            expect(page.locator('.match-card')).to_have_count(5)
            expect(page.locator('#run-agent')).to_be_enabled()
            assert not posts
            ctx.close()
        checks.append("Wrong service or schema explicitly reports connection/contract failure; no demo matches or g-unavailable inference")

        ctx, page, posts = fresh()
        page.goto(url+'/?workspace=legacy')
        expect(page.locator('#detail-content')).to_be_visible(timeout=20000)
        expect(page.locator('#run-agent')).to_have_text(re.compile('本轮'),timeout=20000)
        assert posts and all('/api/research-g/' not in x['url'] for x in posts)
        legacy_id = next(x for x in reversed(posts) if '/action' in x['url'])['url'].split('/')[-2]
        legacy = api(f'/api/runs/{legacy_id}/view')
        assert legacy['schema_version']=='footballmind-ui-run-view-v1'
        prediction = legacy['prediction']['current']
        expect(page.locator('#prob-home')).to_have_text(f'{prediction["home_win_probability"]*100:.0f}%' if prediction else '—')
        count = len(posts)
        page.locator('#legacy-run-id').fill(legacy_id)
        page.locator('#open-legacy-run').click()
        expect(page.locator('#notice')).to_contain_text('已只读打开旧档案实验')
        assert len(posts) == count
        shot(page,'14-legacy-independent-view')
        save('legacy-view.json',legacy)
        checks.append("Legacy workspace uses only /api/runs, preserves old schema/probabilities, and saved-ID open is readonly")
        ctx.close()

        assert not errors, errors
        browser.close()
    result = {'checks':checks,'count':len(checks),'page_errors':errors,'screenshots':screenshots,'base_url':url,
              'failure_testing':'Only error transport injected; successful responses, probabilities and bodies are from actual backend HTTP.'}
    save('acceptance.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8766')
    parser.add_argument('--output',type=Path,default=Path('var/work/fm-research-g-frontend-20261019-v1/browser'))
    args=parser.parse_args()
    verify(args.url,args.output)
