"""Offline browser acceptance for the live console, using captured runs and UI fixtures.

All API requests are intercepted. No source, provider, model or training call is made.
UI-only state fixtures are kept separate from screenshots of real archived records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright

from demo.server import create_server
from demo.live_g import LIMITATIONS
from demo.live_coverage import availability, summarize


ROOT = Path(__file__).resolve().parents[2]
BASE = "2026-10-25T08:00:00+00:00"
RUN_ID = "lg-" + "a" * 32


def ui_fixture():
    match = {"match_id": "ui-fixture-001", "fixture_id": "ui-fixture-001",
        "source_match_id": "001", "home": "界面测试主队", "away": "界面测试客队",
        "competition_key": "Premier League", "kickoff": "2026-10-25T12:00:00+00:00",
        "status": "scheduled", "supported": True, "can_start": True,
        "availability_reason": "可启动实验 · 启动时重新核验身份与赔率",
        "source": "espn", "sources": ["espn"], "quote_observed": True,
        "quote_observed_at": "2026-10-25T07:55:00+00:00",
        "quote_snapshot": {"provider": "UI fixture", "no_vig_probabilities": {"home": .6, "draw": .25, "away": .15}},
        "model_scope": "premier_league_research", "source_url": "https://example.test/fixture"}
    catalog = {"schedule_ui_contract": "footballmind-daily-schedule-ui-v1", "server_time": BASE,
        "status": "ready", "updated_at": BASE, "last_checked_at": BASE, "provider_configured": True,
        "refreshing": False, "auto_refresh_seconds": None, "matches": [match], "runs": [],
        "coverage": summarize([match]), "sources": {"espn": {"status": "ready"}},
        "model_bindings": {}, "limitations": ["UI state fixture: no model execution"]}
    run = {"run_id": RUN_ID, "revision": 0, "state": "PREPARING", "created_at": BASE,
        "updated_at": BASE, "match": deepcopy(match), "stage": "核验目标比赛身份与新鲜报价",
        "reason": None, "current": None, "before": None, "prior": None,
        "steps": [], "attempts": [], "decisions": [], "menus": [], "documents": [],
        "counts": {"source": 0, "provider": 0, "body_attempts": 0}, "max_reads": 5,
        "cost_loss_units": .001, "source_request_budget": 10, "provider_request_budget": 5,
        "run_timeout_minutes": 15, "forecast_at": None, "outcome": None, "gain": None}
    return catalog, run


class NoAPI:
    def get(self, *_args):
        raise AssertionError("Browser API escaped the local fixture interceptor")

    post = get


def verify(output):
    output.mkdir(parents=True, exist_ok=True)
    server = create_server("127.0.0.1", 0, NoAPI())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    checks, errors, external_requests, screenshots = [], [], [], []

    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)

            def fresh(data=None, records=None, viewport=None):
                catalog, run = ui_fixture()
                state = {"catalog": deepcopy(data or catalog), "run": deepcopy(run), "posts": [],
                         "records": records or {}, "gets": []}
                context = browser.new_context(viewport=viewport or {"width": 1440, "height": 1000})
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))

                def dispatch(route):
                    req = route.request
                    target = urlsplit(req.url)
                    if target.netloc != urlsplit(url).netloc:
                        external_requests.append(req.url)
                        route.abort()
                        return
                    if not target.path.startswith("/api/live-g/"):
                        route.continue_()
                        return
                    path = target.path.removeprefix("/api/live-g/")
                    if path == "catalog" and state.get("fail_catalog"):
                        route.fulfill(status=503, content_type="application/json", body=json.dumps({"error": "UI fixture: catalog unavailable"}))
                        return
                    if req.method == "POST":
                        state["posts"].append({"path": path, "body": req.post_data_json})
                        if path == "runs":
                            result = state["run"]
                        elif path.endswith("/stop"):
                            result = {**state["run"], "cancel_requested": True}
                        elif path == "refresh":
                            result = state["catalog"]
                        else:
                            raise AssertionError(path)
                    else:
                        state["gets"].append(path)
                        result = state["catalog"] if path == "catalog" else state["records"].get(path.split("/")[1], state["run"])
                    route.fulfill(status=200, content_type="application/json", body=json.dumps(result, ensure_ascii=False))

                page.route("**/*", dispatch)
                return context, page, state

            def ready(page, suffix="/live.html"):
                page.goto(url + suffix)
                expect(page.locator("#connection")).to_contain_text("连接")
                expect(page.locator("#coverage-total")).not_to_have_text("—")

            def screenshot(page, name):
                path = output / (name + ".png")
                page.screenshot(path=str(path), full_page=True)
                screenshots.append(str(path))

            ctx, page, state = fresh()
            ready(page)
            expect(page.locator("#market-probabilities")).to_contain_text("60.00%")
            expect(page.locator("#probabilities")).to_be_hidden()
            expect(page.locator("#model-output-status")).to_have_text("尚未生成")
            expect(page.locator("#start")).to_have_text("开始自主分析")
            expect(page.locator("#readiness-title")).to_have_text("目录条件就绪")
            assert not state["posts"]
            expect(page.locator("#research-config details[open]")).to_have_count(0)
            page.evaluate("() => { const b=document.getElementById('start'); b.click(); b.click(); }")
            expect(page.locator("#start")).to_have_text("停止本轮")
            assert [r["path"] for r in state["posts"]] == ["runs"]
            assert state["posts"][0]["body"] == {"match_id": "ui-fixture-001", "max_reads": 5, "cost_loss_units": .001}
            expect(page.locator("#probabilities")).to_be_hidden()
            page.locator("#start").click()
            expect(page.locator("#start")).to_have_text("已请求停止")
            expect(page.locator("#start")).to_be_disabled()
            assert [r["path"] for r in state["posts"]] == ["runs", f"runs/{RUN_ID}/stop"]
            state["run"].update(state="CANCELLED", revision=1, reason="用户停止", stage="已停止")
            page.evaluate("poll()")
            expect(page.locator("#download")).to_be_visible()
            expect(page.locator("#start")).to_be_hidden()
            expect(page.locator("#new")).to_be_visible()
            checks.append("Market baseline is separate from an empty model output; ready/start/stop/terminal controls agree; duplicate clicks issue one create only")
            ctx.close()

            for kind in ["stale", "missing_quote", "provider_missing", "finished"]:
                cat, _ = ui_fixture()
                m = cat["matches"][0]
                if kind == "stale":
                    m.update(can_start=False, availability_reason="先验来源缓存超过 30 分钟，等待刷新；赛程仍可查看", quote_observed_at="2026-10-25T06:00:00+00:00")
                elif kind == "missing_quote":
                    m.update(can_start=False, supported=False, quote_observed=False, quote_snapshot=None, quote_observed_at=None, availability_reason="目录尚无完整 90 分钟赔率，等待来源更新后刷新")
                elif kind == "provider_missing":
                    cat["provider_configured"] = False
                else:
                    m.update(status="finished", can_start=False, availability_reason="比赛已结束")
                cat["coverage"] = summarize(cat["matches"])
                ctx, page, state = fresh(cat)
                ready(page)
                expect(page.locator("#start")).not_to_have_text("开始自主分析")
                expect(page.locator("#probabilities")).to_be_hidden()
                if kind == "stale":
                    expect(page.locator("#start")).to_have_text("刷新市场先验")
                    expect(page.locator("#market-freshness")).to_contain_text("已过期")
                    page.locator("#start").click()
                    assert [r["path"] for r in state["posts"]] == ["refresh"]
                elif kind == "missing_quote":
                    expect(page.locator("#start")).to_have_text("重新检查数据")
                elif kind == "provider_missing":
                    expect(page.locator("#start")).to_have_text("检查服务状态")
                    page.locator("#start").click()
                    assert not state["posts"]
                else:
                    expect(page.locator("#start")).to_have_text("查看可启动比赛")
                ctx.close()
            checks.append("Expired quote refreshes instead of launching; missing quote/provider and finished fixture expose the appropriate recovery action")

            ctx, page, state = fresh()
            state["fail_catalog"] = True
            page.goto(url + "/live.html")
            expect(page.locator("#start")).to_have_text("重新连接")
            expect(page.locator("#start")).to_be_enabled()
            expect(page.locator("#model-output-status")).to_have_text("尚未生成")
            state["fail_catalog"] = False
            page.locator("#start").click()
            expect(page.locator("#start")).to_have_text("开始自主分析")
            assert not state["posts"]
            ctx.close()
            cat, _ = ui_fixture()
            cat["matches"][0]["quote_observed_at"] = "2026-10-25T09:00:00+00:00"
            ctx, page, state = fresh(cat)
            ready(page)
            expect(page.locator("#readiness-title")).to_have_text("来源时间需要复核")
            expect(page.locator("#market-freshness")).to_contain_text("时钟异常")
            expect(page.locator("#start")).to_have_text("重新检查数据")
            assert not state["posts"]
            ctx.close()
            checks.append("Initial API failure exposes GET-only reconnect; future quote clock is marked invalid rather than fresh")

            cat, _ = ui_fixture()
            template = cat["matches"][0]
            cat["matches"] = [{**deepcopy(template), "match_id": f"ui-fixture-{i:04d}", "home": f"测试主队 {i:04d}", "status": "scheduled" if i < 900 else "finished", "can_start": i < 900} for i in range(1200)]
            # Target outside the default date/competition scope remains findable by exact ID.
            cat["matches"][-1].update(kickoff="2026-10-29T18:00:00+00:00", competition_key="UI Other League")
            cat["coverage"] = summarize(cat["matches"])
            ctx, page, state = fresh(cat)
            ready(page)
            assert page.locator("#matches .match-row").count() < 30
            expect(page.locator("#coverage-total")).to_have_text("1200")
            expect(page.locator("#matches .match-row").first).to_contain_text("未开赛")
            page.locator("#matches").focus()
            page.keyboard.press("End")
            page.keyboard.press("Enter")
            expect(page.locator("#fixture")).to_contain_text("1198")
            page.locator("#match-search").fill("ui-fixture-1199")
            expect(page.locator("#matches .match-row")).to_have_count(1)
            page.locator("#match-search").press("Enter")
            expect(page.locator("#fixture")).to_contain_text("1199")
            expect(page.locator("#search-mode")).to_contain_text("已忽略日期")
            assert not state["posts"]
            checks.append("1200-row fixture list is virtualized; unstarted games sort first; keyboard End/Enter and cross-filter exact-ID search work without POST")
            ctx.close()

            archive = ROOT / "var/parallel/agent/round2/runs/live-g"
            raw_catalog = json.loads((archive / "catalog.json").read_text(encoding="utf-8"))
            captured = deepcopy(raw_catalog)
            captured.update(schedule_ui_contract="footballmind-daily-schedule-ui-v1", server_time=datetime.now(timezone.utc).isoformat(), provider_configured=False, refreshing=False, runs=[], limitations=LIMITATIONS, model_bindings={})
            for m in captured["matches"]:
                m["can_start"], m["availability_reason"] = availability(m, captured["server_time"])
            captured["coverage"] = summarize(captured["matches"])
            records = {}
            for p in sorted((archive / "runs").glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                record = json.loads(p.read_text(encoding="utf-8"))
                if terminal_record(record):
                    records[record["run_id"]] = record
                    captured["runs"].append({k: record[k] for k in ("run_id", "match", "state", "created_at")})
                if len(records) >= 30:
                    break
            # This view case exercises a quoted, probability-changing first step.
            # A newer valid empty-claim first step is covered as neutral, and must
            # not be indexed as if every successful extraction had claims[0].
            success = next(r for r in records.values() if r.get("steps")
                and r["steps"][0].get("claims")
                and any(sum(abs(a-b) for a,b in zip(s["prediction_before"],s["prediction_after"]))>1e-12
                        for s in r["steps"]))
            failure = next(r for r in records.values() if r["state"] == "FAILED" and r.get("decisions") and r.get("attempts"))
            no_data = next(r for r in records.values() if any(a.get("purpose") == "g_selected_body" and a.get("status") == "NO_DATA" for a in r.get("attempts", [])))
            ctx, page, state = fresh(captured, records)
            ready(page)
            screenshot(page, "01-real-catalog-baseline-and-empty-model")
            page.goto(url + "/live.html?run=" + success["run_id"])
            expect(page.locator("#evidence-count")).to_have_text(f'{len(success["steps"])} 次更新')
            expect(page.locator("#download")).to_be_visible()
            assert page.locator("#probabilities .number").all_text_contents() == [f"{v * 100:.2f}%" for v in success["current"]]
            assert page.locator("#market-probabilities .number").all_text_contents() == [f"{v * 100:.2f}%" for v in success["prior"]]
            page.locator("#replay-select").select_option("0")
            expect(page.locator("#round-replay")).to_contain_text("f 已执行")
            expect(page.locator("#round-replay .pair-table")).to_have_count(1)
            scored = success["decisions"][0]["candidates"]
            highest = max(scored, key=lambda row: row["two_step_net_value"])
            expect(page.locator("#scores .candidate-name").first).to_contain_text(highest["title"])
            expect(page.locator("#scores .candidate-value strong").first).to_have_text(f'{highest["two_step_net_value"]:.6f}')
            screenshot(page, "02-real-run-output-and-decision")
            page.locator("#experiment-content").evaluate("e => { const card=document.getElementById('trace-title').closest('.card'); e.scrollTop=card.offsetTop-e.offsetTop; }")
            screenshot(page, "03-real-round-replay")
            page.locator("#evidence .evidence-record").first.locator(":scope > summary").click()
            first_step = success["steps"][0]
            expect(page.locator("#evidence .evidence-record").first).to_contain_text(first_step["claims"][0]["quotes"][0])
            assert not state["posts"]
            page.locator("#replay-next").click()
            neutral_index = next(i for i, d in enumerate(success["decisions"]) if any(s["action_id"] == d.get("action_id") and not s["claims"] and sum(abs(a-b) for a,b in zip(s["prediction_before"], s["prediction_after"])) < 1e-12 for s in success["steps"]))
            page.locator("#replay-select").select_option(str(neutral_index))
            expect(page.locator("#round-replay")).to_contain_text("概率保持不变")
            expect(page.locator("#round-replay")).to_contain_text("合法陈述 0 条")
            page.locator("#replay-latest").click()
            assert not state["posts"]
            widths = []
            for width in (1920, 1440, 1280, 1100, 960, 768, 760, 390, 320):
                page.set_viewport_size({"width": width, "height": 1000})
                size = page.evaluate("() => ({viewport:innerWidth,width:document.documentElement.scrollWidth,pageHeight:document.documentElement.scrollHeight,clientHeight:document.documentElement.clientHeight})")
                assert size["width"] <= size["viewport"], size
                if width > 1100:
                    assert size["pageHeight"] <= size["clientHeight"] + 2, size
                widths.append(size)
            page.set_viewport_size({"width": 390, "height": 844})
            screenshot(page, "04-real-run-mobile")
            page.locator("#evidence .evidence-record").first.locator(":scope > summary").click()
            page.screenshot(path=str(output / "04-real-run-mobile-viewport.png"), full_page=False)
            screenshots.append(str(output / "04-real-run-mobile-viewport.png"))
            save("responsive-widths.json", widths)
            checks.append("Archived successful run retains exact model probabilities, ranked scores, evidence quotes and input IDs; replay is GET-only; 9 viewport widths fit without horizontal overflow")
            ctx.close()

            for label, record in [("semantic-failure", failure), ("body-no-data", no_data)]:
                ctx, page, state = fresh(captured, records)
                ready(page, "/live.html?run=" + record["run_id"])
                if label == "semantic-failure":
                    page.locator("#replay-select").select_option(str(len(record["decisions"])-1))
                    expect(page.locator("#round-replay")).to_contain_text("after 未生成")
                    expect(page.locator("#round-replay")).to_contain_text(record["reason"])
                else:
                    target = next(i for i, d in enumerate(record["decisions"]) if candidate_has_no_data(record, d))
                    page.locator("#replay-select").select_option(str(target))
                    expect(page.locator("#round-replay")).to_contain_text("NO_DATA")
                    expect(page.locator("#round-replay")).to_contain_text("after 未生成")
                assert not state["posts"]
                page.locator("#experiment-content").evaluate("e => {e.scrollTop=document.getElementById('trace-title').closest('.card').offsetTop-e.offsetTop}")
                screenshot(page, "05-real-" + label)
                ctx.close()
            checks.append("Real semantic failure and body NO_DATA remain missing-after events; neither is rendered as a successful zero-gain update")
            assert not errors, errors
            assert not external_requests, external_requests
            browser.close()
            save("browser-validation.json", {"status": "PASS", "checks": checks, "page_errors": errors,
                "external_requests": external_requests, "source_http": 0, "provider_calls": 0,
                "f_g_executions": 0, "training": False, "screenshots": screenshots,
                "real_catalog_path": str(archive / "catalog.json"),
                "real_run_ids": {"success": success["run_id"], "semantic_failure": failure["run_id"], "body_no_data": no_data["run_id"]},
                "static_sha256": {name: sha256((ROOT / "demo/static" / name).read_bytes()).hexdigest() for name in ("live.html", "live.css", "live.js")}})
            print(json.dumps({"status": "PASS", "checks": len(checks), "screenshots": len(screenshots), "output": str(output)}, ensure_ascii=False))
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def terminal_record(record):
    return record.get("state") in {"STOPPED", "FAILED", "CANCELLED", "INTERRUPTED"}


def candidate_has_no_data(record, decision):
    candidate = next((c for c in decision.get("candidates", []) if c["action_id"] == decision.get("action_id")), None)
    return candidate and any(a.get("purpose") == "g_selected_body" and a.get("action", {}).get("url") == candidate.get("url") and a.get("status") == "NO_DATA" for a in record.get("attempts", []))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)
