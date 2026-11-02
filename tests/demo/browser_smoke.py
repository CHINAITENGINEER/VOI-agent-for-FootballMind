"""Optional real-browser verification against ``python -m demo``.

Uses the already installed Playwright runtime. It is deliberately separate from
the dependency-free demo and from the regular pytest suite.
"""

import argparse
import json
from pathlib import Path
import re

from playwright.sync_api import expect, sync_playwright


def verify(url: str, output: Path, browser_path: str | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    with sync_playwright() as playwright:
        options = {"headless": True}
        if browser_path:
            options["executable_path"] = browser_path
        browser = playwright.chromium.launch(**options)
        page = browser.new_page(viewport={"width": 1440, "height": 1150})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        expect(page.locator("#reload-matches")).to_be_enabled()
        page.select_option("#mode", "test")
        expect(page.locator("#match")).to_have_value("test:match-001")
        expect(page.locator("#start")).to_be_enabled()
        assert page.input_value("#cutoff").startswith("2026-09-20T19:30")
        page.click("#start")
        expect(page.locator("#workspace")).to_be_visible()
        expect(page.locator("#run-all")).to_be_enabled()
        page.click("#run-all")
        expect(page.locator("#state")).to_have_text("已停止")
        assert "测试模式" in page.inner_text("#mode-badge")
        assert page.inner_text("#p-home") == "47.0%"
        page.screenshot(path=str(output / "synthetic-completed.png"), full_page=True)
        checks.append("synthetic normal: tools returned 47.0% and explicit test-mode label")
        page.click("#first")
        assert page.locator("#probabilities").is_hidden()
        assert page.inner_text("#evidence-count") == "0 条"
        assert "0 / 20" in page.inner_text("#budget-stats")
        checks.append("first event: no future probability/evidence/cost leaks")
        page.click("#play")
        page.wait_for_timeout(1300)
        assert "第 2 /" in page.inner_text("#playhead")
        page.click("#play")
        page.click("#latest")
        assert page.inner_text("#p-home") == "47.0%"
        checks.append("replay controls: first, automatic play, pause and latest")
        with page.expect_download() as download:
            page.click("#download")
        download.value.save_as(str(output / "synthetic-report.json"))
        assert json.loads((output / "synthetic-report.json").read_text(encoding="utf-8"))["test_mode"] is True
        checks.append("downloaded complete JSON report with events and test label")

        page.select_option("#mode", "real")
        expect(page.locator("#reload-matches")).to_be_enabled()
        options = page.locator("#match option").evaluate_all("items => items.map(item => item.value)")
        real = None
        if "spftll:3013707" in options:
            page.select_option("#match", "spftll:3013707")
            page.fill("#cutoff", "2026-09-20T19:10")
            page.click("#start")
            expect(page.locator("#state")).to_have_text("已初始化")
            expect(page.locator("#run-all")).to_be_enabled()
            page.click("#run-all")
            expect(page.locator("#state")).to_have_text(re.compile("^(已停止|暂缓判断|等待中)$"))
            assert "测试模式" not in page.inner_text("#mode-badge")
            assert "零联网" in page.inner_text("#run-meta")
            assert page.locator("#probabilities").is_visible()
            real = {"probability_display": page.locator("#probabilities").inner_text(), "state": page.inner_text("#state"), "reason": page.inner_text("#reason")}
            with page.expect_download() as download:
                page.click("#download")
            download.value.save_as(str(output / "real-report.json"))
            report = json.loads((output / "real-report.json").read_text(encoding="utf-8"))
            assert report["test_mode"] is False
            assert report["prediction"]["model_version"]
            checks.append("real retained archive + actual market model + report download")
            saved_id = report["run_id"]
            page.reload()
            expect(page.locator("#load-run-id")).to_have_value(saved_id)
            expect(page.locator("#probabilities")).to_be_visible()
            assert page.inner_text("#evidence-count") == f"{len(report['evidence'])} 条"
            checks.append("persisted run reopens from its URL after page reload")
            page.screenshot(path=str(output / "real-completed.png"), full_page=True)
            page.click("#first")
            assert page.locator("#probabilities").is_hidden()
            assert page.inner_text("#evidence-count") == "0 条"
            checks.append("real historical first event has no later evidence or probability")
        else:
            checks.append("real selected fixture absent; real acceptance remains unverified")
        trained = None
        page.select_option("#model-id", "frequency-england")
        expect(page.locator("#reload-matches")).to_be_enabled()
        page.fill("#cutoff", "2026-09-20T19:25")
        page.locator("#cutoff").press("Tab")
        expect(page.locator("#reload-matches")).to_be_enabled()
        options = page.locator("#match option").evaluate_all("items => items.map(item => item.value)")
        if "spftll:3003895" in options:
            page.select_option("#match", "spftll:3003895")
            page.fill("#cutoff", "2026-09-20T19:25")
            page.locator("#cutoff").press("Tab")
            expect(page.locator("#reload-matches")).to_be_enabled()
            page.click("#start")
            expect(page.locator("#state")).to_have_text("已初始化")
            page.click("#run-all")
            expect(page.locator("#state")).to_have_text(re.compile("^(已停止|暂缓判断|等待中)$"))
            expect(page.locator("#probabilities")).to_be_visible()
            assert page.inner_text("#p-home") == "47.8%"
            assert "league-frequency-1.0" in page.inner_text("#run-model")
            assert "2026/9/20 19:22:02" in page.inner_text("#run-model")
            assert "operational" in page.inner_text("#run-meta")
            assert "archive_replay" in page.inner_text("#run-meta")
            with page.expect_download() as download:
                page.click("#download")
            download.value.save_as(str(output / "trained-report.json"))
            trained = json.loads((output / "trained-report.json").read_text(encoding="utf-8"))
            assert trained["tool_configuration"]["model_id"] == "frequency-england"
            assert trained["tool_configuration"]["artifact_sha256"]
            trained = {"run_id": trained["run_id"], "model_id": trained["tool_configuration"]["model_id"],
                       "model_version": trained["prediction"]["model_version"], "home_probability": trained["prediction"]["home_win_probability"]}
            page.reload()
            expect(page.locator("#load-run-id")).to_have_value(trained["run_id"])
            expect(page.locator("#model-id")).to_have_value("frequency-england")
            expect(page.locator("#match")).to_have_value("spftll:3003895")
            assert "league-frequency-1.0" in page.inner_text("#selected-model")
            expect(page.locator("#probabilities")).to_be_visible()
            assert "league-frequency-1.0" in page.inner_text("#run-model")
            checks.append("server catalog selection: real frequency model, trained cutoff/hash, exact model restored after reload")
            page.screenshot(path=str(output / "trained-completed.png"), full_page=True)
            page.locator("#events button").filter(has_text="工具调用返回").first.click()
            page.locator("#audit-controls button").filter(has_text="查看实际工具返回").click()
            expect(page.locator("#audit-content")).to_be_visible()
            assert '"status": "OK"' in page.inner_text("#audit-content")
            page.click("#first")
            expect(page.locator("#audit-content")).to_be_hidden()
            checks.append("actual hashed tool output and raw references open from selected event; hidden when rewinding")
        else:
            checks.append("trained real fixture absent; trained browser path remains unverified")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        checks.append("390px mobile viewport has no horizontal overflow")
        assert not errors, errors
        browser.close()
    result = {"url": url, "checks": checks, "real": real, "trained": trained, "javascript_errors": errors}
    (output / "browser-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path("var/parallel/agent/round2/demo"))
    parser.add_argument("--browser-path", default=None, help="Optional installed Chromium/Edge executable; defaults to Playwright's bundled browser")
    args = parser.parse_args()
    print(json.dumps(verify(args.url, args.output, args.browser_path), ensure_ascii=False, indent=2))
