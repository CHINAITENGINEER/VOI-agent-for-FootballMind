"""Inspect an actual saved platform/refresh run through its browser replay.

No data or predictions are fabricated; expected display values come from the
same saved report, whose audit has already been verified by the HTTP service.
"""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


def verify(url, run_id, output):
    output.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{url}/api/runs/{run_id}", timeout=20) as response:
        record = json.load(response)
    config = record["tool_configuration"]
    errors, checks = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"{url}/?run={run_id}")
        expect(page.locator("#load-run-id")).to_have_value(run_id)
        expect(page.locator("#workspace")).to_be_visible()
        assert record["information_mode"] in page.inner_text("#run-meta")
        assert record.get("time_mode", "archive_replay") in page.inner_text("#run-meta")
        assert record["reason"] in page.inner_text("#reason")
        if record.get("time_mode") == "live":
            assert "关联子运行" in page.inner_text("#time-mode-description")
        assert config["model_version"] in page.inner_text("#run-model")
        expect(page.locator("#model-id")).to_have_value(config["model_id"])
        expect(page.locator("#match")).to_have_value(record["match_id"])
        checks.append("persisted model, match, execution mode and information mode match saved report")
        assert page.inner_text("#evidence-count") == f"{len(record['evidence'])} 条"
        prediction = record.get("prediction")
        if prediction:
            expect(page.locator("#probabilities")).to_be_visible()
            for ui, field in [("home", "home_win_probability"), ("draw", "draw_probability"), ("away", "away_win_probability")]:
                expect(page.locator(f"#p-{ui}")).to_have_text(f"{prediction[field] * 100:.1f}%")
            checks.append("all three displayed probabilities equal actual PredictionService outputs")
        else:
            expect(page.locator("#probabilities")).to_be_hidden()
            checks.append("missing prediction remains absent")
        related = record.get("child_run_id") or record.get("parent_run_id")
        if related:
            expect(page.locator("#related-runs a").filter(has_text=related)).to_be_visible()
            checks.append("linked parent/child run visible and retained")
        tool_outputs = page.locator("#events button").filter(has_text="工具调用返回")
        if tool_outputs.count():
            tool_outputs.last.click()
            page.locator("#audit-controls button").filter(has_text="查看实际工具返回").click()
            expect(page.locator("#audit-content")).to_be_visible()
            checks.append("actual tool return/raw references available through verified audit blob")
        page.click("#first")
        expect(page.locator("#probabilities")).to_be_hidden()
        assert page.inner_text("#evidence-count") == "0 条"
        expect(page.locator("#audit-content")).to_be_hidden()
        checks.append("initial event hides later evidence, probabilities and tool return")
        page.click("#latest")
        page.screenshot(path=str(output / "run.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert not errors, errors
        browser.close()
    report = {"url": url, "run_id": run_id, "test_mode": record["test_mode"], "model_id": config["model_id"],
              "model_version": config["model_version"], "information_mode": record["information_mode"],
              "time_mode": record.get("time_mode"), "reader_version": config.get("reader_version"),
              "export_sha256": config.get("export_sha256"), "checks": checks, "javascript_errors": errors}
    (output / "browser-run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.url, args.run_id, args.output), ensure_ascii=False, indent=2))
