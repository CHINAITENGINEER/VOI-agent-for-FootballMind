"""Browser verification for the actual public-state acquisition workflow."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def verify(url, output, compare=True):
    output.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        expect(page.locator("#compare")).to_be_enabled()
        # Configuration is loaded by an asynchronous fetch; wait before using
        # server-provided match/cutoff defaults (especially for the real bundle).
        expect(page.locator("#capabilities")).not_to_have_text("")
        expect(page.locator("button[type=submit]")).to_be_enabled()
        capabilities = json.loads(page.locator("#capabilities").text_content())
        mode = capabilities.get("mode")
        if mode == "unconfigured":
            assert "未配置" in page.locator("#mode").inner_text()
            page.screenshot(path=str(output / "unconfigured.png"), full_page=True)
            checks.append("Missing real dependency is explicitly labelled; no synthetic substitution")
            report = {"mode": mode, "checks": checks, "javascript_errors": errors, "capabilities": capabilities}
            browser.close()
            (output / "browser.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
        # Dispatch the form event explicitly.  This avoids a transient browser
        # click race while the initial capability fetch releases the button.
        page.locator("#start-form").dispatch_event("submit")
        # Let the asynchronous POST settle before asserting the run panel.
        page.wait_for_timeout(250)
        expect(page.locator("#run")).to_be_visible()
        expect(page.locator("#step")).to_be_enabled()
        initial = json.loads(page.locator("#raw").text_content())
        assert not any(initial["visible_state"]["mask"].values())
        assert initial["visible_state"]["groups"] == {}
        assert page.locator("#probabilities").inner_text().count("—") == 3
        checks.append("Initial public state exposes only empty revealed groups and false mask")
        page.click("#step")
        expect(page.locator("#complete")).to_be_enabled()
        predicted = json.loads(page.locator("#raw").text_content())
        assert predicted["prediction"] is not None
        checks.append("First probability is returned by the partial predictor")
        page.click("#complete")
        expect(page.locator("#reload")).to_be_enabled()
        record = json.loads(page.locator("#raw").text_content())
        assert record["state"] in {"STOPPED", "DEFERRED", "WAITING"}
        expected = record["prediction"]["probabilities"]
        for key in ("home", "draw", "away"):
            assert f"{expected[key] * 100:.2f}%" in page.locator("#probabilities").inner_text()
        assert record["reason"] in page.locator("#run-state").inner_text()
        checks.append("Final displayed probabilities and stop reason equal stored actual outputs")
        assert "SOURCE_NOT_CONNECTED" in page.locator("#candidate-json").text_content() or mode != "synthetic_test"
        page.screenshot(path=str(output / "completed.png"), full_page=True)
        page.locator("#event-position").fill("0")
        page.locator("#event-position").dispatch_event("input")
        assert page.locator("#probabilities").inner_text().count("—") == 3
        assert page.locator("#visible").text_content() == "{}"
        assert page.locator("#candidate-json").text_content() == "[]"
        checks.append("Beginning of replay does not backfill final probabilities, revealed groups or candidates")
        page.click("#play")
        page.wait_for_timeout(1450)
        assert int(page.locator("#event-position").input_value()) >= 1
        page.click("#play")
        page.click("#show-all")
        with page.expect_download() as download:
            page.click("#download")
        download.value.save_as(str(output / "run.json"))
        assert json.loads((output / "run.json").read_text(encoding="utf-8"))["run_id"] == record["run_id"]
        checks.append("Step playback and downloadable persisted public audit are usable")
        page.reload()
        expect(page.locator("#run")).to_be_visible()
        restored = json.loads(page.locator("#raw").text_content())
        assert restored == record
        checks.append("Reload restores the same configuration, cost ledger and run without charging again")
        if compare:
            with page.expect_response(lambda response: response.url.endswith("/api/compare") and response.request.method == "POST") as comparison_response:
                page.click("#compare")
            comparison = comparison_response.value.json()
            (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
            fixed, numeric = comparison["runs"]
            assert fixed["strategy"] == "fixed" and numeric["strategy"] == "all_candidates"
            assert fixed["match_id"] == numeric["match_id"] and fixed["prediction_cutoff"] == numeric["prediction_cutoff"]
            assert fixed["seed"] == numeric["seed"] and fixed["configuration"] == numeric["configuration"]
            assert {k:v for k,v in fixed["budget"].items() if k.startswith("max_")} == {k:v for k,v in numeric["budget"].items() if k.startswith("max_")}
            expect(page.locator("#comparison-panel")).to_be_visible()
            expect(page.locator("#comparison .compare-card")).to_have_count(2)
            checks.append("Fixed-order and all-candidate comparison produces two independent same-budget traces")
        if capabilities.get("llm", {}).get("configured") is False:
            page.select_option("#strategy", "pure_llm")
            page.locator("#start-form").dispatch_event("submit")
            expect(page.locator("#complete")).to_be_enabled()
            page.click("#complete")
            expect(page.locator("#reload")).to_be_enabled()
            missing = json.loads(page.locator("#raw").text_content())
            assert missing["state"] == "DEFERRED" and missing["reason"] == "LLM_NEEDS_CONFIGURATION"
            assert missing["budget"]["llm_calls"] == 0 and missing["budget"]["online_reserve"] == 0
            checks.append("Missing LLM configuration defers without network calls or reserved fees")
            (output / "llm-unconfigured-run.json").write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")
            page.goto(f'{url}/?run={record["run_id"]}')
            expect(page.locator("#run")).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert not errors, errors
        report = {"mode": mode, "run_id": record["run_id"], "state": record["state"], "reason": record["reason"],
                  "configuration": record["configuration"], "checks": checks, "javascript_errors": errors}
        if compare:
            report["comparison"] = [{"run_id": r["run_id"], "strategy": r["strategy"], "reason": r["reason"],
                                     "actions": [item["action_id"] for item in r["history"]],
                                     "simulated_cost": r["budget"]["simulated_cost"]} for r in comparison["runs"]]
        browser.close()
    (output / "browser.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8768")
    parser.add_argument("--output", type=Path, default=Path("var/parallel/agent/round3/browser"))
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.url, args.output, not args.no_compare), ensure_ascii=False, indent=2))
