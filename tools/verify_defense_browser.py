"""Exercise the real frozen inference UI; block every non-loopback request."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright, expect


def verify(base, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    origin = urlsplit(base).netloc
    errors, external, api_errors, results = [], [], [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
        def route(request):
            u = urlsplit(request.request.url)
            if u.scheme in ("http", "https") and u.netloc != origin:
                external.append(request.request.url)
                request.abort()
            else:
                request.continue_()
        context.route("**/*", route)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda r: api_errors.append({"url": r.url, "status": r.status})
                if "/api/defense/" in r.url and r.status >= 400 else None)
        page.goto(base.rstrip("/") + "/defense.html", wait_until="networkidle")
        expect(page.locator("#run")).to_be_enabled(timeout=60000)
        expect(page.locator("#model-choice")).to_have_value("fulltext_mlp_seed20261027")
        assert "冻结包已校验" in page.locator("#release-status").inner_text()
        case_name = page.locator("#match-name").inner_text()
        case_count = page.locator("#case-count").inner_text()
        page.screenshot(path=str(output / "01-frozen-ready.png"), full_page=True)
        page.locator("#candidates > details").first.locator("summary").click()
        expect(page.locator(".defense-body")).to_be_visible()
        body = page.locator(".defense-body").inner_text()
        assert len(body) > 50, "Body disclosure must contain a real archived article"
        for policy in ("fulltext_mlp_seed20261027", "state_ridge", "fixed_order", "all_evidence"):
            page.select_option("#model-choice", policy)
            with page.expect_response(lambda r: r.url.endswith("/api/defense/runs") and r.request.method == "POST", timeout=60000) as response:
                page.locator("#run").click()
            run = response.value.json()
            assert response.value.status == 200, run
            expect(page.locator("#run")).to_be_enabled()
            assert run["settlement"] is None
            result = run["result"]
            assert result["model_name"] == policy
            # Labels are separate; no outcome/gain is allowed in the inference response.
            def check_no_labels(value):
                if isinstance(value, dict):
                    for k, v in value.items():
                        if k in {"outcome", "outcome_class", "realized_gain", "loss_before", "loss_after"}:
                            assert v is None, (policy, k)
                        check_no_labels(v)
                elif isinstance(value, list):
                    for v in value:check_no_labels(v)
            check_no_labels(result)
            (output / (policy + "-inference.json")).write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
            page.screenshot(path=str(output / (policy + ".png")), full_page=True)
            with page.expect_download() as download:
                page.locator("#download-run").click()
            download.value.save_as(output / (policy + "-download.json"))
            downloaded = json.loads((output / (policy + "-download.json")).read_text(encoding="utf-8"))
            assert downloaded == run
            with page.expect_response(lambda r: r.url.endswith("/settle"), timeout=60000) as settled_response:
                page.locator("#settle").click()
            settled = settled_response.value.json()
            assert settled_response.value.status == 200 and settled["settlement"] is not None
            assert settled["result"] == result
            expect(page.locator("#settlement-card")).to_be_visible()
            (output / (policy + "-settled.json")).write_text(json.dumps(settled, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append({"policy": policy, "steps": len(result["steps"]), "selected_ids": result["selected_ids"],
                            "stop_reason": result["stop_reason"], "inference_unchanged_after_settlement": True,
                            "download_exact": True})
        page.screenshot(path=str(output / "06-settlement.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "07-mobile.png"), full_page=True)
        dimensions = page.evaluate("({width:innerWidth,scroll:document.documentElement.scrollWidth})")
        assert dimensions["scroll"] <= dimensions["width"] + 1, dimensions
        context.close()
        browser.close()
    assert not errors and not external and not api_errors, (errors, external, api_errors)
    report = {"status": "PASS", "tested_at": datetime.now(timezone.utc).isoformat(), "url": base,
              "case_name": case_name, "case_count": case_count, "body_characters": len(body),
              "policies": results, "browser_errors": errors, "external_requests": external,
              "api_errors": api_errors, "mobile_dimensions": dimensions,
              "scope": "Real frozen local inference, explicit settlement and browser rendering; no training or source/provider requests."}
    (output / "browser-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8767")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.base, a.output), ensure_ascii=False, indent=2))
