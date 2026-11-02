"""Offline audit of the schedule -> quote -> startability -> f seam.

This consumes an existing catalog snapshot only. It never refreshes ESPN,
spftll, news, DeepSeek, or the live Agent worker.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.live_g import LiveBackend
from demo.live_coverage import availability
from football_research.acquisition import entity_evidence_updates as f_model


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "var/parallel/agent/round2/runs/live-g/catalog.json"
OUT = ROOT / "var/repairs/live-coverage-chain-audit-20261024-v1"
F_PATH = ROOT / "var/reviews/fm-historical-transition-g-training-20261014-v2/f-labeler.json"


def finite_probability(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def utc(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def quote_valid(match):
    if match.get("source") == "spftll":
        return bool(match.get("quote_observed")) == bool(match.get("inventory_quality", {}).get("complete_1x2_observed", match.get("quote_observed")))
    quote = match.get("quote_snapshot")
    if not match.get("quote_observed"):
        return quote is None or quote == {}
    if not isinstance(quote, dict):
        return False
    probs = quote.get("no_vig_probabilities")
    odds = quote.get("decimal_odds")
    if not isinstance(probs, dict) or not isinstance(odds, dict):
        return False
    if set(probs) != {"home", "draw", "away"} or set(odds) != {"home", "draw", "away"}:
        return False
    return abs(sum(probs.values()) - 1.0) < 1e-9 and all(finite_probability(probs[x]) and probs[x] > 0 for x in probs)


def audit() -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    matches = catalog.get("matches", [])
    server_time = catalog.get("server_time") or catalog.get("updated_at")
    errors, warnings, rows = [], [], []
    counts = {"discovered": len(matches), "quote_observed": 0, "quote_missing": 0,
              "startable": 0, "not_startable": 0, "espn": 0, "spftll": 0,
              "espn_quote": 0, "espn_startable": 0, "f_smoke_passed": 0}
    for match in matches:
        mid = match.get("match_id")
        required = ("match_id", "home", "away", "kickoff", "status", "competition_key", "source")
        missing = [name for name in required if not match.get(name)]
        if missing:
            errors.append({"match_id": mid, "code": "missing_identity_fields", "fields": missing})
        try:
            kickoff = utc(match["kickoff"])
            if kickoff.tzinfo is None:
                raise ValueError("kickoff has no timezone")
        except Exception as exc:
            errors.append({"match_id": mid, "code": "invalid_kickoff", "detail": str(exc)})
            kickoff = None
        is_quote = bool(match.get("quote_observed"))
        counts["quote_observed" if is_quote else "quote_missing"] += 1
        if is_quote and not quote_valid(match):
            errors.append({"match_id": mid, "code": "invalid_quote_snapshot"})
        if match.get("source") == "espn":
            counts["espn"] += 1
            counts["espn_quote"] += int(is_quote)
            if not match.get("raw_observation", {}).get("content_hash"):
                errors.append({"match_id": mid, "code": "espn_quote_missing_raw_digest"})
        elif match.get("source") == "spftll":
            counts["spftll"] += 1
        startable, availability_reason = availability(match, server_time)
        counts["startable" if startable else "not_startable"] += 1
        if match.get("source") == "espn":
            counts["espn_startable"] += int(startable)
        if startable:
            if not (match.get("status") == "scheduled" and is_quote and match.get("supported") is True):
                errors.append({"match_id": mid, "code": "startable_without_required_gate"})
            if kickoff and server_time and kickoff <= utc(server_time):
                errors.append({"match_id": mid, "code": "startable_after_kickoff"})
            if kickoff and server_time and (kickoff - utc(server_time)).total_seconds() <= 1800:
                errors.append({"match_id": mid, "code": "startable_inside_30m_gate"})
        elif match.get("status") == "scheduled" and match.get("supported") is True and is_quote and not availability_reason:
            warnings.append({"match_id": mid, "code": "not_startable_without_reason"})
        rows.append({"match_id": mid, "source": match.get("source"), "competition": match.get("competition_key"),
                     "group": match.get("competition_group"), "status": match.get("status"),
                     "quote_observed": is_quote, "quote_valid": quote_valid(match),
                     "supported": match.get("supported"), "can_start": startable,
                     "reason": availability_reason or match.get("reason"),
                     "kickoff": match.get("kickoff"), "quote_observed_at": match.get("quote_observed_at")})

    # Exercise the ESPN fallback initializer and identity binding against the
    # already archived catalog rows, with no network transport.
    f_model_loaded = f_model.load(F_PATH)
    with tempfile.TemporaryDirectory(prefix="fm-coverage-chain-") as tmp:
        for match in matches:
            if match.get("source") != "espn" or not match.get("quote_observed") or match.get("status") != "scheduled":
                continue
            backend = LiveBackend()
            backend.fixture_scoreboard = lambda fixture, folder, item=match: {"matches": [item]}
            fixture = {"source": "espn", "match_id": match["match_id"], "fixture_id": match["fixture_id"]}
            try:
                initialization = backend._initialize_espn(fixture, Path(tmp), ValueError("offline seam test"))
                init_fixture = initialization["market_initialization"]["fixture"]
                expected = {"source": "espn", "match_id": match["match_id"], "home": match["home"],
                            "away": match["away"], "home_team_source_id": match["home_team_source_id"],
                            "away_team_source_id": match["away_team_source_id"], "competition_key": match["competition_key"],
                            "kickoff": match["kickoff"]}
                binding = backend.identity(initialization, Path(tmp), expected)
                prior = initialization["market_initialization"]["probabilities"]
                prediction = f_model.predict(f_model_loaded, {"physical_match_id": match["match_id"],
                    "decision_at": initialization["market_initialization"]["prediction_cutoff"],
                    "prior": prior, "used_input_ids": [], "evidence_events": []})
                if binding.get("status") != "SOURCE_CONFIRMED" or len(prediction) != 3:
                    raise ValueError("identity or f smoke output invalid")
                counts["f_smoke_passed"] += 1
            except Exception as exc:
                errors.append({"match_id": match.get("match_id"), "code": "espn_fallback_seam_failed",
                               "detail": f"{type(exc).__name__}: {exc}"})
    summary = {"schema": "footballmind-live-coverage-chain-audit-v1", "catalog_path": str(CATALOG),
               "server_time": server_time, "status": "PASS" if not errors else "FAIL",
               "counts": counts, "errors": errors, "warnings": warnings,
               "scope": ["offline catalog validation", "ESPN observed moneyline shape", "startability gates",
                          "ESPN fallback initialization", "ESPN identity binding", "paired f prior smoke"],
               "network_requests": 0, "deepseek_calls": 0, "formal_run_started": False,
               "limitations": ["catalog is a point-in-time snapshot", "source bookmaker event timestamp remains unknown",
                               "this audit does not claim f/g predictive accuracy", "no real Agent run was launched"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage-chain-audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "coverage-chain-table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["match_id"])
        writer.writeheader(); writer.writerows(rows)
    (OUT / "README.md").write_text("# 覆盖比赛链路离线验收\n\n"
        "本包读取已有 catalog.json，不发起网络请求、不调用 DeepSeek、不启动 Agent。\n"
        "逐场检查身份、开球时间、来源、完整 1X2、启动门禁，并对所有 ESPN 完整报价行做初始化→身份→f 先验冒烟验证。\n",
        encoding="utf-8")
    files = []
    for path in sorted(OUT.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        files.append({"path": path.name, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "manifest.json").write_text(json.dumps({"schema": "footballmind-live-coverage-chain-audit-manifest-v1",
        "files": files, "network_requests": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
