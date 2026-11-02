"""Offline Codex-reviewed f recompute for the Everton-Chelsea run.

This is deliberately separate from prospective collection.  It consumes one
already archived Chelsea article, uses a hand-audited empty typed-v3 response,
and never calls a provider.  The output demonstrates the negative-control
path: the body is consumed by the frozen f, but it contains no supported
this-match predicate, so the market prior should remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_research.acquisition.entity_semantic_f import EntityEvidenceBank, EntitySemanticF, _predict
from football_research.acquisition.historical_context_training import _project
from football_research.acquisition.semantic_f import EXTRACTION_VERSION, public_input
from football_research.acquisition.news_link_projection import VERSION as NEWS_PROJECTION


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def main(output: Path, model_path: Path, body_path: Path, article_path: Path,
         prior_path: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    article = json.loads(article_path.read_text(encoding="utf-8"))
    body = body_path.read_text(encoding="utf-8")
    plan = json.loads(prior_path.read_text(encoding="utf-8"))
    selected = plan["selected"][0]
    init = selected["market_initialization"]
    prior = init["probabilities"]
    fixture = {
        "physical_match_id": "footballmind:premier-league:2026-27:2026-10-17:everton:chelsea",
        "match_id": "spftll:match:3003912",
        "home_team_id": "spftll:team:31", "away_team_id": "spftll:team:24",
        "home_name": "Everton", "away_name": "Chelsea",
        "home_aliases": ["Everton", "埃弗顿"], "away_aliases": ["Chelsea", "切尔西"],
        "competition_id": "spftll:competition:36", "competition": "Premier League",
        "kickoff": "2026-10-17T11:30:00Z",
    }
    captured = article.get("captured_at", article.get("response_at", "2026-10-14T08:07:28.782998Z"))
    document = {
        "url": article["url"], "title": article["title"], "text": body,
        "source_published_at": article.get("source_published_at"),
        "captured_at": captured, "available_at": captured,
        "paragraphs": [p for p in body.splitlines() if p.strip()],
    }
    base = {
        "fixture": fixture, "decision_at": "2026-10-16T10:30:00Z",
        "prediction_cutoff": "2026-10-17T10:30:00Z", "information_mode": "operational",
        "allow_unknown_publication_clock": True, "prior_probabilities": prior,
        "prior_basis": "observed_market_prior_research_only", "history": [],
        "documents": [], "observed_links": [{
            "url": document["url"], "title": document["title"],
            "parent_url": document["url"], "title_basis": "observed_parent_link",
            "available_at": captured,
        }],
    }
    # The Codex review found only team-context training imagery; no supported
    # this-match availability/selection/tactics assertion.  Empty claims are
    # a completed, validated extraction, not a fabricated zero value.
    title_input = public_input(_project(base, text_projection=NEWS_PROJECTION)["fixture"], document, "title")
    body_input = public_input(_project(base, text_projection=NEWS_PROJECTION)["fixture"], document, "body")
    records = []
    for channel, item in (("title", title_input), ("body", body_input)):
        records.append({
            "input": item,
            "response": {"schema": EXTRACTION_VERSION, "input_id": item["input_id"], "claims": []},
            "provenance": {
                "kind": "codex_offline_manual_review",
                "reviewer": "Codex GPT local session",
                "provider_http": 0,
                "review_basis": "explicit claims only; no supported this-match predicate",
                "source_body_sha256": sha(body_path), "channel": channel,
            },
        })
    bank = EntityEvidenceBank(records)
    before_projection = bank.project(base)
    after = dict(base)
    after["documents"] = [document]
    after_projection = bank.project(after)
    model = EntitySemanticF.load(model_path)
    before = _predict(model.coefficients, before_projection).tolist()
    after_prediction = _predict(model.coefficients, after_projection).tolist()
    result = {
        "schema": "footballmind-everton-codex-offline-f-recompute-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_negative_control_not_prospective_training",
        "fixture": fixture,
        "market_prior": prior,
        "body": {"path": str(body_path), "sha256": sha(body_path), "characters": len(body),
                 "available_at": captured, "strictly_before_kickoff": True},
        "review": {"provider": "codex_local_manual_review", "provider_http": 0,
                   "claims_returned": 0, "supported_this_match_claims": 0,
                   "review_conclusion": "team_context training gallery; no modeled predicate"},
        "model": {"path": str(model_path), "model_version": model.model_version,
                  "raw_sha256": sha(model_path), "production_ready": False},
        "prediction": {"before": before, "after": after_prediction,
                        "l1_shift": sum(abs(a-b) for a,b in zip(before, after_prediction)),
                        "body_input_id": body_input["input_id"],
                        "body_consumed": body_input["input_id"] in after_projection["used_input_ids"],
                        "before_projection": before_projection,
                        "after_projection": after_projection},
        "training": {"g_training_eligible": False, "reason": "no_settled_outcome_and_offline_review",
                      "realized_gain": None},
        "limitations": ["Does not claim the failed generic runtime consumed a body.",
                         "Codex review is not a DeepSeek provider call.",
                         "No positive live claim was available in this article.",
                         "No outcome is read or used."],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "codex-offline-f-recompute.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    report = ["FootballMind Everton-Chelsea Codex离线 f 重算", "="*64,
              f"正文：{article['title']}", f"正文SHA256：{sha(body_path)}",
              f"body_consumed：{result['prediction']['body_consumed']}",
              f"supported_this_match_claims：0", f"before：{before}", f"after：{after_prediction}",
              f"L1变化：{result['prediction']['l1_shift']:.17g}",
              "结论：正文进入冻结 f，但只有 team_context，未产生可建模本场特征；这是负对照，不是 g 训练行。"]
    (output / "readable-report.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    verification = {"schema": "footballmind-everton-codex-offline-verification-v1",
                    "json_sha256": sha(output/"codex-offline-f-recompute.json"),
                    "body_sha256": sha(body_path), "model_sha256": sha(model_path),
                    "provider_http": 0, "training": False,
                    "body_consumed": result["prediction"]["body_consumed"],
                    "claims": 0, "l1_shift": result["prediction"]["l1_shift"]}
    (output / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--body", type=Path, required=True)
    ap.add_argument("--article", type=Path, required=True)
    ap.add_argument("--prior", type=Path, required=True)
    args = ap.parse_args()
    main(args.output, args.model, args.body, args.article, args.prior)
