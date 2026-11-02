from __future__ import annotations

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


def make_claim(team: str, entity: str, kind: str, value: str, quote: str, body: str) -> dict:
    body.index(quote)
    return {
        "team": team,
        "entity": entity,
        "kind": kind,
        "value": value,
        "scope": "this_match",
        "statement": quote,
        "quotes": [quote],
    }


def main() -> None:
    root = ROOT / "var/work/fm-today-bournemouth-sunderland-20261018-v1"
    out = root / "frozen-f-positive-control-v2"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output exists: {out}")
    body_path = root / "football365-open.txt"
    prior_path = root / "market-prior.json"
    model_path = ROOT / "var/f-entity-semantic-f-shadow-materialization-20261011-v1/model.json"
    body = body_path.read_text(encoding="utf-8")
    prior_doc = json.loads(prior_path.read_text(encoding="utf-8"))
    prior = prior_doc["implied_probability_normalized"]
    fixture = {
        "physical_match_id": "footballmind:premier-league:2026-27:2026-10-18:bournemouth:sunderland",
        "match_id": "spftll:match:3003909",
        "home_team_id": "spftll:team:348", "away_team_id": "spftll:team:65",
        "home_name": "Bournemouth", "away_name": "Sunderland",
        "home_aliases": ["Bournemouth", "伯恩茅斯"], "away_aliases": ["Sunderland", "桑德兰"],
        "competition_id": "spftll:competition:36", "competition": "Premier League",
        "kickoff": "2026-10-18T13:00:00Z",
    }
    url = "https://www.football365.com/match-preview/bournemouth-v-sunderland-prediction-preview"
    title = "Bournemouth v Sunderland: Prediction, team news, lineups and odds"
    available = "2026-10-18T02:27:00Z"
    document = {
        "url": url, "title": title, "text": body,
        "source_published_at": "2026-10-17T00:00:00Z",
        "captured_at": available, "available_at": available,
        "paragraphs": [p for p in body.splitlines() if p.strip()],
    }
    base = {
        "fixture": fixture, "decision_at": "2026-10-18T02:30:00Z",
        "prediction_cutoff": "2026-10-18T12:00:00Z", "information_mode": "operational",
        "allow_unknown_publication_clock": True, "prior_probabilities": prior,
        "prior_basis": "observed_market_prior_research_only", "history": [], "documents": [],
        "observed_links": [{"url": url, "title": title, "parent_url": url,
                            "title_basis": "observed_parent_link", "available_at": available}],
    }
    claims = [
        make_claim("home", "Alex Scott", "availability", "unavailable",
                   "although the midfielder won’t be involved against Sunderland.", body),
        make_claim("home", "Eli Junior Kroupi", "availability", "unavailable",
                   "Eli Junior Kroupi (foot) and Julian Araujo (thigh) also remain sidelined and aren’t expected to play for the Cherries until November.", body),
        make_claim("away", "Thomas Meunier", "availability", "unavailable",
                   "Thomas Meunier and Dan Ballard are still out with their own hamstring injuries, while Romaine Mundle is a long-term absentee after undergoing surgery on a knee injury.", body),
        make_claim("away", "Romaine Mundle", "availability", "unavailable",
                   "Thomas Meunier and Dan Ballard are still out with their own hamstring injuries, while Romaine Mundle is a long-term absentee after undergoing surgery on a knee injury.", body),
        make_claim("away", "Brian Brobbey", "availability", "recovering",
                   "Brian Brobbey and Habib Diarra were back in the matchday squad against Torreense after recovering from hamstring injuries, although they did remain unused substitutes.", body),
    ]
    fixture_for_input = _project(base, text_projection=NEWS_PROJECTION)["fixture"]
    title_input = public_input(fixture_for_input, document, "title")
    body_input = public_input(fixture_for_input, document, "body")
    title_record = {"input": title_input,
                    "response": {"schema": EXTRACTION_VERSION, "input_id": title_input["input_id"], "claims": []},
                    "provenance": {"kind": "codex_offline_manual_review", "channel": "title",
                                    "provider_http": 0, "source_body_sha256": sha(body_path)}}
    body_record = {"input": body_input,
                   "response": {"schema": EXTRACTION_VERSION, "input_id": body_input["input_id"], "claims": claims},
                   "provenance": {"kind": "codex_offline_manual_review", "channel": "body",
                                   "provider_http": 0, "source_body_sha256": sha(body_path),
                                   "reviewer": "Codex GPT local session"}}
    bank = EntityEvidenceBank([title_record, body_record])
    before_projection = bank.project(base)
    after = dict(base)
    after["documents"] = [document]
    after_projection = bank.project(after)
    model = EntitySemanticF.load(model_path)
    before = _predict(model.coefficients, before_projection).tolist()
    after_prediction = _predict(model.coefficients, after_projection).tolist()
    l1 = float(sum(abs(a - b) for a, b in zip(before, after_prediction)))
    result = {
        "schema": "footballmind-bournemouth-sunderland-f-positive-control-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_prematch_f_positive_negative_paired_control",
        "fixture": fixture, "market_prior": prior,
        "source": {"url": url, "title": title, "body_sha256": sha(body_path),
                   "available_at": available, "strictly_before_kickoff": True,
                   "provenance": "Codex web-search projection body; direct official HTTP was unavailable"},
        "claims": {"returned": len(claims), "modeled_this_match": len(after_projection["novel_body_entity_keys"]),
                   "claim_keys": after_projection["novel_body_entity_keys"]},
        "model": {"path": str(model_path), "model_version": model.model_version,
                  "raw_sha256": sha(model_path), "production_ready": False},
        "prediction": {"before": before, "after": after_prediction, "l1_shift": l1,
                        "title_input_id": title_input["input_id"], "body_input_id": body_input["input_id"],
                        "body_consumed": body_input["input_id"] in after_projection["used_input_ids"],
                        "before_projection": before_projection, "after_projection": after_projection},
        "training": {"g_training_eligible": False, "reason": "unsettled_outcome_offline_control_only",
                      "realized_gain": None},
        "limitations": ["This is a frozen-f plumbing/effect control, not a production prediction.",
                         "The source body is a Codex search projection; direct official HTTP was reset.",
                         "Claims are manually quote-checked and not an independent truth label.",
                         "No result was read; no g row is created."],
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "f-positive-control.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "semantic-records.json").write_text(json.dumps({"title": title_record, "body": body_record}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["FootballMind Bournemouth–Sunderland 冻结 f 正例控制", "=" * 64,
              f"赛前正文：{title}", f"正文SHA256：{sha(body_path)}", f"available_at：{available}",
              f"body_consumed：{result['prediction']['body_consumed']}",
              f"modeled_this_match_claims：{result['claims']['modeled_this_match']}",
              f"before：{before}", f"after：{after_prediction}", f"L1变化：{l1:.17g}",
              "训练资格：否（尚未结算；本次只验证 f 接缝与可建模正文响应）。"]
    (out / "readable-report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    verification = {"schema": "footballmind-f-positive-control-verification-v2",
                    "json_sha256": sha(out / "f-positive-control.json"), "body_sha256": sha(body_path),
                    "model_sha256": sha(model_path), "provider_http": 0, "body_consumed": result["prediction"]["body_consumed"],
                    "modeled_claims": result["claims"]["modeled_this_match"], "l1_shift": l1,
                    "prediction_changed": l1 > 1e-15, "training": False}
    (out / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    main()
