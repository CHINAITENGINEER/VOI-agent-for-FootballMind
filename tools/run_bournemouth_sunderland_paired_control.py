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
from tools.run_bournemouth_sunderland_f_control import make_claim, sha


def main() -> None:
    root = ROOT / "var/work/fm-today-bournemouth-sunderland-20261018-v1"
    out = root / "frozen-f-paired-controls-v1"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output exists: {out}")
    body_path = root / "football365-open.txt"
    model_path = ROOT / "var/f-entity-semantic-f-shadow-materialization-20261011-v1/model.json"
    body = body_path.read_text(encoding="utf-8")
    prior = json.loads((root / "market-prior.json").read_text(encoding="utf-8"))["implied_probability_normalized"]
    fixture = {"physical_match_id": "footballmind:premier-league:2026-27:2026-10-18:bournemouth:sunderland",
               "match_id": "spftll:match:3003909", "home_team_id": "spftll:team:348", "away_team_id": "spftll:team:65",
               "home_name": "Bournemouth", "away_name": "Sunderland", "home_aliases": ["Bournemouth"],
               "away_aliases": ["Sunderland"], "competition_id": "spftll:competition:36", "competition": "Premier League",
               "kickoff": "2026-10-18T13:00:00Z"}
    url = "https://www.football365.com/match-preview/bournemouth-v-sunderland-prediction-preview"
    title = "Bournemouth v Sunderland: Prediction, team news, lineups and odds"
    available = "2026-10-18T02:27:00Z"
    document = {"url": url, "title": title, "text": body, "source_published_at": "2026-10-17T00:00:00Z",
                "captured_at": available, "available_at": available,
                "paragraphs": [p for p in body.splitlines() if p.strip()]}
    base = {"fixture": fixture, "decision_at": "2026-10-18T02:30:00Z", "prediction_cutoff": "2026-10-18T12:00:00Z",
            "information_mode": "operational", "allow_unknown_publication_clock": True, "prior_probabilities": prior,
            "prior_basis": "observed_market_prior_research_only", "history": [], "documents": [],
            "observed_links": [{"url": url, "title": title, "parent_url": url, "title_basis": "observed_parent_link",
                                "available_at": available}]}
    fixture_for_input = _project(base, text_projection=NEWS_PROJECTION)["fixture"]
    title_input = public_input(fixture_for_input, document, "title")
    body_input = public_input(fixture_for_input, document, "body")
    positive_claims = [
        make_claim("home", "Alex Scott", "availability", "unavailable", "although the midfielder won’t be involved against Sunderland.", body),
        make_claim("home", "Eli Junior Kroupi", "availability", "unavailable", "Eli Junior Kroupi (foot) and Julian Araujo (thigh) also remain sidelined and aren’t expected to play for the Cherries until November.", body),
        make_claim("away", "Thomas Meunier", "availability", "unavailable", "Thomas Meunier and Dan Ballard are still out with their own hamstring injuries, while Romaine Mundle is a long-term absentee after undergoing surgery on a knee injury.", body),
        make_claim("away", "Romaine Mundle", "availability", "unavailable", "Thomas Meunier and Dan Ballard are still out with their own hamstring injuries, while Romaine Mundle is a long-term absentee after undergoing surgery on a knee injury.", body),
        make_claim("away", "Brian Brobbey", "availability", "recovering", "Brian Brobbey and Habib Diarra were back in the matchday squad against Torreense after recovering from hamstring injuries, although they did remain unused substitutes.", body),
    ]
    def rec(item, claims, channel):
        return {"input": item, "response": {"schema": EXTRACTION_VERSION, "input_id": item["input_id"], "claims": claims},
                "provenance": {"kind": "codex_offline_manual_review", "channel": channel, "provider_http": 0,
                                "source_body_sha256": sha(body_path), "reviewer": "Codex GPT local session"}}
    model = EntitySemanticF.load(model_path)
    rows = {}
    for label, claims in (("negative_control", []), ("positive_control", positive_claims)):
        bank = EntityEvidenceBank([rec(title_input, [], "title"), rec(body_input, claims, "body")])
        before_state = base
        after_state = dict(base)
        after_state["documents"] = [document]
        before_proj = bank.project(before_state)
        after_proj = bank.project(after_state)
        before = _predict(model.coefficients, before_proj).tolist()
        after = _predict(model.coefficients, after_proj).tolist()
        shift = float(sum(abs(a - b) for a, b in zip(before, after)))
        rows[label] = {"before": before, "after": after, "l1_shift": shift,
                       "body_consumed": body_input["input_id"] in after_proj["used_input_ids"],
                       "modeled_claims": len(after_proj["novel_body_entity_keys"]),
                       "novel_body_entity_keys": after_proj["novel_body_entity_keys"],
                       "prediction_changed": shift > 1e-15,
                       "before_projection": before_proj, "after_projection": after_proj}
    result = {"schema": "footballmind-f-paired-controls-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
              "mode": "offline_prematch_f_plumbing_control", "fixture": fixture, "market_prior": prior,
              "body": {"path": str(body_path), "sha256": sha(body_path), "available_at": available,
                       "strictly_before_kickoff": True, "source_url": url,
                       "provenance": "Codex web-search projection body; direct official HTTP unavailable"},
              "controls": rows, "model": {"path": str(model_path), "model_version": model.model_version,
                                          "raw_sha256": sha(model_path), "production_ready": False},
              "training": {"g_training_eligible": False, "reason": "unsettled_outcome_offline_control_only", "realized_gain": None},
              "limitations": ["No match result was read.", "The positive labels are quote-checked manual semantic inputs, not empirical truth labels.",
                              "This verifies f plumbing and sensitivity, not production accuracy or g training."]}
    out.mkdir(parents=True, exist_ok=False)
    (out / "paired-controls.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["FootballMind Bournemouth–Sunderland f 正负对照", "=" * 64,
              f"正文SHA256：{sha(body_path)}", "", "负对照（正文进入但无支持 claim）:",
              f"  body_consumed={rows['negative_control']['body_consumed']}, modeled_claims={rows['negative_control']['modeled_claims']}",
              f"  before={rows['negative_control']['before']}", f"  after={rows['negative_control']['after']}",
              f"  L1={rows['negative_control']['l1_shift']:.17g}", "", "正对照（正文进入且含本场 availability claim）:",
              f"  body_consumed={rows['positive_control']['body_consumed']}, modeled_claims={rows['positive_control']['modeled_claims']}",
              f"  before={rows['positive_control']['before']}", f"  after={rows['positive_control']['after']}",
              f"  L1={rows['positive_control']['l1_shift']:.17g}", "", "结论：f 接缝与区分能力通过；尚无赛果，因此不生成 g 训练行。"]
    (out / "readable-report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"schema": result["schema"], "positive_l1": rows["positive_control"]["l1_shift"],
                      "negative_l1": rows["negative_control"]["l1_shift"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
