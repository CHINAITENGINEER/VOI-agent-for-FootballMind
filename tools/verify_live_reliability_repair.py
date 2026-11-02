"""Reproduce stored failures and replay source projections without old-run writes.

Optional provider smoke is a new diagnostic, never an original prospective run
or a g-chosen action. No result labels, training or model changes.
"""
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import live_g as live
from demo.live_sources import LiveNewsTools, profile
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive, json_bytes
from football_research.data_adapters.espn_scoreboard import ESPNScoreboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-smoke", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    store = ROOT / "var/parallel/agent/round2/runs/live-g"
    census = []
    old_hashes = {}
    for path in sorted((store/"runs").glob("*/run.json")):
        old_hashes[str(path)] = sha256(path.read_bytes()).hexdigest()
        r = live.read(path)
        census.append({"run_id": r["run_id"], "match_id": r["match"]["match_id"],
            "home": r["match"].get("home"), "away": r["match"].get("away"),
            "state": r["state"], "reason": r.get("reason"), "counts": r["counts"],
            "body_updates": len(r["steps"]), "source_statuses": [
                {"purpose": a["purpose"], "status": a["status"],
                 "http_status": ((a.get("response") or {}).get("document") or {}).get("http_status")}
                for a in r["attempts"]]})
    live.save(out/"run-census.json", {"runs": census, "total": len(census),
        "without_body_update": sum(not r["body_updates"] for r in census),
        "reasons": dict(Counter(r["reason"] for r in census))})
    gais = live.read(store/"runs/lg-d8ea9397ca9140fab26547d4cae939dd/run.json")
    at = live.utc(gais["created_at"])
    calls = []
    def failed_source(params):
        calls.append(params)
        raise ConnectionError("offline reproduction of original disconnect")
    report = ESPNScoreboard(out/"initialization-replay", transport=failed_source,
        clock=lambda: at).collect_fixture(gais["match"], catalog_root=store/"espn")
    live.save(out/"initialization-replay-result.json", {"network_requests": 0,
        "simulated_transport_attempts": len(calls), "historical_clock": at.isoformat(), "report": report})
    assert report["status"] == "ready", report
    backend = live.LiveBackend()
    backend.fixture_scoreboard = lambda *a: report
    initialization = backend._initialize_espn(gais["match"], out/"initialization-replay", None)
    identity = backend.identity(initialization, out/"initialization-replay", gais["match"])
    # All directory and body projections below reuse this task's already captured
    # raw. Their acquisition clocks are current, not the original failed run's.
    replays = []
    canonical = identity["canonical"]
    fixture = {"physical_match_id": canonical["source_event_id"], "home_name": canonical["home_name"],
               "away_name": canonical["away_name"], "kickoff": canonical["kickoff"]}
    body = None
    candidates = []
    entries = profile(identity)["seed_urls"]
    for path in sorted((out/"source-probes").glob("*/result.json")):
        r = live.read(path)
        obs = Observation.model_validate(r["observation"])
        is_body = path.parent.name == "gais-body"
        tools = LiveNewsTools(RawArchive(path.parent), [urlsplit(obs.endpoint).hostname],
            index_urls=[] if is_body else [obs.endpoint], projection_version=13)
        doc = tools.project(obs, {"tool": "read_url", "endpoint": obs.endpoint, "params": obs.params, "source": "open_page"})
        live.save(path.parent/"reprojected.json", doc)
        menu = []
        for link in doc["links"]:
            reason = live._directory_candidate_reason(link["url"], link["title"], fixture, entries, set())
            menu.append({**link, "admission_reason": reason})
            if reason is None and obs.endpoint in entries:
                candidates.append({**link, "action_id": "article-"+sha256(link["url"].encode()).hexdigest()[:24]})
        replays.append({"source": obs.endpoint, "observation": obs.id, "raw_sha256": obs.content_hash,
            "observed_at": obs.observed_at.isoformat(), "http_status": obs.http_status,
            "old_role": r["document"].get("structured_news", {}).get("role"),
            "new_role": doc.get("structured_news", {}).get("role"), "body_characters": len(doc["text"]) if is_body else 0,
            "body_segments": len(doc.get("structured_news", {}).get("segments", [])), "menu": menu})
        if is_body:
            body = doc
    live.save(out/"projection-replay.json", {"network_requests": 0, "sources": replays})
    assert body and body["structured_news"]["role"] == "article"
    app = live.LiveGApplication(out/"diagnostic-models", backend=backend)
    prior = initialization["market_initialization"]["probabilities"]
    row = {"physical_match_id": fixture["physical_match_id"], "decision_at": live.now(),
           "prior": prior, "used_input_ids": [], "evidence_events": []}
    before = live.f_model.predict(app.f, row).tolist()
    menu = []
    for c in candidates:
        public = {"schema": "historical-prequery-visible-features-v1", "probabilities": before,
            "acquired_documents": 0, "visible_candidates": len(candidates), "article_age_hours": 168., "title": c["title"]}
        menu.append({"action_id": c["action_id"], "query": live.g_features.build_query(public, live.observed_snapshot(row))})
    decision = live.g_model.choose(app.g, menu, 5, .001) if menu else None
    vectors = [live.g_features.vector(app.g["immediate_model"], m["query"]).tolist() for m in menu]
    live.save(out/"actual-g-diagnostic.json", {"scope": "rank_only_current_probe_menu_historical_prior_diagnostic",
        "not_a_new_prospective_run": True, "model_bindings": app.bindings, "candidates": candidates,
        "decision": decision, "request_executed": False, "candidate_body_used_by_g": False,
        "distinct_immediate_input_vectors": len({tuple(v) for v in vectors}),
        "input_vector_dimension": len(vectors[0]) if vectors else 0})
    smoke_path = out/"f-body-smoke.json"
    if args.provider_smoke:
        if smoke_path.exists():
            raise RuntimeError("Provider diagnostic already recorded; do not retry")
        smoke = {"scope": "new_direct_body_diagnostic_not_original_run_not_g_selected",
            "model_bindings": app.bindings, "outcome": None, "gain": None, "g_training_eligible": False,
            "source_observed_at": body["observed_at"], "raw_sha256": body["raw_sha256"],
            "prediction_before": before, "prediction_after": None, "provider_calls": 0}
        public = live.public_input(fixture, {"url": body["url"], "title": body["title"], "text": body["text"]}, "body")
        live.save(out/"f-body-input.json", public)
        live.save(smoke_path, smoke)
        try:
            if live.utc(live.now()) >= live.utc(fixture["kickoff"]):
                raise ValueError("Do not acquire new semantics after kickoff in this diagnostic")
            smoke["provider_calls"] = 1
            live.save(smoke_path, smoke)
            record = backend.semantics(out/"provider-diagnostic", public, 60)
            live.save(out/"semantic-diagnostic.json", record)
            trial = deepcopy(row)
            trial["used_input_ids"].append(public["input_id"])
            for claim in record["response"]["claims"]:
                event = {"physical_match_id": fixture["physical_match_id"], "input_id": public["input_id"],
                    "entity": claim["entity"], "document_url": public["url"], "source_host": urlsplit(public["url"]).hostname,
                    "source_published_at": body.get("source_published_at"), "historical_available_at": None,
                    "claim": claim, "evidence": live.rich.evidence_tokens(claim, "body", fixture)}
                event["event_id"] = sha256(json_bytes(event)).hexdigest()
                trial["evidence_events"].append(event)
            after = live.f_model.predict(app.f, trial).tolist()
            smoke.update(status="passed", body_consumed=True, used_input_ids=trial["used_input_ids"],
                claims=record["response"]["claims"], prediction_after=after,
                l1_shift=sum(abs(a-b) for a,b in zip(before, after)), executed_at=live.now())
            live.save(out/"f-body-state.json", trial)
        except Exception as exc:
            smoke.update(status="failed", error_class=type(exc).__name__,
                error=str(exc)[:240] if isinstance(exc, ValueError) else None, body_consumed=False)
        live.save(smoke_path, smoke)
    assert all(sha256(Path(p).read_bytes()).hexdigest() == h for p,h in old_hashes.items())
    summary = {"old_runs_unchanged": True, "original_runs": len(census),
        "original_without_body_updates": sum(not r["body_updates"] for r in census),
        "gais_initialization_replay": report["status"], "replayed_source_payloads": len(replays),
        "gais_body_characters": len(body["text"]), "gais_body_segments": len(body["structured_news"]["segments"]),
        "gais_menu_candidates": len(candidates), "g_decision": decision,
        "provider_diagnostic": live.read(smoke_path) if smoke_path.exists() else None,
        "trained_models": False, "original_matches_restarted": False, "created_at": live.now()}
    live.save(out/"verification-summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"g_decision","provider_diagnostic"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
