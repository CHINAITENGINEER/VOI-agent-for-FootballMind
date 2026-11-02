"""Synthetic end-to-end mechanisms, not real fixture or prediction evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest
import requests

from football_agent import learned_voi_replay
from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.inquiry_tools import OpenWebTools
from football_agent.learned_voi_runtime import LearnedVOIRuntime, _save
from football_agent.match_analysis_report import build_analysis_report, write_analysis_report
from football_research.acquisition.conservative_joint import ConservativeJointReturnPredictor
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_spider.storage.raw_archive import RawArchive
from test_joint_returns import ACTIONS, fitted, state


MATCH = {
    "match_id": "future", "home_team": "Arsenal", "away_team": "Leeds United",
    "away_aliases": ["Leeds"],
    "kickoff_at": "2026-10-10T11:30:00Z",
    "prediction_cutoff": "2026-10-09T11:30:00Z",
}
SEED = "https://www.arsenal.com/news/synthetic-match-preview"
HIDDEN_LINK = "https://www.arsenal.com/news/havertz-exclusive-followup"
PAGE = (
    "<html><main><p>Arsenal vs Leeds United Saturday 10th October 2026.</p>"
    "<p>Arsenal have given no official update on Kai Havertz's fitness ahead of the game.</p>"
    "</main></html>"
)
RSS = (
    '<rss><channel><item><link>' + SEED + '</link>'
    '<title>Synthetic club news entry</title></item></channel></rss>'
).encode()


@pytest.fixture
def synthetic_runner(tmp_path, fitted, monkeypatch):
    """Use the real extractor, archive, planner and sampler with synthetic I/O."""
    model = ConservativeJointReturnPredictor(fitted[0], strength=0.)
    policy = LearnedJointVOIPolicy(model, lambda_cost={"money": 1.})
    project = tmp_path / "synthetic-project"
    freeze = project / "var/parallel/agent/round175/frozen-next-period.json"
    _save(freeze, {"synthetic_mechanism_fixture": True})
    monkeypatch.setattr(learned_voi_replay, "load_frozen_models",
                        lambda *_args, **_kwargs: (None, model))

    def no_network(*_args, **_kwargs):
        raise AssertionError("Synthetic integration and replay must never send HTTP")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)

    def run(name, *, page=PAGE, availability_delay=timedelta(), times=None,
            inject_probability_state=False, news=False, evidence_version=None):
        folder = tmp_path / name
        initial = state()
        initial["prediction_cutoff"] = MATCH["prediction_cutoff"]
        initial["base"]["kickoff"] = MATCH["kickoff_at"]
        generator = EvidenceDocumentCandidates(
            match_terms=[MATCH["home_team"], MATCH["away_team"]],
            seed_urls=[SEED], allowed_hosts=["www.arsenal.com"],
            numeric_actions=ACTIONS, web_cost={"money": .01, "requests": 1.},
            match=MATCH,
            profile={"teams": [{"side": "home", "canonical_name": "Arsenal"}],
                     "entries": [{"kind": "article", "side": "home",
                                  "club": "Arsenal", "url": SEED}]},
        )
        current = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
        recorded_times = iter(times) if times is not None else None

        def clock():
            nonlocal current
            current = next(recorded_times) if recorded_times is not None else current + timedelta(seconds=1)
            return current

        archive = RawArchive(folder / "source-archive")
        if news:
            from football_agent.news_discovery import NewsDiscoveryTools
            tools = NewsDiscoveryTools(archive, ["www.arsenal.com"], extraction_version=3,
                provider="bing_rss", projection_version=2,
                clock=lambda: current + availability_delay)
        else:
            tools = OpenWebTools(archive, ["www.arsenal.com"], extraction_version=3,
                                 clock=lambda: current + availability_delay)
        calls, proposals = [], []

        def propose(public):
            proposals.append(deepcopy(public))
            return generator(public)

        def execute(action, public):
            calls.append(deepcopy(action))
            prepared = tools.prepare(action, {
                "task": {"seed_urls": generator.seed_urls}, "documents": public["documents"]})
            body = RSS if prepared["tool"] == "search_web" else page.encode()
            observation = archive.save(prepared["source"], prepared["endpoint"], prepared["params"],
                                       body, status=200, observed_at=current)
            returned = {"status": "OK", "document": tools.project(observation, prepared),
                        "actual_cost": {"money": None, "requests": 1., "latency_ms": 10., "tokens": 0.},
                        "receipt": {"observation": observation.model_dump(mode="json"), "prepared": prepared}}
            if inject_probability_state:
                returned["state_after"] = deepcopy(public["state"])
                returned["state_after"]["base"]["home_team_id"] = "invented-by-page"
            return returned

        metadata = {"generator": generator.configuration(), "transport": tools.configuration(),
                    "input_bindings": {"frozen_model_config_sha256": sha256(freeze.read_bytes()).hexdigest()}}
        runtime = LearnedVOIRuntime(policy, seed=17,
                                    exploration_limits={"money": .04, "requests": 4.},
                                    max_exploration_actions=4)
        result = runtime.run(initial, run_id="synthetic-" + name,
                             limits={"money": .05, "requests": 5.},
                             propose=propose, execute=execute, output=folder,
                             provider_version="synthetic-archived-body-v1",
                             generator_version=generator.version, clock=clock,
                             run_metadata=metadata, evidence_match=MATCH, evidence_version=evidence_version)
        return {"run": result, "folder": folder, "project": project,
                "calls": calls, "proposals": proposals}

    return run


def test_article_segments_drive_an_actual_followup_and_frozen_replay(synthetic_runner, tmp_path):
    """Raw article -> admitted segment -> new request, never numerical gain."""
    from football_agent.match_evidence import ARTICLES_VERSION
    page = '<html><head><script type="application/ld+json">' + json.dumps({
        "@type": "NewsArticle", "headline": "Arsenal match fitness update",
        "datePublished": "2026-10-06T09:00:00Z", "dateModified": "2026-10-06T10:00:00Z",
        "articleBody": "Arsenal vs Leeds United Saturday 10th October 2026. "
            "Arsenal have given no official update on Kai Havertz's fitness ahead of the game.",
    }) + '</script></head><body></body></html>'
    saved = synthetic_runner("article-v3-grounded", page=page, news=True, evidence_version=ARTICLES_VERSION)
    original = saved["run"]
    assert all("Havertz" not in a.get("query", "") for a in original["cycles"][0]["generated_actions"])
    selected = [c for c in original["cycles"] if c["selected_action"]
                and "Havertz" in c["selected_action"].get("query", "")]
    assert selected, "The acquired article must drive a newly executed grounded query"
    for cycle in selected:
        assert cycle["selection_basis"] == "bounded_unknown_information_exploration"
        assert cycle["execution_status"] == "returned"
        evidence = cycle["public_before"]["evidence"]
        assert evidence["schema_version"] == ARTICLES_VERSION
        proposals = evidence["state"]["followup_candidates"]
        grounded = [p for p in proposals if p["query"] == cycle["selected_action"]["query"]]
        assert grounded and grounded[0]["expected_gain"] is None
        docs = {d["document_id"]: d for d in cycle["public_before"]["documents"]}
        for item in evidence["evidence"]:
            anchor = item["anchor"]
            document = docs[anchor["document_id"]]
            assert document["kind"] == "article"
            assert document["text"][anchor["start"]:anchor["end"]] == anchor["quote"]
    assert original["initial_probabilities"] == original["final_probabilities"]
    assert not original["terminal_selection"].get("stop_certificate")
    write_analysis_report(original, match=MATCH, output=saved["folder"])
    output = tmp_path / "article-v3-replay"
    checked = learned_voi_replay.replay_saved_run(saved["folder"], output=output,
                                                  project_root=saved["project"])
    assert checked["identical_replay"] and checked["replay_network_requests"] == 0
    assert (output / "run.json").read_bytes() == (saved["folder"] / "run.json").read_bytes()


def test_body_creates_grounded_query_then_budgeted_call_and_complete_report(synthetic_runner):
    saved = synthetic_runner("grounded-query")
    run = saved["run"]
    first_menu = run["cycles"][0]["generated_actions"]
    assert all("Havertz" not in action.get("query", "") for action in first_menu)
    followups = [(i, cycle) for i, cycle in enumerate(run["cycles"])
                 if cycle["selected_action"] and "Havertz" in cycle["selected_action"].get("query", "")]
    assert followups, "The acquired gap must become an actually selected new query"
    for index, cycle in followups:
        assert index > 0
        assert "2026-10-10" in cycle["selected_action"]["query"]
        assert cycle["selection_basis"] == "bounded_unknown_information_exploration"
        assert cycle["execution_status"] == "returned"
        evidence = cycle["public_before"]["evidence"]
        proposals = evidence["state"]["followup_candidates"]
        matching = [item for item in proposals if item["query"] == cycle["selected_action"]["query"]]
        assert matching and matching[0]["expected_gain"] is None
        anchors = {item["evidence_id"] for item in evidence["evidence"]}
        assert set(matching[0]["based_on_evidence_ids"]) <= anchors
        assert matching[0]["based_on_evidence_ids"]
        assert any("Kai Havertz" in doc["text"] for doc in cycle["public_before"]["documents"])
    assert run["reserved_cost"]["money"] <= .05
    assert run["reserved_cost"]["requests"] <= 5
    assert run["exploration_actions"] <= 4
    assert run["initial_probabilities"] == run["final_probabilities"]
    assert run["unmodeled_return_probability_update"] is False
    assert all(c["prediction_before"] == c["prediction_after"]
               for c in run["cycles"] if c["execution_status"] == "returned")
    written = write_analysis_report(run, match=MATCH, output=saved["folder"])
    analysis = json.loads(Path(written["analysis_path"]).read_text(encoding="utf-8"))
    assert analysis["status"] == "completed_with_gaps"
    assert analysis["probabilities"]["changed"] is False
    assert analysis["unresolved"] and analysis["followup_candidates"]
    assert any("Kai Havertz" in item["source_quote"] for item in analysis["evidence"])
    assert any(item["modality"] == "uncertain_source_statement" for item in analysis["evidence"])
    assert all(item["calibrated_confidence"] is None for item in analysis["evidence"])
    assert analysis["budget"]["measured_resources"]["money"]["total"] is None
    assert analysis["budget"]["measured_resources"]["requests"]["total"] == len(saved["calls"])
    assert analysis["terminal"]["stop_certificate_present"] is False
    assert "Kai Havertz" in Path(written["report_path"]).read_text(encoding="utf-8")


def test_grounded_run_reprojects_real_archive_bytes_and_replays_without_network(synthetic_runner, tmp_path):
    saved = synthetic_runner("time-aware-replay")
    original = saved["run"]
    assert len({c["public_before"]["decision_at"] for c in original["cycles"]}) > 1
    assert any("Havertz" in c.get("query", "") for c in saved["calls"])
    written = write_analysis_report(original, match=MATCH, output=saved["folder"])
    output = tmp_path / "replayed"
    acceptance = learned_voi_replay.replay_saved_run(saved["folder"], output=output,
                                                     project_root=saved["project"])
    assert acceptance["identical_replay"] is True
    assert acceptance["replay_network_requests"] == 0
    assert acceptance["raw_responses_reprojected"] == len(saved["calls"])
    assert json.loads((output / "run.json").read_text(encoding="utf-8")) == original
    assert (output / "analysis.json").read_bytes() == Path(written["analysis_path"]).read_bytes()
    assert (output / "report.md").read_bytes() == Path(written["report_path"]).read_bytes()
    learning = json.loads((output / "learning-records.json").read_text(encoding="utf-8"))
    assert all(item["estimated_net_voi"] is None for item in learning)
    assert all(item["realized_predictive_gain"] is None for item in learning)
    assert all(item["numeric_voi_training_eligible"] is False for item in learning)


def test_unmodeled_content_cannot_write_probability_state(synthetic_runner):
    saved = synthetic_runner("reject-page-state", inject_probability_state=True)
    run = saved["run"]
    assert len(saved["calls"]) == 1
    assert run["cycles"][0]["execution_status"] == "receipt_rejected"
    assert run["terminal_selection"]["reason"] == "receipt_reconciliation_required"
    assert run["final_state"]["base"]["home_team_id"] == "t0"
    assert run["initial_probabilities"] == run["final_probabilities"]
    assert not run["documents"]
    assert build_analysis_report(run, match=MATCH)["status"] == "incomplete"


def test_future_available_body_cannot_leak_grounded_query_or_discovered_link(synthetic_runner):
    page = PAGE.replace("</main>", '<a href="' + HIDDEN_LINK + '">Arsenal Havertz update</a></main>')
    saved = synthetic_runner("future-evidence", page=page, availability_delay=timedelta(days=1))
    run = saved["run"]
    assert any(doc["kind"] == "page" and doc["links"] for doc in run["documents"])
    assert not run["evidence"]["evidence"]
    assert not run["evidence"]["state"]["followup_candidates"]
    assert all(not public["documents"] for public in saved["proposals"])
    generated = [a for cycle in run["cycles"] for a in cycle["generated_actions"]]
    assert all(a.get("url") != HIDDEN_LINK for a in generated)
    assert all("Havertz" not in a.get("query", "") for a in generated)
    assert len(saved["calls"]) == 2  # Only the initial search and known seed.
    assert run["initial_probabilities"] == run["final_probabilities"]


def test_cutoff_crossing_stops_candidate_generation_and_further_execution(synthetic_runner):
    cutoff = datetime(2026, 10, 9, 11, 30, tzinfo=timezone.utc)
    times = [cutoff - timedelta(seconds=2), cutoff - timedelta(seconds=1),
             cutoff + timedelta(seconds=1), cutoff + timedelta(seconds=2)]
    saved = synthetic_runner("deadline", times=times)
    run = saved["run"]
    assert len(saved["calls"]) == len(saved["proposals"]) == 1
    assert run["cycles"][-1]["generated_actions"] == []
    assert run["cycles"][-1]["selected_action"] is None
    assert run["reserved_cost"]["requests"] == 1
    assert run["terminal_selection"]["reason"] == "forecast_cutoff_elapsed"
    assert run["initial_probabilities"] == run["final_probabilities"]
    analysis = build_analysis_report(run, match=MATCH)
    assert analysis["status"] == "incomplete"
    assert "forecast_cutoff_elapsed" in analysis["incomplete_reasons"]
