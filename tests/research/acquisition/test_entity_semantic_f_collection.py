"""Actual frozen f + local mocked sources/extractor: never empirical B data."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from tests.research.acquisition.test_verified_collection import verified_inputs
from tests.research.acquisition.test_f_only_journal_collection import freeze_collector_clocks
from football_agent.entity_semantic_f_only_policy import (
    EntitySemanticFOnlyPolicy, CONFIG_SCHEMA, PRIOR_BASIS, TRACE_FILE,
)
from football_agent.historical_f_only_policy import fixture_from_verified_identity
from football_agent.learned_voi_replay import replay_saved_run
from football_agent.news_discovery import NewsDiscoveryTools
from football_research.acquisition.entity_semantic_f import EntitySemanticF
from football_research.acquisition.semantic_f import SourceAssertionProtocol
from football_research.data_adapters.prospective_text import MarketPriorPolicy, collect_prepared_batch

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "var/f-entity-semantic-f-shadow-materialization-20261011-v1/model.json"
QUOTE = "Spain confirm Alex Test is unavailable against Croatia."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("This is an offline integration test, no network or g calls")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    from football_research.acquisition.historical_context_training import HistoricalContextModel
    monkeypatch.setattr(HistoricalContextModel, "estimate", forbidden)
    if not MODEL.is_file():
        pytest.skip("Independently retained research-shadow model is absent")


def specification(inputs, *, max_calls=32):
    return {"schema": CONFIG_SCHEMA, "model_sha256": sha256(MODEL.read_bytes()).hexdigest(),
        "frozen_at": inputs["now"].isoformat(), "prior_basis": PRIOR_BASIS,
        "provider": {"kind": "deepseek_api", "model": "offline-test-extractor",
            "thinking": "disabled", "max_tokens": 4096, "timeout_seconds": 30},
        "max_calls": max_calls}


def collect_entity(inputs, monkeypatch, *, mode="ok", status="OK", max_calls=32,
                   protocol_version="v1", language_links=False, unknown_publication=False):
    root = inputs["root"]
    tick = freeze_collector_clocks(monkeypatch, inputs["now"] + timedelta(seconds=1))
    import football_agent.entity_semantic_f_only_policy as module
    import football_research.data_adapters.prospective_text as collector
    monkeypatch.setattr(module, "datetime", collector.datetime)
    cutoff = datetime.fromisoformat(MarketPriorPolicy.from_manifest(inputs["plan"]["selected"][0]).model.cutoff)
    # The fixture and extraction are synthetic test inputs. Model bytes are real.
    body = inputs["body"].replace(b"Spain will assess player fitness before the match.", QUOTE.encode())
    body = body.replace(b"<html>", b"<html><head><title>Croatia vs Spain preview</title></head>")
    if unknown_publication:
        # A regular article with an omitted clock is valid operational input.
        # Liveblog updates intentionally retain their per-update clock rule.
        article = {"@type": "NewsArticle", "headline": "Croatia vs Spain preview",
                   "url": "https://www.uefa.com/uefanationsleague/match/2046001--croatia-vs-spain/",
                   "articleBody": "Croatia may rotate their starting lineup against Spain. " + QUOTE}
        body = ('<html><head><title>Croatia vs Spain preview</title></head>'
                '<script type="application/ld+json">' + json.dumps(article) + '</script></html>').encode()
    if language_links:
        controls = ''.join('<a href="https://www.uefa.com/news/test-team-news/">'+label+'</a>'
                           for label in ('English', 'ع', 'Español', 'Thai'))
        assert b"</html>" in body
        body = body.replace(b"</html>", controls.encode('utf-8') + b"</html>")
    protocol = SourceAssertionProtocol
    if protocol_version == "v2":
        from football_agent.semantic_request_binding import RequestBoundSourceAssertionProtocol
        protocol = RequestBoundSourceAssertionProtocol
    elif protocol_version == "v3":
        from football_agent.semantic_request_binding import TypedRequestBoundSourceAssertionProtocol
        protocol = TypedRequestBoundSourceAssertionProtocol
    elif protocol_version == "v4":
        from football_agent.semantic_request_binding import TypedRequestBoundSourceAssertionProtocol
        protocol = TypedRequestBoundSourceAssertionProtocol
        from football_agent.entity_semantic_f_only_policy import REPAIR_REQUEST_CONFIG_SCHEMA
        repair_protocol = __import__(
            "football_agent.semantic_request_binding",
            fromlist=["RepairTypedRequestBoundSourceAssertionProtocol"],
        ).RepairTypedRequestBoundSourceAssertionProtocol
    else:
        repair_protocol = None
    calls = []

    class FakeProvider:
        def __init__(self, *a, **kw):
            self.is_repair = protocol_version == "v4" and kw["protocol"] is repair_protocol
            assert kw["protocol"] is (repair_protocol if self.is_repair else protocol)

        def configuration(self):
            return {"configured": mode != "unconfigured"}

        def decide(self, public):
            fields = {"schema", "channel", "fixture", "url", "title", "text"}
            assert set(public) == (fields | {"input_id"} if protocol_version == "v1" else fields)
            assert set(public["fixture"]) == {"physical_match_id", "home_name", "away_name", "kickoff"}
            calls.append(deepcopy(public))
            tick[0] += timedelta(milliseconds=100)
            claims = []
            if public["channel"] == "body" and QUOTE in public["text"]:
                if mode == "failure":
                    raise TimeoutError("Secret exception text must not be archived")
                if mode == "late_provider":
                    tick[0] = cutoff + timedelta(seconds=1)
                claims = [{"team": "away", "entity": "Alex Test", "kind": "availability",
                    "value": "unavailable", "scope": "this_match", "statement": QUOTE,
                    "quotes": [QUOTE if mode != "bad_quote" else "Invented quote that was not acquired."]}]
                if protocol_version == "v4" and not self.is_repair:
                    # Deliberately violate the frozen kind/value contract on
                    # the first provider response. The v4 policy must retain
                    # this failure and make one separate, typed repair call.
                    claims = [{"team": "away", "entity": "Alex Test", "kind": "tactical_plan",
                        "value": "unmodeled", "scope": "this_match", "statement": QUOTE,
                        "quotes": [QUOTE]}]
            response = {"schema": protocol.version, "claims": claims}
            if protocol_version == "v1":
                response["input_id"] = public["input_id"]
            return {"response": response,
                "receipt": {"fixture": "synthetic_test_only", "http_requests": 0}}

    from football_agent import deepseek_provider
    monkeypatch.setattr(deepseek_provider, "DeepSeekInquiryProvider", FakeProvider)
    if mode in {"late_local_inference", "late_initial_inference"}:
        original = EntitySemanticFOnlyPolicy._forecast
        def slow(self, state, visible, skipped):
            result = original(self, state, visible, skipped)
            if state["documents"] or mode == "late_initial_inference":
                tick[0] = cutoff + timedelta(seconds=1)
            return result
        monkeypatch.setattr(EntitySemanticFOnlyPolicy, "_forecast", slow)

    def execute(self, prepared, *, run_id, request_id):
        start = tick[0]
        tick[0] += timedelta(milliseconds=100)
        payload = (b"<rss><channel></channel></rss>" if prepared["tool"] == "search_web" else body)
        if status != "OK":
            payload = b""
        obs = self.archive.save(prepared["source"], prepared["endpoint"], prepared["params"], payload,
            status=503 if status == "FAILED" else 200, observed_at=tick[0],
            context={"requested_at": start.isoformat()})
        return {"document": self.project(obs, prepared), "observation": obs.model_dump(mode="json"),
            "http_attempts": 1, "money": None, "money_status": "not_measured", "latency_seconds": .1}

    monkeypatch.setattr(NewsDiscoveryTools, "execute", execute)
    spec = specification(inputs, max_calls=max_calls)
    if protocol_version == "v2":
        from football_agent.entity_semantic_f_only_policy import REQUEST_BOUND_CONFIG_SCHEMA
        spec["schema"] = REQUEST_BOUND_CONFIG_SCHEMA
    elif protocol_version == "v3":
        from football_agent.entity_semantic_f_only_policy import TYPED_REQUEST_CONFIG_SCHEMA
        spec["schema"] = TYPED_REQUEST_CONFIG_SCHEMA
    elif protocol_version == "v4":
        from football_agent.entity_semantic_f_only_policy import REPAIR_REQUEST_CONFIG_SCHEMA
        spec["schema"] = REPAIR_REQUEST_CONFIG_SCHEMA
        spec["repair"] = {"enabled": True, "max_attempts": 1,
            "trigger": "typed_validation_failure_only",
            "prompt_version": "request-bound-football-source-assertions-v3-repair-v1"}
    (root / "semantic-config.json").write_text(json.dumps(spec), encoding="utf-8")
    report = collect_prepared_batch(root / "plan.json", output=root / "batch", archive_root=root / "market",
        allowed_hosts=["www.uefa.com"], identity_bundle=root / "bindings.json", identity_archive=root / "identity",
        seed_urls=["https://www.uefa.com/uefanationsleague/news/test-team-news/"] if max_calls == 1 else [],
        news_provider="bing_rss", source_profile_version=1, news_projection_version=4,
        evidence_version="match-evidence-extraction-v5", candidate_discovery_version=7,
        requests_per_match=3, seed=20261010, policy_arm="fixed_order",
        policy_arm_version="footballmind-policy-arm-v2", journal_enabled=True,
        f_only_predictor_path=MODEL, f_only_semantic_config_path=root / "semantic-config.json")
    folder = root / "batch/match-101"
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    journal = json.loads(next((root / "batch/acquisition-journal").glob("*/run.json")).read_text(encoding="utf-8"))
    return folder, run, journal, report, calls


def replay_without_provider(folder, inputs, monkeypatch):
    from football_agent import deepseek_provider
    monkeypatch.setattr(deepseek_provider, "DeepSeekInquiryProvider",
        lambda *a, **k: pytest.fail("Replay may not instantiate a provider"))
    result = replay_saved_run(folder, output=inputs["root"] / "replay", project_root=ROOT)
    assert result["identical_replay"] and result["replay_network_requests"] == 0
    assert (folder / "run.json").read_bytes() == (inputs["root"] / "replay/run.json").read_bytes()
    assert (folder / TRACE_FILE).read_bytes() == (inputs["root"] / "replay" / TRACE_FILE).read_bytes()


def exported_quality_rows(folder, inputs, *, with_result=True):
    """A synthetic independent settlement exercises the label join, not B data."""
    from football_research.data_adapters.prospective_stream import export_batch
    from football_research.data_adapters.prospective_outcomes import project_result_rows, derive_g_candidates
    from football_spider.storage.raw_archive import RawArchive
    from football_platform.information.stream_validator import validate_stream
    output = inputs["root"] / "test-only-stream"
    export_batch(folder.parent, output, mode="prospective_capture")
    report = validate_stream(output)
    assert report["valid"], report
    def rows(name):
        return [json.loads(line) for line in (output / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    fixtures = rows("fixtures.jsonl")
    result_rows = []
    if with_result:
        kickoff = datetime.fromisoformat(fixtures[0]["kickoff_at"])
        observed = kickoff + timedelta(hours=3)
        archive = RawArchive(inputs["root"] / "test-only-results")
        obs = archive.save("independent_test_result", "https://result.example.test/fixture", {},
            b'{"synthetic_test_only":true,"home_goals":1,"away_goals":0}', status=200, observed_at=observed)
        result_rows, audit = project_result_rows([{
            "match_id": fixtures[0]["physical_match_id"], "kickoff": kickoff.isoformat(),
            "home_goals": 1, "away_goals": 0, "status": "final", "score_period": "regulation",
            "observed_at": observed.isoformat(), "independent_outcome_verified": True,
            "input_manifest": [f"observation:{obs.id}|sha256:{obs.content_hash}"]}],
            result_archive=archive.root, output_root=output)
        assert audit["accepted"] == 1
    attempts = rows("attempts.jsonl")
    for attempt in attempts:
        # Actual currency conversion is deliberately unavailable to this
        # quality-target test. No simulated request price can stand in for it.
        attempt["monetary_cost_if_known"] = None
    derived = derive_g_candidates(fixtures=fixtures, discoveries=rows("discovery.jsonl"),
        attempts=attempts, contexts=rows("context_snapshots.jsonl"), results=result_rows,
        documents=rows("documents.jsonl"), archive_root=output)
    (inputs["root"] / "test-only-quality-targets.json").write_text(
        json.dumps({"synthetic_test_only": True, "rows": derived}, indent=2), encoding="utf-8")
    return derived


def test_entity_f_consumes_revealed_text_once_and_journals_before_after(verified_inputs, monkeypatch):
    folder, run, journal, report, calls = collect_entity(verified_inputs, monkeypatch)
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    manifest = config["policy_manifest"]
    assert manifest["prior_domain_shift"] == "training_poisson_to_market_not_calibrated"
    assert manifest["extraction_protocol"]["version"] == SourceAssertionProtocol.version
    assert manifest["bundled_g_enabled"] is False and not (folder / "context-value-model.json").exists()
    assert run["initial_probabilities"] != run["final_probabilities"]
    updates = [c for c in run["cycles"] if (c.get("update") or {}).get("probability_changed")]
    assert updates and any(c["public_after"]["prediction"]["modeled_read_documents"] > 0 for c in updates)
    assert len(calls) == len({p["input_id"] for p in calls})
    assert any(p["channel"] == "body" and QUOTE in p["text"] for p in calls)
    for cycle in updates:
        assert cycle["returned_at"] < cycle["update"]["as_of"]
        assert cycle["semantic_prediction"]["available_at"] == cycle["update"]["as_of"]
        assert cycle["after_state_timing"]["captured_at"] >= cycle["update"]["as_of"]
        assert cycle["after_state_timing"]["prediction_processing"]["effective_at"] == cycle["update"]["as_of"]
    assert journal["state"] == "STOPPED" and journal["model_status"] == "PREDICTOR_ONLY"
    assert journal["candidate_expected_gains"] is None
    assert any(s["prediction_before"] != s["prediction_after"] for s in journal["context_snapshots"]
               if s["snapshot_kind"] == "after_action")
    assert run["outcome"] is None and run["numeric_voi_training_eligible"] is False
    for cycle in run["cycles"]:
        for candidate in cycle["strict_decision"]["candidates"]:
            if candidate.get("value"):
                assert candidate["value"]["expected_signed_gain"] is None
                assert candidate["value"]["net_voi"] is None
    replay_without_provider(folder, verified_inputs, monkeypatch)


def test_v4_repairs_one_typed_validation_failure_without_local_coercion(verified_inputs, monkeypatch):
    """The repair branch is bounded, separately typed, and exactly replayable."""
    folder, run, journal, report, calls = collect_entity(
        verified_inputs, monkeypatch, protocol_version="v4")
    trace = json.loads((folder / TRACE_FILE).read_text(encoding="utf-8"))
    body_attempts = [attempt for event in trace["events"] for attempt in event["attempts"]
                     if attempt["input"]["channel"] == "body"]
    assert len(body_attempts) == 1
    attempt = body_attempts[0]
    assert attempt["status"] == "OK"
    assert attempt["provider_calls"] == 2
    assert attempt["repair_status"] == "validated"
    assert attempt["initial_failure"]["status"] == "FAILED"
    assert attempt["initial_failure"]["error_type"] in {"ValueError", "ValidationError"}
    assert attempt["initial_failure"]["failure_stage"] == "validation"
    assert attempt["initial_failure"]["completed_at"] == attempt["initial_completed_at"]
    assert attempt["repair_attempt"]["status"] == "OK"
    assert attempt["repair_attempt"]["provider_input"]["schema"] == (
        "request-bound-football-source-assertions-v3")
    # The invalid first response is retained as evidence; the repaired
    # response is what passes the normal validator and reaches f.
    assert attempt["initial_provider_response"]["claims"][0]["value"] == "unmodeled"
    assert attempt["initial_provider_response"]["claims"][0]["kind"] == "tactical_plan"
    assert attempt["response"]["claims"][0]["kind"] == "availability"
    assert run["initial_probabilities"] != run["final_probabilities"]
    assert journal["model_status"] == "PREDICTOR_ONLY"
    assert run["numeric_voi_training_eligible"] is False
    assert len([call for call in calls if call["channel"] == "body"]) == 2
    replay_without_provider(folder, verified_inputs, monkeypatch)


def test_entity_f_consumes_body_when_publication_clock_is_unknown_but_available(verified_inputs, monkeypatch):
    folder, run, journal, report, calls = collect_entity(
        verified_inputs, monkeypatch, unknown_publication=True)
    assert any(call["channel"] == "body" and QUOTE in call["text"] for call in calls)
    assert run["initial_probabilities"] != run["final_probabilities"]
    modeled = [cycle["public_after"]["prediction"]["modeled_read_documents"]
               for cycle in run["cycles"] if cycle.get("public_after")]
    assert max(modeled, default=0) > 0
    audits = [cycle["public_after"]["prediction"]["text_projection_audit"]
              for cycle in run["cycles"] if cycle.get("public_after")]
    assert all(item.get("reason") != "unknown_publication_time"
               for audit in audits for item in audit["unmodeled_documents"])
    assert any(document.get("source_published_at") is None
               for document in run["documents"] if document.get("status") == "OK")
    replay_without_provider(folder, verified_inputs, monkeypatch)


def test_real_f_pair_can_form_quality_target_without_inventing_costs(verified_inputs, monkeypatch):
    from math import log
    folder, *_ = collect_entity(verified_inputs, monkeypatch)
    rows = exported_quality_rows(folder, verified_inputs)
    eligible = [row for row in rows if row["g_quality_target"]["status"] == "eligible"]
    assert len(eligible) == 1, [
        {"attempt": row["attempt_id"], "target": row["g_quality_target"],
         "pair": row["predictor_pair_audit"]} for row in rows]
    # The two empty search responses have no admitted document. Preserve their
    # exclusions rather than claiming an observed zero-value text response.
    assert all(row["g_quality_target"]["realized_signed_gain"] is None
               and "document_not_admitted" in row["g_quality_target"]["exclusion_reason"]
               for row in rows if row not in eligible)
    assert eligible[0]["g_quality_target"]["realized_signed_gain"] != 0
    for row in eligible:
        target = row["g_quality_target"]
        assert row["predictor_pair_audit"]["valid"], row["predictor_pair_audit"]
        expected = log(row["prediction_after"]["home"]) - log(row["prediction_before"]["home"])
        assert target["realized_signed_gain"] == pytest.approx(expected)
        assert target["prediction_pipeline_version"].startswith("entity-market-f:")
        assert target["cost_subtracted"] is False and target["production_ready"] is False
        assert row["label_status"] == "ineligible" and row["net_gain"] is None
        assert row["exclusion_reason"] == ["acquisition_cost_unknown"]


def test_no_outcome_produces_no_quality_label(verified_inputs, monkeypatch):
    folder, *_ = collect_entity(verified_inputs, monkeypatch)
    rows = exported_quality_rows(folder, verified_inputs, with_result=False)
    assert rows and all(row["g_quality_target"]["status"] == "ineligible" for row in rows)
    assert all(row["g_quality_target"]["realized_signed_gain"] is None for row in rows)


@pytest.mark.parametrize("mode,reason", [
    ("failure", "semantic_extraction_failed"), ("bad_quote", "semantic_extraction_failed"),
    ("late_provider", "semantic_prediction_deadline_elapsed"),
    ("late_local_inference", "semantic_prediction_deadline_elapsed"),
])
def test_semantic_failure_and_lateness_retain_valid_forecast(verified_inputs, monkeypatch, mode, reason):
    folder, run, journal, report, calls = collect_entity(verified_inputs, monkeypatch, mode=mode)
    assert run["terminal_selection"]["reason"] == reason
    assert run["cycles"][-1]["update"]["status"] == "not_updated"
    assert run["cycles"][-1]["update"]["prediction_retained"]
    assert run["initial_probabilities"] == run["final_probabilities"]
    assert journal["state"] == "STOPPED" and journal["pending_request_id"] is None
    assert journal["budget"]["reserved_requests"] == 0
    assert "Secret exception" not in (folder / TRACE_FILE).read_text(encoding="utf-8")
    assert run["outcome"] is None and run["numeric_voi_training_eligible"] is False
    if mode in {"failure", "bad_quote"}:
        assert journal["context_snapshots"][-1]["forecast_eligible"] is False
        assert journal["context_snapshots"][-1]["recording_scope"] == "audit_only"
    replay_without_provider(folder, verified_inputs, monkeypatch)


def test_failed_extraction_is_not_a_zero_gain_training_label(verified_inputs, monkeypatch):
    folder, *_ = collect_entity(verified_inputs, monkeypatch, mode="failure")
    rows = exported_quality_rows(folder, verified_inputs)
    failed = rows[-1]
    assert failed["g_quality_target"]["status"] == "ineligible"
    assert failed["g_quality_target"]["realized_signed_gain"] is None
    assert failed["predictor_pair_audit"]["valid"] is False


def test_semantic_allowance_does_not_silently_create_empty_assertions(verified_inputs, monkeypatch):
    folder, run, journal, report, calls = collect_entity(verified_inputs, monkeypatch, max_calls=1)
    assert len(calls) == 1
    assert run["terminal_selection"]["reason"] == "semantic_extraction_allowance_exhausted"
    assert run["initial_probabilities"] != run["final_probabilities"]
    assert run["cycles"][-1]["prediction_before"] == run["cycles"][-1]["prediction_after"] == run["final_probabilities"]
    assert run["cycles"][-1]["update"]["status"] == "not_updated"
    replay_without_provider(folder, verified_inputs, monkeypatch)


@pytest.mark.parametrize("status", ["NO_DATA", "FAILED"])
def test_no_data_is_not_body_consumption(verified_inputs, monkeypatch, status):
    folder, run, journal, report, calls = collect_entity(verified_inputs, monkeypatch, status=status)
    assert not any(c["channel"] == "body" for c in calls)
    assert run["initial_probabilities"] == run["final_probabilities"]
    replay_without_provider(folder, verified_inputs, monkeypatch)


def test_unconfigured_extractor_fails_before_source_io(verified_inputs, monkeypatch):
    with pytest.raises(ValueError, match="not configured"):
        collect_entity(verified_inputs, monkeypatch, mode="unconfigured")
    assert not list((verified_inputs["root"] / "batch").rglob("receipt-*.json"))
    journal = json.loads(next((verified_inputs["root"] / "batch/acquisition-journal").glob("*/run.json")).read_bytes())
    assert journal["state"] == "STOPPED" and journal["pending_request_id"] is None
    assert journal["budget"]["reserved_requests"] == 0


def test_initial_local_f_crossing_deadline_terminates_journal_without_prediction(verified_inputs, monkeypatch):
    with pytest.raises(ValueError, match="Empty initial semantic state"):
        collect_entity(verified_inputs, monkeypatch, mode="late_initial_inference")
    folder = verified_inputs["root"] / "batch/match-101"
    receipt = json.loads((folder / "initialization-failure.json").read_bytes())
    assert receipt["reason"] == "semantic_prediction_deadline_elapsed"
    assert receipt["prediction"] is None and receipt["source_requests"] == 0
    assert receipt["prediction_processing"]["effective_at"] is None
    assert receipt["numeric_voi_training_eligible"] is False
    assert not list(folder.glob("receipt-*.json")) and not (folder / "run.json").exists()
    trace = json.loads((folder / TRACE_FILE).read_bytes())
    assert len(trace["events"]) == 1 and trace["events"][0]["attempts"] == []
    journal = json.loads(next((verified_inputs["root"] / "batch/acquisition-journal").glob("*/run.json")).read_bytes())
    assert journal["state"] == "STOPPED" and journal["pending_request_id"] is None
    assert journal["budget"]["charged_requests"] == journal["budget"]["reserved_requests"] == 0


def test_pipeline_groups_fixtures_but_separates_extractor_changes(verified_inputs):
    base = MarketPriorPolicy.from_manifest(verified_inputs["plan"]["selected"][0])
    fixture = fixture_from_verified_identity(base, verified_inputs["binding"])
    model = EntitySemanticF.load(MODEL)
    spec = specification(verified_inputs)
    one = EntitySemanticFOnlyPolicy(base, model, fixture=fixture, artifact=MODEL.read_bytes(), semantic_config=spec)
    other_fixture = {**fixture, "physical_match_id": fixture["physical_match_id"] + "-test-second-match"}
    other_spec = {**spec, "max_calls": 16, "frozen_at": (verified_inputs["now"] + timedelta(seconds=1)).isoformat()}
    two = EntitySemanticFOnlyPolicy(base, model, fixture=other_fixture, artifact=MODEL.read_bytes(), semantic_config=other_spec)
    assert one.version != two.version
    assert one.prediction_pipeline_version == two.prediction_pipeline_version
    changed = deepcopy(spec)
    changed["provider"]["model"] = "different-test-extractor"
    three = EntitySemanticFOnlyPolicy(base, model, fixture=fixture, artifact=MODEL.read_bytes(), semantic_config=changed)
    assert one.prediction_pipeline_version != three.prediction_pipeline_version


def test_modified_semantic_trace_is_not_replayable(verified_inputs, monkeypatch):
    folder, *_ = collect_entity(verified_inputs, monkeypatch)
    path = folder / TRACE_FILE
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="extraction trace.*SHA binding"):
        replay_saved_run(folder, output=verified_inputs["root"] / "tampered", project_root=ROOT)


@pytest.mark.parametrize("change", ["prior", "base_manifest", "artifact", "scope", "training_fixture"])
def test_frozen_new_pipeline_rejects_binding_changes(verified_inputs, change):
    base = MarketPriorPolicy.from_manifest(verified_inputs["plan"]["selected"][0])
    fixture = fixture_from_verified_identity(base, verified_inputs["binding"])
    model = EntitySemanticF.load(MODEL)
    spec, raw = specification(verified_inputs), MODEL.read_bytes()
    if change == "artifact":
        raw += b"\n"
    if change == "scope":
        model.metadata["scope"] = "full_A_fit_training_response_diagnostic"
    if change == "training_fixture":
        fixture["physical_match_id"] = model.metadata["training_match_ids"][0]
    with pytest.raises(ValueError):
        policy = EntitySemanticFOnlyPolicy(base, model, fixture=fixture, artifact=raw, semantic_config=spec)
        if change == "prior":
            base.model.probabilities[:] = [.2, .3, .5]
        if change == "base_manifest":
            base.manifest["market_initialization"]["probabilities"] = [.2, .3, .5]
        policy.public_state(base.model.initial)
