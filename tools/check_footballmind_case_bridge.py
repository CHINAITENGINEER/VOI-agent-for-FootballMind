"""Offline end-to-end acceptance bridge for one FootballMind semantic case.

The bridge intentionally uses a controlled ``example.org`` transport and a
scripted semantic provider.  It exercises the production case preparation,
materialisation, bootstrap admission, directional two-step VoI runtime,
report rendering and exact mathematical replay without network access or an
external provider call.  Its synthetic XOR scenario is a mechanism test only;
it is never presented as a real football prediction or an empirical ``g``
calibration result.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.autonomous_match_contract import build_match_bundle
from football_agent.semantic_scenario_protocol import (
    CandidateDiscoveryProtocol, DirectionalObservationProtocol,
    DirectionalProposalProtocol,
)
from football_agent.semantic_scenario_report import write_semantic_report
from football_agent.semantic_scenario_runtime import (
    SemanticScenarioRuntime, replay_math, save,
)
from tools.materialize_footballmind_case import materialize
from tools.run_footballmind_semantic_case import resolve_bootstrap_url


NOW = datetime(2026, 10, 8, 8, tzinfo=timezone.utc)
SEED = "https://example.org/synthetic/index"
# The live projection admits observed article-shaped links. Keep this fixture
# representative of that contract rather than giving the synthetic transport
# an exception for arbitrary landing-page paths.
ARTICLE_A = "https://example.org/news/synthetic-context-a"
ARTICLE_B = "https://example.org/news/synthetic-context-b"


def _article(document_id: str, url: str, text: str, *, kind: str = "article",
             links: tuple[dict[str, str], ...] = ()) -> dict[str, Any]:
    return {
        "document_id": document_id, "url": url, "title": "Synthetic controlled source",
        "kind": kind, "status": "OK", "text": text, "links": list(links),
        "observed_at": NOW.isoformat(), "available_at": NOW.isoformat(),
        "source_published_at": "2026-10-07T12:00:00Z",
        "structured_news": {"segments": [{"content_role": "paragraph", "text": text,
                                               "source_published_at": "2026-10-07T12:00:00Z"}]},
    }


class ControlledExampleTransport:
    """A no-network transport with explicit audit receipts."""

    def __init__(self):
        self.executions: list[dict[str, Any]] = []

    def configuration(self) -> dict[str, Any]:
        return {"provider": "controlled_example_org_test_only", "actual_network": False,
                "test_only": True, "real_provider_call": False}

    def prepare(self, candidate, state):
        # Discovery is declared by the case.  Article URLs are returned by the
        # controlled search response, so the production runtime can exercise
        # the same search-then-read capability path without a network call.
        if candidate.get("tool") == "read_url" and candidate.get("url") not in {
                SEED, ARTICLE_A, ARTICLE_B}:
            raise ValueError("controlled transport received an unregistered URL")
        return deepcopy(candidate)

    def execute(self, prepared, *, run_id, request_id):
        self.executions.append(deepcopy(prepared))
        if prepared["tool"] == "search_web":
            query = prepared.get("query", "")
            target = ARTICLE_A if "context a" in query.casefold() else ARTICLE_B
            links = ({"url": target, "title": "Synthetic context article"},)
            document = _article("search-" + request_id, "https://example.org/synthetic/search",
                                "Synthetic search result; snippet is not evidence.",
                                kind="search_results", links=links)
        elif prepared["url"] == SEED:
            # This controlled case tests the composite search-then-read
            # capability. The bootstrap is an index with no observed article
            # cards; each article URL first appears in its search response.
            document = _article("index", SEED, "Synthetic source search index.", kind="index")
        else:
            document = _article(
                "article-a" if prepared["url"] == ARTICLE_A else "article-b",
                prepared["url"],
                "Synthetic match-specific effect evidence for the declared home-side hypothesis.",
            )
        return {"http_attempts": 1, "document": document, "latency_seconds": .001,
                "run_id": run_id, "request_id": request_id, "test_only": True,
                "real_provider_call": False}


def _xor_proposal(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if len(candidates) != 2:
        raise ValueError("controlled XOR bridge requires exactly two visible candidates")
    names = ("00", "01", "10", "11")
    hypotheses = [{
        "id": name, "description": "Synthetic XOR latent state " + name,
        "prior": .25,
        "outcome_probabilities": list((.9, .05, .05) if name[0] == name[1]
                                       else (.05, .05, .9)),
    } for name in names]
    actions = []
    for index, candidate in enumerate(candidates):
        likelihoods = [{
            "hypothesis_id": name,
            "probabilities": list((.97, .02, .01) if name[index] == "0"
                                   else (.02, .97, .01)),
        } for name in names]
        actions.append({
            "action_id": candidate["action_id"],
            "question": "Does synthetic context bit %s hold for this fixture?" % ("A" if index == 0 else "B"),
            "observation_group": "synthetic-context-%s" % ("a" if index == 0 else "b"),
            "direction_contract": {"supports": ["home_positive"],
                                    "refutes": ["home_negative"]},
            "likelihoods": likelihoods,
        })
    return {"schema": DirectionalProposalProtocol.version, "hypotheses": hypotheses,
            "actions": actions,
            "assumptions": ["Synthetic XOR continuation mechanism; not football evidence."]}


class ControlledSemanticProviderFactory:
    """Scripted semantic provider with an explicit non-live receipt."""

    def __init__(self):
        self.inputs: list[dict[str, Any]] = []

    def __call__(self, workspace, protocol):
        factory = self

        class Provider:
            def configuration(self):
                return {"provider": "controlled_semantic_test_only",
                        "protocol": protocol.version, "test_only": True,
                        "real_provider_call": False}

            def decide(self, public):
                factory.inputs.append({"protocol": protocol.version, "public": deepcopy(public)})
                if protocol.version == CandidateDiscoveryProtocol.version:
                    response = {"schema": protocol.version, "queries": [
                        {"query": "synthetic context A", "information_need": "synthetic bit A"},
                        {"query": "synthetic context B", "information_need": "synthetic bit B"},
                    ]}
                elif protocol.version == DirectionalProposalProtocol.version:
                    response = _xor_proposal(public["candidates"])
                elif protocol.version == DirectionalObservationProtocol.version:
                    document = public["documents"][0]
                    response = {
                        "schema": protocol.version,
                        "action_id": public["action"]["action_id"],
                        "outcome": "supports", "direction": "home_positive",
                        "subject": "home", "direction_basis": "explicit_effect",
                        "directional_claim": "正文明确给出本场主队方向性影响。",
                        "evidence": [{"document_id": document["document_id"],
                                      "quote": document["text"]}],
                        "applicability": "match_specific", "reason": "Controlled synthetic classification.",
                    }
                else:
                    raise AssertionError("unexpected protocol " + protocol.version)
                return {"response": response,
                        "receipt": {"provider": "controlled_semantic_test_only",
                                     "real_provider_call": False, "test_only": True,
                                     "usage": {}, "latency_seconds": .001}}

        return Provider()


def _synthetic_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build immutable synthetic request, source profile and baseline."""
    match = {
        "match_id": "synthetic:xor:case-bridge",
        "home": "Synthetic North", "away": "Synthetic South",
        "competition": "Synthetic Cup", "kickoff": "2026-10-10T12:00:00Z",
        "prediction_cutoff": "2026-10-09T12:00:00Z", "mode": "prematch",
    }
    bundle = build_match_bundle(match, now=NOW)
    profile = {
        "version": "synthetic-source-profile-v1", "status": "registered",
        "test_only": True, "real_provider_call": False,
        "teams": [{"side": "home", "input_name": "Synthetic North",
                    "canonical_name": "Synthetic North"},
                   {"side": "away", "input_name": "Synthetic South",
                    "canonical_name": "Synthetic South"}],
        "allowed_hosts": ["example.org"], "seed_urls": [SEED],
        "bootstrap_url": SEED, "scope": "synthetic_test_only",
    }
    baseline = {
        "match_id": match["match_id"], "prediction_cutoff": match["prediction_cutoff"],
        "source": "synthetic_controlled_bridge", "model_version": "synthetic-xor-v1",
        "probabilities": [.475, .05, .475], "scope": "synthetic_test_only",
    }
    return bundle, profile, baseline


def _assert_receipts(run: dict[str, Any], transport: ControlledExampleTransport,
                     factory: ControlledSemanticProviderFactory) -> None:
    if not transport.executions:
        raise AssertionError("controlled transport was never exercised")
    for receipt in run.get("http_receipts", []):
        returned = receipt.get("returned", {})
        if returned.get("test_only") is not True or returned.get("real_provider_call") is not False:
            raise AssertionError("HTTP receipt lacks explicit test_only/real_provider_call=false")
    for call in run.get("semantic_calls", []):
        receipt = call.get("receipt", {})
        if receipt.get("test_only") is not True or receipt.get("real_provider_call") is not False:
            raise AssertionError("provider receipt lacks explicit test_only/real_provider_call=false")
    if len(factory.inputs) != len(run.get("semantic_calls", [])):
        raise AssertionError("semantic input/receipt accounting differs")


def run_synthetic_bridge(output: Path) -> dict[str, Any]:
    """Run the full offline synthetic bridge and return an audit summary."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    bundle, profile, baseline = _synthetic_inputs()
    case = materialize(bundle, profile, baseline)
    case_path = output / "case"
    case_path.mkdir()
    save(case_path / "bundle.json", bundle)
    save(case_path / "source-profile.json", profile)
    save(case_path / "baseline.json", baseline)
    save(case_path / "case.json", case)
    seeds, bootstrap = resolve_bootstrap_url(case)
    transport = ControlledExampleTransport()
    factory = ControlledSemanticProviderFactory()
    run_dir = output / "run"
    runtime = SemanticScenarioRuntime(
        provider_factory=factory, transport=transport, output=run_dir,
        case=case, seed_urls=seeds, bootstrap_url=bootstrap,
        request_budget=5, request_loss_cost=.02, semantic_call_budget=12,
        max_extensions=0, max_rediscoveries=0, planning_horizon=2,
        require_directional_evidence=True, clock=lambda: NOW,
    )
    # This controlled mechanism uses the single nominal model so the robust
    # lower envelope exposes the exact finite XOR value; it remains explicitly
    # synthetic and is not an empirical calibration choice.
    runtime.configuration["contamination_levels"] = [0.0]
    save(run_dir / "configuration.json", runtime.configuration)
    run = runtime.run()
    report = write_semantic_report(run, run_dir)
    replay = replay_math(run)
    _assert_receipts(run, transport, factory)
    if run["status"] not in {"completed", "completed_with_resource_stop"}:
        raise AssertionError("synthetic runtime did not complete")
    if run["configuration"]["planning_horizon"] != 2:
        raise AssertionError("planning horizon was not persisted")
    first = run["cycles"][0]["decision"]
    h1 = runtime.model.evaluate([], 4, horizon=1)
    if h1["selected_action_id"] is not None:
        raise AssertionError("XOR h1 should stop before the first query")
    selected = first["selected_action_id"]
    chosen = next(row for row in first["candidates"] if row["action_id"] == selected)
    if not chosen["net"] < 0 or not chosen["planning_net"] > 0:
        raise AssertionError("XOR bridge did not show immediate-negative/planning-positive first action")
    if len(run["observations"]) != 2:
        raise AssertionError("XOR bridge must acquire and classify both evidence branches")
    if replay.get("status") != "passed":
        raise AssertionError("math replay did not pass")
    if report.get("report_path") is None or not Path(report["report_path"]).exists():
        raise AssertionError("semantic report was not materialized")
    summary = {
        "status": "passed", "scope": "synthetic_offline_mechanism_only",
        "claims_boundary": "not_real_match_evidence; not_empirical_g_calibration; no_network",
        "checks": {
            "build_match_bundle": "passed", "materialize": "passed",
            "bootstrap_resolution": "passed", "directional_runtime": "passed",
            "two_step_xor": {"h1_selected": h1["selected_action_id"],
                              "h2_selected": selected,
                              "first_immediate_net": chosen["net"],
                              "first_planning_net": chosen["planning_net"],
                              "observations": len(run["observations"])},
            "report": "passed", "math_replay": replay,
            "receipts_explicitly_test_only": "passed",
        },
        "evidence_paths": {
            "bundle": str((case_path / "bundle.json").resolve()),
            "case": str((case_path / "case.json").resolve()),
            "run": str((run_dir / "run.json").resolve()),
            "report": str(Path(report["report_path"]).resolve()),
            "analysis": str(Path(report["analysis_path"]).resolve()),
        },
        "resources": {"http_requests": run["resources"]["http_requests"],
                      "semantic_calls": run["resources"]["semantic_calls"],
                      "actual_network": False, "real_provider_calls": 0,
                      "test_only_receipts": len(run["http_receipts"]) + len(run["semantic_calls"])},
        "condition_range": {
            "planning_horizon": 2, "request_loss_cost": .02,
            "contamination_levels": [0.0],
            "visible_menu": 2,
            "interpretation": "exact finite Bellman value under declared synthetic XOR laws",
        },
    }
    save(output / "bridge-summary.json", summary)
    return summary


def check_source_case(source_case: Path, output: Path) -> dict[str, Any]:
    """Convert an existing case into the strict bundle/materialisation path."""
    source_case = Path(source_case)
    original = json.loads((source_case / "case.json").read_text(encoding="utf-8"))
    match = original["match"]
    request = {"match_id": match.get("match_id"), "match": match.get("match"),
               "competition": match.get("competition"), "kickoff": match.get("kickoff"),
               "prediction_cutoff": match.get("prediction_cutoff"), "mode": "prematch"}
    bundle = build_match_bundle(request, now=NOW)
    case = materialize(bundle, original["source_profile"], original["structured_baseline"])
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    save(destination / "bundle.json", bundle)
    save(destination / "case.json", case)
    summary = {"status": "passed", "scope": "existing_source_case_conversion_only",
               "claims_boundary": "conversion_integrity_only; no_runtime; no_synthetic_value",
               "checks": {"source_case_bundle": "passed", "source_case_materialize": "passed"},
               "evidence_paths": {"bundle": str((destination / "bundle.json").resolve()),
                                  "case": str((destination / "case.json").resolve())},
               "resources": {"http_requests": 0, "semantic_calls": 0,
                             "actual_network": False, "real_provider_calls": 0}}
    save(destination / "source-case-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-case", type=Path,
                        help="Optional existing case directory for conversion-only integrity")
    args = parser.parse_args()
    summary = run_synthetic_bridge(args.output)
    if args.source_case:
        summary["source_case"] = check_source_case(args.source_case, args.output / "source-case")
        save(args.output / "bridge-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
