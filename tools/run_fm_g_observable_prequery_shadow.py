"""One bounded, explicitly opted-in fresh shadow using existing collection APIs.

No new HTTP/archive implementation and no synthetic selection authority.
The immutable preparation is created before collect_prepared_batch is called.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.evidence_candidates import EvidenceDocumentCandidates
from football_agent.match_source_profiles import resolve_verified_profile
from football_agent.observable_prequery_shadow import (
    SCHEMA, REGISTRY_SCHEMA, ObservablePrequeryShadow, request_binding,
)
from football_research.acquisition.state import parse_time
from football_research.data_adapters.prospective_text import (
    MarketPriorPolicy, _save, collect_prepared_batch, verify_prior_archive,
)
from football_research.data_adapters.verified_fixtures import verify_identity_binding

MATCH = "spftll:match:3003907"
MODEL_SHA = "4967d1068109bf982cd22c1a1a66d8b9f3b2e02a2f121f131ca80fae3c8bca21"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new run directory; completed/prepared runs cannot be repeated")
    binding_path = ROOT / "var/fm3-delivery-20261009-v2/fm3/identity/identity-bundle.json"
    identity_archive = ROOT / "var/prospective-information-stream-v3/phase3-identity-20261009-v1/identity-archive"
    prior_archive = ROOT / "var/prospective-information-stream-v2/phase2-setup-20261009-v1"
    model_path = ROOT / "var/submission/p9-g-observable-only-v3-final3-20261010/model.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))["bindings"][MATCH]
    now = datetime.now(timezone.utc)
    checked = verify_identity_binding(binding, archive_root=identity_archive, as_of=now)
    manifest = binding["initialization_manifest"]
    policy = MarketPriorPolicy.from_manifest(manifest)
    verify_prior_archive(manifest, prior_archive)
    if checked["status"] != "CONFIRMED" or now >= parse_time(policy.model.cutoff):
        raise ValueError("Fresh pre-kickoff identity/cutoff gate is not satisfied")
    profile = resolve_verified_profile(binding, version=7)
    match = policy.model.fixture
    identity = {"match_id": MATCH, "home_team": binding["canonical"]["home_name"],
        "away_team": binding["canonical"]["away_name"], "kickoff_at": match["kickoff"],
        "prediction_cutoff": policy.model.cutoff,
        "home_aliases": binding["aliases"]["home"], "away_aliases": binding["aliases"]["away"]}
    generator = EvidenceDocumentCandidates(
        match_terms=[identity["home_team"], identity["away_team"]],
        seed_urls=profile["seed_urls"], allowed_hosts=profile["allowed_hosts"],
        numeric_actions=[], web_cost={"money": .01, "requests": 1.},
        match=identity, profile=profile, discovery_version=13)
    public = {"state": policy.model.initial, "prediction": policy.model.predict_partial(policy.model.initial),
              "documents": [], "attempts": [], "decision_at": now.isoformat()}
    actions = generator(public)
    registry = {"schema": REGISTRY_SCHEMA, "capabilities_by_action_id": {
        action["action_id"]: {"declared_at": now.isoformat(),
            "request_binding_sha256": request_binding(action),
            "basis": "request identity observed in frozen generic menu; v3 semantic capability and scalar declarations not supplied"}
        for action in actions}}
    model_bytes = model_path.read_bytes()
    shadow = ObservablePrequeryShadow(generator, registry=registry, model_bytes=model_bytes,
                                     expected_model_sha256=MODEL_SHA)
    preflight = shadow.candidate_contexts(public, actions)
    args.output.mkdir(parents=True)
    preparation = args.output / "preparation"
    _save(preparation / "initializations.json", {"schema": "prospective-text-initializations-v1",
        "selected": [manifest], "prepared_at": now.isoformat(), "labels": [],
        "selection_basis": "user_predeclared_backup_single_match_shadow",
        "initialization_reused_without_changing_original_clocks": True})
    _save(preparation / "identity-bundle.json", {"bindings": {MATCH: binding}})
    _save(preparation / "shadow-spec.json", {"schema": SCHEMA, "registry": registry,
        "model_path": str(model_path.resolve()), "model_raw_sha256": MODEL_SHA})
    _save(preparation / "menu-preflight.json", {"observed_at": now.isoformat(),
        "actions": actions, "candidate_contexts": preflight,
        "kind": "fresh_local_preflight_not_runtime_selection_or_http"})
    _save(preparation / "plan.json", {"schema": "footballmind-fresh-observable-shadow-plan-v1",
        "frozen_at": now.isoformat(), "fixture_id": MATCH, "identity_binding_sha256": binding["binding_sha256"],
        "kickoff": match["kickoff"], "prediction_cutoff": policy.model.cutoff,
        "policy_arm": "random_explore", "seed": 20261010, "propensity_logging": True,
        "requests_per_match": 4, "setup_http_requests": 0,
        "source_profile": profile, "candidate_generator": shadow.configuration(),
        "model_scope": "market_prior_only_no_text_f", "empirical_g_enabled": False,
        "empirical_training_eligible": False, "synthetic_selection_allowed": False,
        "new_actions_without_exact_registry_entry": "UNAVAILABLE_DIAGNOSTIC_ONLY",
        "missing_v3_features": "no_defaults_no_rank_no_synthetic_stop",
        "quote_observed_at": policy.model.quote["observed_at"],
        "quote_time": policy.model.quote.get("source_timestamp"),
        "quote_frozen_at": policy.model.frozen_at, "quote_is_fresh_claim": False,
        "cutoff_semantics": "existing_model_acquisition_horizon; not substituted with decision clock",
        "independent_outcome": None, "realized_gain": None,
        "future_result_collection_executed": False,
        "input_references": {"identity_bundle": str(binding_path.resolve()),
            "identity_archive": str(identity_archive.resolve()), "prior_archive": str(prior_archive.resolve())}})
    if not args.execute:
        print(json.dumps({"status": "prepared_only", "http": 0, "path": str(args.output.resolve())}))
        return
    try:
        report = collect_prepared_batch(preparation / "initializations.json", output=args.output / "batch",
            archive_root=prior_archive, allowed_hosts=[], requests_per_match=4, seed=20261010,
            identity_bundle=preparation / "identity-bundle.json", identity_archive=identity_archive,
            source_profile_version=7, news_provider="bing_rss", news_projection_version=11,
            evidence_version="match-evidence-extraction-v5", candidate_discovery_version=13,
            policy_arm="random_explore", policy_arm_version="footballmind-observable-shadow-random-v1",
            random_seed=20261010, arm_assigned_at=now.isoformat(), journal_enabled=True,
            observable_shadow_path=preparation / "shadow-spec.json")
        _save(args.output / "execution-status.json", {"status": "completed", "report": report})
        print(json.dumps(report, ensure_ascii=True))
    except Exception as exc:
        _save(args.output / "execution-status.json", {"status": "failed_preserved_no_automatic_retry",
            "failed_at": datetime.now(timezone.utc).isoformat(), "error_class": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
