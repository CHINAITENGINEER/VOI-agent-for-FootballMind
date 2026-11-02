"""Versioned research reprocessing, kept separate from actual actor transitions.

Captured bytes may support a future text predictor. Reprocessing never changes
what the historical actor saw, supplies a historical propensity, or creates a
fixed-predictor risk-calibration observation.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from football_agent.news_discovery import NewsDiscoveryTools
from football_spider.models.contracts import Observation
from football_spider.storage.raw_archive import RawArchive
from football_research.acquisition.state import parse_time
from .information_panel import load_panel
from .open_text_episodes import _digest, _read, document_role
from .prospective_text import join_settlements, verify_run_identity, _verified_result_rows

VERSION = "derived-prematch-text-corpus-v1"


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _audit(joined, derived, eligibility, paths):
    return {"actual_runs": joined["dataset"]["audit"]["unique_runs"],
        "unique_matches": joined["dataset"]["audit"]["unique_matches"],
        "actual_episodes": joined["dataset"]["audit"]["episodes"], "frozen_panel_cases": len(paths),
        "research_responses": len(derived), "derived_article_candidates": sum(r["f_training_candidate"] for r in derived),
        "derived_roles": dict(Counter(r["derived_role"] for r in derived)),
        "actual_training_records": len(joined["training_records"]),
        "derived_supervised_examples": sum(x["supervised_text_eligible"] for x in eligibility),
        "new_operational_voi_calibration_records": 0, "network_requests": 0,
        "scope": "versioned_research_reprocessing_preserving_original_actor_and_panel"}


def _projector(archive, transport, clock):
    if "news_discovery" not in transport:
        raise ValueError("Derived news corpus requires a recorded news transport")
    route = transport["source_route_policy"]
    return NewsDiscoveryTools(archive, transport["allowed_hosts"],
        provider=transport["search_provider"], default_route=route["default_route"],
        route_overrides=route["route_overrides"], extraction_version=3, clock=clock,
        index_urls=transport["news_discovery"]["index_urls"], projection_version=2)


def _local_proof(folder, config, identity):
    from .local_text_scope import build_local_scope_proof
    archive = RawArchive(folder / "source-archive")
    visible = _read(folder / "run.json").get("documents", [])
    observations = {o.id: o for o in archive.observations()}
    candidates = [o for o in observations.values()
        if o.endpoint == "https://jalgpall.ee/koondis/1/uudised"
        and any(d.get("raw_sha256") == o.content_hash for d in visible)]
    if len(candidates) != 1:
        return None
    return build_local_scope_proof(identity, identity_archive=folder / "identity-archive",
                                   index_observation=candidates[0], index_archive=archive.root)


def _derive_panel(path, *, clock, recorded_times=None):
    from .local_text_scope import associate_local_text
    path = Path(path).resolve()
    load_panel(path)  # Original extractor, raw requests, history and identity.
    panel = _read(path)
    design = panel["design"]
    folder = Path(design["actor_run_folder"])
    config = _read(folder / "configuration.json")
    identity = verify_run_identity(folder, config)
    archive = RawArchive(Path(panel["archive_folder"]))
    proof = _local_proof(folder, config, identity)
    records = []
    for capture in panel["captures"]:
        original = capture["result"]["document"]
        observation = Observation.model_validate(capture["result"]["observation"])
        key = design["panel_id"] + ":" + capture["action_id"]
        if recorded_times is None:
            projection_clock = clock
        else:
            at = parse_time(recorded_times[key])
            projection_clock = lambda: at
        tools = _projector(archive, design["transport"], projection_clock)
        document = tools.project(observation, capture["prepared"])
        new_at = parse_time(document["available_at"])
        old_at = parse_time(original["available_at"])
        if new_at < max(old_at, parse_time(panel["panel_as_of"])):
            raise ValueError("New projection cannot predate original response/panel completion")
        raw_before = observation.observed_at < parse_time(design["match_scope"]["prediction_cutoff"])
        scope = (associate_local_text(document, proof=proof) if proof else
            {"scope": "unscoped", "reason": "supported_local_source_proof_absent",
             "alias_status": "not_verified", "value_status": "not_estimated"})
        article = document_role(document) == "article_candidate_not_semantically_verified"
        records.append({"record_id": key, "match_id": design["match_scope"]["match_id"],
            "purpose": "research_alternative_reprocessed_text_not_actor_transition",
            "action": next(deepcopy(a) for a in design["candidates"] if a["action_id"] == capture["action_id"]),
            "prior_probabilities": deepcopy(design["initial_probabilities"]),
            "prediction_cutoff": design["match_scope"]["prediction_cutoff"],
            "source_clocks": {"decision_at": capture["decision_at"], "observed_at": original["observed_at"],
                "original_available_at": original["available_at"], "returned_at": capture["returned_at"],
                "original_panel_as_of": panel["panel_as_of"]},
            "derived_available_at": document["available_at"],
            "raw_captured_before_cutoff": raw_before,
            "derived_before_cutoff": new_at < parse_time(design["match_scope"]["prediction_cutoff"]),
            "original_document": deepcopy(original), "derived_document": document,
            "original_qualification": deepcopy(capture["qualification"]),
            "derived_role": document_role(document), "local_scope": scope,
            "f_training_candidate": raw_before and article,
            "operational_transition_eligible": False, "risk_calibration_eligible": False,
            "excluded_uses": ["original_actor_retroactive_admission", "fabricated_sequential_history",
                              "preexisting_f_calibration", "off_policy_estimation_without_support"],
            "information_value": None, "selection_probability": None,
            "cost": {"declared": next(a.get("cost") for a in design["candidates"] if a["action_id"] == capture["action_id"]),
                     "actual": {k: capture["result"][k] for k in ("http_attempts", "latency_seconds", "money", "money_status")}},
            "lineage": {"panel_path": str(path), "panel_file_sha256": _hash(path),
                "original_document_sha256": _digest(original), "observation_id": observation.id,
                "raw_sha256": observation.content_hash, "prepared_sha256": _digest(capture["prepared"]),
                "identity_binding_sha256": identity["binding_sha256"],
                "original_actor_public_before_sha256": _digest(design["actor_public_before"]),
                "transport": tools.configuration()},
            "local_scope_proof": proof})
    return records


def build_corpus(run_folders, panel_paths, *, output, result_archive=None, clock=None):
    """Create a new immutable corpus; no network, fitting or modified old files."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Corpus exists; verify it, never overwrite frozen processing")
    clock = clock or (lambda: datetime.now(timezone.utc))
    folders = sorted({str(Path(p).resolve()) for p in run_folders})
    paths = sorted({str(Path(p).resolve()) for p in panel_paths})
    if not folders or not paths:
        raise ValueError("Actual actor cohort and all frozen panel cases are required")
    proofs, panel_actors, derived = {}, [], []
    for folder in folders:
        proofs.update(_read(Path(folder) / "prior-bindings.json"))
    for path in paths:
        panel_actors.append(_read(path)["design"]["actor_run_folder"])
        derived.extend(_derive_panel(path, clock=clock))
    if len(panel_actors) != len(set(panel_actors)) or not set(panel_actors).issubset(folders):
        raise ValueError("Panel actor must belong to the explicit corpus cohort")
    as_of = clock()
    if any(parse_time(r["derived_available_at"]) > parse_time(as_of) for r in derived):
        raise ValueError("Corpus cannot be built before derived responses become available")
    joined = join_settlements(folders, settlements=[], prior_bindings=proofs, as_of=as_of,
                              result_archive=result_archive)
    labels, checks = _verified_result_rows(sorted(set(panel_actors)), result_archive=result_archive, as_of=as_of)
    by_match = {row["match_id"]: row for row in labels}
    eligibility = []
    for row in derived:
        reasons = ([] if row["f_training_candidate"] else ["article_projection_or_prematch_capture_missing"])
        if row["match_id"] not in by_match:
            reasons.append("independent_settlement_missing")
        eligibility.append({"record_id": row["record_id"], "supervised_text_eligible": not reasons,
            "label": deepcopy(by_match.get(row["match_id"])), "exclusion_reasons": reasons,
            "training_scope": "offline_text_prediction_corpus_not_online_voi_calibration"})
    document = {"schema": VERSION, "built_at": parse_time(as_of).isoformat(),
        "inputs": {"run_folders": folders, "panel_paths": paths,
            "panel_sha256": {p: _hash(p) for p in paths},
            "actor_configuration_sha256": {p: _hash(Path(p) / "configuration.json") for p in folders},
            "result_archive": str(Path(result_archive).resolve()) if result_archive else None},
        "actor_dataset": joined, "derived_records": derived, "supervision_sidecar": eligibility,
        "panel_result_checks": checks,
        "audit": _audit(joined, derived, eligibility, paths)}
    document["corpus_sha256"] = _digest(document)
    _write(output, document)
    return document


def verify_corpus(path):
    """Recompute original and new projections plus independent labels, offline."""
    corpus = _read(path)
    if (corpus.get("schema") != VERSION or corpus.get("corpus_sha256") !=
            _digest({k: v for k, v in corpus.items() if k != "corpus_sha256"})):
        raise ValueError("Corpus binding changed")
    inputs = corpus["inputs"]
    for name in ("run_folders", "panel_paths"):
        if not inputs[name] or inputs[name] != sorted({str(Path(p).resolve()) for p in inputs[name]}):
            raise ValueError("Corpus input paths must be canonical, sorted and unique")
    actors = [_read(p)["design"]["actor_run_folder"] for p in inputs["panel_paths"]]
    if len(actors) != len(set(actors)) or not set(actors).issubset(inputs["run_folders"]):
        raise ValueError("Duplicate or unbound panel actor")
    if len({r["record_id"] for r in corpus["derived_records"]}) != len(corpus["derived_records"]):
        raise ValueError("Duplicate derived response cannot inflate sample count")
    if inputs["panel_sha256"] != {p: _hash(p) for p in inputs["panel_paths"]} or inputs["actor_configuration_sha256"] != {
            p: _hash(Path(p) / "configuration.json") for p in inputs["run_folders"]}:
        raise ValueError("Frozen original panel or actor configuration changed")
    if any(parse_time(r["derived_available_at"]) > parse_time(corpus["built_at"]) for r in corpus["derived_records"]):
        raise ValueError("Corpus predates derived response availability")
    times = {r["record_id"]: r["derived_available_at"] for r in corpus["derived_records"]}
    rebuilt = [row for p in inputs["panel_paths"] for row in _derive_panel(p, clock=None, recorded_times=times)]
    if rebuilt != corpus["derived_records"]:
        raise ValueError("Derived records do not reproduce original source bytes")
    proofs = {}
    for folder in inputs["run_folders"]:
        proofs.update(_read(Path(folder) / "prior-bindings.json"))
    joined = join_settlements(inputs["run_folders"], settlements=[], prior_bindings=proofs,
                              as_of=corpus["built_at"], result_archive=inputs["result_archive"])
    if joined != corpus["actor_dataset"]:
        raise ValueError("Original actor/settlement dataset changed")
    labels, checks = _verified_result_rows(sorted({_read(p)["design"]["actor_run_folder"] for p in inputs["panel_paths"]}),
        result_archive=inputs["result_archive"], as_of=corpus["built_at"])
    by_match = {row["match_id"]: row for row in labels}
    expected = []
    for row in rebuilt:
        reasons = [] if row["f_training_candidate"] else ["article_projection_or_prematch_capture_missing"]
        if row["match_id"] not in by_match:
            reasons.append("independent_settlement_missing")
        expected.append({"record_id": row["record_id"], "supervised_text_eligible": not reasons,
            "label": deepcopy(by_match.get(row["match_id"])), "exclusion_reasons": reasons,
            "training_scope": "offline_text_prediction_corpus_not_online_voi_calibration"})
    if expected != corpus["supervision_sidecar"] or checks != corpus["panel_result_checks"]:
        raise ValueError("Derived corpus supervision is not independently reproducible")
    if corpus["audit"] != _audit(joined, rebuilt, expected, inputs["panel_paths"]):
        raise ValueError("Corpus statistics differ from actual records")
    return {"status": "passed", "corpus_sha256": corpus["corpus_sha256"],
            "network_requests": 0, "audit": deepcopy(corpus["audit"])}


__all__ = ["build_corpus", "verify_corpus", "VERSION"]


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--run", type=Path, action="append", required=True)
    build.add_argument("--panel", type=Path, action="append", required=True)
    build.add_argument("--result-archive", type=Path)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("corpus", type=Path)
    args = parser.parse_args(argv)
    result = (build_corpus(args.run, args.panel, output=args.output, result_archive=args.result_archive)["audit"]
              if args.command == "build" else verify_corpus(args.corpus))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
