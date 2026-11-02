"""Materialize a strictly purged research shadow f from the capability audit.

The capability audit keeps two different objects for each regularization:

* a full-A fit used only for an in-sample response diagnostic; and
* one model per purged forward-validation fold.

The selection summary historically points at the former.  This tool deliberately
does not reinterpret that object as a prospective model.  It verifies the
selected regularization, chooses the latest chronological purged fold, adds an
explicit top-level ``promotion_allowed=false`` boundary, and writes a new
research-only artifact.  It never changes the formal f or the audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_AUDIT = ROOT / "var" / "f-entity-semantic-selection-audit-20261011-v6" / "audit-report.json"
DEFAULT_OUTPUT = ROOT / "var" / "f-entity-semantic-f-shadow-materialization-20261011-v1"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _model_entries(audit: dict[str, Any], regularization: float) -> list[tuple[str, dict[str, Any]]]:
    entries = []
    for version, model in audit.get("audit_models", {}).items():
        if not isinstance(model, dict) or not isinstance(model.get("metadata"), dict):
            continue
        metadata = model["metadata"]
        if (metadata.get("applied_regularization") == regularization
                and metadata.get("scope") == "A_purged_forward_validation"):
            entries.append((version, model))
    return entries


def _candidate(audit: dict[str, Any], regularization: float) -> dict[str, Any]:
    rows = [row for row in audit.get("candidates", [])
            if row.get("regularization") == regularization]
    if len(rows) != 1:
        raise ValueError("audit must contain exactly one selected-regularization candidate")
    return rows[0]


def materialize(audit_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    audit_path = audit_path.resolve()
    audit = read_json(audit_path)
    protocol = audit.get("protocol")
    selection = audit.get("capability_constrained_selection")
    if not isinstance(protocol, dict) or not isinstance(selection, dict):
        raise ValueError("capability audit protocol and selection are required")
    if protocol.get("promotion_allowed") is not False:
        raise ValueError("audit protocol must be non-promoting")
    if protocol.get("response_scope") != "A_purged_forward_validation_matches_only":
        raise ValueError("audit response scope is not A-purged forward validation")
    if selection.get("promotion_allowed") is not False:
        raise ValueError("capability selection must be non-promoting")
    regularization = selection.get("proposed_candidate_regularization")
    if regularization != 10.0:
        raise ValueError("this materializer is intentionally bound to regularization=10")
    candidate = _candidate(audit, regularization)
    # The summary model is intentionally rejected here: it is a full-A response
    # diagnostic, not an out-of-sample/purged model.
    if candidate.get("model_version") != selection.get("proposed_candidate_model_version"):
        raise ValueError("candidate summary/model selection identity mismatch")
    summary_model = audit["audit_models"].get(candidate["model_version"])
    if not isinstance(summary_model, dict):
        raise ValueError("selected candidate model object is missing")
    if summary_model.get("metadata", {}).get("scope") != "full_A_fit_training_response_diagnostic":
        raise ValueError("selected summary must be the full-A diagnostic this repair excludes")

    # The candidate's forward fold records are the source of the chronological
    # materialization rule.  We choose the latest fold, which maximizes historical
    # training data while keeping all later validation matches out of the fit.
    folds = candidate.get("forward_folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("selected candidate has no purged forward folds")
    fold = max(folds, key=lambda row: (row.get("first_validation_decision", ""), row.get("fold_index", -1)))
    fold_version = fold.get("model_version")
    purged_entries = dict(_model_entries(audit, regularization))
    source_model = purged_entries.get(fold_version)
    if source_model is None:
        raise ValueError("latest forward fold model object is missing")
    source_meta = source_model.get("metadata", {})
    if source_meta.get("scope") != "A_purged_forward_validation":
        raise ValueError("latest fold is not A-purged")
    if source_meta.get("candidate_role") != "selection_capability_audit_only":
        raise ValueError("latest fold has unexpected candidate role")
    if source_meta.get("applied_regularization") != regularization:
        raise ValueError("latest fold regularization differs from selection")
    if source_meta.get("production_ready") is not False or source_meta.get("gain_model_enabled") is not False:
        raise ValueError("latest fold is missing research-only gates")
    if source_model.get("model_version") != fold_version:
        raise ValueError("forward fold model identity mismatch")
    if fold.get("coefficients_sha256") != digest_json(source_model.get("coefficients")):
        raise ValueError("forward fold coefficient digest mismatch")

    metadata = deepcopy(source_meta)
    metadata.update({
        "candidate_role": "selection_capability_audit_only",
        "scope": "A_purged_forward_validation",
        "promotion_allowed": False,
        "production_ready": False,
        "gain_model_enabled": False,
        "evidence_origin": "historical_reconstruction",
        "materialization_role": "latest_A_purged_forward_validation_fold_shadow",
        "materialization_rule": (
            "selected_regularization_from_A_capability_audit_then_latest_chronological_purged_fold"
        ),
        "source_audit_candidate_model_version": candidate["model_version"],
        "source_audit_fold_model_version": fold_version,
        "source_audit_fold_index": fold.get("fold_index"),
        "source_audit_fold_first_validation_decision": fold.get("first_validation_decision"),
        "source_audit_selection_sha256": digest_bytes(audit_path.read_bytes()),
    })
    output_model = {
        "schema": source_model.get("schema"),
        "parameters": source_model.get("parameters"),
        "representation_sha256": source_model.get("representation_sha256"),
        "coefficients": source_model.get("coefficients"),
        "metadata": metadata,
    }

    # Recompute the model identity using the same canonical payload rule as
    # EntitySemanticF.  Importing the class also lets us validate the resulting
    # artifact with the production loader without enabling it in runtime.
    from football_research.acquisition.entity_semantic_f import EntitySemanticF
    model = EntitySemanticF(output_model["coefficients"], output_model["metadata"])
    output_model["model_version"] = model.model_version
    output.mkdir(parents=True)
    model_path = output / "model.json"
    write_json(model_path, output_model)
    loaded = EntitySemanticF.load(model_path)
    if loaded.model_version != output_model["model_version"]:
        raise AssertionError("materialized model does not round-trip through EntitySemanticF.load")

    audit_sha = digest_bytes(audit_path.read_bytes())
    model_sha = digest_bytes(model_path.read_bytes())
    model_canonical_sha = digest_json(output_model)
    reconciliation = {
        "schema": "footballmind-f-shadow-materialization-reconciliation-v1",
        "status": "materialized_research_shadow",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_audit": {"path": str(audit_path), "sha256": audit_sha},
        "selection": {
            "rule": protocol.get("rule"),
            "selected_regularization": regularization,
            "selection_summary_model_version": candidate["model_version"],
            "selection_summary_scope": summary_model["metadata"].get("scope"),
            "selection_summary_rejected_for_shadow": True,
            "selection_summary_rejection_reason": "full_A_fit_training_response_diagnostic",
        },
        "materialization": {
            "rule": metadata["materialization_rule"],
            "source_fold_index": fold.get("fold_index"),
            "source_fold_model_version": fold_version,
            "source_fold_scope": source_meta.get("scope"),
            "source_fold_training_match_ids": source_meta.get("training_match_ids"),
            "source_fold_training_rows_sha256": source_meta.get("training_rows_sha256"),
            "source_fold_coefficients_sha256": fold.get("coefficients_sha256"),
            "materialized_model_version": output_model["model_version"],
            "materialized_model_sha256": model_sha,
            "materialized_model_canonical_sha256": model_canonical_sha,
            "promotion_allowed": False,
            "production_ready": False,
            "gain_model_enabled": False,
        },
        "formal_f": {
            "modified": False,
            "path": str(ROOT / "var/f-entity-semantic-20261011-v2/model.json"),
        },
        "collection_gate": {
            "shadow_f_materialized": True,
            "collection_authorized": False,
            "reason": "materialization_only; fresh inventory and non-g f manifest still required",
            "g_training_enabled": False,
        },
    }
    recon_path = output / "materialization-reconciliation.json"
    write_json(recon_path, reconciliation)
    readme = """# FootballMind entity-semantic shadow f（研究用途）

本目录是从 capability audit 的最后一个时间顺序 A-purged forward-validation fold
独立物化的 shadow f。审计摘要中的 c4f710… 是 full-A response diagnostic，已明确拒绝，
没有被当作 prospective 模型使用。

本模型的 `promotion_allowed=false`、`production_ready=false`、`gain_model_enabled=false`
均为硬边界。它不替换正式 f，不表示信息有价值，也不表示 g 已训练。只有在采集线程
重新核对本目录、冻结 non-g manifest、获得新鲜赛前 inventory 后，才可以开始真实 B 轨迹。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": path.name, "bytes": path.stat().st_size,
                          "sha256": digest_bytes(path.read_bytes())})
    manifest = {
        "schema": "footballmind-f-shadow-materialization-manifest-v1",
        "status": "materialized_research_shadow",
        "source_audit_sha256": audit_sha,
        "model_version": output_model["model_version"],
        "model_sha256": model_sha,
        "model_canonical_sha256": model_canonical_sha,
        "promotion_allowed": False,
        "production_ready": False,
        "gain_model_enabled": False,
        "files": files,
    }
    write_json(output / "manifest.json", manifest)
    return {"output": str(output), "model_version": output_model["model_version"],
            "model_sha256": model_sha, "audit_sha256": audit_sha,
            "source_fold_model_version": fold_version,
            "source_fold_index": fold.get("fold_index")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(materialize(args.audit, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
