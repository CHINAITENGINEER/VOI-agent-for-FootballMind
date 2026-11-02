"""Recompute a repaired f forecast from an archived pre-kickoff shadow.

The archived generic run is immutable.  This tool reuses only documents whose
recorded availability preceded the decision cutoff, constructs a separate
semantic extraction bank, runs the frozen research f, and joins the independent
settlement only after f has completed.  It never treats the repaired output as
the output of the original online run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from football_research.acquisition.entity_semantic_f import (
    EntityEvidenceBank,
    EntitySemanticF,
    _predict,
)
from football_research.acquisition.semantic_f import public_input
from football_research.acquisition.state import parse_time
from football_agent.deepseek_provider import DeepSeekInquiryProvider
from football_agent.semantic_request_binding import (
    TypedRequestBoundSourceAssertionProtocol, bind_response, provider_input,
)

ROOT = Path(__file__).resolve().parents[1]
SHADOW = ROOT / "var/shadow/fm-g-single-match-shadow-20261010-v1"
REPORT = SHADOW / "audit-final/shadow-report.json"
MODEL = ROOT / "var/f-entity-semantic-g-shadow-cumulative-b-20261011-pipelinev2-liverpool-v1/batch/match-3003906/context-predictor.json"
SETTLEMENTS = ROOT / "var/submission/fm-g-settlement-evidence-20261011-v1/settled-results.jsonl"


def digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gunzip_json(path: Path):
    return json.loads(gzip.open(path, "rb").read())


def archived_contexts():
    paths = sorted((SHADOW / "batch/match-3003899/source-archive/raw/agent_trajectory").rglob("*.body.gz"))
    contexts = [gunzip_json(path) for path in paths]
    contexts = [item for item in contexts if item.get("schema") == "runtime-context-v1" and item.get("documents")]
    if not contexts:
        raise ValueError("No archived pre-kickoff body contexts found")
    return contexts


def receipt_rows(report):
    return {row.get("url"): row for row in report["runtime"]["receipt_rows"]
            if row.get("status") == "OK" and row.get("document_status") == "OK"}


def fixture_from_report(report, context):
    source = report["fixture"]
    base = context["state"]["base"]
    return {
        "physical_match_id": report["physical_match_id"],
        "match_id": report["physical_match_id"],
        "home_team_id": base["home_team_id"],
        "away_team_id": base["away_team_id"],
        "home_name": source["home_team"],
        "away_name": source["away_team"],
        "competition_id": base["competition_id"],
        "competition": source["competition"],
        "kickoff": base["kickoff"],
    }


def normalize_document(document, row):
    # These values come from the archived runtime context.  publication is
    # deliberately retained as null when the source did not expose it.
    result = deepcopy(document)
    result.setdefault("title", "")
    result.setdefault("source_published_at", None)
    result["captured_at"] = row["archive_save_returned_at"]
    if parse_time(result["captured_at"]) > parse_time(result["available_at"]):
        raise ValueError("Archived capture must precede document availability")
    result["paragraphs"] = [line for line in result["text"].splitlines() if line.strip()]
    return result


def make_state(fixture, prior, decision_at, documents):
    return {
        "fixture": deepcopy(fixture),
        "decision_at": decision_at,
        "prediction_cutoff": decision_at,
        "information_mode": "operational",
        "allow_unknown_publication_clock": True,
        "prior_probabilities": list(prior),
        "prior_basis": "observed_spftll_regulation_market_snapshot",
        "documents": [deepcopy(doc) for doc in documents],
        "observed_links": [],
        "history": [],
    }


def make_extractions(fixture, documents, output):
    """Run the real typed semantic extractor over the archived body text.

    A missing or invalid response aborts this calculation.  It must never be
    converted into an empty claim list, because that was the exact defect this
    audit is intended to remove.
    """
    provider = DeepSeekInquiryProvider(
        output / "semantic-provider-calls",
        protocol=TypedRequestBoundSourceAssertionProtocol,
        model="deepseek-v4-pro", thinking="disabled", max_tokens=8192,
        timeout_seconds=120,
    )
    if not provider.configuration()["configured"]:
        raise RuntimeError("DEEPSEEK_API_KEY_NOT_CONFIGURED")
    records, audit = [], []
    for index, document in enumerate(documents):
        public = public_input(fixture, document, "body")
        request = provider_input(public, protocol=TypedRequestBoundSourceAssertionProtocol)
        returned = None
        binding = None
        validation_failure = None
        # A single same-input retry repairs transient model-format mistakes,
        # but never changes the public input or silently repairs a claim.
        for attempt_number in (1, 2):
            returned = provider.decide(request)
            try:
                bound, binding = bind_response(
                    public, request, returned["response"],
                    protocol=TypedRequestBoundSourceAssertionProtocol,
                )
                break
            except Exception as exc:
                validation_failure = type(exc).__name__ + ":" + str(exc).splitlines()[0]
                if attempt_number == 2:
                    raise RuntimeError(
                        f"SEMANTIC_EXTRACTION_INVALID_AFTER_ONE_RETRY:{public['input_id']}:{validation_failure}"
                    ) from None
        records.append({
            "input": public,
            "response": bound,
            "provenance": {
                "mode": "repaired_offline_recompute",
                "source": "archived_pre_kickoff_body",
                "provider": "deepseek-api",
                "provider_protocol": TypedRequestBoundSourceAssertionProtocol.version,
                "provider_request_folder": provider.last_attempt_folder,
                "provider_receipt": returned["receipt"],
                "binding": binding,
                "validation_failure_before_retry": validation_failure,
            },
        })
        audit.append({
            "index": index, "url": document["url"],
            "input_id": public["input_id"], "claims": len(bound["claims"]),
            "provider_request_folder": provider.last_attempt_folder,
            "receipt_status": returned["receipt"].get("status"),
            "validation_failure_before_retry": validation_failure,
        })
    return records, audit


def loss(probabilities, outcome):
    index = {"home": 0, "draw": 1, "away": 2}[outcome]
    return -math.log(float(probabilities[index]))


def load_settlement(match_id):
    for line in SETTLEMENTS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("physical_match_id") == match_id:
            return row
    raise ValueError(f"No independent settlement for {match_id}")


def main(output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    contexts = archived_contexts()
    fixture = fixture_from_report(report, contexts[-1])
    rows = receipt_rows(report)
    documents = []
    for context in contexts:
        for document in context["documents"]:
            key = (document["url"], document.get("text", ""))
            if key not in {(d["url"], d.get("text", "")) for d in documents}:
                row = rows.get(document["url"])
                if row is None:
                    raise ValueError(f"No immutable receipt row for {document['url']}")
                documents.append(normalize_document(document, row))
    documents.sort(key=lambda d: parse_time(d["available_at"]))
    decision_at = max(parse_time(doc["available_at"]) for doc in documents).isoformat()
    kickoff = parse_time(fixture["kickoff"])
    if any(parse_time(doc["available_at"]) >= kickoff for doc in documents):
        raise ValueError("A repaired input was not available before kickoff")
    prior = report["before_after"]["prediction_before"]
    before = make_state(fixture, prior, decision_at, [])
    after = make_state(fixture, prior, decision_at, documents)
    output.mkdir(parents=True, exist_ok=False)
    extractions, extraction_audit = make_extractions(fixture, documents, output)
    bank = EntityEvidenceBank(extractions)
    before_projection = bank.project(before)
    after_projection = bank.project(after)
    model = EntitySemanticF.load(MODEL)
    before_prediction = _predict(model.coefficients, before_projection).tolist()
    after_prediction = _predict(model.coefficients, after_projection).tolist()
    settlement = load_settlement(report["physical_match_id"])
    outcome = settlement["result"]
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "footballmind-repaired-f-before-after-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calculation_mode": "repaired_offline_recompute",
        "information_mode": "prospective_capture",
        "source_time_basis": "all archived document available_at strictly precedes kickoff",
        "outcome_visible_to_f": False,
        "original_runtime_f_consumed": False,
        "fixture": fixture,
        "timing": {
            "decision_at_used": decision_at,
            "kickoff": fixture["kickoff"],
            "documents": [
                {"url": d["url"], "available_at": d["available_at"],
                 "source_published_at": d.get("source_published_at"),
                 "text_sha256": hashlib.sha256(d["text"].encode()).hexdigest()}
                for d in documents
            ],
        },
        "model": {
            "path": str(MODEL),
            "model_version": model.model_version,
            "model_file_sha256": file_sha(MODEL),
            "production_ready": False,
            "gain_model_enabled": False,
        },
        "prediction": {
            "prediction_before_market_prior": list(prior),
            "prediction_after_repaired_f": after_prediction,
            "probability_l1_change": sum(abs(a-b) for a, b in zip(prior, after_prediction)),
            "before_projection": before_projection,
            "after_projection": after_projection,
            "semantic_inputs_consumed": after_projection["used_input_ids"],
            "body_documents_consumed": len(after_projection["used_input_ids"]),
            "extraction_audit": extraction_audit,
        },
        "settlement": {
            "result": outcome,
            "result_observed_at": settlement["result_observed_at"],
            "independent_outcome_verified": settlement["independent_outcome_verified"],
            "raw_sha256": settlement["raw_sha256"],
        },
        "log_loss": {
            "formula": "-log(p_observed_outcome)",
            "loss_before_market_prior": loss(prior, outcome),
            "loss_after_repaired_f": loss(after_prediction, outcome),
            "repaired_offline_gain": loss(prior, outcome) - loss(after_prediction, outcome),
        },
        "interpretation": {
            "old_equal_loss_explanation": "The original generic run retained market prior because body_f_consumed=false.",
            "repaired_result_status": "separate offline calculation; not a claim about original online output",
            "g_training": "diagnostic_only_until a predeclared repaired trajectory policy is approved",
        },
        "source_hashes": {
            "original_shadow_report": file_sha(REPORT),
            "f_model": file_sha(MODEL),
            "settlement_file": file_sha(SETTLEMENTS),
        },
    }
    (output / "repaired-f-before-after.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "FootballMind 修复后 f 离线重算",
        "=" * 64,
        "",
        f"比赛：{fixture['home_name']}–{fixture['away_name']}",
        f"赛前正文数量：{len(documents)}；全部 available_at < kickoff：是",
        f"原始运行消费正文：否；修复后离线计算消费正文：{len(documents)} 篇",
        "",
        f"market prior before: {list(prior)}",
        f"repaired f after:  {after_prediction}",
        f"probability L1 change: {result['prediction']['probability_l1_change']:.12f}",
        "",
        f"result: {outcome}",
        f"loss before: {result['log_loss']['loss_before_market_prior']:.12f}",
        f"loss after repaired f: {result['log_loss']['loss_after_repaired_f']:.12f}",
        f"repaired offline gain: {result['log_loss']['repaired_offline_gain']:+.12f}",
        "",
        "说明：本次使用的是比赛前已归档的正文；赛果只在最后评估 loss 时使用。",
        "正文先经真实 DeepSeek typed-v3 提取并逐条 quote 校验；若无支持特征，只能由真实返回的 claims=[] 或 unmodeled claims 导致，不能预先伪造空响应。",
    ]
    (output / "repaired-f-before-after.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"schema": "footballmind-repaired-f-before-after-manifest-v1",
                "files": {name: file_sha(output / name) for name in
                          ("repaired-f-before-after.json", "repaired-f-before-after.txt")}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": manifest,
                      "probability": result["prediction"], "loss": result["log_loss"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "var/reviews/fm-repaired-offline-recompute-20261013-v1")
    main(parser.parse_args().output)
