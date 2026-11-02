"""Prepare frozen semantic training inputs and exercise batching without training."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from football_research.acquisition.semantic_training_dataset import (
    SemanticTrainingDataset, file_sha, prepare_panel, read_json, read_rows, write_json)
from football_research.acquisition.postread_semantic_embedding import build_semantic_design

DATA = Path("E:/FootballMind/experiments/postread-clean-expansion-20261025-v1")
EMBED = Path("E:/FootballMind/experiments/postread-qwen-semantic-20261025-v1")
STABILITY = Path("E:/FootballMind/experiments/postread-qwen-stability-20261025-v1")
F_MODEL = ROOT / "var/f-entity-semantic-f-shadow-materialization-20261011-v1/model.json"
OUT = Path("E:/FootballMind/experiments/postread-semantic-training-ready-20261025-v1/supervised-panel")


def smoke(output, data, embedding, reference_fold):
    """Use frozen transformations/heads for exact inference; no fit is called."""
    import joblib
    from threadpoolctl import threadpool_limits
    original = read_rows(data / "features.jsonl")
    labels = {r["sample_id"]: r for r in read_rows(data / "labels.jsonl")}
    with np.load(embedding / "embeddings.npz", allow_pickle=False) as archive:
        vectors = {key: archive[key] for key in archive.files}
    design_checks = []
    for dimension in (32, 256, 1024):
        dataset = SemanticTrainingDataset(output, dimension=dimension)
        if not isinstance(dataset.vectors, np.memmap) or dataset.vectors.flags.writeable or dataset.gains is not None:
            raise AssertionError("Expected read-only mmap and no label array")
        batches = list(dataset.iter_batches(31))
        expected = build_semantic_design(original, vectors, dimension)
        errors = {}
        for name, target in expected.items():
            actual = np.concatenate([batch["features"][name] for batch in batches])
            errors[name] = float(np.max(np.abs(actual-target)))
            if not np.array_equal(actual, target):
                raise AssertionError("Batched design differs from previous frozen design")
        design_checks.append(dict(dimension=dimension, batches=len(batches), rows=len(dataset), max_absolute_errors=errors,
                                  design_width=86+4*dimension, vectors_read_only_mmap=True, labels_opened_by_loader=False))
    split_checks = []
    for split in ("train", "validation", "test"):
        dataset = SemanticTrainingDataset(output, split=split, include_labels=True)
        seen = []
        for batch in dataset.iter_batches(17):
            seen.extend(batch["sample_id"])
            if not np.array_equal(batch["realized_gain"], np.array([labels[sid]["realized_gain"] for sid in batch["sample_id"]])):
                raise AssertionError("Batch labels are not aligned by sample_id")
        expected = [r["sample_id"] for r in original if r["split"] == split]
        if seen != expected:
            raise AssertionError("Split batch order differs")
        y = np.array([labels[sid]["realized_gain"] for sid in seen])
        split_checks.append(dict(split=split, rows=len(seen), matches=len({labels[sid]["physical_match_id"] for sid in seen}),
                                 positive=int((y>0).sum()), zero=int((y==0).sum()), negative=int((y<0).sum())))
    bundle = joblib.load(reference_fold / "models.joblib")
    if bundle["encoder_sha256"] != file_sha(embedding / "embeddings.npz"):
        raise AssertionError("Reference model embedding binding differs")
    expected_assignment = {r["physical_match_id"]:r["split"] for r in original}
    if bundle["assignments"] != expected_assignment:
        raise AssertionError("Reference model split differs from panel")
    saved = {}
    for r in read_rows(reference_fold / "predictions.jsonl"):
        saved.setdefault(r["model"], {})[r["sample_id"]] = r["predicted_gain"]
    prediction_checks = []
    with threadpool_limits(limits=2):
        for model, spec in bundle["fitted"].items():
            key = spec.get("input_key", "")
            if not key.startswith("qwen"):
                continue
            dimension = 256 if key == "qwen256" else 32
            dataset = SemanticTrainingDataset(output, dimension=dimension)
            pp = []; ids = []
            for batch in dataset.iter_batches(19):
                design = batch["features"]
                if key.endswith("_body"):
                    x = design["body"]
                elif key.endswith("_no_candidate_claims"):
                    x = design["fulltext_state_without_candidate_claims"]
                elif key.endswith("_no_adopted_text"):
                    x = np.c_[design["state"], design["claims"], design["body"]]
                else:
                    x = design["fulltext_state"]
                pp.extend(spec["estimator"].predict(spec["scaler"].transform(x))*bundle["target_scale"]+bundle["target_mean"])
                ids.extend(batch["sample_id"])
            delta = float(np.max(np.abs(np.array(pp)-np.array([saved[model][sid] for sid in ids]))))
            # sklearn's float32 body-only head uses different accumulation for
            # different batch shapes. Verify the old full-batch shape separately
            # instead of claiming bitwise batching equivalence.
            all_design = dataset.batch(np.arange(len(dataset)))["features"]
            if key.endswith("_body"):
                all_x = all_design["body"]
            elif key.endswith("_no_candidate_claims"):
                all_x = all_design["fulltext_state_without_candidate_claims"]
            elif key.endswith("_no_adopted_text"):
                all_x = np.c_[all_design["state"], all_design["claims"], all_design["body"]]
            else:
                all_x = all_design["fulltext_state"]
            all_pred = spec["estimator"].predict(spec["scaler"].transform(all_x))*bundle["target_scale"]+bundle["target_mean"]
            full_delta = float(np.max(np.abs(all_pred-np.array([saved[model][sid] for sid in ids]))))
            tolerance = 1e-7 if all_x.dtype == np.float32 else 1e-9
            if delta > tolerance or full_delta > 1e-12:
                raise AssertionError("Frozen head prediction differs")
            prediction_checks.append(dict(model=model, rows=len(pp), batch_size=19,
                max_absolute_prediction_difference=delta, full_batch_max_prediction_difference=full_delta,
                batch_absolute_tolerance=tolerance, input_dtype=str(all_x.dtype),
                batch_numerical_drift=(delta>1e-9), no_retraining=True))
    return dict(schema="semantic-training-panel-smoke-v1", status="passed", design_checks=design_checks,
        split_checks=split_checks, frozen_head_checks=prediction_checks, network_requests=0, encoder_inference=0,
        scaler_fits=0, training_calls=0, production_ready=False,
        limitations=["51 existing historical matches only; no new independent matches", "Historical version/time assumptions inherited",
                     "Current split includes previously examined validation/test matches; no new blind test",
                     "Existing adopted_text composition retained; context sidecar is independent and not silently concatenated"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--embedding-dir", type=Path, default=EMBED)
    parser.add_argument("--binding-plan", type=Path, default=STABILITY / "experiment-plan.json")
    parser.add_argument("--f-model", type=Path, default=F_MODEL)
    parser.add_argument("--reference-fold", type=Path, default=STABILITY / "P0_original")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--verify-existing", action="store_true", help="Verify an already prepared panel without rewriting its feature/label files")
    args = parser.parse_args()
    def offline(*a, **k):
        raise RuntimeError("Panel preparation is offline")
    socket.socket.connect = socket.create_connection = socket.getaddrinfo = offline
    plan = read_json(args.binding_plan)
    if args.verify_existing:
        config = read_json(args.output / "config.json")
        expected = {str(Path(p).resolve()):v for p,v in plan["source_hashes"].items()}
        if any(expected.get(str(Path(p).resolve())) != digest or file_sha(p) != digest for p,digest in config["source_sha256"].items()):
            raise AssertionError("Existing panel source bindings differ")
    else:
        config = prepare_panel(args.data, args.embedding_dir, args.f_model, plan["source_hashes"], args.output)
    checks = smoke(args.output, args.data, args.embedding_dir, args.reference_fold)
    if any(file_sha(p) != digest for p, digest in config["source_sha256"].items()):
        raise AssertionError("Frozen input changed")
    checks["source_hashes_unchanged"] = True
    checks["f_model_sha256"] = config["f_model_sha256"]
    checks["unique_vectors"] = config["text_count"]
    checks["stored_vector_bytes"] = (args.output / "text-vectors.npy").stat().st_size
    checks["binding_plan"] = dict(path=str(args.binding_plan), sha256=file_sha(args.binding_plan))
    write_json(args.output / "readiness-report.json", checks)
    write_json(args.output / "manifest.json", {str(p.relative_to(args.output)).replace('\\','/'):file_sha(p)
        for p in sorted(args.output.rglob("*")) if p.is_file()})
    print(json.dumps(dict(status=checks["status"], output=str(args.output), rows=config["rows"], matches=config["matches"],
                         vectors=config["text_count"], frozen_head_checks=len(checks["frozen_head_checks"]))))


if __name__ == "__main__":
    main()
