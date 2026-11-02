"""Read-only, ID-aligned verification of completed sequential g experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from football_research.acquisition.sequential_postread_g import predict_head


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").split("\n") if line]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    complete = read_json(directory / "training-completion.json")
    require(complete["status"] == "completed" and not complete["validation_used"] and not complete["test_used"], "Training stage boundary")
    model = read_json(directory / "model.json")
    manifest = read_json(directory / "manifest.json")
    require(all(sha(directory / name) == digest for name, digest in manifest.items()), "Output manifest mismatch")
    require(all(sha(path) == digest for path, digest in model["source_sha256"].items()), "Input source changed")
    bundle = joblib.load(directory / "g-models.joblib")
    require(sha(directory / "g-models.joblib") == model["g_model_sha256"], "Model hash mismatch")
    rows = read_rows(directory / "labels.jsonl")
    x = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    ids = read_json(directory / "feature-ids.json")
    require(ids == [r["sample_id"] for r in rows] and len(ids) == len(set(ids)) == len(x), "Features/labels unique ID alignment")
    require(x.shape[1] == model["feature_dimension"] and np.isfinite(x).all(), "Invalid features")
    lookup = {r["sample_id"]: i for i, r in enumerate(rows)}
    role = np.asarray([r["split_role"] for r in rows])
    mids = np.asarray([r["physical_match_id"] for r in rows])
    y = np.asarray([r["realized_gain"] for r in rows])
    require(set(role) == {"g_fit", "g_dev"}, "Held-out labels in training")
    require(not (set(mids[role == "g_fit"]) & set(mids[role == "g_dev"])), "Cross-match split overlap")
    max_gain_difference = 0.
    for r in rows:
        yy = r["outcome_class"]
        before, after = np.asarray(r["p_before"]), np.asarray(r["p_after"])
        require(np.isfinite(before).all() and np.isfinite(after).all() and (before > 0).all() and (after > 0).all(), "Invalid f probability")
        require(abs(before.sum() - 1) < 1e-6 and abs(after.sum() - 1) < 1e-6, "Non-normalized f probability")
        delta = float(np.log(after[yy]) - np.log(before[yy]))
        max_gain_difference = max(max_gain_difference, abs(delta-r["realized_gain"]))
        require(abs(delta-r["realized_gain"]) < 1e-13, "Wrong gain")
        require(r["candidate_id"] not in r["adopted_ids"] and len(r["adopted_ids"]) == len(set(r["adopted_ids"])), "Duplicate adoption")
        require(r["f_model_sha256"] == bundle["f_model_sha256"] and r["f_executed"] is True, "Frozen f binding mismatch")
    require(np.allclose(x[:, :3], np.asarray([r["p_before"] for r in rows]), atol=1e-6, rtol=0), "Feature probability not before state")
    predictions = read_rows(directory / "predictions.jsonl")
    saved_metrics = {(r["model"], r["split_role"]): r for r in read_json(directory / "metrics.json")}
    scaler_reports, metric_reports = [], []
    maximum_prediction_difference = 0.
    for name, spec in bundle["fitted"].items():
        pp = predict_head(spec, x, bundle["target_mean"], bundle["target_scale"])
        require(np.isfinite(pp).all(), "Nonfinite model prediction")
        rr = [r for r in predictions if r["model"] == name]
        require(len(rr) == len(rows) and {r["sample_id"] for r in rr} == set(ids), "Prediction ID coverage")
        for r in rr:
            i = lookup[r["sample_id"]]
            require(r["realized_gain"] == y[i] and r["physical_match_id"] == mids[i] and r["split_role"] == role[i], "Prediction/label mismatch")
            maximum_prediction_difference = max(maximum_prediction_difference, abs(pp[i]-r["predicted_gain"]))
        for split in ("g_fit", "g_dev"):
            ii = np.flatnonzero(role == split)
            residual = pp[ii]-y[ii]
            actual = dict(mse=float(np.mean(residual**2)), mae=float(np.mean(abs(residual))),
                match_macro_mse=float(np.mean([np.mean((pp[mids == mid]-y[mids == mid])**2) for mid in set(mids[ii])])),
                mean_residual=float(np.mean(residual)))
            require(all(abs(value-saved_metrics[(name, split)][key]) < 1e-12 for key,value in actual.items()), "Independent metric mismatch")
            metric_reports.append(dict(model=name,split_role=split,**actual))
        scaler = spec.get("scaler")
        if scaler is not None:
            xx = x[:, :spec["state_width"]] if spec.get("input") == "state" else x[:, spec["column_indices"]] if spec.get("input") == "metadata" else x
            transformed = scaler.transform(xx)
            negative = np.flatnonzero(scaler.var_ < 0)
            require(np.isfinite(scaler.var_).all() and np.isfinite(scaler.scale_).all() and np.isfinite(transformed).all(), "Nonfinite scaler values")
            require((scaler.scale_ > 0).all(), "Nonpositive scaler scale")
            details = [dict(column=int(j),variance=float(scaler.var_[j]),scale=float(scaler.scale_[j]),
                fit_raw_range=float(np.ptp(xx[role == "g_fit", j]))) for j in negative]
            require(all(d["scale"] == 1 and d["fit_raw_range"] == 0 for d in details), "Negative scaler variance not an exact constant feature")
            scaler_reports.append(dict(model=name,negative_variance_count=len(details),negative_variance_details=details,
                minimum_variance=float(scaler.var_.min()),all_transformed_finite=True))
    require(maximum_prediction_difference < 1e-12, "Reload prediction mismatch")
    selection = read_json(directory / "selection-frozen.json")
    paths = read_rows(directory / "dev-policy-paths.jsonl")
    require({p["physical_match_id"] for p in paths} == set(mids[role == "g_dev"]), "Selection policy includes non-dev match")
    for choice in selection["all_dev_policy_choices"]:
        rr = [p for p in paths if p["model"] == choice["model"] and p["threshold"] == choice["threshold"]]
        require(len(rr) == choice["matches"], "Dev policy missing paths")
        ll = float(np.mean([-np.log(p["p_after"][p["outcome_class"]]) for p in rr]))
        require(abs(ll-choice["mean_final_log_loss"]) < 1e-12, "Policy log loss mismatch")
    optimum = min(selection["all_dev_policy_choices"], key=lambda r:(r["mean_final_log_loss"],r["mean_updates"],r["model"],r["threshold"]))
    require(optimum == selection["selected"] == bundle["selected"], "Incorrect selection rule")
    return dict(status="passed",directory=str(directory),f_model_sha256=bundle["f_model_sha256"],g_model_sha256=model["g_model_sha256"],
        rows=len(rows),feature_shape=list(x.shape),roles={s:dict(rows=int((role == s).sum()),matches=len(set(mids[role == s]))) for s in ("g_fit","g_dev")},
        gain_counts=dict(positive=int((y>0).sum()),zero=int((y==0).sum()),negative=int((y<0).sum())),
        max_gain_recalculation_difference=max_gain_difference,max_reloaded_prediction_difference=maximum_prediction_difference,
        independent_metrics=metric_reports,scalers=scaler_reports,selected=optimum,
        output_manifest_files=len(manifest),input_hashes_verified=len(model["source_sha256"]),
        heldout_labels_used=False,new_training=False,network_requests=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--models",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    with threadpool_limits(limits=1):reports=[verify(p) for p in args.models]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(status="passed",models=reports),ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(dict(status="passed",output=str(args.output),models=len(reports)),ensure_ascii=False))


if __name__ == "__main__":main()
