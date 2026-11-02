from __future__ import annotations

import json
import math
import hashlib
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_entity_semantic_f_experiment import (
    CASES, PRIORS, SOURCE, RECORDS, _prepare_rows, read,
)
from football_research.acquisition.entity_semantic_f import (
    EntityEvidenceBank, EntitySemanticF, vector as entity_vector,
)
from football_research.acquisition.historical_context_training import prepare_historical_cases


OUT = ROOT / "var" / "work" / "fm-entity-f-v2-research-20261025-v1"
V1_MODEL = ROOT / "var" / "f-entity-semantic-20261011-v2" / "model.json"
V1_DIAGNOSTIC = ROOT / "var" / "f-entity-semantic-20261011-v2" / "diagnostics" / "candidate-model.json"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(state):
    base = np.asarray(entity_vector(state), dtype=float)
    counts = np.asarray(state["title_entity_counts"] + state["novel_body_entity_counts"], dtype=float)
    prior = np.asarray(state["prior"], dtype=float)
    entropy = float(-np.sum(prior * np.log(np.clip(prior, 1e-12, 1.0))))
    margin = float(np.max(prior) - np.partition(prior, -2)[-2])
    # v2 research representation: v1 channels + log-count channels + prior basis
    # + summary uncertainty. It is deterministic and contains no outcome fields.
    return np.concatenate([base, np.log1p(counts), np.repeat(prior, len(counts) // 3), [entropy, margin]])


def brier(probabilities, labels):
    y = np.zeros((len(labels), 3), dtype=float)
    y[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((np.asarray(probabilities) - y) ** 2, axis=1)))


def score(rows, probabilities):
    labels = np.asarray([int(r["outcome"]) for r in rows], dtype=int)
    p = np.asarray(probabilities, dtype=float)
    ll = float(np.mean([-math.log(max(float(p[i, y]), 1e-12)) for i, y in enumerate(labels)]))
    return {"rows": len(rows), "matches": len({r["physical_match_id"] for r in rows}), "log_loss": ll, "brier": brier(p, labels), "accuracy": float(np.mean(np.argmax(p, axis=1) == labels))}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    records = read(RECORDS)
    bank = EntityEvidenceBank(records)
    cases, _ = prepare_historical_cases(read(CASES))
    snapshots = read(PRIORS)
    protocol = read(SOURCE / "protocol.json")
    splits = protocol["splits"]
    prepared, training = _prepare_rows(cases, snapshots, bank, splits)
    rows = [r for mids in prepared.values() for r in mids]
    train_rows = [r for r in training]
    X_train = np.asarray([features(r["state"]) for r in train_rows], dtype=float)
    y_train = np.asarray([r["outcome"] for r in train_rows], dtype=int)
    counts = Counter(r["physical_match_id"] for r in train_rows)
    weights = np.asarray([1.0 / counts[r["physical_match_id"]] for r in train_rows], dtype=float)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="tanh", solver="lbfgs", alpha=1.0,
                        max_iter=2000, random_state=20261025)
    clf.fit(X_train, y_train, sample_weight=weights)
    selected = EntitySemanticF.load(V1_MODEL)
    diagnostic = EntitySemanticF.load(V1_DIAGNOSTIC)
    by_group = {g: [r for r in rows if r["physical_match_id"] in splits[g]] for g in ("A", "B", "C")}
    predictions = []
    metrics = {"v2_mlp": {}, "prior": {}, "f_v1_selected": {}, "f_v1_diagnostic": {}}
    for group, group_rows in by_group.items():
        x = np.asarray([features(r["state"]) for r in group_rows], dtype=float)
        v2 = clf.predict_proba(x)
        prior = np.asarray([r["state"]["prior"] for r in group_rows], dtype=float)
        v1 = np.asarray([np.asarray(r["state"]["prior"]) @ np.eye(3) for r in group_rows], dtype=float)
        selected_p = np.asarray([selected.coefficients for _ in group_rows])
        # EntitySemanticF's frozen residual formula, evaluated on already projected states.
        from football_research.acquisition.entity_semantic_f import _predict
        selected_probs = np.asarray([_predict(selected.coefficients, r["state"]) for r in group_rows])
        diagnostic_probs = np.asarray([_predict(diagnostic.coefficients, r["state"]) for r in group_rows])
        metrics["v2_mlp"][group] = score(group_rows, v2)
        metrics["prior"][group] = score(group_rows, prior)
        metrics["f_v1_selected"][group] = score(group_rows, selected_probs)
        metrics["f_v1_diagnostic"][group] = score(group_rows, diagnostic_probs)
        for i, r in enumerate(group_rows):
            predictions.append({"physical_match_id": r["physical_match_id"], "group": group, "outcome": r["outcome"], "v2_mlp": v2[i].tolist(), "prior": prior[i].tolist(), "f_v1_selected": selected_probs[i].tolist(), "f_v1_diagnostic": diagnostic_probs[i].tolist()})
    parameter_count = int(sum(x.size for x in clf.coefs_) + sum(x.size for x in clf.intercepts_))
    model = {"schema": "footballmind-entity-f-v2-research-mlp-v1", "status": "research_only_not_promoted", "frozen_v1_model_sha256": digest(V1_MODEL), "frozen_v1_diagnostic_sha256": digest(V1_DIAGNOSTIC), "input_dimension": int(X_train.shape[1]), "hidden_layer_sizes": [128, 64], "activation": "tanh", "alpha": 1.0, "parameter_count": parameter_count, "split": splits, "weights": "equal physical-match weight within A", "coefficients": [x.tolist() for x in clf.coefs_], "intercepts": [x.tolist() for x in clf.intercepts_], "selection_allowed": False, "production_ready": False}
    dump(OUT / "model.json", model)
    dump(OUT / "metrics.json", metrics)
    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump(OUT / "feature-schema.json", {"schema": "footballmind-entity-f-v2-research-features-v1", "input_dimension": int(X_train.shape[1]), "components": ["entity_semantic_v1_vector", "log1p_entity_counts", "prior_basis", "entropy", "margin"], "forbidden_inputs": ["outcome", "prediction_after", "realized_gain"]})
    # Visual comparison on held-out C: only a diagnostic visualization, not model selection.
    labels = ["prior", "f_v1_selected", "f_v1_diagnostic", "v2_mlp"]
    cvals = [metrics[k]["C"]["log_loss"] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(labels, cvals, color=["#8aa6a3", "#3f7f73", "#c8903b", "#6a5acd"]); ax.set_ylabel("held-out C log loss (lower is better)"); ax.set_title("Entity f-v2 research comparison"); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(OUT / "heldout-logloss-comparison.png", dpi=160); plt.close(fig)
    report = {"schema": "footballmind-entity-f-v2-research-report-v1", "train_matches": len(splits["A"]), "validation_matches": len(splits["B"]), "test_matches": len(splits["C"]), "train_rows": len(train_rows), "parameter_count": parameter_count, "metrics": metrics, "decision": "REJECT_PROMOTION_UNTIL_INDEPENDENT_DATA", "reason": "v2 MLP has larger parameter count but must beat selected f-v1 and prior on held-out C; this package records the result without promotion.", "network_requests": 0, "api_calls": 0}
    dump(OUT / "report.json", report)
    (OUT / "readable-report.txt").write_text("Entity f-v2 research\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in {"manifest.json", "verification.json"})
    dump(OUT / "manifest.json", {p.name: digest(p) for p in files})
    dump(OUT / "verification.json", {"schema": "footballmind-entity-f-v2-research-verification-v1", "manifest_entries": len(files), "network_requests": 0, "api_calls": 0, "formal_f_modified": False, "formal_runtime_modified": False, "selection_allowed": False, "production_ready": False, "status": "PASS_OFFLINE_RESEARCH_RUN"})


if __name__ == "__main__":
    main()
