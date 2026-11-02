"""Label-isolated, memory-mapped research inputs for a frozen post-read g.

This module does not encode text, fit preprocessing, train a model, or import
runtime.  Label files are opened only with explicit ``include_labels=True``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .postread_semantic_embedding import EncoderContract, normalize_prefix, text_sha, validate_feature_rows


SCHEMA = "postread-semantic-training-panel-v1"
DIMENSIONS = (32, 256, 1024)
SPLITS = ("train", "validation", "test")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _by_id(rows):
    result = {r["sample_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate sample_id")
    return result


def validate_groups(rows, groups):
    assignments = {}
    for split, matches in groups.items():
        if split not in SPLITS:
            raise ValueError("Unknown split")
        for match in matches:
            if match in assignments:
                raise ValueError("Match appears more than once in split groups")
            assignments[match] = split
    if set(assignments) != {r["physical_match_id"] for r in rows}:
        raise ValueError("Split groups do not cover exactly the input matches")
    for row in rows:
        if assignments[row["physical_match_id"]] != row["split"]:
            raise ValueError("Row split disagrees with physical_match_id group")
    return assignments


def prepare_panel(data_dir, embedding_dir, f_model, expected_source_hashes, output):
    """Convert existing frozen inputs, retaining all positive/zero/negative labels.

    ``expected_source_hashes`` must come from an already frozen experiment plan;
    newly computed hashes alone are not accepted as source verification.
    """
    data_dir, embedding_dir, f_model, output = map(Path, (data_dir, embedding_dir, f_model, output))
    sources = [data_dir / n for n in ("features.jsonl", "labels.jsonl", "samples.jsonl", "split.json")]
    sources += [embedding_dir / "embeddings.npz", embedding_dir / "embedding-report.json", f_model]
    expected = {str(Path(p).resolve()): value for p, value in expected_source_hashes.items()}
    hashes = {}
    for path in sources:
        digest = file_sha(path)
        if expected.get(str(path.resolve())) != digest:
            raise ValueError("Frozen source digest missing or changed: " + str(path))
        hashes[str(path.resolve())] = digest
    features = read_rows(data_dir / "features.jsonl")
    validate_feature_rows(features)
    feature_by = _by_id(features)
    labels = _by_id(read_rows(data_dir / "labels.jsonl"))
    samples = _by_id(read_rows(data_dir / "samples.jsonl"))
    if set(feature_by) != set(labels) or set(feature_by) != set(samples):
        raise ValueError("Feature/label/sample IDs differ")
    split = read_json(data_dir / "split.json")
    validate_groups(features, split["groups"])
    report = read_json(embedding_dir / "embedding-report.json")
    contract = EncoderContract(**report["contract"])
    if contract != EncoderContract() or report["source_sha256"] != file_sha(data_dir / "features.jsonl"):
        raise ValueError("Frozen semantic model or feature binding differs")
    if report["vectors_file_sha256"] != file_sha(embedding_dir / "embeddings.npz"):
        raise ValueError("Frozen embedding digest differs")
    f_sha = file_sha(f_model)
    f_versions = {label.get("f_model_version") for label in labels.values()}
    if len(f_versions) != 1 or None in f_versions:
        raise ValueError("Labels must identify a single frozen f version")
    texts = {}
    locations = defaultdict(set)
    metadata = []
    gains = []
    for row in features:
        sid = row["sample_id"]
        label, sample = labels[sid], samples[sid]
        if any(item["physical_match_id"] != row["physical_match_id"] or item["split"] != row["split"] for item in (label, sample)):
            raise ValueError("Label/sample identity differs from feature ID")
        if label["f_model_sha256"] != f_sha or not label.get("offline_f_evaluated") or not label.get("training_eligible"):
            raise ValueError("Label is not eligible or is bound to another f")
        if not np.isfinite(label["realized_gain"]):
            raise ValueError("Nonfinite gain")
        if not np.allclose(row["state_numeric"][:3], label["p_before"], atol=1e-12, rtol=0):
            raise ValueError("State probabilities do not match label before state")
        if sample["text_sha256"] != text_sha(row["candidate_text"]):
            raise ValueError("Sample body differs from feature text")
        for field in ("candidate_text", "adopted_text"):
            if row[field].strip():
                digest = text_sha(row[field]); texts[digest] = row[field]
                locations[digest].add(row["split"])
        candidate_sha = text_sha(row["candidate_text"])
        adopted_sha = text_sha(row["adopted_text"]) if row["adopted_text"].strip() else None
        metadata.append(dict(sample_id=sid, physical_match_id=row["physical_match_id"], split=row["split"],
            candidate_text_sha256=candidate_sha, adopted_text_sha256=adopted_sha,
            candidate_text_id=sample.get("candidate_text_id"), candidate_title=row["candidate_title"],
            source_url=sample.get("source_url"), source_domain=sample.get("source_domain"),
            source_published_at=sample.get("source_published_at"), source_available_at=sample.get("source_available_at"),
            captured_at=sample.get("captured_at"), kickoff=sample.get("kickoff"), prediction_cutoff=sample.get("prediction_cutoff"),
            discovered_mask=sample.get("discovered_mask"), acquired_mask=sample.get("acquired_mask"),
            candidate_index=sample.get("candidate_index"), f_model_sha256=f_sha,
            g_has_read_candidate=True, f_has_adopted_candidate_before=sample.get("f_has_adopted_candidate_before"),
            provenance="historical_reconstruction_research_only"))
        gains.append(label["realized_gain"])
    cross_split = [key for key, values in locations.items() if len(values) > 1]
    if cross_split:
        raise ValueError("Exact text spans multiple data splits")
    state = np.asarray([r["state_numeric"] for r in features], dtype=np.float64)
    claims = np.asarray([r["semantic_numeric"] for r in features], dtype=np.float64)
    if state.shape[1] != 57 or claims.shape[1] != 29 or not np.isfinite(state).all() or not np.isfinite(claims).all():
        raise ValueError("Expected finite state57/claims29")
    if output.exists():
        raise FileExistsError("Independent output must not exist: " + str(output))
    keys = sorted(texts)
    text_index = {key: i for i, key in enumerate(keys)}
    indices = np.array([[text_index[r["candidate_text_sha256"]], text_index.get(r["adopted_text_sha256"], -1)] for r in metadata], dtype=np.int64)
    with np.load(embedding_dir / "embeddings.npz", allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(keys):
            raise ValueError("Embedding keys do not match unique nonempty input texts")
        # Validate all source vectors before creating the output directory.
        for key in keys:
            vector = archive[key]
            if vector.shape != (1024,) or vector.dtype != np.float32 or not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > 1e-5:
                raise ValueError("Invalid source embedding")
        output.mkdir(parents=True)
        matrix = np.lib.format.open_memmap(output / "text-vectors.npy", mode="w+", dtype=np.float32, shape=(len(keys), 1024))
        for i, key in enumerate(keys):
            matrix[i] = archive[key]
        matrix.flush(); del matrix
    np.save(output / "state.npy", state, allow_pickle=False)
    np.save(output / "claims.npy", claims, allow_pickle=False)
    np.save(output / "text-indices.npy", indices, allow_pickle=False)
    write_json(output / "text-index.json", text_index)
    write_rows(output / "texts.jsonl", (dict(text_sha256=key, text=texts[key]) for key in keys))
    write_rows(output / "rows.jsonl", metadata)
    write_json(output / "split.json", split)
    (output / "labels").mkdir()
    np.save(output / "labels/realized-gain.npy", np.array(gains, dtype=np.float64), allow_pickle=False)
    write_rows(output / "labels/records.jsonl", (labels[r["sample_id"]] for r in features))
    write_json(output / "labels/sample-ids.json", [r["sample_id"] for r in features])
    feature_schema = dict(schema=SCHEMA, state=dict(dtype="float64", dimension=57, names=features[0]["state_feature_names"]),
        claims=dict(dtype="float64", dimension=29, names=features[0]["semantic_feature_names"]),
        text=dict(dtype="float32", stored_dimension=1024, selectable_dimensions=list(DIMENSIONS),
                  transformation="float32 prefix, then per-vector L2; empty adopted text is zero"),
        fulltext_state=dict(order=["state", "claims", "candidate", "adopted", "abs(candidate-adopted)", "candidate*adopted"],
                            dimensions={str(d): 86 + 4*d for d in DIMENSIONS}),
        no_fit=True, post_outcome_fields_in_inputs=False,
        state_scope="adopted text is the existing serialized adopted-evidence string; no candidate-pool context added")
    write_json(output / "feature-schema.json", feature_schema)
    feature_files = {p.name: file_sha(p) for p in output.iterdir() if p.is_file()}
    label_files = {str(p.relative_to(output)).replace('\\', '/'): file_sha(p) for p in (output / "labels").iterdir()}
    config = dict(schema=SCHEMA, rows=len(features), matches=len(split["groups"]["train"])+len(split["groups"]["validation"])+len(split["groups"]["test"]),
        text_count=len(keys), f_model_sha256=f_sha, f_model_version=labels[features[0]["sample_id"]]["f_model_version"],
        encoder_contract=report["contract"], source_sha256=hashes, feature_files=feature_files, label_files=label_files,
        label_access_default=False, production_ready=False, provenance="historical_reconstruction_research_only")
    write_json(output / "config.json", config)
    return config


class SemanticTrainingDataset:
    """Read-only split view. Only requested batches materialize text vectors.

    ``batch`` indices are local to this dataset's split view. The default object
    neither opens nor hashes label files. No fitted scaler/encoder is created.
    """
    def __init__(self, root, *, split=None, dimension=32, include_labels=False):
        self.root = Path(root)
        if dimension not in DIMENSIONS:
            raise ValueError("Supported dimensions: 32, 256, 1024")
        if split is not None and split not in SPLITS:
            raise ValueError("Unknown split")
        if not isinstance(include_labels, bool):
            raise TypeError("include_labels must be an explicit boolean")
        self.dimension = dimension
        self.include_labels = include_labels
        self.config = read_json(self.root / "config.json")
        if self.config["schema"] != SCHEMA:
            raise ValueError("Wrong panel schema")
        if EncoderContract(**self.config["encoder_contract"]) != EncoderContract():
            raise ValueError("Unexpected encoder binding")
        for relative, digest in self.config["feature_files"].items():
            path = (self.root / relative).resolve()
            if path.parent != self.root.resolve() or file_sha(path) != digest:
                raise ValueError("Feature path or digest invalid: " + relative)
        self.rows = read_rows(self.root / "rows.jsonl")
        _by_id(self.rows)
        validate_groups(self.rows, read_json(self.root / "split.json")["groups"])
        if any(r["f_model_sha256"] != self.config["f_model_sha256"] for r in self.rows):
            raise ValueError("Row f binding differs")
        self.text_index = read_json(self.root / "text-index.json")
        if sorted(self.text_index.values()) != list(range(len(self.text_index))):
            raise ValueError("Text index must be a complete bijection")
        self.vectors = np.load(self.root / "text-vectors.npy", mmap_mode="r", allow_pickle=False)
        self.state = np.load(self.root / "state.npy", mmap_mode="r", allow_pickle=False)
        self.claims = np.load(self.root / "claims.npy", mmap_mode="r", allow_pickle=False)
        self.text_indices = np.load(self.root / "text-indices.npy", mmap_mode="r", allow_pickle=False)
        count = len(self.rows)
        expected = [(self.vectors, (len(self.text_index), 1024), np.float32), (self.state, (count, 57), np.float64),
                    (self.claims, (count, 29), np.float64), (self.text_indices, (count, 2), np.int64)]
        for matrix, shape, dtype in expected:
            if matrix.shape != shape or matrix.dtype != dtype:
                raise ValueError("Matrix shape/dtype differs from panel contract")
        if count != self.config["rows"] or len(self.text_index) != self.config["text_count"]:
            raise ValueError("Panel counts differ")
        for i, row in enumerate(self.rows):
            target = (self.text_index[row["candidate_text_sha256"]], self.text_index.get(row["adopted_text_sha256"], -1))
            if row["adopted_text_sha256"] is not None and row["adopted_text_sha256"] not in self.text_index:
                raise ValueError("Unknown adopted text")
            if tuple(self.text_indices[i]) != target:
                raise ValueError("Row text indices disagree with SHA bindings")
        self.indices = np.array([i for i, row in enumerate(self.rows) if split is None or row["split"] == split], dtype=np.int64)
        self.gains = None
        if include_labels:
            for relative, digest in self.config["label_files"].items():
                path = (self.root / relative).resolve()
                if path.parent != (self.root / "labels").resolve() or file_sha(path) != digest:
                    raise ValueError("Label path or digest invalid")
            if read_json(self.root / "labels/sample-ids.json") != [r["sample_id"] for r in self.rows]:
                raise ValueError("Label IDs do not align with feature rows")
            self.gains = np.load(self.root / "labels/realized-gain.npy", mmap_mode="r", allow_pickle=False)
            if self.gains.shape != (count,) or self.gains.dtype != np.float64 or not np.isfinite(self.gains).all():
                raise ValueError("Invalid label array")

    def __len__(self):
        return len(self.indices)

    def batch(self, indices):
        ii = np.asarray(indices)
        if ii.ndim != 1 or not np.issubdtype(ii.dtype, np.integer) or np.any(ii < 0) or np.any(ii >= len(self)):
            raise IndexError("Expected valid integer indices local to the split view")
        absolute = self.indices[ii]
        state, claims = self.state[absolute], self.claims[absolute]
        pair = self.text_indices[absolute]
        def lookup(column):
            result = np.zeros((len(absolute), self.dimension), dtype=np.float32)
            for j, key in enumerate(pair[:, column]):
                if key >= 0:
                    result[j] = normalize_prefix(self.vectors[key], self.dimension)
            return result
        body, adopted = lookup(0), lookup(1)
        delta, product = abs(body-adopted), body*adopted
        features = dict(state=state, claims=claims, body=body, adopted=adopted, absolute_difference=delta, product=product,
            semantic=np.c_[state, claims], fulltext_state=np.c_[state, claims, body, adopted, delta, product],
            fulltext_state_without_candidate_claims=np.c_[state, body, adopted, delta, product])
        if not all(np.isfinite(x).all() for x in features.values()):
            raise ValueError("Nonfinite batch inputs")
        result = dict(sample_id=[self.rows[i]["sample_id"] for i in absolute],
            physical_match_id=[self.rows[i]["physical_match_id"] for i in absolute],
            split=[self.rows[i]["split"] for i in absolute], features=features,
            candidate_text_index=pair[:, 0], adopted_text_index=pair[:, 1], f_model_sha256=self.config["f_model_sha256"])
        if self.include_labels:
            result["realized_gain"] = self.gains[absolute]
        return result

    def iter_batches(self, batch_size=64):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        for start in range(0, len(self), batch_size):
            yield self.batch(np.arange(start, min(start + batch_size, len(self))))
