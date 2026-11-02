"""Frozen semantic text representation for an independent post-read g experiment.

This is not imported by production runtime. The full-text and adopted-evidence
channels stay separate; outcomes and candidate-after predictions are not inputs.
No model downloads take place here. All inputs must fit the explicit token limit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np


FEATURE_FIELDS = frozenset({
    "sample_id", "physical_match_id", "split", "candidate_text", "candidate_title",
    "adopted_text", "state_numeric", "state_feature_names", "semantic_numeric",
    "semantic_feature_names",
})


def text_sha(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Text must be an explicit string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EncoderContract:
    repository: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    model_sha256: str = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
    dimension: int = 1024
    pooling: str = "last_nonpadding_token"
    normalization: str = "l2_then_matryoshka_prefix_then_l2"
    text_policy: str = "verbatim_no_prompt_no_title_injection_no_truncation"
    max_input_tokens: int = 8192
    dtype: str = "float32"
    implementation: str = "postread-qwen-frozen-v1"

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def normalize_prefix(vector: np.ndarray, dimension: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a finite vector")
    if dimension < 32 or dimension > len(vector):
        raise ValueError("Requested dimension outside the declared MRL range")
    prefix = vector[:dimension].copy()
    norm = np.linalg.norm(prefix)
    return prefix / norm if norm > 0 else prefix


def last_token_pool(hidden, attention_mask):
    """Last attended position, including when left padding is used."""
    import torch
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden/mask shape mismatch")
    if (attention_mask.sum(dim=1) == 0).any():
        raise ValueError("An entirely padded sequence is not text")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = (attention_mask.to(torch.bool) * (positions + 1)).max(dim=1).values - 1
    return hidden[torch.arange(len(hidden), device=hidden.device), last]


class LocalSemanticEncoder:
    """Actual pretrained inference; one text per forward for bounded CPU memory."""
    def __init__(self, model_path: Path, contract: EncoderContract | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.contract = contract or EncoderContract()
        self.model_path = Path(model_path)
        if self.model_path.name != self.contract.revision:
            raise ValueError("Local snapshot revision differs from the frozen contract")
        digest = hashlib.sha256()
        with (self.model_path / "model.safetensors").open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.contract.model_sha256:
            raise ValueError("Local encoder weights differ from the frozen contract")
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, padding_side="left", trust_remote_code=False)
        self.model = AutoModel.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="sdpa").eval()
        self.model.requires_grad_(False)
        if self.model.config.hidden_size != self.contract.dimension:
            raise ValueError("Loaded hidden size differs from the declared embedding size")

    def encode_one(self, text: str):
        import torch
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty evidence is represented by zero, not encoded as language")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        count = int(inputs["attention_mask"].sum())
        if count > self.contract.max_input_tokens:
            raise ValueError(f"Full input has {count} tokens; explicit limit is {self.contract.max_input_tokens}; not truncated")
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model(**inputs, use_cache=False)
            vector = last_token_pool(result.last_hidden_state, inputs["attention_mask"])
            vector = torch.nn.functional.normalize(vector.float(), p=2, dim=1)[0].cpu().numpy()
        if vector.shape != (self.contract.dimension,) or not np.isfinite(vector).all():
            raise ValueError("Invalid model output")
        return vector, {"token_count": count, "truncated_tokens": 0,
                        "characters": len(text), "inference_seconds": time.perf_counter() - started}


class SemanticCache:
    """Immutable text/model-keyed cache. Incomplete entries never count as hits."""
    def __init__(self, root: Path, contract: EncoderContract):
        self.contract = contract
        self.root = Path(root) / contract.fingerprint()
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, text: str):
        key = text_sha(text); meta_path = self.root / (key + ".json")
        vector_path = self.root / (key + ".npy")
        if not meta_path.exists() or not vector_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["text_sha256"] != key or meta["contract_fingerprint"] != self.contract.fingerprint():
            raise ValueError("Embedding cache identity mismatch")
        if hashlib.sha256(vector_path.read_bytes()).hexdigest() != meta["vector_file_sha256"]:
            raise ValueError("Embedding cache digest mismatch")
        vector = np.load(vector_path, allow_pickle=False)
        if vector.shape != (self.contract.dimension,) or not np.isfinite(vector).all():
            raise ValueError("Invalid cached embedding")
        if abs(float(np.linalg.norm(vector)) - 1.0) > 1e-5:
            raise ValueError("Cached nonempty text must have normalized embedding")
        return vector, meta

    def put(self, text: str, vector: np.ndarray, metadata: dict):
        if self.get(text) is not None:
            raise FileExistsError("A valid immutable embedding already exists")
        key = text_sha(text)
        path = self.root / (key + ".npy")
        # The experiment has one writer. Incomplete entries can be resumed without
        # accepting partial vector bytes as an inference result.
        temporary = self.root / (key + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("xb") as f:
            np.save(f, vector, allow_pickle=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        meta = {**metadata, "text_sha256": key, "contract_fingerprint": self.contract.fingerprint(),
                "vector_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        temporary_meta = self.root / (key + "." + uuid.uuid4().hex + ".tmp")
        with temporary_meta.open("x", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary_meta, self.root / (key + ".json"))
        return meta


def validate_feature_rows(features):
    if not features:
        raise ValueError("No feature rows")
    state_names = features[0]["state_feature_names"]
    semantic_names = features[0]["semantic_feature_names"]
    for row in features:
        if set(row) != FEATURE_FIELDS:
            raise ValueError("Explicit pre-label feature contract required; unknown fields rejected")
        for key in ("state", "semantic"):
            names = row[key + "_feature_names"]
            if names != (state_names if key == "state" else semantic_names):
                raise ValueError("Feature order differs across rows")
            if len(row[key + "_numeric"]) != len(names):
                raise ValueError("Feature names/value dimensions differ")
            if any(any(t in name.lower() for t in ("outcome", "realized", "gain", "after")) for name in names):
                raise ValueError("Post-candidate or outcome feature forbidden")
        text_sha(row["candidate_text"]); text_sha(row["adopted_text"])
        if not row["candidate_text"].strip():
            raise ValueError("Candidate body missing")


def build_semantic_design(features, vectors_by_text_sha, dimension: int):
    """Same four text blocks as the lexical experiment, in the same order."""
    validate_feature_rows(features)
    def lookup(text):
        if not text.strip():
            return np.zeros(dimension, dtype=np.float32)
        return normalize_prefix(vectors_by_text_sha[text_sha(text)], dimension)
    body = np.stack([lookup(r["candidate_text"]) for r in features])
    adopted = np.stack([lookup(r["adopted_text"]) for r in features])
    state = np.asarray([r["state_numeric"] for r in features], dtype=float)
    semantic = np.asarray([r["semantic_numeric"] for r in features], dtype=float)
    design = dict(state=state, body=body, semantic=np.c_[state, semantic],
                  fulltext_state=np.c_[state, semantic, body, adopted, np.abs(body-adopted), body*adopted],
                  fulltext_state_without_candidate_claims=np.c_[state, body, adopted, np.abs(body-adopted), body*adopted])
    if not all(np.isfinite(x).all() for x in design.values()):
        raise ValueError("Nonfinite numeric input")
    return design


def predict_frozen_head(bundle, features, vectors_by_text_sha, model_name=None):
    """Research inference from pre-label inputs; no label file is opened."""
    name = model_name or bundle["selected_semantic_model"]
    spec = bundle["fitted"][name]
    if spec["kind"] == "zero":
        validate_feature_rows(features)
        return np.zeros(len(features))
    if spec["kind"] == "constant":
        validate_feature_rows(features)
        return np.full(len(features), spec["value"])
    design = build_semantic_design(features, vectors_by_text_sha, spec["dimension"])
    x = spec["scaler"].transform(design[spec["input_key"]])
    return spec["estimator"].predict(x) * bundle["target_scale"] + bundle["target_mean"]
