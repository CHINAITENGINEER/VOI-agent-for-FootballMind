"""Freeze existing historical bodies for local, resumable Qwen encoding.

This prepares an unlabeled research pool, not g training rows. It never opens
match outcomes, generates claims, fits a model, or accesses the network.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from football_research.acquisition.postread_semantic_embedding import (
    EncoderContract, LocalSemanticEncoder, SemanticCache, text_sha,
)

ROOT = Path("E:/FootballMind")
DEFAULT_OUTPUT = ROOT / "experiments/postread-semantic-training-ready-20261025-v1/corpus-queue"
FIELDS = ("candidate_text_id", "physical_match_id", "division", "season", "match_date",
          "home_team", "away_team", "source_url", "source_domain", "title", "published_at",
          "updated_at", "captured_at", "source_available_at", "historical_version_unknown",
          "association_evidence", "match_binding_review_status")


def read_rows(path):
    # Deliberately split only physical newlines: U+0085/U+2028 may occur in text.
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_tokenizer(model_path, contract):
    receipt = json.loads((ROOT / "models/qwen3-embedding-0.6b-pinned.json").read_text(encoding="utf-8"))
    if Path(model_path).name != contract.revision or receipt["revision"] != contract.revision:
        raise ValueError("Tokenizer snapshot revision differs from frozen contract")
    for record in receipt["files"]:
        if record["file"] in {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}:
            if digest(Path(model_path) / record["file"]) != record["sha256"]:
                raise ValueError("Tokenizer asset differs from pinned receipt")


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def immutable_bytes(path, value):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"Existing preparation differs: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(value); stream.flush(); os.fsync(stream.fileno())


def immutable_rows(path, rows):
    immutable_bytes(path, ("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)
                                  + "\n" for r in rows)).encode("utf-8"))


def extract_bodies(candidates):
    """Resolve the declared nonempty JSONL record; require article ID and SHA."""
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["text_reference"]["path"]].append(row)
    bodies = {}
    for path, requested in grouped.items():
        positions = {int(r["text_reference"]["jsonl_record"]) for r in requested}
        records = {i: row for i, row in enumerate(read_rows(path), 1) if i in positions}
        for candidate in requested:
            ref = candidate["text_reference"]
            article = records[int(ref["jsonl_record"])]
            if article["article_id"] != candidate["candidate_text_id"] or article["article_id"] != ref["article_id"]:
                raise ValueError("Article ID differs from declared source record")
            text = article["body_text"]
            sha = text_sha(text)
            if not text.strip() or sha != candidate["text_sha256"] or sha != ref["body_sha256"]:
                raise ValueError("Body differs from declared text SHA")
            bodies[sha] = text
    return bodies


@contextmanager
def writer_lock(path):
    """OS releases the lock on a crash; the empty lock file may safely remain."""
    stream = Path(path).open("a+b")
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def prepare(args):
    from transformers import AutoTokenizer
    contract = EncoderContract()
    verify_tokenizer(args.model, contract)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "texts").mkdir(exist_ok=True)
    with writer_lock(output / ".prepare.lock"):
        source_hash = digest(args.candidates)
        exclusion_hash = digest(args.exclusions)
        candidates = list(read_rows(args.candidates))
        # Use only established quality exclusion metadata; no numeric target is read.
        excluded = {(r["candidate_text_id"], r["body_sha256"]):
                    {k: r[k] for k in ("candidate_text_id", "body_sha256", "status", "reason")}
                    for r in read_rows(args.exclusions)}
        rejected, accepted = [], []
        seen = set()
        for row in candidates:
            key = (row["physical_match_id"], row["candidate_text_id"], row["text_sha256"])
            if key in seen:
                raise ValueError("Duplicate candidate-to-match record")
            seen.add(key)
            exclusion = excluded.get((row["candidate_text_id"], row["text_sha256"]))
            if exclusion:
                rejected.append({"physical_match_id": row["physical_match_id"], **exclusion})
            else:
                accepted.append(row)
        bodies = extract_bodies(accepted)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        tokens = {sha: len(tokenizer.encode(text, truncation=False)) for sha, text in bodies.items()}
        for sha, text in bodies.items():
            immutable_bytes(output / "texts" / f"{sha}.txt", text.encode("utf-8"))
        queue = []
        for row in accepted:
            sha = row["text_sha256"]
            queue.append({**{k: row.get(k) for k in FIELDS}, "text_sha256": sha,
                          "text_path": str(output / "texts" / f"{sha}.txt"),
                          "source_text_reference": row["text_reference"],
                          "characters": len(bodies[sha]), "token_count": tokens[sha],
                          "encoding_status": "ready" if tokens[sha] <= contract.max_input_tokens else "over_token_limit",
                          "historical_system_available_at": None,
                          "source_record_system_available_at": row.get("system_available_at"),
                          "system_clock_interpretation": "2026 current-version retrieval only; not historical availability",
                          "research_stratum": "historical_reconstruction_current_body_version",
                          "same_f_label_status": "not_generated_by_this_preparation",
                          "supervised_training_eligible": False})
        immutable_rows(output / "candidates.jsonl", queue)
        immutable_rows(output / "exclusions.jsonl", rejected)
        summary = {
            "schema": "postread-semantic-unlabeled-corpus-v1", "encoder": asdict(contract),
            "source": str(args.candidates), "source_sha256": source_hash,
            "exclusion_source": str(args.exclusions), "exclusion_source_sha256": exclusion_hash,
            "candidate_rows_before": len(candidates), "excluded_known_contaminated": len(rejected),
            "candidate_rows": len(queue), "unique_texts": len(bodies),
            "independent_matches": len({r["physical_match_id"] for r in queue}),
            "division_matches": {div: len({r["physical_match_id"] for r in queue if r["division"] == div})
                                 for div in sorted({r["division"] for r in queue})},
            "encoding_status_counts": dict(Counter(r["encoding_status"] for r in queue)),
            "body_bytes": sum(len(t.encode("utf-8")) for t in bodies.values()),
            "total_tokens": sum(tokens.values()), "maximum_tokens": max(tokens.values()),
            "no_truncation": True, "training_started": False, "claims_generated": False,
            "outcomes_opened": False, "labels_generated": 0, "http_requests": 0,
            "inputs_unchanged": source_hash == digest(args.candidates) and exclusion_hash == digest(args.exclusions),
            "candidate_file_sha256": digest(output / "candidates.jsonl"),
        }
        immutable_bytes(output / "preparation.json", (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode())
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def encode(args):
    contract = EncoderContract()
    verify_tokenizer(args.model, contract)
    output = args.output
    preparation = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    if preparation["encoder"] != asdict(contract) or digest(output / "candidates.jsonl") != preparation["candidate_file_sha256"]:
        raise ValueError("Prepared pool or encoder contract changed")
    with writer_lock(output / ".encode.lock"):
        cache = SemanticCache(ROOT / "embeddings/postread-qwen-frozen-v1", contract)
        rows = list(read_rows(output / "candidates.jsonl"))
        unique = {r["text_sha256"]: r for r in rows}
        encoder = None; fresh = 0; cache_hits = 0; statuses = []
        started = time.perf_counter()
        for sha, row in sorted(unique.items(), key=lambda item: (item[1]["token_count"], item[0])):
            # Fixed SHA verification protects resume even if files are later edited.
            text = Path(row["text_path"]).read_bytes().decode("utf-8")
            if text_sha(text) != sha:
                raise ValueError("Prepared text changed")
            status = {"text_sha256": sha, "token_count": row["token_count"]}
            entry = cache.get(text)
            if entry is not None:
                status.update(status="encoded", cache_hit=True)
                cache_hits += 1
            elif row["token_count"] > contract.max_input_tokens:
                status.update(status="over_token_limit", truncated=False)
            elif fresh >= args.limit:
                status.update(status="pending", reason="bounded_encoding_limit")
            else:
                if encoder is None:
                    encoder = LocalSemanticEncoder(args.model, contract)
                vector, meta = encoder.encode_one(text)
                cache.put(text, vector, meta)
                fresh += 1
                status.update(status="encoded", cache_hit=False, inference_seconds=meta["inference_seconds"])
                print(json.dumps({"fresh_encoded": fresh, **status}), flush=True)
            statuses.append(status)
        # Status is a disposable projection of the immutable content-addressed cache.
        temp = output / ("encoding-status." + uuid.uuid4().hex + ".tmp")
        immutable_rows(temp, statuses)
        os.replace(temp, output / "encoding-status.jsonl")
        report = {"fresh_encoded": fresh, "verified_cache_hits": cache_hits,
                  "status_counts": dict(Counter(r["status"] for r in statuses)),
                  "elapsed_seconds": time.perf_counter() - started,
                  "limit": args.limit, "http_requests": 0, "training": False,
                  "encoder_contract_fingerprint": contract.fingerprint(), "cache_path": str(cache.root)}
        atomic_json(output / "encoding-status.json", report)
        print(json.dumps(report), flush=True)


def main():
    def no_network(*args, **kwargs):
        raise RuntimeError("Corpus preparation/encoding is offline")
    socket.create_connection = socket.socket.connect = socket.getaddrinfo = no_network
    contract = EncoderContract()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "encode"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=ROOT / "models/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots" / contract.revision)
    parser.add_argument("--candidates", type=Path, default=ROOT / "experiments/postread-usable-pool-20261025-v2/usable-candidates.jsonl")
    parser.add_argument("--exclusions", type=Path, default=ROOT / "experiments/postread-clean-expansion-20261025-v1/excluded-review.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Maximum new forward calls; 0 only audits/reuses cache")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    (prepare if args.mode == "prepare" else encode)(args)


if __name__ == "__main__":
    main()
