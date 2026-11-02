"""Encode existing candidate/adopted text offline; never open outcome labels."""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import socket
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from football_research.acquisition.postread_semantic_embedding import (
    EncoderContract, LocalSemanticEncoder, SemanticCache, text_sha, validate_feature_rows)


def main():
    def forbid_network(*args, **kwargs):
        raise RuntimeError("Encoding is local-only; network access is forbidden")
    socket.socket.connect = socket.create_connection = socket.getaddrinfo = forbid_network
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("E:/FootballMind/experiments/postread-clean-expansion-20261025-v1"))
    parser.add_argument("--output", type=Path, default=Path("E:/FootballMind/experiments/postread-qwen-semantic-20261025-v1"))
    args = parser.parse_args()
    contract = EncoderContract()
    model = Path("E:/FootballMind/models/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots") / contract.revision
    source = args.dataset / "features.jsonl"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    features = [json.loads(s) for s in source.read_text(encoding="utf-8").split("\n") if s.strip()]
    validate_feature_rows(features)
    texts = sorted({r[k] for r in features for k in ("candidate_text", "adopted_text") if r[k].strip()}, key=lambda t:(len(t),text_sha(t)))
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "embedding-protocol.json"
    protocol = {"contract": asdict(contract), "dataset": str(args.dataset), "source_sha256": source_hash,
                "text_count": len(texts), "body_count": len({r["candidate_text"] for r in features}),
                "outcomes_opened": False, "encoder_trainable": False, "device": "cpu",
                "representation_only_no_g_scoring": True, "input_texts_uploaded": False,
                "pool_visibility": "all candidates read by g; adopted_text contains only evidence adopted by f",
                "limits": ["Pretrained corpus/football match contamination cannot be independently excluded",
                           "Historical version and timestamp assumptions inherited unchanged"]}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Resume protocol changed")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cache = SemanticCache(Path("E:/FootballMind/embeddings/postread-qwen-frozen-v1"), contract)
    encoder = None; records = []; vectors = {}; started = time.perf_counter()
    for i, text in enumerate(texts):
        entry = cache.get(text); hit = entry is not None
        if entry is None:
            if encoder is None:
                print(json.dumps({"stage": "loading_local_encoder", "path": str(model)}), flush=True)
                encoder = LocalSemanticEncoder(model, contract)
                print(json.dumps({"stage":"loaded","frozen_parameters":sum(p.numel() for p in encoder.model.parameters()),"trainable_parameters":sum(p.numel() for p in encoder.model.parameters() if p.requires_grad)}),flush=True)
            vector, meta = encoder.encode_one(text)
            meta = cache.put(text, vector, meta)
        else:
            vector, meta = entry
        vectors[text_sha(text)] = vector
        records.append({**meta, "cache_hit_this_run": hit})
        print(json.dumps({"stage":"encoded","done":i+1,"total":len(texts),"tokens":meta["token_count"],"seconds":round(meta["inference_seconds"],3),"cache_hit":hit}), flush=True)
    dest = args.output / "embeddings.npz"
    with dest.open("xb") as f:
        np.savez_compressed(f, **vectors)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    report = {**protocol, "records": records, "total_tokens": sum(r["token_count"] for r in records),
              "max_tokens": max(r["token_count"] for r in records), "cache_path": str(cache.root),
              "elapsed_seconds": time.perf_counter()-started, "truncated_tokens":0,
              "vectors_file_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
              "feature_source_unchanged": True}
    (args.output/"embedding-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"stage":"complete","output":str(args.output),"texts":len(vectors),"elapsed_seconds":report["elapsed_seconds"]}),flush=True)


if __name__ == "__main__":
    main()
