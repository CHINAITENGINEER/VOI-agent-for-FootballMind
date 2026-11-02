"""Offline full-history Qwen cache builder with durable per-text progress.

All existing float32 vectors are reused unchanged. Longer texts use the same
weights/pooling with a separately identified 32768-token contract. No text is
silently shortened. A JSONL manifest is captured before inference begins.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from football_research.acquisition.postread_semantic_embedding import (
    EncoderContract, LocalSemanticEncoder, SemanticCache, text_sha)


def json_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new', type=int, default=100000)
    parser.add_argument('--cache-root', type=Path, default=Path('E:/FootballMind/embeddings/postread-qwen-frozen-v1'))
    args = parser.parse_args()
    def offline(*a, **kw): raise RuntimeError('Offline encoder: network forbidden')
    socket.socket.connect = socket.create_connection = socket.getaddrinfo = offline
    args.output.mkdir(parents=True, exist_ok=True)
    source_bytes = args.input.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode('utf-8').split('\n') if line.strip()]
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    (args.output / ('input-' + source_digest + '.jsonl')).write_bytes(source_bytes)
    # One encoding can be shared across matches; do not collapse match records.
    unique = {}
    for row in rows:
        sha = row['text_sha256']
        if sha not in unique or (row.get('match_date') or '9999') < (unique[sha].get('match_date') or '9999'):
            unique[sha] = row
    contract = EncoderContract()
    long_contract = replace(contract, max_input_tokens=32768, implementation='postread-qwen-full-history-long-v1')
    cache = SemanticCache(args.cache_root, contract)
    long_cache = SemanticCache(args.cache_root, long_contract)
    model_path = Path('E:/FootballMind/models/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots') / contract.revision
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    prepared = []
    for sha, row in sorted(unique.items(), key=lambda kv: ((kv[1].get('match_date') or '9999'), kv[0])):
        path = Path(row.get('text_path') or row.get('body_path'))
        text = path.read_bytes().decode('utf-8')
        if text_sha(text) != sha:
            raise ValueError(f'Original text SHA mismatch: {sha}')
        tokens = len(tokenizer.encode(text, truncation=False))
        prepared.append({'text_sha256': sha, 'text_path': str(path), 'token_count': tokens,
                         'match_date': row.get('match_date'), 'physical_match_id': row.get('physical_match_id')})
    atomic_json(args.output/'encoding-plan.json', {'input':str(args.input),'input_sha256':source_digest,
                'input_rows':len(rows),'unique_texts':len(prepared),'short_contract':asdict(contract),
                'long_contract':asdict(long_contract),'total_tokens':sum(r['token_count'] for r in prepared),
                'http_requests':0,'source_texts_uploaded':False,'new_f_or_g_training':False})
    encoder = None
    started = time.perf_counter(); hits = fresh = 0; entries = []; failures = []
    progress_file = args.output / 'encoding-progress.jsonl'
    with progress_file.open('a', encoding='utf-8', buffering=1) as log:
        for i, row in enumerate(prepared):
            sha = row['text_sha256']; text = Path(row['text_path']).read_bytes().decode('utf-8')
            active_cache = cache if row['token_count'] <= contract.max_input_tokens else long_cache
            active_contract = active_cache.contract
            if row['token_count'] > active_contract.max_input_tokens:
                failures.append({**row,'status':'over_model_context','truncated':False}); continue
            entry = active_cache.get(text)
            if entry is not None:
                _, meta = entry; hits += 1; hit = True
            elif fresh >= args.max_new:
                failures.append({**row,'status':'pending_bounded_limit'}); continue
            else:
                if encoder is None:
                    encoder = LocalSemanticEncoder(model_path, contract)
                # The only contract difference is a declared length bound; weights,
                # dtype, pooling and the exact text input remain identical.
                encoder.contract = active_contract
                vector, meta = encoder.encode_one(text)
                meta = active_cache.put(text, vector, meta)
                fresh += 1; hit = False
            item = {**row, 'vector_path':str(active_cache.root/(sha+'.npy')),
                    'vector_file_sha256':meta['vector_file_sha256'],
                    'contract_fingerprint':active_contract.fingerprint(), 'cache_hit':hit,
                    'encoded_at':datetime.now(timezone.utc).isoformat(), 'status':'encoded'}
            entries.append(item)
            log.write(json.dumps(item,ensure_ascii=False)+'\n'); log.flush(); os.fsync(log.fileno())
            if i % 10 == 0 or not hit:
                status={'done':len(entries),'total':len(prepared),'fresh':fresh,'hits':hits,
                        'tokens':row['token_count'],'elapsed_seconds':time.perf_counter()-started,
                        'latest_match_date':row['match_date']}
                atomic_json(args.output/'progress.json',status)
                print(json.dumps(status),flush=True)
    import numpy as np
    vectors=np.stack([np.load(r['vector_path'],allow_pickle=False) for r in entries])
    with (args.output/'vectors.npy').open('wb') as stream:
        np.save(stream,vectors,allow_pickle=False)
    with (args.output/'embedding-index.jsonl').open('w',encoding='utf-8') as stream:
        for index, row in enumerate(entries):
            stream.write(json.dumps({'index':index,**row},ensure_ascii=False)+'\n')
    # An actual portable npz is also supplied to the training interface.
    with (args.output/'embeddings.npz').open('wb') as stream:
        np.savez_compressed(stream, **{r['text_sha256']:vectors[i] for i,r in enumerate(entries)})
    atomic_json(args.output/'failures.json',failures)
    atomic_json(args.output/'completion.json',{'input_sha256':source_digest,'encoded':len(entries),
                'unique_texts':len(prepared),'fresh':fresh,'hits':hits,'failures':len(failures),
                'elapsed_seconds':time.perf_counter()-started,'http_requests':0,
                'vectors_sha256':hashlib.sha256((args.output/'vectors.npy').read_bytes()).hexdigest()})
    print(json.dumps({'complete':True,'encoded':len(entries),'failures':len(failures)}),flush=True)


if __name__ == '__main__':
    main()
