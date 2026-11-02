"""Freeze a role-specific vector archive once every required local cache exists."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from dataclasses import replace

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from football_research.acquisition.postread_semantic_embedding import EncoderContract


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--roles',nargs='+',default=['f_train','f_dev'])
    p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'embeddings.npz').exists(): raise FileExistsError('Already frozen')
    source=args.data/'sequences.jsonl'
    source_bytes=source.read_bytes()
    sequences=[json.loads(s) for s in source_bytes.decode('utf-8').split('\n') if s.strip()]
    selected=[r for r in sequences if r.get('eligible') and r.get('fit_role',r['split_role']) in args.roles]
    required=sorted({doc.removeprefix('sha256:') for r in selected for doc in r['document_ids']})
    base=EncoderContract()
    contracts=[base,replace(base,max_input_tokens=32768,implementation='postread-qwen-full-history-long-v1')]
    roots=[Path('E:/FootballMind/embeddings/postread-qwen-frozen-v1')/c.fingerprint() for c in contracts]
    vectors={}; records={}
    while True:
        missing=[]
        for sha in required:
            if sha in vectors: continue
            found=False
            for root,contract in zip(roots,contracts):
                meta=root/(sha+'.json');file=root/(sha+'.npy')
                if not meta.exists() or not file.exists():continue
                entry=json.loads(meta.read_text(encoding='utf-8'))
                if entry['text_sha256']!=sha or entry['contract_fingerprint']!=contract.fingerprint():
                    raise ValueError('Cache binding mismatch')
                raw=file.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=entry['vector_file_sha256']:raise ValueError('Cache SHA mismatch')
                v=np.load(file,allow_pickle=False)
                if v.shape!=(1024,) or v.dtype!=np.float32 or not np.isfinite(v).all() or abs(np.linalg.norm(v)-1)>1e-5:
                    raise ValueError('Invalid frozen vector')
                vectors[sha]=v;records[sha]={'text_sha256':sha,'vector_path':str(file),'vector_file_sha256':entry['vector_file_sha256'],'contract_fingerprint':contract.fingerprint()}
                found=True;break
            if not found:missing.append(sha)
        status={'roles':args.roles,'matches':len(selected),'required':len(required),'ready':len(vectors),'missing':len(missing)}
        (args.output/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        print(json.dumps(status),flush=True)
        if not missing:break
        if not args.wait:raise SystemExit(2)
        time.sleep(15)
    if source.read_bytes()!=source_bytes:raise ValueError('Dataset changed while packing')
    with (args.output/'embeddings.npz').open('xb') as f:
        np.savez_compressed(f,**vectors)
    (args.output/'index.jsonl').write_text(''.join(json.dumps(records[s])+'\n' for s in required),encoding='utf-8')
    (args.output/'manifest.json').write_text(json.dumps({**status,
        'sequences_sha256':hashlib.sha256(source_bytes).hexdigest(),
        'embedding_file_sha256':hashlib.sha256((args.output/'embeddings.npz').read_bytes()).hexdigest(),
        'encoding_repeated':False,'outcomes_read':False,'model_trained':False},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
