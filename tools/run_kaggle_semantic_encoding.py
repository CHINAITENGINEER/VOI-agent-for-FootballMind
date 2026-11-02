"""Standalone immutable Qwen encoding job runner for Kaggle or another CUDA host.

Only numpy is needed for --verify-input-only. Encoding additionally requires
torch, transformers and huggingface_hub. No FootballMind package is imported.
The output is a separate, job-bound artifact; this script never imports caches.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import socket
import time
import uuid
import numpy as np

REPO='Qwen/Qwen3-Embedding-0.6B'
REVISION='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
WEIGHTS_SHA='0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'
BASE_CONTRACT=dict(repository=REPO,revision=REVISION,model_sha256=WEIGHTS_SHA,dimension=1024,
    pooling='last_nonpadding_token',normalization='l2_then_matryoshka_prefix_then_l2',
    text_policy='verbatim_no_prompt_no_title_injection_no_truncation',max_input_tokens=8192,
    dtype='float32',implementation='postread-qwen-frozen-v1')
MIN_COSINE=.99999
MAX_ABS=.001
LONG_CONTRACT=dict(BASE_CONTRACT,max_input_tokens=32768,implementation='postread-qwen-full-history-long-v1')


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def fingerprint(contract):return hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def read_json(path):return json.loads(Path(path).read_bytes().decode('utf-8-sig'))
def valid_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def within(root,relative):
    if not isinstance(relative,str) or not relative or '\\' in relative or PureWindowsPath(relative).drive:
        raise ValueError('Input paths must be explicit relative POSIX paths')
    p=PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Path escapes input directory')
    root=Path(root).resolve();target=(root/relative).resolve()
    if not target.is_relative_to(root):raise ValueError('Symlink escapes input directory')
    return target


def atomic_bytes(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
def atomic_json(path,data):atomic_bytes(path,json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False).encode('utf8'))
def atomic_vector(path,vector):
    import io
    b=io.BytesIO();np.save(b,vector,allow_pickle=False);atomic_bytes(path,b.getvalue())


def verify_vector(vector):
    if vector.shape!=(1024,) or vector.dtype!=np.float32 or not np.isfinite(vector).all():raise ValueError('Expected finite float32 vector with 1024 elements')
    if abs(float(np.linalg.norm(vector))-1)>1e-5:raise ValueError('Vector must be L2-normalized')
    return vector


def compare_control(reference,actual):
    verify_vector(reference);verify_vector(actual)
    a=reference.astype(np.float64);b=actual.astype(np.float64)
    cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)));difference=float(np.max(np.abs(a-b)))
    return dict(cosine=cosine,max_abs_difference=difference,passed=bool(cosine>=MIN_COSINE and difference<=MAX_ABS))


def verify_inputs(input_dir):
    input_dir=Path(input_dir).resolve();job_path=input_dir/'job.json';list_path=input_dir/'texts.jsonl'
    job=read_json(job_path)
    if job.get('schema')!='footballmind-kaggle-semantic-job-v1' or not isinstance(job.get('job_id'),str) or not job['job_id']:raise ValueError('Invalid job identity/schema')
    model=job.get('model',{})
    if model.get('repo')!=REPO or model.get('revision')!=REVISION:raise ValueError('Frozen model repository/revision differs')
    hashes=model.get('files_sha256')
    if not isinstance(hashes,dict) or hashes.get('model.safetensors')!=WEIGHTS_SHA:raise ValueError('Frozen weights SHA missing or different')
    for name in ('config.json','tokenizer.json','tokenizer_config.json'):
        if name not in hashes:raise ValueError('Missing frozen model/tokenizer file hash: '+name)
    for name,digest in hashes.items():
        within(input_dir,name)
        if not valid_sha(digest):raise ValueError('Invalid model file SHA')
    contracts=job.get('contracts')
    if not isinstance(contracts,dict) or not contracts:raise ValueError('No encoder contracts')
    for key,contract in contracts.items():
        if not isinstance(contract,dict) or set(contract)!=set(BASE_CONTRACT):raise ValueError('Unexpected effective contract fields')
        if contract not in (BASE_CONTRACT,LONG_CONTRACT):raise ValueError('Frozen encoding contract changed')
        limit=contract['max_input_tokens']
        if isinstance(limit,bool) or not isinstance(limit,int) or limit<1:raise ValueError('Explicit positive token limit required')
        if fingerprint(contract)!=key:raise ValueError('Contract fingerprint mismatch')
    records=[];seen=set();total_bytes=0
    for number,line in enumerate(list_path.read_bytes().decode('utf-8-sig').split('\n'),1):
        if not line.strip():continue
        r=json.loads(line)
        if set(r)-{'text_sha256','text_path','contract_fingerprint','token_count','role','task_group'}:raise ValueError('Unexpected text record fields, including labels, are forbidden')
        if any(k in r and not isinstance(r[k],str) for k in ('role','task_group')):raise ValueError('Role/task_group must be plain metadata strings')
        digest=r.get('text_sha256');contract=contracts.get(r.get('contract_fingerprint'))
        if not valid_sha(digest) or digest in seen:raise ValueError('Text SHA must be unique and valid')
        if contract is None:raise ValueError('Unknown text encoder contract')
        count=r.get('token_count')
        if isinstance(count,bool) or not isinstance(count,int) or count<1 or count>contract['max_input_tokens']:raise ValueError('Invalid declared token count; no truncation allowed')
        body=within(input_dir,r['text_path']).read_bytes()
        text=body.decode('utf8')
        if not text.strip() or hashlib.sha256(body).hexdigest()!=digest:raise ValueError('Exact UTF8 body SHA mismatch or empty text')
        records.append(r);seen.add(digest);total_bytes+=len(body)
    if not records:raise ValueError('No texts')
    if job.get('expected_texts',len(records))!=len(records) or job.get('texts_sha256',sha_file(list_path))!=sha_file(list_path):raise ValueError('Job text-list count/hash differs')
    controls=job.get('controls',[])
    if len(controls)!=8 or len({r.get('text_sha256') for r in controls})!=8:raise ValueError('Exactly eight distinct CPU controls required')
    control_info=[]
    for r in controls:
        if r.get('text_sha256') not in seen:raise ValueError('Control body must be in texts.jsonl')
        path=within(input_dir,r['vector_path']);digest=sha_file(path)
        if r.get('vector_file_sha256')!=digest:raise ValueError('CPU reference vector SHA differs from job binding')
        verify_vector(np.load(path,allow_pickle=False))
        control_info.append(dict(text_sha256=r['text_sha256'],reference_vector_path=r['vector_path'],reference_file_sha256=digest))
    audit=dict(schema='footballmind-kaggle-semantic-input-verification-v1',job_id=job['job_id'],job_sha256=sha_file(job_path),
        texts_sha256=sha_file(list_path),expected_texts=len(records),body_bytes=total_bytes,model=model,
        contract_fingerprints=sorted(contracts),controls=control_info,token_counts_tokenizer_checked=False,
        note='Declared counts and byte hashes checked; exact tokenizer counts are rechecked during encoding',verified_at=utc())
    return job,records,audit


def verify_model_files(model_dir,job):
    actual={}
    for name,expected in job['model']['files_sha256'].items():
        # HF snapshots legitimately symlink to their cache blob directory. Exact
        # file hashes, rather than snapshot-path containment, bind model bytes.
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name or PureWindowsPath(name).drive:raise ValueError('Unsafe model filename')
        digest=sha_file(Path(model_dir)/name)
        if digest!=expected:raise ValueError('Frozen model/tokenizer hash mismatch: '+name)
        actual[name]=digest
    return actual


class Encoder:
    def __init__(self,model_dir,job,device):
        import torch
        from transformers import AutoModel,AutoTokenizer
        if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
        self.device=torch.device(device);self.torch=torch
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,padding_side='left',trust_remote_code=False)
        self.model=AutoModel.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='sdpa').to(self.device).eval()
        self.model.requires_grad_(False)
        if self.model.config.hidden_size!=1024:raise ValueError('Unexpected model hidden dimension')
        if any(p.dtype!=torch.float32 for p in self.model.parameters()):raise ValueError('Model parameters must stay float32')
        self.execution=dict(device=str(self.device),device_name=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'CPU',
            model_dtype='float32',output_dtype='float32',attention_implementation='sdpa',tf32_enabled=False,
            batch_size=1,model_training=False,prompt_injected=False,
            software={name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','huggingface_hub')},cuda_version=torch.version.cuda)

    def encode(self,text,contract,expected_tokens):
        torch=self.torch;inputs=self.tokenizer(text,return_tensors='pt',truncation=False)
        count=int(inputs['attention_mask'].sum())
        if count!=expected_tokens:raise ValueError(f'Token count differs: declared {expected_tokens}, actual {count}')
        if count>contract['max_input_tokens'] or count>getattr(self.model.config,'max_position_embeddings',count):raise ValueError('Full input exceeds explicit/model limit; never truncated')
        inputs={k:v.to(self.device) for k,v in inputs.items()};start=time.perf_counter()
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        with torch.inference_mode(),torch.autocast(device_type=self.device.type,enabled=False):
            hidden=self.model(**inputs,use_cache=False).last_hidden_state
            mask=inputs['attention_mask'];positions=torch.arange(mask.shape[1],device=mask.device)
            last=(mask.to(torch.bool)*(positions+1)).max(dim=1).values-1
            if (last<0).any():raise ValueError('Entirely padded input')
            pooled=hidden[torch.arange(len(hidden),device=hidden.device),last]
            vector=torch.nn.functional.normalize(pooled.float(),p=2,dim=1)[0].cpu().numpy()
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        return verify_vector(vector),dict(token_count=count,inference_seconds=time.perf_counter()-start,characters=len(text),truncated_tokens=0)


def cached_entry(output,r,job,audit):
    meta_path=output/(r['text_sha256']+'.json');vector_path=output/(r['text_sha256']+'.npy')
    if not meta_path.exists():return None  # orphan vector is uncommitted, safe to recompute
    meta=read_json(meta_path)
    for key,expected in (('text_sha256',r['text_sha256']),('job_id',job['job_id']),('job_sha256',audit['job_sha256']),('texts_sha256',audit['texts_sha256']),('contract_fingerprint',r['contract_fingerprint']),('token_count',r['token_count'])):
        if meta.get(key)!=expected:raise ValueError('Resume cache binding differs: '+key)
    if meta.get('effective_contract')!=job['contracts'][r['contract_fingerprint']] or meta.get('model')!=job['model']:raise ValueError('Resume model/contract differs')
    if not vector_path.exists() or sha_file(vector_path)!=meta.get('vector_file_sha256'):raise ValueError('Committed vector missing or corrupted')
    verify_vector(np.load(vector_path,allow_pickle=False))
    return meta


def write_entry(output,r,vector,measurement,execution,job,audit):
    verify_vector(vector);path=output/(r['text_sha256']+'.npy');atomic_vector(path,vector)
    meta=dict(**measurement,text_sha256=r['text_sha256'],contract_fingerprint=r['contract_fingerprint'],job_id=job['job_id'],
        job_sha256=audit['job_sha256'],texts_sha256=audit['texts_sha256'],model=job['model'],effective_contract=job['contracts'][r['contract_fingerprint']],
        vector_path=path.name,metadata_path=r['text_sha256']+'.json',vector_file_sha256=sha_file(path),dimension=1024,dtype='float32',
        execution_metadata=execution,encoded_at=utc(),no_prompt=True)
    atomic_json(output/meta['metadata_path'],meta);return meta


def _execute(args):
    job,records,audit=verify_inputs(args.input_dir);output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    old=output/'input-verification.json'
    if old.exists():
        previous=read_json(old)
        if previous.get('job_sha256')!=audit['job_sha256'] or previous.get('texts_sha256')!=audit['texts_sha256']:raise ValueError('Output belongs to another job; choose a fresh output directory')
    atomic_json(old,audit)
    for name in ('job.json','texts.jsonl'):atomic_bytes(output/'input-contract'/name,(Path(args.input_dir)/name).read_bytes())
    completion=dict(schema='footballmind-kaggle-semantic-completion-v1',job_id=job['job_id'],job_sha256=audit['job_sha256'],texts_sha256=audit['texts_sha256'],
        model=job['model'],expected_texts=len(records),completed_texts=0,controls_passed=False,started_at=utc(),status='started')
    if args.verify_input_only:
        completion['status']='inputs_verified_only';atomic_json(output/'completion.json',completion);return completion
    if args.batch_size!=1:raise ValueError('This frozen runner supports batch_size=1 only; no hidden batching change')
    try:
        if args.model_dir:model_dir=Path(args.model_dir)
        else:
            from huggingface_hub import snapshot_download
            model_dir=Path(snapshot_download(repo_id=REPO,revision=REVISION,allow_patterns=list(job['model']['files_sha256'])))
        verify_model_files(model_dir,job);encoder=Encoder(model_dir,job,args.device)
        lookup={r['text_sha256']:r for r in records};control_vectors={};control_measurements={};comparisons=[]
        for control in job['controls']:
            r=lookup[control['text_sha256']];text=within(args.input_dir,r['text_path']).read_bytes().decode('utf8')
            vector,measurement=encoder.encode(text,job['contracts'][r['contract_fingerprint']],r['token_count'])
            reference=np.load(within(args.input_dir,control['vector_path']),allow_pickle=False)
            relative='controls/'+r['text_sha256']+'.npy';atomic_vector(output/relative,vector)
            comparisons.append(dict(text_sha256=r['text_sha256'],reference_vector_path=control['vector_path'],reference_file_sha256=control['vector_file_sha256'],
                gpu_vector_path=relative,gpu_vector_file_sha256=sha_file(output/relative),**compare_control(reference,vector)))
            control_vectors[r['text_sha256']]=vector;control_measurements[r['text_sha256']]=measurement
        passed=all(r['passed'] for r in comparisons)
        atomic_json(output/'controls.json',dict(schema='footballmind-kaggle-semantic-controls-v1',job_id=job['job_id'],job_sha256=audit['job_sha256'],
            thresholds=dict(minimum_cosine=MIN_COSINE,maximum_absolute_difference=MAX_ABS),all_passed=passed,controls=comparisons,execution_metadata=encoder.execution))
        if not passed:
            completion['status']='control_validation_failed';atomic_json(output/'completion.json',completion)
            raise RuntimeError('GPU/CPU control tolerance failed; no encoding batch accepted')
        completion['controls_passed']=True;index=[];resumed=0
        for r in records:
            meta=cached_entry(output,r,job,audit)
            if meta is not None:resumed+=1
            else:
                key=r['text_sha256']
                if key in control_vectors:vector,measurement=control_vectors[key],control_measurements[key]
                else:
                    text=within(args.input_dir,r['text_path']).read_bytes().decode('utf8')
                    if hashlib.sha256(text.encode('utf8')).hexdigest()!=key:raise ValueError('Body changed during encoding')
                    vector,measurement=encoder.encode(text,job['contracts'][r['contract_fingerprint']],r['token_count'])
                meta=write_entry(output,r,vector,measurement,encoder.execution,job,audit)
            index.append({k:meta[k] for k in ('text_sha256','contract_fingerprint','job_id','job_sha256','vector_path','metadata_path','vector_file_sha256','dimension','dtype','token_count')})
            completion['completed_texts']=len(index)
            if len(index)%25==0:atomic_json(output/'progress.json',dict(completion,updated_at=utc(),resumed=resumed))
        atomic_bytes(output/'index.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in index).encode('utf8'))
        tmp=output/('embeddings.'+uuid.uuid4().hex+'.tmp')
        with tmp.open('xb') as f:
            np.savez_compressed(f,**{r['text_sha256']:np.load(output/r['vector_path'],allow_pickle=False) for r in index});f.flush();os.fsync(f.fileno())
        os.replace(tmp,output/'embeddings.npz')
        if sha_file(Path(args.input_dir)/'job.json')!=audit['job_sha256'] or sha_file(Path(args.input_dir)/'texts.jsonl')!=audit['texts_sha256']:raise ValueError('Job source changed during encoding')
        completion.update(status='completed',completed_texts=len(index),resumed_texts=resumed,finished_at=utc(),execution_metadata=encoder.execution,
            index_sha256=sha_file(output/'index.jsonl'),embeddings_sha256=sha_file(output/'embeddings.npz'),controls_sha256=sha_file(output/'controls.json'))
        atomic_json(output/'completion.json',completion);return completion
    except Exception as exc:
        if completion['status']!='control_validation_failed':completion['status']='failed'
        completion.update(error_type=type(exc).__name__,error=str(exc),finished_at=utc())
        atomic_json(output/'completion.json',completion);raise


@contextmanager
def output_lock(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);lock=output/'.encoding.lock'
    if lock.exists():
        previous=read_json(lock)
        if previous.get('host')==socket.gethostname():
            try:os.kill(int(previous['pid']),0)
            except ProcessLookupError:lock.unlink()
        if lock.exists():raise RuntimeError('Output has another/live or unverified stale writer lock; do not run concurrent writers')
    with lock.open('x',encoding='utf8') as f:json.dump(dict(pid=os.getpid(),host=socket.gethostname(),created_at=utc()),f)
    try:yield
    finally:lock.unlink(missing_ok=True)


def execute(args):
    with output_lock(args.output_dir):return _execute(args)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=1);p.add_argument('--verify-input-only',action='store_true')
    result=execute(p.parse_args());print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
