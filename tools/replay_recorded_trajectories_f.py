"""Frozen new-f diagnosis on actual recorded adoption order; never old-gain training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from football_research.acquisition.sequential_evidence_f import FrozenF
from football_research.acquisition.sequential_postread_g import VectorStore,rows,sha_file,save_rows,save_json
from football_research.acquisition.full_history_sequence_metadata import sequence_metadata,METADATA_NAMES


def execute(args):
    if args.output.exists():raise FileExistsError('Frozen replay output must be new')
    records=rows(args.inputs/'new-f-replay-inputs.jsonl');source_sha=sha_file(args.inputs/'new-f-replay-inputs.jsonl')
    model=FrozenF.load(args.f_model);vectors=VectorStore(args.vectors)
    if list(model.config.metadata_names)!=METADATA_NAMES:raise ValueError('Frozen f metadata contract differs')
    output=[];failures=[];started=time.perf_counter()
    for record in records:
        if not record['new_f_input_replayable']:continue
        run_id=record['run_id'];mid=record['physical_match_id']
        try:
            docs=[];vv=[]
            for item in record['documents']:
                p=Path(item['body_path']);text=p.read_text(encoding='utf8')
                if sha_file(p)!=item['text_sha256']:raise ValueError('Body SHA mismatch')
                docs.append(dict(item,body_characters=len(text)))
                vv.append(vectors.get(item['text_sha256']))
            if not docs:raise ValueError('No adopted full-body sequence')
            meta=np.asarray(sequence_metadata(docs,record['identity']['kickoff'][:10]),dtype=float)
            trace=model.predict_prefixes(record['prior'],np.stack(vv),meta,order=[d['document_id'] for d in docs])
            # This point is after all model inference. Outcome is used only for loss.
            outcome=record.get('independent_outcome');y=outcome['outcome_class'] if outcome else None
            for i,step in enumerate(trace):
                p=step['probabilities'];before=trace[i-1]['probabilities'] if i else None
                gain=float(np.log(p[y])-np.log(before[y])) if y is not None and before else None
                output.append(dict(run_id=run_id,physical_match_id=mid,stratum=record['stratum'],prefix_length=i,
                    adopted_document_ids=step['used_input_ids'],new_f_probabilities=p,new_f_hidden_state=step['hidden_state'],
                    new_f_model_sha256=model.f_model_sha256,original_f_binding=record['original_f_binding'],
                    actual_adoption_order_preserved=True,source_trace=record['source_path'],
                    outcome_class=y,independent_outcome_source=outcome,loss=float(-np.log(p[y])) if y is not None else None,
                    new_f_step_gain=gain,old_gain_reused=False,training_row_created=False,
                    metadata_unknown_policy='missing source type/target aliases/publication retain shared helper unknown flags',
                    source_available_at_policy='no reconstruction of first public availability',
                    diagnostic_not_new_prospective_prediction=True))
        except (ValueError,KeyError,FileNotFoundError,RuntimeError) as exc:
            failures.append(dict(run_id=run_id,physical_match_id=mid,error_type=type(exc).__name__,reason=str(exc),gain=None))
    args.output.mkdir(parents=True);save_rows(args.output/'prefix-predictions.jsonl',output);save_rows(args.output/'failures.jsonl',failures)
    final=[]
    for rid in sorted({r['run_id'] for r in output}):
        rr=[r for r in output if r['run_id']==rid];initial=rr[0];last=rr[-1]
        final.append(dict(run_id=rid,physical_match_id=last['physical_match_id'],stratum=last['stratum'],adopted_documents=last['prefix_length'],
            p_before=initial['new_f_probabilities'],p_after=last['new_f_probabilities'],loss_before=initial['loss'],loss_after=last['loss'],
            total_new_f_gain=initial['loss']-last['loss'] if last['loss'] is not None else None))
    save_json(args.output/'summary.json',dict(frozen_f_sha256=model.f_model_sha256,input_sha256=source_sha,completed_runs=len(final),
        independent_matches=len({r['physical_match_id'] for r in final}),runs=final,failures=len(failures),
        elapsed_seconds=time.perf_counter()-started,new_network_requests=0,training_calls=0,old_labels_reused=False,
        limitations=['Prior recorded adoption order is not chosen by new g','Repeated control versions are one physical match',
            'Previously examined settled runs are development diagnosis, not an independent blind test',
            'New f outputs are offline replay results, never backdated prospective predictions']))
    if source_sha!=sha_file(args.inputs/'new-f-replay-inputs.jsonl'):raise ValueError('Input changed during replay')
    save_json(args.output/'manifest.json',{p.name:sha_file(p) for p in args.output.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--f-model',type=Path,required=True)
    p.add_argument('--vectors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import torch
    torch.set_num_threads(1)
    def offline(*a,**k):raise RuntimeError('Offline frozen trajectory replay only')
    socket.create_connection=socket.socket.connect=socket.getaddrinfo=offline
    execute(args)


if __name__=='__main__':main()
