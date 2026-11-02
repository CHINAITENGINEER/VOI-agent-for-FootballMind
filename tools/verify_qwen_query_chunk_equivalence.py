"""CPU-only numerical investigation of full-key/query-chunk SDPA.

This does not patch the Kaggle runner, load old checkpoints, or train a model.
The reference implementation intentionally rejects cached/non-square sequences
and sliding windows: the encoding job is full-sequence, batch-one inference.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F


def repeat_kv_once(value,groups):
    if groups==1:return value
    b,h,l,d=value.shape
    return value[:,:,None,:,:].expand(b,h,groups,l,d).reshape(b,h*groups,l,d)


def query_chunk_sdpa(query,key,value,attention_mask=None,*,chunk_size=256,is_causal=True,scale=None,dropout_p=0.):
    if dropout_p!=0.:raise ValueError('Inference-only: dropout changes RNG sequence across blocks')
    if chunk_size<1:raise ValueError('Positive query chunk required')
    if query.ndim!=4 or key.ndim!=4 or value.ndim!=4:raise ValueError('Expected BHLD tensors')
    if query.shape[0]!=key.shape[0] or key.shape!=value.shape or query.shape[-2]!=key.shape[-2] or query.shape[-1]!=key.shape[-1]:raise ValueError('Full no-cache square sequence required')
    if query.shape[1]%key.shape[1]:raise ValueError('Invalid GQA head ratio')
    length=query.shape[-2]
    if attention_mask is not None:
        if attention_mask.ndim!=4 or attention_mask.shape[-1]!=length or attention_mask.shape[-2] not in (1,length):raise ValueError('Expected complete broadcastable 4D mask')
    groups=query.shape[1]//key.shape[1];key=repeat_kv_once(key,groups);value=repeat_kv_once(value,groups)
    # Full K/V stay available; only the query rows and corresponding mask rows
    # are batched. No key truncation, document segmentation, or output averaging.
    output=torch.empty_like(query);keys=torch.arange(length,device=query.device)
    for start in range(0,length,chunk_size):
        end=min(start+chunk_size,length)
        if attention_mask is not None:
            mask=attention_mask if attention_mask.shape[-2]==1 else attention_mask[...,start:end,:]
        elif is_causal and length>1:
            mask=(keys[None,:]<=torch.arange(start,end,device=query.device)[:,None])[None,None,:,:]
        else:mask=None
        output[:,:,start:end,:]=F.scaled_dot_product_attention(query[:,:,start:end,:],key,value,
            attn_mask=mask,dropout_p=0.,scale=scale,is_causal=False,enable_gqa=False)
    return output


def hf_query_chunk(module,query,key,value,attention_mask,dropout=0.,scaling=None,is_causal=None,**kwargs):
    if kwargs.get('sliding_window') is not None:raise ValueError('Sliding attention not supported by this narrow encoding replacement')
    causal=getattr(module,'is_causal',True) if is_causal is None else is_causal
    output=query_chunk_sdpa(query,key,value,attention_mask,chunk_size=16,is_causal=causal,scale=scaling,dropout_p=dropout)
    return output.transpose(1,2).contiguous(),None


def last_pool(hidden,mask):
    positions=torch.arange(mask.shape[-1]);last=(mask.bool()*(positions+1)).max(dim=-1).values-1
    return F.normalize(hidden[torch.arange(len(hidden)),last].float(),p=2,dim=-1)


def small_qwen_comparison(length=69,padded=False):
    from transformers import Qwen3Config,Qwen3Model
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    torch.manual_seed(91)
    config=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=96,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=1024,
        use_sliding_window=False,sliding_window=None,attention_dropout=0.,pad_token_id=0)
    config._attn_implementation='sdpa';model=Qwen3Model(config).float().eval()
    ids=torch.randint(1,128,(1,length));mask=torch.ones_like(ids)
    if padded:mask[:,:3]=0;ids[:,:3]=0
    observed=[]
    def capture(module,q,k,v,attention_mask,**kwargs):
        observed.append(None if attention_mask is None else list(attention_mask.shape))
        return sdpa_attention_forward(module,q,k,v,attention_mask,**kwargs)
    name='footballmind_audit_query_chunk_only'
    ALL_ATTENTION_FUNCTIONS.register(name,hf_query_chunk)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name,ALL_MASK_ATTENTION_FUNCTIONS['sdpa'])
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    try:
        ALL_ATTENTION_FUNCTIONS.register('sdpa',capture)
        with torch.inference_mode():reference=model(ids,attention_mask=mask,use_cache=False).last_hidden_state
        ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
        model.config._attn_implementation=name
        with torch.inference_mode():chunked=model(ids,attention_mask=mask,use_cache=False).last_hidden_state
    finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
    a=last_pool(reference,mask);b=last_pool(chunked,mask)
    return dict(tokens=length,left_padding=padded,upstream_masks=observed,chunk_queries=16,
        full_hidden_max_abs_difference=float((reference-chunked).abs().max()),
        normalized_embedding_max_abs_difference=float((a-b).abs().max()),
        cosine=float(F.cosine_similarity(a.double(),b.double()).item()),
        model='small_random_Qwen3_two_layers_not_the_frozen_0.6B_model',device='cpu',dtype='float32',
        real_gpu_validated=False,training_performed=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    cases=[small_qwen_comparison(n) for n in (1,17,69,257)]+[small_qwen_comparison(69,True)]
    if any(r['normalized_embedding_max_abs_difference']>1e-5 or r['cosine']<.999999 for r in cases):raise ValueError('CPU small-Qwen equivalence failed')
    result=dict(status='cpu_small_model_equivalence_passed',torch_version=torch.__version__,cases=cases,
        boundary='not a substitute for actual Kaggle GPU controls or the longest real body',
        scores_gib_full_15446_16heads=16*15446**2*4/2**30,
        scores_gib_query256_15446_16heads=16*256*15446*4/2**30)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
