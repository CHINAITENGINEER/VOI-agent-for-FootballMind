"""Direct checks of the actual v2 runner, not merely an independent prototype."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/'tools'))
import run_kaggle_semantic_encoding_v2 as runner
import pytest
import torch
import torch.nn.functional as F
torch.set_num_threads(1)


def tensors(length=37,key_length=None,gqa=True):
    torch.manual_seed(41);key_length=key_length or length;heads=2 if gqa else 4
    return torch.randn(1,4,length,8),torch.randn(1,heads,key_length,8),torch.randn(1,heads,key_length,12)


@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('gqa',[False,True])
def test_actual_v2_full_key_gqa_and_causal(causal,gqa):
    q,k,v=tensors(gqa=gqa)
    expected=F.scaled_dot_product_attention(q,k,v,is_causal=causal,enable_gqa=gqa,scale=.27)
    actual=runner.tiled_sdpa(F.scaled_dot_product_attention,q,k,v,is_causal=causal,enable_gqa=gqa,scale=.27,chunk_size=8)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('mask_type',['boolean_2d','float_2d','boolean_4d','float_4d','broadcast_2d','left_padding_4d'])
def test_actual_v2_slices_complete_masks_and_preserves_bias(mask_type):
    q,k,v=tensors();mask=torch.tril(torch.ones(37,37,dtype=torch.bool))
    if mask_type=='broadcast_2d':mask=torch.ones(1,37,dtype=torch.bool);mask[:,:2]=False
    if mask_type=='left_padding_4d':mask[:,:3]=False
    if 'float' in mask_type:
        bias=torch.randn(37,37)*.03;mask=bias.masked_fill(~mask,float('-inf'))
    if '4d' in mask_type:mask=mask[None,None]
    expected=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True)
    actual=runner.tiled_sdpa(F.scaled_dot_product_attention,q,k,v,attn_mask=mask,enable_gqa=True,chunk_size=8)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_actual_v2_rectangular_global_upper_left_causal_matches_torch():
    q,k,v=tensors(17,23)
    expected=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
    actual=runner.tiled_sdpa(F.scaled_dot_product_attention,q,k,v,is_causal=True,enable_gqa=True,chunk_size=4)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_actual_v2_never_crops_keys_or_allocates_full_query_scores_in_call():
    q,k,v=tensors();calls=[];original=F.scaled_dot_product_attention
    def spy(query,key,value,**kwargs):
        calls.append((query.shape[-2],key.shape[-2],kwargs['is_causal'],kwargs['attn_mask'].shape))
        return original(query,key,value,**kwargs)
    runner.tiled_sdpa(spy,q,k,v,is_causal=True,enable_gqa=True,chunk_size=8)
    assert len(calls)==5
    assert all(qn<=8 and kn==37 and causal is False and mask[-2]<=8 and mask[-1]==37 for qn,kn,causal,mask in calls)


def test_actual_v2_causal_position_does_not_reset_in_each_chunk():
    q=k=torch.zeros(1,1,6,1);v=torch.arange(6,dtype=torch.float32).reshape(1,1,6,1)
    actual=runner.tiled_sdpa(F.scaled_dot_product_attention,q,k,v,is_causal=True,chunk_size=2)
    torch.testing.assert_close(actual.flatten(),torch.arange(6,dtype=torch.float32)/2,atol=2e-7,rtol=0)


def test_actual_v2_rejects_training_and_restores_patch_after_exception():
    q,k,v=tensors()
    with pytest.raises(ValueError,match='dropout'):runner.tiled_sdpa(F.scaled_dot_product_attention,q,k,v,dropout_p=.1,chunk_size=8)
    with pytest.raises(ValueError,match='frozen inference'):runner.tiled_sdpa(F.scaled_dot_product_attention,q.requires_grad_(),k,v,chunk_size=8)
    original=F.scaled_dot_product_attention
    with pytest.raises(RuntimeError,match='deliberate'):
        with runner.bounded_query_attention(8):raise RuntimeError('deliberate')
    assert F.scaled_dot_product_attention is original
    with runner.bounded_query_attention(8):
        with pytest.raises(RuntimeError,match='Nested'):
            with runner.bounded_query_attention(8):pass
    assert F.scaled_dot_product_attention is original


@pytest.mark.parametrize('padded',[False,True])
def test_actual_v2_complete_small_qwen_embedding(padded):
    from transformers import Qwen3Config,Qwen3Model
    torch.manual_seed(19)
    cfg=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=96,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=2,head_dim=16,max_position_embeddings=512,use_sliding_window=False,sliding_window=None,
        attention_dropout=0.,pad_token_id=0)
    cfg._attn_implementation='sdpa';model=Qwen3Model(cfg).float().eval()
    ids=torch.randint(1,128,(1,257));mask=torch.ones_like(ids)
    if padded:mask[:,:3]=0;ids[:,:3]=0
    with torch.inference_mode():
        ref=model(ids,attention_mask=mask,use_cache=False).last_hidden_state
        original=F.scaled_dot_product_attention
        with runner.bounded_query_attention(256) as counters:
            actual=model(ids,attention_mask=mask,use_cache=False).last_hidden_state
        assert F.scaled_dot_product_attention is original
    assert counters['attention_calls']==2 and counters['tiled_attention_calls']==2
    torch.testing.assert_close(actual,ref,atol=2e-6,rtol=2e-5)
    a=F.normalize(ref[:,-1],p=2,dim=-1);b=F.normalize(actual[:,-1],p=2,dim=-1)
    assert (a-b).abs().max()<1e-6 and F.cosine_similarity(a.double(),b.double()).item()>.999999
