import importlib.util
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

SPEC=importlib.util.spec_from_file_location('query_chunk_audit',Path(__file__).parents[2]/'tools/verify_qwen_query_chunk_equivalence.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
torch.set_num_threads(1)


@pytest.mark.parametrize('length,chunk',[(1,1),(7,2),(19,5),(37,16)])
def test_full_causal_gqa_and_chunked_queries_match(length,chunk):
    torch.manual_seed(15)
    q=torch.randn(1,4,length,8);k=torch.randn(1,2,length,8);v=torch.randn(1,2,length,8)
    expected=F.scaled_dot_product_attention(q,k,v,is_causal=length>1,enable_gqa=True,scale=.27)
    actual=m.query_chunk_sdpa(q,k,v,chunk_size=chunk,scale=.27)
    torch.testing.assert_close(actual,expected,atol=8e-7,rtol=8e-6)


def test_causal_mask_uses_global_query_position_not_block_local_position():
    q=k=torch.zeros(1,1,6,1);v=torch.arange(6,dtype=torch.float32).reshape(1,1,6,1)
    actual=m.query_chunk_sdpa(q,k,v,chunk_size=2)
    torch.testing.assert_close(actual.flatten(),torch.arange(6,dtype=torch.float32)/2,atol=2e-7,rtol=0)


@pytest.mark.parametrize('additive',[False,True])
def test_existing_four_dimensional_padding_causal_mask_is_sliced(additive):
    torch.manual_seed(92);q=torch.randn(1,4,19,8);k=torch.randn(1,2,19,8);v=torch.randn(1,2,19,8)
    mask=torch.tril(torch.ones(19,19,dtype=torch.bool))[None,None];mask[:,:,:,:3]=False
    if additive:mask=torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,float('-inf'))
    expected=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=False,enable_gqa=True)
    actual=m.query_chunk_sdpa(q,k,v,mask,chunk_size=5)
    torch.testing.assert_close(actual,expected,atol=8e-7,rtol=8e-6)


def test_full_key_noncausal_mode_not_forced_causal():
    torch.manual_seed(3);q=torch.randn(1,4,17,8);k=torch.randn(1,2,17,8);v=torch.randn(1,2,17,8)
    expected=F.scaled_dot_product_attention(q,k,v,is_causal=False,enable_gqa=True)
    torch.testing.assert_close(m.query_chunk_sdpa(q,k,v,chunk_size=4,is_causal=False),expected,atol=8e-7,rtol=8e-6)


def test_reject_cached_decode_and_training_dropout():
    q=torch.zeros(1,4,3,8);k=v=torch.zeros(1,2,5,8)
    with pytest.raises(ValueError,match='no-cache'):m.query_chunk_sdpa(q,k,v)
    with pytest.raises(ValueError,match='dropout'):m.query_chunk_sdpa(q,k,v,dropout_p=.1)


@pytest.mark.parametrize('padded',[False,True])
def test_random_small_qwen_entire_embedding_matches(padded):
    result=m.small_qwen_comparison(69,padded)
    assert result['normalized_embedding_max_abs_difference']<1e-5 and result['cosine']>.999999
    assert result['upstream_masks']==([[1,1,69,69]]*2 if padded else [None,None])
