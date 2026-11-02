from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from football_research.acquisition.postread_semantic_embedding import (
    EncoderContract, LocalSemanticEncoder, SemanticCache, build_semantic_design,
    last_token_pool, normalize_prefix, text_sha,
)


def feature(**overrides):
    return dict(sample_id="match:one:0:1", physical_match_id="match:one", split="train",
        candidate_text="He trained, but the coach says he cannot play tomorrow.",
        candidate_title="Training update", adopted_text="", state_numeric=[.4,.3,.3],
        state_feature_names=["p_home","p_draw","p_away"], semantic_numeric=[1.],
        semantic_feature_names=["unavailable_claims"], **overrides)


def test_left_and_right_padding_pool_last_real_token():
    hidden=torch.arange(2*4*3).reshape(2,4,3)
    masks=torch.tensor([[0,0,1,1],[1,1,0,0]])
    pooled=last_token_pool(hidden,masks)
    assert torch.equal(pooled[0],hidden[0,3])
    assert torch.equal(pooled[1],hidden[1,1])
    with pytest.raises(ValueError,match="entirely padded"):
        last_token_pool(hidden,torch.zeros_like(masks))


def test_overlength_is_rejected_before_forward_instead_of_truncated():
    encoder=object.__new__(LocalSemanticEncoder)
    encoder.contract=replace(EncoderContract(),max_input_tokens=5)
    calls=[]
    def tokenize(text,**kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids":torch.ones((1,6),dtype=torch.long),"attention_mask":torch.ones((1,6),dtype=torch.long)}
    encoder.tokenizer=tokenize
    encoder.model=lambda **kwargs:calls.append(kwargs)
    with pytest.raises(ValueError,match="not truncated"):
        encoder.encode_one("The decisive injury statement is at the end.")
    assert not calls


def test_empty_adopted_evidence_is_not_a_language_embedding():
    r=feature();v=np.arange(1,1025,dtype=np.float32);v/=np.linalg.norm(v)
    x=build_semantic_design([r],{text_sha(r["candidate_text"]):v},32)["fulltext_state"]
    assert x.shape==(1,4+4*32)
    np.testing.assert_array_equal(x[0,4+32:4+64],np.zeros(32))
    np.testing.assert_allclose(x[0,4:4+32],normalize_prefix(v,32))
    np.testing.assert_array_equal(x[0,-32:],np.zeros(32))


def test_same_dimensions_do_not_allow_outcome_or_after_features():
    r=feature();r["realized_gain"]=0.1
    with pytest.raises(ValueError,match="unknown fields"):
        build_semantic_design([r],{},32)
    r=feature();r["state_feature_names"][0]="p_after_home"
    with pytest.raises(ValueError,match="forbidden"):
        build_semantic_design([r],{},32)


def test_reordered_state_fields_rejected():
    a,b=feature(),feature();b["state_feature_names"]=list(reversed(b["state_feature_names"]))
    with pytest.raises(ValueError,match="order differs"):
        build_semantic_design([a,b],{},32)


def test_model_contract_and_full_text_tail_change_cache_key(tmp_path):
    c=EncoderContract();cache=SemanticCache(tmp_path,c)
    text="Returned to training. However, he is ruled out of this match."
    vector=np.ones(1024,dtype=np.float32)/32
    cache.put(text,vector,{"token_count":15,"truncated_tokens":0})
    np.testing.assert_array_equal(cache.get(text)[0],vector)
    assert cache.get("Returned to training. However, he is available for this match.") is None
    other=SemanticCache(tmp_path,replace(c,revision="different-revision"))
    assert other.get(text) is None
    with pytest.raises(FileExistsError): cache.put(text,vector,{})


def test_corrupt_cache_not_silently_used(tmp_path):
    cache=SemanticCache(tmp_path,EncoderContract());text="Official team news"
    cache.put(text,np.ones(1024,dtype=np.float32)/32,{})
    path=cache.root/(text_sha(text)+".npy");path.write_bytes(b"broken")
    with pytest.raises(ValueError,match="digest mismatch"):cache.get(text)


@pytest.mark.parametrize("dim",[32,256,1024])
def test_prefix_re_normalized_without_fitting(dim):
    v=np.arange(1,1025,dtype=np.float32)
    p=normalize_prefix(v,dim)
    assert p.shape==(dim,)
    assert abs(float(np.linalg.norm(p))-1)<1e-6
    assert p[-1]>p[0]


def test_actual_existing_feature_geometry():
    # Use the real 57 + 29 public/claim feature layout, not labels or after states.
    r=feature();r["state_numeric"]=[0.]*57;r["state_feature_names"]=[f"state_{i}" for i in range(57)]
    r["semantic_numeric"]=[0.]*29;r["semantic_feature_names"]=[f"claim_{i}" for i in range(29)]
    v={text_sha(r["candidate_text"]):np.ones(1024,dtype=np.float32)/32}
    assert build_semantic_design([r],v,32)["fulltext_state"].shape==(1,214)
    assert build_semantic_design([r],v,256)["fulltext_state"].shape==(1,1110)
    assert build_semantic_design([r],v,256)["fulltext_state_without_candidate_claims"].shape==(1,1081)
