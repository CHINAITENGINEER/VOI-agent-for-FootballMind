"""Contract tests for label isolation, ID alignment and bounded batch indexing."""
import json
import builtins
from pathlib import Path

import numpy as np
import pytest

from football_research.acquisition.postread_semantic_embedding import EncoderContract, text_sha
from football_research.acquisition.semantic_training_dataset import (
    SemanticTrainingDataset, file_sha, prepare_panel, validate_groups)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def rows(path, values):
    path.write_text("".join(json.dumps(v)+"\n" for v in values), encoding="utf-8")


@pytest.fixture
def source(tmp_path):
    data=tmp_path/"data"; embed=tmp_path/"embed"; data.mkdir(); embed.mkdir()
    f=tmp_path/"model.json"; write(f,{"frozen":True})
    features=[]; labels=[]; samples=[]
    for i, split in enumerate(("train","validation","test")):
        candidate=f"full body {i}"; adopted="" if i==0 else f"already adopted evidence {i}"
        feature=dict(sample_id=f"s{i}",physical_match_id=f"match{i}",split=split,
            candidate_text=candidate,adopted_text=adopted,candidate_title=f"title{i}",
            state_numeric=[.3,.3,.4]+[0.]*54,state_feature_names=[f"state_{j}" for j in range(57)],
            semantic_numeric=[0.]*29,semantic_feature_names=[f"claim_{j}" for j in range(29)])
        features.append(feature)
        labels.append(dict(sample_id=f"s{i}",physical_match_id=f"match{i}",split=split,realized_gain=float(i-1),
            p_before=[.3,.3,.4],f_model_sha256=file_sha(f),f_model_version="fixed",offline_f_evaluated=True,training_eligible=True))
        samples.append(dict(sample_id=f"s{i}",physical_match_id=f"match{i}",split=split,text_sha256=text_sha(candidate)))
    rows(data/"features.jsonl",features);rows(data/"labels.jsonl",labels[::-1]);rows(data/"samples.jsonl",samples[::-1])
    write(data/"split.json",{"groups":{s:[f"match{i}"] for i,s in enumerate(("train","validation","test"))}})
    vectors={}
    for feature in features:
        for field in ("candidate_text","adopted_text"):
            if feature[field]:
                v=np.arange(1,1025,dtype=np.float32)+len(vectors);v/=np.linalg.norm(v)
                vectors[text_sha(feature[field])]=v
    np.savez(embed/"embeddings.npz",**vectors)
    from dataclasses import asdict
    write(embed/"embedding-report.json",dict(contract=asdict(EncoderContract()),source_sha256=file_sha(data/"features.jsonl"),vectors_file_sha256=file_sha(embed/"embeddings.npz")))
    hashes={str(p.resolve()):file_sha(p) for p in [*data.iterdir(),*embed.iterdir(),f]}
    return data,embed,f,hashes,tmp_path/"panel"


def test_default_does_not_open_any_label_file(source,monkeypatch):
    prepare_panel(*source)
    real_open=Path.open
    def guard(path,*a,**k):
        if "labels" in path.parts:
            raise AssertionError("Default loader opened a label file")
        return real_open(path,*a,**k)
    monkeypatch.setattr(Path,"open",guard)
    builtin_open=builtins.open
    def builtin_guard(path,*a,**k):
        if isinstance(path,(str,Path)) and "labels" in Path(path).parts:
            raise AssertionError("Default loader opened labels through builtin open")
        return builtin_open(path,*a,**k)
    monkeypatch.setattr(builtins,"open",builtin_guard)
    dataset=SemanticTrainingDataset(source[-1])
    batch=dataset.batch(np.array([2,0]))
    assert "realized_gain" not in batch
    assert isinstance(dataset.vectors,np.memmap) and not dataset.vectors.flags.writeable
    assert batch["sample_id"]==["s2","s0"]
    assert np.array_equal(batch["features"]["adopted"][1],np.zeros(32,np.float32))
    assert batch["adopted_text_index"][1]==-1


def test_labels_align_by_id_despite_source_order_and_split_batch(source):
    prepare_panel(*source)
    dataset=SemanticTrainingDataset(source[-1],include_labels=True)
    assert dataset.batch(np.array([2,0,1]))["realized_gain"].tolist()==[1.,-1.,0.]
    validation=SemanticTrainingDataset(source[-1],split="validation",include_labels=True,dimension=256)
    b=list(validation.iter_batches(2))[0]
    assert b["sample_id"]==["s1"] and b["realized_gain"].tolist()==[0.]
    assert b["features"]["fulltext_state"].shape==(1,1110)
    assert np.isclose(np.linalg.norm(b["features"]["body"][0]),1)


def test_corrupt_label_only_blocks_explicit_supervision(source):
    prepare_panel(*source)
    (source[-1]/"labels/realized-gain.npy").write_bytes(b"corrupt")
    assert len(SemanticTrainingDataset(source[-1]))==3
    with pytest.raises(ValueError,match="Label path or digest"):
        SemanticTrainingDataset(source[-1],include_labels=True)


def test_frozen_source_digest_and_group_validation(source):
    source[0].joinpath("features.jsonl").write_text("{}\n",encoding="utf8")
    with pytest.raises(ValueError,match="digest"):
        prepare_panel(*source)
    with pytest.raises(ValueError,match="more than once"):
        validate_groups([dict(physical_match_id="m",split="train")],{"train":["m"],"test":["m"]})


def test_reject_index_tampering_even_if_file_digest_updated(source):
    prepare_panel(*source)
    path=source[-1]/"text-indices.npy"
    a=np.load(path);a[0,0]=-1;np.save(path,a)
    config_path=source[-1]/"config.json";config=json.loads(config_path.read_text())
    config["feature_files"]["text-indices.npy"]=file_sha(path);write(config_path,config)
    with pytest.raises(ValueError,match="indices disagree"):
        SemanticTrainingDataset(source[-1])


def test_batch_bounds_and_dimension_contract(source):
    prepare_panel(*source)
    dataset=SemanticTrainingDataset(source[-1],dimension=1024)
    assert dataset.batch(np.array([0]))["features"]["fulltext_state"].shape==(1,4182)
    for bad in ([-1],[3],[0.1]):
        with pytest.raises(IndexError):dataset.batch(bad)
    with pytest.raises(ValueError):SemanticTrainingDataset(source[-1],dimension=128)
    with pytest.raises(FileExistsError):prepare_panel(*source)


def test_f_binding_checked_beyond_source_digest(source):
    path=source[0]/"labels.jsonl"
    labels=[json.loads(line) for line in path.read_text().splitlines()]
    labels[0]["f_model_sha256"]="0"*64
    rows(path,labels);source[3][str(path.resolve())]=file_sha(path)
    with pytest.raises(ValueError,match="bound to another f"):
        prepare_panel(*source)


def test_duplicate_feature_id_rejected(source):
    path=source[0]/"features.jsonl"
    features=[json.loads(line) for line in path.read_text().splitlines()]
    features[1]["sample_id"]=features[0]["sample_id"]
    rows(path,features);source[3][str(path.resolve())]=file_sha(path)
    with pytest.raises(ValueError,match="Duplicate sample_id"):
        prepare_panel(*source)
