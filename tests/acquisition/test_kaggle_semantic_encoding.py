"""Small fake-vector checks; never load or download the real encoder."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('kaggle_runner',Path(__file__).parents[2]/'tools/run_kaggle_semantic_encoding.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def vector_for(text):
    v=np.zeros(1024,dtype=np.float32);v[int(hashlib.sha256(text.encode()).hexdigest()[:4],16)%1024]=1;return v


@pytest.fixture
def job(tmp_path):
    root=tmp_path/'input';root.mkdir();(root/'texts').mkdir();(root/'controls').mkdir();records=[];controls=[]
    for i in range(10):
        text=f'Exact text {i}: café\r\nFull body.';digest=hashlib.sha256(text.encode()).hexdigest()
        path=root/'texts'/(digest+'.txt');path.write_bytes(text.encode())
        contract=m.LONG_CONTRACT if i==9 else m.BASE_CONTRACT
        records.append(dict(text_sha256=digest,text_path='texts/'+path.name,contract_fingerprint=m.fingerprint(contract),token_count=7,role='f_fit',task_group='main_historical_corpus'))
        if i<8:
            vp=root/'controls'/(digest+'.npy');np.save(vp,vector_for(text),allow_pickle=False)
            controls.append(dict(text_sha256=digest,vector_path='controls/'+vp.name,vector_file_sha256=m.sha_file(vp)))
    lines=''.join(json.dumps(r)+'\n' for r in records).encode();(root/'texts.jsonl').write_bytes(lines)
    value=dict(schema='footballmind-kaggle-semantic-job-v1',job_id='unit-test-job',
        model=dict(repo=m.REPO,revision=m.REVISION,files_sha256={'model.safetensors':m.WEIGHTS_SHA,'config.json':'0'*64,'tokenizer.json':'1'*64,'tokenizer_config.json':'2'*64}),
        contracts={m.fingerprint(c):c for c in (m.BASE_CONTRACT,m.LONG_CONTRACT)},controls=controls,
        expected_texts=len(records),texts_sha256=hashlib.sha256(lines).hexdigest())
    (root/'job.json').write_text(json.dumps(value),encoding='utf8')
    return SimpleNamespace(input_dir=root,output_dir=tmp_path/'output',model_dir=tmp_path/'model',device='cuda',batch_size=1,verify_input_only=False)


def fake_model(monkeypatch,*,bad=False):
    calls=[]
    class FakeEncoder:
        execution={'device':'fake-test-only','software':{},'model_dtype':'float32'}
        def __init__(self,*args):pass
        def encode(self,text,contract,expected_tokens):
            calls.append(text);v=vector_for(text)
            if bad:v=np.roll(v,1)
            return v,dict(token_count=expected_tokens,inference_seconds=0.,characters=len(text),truncated_tokens=0)
    monkeypatch.setattr(m,'Encoder',FakeEncoder);monkeypatch.setattr(m,'verify_model_files',lambda *a:None)
    return calls


def test_input_only_checks_bytes_without_model_loading(job,monkeypatch):
    job.verify_input_only=True
    monkeypatch.setattr(m,'Encoder',lambda *a:pytest.fail('must not load model'))
    result=m.execute(job)
    assert result['status']=='inputs_verified_only' and result['expected_texts']==10
    assert not list(job.output_dir.glob('*.npy'))
    assert m.read_json(job.output_dir/'input-verification.json')['token_counts_tokenizer_checked'] is False


def test_text_sha_preserves_exact_crlf_and_unicode(job):
    _,records,_=m.verify_inputs(job.input_dir);path=job.input_dir/records[0]['text_path']
    path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    with pytest.raises(ValueError,match='body SHA'):m.verify_inputs(job.input_dir)


def test_control_reference_bytes_are_bound_by_job(job):
    j=m.read_json(job.input_dir/'job.json');p=job.input_dir/j['controls'][0]['vector_path']
    np.save(p,np.roll(np.load(p),1),allow_pickle=False)
    with pytest.raises(ValueError,match='reference vector SHA'):m.verify_inputs(job.input_dir)


def test_unknown_contract_or_model_revision_rejected(job):
    p=job.input_dir/'job.json';j=m.read_json(p);j['model']['revision']='main';p.write_text(json.dumps(j),encoding='utf8')
    with pytest.raises(ValueError,match='revision differs'):m.verify_inputs(job.input_dir)


def test_fake_full_encoding_and_resume_match_exactly(job,monkeypatch):
    calls=fake_model(monkeypatch);first=m.execute(job);assert first['status']=='completed' and len(calls)==10
    original={p.name:m.sha_file(p) for p in job.output_dir.glob('*.npy')}
    second=m.execute(job);assert second['resumed_texts']==10 and len(calls)==18
    assert original=={p.name:m.sha_file(p) for p in job.output_dir.glob('*.npy')}
    index=[json.loads(x) for x in (job.output_dir/'index.jsonl').read_text().split('\n') if x]
    with np.load(job.output_dir/'embeddings.npz',allow_pickle=False) as archive:
        assert set(archive.files)=={r['text_sha256'] for r in index}
        for r in index:np.testing.assert_array_equal(archive[r['text_sha256']],np.load(job.output_dir/r['vector_path']))
    assert m.read_json(job.output_dir/'controls.json')['all_passed']
    assert (job.output_dir/'input-contract/job.json').read_bytes()==(job.input_dir/'job.json').read_bytes()


def test_bad_gpu_control_stops_before_batch_acceptance(job,monkeypatch):
    calls=fake_model(monkeypatch,bad=True)
    with pytest.raises(RuntimeError,match='tolerance failed'):m.execute(job)
    assert len(calls)==8
    assert m.read_json(job.output_dir/'completion.json')['status']=='control_validation_failed'
    assert not (job.output_dir/'embeddings.npz').exists() and not list(job.output_dir.glob('*.npy'))


def test_committed_corrupt_vector_is_not_silently_reencoded(job,monkeypatch):
    fake_model(monkeypatch);m.execute(job);p=next(job.output_dir.glob('*.npy'));p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupted'):m.execute(job)
    assert p.read_bytes()==b'corrupt'


def test_resume_rejects_changed_job_identity(job,monkeypatch):
    fake_model(monkeypatch);m.execute(job);p=job.input_dir/'job.json';j=m.read_json(p);j['job_id']='other';p.write_text(json.dumps(j),encoding='utf8')
    with pytest.raises(ValueError,match='another job'):m.execute(job)


def test_postoutcome_record_fields_are_not_accepted(job):
    p=job.input_dir/'texts.jsonl';records=[json.loads(s) for s in p.read_text().split('\n') if s];records[0]['realized_gain']=1
    p.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf8')
    with pytest.raises(ValueError,match='labels'):m.verify_inputs(job.input_dir)


@pytest.mark.parametrize('path',['../escape','/absolute','C:/secret','a\\b'])
def test_paths_cannot_escape_or_use_platform_ambiguous_spelling(tmp_path,path):
    with pytest.raises(ValueError):m.within(tmp_path,path)


def test_model_file_checksum_is_checked(tmp_path):
    p=tmp_path/'config.json';p.write_bytes(b'actual')
    with pytest.raises(ValueError,match='hash mismatch'):m.verify_model_files(tmp_path,{'model':{'files_sha256':{'config.json':'0'*64}}})


def test_vector_normalization_dtype_and_finiteness_required():
    v=vector_for('test');assert m.compare_control(v,v)['passed']
    with pytest.raises(ValueError):m.verify_vector(v.astype(np.float64))
    with pytest.raises(ValueError):m.verify_vector(v*2)
    v[2]=np.nan
    with pytest.raises(ValueError):m.verify_vector(v)


def test_oom_stops_without_retry_precision_change_or_truncation(job,monkeypatch):
    calls=[]
    class OOMEncoder:
        execution={'device':'fake-test-only'}
        def __init__(self,*a):pass
        def encode(self,*a):calls.append(a);raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(m,'Encoder',OOMEncoder);monkeypatch.setattr(m,'verify_model_files',lambda *a:None)
    with pytest.raises(RuntimeError,match='out of memory'):m.execute(job)
    assert len(calls)==1
    assert m.read_json(job.output_dir/'completion.json')['status']=='failed'
    assert not (job.output_dir/'embeddings.npz').exists()
