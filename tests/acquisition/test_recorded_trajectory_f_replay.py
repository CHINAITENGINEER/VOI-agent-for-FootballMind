import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np

SPEC=importlib.util.spec_from_file_location('recorded_replay',Path(__file__).parents[2]/'tools/replay_recorded_trajectories_f.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def setup_inputs(tmp_path):
    inp=tmp_path/'inputs';inp.mkdir();body=inp/'body.txt';body.write_text('Exact archived body.',encoding='utf8');digest=hashlib.sha256(body.read_bytes()).hexdigest()
    record=dict(run_id='r',physical_match_id='m',new_f_input_replayable=True,prior=[.4,.3,.3],
        documents=[dict(document_id='sha256:'+digest,text_sha256=digest,body_path=str(body))],
        identity=dict(kickoff='2026-10-10T12:00:00Z'),independent_outcome=dict(outcome_class=0),
        stratum='recorded_agent_trajectory_frozen_model_replay_diagnostic',original_f_binding={'raw_sha256':'OLD'},source_path='oldrun',old_gain=999.)
    (inp/'new-f-replay-inputs.jsonl').write_text(json.dumps(record)+'\n',encoding='utf8')
    return SimpleNamespace(inputs=inp,output=tmp_path/'out',vectors=tmp_path/'vectors',f_model=tmp_path/'model')


def test_new_f_gain_recomputed_and_old_gain_never_reused(tmp_path,monkeypatch):
    args=setup_inputs(tmp_path)
    class Model:
        config=SimpleNamespace(metadata_names=m.METADATA_NAMES);f_model_sha256='NEW'
        def predict_prefixes(self,prior,vectors,meta,order):
            assert meta.shape==(1,len(m.METADATA_NAMES)) and len(order)==1
            return [dict(probabilities=[.4,.3,.3],hidden_state=[0.],used_input_ids=[]),dict(probabilities=[.5,.25,.25],hidden_state=[1.],used_input_ids=order)]
    monkeypatch.setattr(m.FrozenF,'load',lambda _:Model())
    monkeypatch.setattr(m,'VectorStore',lambda _:SimpleNamespace(get=lambda _:np.ones(1024,dtype=np.float32)))
    m.execute(args);rows=m.rows(args.output/'prefix-predictions.jsonl')
    assert np.isclose(rows[1]['new_f_step_gain'],np.log(.5)-np.log(.4))
    assert rows[1]['new_f_model_sha256']=='NEW' and not rows[1]['old_gain_reused'] and not rows[1]['training_row_created']


def test_missing_vector_failure_has_null_gain(tmp_path,monkeypatch):
    args=setup_inputs(tmp_path)
    monkeypatch.setattr(m.FrozenF,'load',lambda _:SimpleNamespace(config=SimpleNamespace(metadata_names=m.METADATA_NAMES),f_model_sha256='NEW'))
    def missing(_):raise FileNotFoundError('not encoded')
    monkeypatch.setattr(m,'VectorStore',lambda _:SimpleNamespace(get=missing))
    m.execute(args);failure=m.rows(args.output/'failures.jsonl')[0]
    assert failure['gain'] is None and not m.rows(args.output/'prefix-predictions.jsonl')
