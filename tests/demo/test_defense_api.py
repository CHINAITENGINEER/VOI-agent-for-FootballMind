"""Endpoint contract, model ownership and separate settlement boundary."""
import pytest
from demo.defense import DefenseApplication
from demo.server import DemoApplication

class FrozenStub:
    def __init__(self,root):self.settlements=0;self.calls=[]
    def catalog(self):return [dict(case_id='match-1',home_team='A',away_team='B')]
    def get_case(self,case_id):
        if case_id!='match-1':raise KeyError(case_id)
        return dict(case_id=case_id,prior=[.4,.3,.3],candidates=[])
    def get_evidence(self,case_id,document_id):return dict(text='Archived body',sha256='digest')
    def run_case(self,case_id,model_name=None):
        self.calls.append((case_id,model_name))
        return dict(run_id='internal-run',p_before=[.4,.3,.3],p_after=[.42,.29,.29],selected_ids=['doc-1'],steps=[])
    def settle_run(self,run):self.settlements+=1;return dict(outcome_class=0,realized_gain=.04879)

def app(tmp_path):
    a=DemoApplication(store_dir=tmp_path)
    a._defense=DefenseApplication(tmp_path,factory=FrozenStub)
    return a

def test_catalog_distinguishes_default_from_development_choice(tmp_path):
    a=app(tmp_path);r=a.get('/api/defense/catalog',{})
    assert r['release_ready'] and r['frozen_research_demo'] and not r['network_required']
    assert r['presentation_default']=='fulltext_mlp_seed20261027'
    assert r['development_selected_policy']=='state_ridge'
    assert r['workspace_root'] and len(r['matches'])==1

def test_prediction_then_explicit_settlement_and_copy_boundary(tmp_path):
    a=app(tmp_path);r=a.post('/api/defense/runs',{'case_id':'match-1'})
    assert r['settlement'] is None and a._defense.backend.settlements==0
    r['result']['p_after']=[0,0,1]
    saved=a.get('/api/defense/runs/'+r['run_id'],{})
    assert saved['result']['p_after']==[.42,.29,.29]
    settled=a.post('/api/defense/runs/'+r['run_id']+'/settle',{})
    assert settled['settlement']['outcome_class']==0
    a.post('/api/defense/runs/'+r['run_id']+'/settle',{})
    assert a._defense.backend.settlements==1

def test_reject_custom_paths_heads_and_outcomes(tmp_path):
    a=app(tmp_path)
    for payload in ({'case_id':'match-1','model_path':'anything'}, {'case_id':'match-1','model_name':'unfrozen'}):
        with pytest.raises(ValueError):a.post('/api/defense/runs',payload)
    r=a.post('/api/defense/runs',{'case_id':'match-1','model_name':'state_ridge'})
    with pytest.raises(ValueError):a.post('/api/defense/runs/'+r['run_id']+'/settle',{'outcome_class':2})

def test_case_and_evidence_are_separate_local_reads(tmp_path):
    a=app(tmp_path)
    assert a.get('/api/defense/case',{'case_id':['match-1']})['prior']==[.4,.3,.3]
    assert a.get('/api/defense/evidence',{'case_id':['match-1'],'document_id':['doc-1']})['text']=='Archived body'
    assert a._defense.backend.settlements==0
