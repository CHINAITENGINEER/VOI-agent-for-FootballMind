import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'tools/correct_postread_reversal_experiment.py'
spec = importlib.util.spec_from_file_location('correct_postread_reversal', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def doc(mid, domain, text=None):
    return dict(physical_match_id=mid, source_domain=domain, text_sha256=text or mid+domain, source_url=mid+domain)


def test_heldout_domain_cannot_hide_behind_first_source():
    docs = [doc('a','brentford'),doc('a','chelsea'),doc('b','chelsea'),doc('c','arsenal'),doc('d','liverpool')]
    assignment = dict(a='train',b='test',c='validation',d='train')
    corrected = module.correct_source_split(assignment,docs,'chelsea')
    assert corrected['a']=='unused'
    assert assignment['a']=='train'
    assert corrected['b']=='test'


def test_duplicate_body_quarantines_whole_match():
    docs = [doc('a','source_a','duplicate'),doc('b','source_b','duplicate'),doc('c','source_c'),doc('d','source_d')]
    corrected=module.correct_source_split(dict(a='test',b='train',c='validation',d='train'),docs,'source_a')
    assert corrected['b']=='unused'


def test_source_mean_unknown_uses_equal_matches_not_equal_domains():
    domains=['a','a','b','new'];mids=np.array(['m1','m2','m3','m4']);y=np.array([1.,1.,-1.,999.])
    pred,meta=module.source_predict(domains,y,mids,np.array([0,1,2]))
    assert pred[-1]==pytest.approx(1/3)
    assert meta['fallback_unknown_domains']==['new']


def test_source_mean_no_row_multiplicity_bias():
    pred,_=module.source_predict(['a']*5+['a','new'],np.array([1.]*5+[3.,-999.]),np.array(['m1']*5+['m2','m3']),np.arange(6))
    assert pred[-1]==pytest.approx(2.)


def test_mlp_name_mapping_cannot_silently_skip():
    old=[dict(model='mlp_32_20261025',train_mse=.1,validation_mse=.2,test_mse=.3)]
    published=[dict(model='mlp_32_seed20261025',split=s,match_macro_mse=v) for s,v in zip(module.base.SPLITS,(.1,.2,.3))]
    assert module.check_saved_reproduction(old,published)['count']==3
    with pytest.raises(ValueError,match='Missing'):
        module.check_saved_reproduction([],published)


def test_centroids_are_train_only_and_unseen_class_reported():
    docs=[doc('m1','a'),doc('m2','b'),doc('m3','a'),doc('m4','c')]
    vec=np.array([[1.,0.],[0.,1.],[0.,1.],[1.,0.]])
    report=module.centroid_probe(vec,docs,dict(m1='train',m2='train',m3='validation',m4='test'))
    val=next(r for r in report['records'] if r['split']=='validation')
    assert val['predicted_domain']=='b'
    test=next(r for r in report['metrics'] if r['split']=='test')
    assert test['known_domain_articles']==0
    assert test['known_domain_accuracy'] is None


def test_mask_word_boundaries_preserve_nonclub_substrings():
    assert module.mask_identity('aromatic hundreds of players')=='aromatic hundreds of players'
    assert module.mask_identity('Chelsea face Arsenal')=='team face team'


def test_shrinkage_can_reject_overconfident_text():
    choice=module.choose_shrinkage(np.array([1.,-1.]),np.zeros(2),np.zeros(2),np.array(['m1','m2']))
    assert choice['weight']==0.


def test_shrinkage_keeps_correct_signed_predictions():
    choice=module.choose_shrinkage(np.array([1.,-1.]),np.zeros(2),np.array([1.,-1.]),np.array(['m1','m2']))
    assert choice['weight']==1.


def test_global_time_rejects_future_training_matches():
    docs=[doc('a','x'),doc('b','y'),doc('c','z')]
    with pytest.raises(ValueError,match='chronological'):
        module.audit_split(dict(a='train',b='validation',c='test'),docs,dates=dict(a='2026-01-01',b='2024-01-01',c='2025-01-01'))


def test_actual_input_labels_and_current_masks_are_consistent():
    b=module.base
    if not b.DATA.exists():pytest.skip('Archive is not mounted')
    data=[b.rows(b.DATA/n) for n in ('features.jsonl','samples.jsonl','labels.jsonl','states.jsonl','documents.jsonl')]
    assert module.validate_labels(*data,b.sha(module.F_PATH))['rows']==289


def test_masked_body_encoder_fits_masked_training_text_only():
    b=module.base
    features=[]
    for i,text in enumerate(['Chelsea player injured','Arsenal goalkeeper returns','heldoutuniqueword Liverpool unavailable']):
        features.append(dict(sample_id=str(i),physical_match_id=str(i),split='train' if i<2 else 'test',
            candidate_text=text,candidate_title=text,adopted_text='',state_numeric=[.2,.3,.5],
            state_feature_names=['a','b','c'],semantic_numeric=[0.],semantic_feature_names=['x']))
    _,enc=module.make_design(features,np.array([0,1]))
    vocabulary=enc['body_masked']['vectorizer'].vocabulary_
    assert 'team' in vocabulary
    assert 'chelsea' not in vocabulary and 'arsenal' not in vocabulary and 'heldoutuniqueword' not in vocabulary
