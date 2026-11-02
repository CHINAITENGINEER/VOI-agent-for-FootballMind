import importlib.util
import json
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('recover_trajectories',Path(__file__).parents[2]/'tools/build_full_history_recorded_trajectories.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def test_failed_semantics_rejects_stale_after(tmp_path):
    (tmp_path/'entity-semantic-extractions.json').write_text(json.dumps({'events':[{'attempts':[{'input':{'channel':'body','input_id':'body1','text':'full body','url':'https://source/a'},'status':'FAILED'}]}]}),encoding='utf8')
    run={'cycles':[{'prediction_before':[.4,.3,.3],'prediction_after':[.4,.3,.3],'public_after':None,'semantic_prediction':{'status':'not_updated','reason':'semantic_extraction_failed'}}]}
    _,_,steps,bodies,_=m.extract_learned(tmp_path/'run.json',run)
    assert steps[0]['old_prediction_after'] is None and steps[0]['stale_saved_after_rejected']
    assert steps[0]['newly_adopted_body_input_ids']==[] and not bodies[0]['original_f_consumption_proven']


def test_exact_body_consumption_uses_id_not_probability_change(tmp_path):
    inputs=[{'input':{'channel':c,'input_id':i,'text':t,'url':'https://source/a'},'status':'OK'} for c,i,t in [('title','t','title'),('body','b','Full exact body.')]]
    (tmp_path/'entity-semantic-extractions.json').write_text(json.dumps({'events':[{'attempts':inputs}]}),encoding='utf8')
    run={'cycles':[{'prediction_before':[.4,.3,.3],'prediction_after':[.4,.3,.3],'public_after':{'prediction':{'used_semantic_input_ids':['t','b']}},'semantic_prediction':{'status':'ready'}}]}
    _,_,steps,bodies,_=m.extract_learned(tmp_path/'run.json',run)
    assert len(bodies)==1 and bodies[0]['text']=='Full exact body.'
    assert steps[0]['newly_adopted_body_input_ids']==['b'] and steps[0]['old_prediction_after']==[.4,.3,.3]


def test_live_body_preserves_order_and_unknown_publication(tmp_path):
    r={'fixture':{'physical_match_id':'match'},'prior':[.4,.3,.3],'steps':[{'body':'one','input_id':'x','f_executed':True,'prediction_after':[.4,.3,.3]},{'body':'two','input_id':'y','f_executed':True,'prediction_after':[.4,.3,.3]}]}
    _,_,ss,bb,_=m.extract_live(tmp_path/'run.json',r)
    assert ss[-1]['adopted_body_input_ids_after']==['x','y']
    assert bb[0]['source_available_at'] is None and bb[0]['published_at'] is None


def test_path_scope_does_not_claim_replay_is_real():
    assert m.path_scope('var/x/replay/run.json')=='replay_or_development_diagnostic'
    assert m.path_scope('var/x/pytest-temp/run.json')=='test_or_constructed_excluded'
    assert m.path_scope('var/live/runs/x/run.json')=='recorded_execution'


def test_semantic_digest_mismatch_does_not_prove_consumption(tmp_path):
    (tmp_path/'entity-semantic-extractions.json').write_text('{"events": []}',encoding='utf8')
    _,_,_,b,e=m.extract_learned(tmp_path/'run.json',{'semantic_extraction_archive':{'sha256':'wrong'}})
    assert not b and e=='semantic archive SHA mismatch'


def test_unknown_or_equal_timestamp_not_before():
    assert m.before(None,'2026-10-10T11:30:00Z') is None
    assert m.before('2026-10-10T11:30:00Z','2026-10-10T11:30:00Z') is False
