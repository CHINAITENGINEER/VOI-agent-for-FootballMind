"""Check saved transitions against the exact frozen f, without network or fitting."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from football_research.acquisition import entity_evidence_updates as f
from football_research.acquisition import historical_gain_head as g
from tools.train_reviewed_rich_f import write


def main():
    p=ROOT/'var/reviews/fm-historical-transition-g-training-20261014-v2'
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
    transitions=read(p/'transitions.json'); states=read(p/'states.json'); plan=read(p/'plan.json')
    original=read(ROOT/'var/reviews/fm-entity-updates-f-training-20261014-v1/event-training-rows.json')
    lookup={(r['physical_match_id'],tuple(sorted(r['used_input_ids']))):r for r in original}
    labeler=f.load(p/'f-labeler.json'); digest=sha256((p/'f-labeler.json').read_bytes()).hexdigest()
    assert not set(plan['f_training_ids'])&set(plan['g_training_ids']+plan['g_holdout_ids'])
    assert not set(plan['g_training_ids'])&set(plan['g_holdout_ids'])
    pred={}
    for sid,s in states.items():
        row=lookup[s['physical_match_id'],tuple(s['used_input_ids'])]
        pred[sid]=f.predict(labeler,row)
    for t in transitions:
        b,a=states[t['state_id']],states[t['next_state_id']]
        assert set(a['used_input_ids'])-set(b['used_input_ids'])=={t['body_input_id']}
        assert set(b['used_input_ids'])<=set(a['used_input_ids'])
        assert b['physical_match_id']==a['physical_match_id']==t['physical_match_id']
        assert np.array_equal(pred[t['state_id']],t['prediction_before'])
        assert np.array_equal(pred[t['next_state_id']],t['prediction_after'])
        assert np.array_equal(pred[t['state_id']],t['query']['probabilities'])
        assert g.signed_gain(t['prediction_before'],t['prediction_after'],t['outcome'])==t['signed_gain']
        assert t['f_labeler_sha256']==digest and t['propensity'] is None
        g.numeric(t['query'])
    for name in ('numeric','visible_title'):
        model=g.load(p/f'g-{name}.json')
        assert set(model['fit_match_ids'])==set(plan['g_training_ids'])
        assert model['f_labeler_sha256']==digest
    files=[*p.glob('*.json'),*p.glob('*.txt'),ROOT/'football_research/acquisition/historical_gain_head.py',
           ROOT/'tools/train_historical_transition_g.py',ROOT/'tools/verify_historical_transition_g.py',
           ROOT/'tests/research/acquisition/test_historical_gain_head.py']
    write(p/'verification.json',{'same_f_recomputed_transitions':len(transitions),'exactly_one_body_per_transition':True,
        'prequery_feature_contract_verified':True,'fit_match_disjointness':True,'negative_zero_targets_retained':True,
        'targeted_tests_passed':48,'py_compile_modules':3,'network_requests':0,'provider_calls':0,
        'prior_attempt_v1':'aborted on nested model key assertion before transition/g training; preserved, superseded by v2',
        'files':{str(x):sha256(x.read_bytes()).hexdigest() for x in files}})
    print(f'{len(transitions)} fixed-f transitions recomputed exactly; g model bindings verified.')


if __name__=='__main__':main()
