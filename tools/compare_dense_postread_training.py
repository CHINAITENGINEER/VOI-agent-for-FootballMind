"""Controlled comparison with frozen f, unchanged evaluation pools and original vectors."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_e_drive_postread_g import ResearchData,SelectorInputs,rollout,evaluate
from tools.train_e_drive_rolling_g import transitions,dump,digest,write_rows


def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]


def main():
    ap=argparse.ArgumentParser()
    for key in ('root','old','new'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();oldp=json.loads((a.old/'protocol.json').read_text(encoding='utf8'))
    newp=json.loads((a.new/'protocol.json').read_text(encoding='utf8'))
    for key in ('f_fit_ids','g_fit_ids','g_validation_ids','evaluation_ids'):assert oldp[key]==newp[key]
    assert digest(a.old/'f-body.json')==digest(a.new/'f-body.json')
    oldpanel=rows(a.old/'panel.jsonl');panel=rows(a.new/'panel.jsonl')
    train=set(newp['g_fit_ids'])
    assert [p for p in oldpanel if p['match_id'] not in train]==[p for p in panel if p['match_id'] not in train]
    for split in ('validation','test'):
        assert rows(a.old/f'{split}-transitions.jsonl')==rows(a.new/f'{split}-transitions.jsonl')
    oldr=json.loads((a.old/'report.json').read_text(encoding='utf8'))
    newr=json.loads((a.new/'report.json').read_text(encoding='utf8'))
    data=ResearchData(a.root,a.new,panel);inputs=SelectorInputs(data)
    f=json.loads((a.new/'f-body.json').read_text(encoding='utf8'));fsha=digest(a.new/'f-body.json')
    result={'fixed_f_exact':True,'evaluation_panels_and_all_transition_labels_exact':True,
        'independent_g_matches_unchanged':len(train),
        'training_transitions_before':oldr['g_training_transitions'],'training_transitions_after':newr['g_training_transitions'],
        'multi_candidate_matches_before':oldr['training_matches_with_multiple_candidates'],
        'multi_candidate_matches_after':newr['training_matches_with_multiple_candidates'],
        'selection_by_validation_only':True,'test_previously_examined':True,'production_promoted':False,'splits':{}}
    changes=[]
    for split,key in [('validation','g_validation_ids'),('test','evaluation_ids')]:
        group=[p for p in panel if p['match_id'] in set(newp[key])]
        _,_,states=transitions(data,group,f,fsha,split)
        family={};details={}
        for name,path in [('old_fixed_ridge',a.old/'g-body_state-ridge-10.0.json'),
                          ('new_fixed_ridge',a.new/'g-body_state-ridge-10.0.json')]:
            model=json.loads(path.read_text(encoding='utf8'))
            summary,detail=evaluate(data,group,states,rollout(inputs,group,states,model,3),3)
            family[name]=summary;details[name]={r['match_id']:r for r in detail}
        diff=np.array([details['old_fixed_ridge'][p['match_id']]['loss']['g']-
                       details['new_fixed_ridge'][p['match_id']]['loss']['g'] for p in group])
        for p,d in zip(group,diff):
            changes.append({'split':split,'match_id':p['match_id'],'loss_reduction_same_ridge':float(d)})
        result['splits'][split]={'selected_models':{'old':oldr['comparisons'][split]['body_state']['budgets']['3'],
            'new':newr['comparisons'][split]['body_state']['budgets']['3']},'same_ridge_family':family,
            'same_ridge_loss_reduction':float(diff.mean()),'same_ridge_improved_matches':int(sum(diff>1e-12)),
            'same_ridge_worsened_matches':int(sum(diff < -1e-12)), 'same_ridge_unchanged_matches':int(sum(abs(diff)<=1e-12))}
    dump(a.new/'dense-training-comparison.json',result)
    write_rows(a.new/'same-family-per-match.jsonl',changes)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
