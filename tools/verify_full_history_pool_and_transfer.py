"""Final read-only pool coverage and saved cross-f transfer verification."""
from __future__ import annotations
import ast
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path('E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1')
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('E:/FootballMind/experiments/postread-clean-expansion-20261025-v1')
SEM=Path('E:/FootballMind/experiments/postread-qwen-semantic-20261025-v1')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').split('\n') if s]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(ok,msg):
    if not ok:raise ValueError(msg)


def main():
    out=BASE/'code-checks/g-training/final-pool-and-legacy-transfer-verification.json'
    if out.exists():raise FileExistsError(out)
    sources=[ROOT/'football_research/acquisition/sequential_postread_g.py',ROOT/'tools/train_full_history_sequence_g.py',ROOT/'tools/evaluate_full_history_sequence_policies.py',ROOT/'tools/evaluate_legacy_g_transfer_subset.py']
    tree=ast.parse(sources[0].read_text(encoding='utf8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EvidencePanel')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    defaults=dict(zip([a.arg for a in init.args.kwonlyargs],init.args.kw_defaults))
    check(isinstance(defaults['max_candidates'],ast.Constant) and defaults['max_candidates'].value is None,'Default candidate cap exists')
    calls=[]
    for path in sources[1:]:
        nodes=[n for n in ast.walk(ast.parse(path.read_text(encoding='utf8'))) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='EvidencePanel']
        check(nodes and all('max_candidates' not in {k.arg for k in n.keywords} for n in nodes),'Caller supplies candidate cap')
        calls.extend(dict(path=str(path),line=n.lineno,max_candidates_keyword_absent=True) for n in nodes)
    sequences={r['physical_match_id']:r for r in rows(BASE/'data/sequences.jsonl') if r['eligible']}
    matches={r['physical_match_id']:r for r in rows(BASE/'data/matches.jsonl')}
    outcomes={r['physical_match_id']:r['outcome_class'] for r in rows(BASE/'data/outcomes.jsonl')}
    old_docs=defaultdict(dict)
    for r in rows(OLD/'documents.jsonl'):old_docs[r['physical_match_id']][r['candidate_index']]=r
    poolstats=[]
    for role in ('g_fit','g_dev','validation','test'):
        ss=[s for s in sequences.values() if s['split_role']==role]
        poolstats.append(dict(split_role=role,matches=len(ss),candidate_occurrences=sum(len(s['document_ids']) for s in ss),maximum_candidates=max(len(s['document_ids']) for s in ss),
            matches_at_least_seven=sum(len(s['document_ids'])>=7 for s in ss),distribution=dict(Counter(len(s['document_ids']) for s in ss)),
            large_pools=[dict(physical_match_id=s['physical_match_id'],candidates=len(s['document_ids'])) for s in ss if len(s['document_ids'])>=7]))
    archreports=[];bindings_previous=None
    with np.load(BASE/'gpu-import-v1/embeddings.npz',allow_pickle=False) as vectors:
        vector_keys=set(vectors.files)
    for arch,fname in (('set','set_residual_mlp'),('gru','gru')):
        gdir=BASE/'g-models'/arch;edir=BASE/'policy-evaluation'/arch;tdir=BASE/'legacy-g-transfer-v1'/arch
        labels=rows(gdir/'labels.jsonl')+rows(edir/'labels.jsonl')
        s0=defaultdict(set);reference={};reference_variation=0.
        def put(mid,adopted,p):
            nonlocal reference_variation
            key=(mid,tuple(sorted(adopted)) if arch=='set' else tuple(adopted));arr=np.asarray(p,dtype=float)
            if key in reference:reference_variation=max(reference_variation,float(np.max(np.abs(reference[key]-arr))))
            else:reference[key]=arr
        for r in labels:
            mid=r['physical_match_id']
            if not r['adopted_ids']:s0[mid].add(r['candidate_id'])
            put(mid,r['adopted_ids'],r['p_before']);put(mid,r['adopted_ids']+[r['candidate_id']],r['p_after'])
        target={mid:s for mid,s in sequences.items() if s['split_role'] in ('g_fit','g_dev','validation','test')}
        check(set(s0)==set(target) and all(s0[mid]==set(s['document_ids']) for mid,s in target.items()),'Missing candidate s0 supervision')
        mainpaths=rows(edir/'per-match-paths.jsonl')
        for r in mainpaths:
            check(set(r['selected_ids'])|set(r['unselected_ids'])==set(sequences[r['physical_match_id']]['document_ids']),'Heldout policy candidate pool shortened')
            put(r['physical_match_id'],[],r['p_before']);put(r['physical_match_id'],r['selected_ids'],r['p_after'])
        comparison=read(tdir/'comparison.json');bindings=rows(tdir/'bindings.jsonl');paths=rows(tdir/'paths.jsonl')
        if bindings_previous is not None:check(bindings==bindings_previous,'Different legacy common pools for set and GRU')
        bindings_previous=bindings
        bmap={b['physical_match_id']:b for b in bindings};check(len(bmap)==len(bindings)==46,'Legacy match count')
        check(len(paths)==46*3*15,'Legacy strategy path count')
        for b in bindings:
            mid=b['physical_match_id'];old=b['old_id']
            check(old in [mid,*matches[mid].get('aliases',[])],'Unverified alias mapping')
            ordered=['sha256:'+old_docs[old][i]['text_sha256'] for i in range(len(old_docs[old]))]
            check(b['documents']==ordered and set(ordered).issubset(sequences[mid]['document_ids']),'Wrong candidate/body SHA mapping')
            check(all(d.removeprefix('sha256:') in vector_keys for d in ordered),'Missing vector SHA')
        fsha=digest(BASE/'f-models'/('selected-'+fname)/'model.pt');gsha=digest(gdir/'g-models.joblib');oldsha=digest(SEM/'training/models.joblib')
        check(comparison['new_f_sha256']==fsha and comparison['new_g_sha256']==gsha and comparison['old_g_sha256']==oldsha,'Transfer model SHA mismatch')
        maxloss=maxgain=maxp=0.;compared=0;missing=set();groups=defaultdict(list);basep={};selectedprob={}
        for r in paths:
            mid=r['physical_match_id'];b=bmap[mid];y=outcomes[mid]
            check(y==r['outcome_class'] and fsha==r['f_model_sha256'],'Outcome/f binding mismatch')
            check(r['selected_ids']==[b['documents'][i] for i in r['selected_indices']],'Adoption index/body mismatch')
            check(set(r['selected_ids'])|set(r['unselected_ids'])==set(b['documents']) and not set(r['selected_ids'])&set(r['unselected_ids']),'Different transfer policy pools')
            before=np.asarray(r['p_before']);after=np.asarray(r['p_after'])
            check(np.isfinite(before).all() and np.isfinite(after).all() and min(before)>0 and min(after)>0 and abs(sum(before)-1)<1e-6 and abs(sum(after)-1)<1e-6,'Invalid transfer probability')
            maxloss=max(maxloss,abs(float(-np.log(after[y]))-r['final_log_loss']))
            maxgain=max(maxgain,abs(float(np.log(after[y])-np.log(before[y]))-r['realized_gain']))
            if mid in basep:check(np.array_equal(before,basep[mid]),'Different baseline across common pool policies')
            else:basep[mid]=before
            for adopted,p in (([],before),(r['selected_ids'],after)):
                key=(mid,tuple(sorted(adopted)) if arch=='set' else tuple(adopted))
                if key in reference:maxp=max(maxp,float(np.max(abs(reference[key]-p))));compared+=1
                else:missing.add(key)
                if key in selectedprob:check(np.max(abs(selectedprob[key]-p))<1e-6,'Inconsistent transfer prefix probability')
                else:selectedprob[key]=p
            groups[(r['strategy'],r['budget'])].append(r)
        summary_diff=0.
        for s in comparison['summary']:
            rr=groups[(s['strategy'],s['budget'])]
            counts=Counter(r['physical_match_id'] for r in rr)
            check(set(counts)==set(bmap) and set(counts.values())=={10 if s['strategy']=='random' else 1},'Transfer path coverage mismatch')
            summary_diff=max(summary_diff,abs(float(np.mean([r['final_log_loss'] for r in rr]))-s['mean_final_log_loss']))
        check(maxloss<1e-12 and maxgain<1e-12 and summary_diff<1e-12 and maxp<1e-6,'Transfer numerical mismatch')
        archreports.append(dict(architecture=arch,full_s0_pool_matches=len(s0),full_s0_candidate_occurrences=sum(map(len,s0.values())),
            main_heldout_pool_paths_verified=len(mainpaths),legacy_matches=len(bindings),legacy_paths=len(paths),
            legacy_candidate_occurrences=sum(len(b['documents']) for b in bindings),legacy_unique_bodies=len({d for b in bindings for d in b['documents']}),
            old_g_sha256=oldsha,new_g_sha256=gsha,new_f_sha256=fsha,max_log_loss_difference=maxloss,max_gain_difference=maxgain,max_summary_difference=summary_diff,
            saved_main_f_probability_comparisons=compared,max_saved_main_f_probability_difference=maxp,main_saved_reference_numerical_variation=reference_variation,
            transfer_states_without_main_saved_reference=len(missing),missing_reference_states=[dict(physical_match_id=k[0],adopted_ids=list(k[1])) for k in sorted(missing)],
            no_new_inference=True,no_training=True,legacy_scope=comparison['scope'],
            budget3_summary=[s for s in comparison['summary'] if s['budget']==3],
            source_hashes={str(p):digest(p) for p in sorted(tdir.iterdir()) if p.is_file()}))
    result=dict(status='passed',candidate_cap_default=None,caller_checks=calls,pool_statistics=poolstats,architectures=archreports,
        sources_sha256={str(p):digest(p) for p in sources},new_network_requests=0,new_inference=False,new_training=False,
        limitations=['Legacy transfer is previously studied diagnostic, not independent validation.','Saved-state probability comparison is limited to available exact current-prefix references; unavailable references are listed, not regenerated.'])
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='passed',output=str(out),pool_statistics=poolstats,architectures=[{k:v for k,v in r.items() if k not in ('source_hashes','missing_reference_states','budget3_summary')} for r in archreports]),ensure_ascii=False))


if __name__=='__main__':main()
