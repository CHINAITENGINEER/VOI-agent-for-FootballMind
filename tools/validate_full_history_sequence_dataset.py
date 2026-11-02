"""Independent integrity and causal-metadata checks; no model calls."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,math,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from football_research.acquisition.full_history_sequence_metadata import document_metadata,METADATA_NAMES


def validate(path):
    def rows(name):
        with (path/name).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
    mm=rows('matches.jsonl');m={r['physical_match_id']:r for r in mm}
    ss=rows('sequences.jsonl');yy=rows('outcomes.jsonl');y={r['physical_match_id']:r for r in yy}
    dd=rows('documents.jsonl');d={r['document_id']:r for r in dd};q=rows('eligible-encoding-inputs.jsonl')
    groups=defaultdict(set);bad=[]
    for s in ss:
        if not s['eligible']:continue
        assert s['physical_match_id'] in m and y[s['physical_match_id']]['outcome_class'] in (0,1,2)
        prefix=[]
        for item,values in zip(s['documents'],s['document_meta']):
            doc={**d[item['document_id']],**item['metadata_context']}
            if document_metadata(doc,prefix,s['match_date'])!=values:bad.append((s['physical_match_id'],item['document_id']))
            prefix.append(doc);groups[item['document_id']].add(s['fit_role'])
    checks={
        'unique_master_ids':len(m)==len(mm),'outcome_registry_alignment':set(y)==set(m),
        'sequence_registry_alignment':{r['physical_match_id'] for r in ss}==set(m),
        'metadata_width_matches_schema':all(len(v)==len(METADATA_NAMES) and all(math.isfinite(x) for x in v) for s in ss for v in s['document_meta']),
        'metadata_exact_reproduction':not bad,'prefix_weight_match_balance':all(abs(sum(s['prefix_weights'])-1)<1e-12 for s in ss),
        'no_cross_role_exact_body':all(len(v)==1 for v in groups.values()),
        'every_trainable_body_in_encoding_queue':set(groups)=={r['document_id'] for r in q},
        'no_duplicate_encoding_sha':len({r['text_sha256'] for r in q})==len(q),
        'all_encoding_paths_present':all(Path(r['text_path']).is_file() for r in q),
        'empty_prefix_retained':all(s['prefix_lengths'][0]==0 for s in ss),
        'f_dev_before_gfit':max(r['match_date'] for r in mm if r['fit_role']=='f_dev')<min(r['match_date'] for r in mm if r['fit_role']=='g_fit'),
        'link_meta_matches_schema':all('document_meta' not in r or len(r['document_meta'])==len(METADATA_NAMES) for r in rows('match-document-links.jsonl')),
    }
    result={'checks':checks,'all_passed':all(checks.values()),
        'main_sequence_claims_known_documents':sum(d[k].get('existing_claims') is not None for k in groups),
        'all_registry_claims_known_documents':sum(z.get('existing_claims') is not None for z in dd),
        'conflicting_outcome_matches':sum(z['conflicting_results'] for z in yy),
        'missing_outcome_matches':sum(z['outcome_class'] is None for z in yy),
        'registry_fit_roles':dict(Counter(z['fit_role'] for z in mm)),
        'previously_exposed_main_test_matches':sum(s['eligible'] and s['fit_role']=='test' and s['previous_development_exposure'] for s in ss),
        'network_requests':0,'training':0}
    (path/'data-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert result['all_passed'],result
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1/data'))
    print(json.dumps(validate(parser.parse_args().data),ensure_ascii=False,indent=2))
