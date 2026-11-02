"""Incrementally encode new bodies; preserve old embeddings and held-out pools exactly."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_e_drive_news_corpus import read_rows
from build_e_drive_football_corpus import write_json,write_rows
from vectorize_e_drive_football_corpus import encode,link_matches
from football_research.acquisition.corpus_text_f import make_panel


def enrich_panel(original, proposed, train_ids, cap=4):
    protected={c['body_sha256'] for p in original if p['match_id'] not in train_ids for c in p['candidates']}
    byid={p['match_id']:p for p in proposed};output=[];added=0
    for p in original:
        if p['match_id'] not in train_ids:
            output.append(p);continue
        old=list(p['candidates']);ids={c['article_id'] for c in old};hashes={c['body_sha256'] for c in old}
        candidates=[c for c in byid.get(p['match_id'],{}).get('candidates',[])
                    if c['article_id'] not in ids and c['body_sha256'] not in protected and c['body_sha256'] not in hashes]
        # Latest eligible additions, preserving every original candidate. No label access.
        additions=[]
        for c in reversed(candidates):
            if len(old)+len(additions)>=cap:break
            if c['body_sha256'] in hashes:continue
            additions.append(c);hashes.add(c['body_sha256'])
        full=old+additions
        order={c['article_id']:i for i,c in enumerate(byid.get(p['match_id'],p)['candidates'])}
        full.sort(key=lambda c:order.get(c['article_id'],-1))
        output.append({**p,'candidates':full});added+=len(additions)
    return output,added


def main():
    ap=argparse.ArgumentParser()
    for key in ('old','batch','experiment','output','panel_output'):ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    for directory in ('structured','models'):shutil.copytree(a.old/directory,out/directory)
    for directory in ('vectors','news','research','reports'):(out/directory).mkdir()
    for path in (a.old/'vectors').glob('prematch_numeric*'):shutil.copy2(path,out/'vectors'/path.name)
    old=read_rows(a.old/'news/articles.jsonl');new=read_rows(a.batch/'news/articles.jsonl')
    assert not {r['article_id'] for r in old}&{r['article_id'] for r in new}
    write_rows(out/'news/articles.jsonl',new);encode(out)
    # New arrays are small. Concatenation retains all original vectors byte-for-byte.
    for name in ('article_titles','article_body_mean','article_body_chunks'):
        ov=np.load(a.old/f'vectors/{name}.npy');nv=np.load(out/f'vectors/{name}.npy')
        np.save(out/f'vectors/{name}.npy',np.concatenate([ov,nv]))
        orows=read_rows(a.old/f'vectors/{name}.rows.jsonl');nrows=read_rows(out/f'vectors/{name}.rows.jsonl')
        write_rows(out/f'vectors/{name}.rows.jsonl',orows+[{**r,'row':r['row']+len(orows)} for r in nrows])
        assert np.array_equal(np.load(out/f'vectors/{name}.npy')[:len(ov)],ov)
    embedding=json.loads((out/'reports/embeddings.json').read_text(encoding='utf8'))
    embedding.update(newly_encoded_articles=len(new),articles=len(old)+len(new),
        body_chunks=len(read_rows(out/'vectors/article_body_chunks.rows.jsonl')),old_vectors_preserved_exactly=True)
    write_json(out/'reports/embeddings.json',embedding)
    write_rows(out/'news/articles.jsonl',old+new);link_matches(out)
    articles={r['article_id']:r for r in old+new}
    matches={r['match_id']:r for r in read_rows(out/'structured/matches.jsonl')}
    proposed,_=make_panel(matches,read_rows(out/'research/match_article_candidates.jsonl'),articles)
    original=read_rows(a.experiment/'panel.jsonl')
    protocol=json.loads((a.experiment/'protocol.json').read_text(encoding='utf8'));train_ids=set(protocol['g_fit_ids'])
    panel,added=enrich_panel(original,proposed,train_ids)
    a.panel_output.mkdir(parents=True,exist_ok=False)
    write_rows(a.panel_output/'panel.jsonl',panel);shutil.copyfile(a.experiment/'f-base.json',a.panel_output/'f-base.json')
    before=[p for p in original if p['match_id'] in train_ids];after=[p for p in panel if p['match_id'] in train_ids]
    report=dict(schema='dense-training-corpus-v1',new_articles=len(new),total_articles=len(old)+len(new),
        original_f_validation_test_panels_unchanged=True,old_vectors_preserved_exactly=True,
        fixed_f_source=str(a.experiment/'f-body.json'),g_training_matches=len(after),added_training_links=added,
        before_candidates=sum(len(p['candidates']) for p in before),after_candidates=sum(len(p['candidates']) for p in after),
        before_multi=sum(len(p['candidates'])>1 for p in before),after_multi=sum(len(p['candidates'])>1 for p in after),
        candidate_count_distribution=dict(Counter(len(p['candidates']) for p in after)),
        train_pool_cap=4,selection_uses_labels=False,historical_reconstruction=True,
        limitations=['sitemap proximity is not relevance','date/club links are heuristic','current historical body versions'])
    write_json(a.panel_output/'panel-protocol.json',report);write_json(out/'reports/assembly.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
