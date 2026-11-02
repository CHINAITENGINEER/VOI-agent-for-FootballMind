"""Select archived sitemap neighbours of g-training articles, without outcome access."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from urllib.parse import urlsplit
from bs4 import BeautifulSoup
from build_e_drive_football_corpus import digest, write_json, write_rows
from build_e_drive_news_corpus import read_rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    articles={a['article_id']:a for a in read_rows(args.root/'news/articles.jsonl')}
    existing={a['url'] for a in articles.values()}
    protocol=json.loads((args.experiment/'protocol.json').read_text(encoding='utf8'))
    train=set(protocol['g_fit_ids'])
    panels=[p for p in read_rows(args.experiment/'panel.jsonl') if p['match_id'] in train]
    maps={}; candidates={};missing=[]
    for p in panels:
        for c in p['candidates']:
            a=articles[c['article_id']];path=a.get('sitemap_raw_path')
            if not path or not Path(path).exists():
                missing.append(a['article_id']);continue
            if path not in maps:
                raw=Path(path).read_bytes()
                urls=[r.loc.text for r in BeautifulSoup(raw,'xml').select('url') if r.loc]
                maps[path]=(urls,{u:i for i,u in enumerate(urls)},digest(raw))
            urls,positions,sha=maps[path]
            if a['url'] not in positions:
                missing.append(a['article_id']);continue
            pos=positions[a['url']]
            for distance in (-1,1,-2,2):
                ix=pos+distance
                if not 0<=ix<len(urls):continue
                url=urls[ix]
                if url in existing or urlsplit(url).hostname!=a['host']:continue
                if '/news/' not in url:continue
                record=candidates.setdefault(url,dict(url=url,host=a['host'],source_team=a['source_team'],
                    sitemap_raw_path=path,sitemap_sha256=sha,sitemap_position=ix,
                    selection_basis='g_training_article_sitemap_neighbour_not_date_or_relevance_proof',
                    anchors=[],selected_for_corpus=False))
                record['anchors'].append(dict(match_id=p['match_id'],article_id=a['article_id'],distance=distance))
    # Bound at 60 per host; nearest items first, stable hash tie-break independent of results.
    for host in sorted({c['host'] for c in candidates.values()}):
        pool=sorted((c for c in candidates.values() if c['host']==host),
            key=lambda c:(min(abs(a['distance']) for a in c['anchors']),digest(c['url'].encode())))
        for c in pool[:60]:c['selected_for_corpus']=True
    values=list(candidates.values());write_rows(args.output/'candidate-plan.jsonl',values)
    result=dict(schema='dense-g-training-body-plan-v1',training_match_ids=sorted(train),
        outcomes_read=False,selection_uses_outcome_or_gain=False,per_host_limit=60,
        neighbours_are_not_verified_match_links=True,match_linking_requires_actual_publication_after_fetch=True,
        heldout_panels_to_remain_unchanged=True,missing_anchor_maps=missing,
        candidates=len(values),selected=dict(Counter(c['host'] for c in values if c['selected_for_corpus'])))
    write_json(args.output/'protocol.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
