"""Outcome-free, append-only historical body expansion and corpus assembly."""
from __future__ import annotations
import argparse
import concurrent.futures
from collections import Counter
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urlsplit
from bs4 import BeautifulSoup
from build_e_drive_football_corpus import fetch, digest, write_json, write_rows
from build_e_drive_news_corpus import TOPICS, parse_article, read_rows


def plan(old, batch):
    existing = {r['url'] for r in read_rows(old/'news/articles.jsonl')}
    candidates = []
    # These exact child sitemap URLs were observed in the archived official index.
    for n in (6,7,8,9,10,11):
        path = batch/f'raw/arsenal-map-{n}.xml'
        if not path.exists():
            raise ValueError(f'Missing observed sitemap {path}')
        data=path.read_bytes(); map_sha=digest(data)
        soup = BeautifulSoup(data,'xml')
        group = []
        for i,row in enumerate(soup.select('url')):
            if row.loc is None: continue
            url = row.loc.text
            if urlsplit(url).hostname != 'www.arsenal.com' or '/news/' not in url: continue
            group.append(dict(url=url,host='www.arsenal.com',source_team='Arsenal',sitemap_position=i,
                sitemap_raw_path=str(path),sitemap_sha256=map_sha,sitemap_page=n,
                sitemap_lastmod=row.lastmod.text if row.lastmod else None))
        pool = [i for i,c in enumerate(group) if TOPICS.search(c['url']) and c['url'] not in existing]
        # Ten dispersed topic anchors and their following two articles per page.
        # Adjacent items are retained regardless of topic; dates are parsed later.
        anchors = sorted(pool,key=lambda i:digest(group[i]['url'].encode()))[:10]
        chosen = {j for i in anchors for j in range(i,min(i+3,len(group)))}
        for i,c in enumerate(group):
            c['selected_for_corpus'] = i in chosen and c['url'] not in existing
            c['selection_basis'] = 'url_hash_topic_anchor_and_two_sitemap_neighbors_no_labels'
            candidates.append(c)
    pool = [c for c in read_rows(old/'news/discovery.jsonl') if c['host']=='www.liverpoolfc.com' and c['url'] not in existing]
    # Balance additional topic and ordinary material without using scores/gains.
    selected = set()
    for topic,count in [('evidence_topic',60),('other_topic',20)]:
        selected.update(c['url'] for c in sorted((c for c in pool if c['topic_stratum']==topic),
                        key=lambda c:digest(c['url'].encode()))[:count])
    candidates.extend({**c,'selected_for_corpus':c['url'] in selected,
        'selection_basis':'url_hash_host_topic_no_labels'} for c in pool)
    unique = {}
    for c in candidates:
        if c['url'] not in unique or c['selected_for_corpus']: unique[c['url']]=c
    values = list(unique.values())
    write_rows(batch/'candidate-plan.jsonl',values)
    write_json(batch/'protocol.json',dict(schema='historical-text-expansion-v2',
        old_corpus=str(old),source_sitemap_date_is_not_publication=True,selection_uses_labels=False,
        modified_after_match_excluded_by_existing_panel_builder=True,per_host_workers=1,
        retry_failed_requests=False,production_modified=False,
        selected=dict(Counter(c['host'] for c in values if c['selected_for_corpus']))))
    return values


def collect(old,batch):
    batch.mkdir(parents=True,exist_ok=True)
    values = read_rows(batch/'candidate-plan.jsonl') if (batch/'candidate-plan.jsonl').exists() else plan(old,batch)
    def worker(host):
        good,bad = [],[]
        cs=[c for c in values if c['selected_for_corpus'] and c['host']==host]
        for i,c in enumerate(cs):
            aid=digest(c['url'].encode())[:24]; rel=f'raw/news/{host}/{aid}.html'
            try:
                record_path=batch/f'news/records/{aid}.json'
                if record_path.exists():
                    record=json.loads(record_path.read_text(encoding='utf-8'))
                    if digest(Path(record['body_path']).read_bytes())!=record['body_sha256']: raise ValueError('body digest')
                else:
                    receipt_path=(batch/rel).with_suffix('.html.receipt.json')
                    if receipt_path.exists():
                        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
                        if receipt.get('error') or receipt.get('status')!=200: raise ValueError('prior_failed_request_not_retried')
                    path,receipt=fetch(batch,c['url'],rel,max_bytes=8_000_000)
                    record=parse_article(path.read_bytes(),host)
                    body=batch/f'news/bodies/{aid}.txt';body.parent.mkdir(parents=True,exist_ok=True)
                    body.write_text(record.pop('body'),encoding='utf-8',newline='\n');record.pop('paragraphs')
                    record.update(c,article_id=aid,body_path=str(body),raw_path=str(path),raw_sha256=receipt['sha256'],
                        captured_at=receipt['response_completed_at'],available_at=None,historical_version_unknown=True,
                        provenance='historical_reconstruction',strict_prospective=False,
                        semantic_review_status='not_reviewed',training_eligibility='requires_match_binding_and_f_transition')
                    write_json(record_path,record)
                good.append(record)
            except Exception as exc:
                bad.append(dict(article_id=aid,url=c['url'],host=host,reason=str(exc)))
            if (i+1)%10==0 or i+1==len(cs):print('EXPAND',host,i+1,len(cs),'bodies',len(good),'failures',len(bad),flush=True)
            time.sleep(.35)
        return good,bad
    good,bad=[],[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for gs,bs in pool.map(worker,sorted({c['host'] for c in values if c['selected_for_corpus']})):
            good.extend(gs);bad.extend(bs)
    write_rows(batch/'news/articles.jsonl',sorted(good,key=lambda c:c['article_id']))
    write_rows(batch/'news/failures.jsonl',bad)
    write_json(batch/'collection-report.json',dict(bodies=len(good),failures=len(bad),
        source_years={str(k):v for k,v in Counter((c['source_team'],str(c['published_at'])[:4]) for c in good).items()},
        network_ledger=str(batch/'http-ledger.jsonl'),g_training_rows=0))


def assemble(old,batch,out):
    out.mkdir(parents=True,exist_ok=False)
    for directory in ['structured','models']:
        shutil.copytree(old/directory,out/directory)
    (out/'vectors').mkdir();(out/'reports').mkdir();(out/'news').mkdir();(out/'research').mkdir()
    for path in (old/'vectors').glob('prematch_numeric*'):shutil.copy2(path,out/'vectors'/path.name)
    articles=read_rows(old/'news/articles.jsonl')+read_rows(batch/'news/articles.jsonl')
    assert len({a['article_id'] for a in articles})==len(articles)
    write_rows(out/'news/articles.jsonl',sorted(articles,key=lambda a:a['article_id']))
    write_json(out/'reports/assembly.json',dict(old_corpus=str(old),supplement=str(batch),articles=len(articles),
        body_storage='immutable_referenced_original_paths',production_modified=False))
    from vectorize_e_drive_football_corpus import link_matches,encode
    link_matches(out);encode(out)
    print('ASSEMBLED',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--assemble',type=Path);args=p.parse_args()
    if args.assemble:assemble(args.old,args.batch,args.assemble)
    else:collect(args.old,args.batch)
