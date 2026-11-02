"""Outcome-free temporal panel for postread g, with a bounded common evidence pool."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_research.acquisition.corpus_text_f import make_panel


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def prepare(matches, links, articles, *, boundary='2024-01-01', cap=8, base_ids=()):
    if cap < 1:
        raise ValueError('Positive pool cap required')
    panel, rejected = make_panel(matches, links, articles)
    def group(p):
        return 'f_fit' if p['split']=='train' and p['date'] < boundary else p['split']
    hashes = defaultdict(set)
    eligible = []
    for p in panel:
        if p['date'] < '2019-07-01' or p['match_id'] in base_ids:
            rejected.extend({**c, 'reason':'overlaps_numeric_base_fit_period_or_identity'} for c in p['candidates'])
            continue
        eligible.append(p)
        for c in p['candidates']:
            hashes[c['body_sha256']].add(group(p))
    output = []
    for p in eligible:
        kept = []
        for c in p['candidates']:
            if len(hashes[c['body_sha256']]) > 1:
                rejected.append({**c, 'reason':'shared_body_crosses_four_temporal_groups'})
            else:
                kept.append(c)
        # Already publication ordered. Bound exponential enumeration identically
        # for all policies, without consulting contents, outcomes, or gains.
        rejected.extend({**c, 'reason':'common_pool_cap_latest_publication_order'} for c in kept[:-cap])
        kept = kept[-cap:]
        if kept:
            output.append({**p, 'candidates':kept})
    return output, rejected


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--previous',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    matches={r['match_id']:r for r in rows(a.root/'structured/matches.jsonl')}
    articles={r['article_id']:r for r in rows(a.root/'news/articles.jsonl')}
    base=json.loads((a.previous/'f-base.json').read_text(encoding='utf-8'))
    panel,rejected=prepare(matches, rows(a.root/'research/match_article_candidates.jsonl'), articles,
        base_ids=set(base.get('fit_match_ids',[])))
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(a.previous/'f-base.json',a.output/'f-base.json')
    for name,rs in [('panel.jsonl',panel),('excluded-links.jsonl',rejected)]:
        (a.output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')
    stats={}
    for name,ps in [('f_fit',[p for p in panel if p['date']<'2024-01-01']),
        ('g_fit',[p for p in panel if '2024-01-01'<=p['date']<'2025-07-01']),
        ('validation',[p for p in panel if p['split']=='validation']),
        ('test',[p for p in panel if p['split']=='test'])]:
        stats[name]=dict(matches=len(ps),links=sum(len(p['candidates']) for p in ps),
            multi_candidate_matches=sum(len(p['candidates'])>1 for p in ps),
            source_team_links=dict(Counter(articles[c['article_id']]['source_team'] for p in ps for c in p['candidates'])))
    protocol=dict(schema='expanded-postread-panel-v1',f_g_boundary='2024-01-01',pool_cap=8,
        pool_cap_rule='latest_publication_order_without_outcomes',base_fit_excluded_before='2019-07-01',
        cross_group_body_hashes_removed=True,outcomes_loaded=False,
        historical_reconstruction=True,production_modified=False,coverage=stats)
    (a.output/'panel-protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(protocol,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
