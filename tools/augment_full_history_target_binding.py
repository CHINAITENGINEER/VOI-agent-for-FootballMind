"""Add observed target/source relationships, preserving split and label bytes."""
from __future__ import annotations
import argparse,hashlib,json,re,sys
from pathlib import Path
from collections import Counter,defaultdict
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from football_research.acquisition.full_history_sequence_metadata import METADATA_NAMES,document_metadata
from tools.build_full_history_sequence_dataset import ALIASES,norm,rows,write_rows,write_json


def exact_mentions(text,aliases):
    """Whole lexical tokens; overlapping aliases count once. No semantic claim."""
    spans=set()
    for alias in aliases:
        tokens=alias.split()
        if not tokens:continue
        pattern=r'(?<!\w)'+r'\s+'.join(re.escape(t) for t in tokens)+r'(?!\w)'
        spans.update(m.span() for m in re.finditer(pattern,text,flags=re.IGNORECASE))
    # Prefer the longest overlapping explicit name, rather than double-counting it.
    chosen=[]
    for a,b in sorted(spans,key=lambda x:(-(x[1]-x[0]),x[0])):
        if not any(a<d and b>c for c,d in chosen):chosen.append((a,b))
    return len(chosen)


def augment(out):
    registry_path=ROOT/'var/corpora/football-news-pipeline-v1/00_registry/club_site_registry.json'
    aliases_path=ROOT/'var/corpora/football-news-pipeline-v1/config/club_sites_extra.json'
    registry=json.loads(registry_path.read_text(encoding='utf-8'))
    saved_aliases=json.loads(aliases_path.read_text(encoding='utf-8')).get('team_search_aliases',{})
    hosts=defaultdict(set);hostrefs=defaultdict(list)
    for key,record in registry.get('verified',{}).items():
        if not record.get('verified_domain'):continue
        team=record.get('team') or key
        observed=[record['verified_domain']]+[urlsplit(s).hostname for s in record.get('sitemaps',[])]
        for h in observed:
            if h:
                hosts[h.casefold()].add(team)
                hostrefs[h.casefold()].append({'path':str(registry_path),'record_key':['verified',key],'basis':'saved_verified_domain_or_observed_sitemap_host'})
    article_owner={}
    p=Path('E:/FootballMind/corpus-v3/news/articles.jsonl')
    for _,r in rows(p):
        if r.get('source_team'):article_owner[r['article_id']]={'team':r['source_team'],'source':{'path':str(p),'article_id':r['article_id'],'field':'source_team'}}
    matches={r['physical_match_id']:r for _,r in rows(out/'matches.jsonl')}
    documents={r['document_id']:r for _,r in rows(out/'documents.jsonl')}
    links={(r['physical_match_id'],r['document_id']):r for _,r in rows(out/'match-document-links.jsonl')}
    sequences=[r for _,r in rows(out/'sequences.jsonl')]
    queue=[r for _,r in rows(out/'eligible-encoding-inputs.jsonl')]
    texts={}
    for r in queue:
        text=Path(r['text_path']).read_bytes().decode('utf-8')
        if hashlib.sha256(text.encode('utf-8')).hexdigest()!=r['text_sha256']:raise ValueError('Body changed before target binding')
        texts[r['document_id']]=text
    protected={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['matches.jsonl','outcomes.jsonl','split.json','eligible-encoding-inputs.jsonl','encoding-candidates.jsonl']}
    aliasmap=defaultdict(set);aliasrefs=defaultdict(list)
    for key,aa in saved_aliases.items():
        aliasmap[norm(key)].update([key]+aa)
        aliasrefs[norm(key)].append({'path':str(aliases_path),'record_key':['team_search_aliases',key]})
    def team_aliases(team):
        if not isinstance(team,str) or not team.strip():return [],[],True
        normalized=norm(team);names={team};refs=[{'basis':'saved_structured_team_name','name':team}]
        for source,target in ALIASES.items():
            if normalized==target:names.update([source,target]);refs.append({'path':str(ROOT/'tools/build_full_history_sequence_dataset.py'),'basis':'frozen_explicit_identity_alias','source':source,'target':target})
        names.update(aliasmap.get(normalized,set()));refs.extend(aliasrefs.get(normalized,[]))
        # Registered generic search shortcuts are not sufficiently specific lexical names.
        generic={'city','united','town','rovers','forest','wednesday','fc','ac','sc'}
        names={n for n in names if n.strip() and (n.casefold() not in generic or norm(n)==normalized)}
        return sorted(names,key=str.casefold),refs,not bool(aliasrefs.get(normalized))
    for s in sequences:
        if not s['document_ids']:continue
        m=matches[s['physical_match_id']]
        ha,hr,hpartial=team_aliases(m['home_team']);aa,ar,apartial=team_aliases(m['away_team'])
        objects=[]
        for item in s['documents']:
            did=item['document_id'];d=documents[did];link=links[(s['physical_match_id'],did)]
            text=texts.get(did)
            owners=set();refs=[]
            for aid in d['article_ids']:
                if aid in article_owner:owners.add(article_owner[aid]['team']);refs.append(article_owner[aid]['source'])
            for source in d['source_versions']:
                h=(source.get('source_domain') or '').casefold()
                owners.update(hosts.get(h,set()));refs.extend(hostrefs.get(h,[]))
            normalized_owners={norm(t) for t in owners}
            owner=next(iter(owners)) if len(normalized_owners)==1 else None
            unknown=text is None or not ha or not aa
            binding={'exact_home_team_mentions':exact_mentions(text,ha) if text is not None and ha else 0,
                'exact_away_team_mentions':exact_mentions(text,aa) if text is not None and aa else 0,
                'target_aliases_unknown':unknown,'target_aliases_partial':hpartial or apartial,
                'source_owner_team':owner,'source_owner_candidates':sorted(owners),'source_owner_evidence':refs,
                'source_target_home':bool(owner and norm(owner)==norm(m['home_team'])),
                'source_target_away':bool(owner and norm(owner)==norm(m['away_team'])),
                'source_target_unknown':owner is None,'home_team':m['home_team'],'away_team':m['away_team'],
                'home_aliases':ha,'away_aliases':aa,'home_alias_evidence':hr,'away_alias_evidence':ar,
                'mention_basis':'case-insensitive Unicode whole-token lexical match; overlapping aliases count once; not a factual/mens-team/scope claim',
                'aliases_partial_meaning':'at least one target lacks an additional saved team_search_aliases entry; no exhaustive multilingual-coverage claim',
                'full_body_text_sha256':d['text_sha256']}
            item['metadata_context']['target_binding']=binding
            link['target_binding']=binding
            obj={**d,**item['metadata_context']};objects.append(obj)
        metas=[document_metadata(d,objects[:i],m['match_date']) for i,d in enumerate(objects)]
        s['document_meta']=metas
        for i,item in enumerate(s['documents']):
            item['meta']=metas[i];links[(s['physical_match_id'],item['document_id'])]['document_meta']=metas[i]
    # Diagnostics have no inferred text relationships. Unknown flags are computed if requested.
    for link in links.values():
        if 'target_binding' not in link:link['target_binding']=None
        if not link['historical_sequence_eligible']:link.pop('document_meta',None)
    schema=json.loads((out/'schema.json').read_text(encoding='utf-8'))
    schema['schema']='footballmind-full-history-sequential-fg-v2-target-binding'
    schema['document_meta_names']=METADATA_NAMES
    schema['target_binding_schema']='observable-target-binding-v1'
    schema['target_binding_rule']='Saved structure/explicit aliases lexical mentions and saved registry owner only; no URL/name guessing; no outcome or f output.'
    write_json(out/'schema.json',schema);write_rows(out/'sequences.jsonl',sequences);write_rows(out/'match-document-links.jsonl',links.values())
    values=[item['metadata_context']['target_binding'] for s in sequences if s['eligible'] for item in s['documents']]
    report={'schema':'observable-target-binding-augmentation-v1','metadata_dimension':len(METADATA_NAMES),'main_sequence_document_occurrences':len(values),
        'full_texts_read_once':len(texts),'known_target_lexical_counts':sum(not b['target_aliases_unknown'] for b in values),
        'mentions_home':sum(b['exact_home_team_mentions']>0 for b in values),'mentions_away':sum(b['exact_away_team_mentions']>0 for b in values),
        'owner_home':sum(b['source_target_home'] for b in values),'owner_away':sum(b['source_target_away'] for b in values),'owner_unknown':sum(b['source_target_unknown'] for b in values),
        'protected_inputs_unchanged':{name:hashlib.sha256((out/name).read_bytes()).hexdigest()==sha for name,sha in protected.items()},
        'protected_sha256':protected,'network_requests':0,'training':0}
    assert all(report['protected_inputs_unchanged'].values())
    write_json(out/'target-binding-augmentation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1/data'))
    augment(p.parse_args().data)
