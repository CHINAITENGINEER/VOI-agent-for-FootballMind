from pathlib import Path
from collections import Counter, defaultdict
import json
ROOT=Path(__file__).resolve().parents[1]
trans=json.loads((ROOT/'var/reviews/fm-historical-transition-g-training-20261014-v2/transitions.json').read_text(encoding='utf8'))
states=json.loads((ROOT/'var/reviews/fm-historical-transition-g-training-20261014-v2/states.json').read_text(encoding='utf8'))
events=json.loads((ROOT/'var/reviews/fm-entity-updates-f-training-20261014-v1/event-training-rows.json').read_text(encoding='utf8'))
bykey={(r['physical_match_id'],tuple(sorted(r['used_input_ids']))):r for r in events}
sid_to_key={sid:(s['physical_match_id'],tuple(sorted(s['used_input_ids']))) for sid,s in states.items()}
def claims(t):
 key=sid_to_key[t['next_state_id']]; row=bykey.get(key,{})
 out=[]
 for e in row.get('evidence_events',[]):
  c=e.get('claim',{}); out.append({'input_id':e.get('input_id'),'url':e.get('document_url'),'team':c.get('team'),'entity':c.get('entity'),'kind':c.get('kind'),'value':c.get('value'),'scope':c.get('scope'),'statement':c.get('statement'),'quote':(c.get('quotes') or [''])[0]})
 return out
pos=[]; neg=[]; facet=Counter(); match=defaultdict(lambda:Counter())
for t in trans:
 cs=claims(t)
 item={k:t.get(k) for k in ('physical_match_id','action_id','body_input_id','body_url','signed_gain','loss_before','loss_after','effect_class','g_split')}
 item['claims']=cs
 if t['signed_gain']>1e-12: pos.append(item); group='positive'
 elif t['signed_gain']<-1e-12: neg.append(item); group='negative'
 else: group='zero'
 for c in cs: facet[(group,c['kind'],c['value'],c['scope'])]+=1
 match[t['physical_match_id']][group]+=1
pos.sort(key=lambda x:x['signed_gain'],reverse=True)
neg.sort(key=lambda x:x['signed_gain'])
out=ROOT/'var/reviews/fm-g-positive-negative-panel-20261015-v1'; out.mkdir(parents=True,exist_ok=False)
summary={'schema':'fm-f-positive-negative-panel-v1','source_transitions':len(trans),'positive_transitions':len(pos),'negative_transitions':len(neg),'zero_transitions':len(trans)-len(pos)-len(neg),'top_positive':pos[:20],'top_negative':neg[:20],'facet_counts':{'%s|%s|%s|%s'%k:v for k,v in facet.items()},'match_counts':{k:dict(v) for k,v in match.items()},'use':'diagnostic sample selection for next g round; no rows promoted or runtime changed','leakage_note':'outcomes joined after fixed-f predictions; not blind to historical results'}
(out/'panel.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['FootballMind f 正负收益对照面板（离线）','',f"总转移：{len(trans)}，正收益：{len(pos)}，负收益：{len(neg)}，零变化：{summary['zero_transitions']}",'','该面板用于选择下一轮人工复核样本，不改变 f，也不直接训练或启用 g。','']
for title,arr in [('正收益最高的5个转移',pos[:5]),('负收益最大的5个转移',neg[:5])]:
 lines.append(title)
 for x in arr:
  lines.append(f"- {x['physical_match_id']} | signed_gain={x['signed_gain']:.9f} | {x['body_url']}")
  for c in x['claims'][:3]: lines.append(f"  {c['kind']} / {c['value']} / {c['scope']} / {c['statement']}")
 lines.append('')
(out/'readable-summary.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'positive':len(pos),'negative':len(neg),'zero':summary['zero_transitions'],'out':str(out)},ensure_ascii=False,indent=2))
