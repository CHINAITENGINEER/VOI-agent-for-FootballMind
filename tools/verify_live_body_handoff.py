"""Recompute one completed archived live run, without requesting sources/providers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run_dir, output):
    import numpy as np
    from football_research.acquisition import entity_evidence_updates as f
    from football_agent import live_source_assertions as source, live_claim_review_v2 as review
    from demo.live_f_consumption import consumption_audit, pipeline_summary
    from demo.live_g import utc

    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    run_path=run_dir/'run.json';initial_sha=sha(run_path);r=read(run_path)
    if r['state'] not in {'STOPPED','FAILED','CANCELLED','INTERRUPTED'}:
        raise ValueError('Archive must be terminal before reproducibility verification')
    assert sha(run_dir/'f.json')==r['model_bindings']['f']
    assert sha(run_dir/'g.json')==r['model_bindings']['g']
    model=f.load(run_dir/'f.json');results=[];used=[];body_files=[]
    for step in r['steps']:
        record_path=run_dir/step['semantic_record'];record=read(record_path);public=record['input']
        assert public['text']==step['body'] and public['input_id']==step['input_id']
        original=record['response']['claims']
        bound,binding=source.bind_response(public,source.provider_input(public),{'schema':source.VERSION,'claims':original})
        assert bound==record['response']
        admission=record['claim_admission']
        if original:
            recalculated,_=review.bind_response(public,original,admission['input'],admission['response'])
        else:
            recalculated,_=review.skip_review(public,original)
        assert recalculated==admission['review']
        admitted=[original[i] for i in recalculated['retained_indices']]
        assert admitted==step['claims']
        def row(ids):
            return dict(physical_match_id=r['fixture']['physical_match_id'],decision_at=step['executed_at'],
                prior=r['prior'],used_input_ids=list(ids),evidence_events=[e for e in r['evidence_events'] if e['input_id'] in ids])
        previous=row(used);used.append(step['input_id']);following=row(used)
        before=f.predict(model,previous).tolist();after=f.predict(model,following).tolist()
        np.testing.assert_allclose(before,step['prediction_before'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(after,step['prediction_after'],rtol=0,atol=1e-12)
        audit=consumption_audit(model,previous,following,step['input_id'],step['body'],admitted,before,after)
        assert audit==step['f_consumption']
        docs=[d for d in r['documents'] if d.get('raw_sha256')==step['raw_sha256'] and d.get('text')==step['body']]
        assert len(docs)==1
        doc=docs[0]
        assert utc(doc['observed_at'])<utc(step['executed_at'])<utc(r['prediction_cutoff'])<utc(r['fixture']['kickoff'])
        clocks=[{k:a.get(k) for k in ['kind','status','requested_at','completed_at']} for a in r['semantic_attempts'] if a['input_id']==step['input_id']]
        results.append(dict(step=step['step'],title=step['title'],url=step['url'],body_characters=len(step['body']),
            input_id=step['input_id'],action_id=step['action_id'],body_sha256=audit['body_sha256'],raw_sha256=step['raw_sha256'],
            body_consumed=True,original_claims=len(original),retained_claims=len(admitted),modeled_claims=audit['modeled_claims'],
            nonzero_feature_delta=audit['new_nonzero_features'],prediction_before=before,prediction_after=after,
            probability_l1=audit['probability_l1_change'],no_change_reason=audit['reason'],executed_at=step['executed_at'],
            provider_attempts=clocks,source_published_at=doc.get('source_published_at'),response_at=doc['observed_at'],
            current_run_model_family='legacy_title_state_g_and_claim_entity_f',new_qwen_model_used=False,
            result=r.get('outcome'),gain=r.get('gain'),source_record=str(record_path),source_record_sha256=sha(record_path)))
        body_files.append((f"body-{step['step']:02d}.txt",step['body']))
    assert used==r['used_input_ids'] and sha(run_path)==initial_sha
    output.mkdir(parents=True,exist_ok=False)
    payload=dict(schema='footballmind-live-body-handoff-verification-v1',status='PASS',run_id=r['run_id'],
        source_run=str(run_path),source_run_sha256=initial_sha,original_state=r['state'],original_reason=r.get('reason'),
        original_counts=r['counts'],body_pipeline=pipeline_summary(r),models=r['model_bindings'],steps=results,
        replay_source_http=0,replay_provider_http=0,training=0,old_run_unchanged=True,
        claim_check_scope='exact source binding plus replay of saved review; no independent truth assertion',
        conclusion='The screenshot captured pending provider review; completed archive has committed f updates.')
    (output/'verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,body in body_files:(output/name).write_text(body,encoding='utf-8')
    lines=['GAIS–IFK Göteborg 正文进入 f 的实际复算',
        '截图显示的是等待语义复核期间的中间状态；最终运行已完成，不能把当时0次更新理解为永久没有送入f。',
        f"run={r['run_id']}；状态={r['state']}；停止原因={r.get('reason')}。",
        '本检查只读原记录并用其冻结f重算，没有重新请求网页、调用DeepSeek或改写原记录。',
        '当前在线使用旧标题/状态g与实体claim f；不是新Qwen全文筛选研究模型。',
        '每篇逐项验证：原文及input_id一致；原始claim精确引文校验；保存的复核结果复算；采用集合；f前后概率；时钟先于cutoff。','']
    for s in results:
        lines += [f"第{s['step']}篇：{s['title']}",f"正文{s['body_characters']}字符；原始陈述{s['original_claims']}，复核保留{s['retained_claims']}，f可建模{s['modeled_claims']}；新增非零特征{s['nonzero_feature_delta']}。",
            f"before={s['prediction_before']}",f"after ={s['prediction_after']}",f"概率L1={s['probability_l1']}；原因={s['no_change_reason']}。",
            f"f完成时间UTC={s['executed_at']}；input_id={s['input_id']}。",'']
    lines += ['概率变化表示模型响应，不表示正确率改善；本轮尚未结算，outcome/gain保持未知。',
        '无可建模特征的正文也保留真实消费记录；不得为了演示强迫概率变化。',
        '本次改动只细分待处理/复核/f执行阶段，未修改模型参数或放松语义准入。']
    (output/'readable-report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return {'status':'PASS','updates':len(results),'output':str(output),'max_probability_difference':0.0}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    def denied(*a,**kw):raise RuntimeError('Network prohibited during archived verification')
    with patch.object(socket.socket,'connect',denied),patch.object(socket,'create_connection',denied):
        print(json.dumps(verify(args.run,args.output),ensure_ascii=False))
