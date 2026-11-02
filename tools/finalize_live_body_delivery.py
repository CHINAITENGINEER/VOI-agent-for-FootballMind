"""Read-only final evidence export: no fitting, source requests or provider calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import py_compile
import re
import socket
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from football_research.acquisition import entity_evidence_updates as fm
from football_agent.live_source_assertions import canonical_body
from football_agent import live_claim_review as review_v1
from football_research.acquisition.semantic_f import EXTRACTION_VERSION,validate_response
from demo.live_f_consumption import consumption_audit

TASK=ROOT/'var/work/fm-live-body-chain-repair-20261025'
RUNS=ROOT/'var/parallel/agent/round2/runs/live-g/runs'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def utc(value):return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(timezone.utc)
def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def export(run_id,out):
    folder=RUNS/run_id;run=read(folder/'run.json');source_hash=digest(folder/'run.json')
    assert run['state'] in {'STOPPED','FAILED','CANCELLED','INTERRUPTED'}
    f=fm.load(folder/'f.json');g_path=ROOT/'var/reviews/fm-g-policy-return-continuation-20261014-v1/g-policy-return.json'
    f_path=ROOT/'var/reviews/fm-historical-transition-g-training-20261014-v2/f-labeler.json'
    assert digest(folder/'f.json')==digest(f_path)==run['model_bindings']['f']
    assert digest(g_path)==run['model_bindings']['g']
    checked_sources=[]
    for attempt in run['attempts']:
        returned=attempt.get('response')
        if not returned:continue
        obs=returned['observation'];p=folder/'sources'/obs['payload_path']
        payload=gzip.decompress(p.read_bytes())
        assert sha256(payload).hexdigest()==obs['content_hash']
        checked_sources.append({'request_id':attempt['request_id'],'purpose':attempt['purpose'],
            'observation_id':obs['id'],'raw_path':str(p),'raw_sha256':obs['content_hash'],
            'http_status':obs['http_status'],'request_at':attempt['requested_at'],
            'response_observed_at':obs['observed_at'],'handoff_at':attempt.get('handoff_at'),
            'pre_cutoff':utc(obs['observed_at'])<utc(run['prediction_cutoff'])})
    steps=[];used=[]
    for step in run['steps']:
        sid=step['input_id'];semantic=read(folder/step['semantic_record'])
        canonical=canonical_body(semantic['input'])
        base_claims=[{k:v for k,v in c.items() if k!='concept'} for c in semantic['response']['claims']]
        validate_response(canonical,dict(schema=EXTRACTION_VERSION,input_id=sid,claims=base_claims))
        assert canonical['input_id']==sid and canonical['text']==step['body']
        admitted=semantic.get('claim_admission',{}).get('review',{}).get('retained_indices')
        extracted=semantic['response']['claims']
        assert (extracted if admitted is None else [extracted[i] for i in admitted])==step['claims']
        admission=semantic.get('claim_admission')
        if admission is not None:
            if admission['review']['schema']==review_v1.BOUND_VERSION:
                review_protocol=review_v1
            else:
                from football_agent import live_claim_review_v2 as review_protocol
            if extracted:
                check_review,check_binding=review_protocol.bind_response(semantic['input'],extracted,
                    admission['input'],admission['response'])
            else:
                check_review,check_binding=review_protocol.skip_review(semantic['input'],extracted)
            assert check_review==admission['review']==step['claim_review']
            assert check_binding==admission['binding']
        source=next(a for a in run['attempts'] if a['purpose']=='g_selected_body' and a['action'].get('url')==step['url'])
        doc=source['response']['document']
        assert doc['text']==step['body'] and doc['raw_sha256']==step['raw_sha256']
        before=dict(physical_match_id=run['fixture']['physical_match_id'],decision_at=step['executed_at'],
            prior=run['prior'],used_input_ids=list(used),
            evidence_events=[e for e in run['evidence_events'] if e['input_id'] in used])
        pb=fm.predict(f,before).tolist();used.append(sid)
        after={**before,'used_input_ids':list(used),
            'evidence_events':[e for e in run['evidence_events'] if e['input_id'] in used]}
        pa=fm.predict(f,after).tolist()
        np.testing.assert_allclose(pb,step['prediction_before'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(pa,step['prediction_after'],rtol=0,atol=1e-12)
        audit=consumption_audit(f,before,after,sid,step['body'],step['claims'],pb,pa)
        assert audit==step['f_consumption']
        steps.append({'step':step['step'],'action_id':step['action_id'],'title':step['title'],'url':step['url'],
            'input_id':sid,'body_sha256':audit['body_sha256'],'raw_sha256':step['raw_sha256'],
            'claim_review':step.get('claim_review'),'extracted_claims_count':len(extracted),
            'body_characters':len(step['body']),'claims':step['claims'],
            'claim_count':len(step['claims']),'modeled_claims':audit['modeled_claims'],
            'nonzero_features':audit['nonzero_model_features'],'new_features':audit['new_nonzero_features'],
            'reason':audit['reason'],'prediction_before':pb,'prediction_after':pa,
            'probability_l1':audit['probability_l1_change'],'market_l1':audit['market_probability_l1_difference'],
            'source_published_at':semantic['input'].get('source_context',{}).get('source_published_at'),
            'source_observed_at':doc.get('observed_at'),'handoff_at':source.get('handoff_at'),
            'f_executed_at':step['executed_at'],'source_pre_cutoff':utc(doc['observed_at'])<utc(run['prediction_cutoff']),
            'f_pre_cutoff':utc(step['executed_at'])<utc(run['prediction_cutoff']),
            'first_public_availability':'UNKNOWN','f_model_sha256':run['model_bindings']['f'],
            'body_consumed':True,'replay_exact':True,'training_performed':False})
    assert used==run['used_input_ids'] and all(s['source_pre_cutoff'] and s['f_pre_cutoff'] for s in steps)
    assert len(run['semantic_attempts'])==run['counts']['provider']
    receipts=[]
    for a in run['semantic_attempts']:
        receipt=(a.get('provider_artifact') or {}).get('receipt') or {}
        receipts.append({'input_id':a['input_id'],'kind':a['kind'],'status':a['status'],
            'requested_at':a['requested_at'],'completed_at':a['completed_at'],
            'provider_real_call':receipt.get('real_provider_call'),'requests':receipt.get('requests'),
            'model_requested':receipt.get('model_requested'),'response_model':receipt.get('response_model'),
            'usage':receipt.get('usage'),'actual_money':receipt.get('actual_money'),
            'receipt_path':str(Path(a['provider_artifact']['folder'])/'receipt.json') if a.get('provider_artifact') else None})
        assert read(receipts[-1]['receipt_path'])==receipt
    tokens=sum(u.get('total_tokens',0) for r in receipts for u in (r.get('usage') or []))
    assert all(r['provider_real_call'] is True and r['requests']==1 for r in receipts)
    assert digest(folder/'run.json')==source_hash
    save(out/'step-evidence.json',steps);save(out/'source-and-provider-receipts.json',{'sources':checked_sources,'semantics':receipts})
    fields=['step','title','url','input_id','body_characters','claim_count','modeled_claims','nonzero_features',
        'new_features','reason','probability_l1','market_l1','source_published_at','source_observed_at',
        'f_executed_at','source_pre_cutoff','f_pre_cutoff','body_consumed','replay_exact','f_model_sha256']
    write_csv(out/'per-body-f-response.csv',[{k:s[k] for k in fields} for s in steps])
    save(out/'accepted-body-records.json',[{'input_id':s['input_id'],'title':s['title'],'url':s['url'],
        'body':run['steps'][i]['body'],'claims':s['claims'],'body_sha256':s['body_sha256']} for i,s in enumerate(steps)])
    run_history=[]
    notes={
        'lg-931eb30353114f1db6da2ebdcdd36a30':'裸根 URL 授权规范化故障；已修复',
        'lg-496ae94f6b9c4e7fa8b3a87ddcc4d60a':'旧卡/同分选择诊断；不强迫中性正文改变概率',
        'lg-5e9eb6936d3d4cc7ab135779911c5d2e':'真实正文中性样本；0 modeled 合法保留市场',
        'lg-d88769813469435c869259fd3a2570d1':'publication_raw 本地类型失败；该旧记录 provider 计数不代表真实 HTTP，旧 receipt 串用已修',
        'lg-8b2cb8d051c64e6586af484f2c8f8f37':'已有一次真实 f 更新后后续引句校验失败；原记录保留',
        'lg-35b5a306c62c47078b02c1058a4ea6aa':'中间运行存在1条已确认训练概念误判；不是语义验收通过',
        'lg-74d1369b68d2425da71dbc530970e6d9':'初次 kind/concept 不合法，repair 32>24；正确拒绝且f更新0',
        'lg-a807f42ea5284385a1cb98474e68d6fe':'5篇真实进入f，但人工复核发现资格/名单、训练否定及同名歧义；不是最终语义通过',
        'lg-e33662b5d3c748c5ab508f39dda5aab3':'V1单一verdict复核路径：9实际modeled有源支持，但未建模must_win仍假阳性；随后真实旧错误控制证明V1不足',
    }
    for rid in list(notes)+([run_id] if run_id not in notes else []):
        p=RUNS/rid/'run.json';r=read(p)
        run_history.append({'run_id':rid,'physical_match_id':r.get('fixture',{}).get('physical_match_id'),
            'state':r['state'],'reason':r['reason'],'source_count':r['counts']['source'],
            'provider_counter':r['counts']['provider'],'body_attempts':r['counts']['body_attempts'],
            'f_updates':len(r['steps']),'market_l1':r.get('body_pipeline',{}).get('market_difference_l1'),
            'note':notes.get(rid,'最终新正常API验收；逐条语义复核另附'),'run_sha256':digest(p),'run_path':str(p)})
    write_csv(out/'all-repair-runs.csv',run_history)
    junit=ET.parse(TASK/'verification-tests-final.xml').getroot()
    cases=junit.findall('.//testcase');failed=junit.findall('.//failure')+junit.findall('.//error')
    checks={'run_id':run_id,'state':run['state'],'stop_reason':run['reason'],'fixture':run['fixture'],
        'prediction_cutoff':run['prediction_cutoff'],'market':run['prior'],'initial_f':run['before'],'final_f':run['current'],
        'market_l1':float(np.abs(np.array(run['current'])-run['prior']).sum()),'counts':run['counts'],
        'body_pipeline':run['body_pipeline'],'provider_total_tokens':tokens,'actual_money':None,
        'f_model_sha256':digest(f_path),'g_model_sha256':digest(g_path),'frozen_models_unchanged':True,
        'committed_steps_independently_replayed':len(steps),'raw_sources_checked':len(checked_sources),
        'all_exact_source_quotes_valid':True,'same_f_before_after':True,'all_steps_before_cutoff':True,
        'new_http_in_this_export':0,'provider_calls_in_this_export':0,'training_performed':False,
        'run_unchanged_by_export':True,'original_run_sha256':source_hash,'runtime_bindings':run['runtime_bindings'],
        'tests':{'testcases':len(cases),'failures':len(failed),'junit':str(TASK/'verification-tests-final.xml')},
        'page_url':'http://127.0.0.1:8766/live.html?run='+run_id,
        'semantic_truth_review':'See separate direct-full-text review; exact quotes alone do not prove correct semantics.',
        'not_established':['forecast_accuracy_improvement','per_match_nonzero_guarantee','all_leagues_forecasting_calibration',
            'all_startable_leagues_live_body_executed','new_empirical_g_training_labels']}
    assert checks['market_l1']>1e-12 and not failed
    save(out/'acceptance-report.json',checks)
    return run,checks,steps


def plot(out,run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');x=np.arange(3);w=.34
    for shift,p,label,color in [(-w/2,run['prior'],'市场基线','#9baaa0'),(w/2,run['current'],'正文进入固定 f 后','#217b55')]:
        bars=ax.bar(x+shift,p,w,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v:.4%}' for v in p],padding=4,fontsize=11)
    ax.set_xticks(x,['主胜','平局','客胜']);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0,max(run['prior']+run['current'])*1.25);ax.set_ylabel('模型输出概率');ax.legend()
    ax.set_title('Liverpool – Brighton：真实在线正文更新\n概率变化表示模型响应，尚无赛果改善结论')
    ax.spines[['top','right']].set_visible(False);fig.savefig(out/'market-vs-f.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--output-name',default='delivery-v2');args=p.parse_args()
    if not re.fullmatch(r'delivery-[a-zA-Z0-9_-]+',args.output_name):raise ValueError('Use a plain delivery directory name')
    out=TASK/args.output_name;out.mkdir(exist_ok=False)
    # This exporter has no reason to reach a source or a model provider.
    def offline(*a,**kw):raise RuntimeError('Offline evidence export: network forbidden')
    socket.socket.connect=socket.create_connection=socket.getaddrinfo=offline
    run,checks,steps=export(args.run_id,out);plot(out,run)
    from demo.live_g import RUNTIME_CODE_FILES
    files=list(RUNTIME_CODE_FILES)+['demo/static/live.js','demo/static/live.html',
        'tools/restart_footballmind_demo.py','tools/run_live_body_acceptance.py','tools/finalize_live_body_delivery.py',
        'tools/capture_live_body_delivery.py','tools/package_live_body_delivery.py']
    save(out/'code-snapshot-sha256.json',{name:digest(ROOT/name) for name in files})
    compiled=[]
    for name in files:
        if name.endswith('.py'):
            target=out/'compile'/(name.replace('/','__')+'c');target.parent.mkdir(exist_ok=True)
            py_compile.compile(str(ROOT/name),cfile=str(target),doraise=True);compiled.append(name)
    save(out/'compile-check.json',{'passed':compiled,'count':len(compiled),'original_source_pycache_modified':False})
    text=['在线正文 → 语义 → 固定 f 修复交付','',
        '最终真实运行：'+checks['run_id'],'页面：'+checks['page_url'],
        '比赛：Liverpool – Brighton；开球 '+run['fixture']['kickoff'],
        '市场：[主胜/平/客胜] '+json.dumps(run['prior']),
        '最终 f：[主胜/平/客胜] '+json.dumps(run['current']),
        '市场 L1 差异：'+str(checks['market_l1']),
        '真实来源/语义/正文尝试：'+json.dumps(run['counts'],ensure_ascii=False),
        '逐正文已采用/特征/概率/时钟/离线精确复算见 per-body-f-response.csv、step-evidence.json。',
        '正文与 claims 见 accepted-body-records.json；本导出没有请求来源或 provider。','',
        '截图原问题：Fortuna–Excelsior 的5是来源请求，不是5篇已进入f的正文。该旧运行无候选、无语义、无f正文步骤。',
        '修复：服务器源码版本检测；裸根URL一致规范化；Vue/JSON-LD/可见锚文本目录解析；国家/男女/青年同名隔离；',
        '来源明确日历日期过滤及同分并列处理；显式concept接入既有f分支；正文input_id/特征/概率消费审计；',
        '精确源引句校验、最多一次span-ID修复，并对非空合法陈述额外做一次完整原文语义复核；仅保留原claims中supported索引。',
        '原抽取response/binding保持，复核不新增/改写fact或quote。失败候选不伪造after/gain，保留已有有效f。',
        '语义预算15次对应最多5篇各初次+一次协议修复+非空claims来源复核；来源10次、最长15分钟、赛前cutoff均保留。','',
        '重要复核：前一轮lg-35b5...有一条“指导年轻后卫”被误标训练参与并影响概率；原运行保留，不列为语义验收通过。',
        '已收紧抽取prompt并重新运行；后续lg-74d初次kind/concept错误、修复32条超24上限，也被完整拒绝。',
        '所有中间运行见all-repair-runs.csv，不把失败或存在语义错误的运行只因出现概率差异就算验收成功。',
        '最终语义逐条审阅见 trace 下对应新run审阅文件，勿仅凭概率变化判为正确。','',
        '范围：全部可启动来源条目共用这一f适配路径；快照范围/逐联赛数量见 final-coverage。',
        '通用契约检查不等于每个联赛、每场均已实测取得有用正文，更不证明跨联赛预测准确。',
        '五大联赛、苏超原始1X2路径通过；中甲已新增3场可见真实赛程，但完整90分钟1X2仍缺，保持不可启动。',
        '中超/K1当前窗口为空与渠道不支持分开；来源探测和缺口证据见 coverage/coverage-next。','',
        '模型：'+checks['f_model_sha256']+' (f)，'+checks['g_model_sha256']+' (g)。',
        '固定模型权重未改；每次用户运行执行predict，不执行fit。当前在线g是原配套标题/状态g，Qwen全文g仍是独立研究模型。',
        '中性、重复或当前f无法建模的正文允许概率不变；不添加人为扰动来保证每场偏离市场。',
        '本轮未读取赛果、没有loss/gain结算、未生成训练行；概率变化不代表准确率提高。','',
        '测试：'+str(checks['tests']['testcases'])+'项通过，失败'+str(checks['tests']['failures'])+'；py_compile '+str(len(compiled))+'项通过。',
        '较早一次测试有1项Windows磁盘负载下20秒归档复制等待超时，单独复核通过后测试等待放宽60秒；未放宽业务cutoff或语义校验。']
    (out/'readable-report.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    save(out/'manifest.json',{'schema':'footballmind-live-body-delivery-manifest-v1',
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'files':{str(f.relative_to(out)).replace('\\','/'):digest(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='manifest.json'}})
    print(json.dumps({'output':str(out),'run_id':run['run_id'],'market':run['prior'],'final_f':run['current'],
        'f_updates':len(steps),'market_l1':checks['market_l1'],'tests':checks['tests'],'py_compile':len(compiled)},ensure_ascii=False))


if __name__=='__main__':main()
