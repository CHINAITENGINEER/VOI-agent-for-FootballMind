/* Pure projection of saved provider attempts. No inference or request side effects. */
(function(root){
 'use strict';
 function isPostread(record){return record?.pipeline_mode==='postread_fulltext_v1'||record?.schema_mode==='postread_fulltext';}
 function candidateKey(row){return row?.action_id||row?.candidate_id||row?.document_id||null;}
 function sameCandidate(row,id){return !!id&&[row?.action_id,row?.candidate_id,row?.document_id,row?.input_id,row?.text_sha256].includes(id);}
 function postreadCounts(record){
  const pool=record?.candidate_pool||[],steps=(record?.steps||[]).filter(s=>s.f_executed===true),scored=new Set();
  for(const d of record?.decisions||[])for(const c of d.candidates||d.candidate_scores||[]){const id=candidateKey(c);if(id&&[c.score,c.predicted_gain,c.immediate_gain].some(Number.isFinite))scored.add(id);}
  return {body_documents:pool.filter(c=>String(c.body||c.text||'').trim()).length,
   encoded:pool.filter(c=>c.status==='encoded').length,g_scored:scored.size,f_adopted:steps.length,
   body_failed:pool.filter(c=>c.status==='body_failed').length,encoding_failed:pool.filter(c=>c.status==='encoding_failed').length};
 }
 function postreadProgress(record,actionId,documentId,now){
  const terminal=['STOPPED','FAILED','CANCELLED','INTERRUPTED'].includes(record?.state);
  const committed=(record?.steps||[]).find(s=>s.f_executed===true&&(sameCandidate(s,actionId)||sameCandidate(s,documentId)));
  if(committed)return {status:'committed',message:'已编码正文由冻结 f 实际采用，预测已保存；此路径直接使用正文语义向量，不需要 claim 抽取。',attempts:[]};
  const row=(record?.candidate_pool||[]).find(c=>sameCandidate(c,actionId)||sameCandidate(c,documentId));
  if(!row)return {status:'unknown',message:'尚未保存可精确关联的正文与编码记录；不能确认已进入 f。',attempts:[]};
  const start=Date.parse(row.encoding_started_at),end=Date.parse(row.encoding_completed_at),clock=Number.isFinite(now)?now:Date.now();
  const pending=['encoding','encoding_pending'].includes(row.status);
  const attempts=Number.isFinite(start)?[{kind:'local_encoding',label:'本地 Qwen 编码',status:row.status||'UNKNOWN',input_id:row.text_sha256||row.document_id,
   elapsed_seconds:Number.isFinite(end)?Math.max(0,(end-start)/1000):pending&&!terminal?Math.max(0,(clock-start)/1000):null,
   requested_at:row.encoding_started_at,completed_at:row.encoding_completed_at,error_class:row.error_class||null}]:[];
  if(row.status==='body_failed')return {status:'failed',message:'正文获取或投影失败，未进入编码与 f；没有 after 或收益。',attempts};
  if(row.status==='encoding_failed')return {status:'failed',message:'本地正文编码失败，未进入 f；这不是 claim 抽取失败，也没有 after 或收益。',attempts};
  if(row.status==='encoding_budget_skipped')return {status:'budget_skipped',message:'全文已保存，超出本轮 CPU 编码预算；未送 g / f，不表示没有价值。没有截断全文，也没有生成 after 或收益。',attempts};
  if(row.status==='duplicate_body')return {status:'duplicate',message:'正文与候选池中已有全文完全相同，已去重；本记录没有重复编码或另一次 f 采用。',attempts};
  if(pending)return {status:terminal?'interrupted':'pending',message:terminal?'运行已结束，正文编码未记录完成；f 未确认执行。':'正文已取得，正在本地 Qwen 编码；f 尚未采用，等待实际编码结果。',attempts};
  const binding=record.active_body||{};
  if(!terminal&&binding.status==='f_running'&&(sameCandidate(binding,actionId)||sameCandidate(binding,documentId)))return {status:'f_running',message:'正文已编码并被本轮策略选中，冻结 f 正在计算；after 尚未保存。',attempts};
  if(row.status==='encoded')return {status:'encoded',message:'正文已编码，可用于 g 评分；没有本次 f 采用记录，不能生成 after 或收益。',attempts};
  return {status:'pending',message:'正文处于获取或待编码阶段；尚未确认编码完成及 f 采用。',attempts};
 }
 function bodyProgress(record,actionId,documentId,now){
  if(isPostread(record))return postreadProgress(record,actionId,documentId,now);
  record=record||{};const terminal=['STOPPED','FAILED','CANCELLED','INTERRUPTED'].includes(record.state);
  const binding=record.active_body||{},steps=record.steps||[];
  const committed=steps.find(s=>s.f_executed===true&&s.action_id===actionId);
  if(committed)return {status:'committed',message:'正文已交给 f 并保存预测。',attempts:[]};
  const inputId=binding.action_id===actionId?binding.input_id:null;
  const attempts=(record.semantic_attempts||[]).filter(a=>(a.action_id&&a.action_id===actionId)||(documentId&&a.document_id===documentId)||(inputId&&a.input_id===inputId));
  const rejection=(record.body_rejections||[]).find(r=>r.action_id===actionId);
  const latest=attempts.at(-1);
  const stages={initial:'抽取陈述',validation_repair:'修复语义协议',claim_review:'复核来源陈述'};
  const clock=Number.isFinite(now)?now:Date.now();
  const details=attempts.map(a=>{const start=Date.parse(a.requested_at),end=Date.parse(a.completed_at);return {
   kind:a.kind,label:stages[a.kind]||'阶段未知',status:a.status||'UNKNOWN',input_id:a.input_id,
   elapsed_seconds:Number.isFinite(start)&&((a.status==='PENDING'&&!terminal)||Number.isFinite(end))?Math.max(0,Math.floor(((a.status==='PENDING'&&!terminal)?clock:end)-start)/1000):null,
   requested_at:a.requested_at,completed_at:a.completed_at,error_class:a.error_class||null};});
  if(rejection)return {status:'rejected',message:'本候选语义校验未通过，f 未执行；未生成 after 或收益。',attempts:details};
  if(!terminal&&latest?.status==='PENDING')return {status:'pending',message:'正文已取得，正在'+(stages[latest.kind]||'处理语义请求')+'；f 尚未执行，等待实际返回。',attempts:details};
  if(!terminal&&binding.action_id===actionId&&binding.status==='f_running')return {status:'f_running',message:'语义验证已完成，f 正在计算；after 尚未保存。',attempts:details};
  if(!terminal&&binding.action_id===actionId&&!['f_committed','semantic_validation_failed'].includes(binding.status))return {status:'pending',message:latest?.status==='FAILED'?'本次语义请求失败，运行仍在处理中；尚未记录后续处理结果，f 未执行。':'正文已取得，正在等待下一处理阶段；f 尚未执行。',attempts:details};
  if(terminal&&latest?.status==='FAILED')return {status:'failed',message:'语义请求已失败，本轮已结束；f 未执行，after 未生成。',attempts:details};
  if(terminal&&latest?.status==='PENDING')return {status:'interrupted',message:'本轮已结束，但这次语义请求没有保存完成状态；结果未知，不能认定 f 已执行。',attempts:details};
  return {status:'unknown',message:'未保存可关联的完整语义阶段或 f 更新记录；状态未知。',attempts:details};
 }
 const api={bodyProgress,isPostread,postreadCounts,candidateKey,sameCandidate};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
 else root.FootballMindProgress=api;
})(typeof globalThis!=='undefined'?globalThis:this);
