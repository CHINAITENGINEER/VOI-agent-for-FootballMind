'use strict';

const $ = id => document.getElementById(id);
const state = {matches:[], selected:null, run:null, meta:null, search:'', league:'all', sort:'asc', tab:'overview', evidenceFilter:'all', running:false, runCache:new Map(), request:0};
const terminalStates = new Set(['STOPPED','DEFERRED','COMPLETED','FAILED','BUDGET_EXHAUSTED','DEADLINE_EXPIRED']);
const typeLabels = {initialized:'初始化',transition_started:'状态推进开始',transition_completed:'状态推进完成',tool_started:'工具调用开始',tool_returned:'工具调用返回',match_selected:'比赛身份确定',evidence_arrived:'来源返回',evidence_ready:'证据就绪',evidence_duplicate:'重复证据',features_ready:'特征进入 f',prediction_updated:'概率更新',policy_decision:'Agent 选择动作',stopped:'停止',deferred:'暂缓',quality_issue:'质量问题',failed:'失败'};
const toolLabels = {query_matches:'查询比赛',read_evidence:'读取来源',get_features:'构造 f 特征',predict:'调用预测模型',refresh_information:'刷新信息'};

function esc(value){return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));}
function text(value, fallback='—'){return value === null || value === undefined || value === '' ? fallback : String(value);}
function finite(value){return typeof value === 'number' && Number.isFinite(value);}
function pct(value){return finite(value) ? `${Math.round(value * 100)}%` : '—';}
function pctExact(value){return finite(value) ? `${(value * 100).toFixed(1)}%` : '—';}
function fmtTime(value, withDate=true){if(!value) return '时间未知'; const d=new Date(value); if(Number.isNaN(d.valueOf())) return String(value); return d.toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false,year:withDate?'numeric':undefined,month:withDate?'2-digit':undefined,day:withDate?'2-digit':undefined,hour:'2-digit',minute:'2-digit'});}
function fmtClock(value){if(!value) return '—';const d=new Date(value);return Number.isNaN(d.valueOf())?'—':d.toLocaleTimeString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false,hour:'2-digit',minute:'2-digit'});}
function lastEventAt(run){return run?.events?.length ? run.events.at(-1).at : run?.created_at;}
function setNotice(message=''){const el=$('notice');el.hidden=!message;el.textContent=message;}
function setConnection(label, ok=true){$('connection').textContent=label;$('connection-dot').classList.toggle('offline',!ok);}
function statusText(status){return ({scheduled:'未开赛',unknown:'状态未知',finished:'已结束',live:'进行中'})[status] || text(status);}
function eventStatus(type){return ({tool_returned:'已返回',tool_started:'执行中',prediction_updated:'已产生结果',evidence_arrived:'已发现',evidence_duplicate:'重复',stopped:'已停止',failed:'失败'})[type] || '已记录';}
function eventTitle(event){const base=typeLabels[event.type] || text(event.type);const tool=event.data?.tool ? ` · ${toolLabels[event.data.tool] || event.data.tool}` : '';return base+tool;}
function jsonPreview(value){try{return JSON.stringify(value ?? {},null,2)}catch(_){return '{}'}}
function uiView(run){return run?.ui_view || run;}

function filteredMatches(){
  const q=state.search.trim().toLowerCase();
  return state.matches.filter(m => (state.league==='all' || m.competition===state.league) && (!q || `${m.home} ${m.away} ${m.competition}`.toLowerCase().includes(q))).sort((a,b)=>{
    const da=Date.parse(a.kickoff)||0, db=Date.parse(b.kickoff)||0; return state.sort==='asc' ? da-db : db-da;
  });
}

function renderMatchList(){
  if(state.workspace==='research')return renderResearchMatchList();
  const list=$('match-list');list.replaceChildren();const rows=filteredMatches();$('fixture-count').textContent=`${rows.length} 场`;$('match-empty').hidden=rows.length>0;
  for(const match of rows){
    const run=state.runCache.get(match.match_id);const prediction=run?.prediction || run?.last_prediction;const card=document.createElement('button');card.type='button';card.className='match-card'+(state.selected?.match_id===match.match_id?' active':'');card.dataset.matchId=match.match_id;card.setAttribute('aria-pressed',String(state.selected?.match_id===match.match_id));
    const gap=run && (!prediction || !run.features || (run.quality||[]).length) ? '<span class="mini-tag warning">有缺口</span>' : (run?'':'<span class="mini-tag">待研究</span>');
    card.innerHTML=`<div class="match-card-top"><span class="match-time">${esc(fmtClock(match.kickoff))}</span><span class="league-label">${esc(match.competition)}</span>${match.is_demo?'<span class="mini-tag demo">演示</span>':''}</div><div class="match-teams"><strong>${esc(match.home)}</strong><span>vs</span><strong>${esc(match.away)}</strong></div><div class="match-probs"><span>主 ${esc(pct(prediction?.home_win_probability))}</span><span>平 ${esc(pct(prediction?.draw_probability))}</span><span>客 ${esc(pct(prediction?.away_win_probability))}</span></div><div class="match-card-foot"><span class="mini-status ${run?'ready':''}"><i></i>${run? (terminalStates.has(String(run.state||'').toUpperCase())?'已分析':'分析中'):'未开始'}</span>${gap}<span class="updated-label">${esc(fmtTime(lastEventAt(run)||match.fixture_available_at,false))}</span></div>`;
    card.addEventListener('click',()=>selectMatch(match.match_id));list.append(card);
  }
}

function populateLeagues(){const select=$('league-filter');const current=state.league;const leagues=[...new Set(state.matches.map(m=>m.competition).filter(Boolean))].sort();select.replaceChildren(new Option('全部联赛','all'),...leagues.map(x=>new Option(x,x)));select.value=leagues.includes(current)?current:'all';state.league=select.value;}

function resetDetail(){
  $('detail-content').hidden=true;$('detail-error').hidden=true;$('detail-loading').hidden=false;$('detail-error-text').textContent='';$('prediction-breakdown').hidden=true;$('agent-status').className='run-status';$('agent-status').innerHTML='<i></i>未开始';$('timeline').replaceChildren();$('candidate-actions').replaceChildren();$('status-checks').replaceChildren();$('toggle-prediction').setAttribute('aria-expanded','false');$('agent-outcome').hidden=true;$('agent-status').dataset.state='INITIALIZED';$('agent-log-count').textContent='—';
}

async function selectMatch(matchId){
  if(state.workspace==='research')return selectResearchMatch(matchId);
  const match=state.matches.find(x=>x.match_id===matchId);if(!match)return;state.selected=match;state.request++;const ticket=state.request;renderMatchList();resetDetail();setNotice('');
  try{
    let run=state.runCache.get(match.match_id);
    if(!run){run=await FootballMindAdapter.startRun(match);state.runCache.set(match.match_id,run);if(ticket!==state.request)return;renderRun(run);}
    else renderRun(run);
    if(!match.is_demo && !terminalStates.has(String(run.state||'').toUpperCase())){
      state.running=true;renderAgentRunning();run=await FootballMindAdapter.runToCompletion(run);state.runCache.set(match.match_id,run);state.running=false;if(ticket!==state.request)return;renderRun(run);renderMatchList();
    }
  }catch(error){state.running=false;if(ticket!==state.request)return;$('detail-loading').hidden=true;$('detail-error').hidden=false;$('detail-error-text').textContent=error.message;setConnection('读取失败',false);}
}

function renderHero(run){
  const match=run?.match || state.selected || {};const pred=run?.prediction || run?.last_prediction;
  $('detail-loading').hidden=true;$('detail-error').hidden=true;$('detail-content').hidden=false;
  $('hero-league').textContent=text(match.competition,'联赛未知');$('hero-status').textContent=statusText(match.status);$('hero-kickoff').textContent=fmtTime(match.kickoff);$('home-name').textContent=text(match.home,'主队未命名');$('away-name').textContent=text(match.away,'客队未命名');$('home-id').textContent=text(match.home_team_id,'身份 ID 未返回');$('away-id').textContent=text(match.away_team_id,'身份 ID 未返回');$('home-badge').textContent=(match.home||'H').slice(0,1);$('away-badge').textContent=(match.away||'A').slice(0,1);$('match-id').textContent=text(match.match_id);$('hero-cutoff').textContent=fmtTime(run?.prediction_cutoff || FootballMindAdapter.DEFAULT_CUTOFF);$('hero-updated').textContent=fmtTime(lastEventAt(run));
  const identity=match.identity_quality?.home==='stable' && match.identity_quality?.away==='stable';$('identity-status').textContent=identity?'比赛身份已核验':'比赛身份仍需确认';$('identity-status').className=identity?'verified':'warning-text';
  if(match.is_demo || run?.is_demo){$('hero-status').textContent='演示数据';$('hero-status').className='status-chip demo-chip';}
}

function predictionEvents(run){const events=(run?.events||[]).filter(e=>e.type==='prediction_updated' && e.data?.after);const canonical=run?.ui_view?.prediction;if(!canonical)return events;if(!canonical.after)return events.filter(e=>e.data?.update_reason==='INITIAL_PREDICTION');return events;}
function renderProbability(run){
  const view=uiView(run);const canonical=run?.ui_view?.prediction;const prediction=canonical?.current || run?.prediction || run?.last_prediction;const events=predictionEvents(run);const first=canonical ? canonical.before : (events[0]?.data?.before || null);const after=canonical ? canonical.after : (events.at(-1)?.data?.after || null);const current=prediction || after;
  const values=[current?.home_win_probability,current?.draw_probability,current?.away_win_probability];
  const complete=values.every(finite);const maximum=complete?Math.max(...values):null;const tied=complete&&values.filter(v=>v===maximum).length>1;
  ['home','draw','away'].forEach((key,i)=>{
    const value=values[i];const leading=complete&&value===maximum;const item=$(`prob-${key}`).closest('.prob-item');
    $(`prob-${key}`).textContent=pct(value);$(`bar-${key}`).style.width=finite(value)?`${Math.max(0,Math.min(1,value))*100}%`:'0%';
    item.classList.toggle('is-leading',leading);const rank=item.querySelector('.prob-rank');rank.hidden=!leading;rank.textContent=tied?'并列最高':'最高项';
  });
  $('prediction-source').textContent=current?(run?.is_demo?'演示数据 · 非实时预测':`模型 ${text(current.model_version,'预测工具')}`):'暂无可核验结果';
  $('prediction-time').textContent=`数据截止 ${fmtTime(current?.prediction_cutoff || run?.prediction_cutoff)}`;
  $('support-count').textContent=`${(run?.evidence||[]).length} 条返回来源`;
  const fStatus=view?.f?.status;const fCalled=(run?.events||[]).some(e=>e.type==='features_ready' || e.type==='prediction_updated');const featureValid=Boolean(run?.features && Object.keys(run.features.missing||{}).length===0);const changed=canonical ? canonical.change_status==='changed' : (first && current ? values.map((v,i)=>v-first[['home_win_probability','draw_probability','away_win_probability'][i]]).some(v=>Math.abs(v)>1e-9):false);
  let explanation='等待 Agent 完成预测工具调用。';if(run?.is_demo) explanation='演示数据：没有调用真实 f 或 Agent runtime。';else if(canonical?.change_status==='not_applicable') explanation='本次运行没有经过已确认的额外信息与 f 更新，保持 prediction_after 为空。';else if(!current) explanation='当前运行没有返回可核验的预测；保持空状态。';else if(!first) explanation='没有独立 prediction_before，因此无法计算概率变化量。';else if(canonical?.change_status==='unknown') explanation='运行记录缺少完整的 f 消费链，无法确认 before / after 的业务变化。';else if(!changed) explanation='f 已重新生成 after，但概率与 before 相同。';else explanation='f 已基于准入特征重新计算概率；变化来自完整特征快照，不归因于单条来源。';
  $('prediction-explain').innerHTML=`<span class="explain-label">变化说明</span><p>${esc(explanation)}</p>`;
  $('before-value').textContent=first?`${pct(first.home_win_probability)} / ${pct(first.draw_probability)} / ${pct(first.away_win_probability)}`:'未返回';$('after-value').textContent=after?`${pct(after.home_win_probability)} / ${pct(after.draw_probability)} / ${pct(after.away_win_probability)}`:'未返回';$('delta-value').textContent=canonical?.delta?`${pct(canonical.delta.home)} / ${pct(canonical.delta.draw)} / ${pct(canonical.delta.away)}`:(first&&after?`${pct(after.home_win_probability-first.home_win_probability)} / ${pct(after.draw_probability-first.draw_probability)} / ${pct(after.away_win_probability-first.away_win_probability)}`:'无法计算');$('f-state').textContent=run?.is_demo?'演示数据':fStatus?text(fStatus):fCalled?(featureValid?'已调用 · 产生有效特征':'已调用 · 特征缺失'):'尚未调用';
}

function renderTrend(run){
  if(state.workspace==='research')return renderResearchTrend(run);
  const events=predictionEvents(run);const points=events.map(e=>({at:e.at||e.data?.after?.prediction_cutoff,home:e.data.after.home_win_probability,draw:e.data.after.draw_probability,away:e.data.after.away_win_probability})).filter(p=>[p.home,p.draw,p.away].every(finite));const chart=$('trend-chart');chart.replaceChildren();
  $('trend-empty').hidden=points.length>1;$('trend-badge').textContent=run?.is_demo?'演示趋势':points.length>1?`${points.length} 个有效点`:'等待有效时间序列';
  $('trend-empty').textContent=points.length?'当前只有一个有效预测时间点，暂无可绘制的历史趋势':'当前没有有效预测时间点，暂无可绘制的历史趋势';
  if(points.length<2)return;
  const width=Math.max(240,chart.clientWidth),height=Math.max(180,chart.clientHeight-16),pad={l:40,r:28,t:16,b:30};
  const x=i=>pad.l+(width-pad.l-pad.r)*(i/(points.length-1)),y=v=>pad.t+(height-pad.t-pad.b)*(1-v);
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="模型预测概率趋势：主胜、平局、客胜"><g class="grid-lines">${[0,.25,.5,.75,1].map(v=>`<line x1="${pad.l}" x2="${width-pad.r}" y1="${y(v)}" y2="${y(v)}"></line><text x="0" y="${y(v)+4}">${Math.round(v*100)}%</text>`).join('')}</g>`;
  for(const [name,label,color] of [['home','主胜','var(--home)'],['draw','平局','var(--draw)'],['away','客胜','var(--away)']]){
    const path=points.map((p,i)=>`${i?'L':'M'} ${x(i).toFixed(1)} ${y(p[name]).toFixed(1)}`).join(' ');
    svg+=`<path class="trend-line ${name}" stroke="${color}" d="${path}"></path>${points.map((p,i)=>`<circle cx="${x(i)}" cy="${y(p[name])}" r="3" fill="${color}"><title>${esc(label)} ${pctExact(p[name])} · ${esc(fmtTime(p.at))}</title></circle>`).join('')}`;
  }
  // Keep all observations; reduce only axis labels when the plot is narrow.
  const tickSlots=Math.max(1,Math.floor((width-pad.l-pad.r)/64));const stride=Math.max(1,Math.ceil((points.length-1)/tickSlots));
  svg+=`<g class="x-labels">${points.map((p,i)=>(i===points.length-1||(i%stride===0&&i<points.length-1-stride/2))?`<text x="${x(i)}" y="${height-6}" text-anchor="middle">${esc(fmtTime(p.at,false))}</text>`:'').join('')}</g></svg>`;chart.innerHTML=svg;
}

function renderOdds(run){
  const f=run?.features;const odds=[f?.odds_home,f?.odds_draw,f?.odds_away];const labels=['主胜','平局','客胜'];const grid=$('odds-grid');grid.replaceChildren();const has=odds.every(finite);$('odds-status').textContent=has?'已核验':'暂无数据';$('odds-status').className=`small-badge ${has?'success-badge':''}`;
  odds.forEach((value,i)=>{const implied=has?(1/value)/odds.reduce((sum,v)=>sum+1/v,0):null;const el=document.createElement('div');el.className='odds-item';el.innerHTML=`<span>${labels[i]}</span><strong>${finite(value)?value.toFixed(2):'—'}</strong><small>隐含概率 ${pct(implied)}</small>`;grid.append(el);});
  $('odds-note').textContent=has?`来源：${text(f.provider)} · 读取 ${fmtTime(f.odds_available_at)} · 已归一化隐含概率 · 赔率变化方向：后端未返回时间序列${run.is_demo?' · 演示数据':''}`:'暂无已核验数据；不会显示 0 代替缺失；赔率变化方向暂无已核验数据。';
}

function modelCards(run){
  const prediction=run?.prediction || run?.last_prediction;const modelName=prediction?.model_version || run?.tool_configuration?.model_name;const card=(name,body,cls='')=>`<article class="model-card ${cls}"><span>${esc(name)}</span><strong>${body}</strong><small>${cls==='active'?'已参与最终预测':'暂无可核验输出'}</small></article>`;
  return [card('Elo','—'),card('Poisson','—'),card('xG','—'),card('机器学习','—'),card(text(modelName,'f 更新后的结果'),prediction?`${pct(prediction.home_win_probability)} / ${pct(prediction.draw_probability)} / ${pct(prediction.away_win_probability)}`:'—','active')].join('');
}
function renderModels(run){$('models-row').innerHTML=modelCards(run);$('models-detail').innerHTML=`<div class="model-detail-grid">${modelCards(run)}</div><div class="micro-note">每个模型只展示后端实际返回的输出；缺少真实 Elo、Poisson、xG 或机器学习结果时保持空状态。</div>`;}

function evidenceStatus(doc,run){if(doc?.f_consumed===true || doc?.admission==='verified')return {label:'已被 f 使用',kind:'verified'};if(doc?.admission==='unmodeled')return {label:'未进入 f',kind:'model_inferred'};if(doc?.admission==='rejected')return {label:'已拒绝',kind:'unknown'};return {label:'未知',kind:'unknown'};}
function renderEvidence(run){
  if(state.workspace==='research')return renderResearchEvidence(run);
  const docs=run?.evidence||[];const filter=state.evidenceFilter;const list=$('evidence-list');list.replaceChildren();const visible=docs.filter(d=>filter==='all'||evidenceStatus(d,run).kind===filter);if(!visible.length){list.innerHTML=`<div class="empty-module"><div class="empty-glyph">∅</div><strong>${docs.length?'该筛选下暂无证据':'尚无来源返回'}</strong><p>来源返回后，Agent 会继续做时间、实体和正文准入检查。</p></div>`;return;}
  for(const doc of visible){const st=evidenceStatus(doc,run);const content=doc.content||{};const uncertainty=[];if(doc.source_published_at==null)uncertainty.push('发布时间未知');if(content.missing?.xg==='not_integrated')uncertainty.push('xG 未接入');if(content.missing?.score==='not_yet_reported')uncertainty.push('赛果未报告');const article=document.createElement('article');article.className='evidence-card';article.innerHTML=`<div class="evidence-main"><div class="evidence-top"><span class="source-badge">${esc(doc.source||'来源未知')}</span><span class="evidence-kind">${esc(doc.kind||'document')}</span><span class="evidence-state ${st.kind}">${esc(st.label)}</span></div><h3>${esc(content.title||`${doc.kind==='odds'?'赔率快照':'比赛身份'} · ${doc.evidence_id||'未命名'}`)}</h3><p class="evidence-meta">观察 ${esc(fmtTime(doc.observed_at))} · 可用 ${esc(fmtTime(doc.available_at))} · ${doc.source_published_at?`发布 ${esc(fmtTime(doc.source_published_at))}`:'发布时间未知'}</p><div class="evidence-tags">${uncertainty.map(x=>`<span class="mini-tag warning">${esc(x)}</span>`).join('')}<span class="mini-tag">cutoff ${doc.available_at && run?.prediction_cutoff && new Date(doc.available_at)<=new Date(run.prediction_cutoff)?'之前':'之后'}</span></div></div><details class="evidence-details"><summary>展开证据摘要与引用</summary><div class="detail-lines"><p><b>正文准入：</b>${st.kind==='verified'?'通过来源和实体检查':'尚未进入 f'}</p><p><b>正文完整：</b>${doc.body_complete===true?'是':doc.body_complete===false?'否':'未知'}</p><p><b>保存时间：</b>${fmtTime(doc.saved_at||doc.persisted_at)}</p><p><b>来源 SHA：</b><code>${esc(doc.content_hash||doc.raw_content_hash||'未返回')}</code></p><p><b>URL：</b>${doc.url?`<a href="${esc(doc.url)}" target="_blank" rel="noopener">${esc(doc.url)}</a>`:'暂无 URL（本地档案引用）'}</p><p><b>原文引用：</b>${content.quote?esc(content.quote):'正文未返回；下面展示结构化来源载荷。'}</p><p><b>引用 offset：</b>${text(content.quote_offset||doc.quote_offset,'未返回')}</p><p><b>claim / 球队 / 球员：</b>${text(content.claim_type,'未返回')} · ${text(content.team,'未返回')} · ${text(content.player,'未返回')}</p><p><b>时间范围：</b>${text(content.time_range,'未返回')} · <b>不确定性：</b>${text(content.uncertainty,'未返回')}</p><pre>${esc(jsonPreview(content).slice(0,3600))}</pre></div></details>`;list.append(article);}
}

function renderTimeline(run){
  const timeline=$('timeline');timeline.replaceChildren();const events=run?.events||[];const status=String(run?.state||'INITIALIZED').toUpperCase();
  const statusMap={INITIALIZED:'未开始',MATCH_READY:'比赛已就绪',EVIDENCE_READY:'证据已就绪',FEATURES_READY:'等待 f',PREDICTED:'预测已生成',COMPLETED:'已完成',STOPPED:'已停止',DEFERRED:'已暂缓',FAILED:'失败',BUDGET_EXHAUSTED:'预算耗尽',DEADLINE_EXPIRED:'已超时'};
  if(!events.length)timeline.innerHTML='<div class="empty-inline">尚无 Agent 事件。</div>';
  events.slice(-24).forEach(event=>{
    const data=event.data||{};const item=document.createElement('details');item.className=`timeline-item ${event.type==='prediction_updated'?'highlight':''}`;item.dataset.event=event.type;
    const action=data.action_id?` · ${data.action_id}`:'';
    item.innerHTML=`<summary><span class="timeline-dot"></span><span class="timeline-title">${esc(eventTitle(event))}${esc(action)}</span><span class="timeline-time">${esc(fmtTime(event.at,false))}</span><span class="timeline-state">${esc(eventStatus(event.type))}${data.status?` · ${esc(data.status)}`:''}</span></summary><div class="timeline-detail"><p>${esc(data.reason||data.message||toolLabels[data.tool]||'该步骤已写入审计链。')}</p><div class="timeline-meta"><span>seq #${event.seq||'—'}</span><span>action_id ${esc(text(data.action_id,'未提供'))}</span><span>来源 ${esc(text(data.source||data.tool,'runtime'))}</span><span>耗时 ${data.duration_ms!=null?`${data.duration_ms} ms`:'未返回'}</span></div><pre>${esc(jsonPreview(data).slice(0,3000))}</pre></div>`;timeline.append(item);
  });
  $('agent-status').className=`run-status ${terminalStates.has(status)?'done':''}`;$('agent-status').dataset.state=status;$('agent-status').innerHTML=`<i></i>${esc(statusMap[status]||status)}`;
  const reason=uiView(run)?.lifecycle?.reason || run?.reason;const reasonLabels={NO_NEW_VALID_EVIDENCE:'未发现新的有效证据',POLICY_STOP:'策略决定停止'};
  $('agent-outcome').hidden=!reason;$('agent-outcome-label').textContent=terminalStates.has(status)?'结束原因':'运行原因';
  $('agent-reason').textContent=reasonLabels[reason] || text(reason,'未返回');$('agent-reason-code').textContent=reasonLabels[reason]?reason:'';
  $('agent-log-count').textContent=`最近 ${Math.min(24,events.length)} / ${events.length} 条`;
  const selectedTopic=events.filter(e=>e.type==='policy_decision').at(-1)?.data?.topic;
  $('agent-summary').textContent=run?.is_demo?'演示数据：本次没有调用真实 Agent runtime。':terminalStates.has(status)?`本轮已结束，返回 ${(run.evidence||[]).length} 条来源，记录 ${events.length} 个执行事件。来源与特征可继续复核。`:selectedTopic?`Agent 选择了查询「${selectedTopic}」，返回结果会再次进入 f / g 边界检查。`:'Agent 正在从比赛身份、市场先验和可用来源建立研究状态。';
  $('agent-next-step').textContent=terminalStates.has(status)?'查看证据或展开执行步骤，复核返回结果与结束原因。':'展开执行步骤可查看工具返回与原始记录。';
}

function renderCandidates(run){
  const view=uiView(run);const g=view?.g||{};const actions=Array.isArray(g.candidate_actions)?g.candidate_actions:[];const box=$('candidate-actions');box.replaceChildren();const gStatus=g.status==='unknown'?'UNKNOWN':text(g.status,'UNKNOWN');$('g-status').textContent=run?.is_demo?'g UNKNOWN · 演示':`g ${gStatus}`;if(!actions.length){box.innerHTML='<div class="empty-inline">后端没有返回候选动作；页面不生成默认主题。</div>';return;}for(const action of actions){const topic=action.information_key||action.topic;const value=action.expected_net_voi ?? action.expected_signed_gain;const row=document.createElement('button');row.type='button';row.className='candidate-row';row.innerHTML=`<span class="candidate-name">${esc(topic||'未命名动作')}</span><span>${finite(value)?value.toFixed(3):'UNKNOWN'}</span><span>${esc(action.status||'未知')}</span><span class="chevron">›</span><div class="candidate-detail"><p><b>action_id：</b>${text(action.action_id,'未提供')}</p><p><b>information_key：</b>${text(action.information_key,'未返回')}</p><p><b>expected_net_voi：</b>${text(action.expected_net_voi,'UNKNOWN')}</p><p><b>expected_signed_gain：</b>${text(action.expected_signed_gain,'UNKNOWN')}</p><p><b>查询成本：</b>${text(action.cost,'未返回')}</p><p><b>propensity：</b>${text(action.propensity,'未返回')}</p><p><b>选择原因：</b>${esc(action.selection_reason||'未返回')}</p><p><b>是否执行：</b>${action.request_executed===true?'是':action.request_executed===false?'否':'未知'}</p><p><b>是否进入 f：</b>${esc(action.f_status||'unknown')}</p></div>`;row.addEventListener('click',()=>row.classList.toggle('expanded'));box.append(row);}
}

function renderChecks(run){
  const view=uiView(run);const match=run?.match||state.selected||{};const f=view?.f||{};const g=view?.g||{};const p=view?.prediction||{};const identity=match.identity_quality?.home==='stable'&&match.identity_quality?.away==='stable';const items=[['比赛身份',identity?'已核验':'仍需确认',identity?'verified':'unknown'],['市场先验 prediction_before',p.before?'已返回':'未返回 · 不做假设',p.before?'verified':'unknown'],['f 状态',text(f.status,'unknown'),f.status?.startsWith('executed')?'verified':'unknown'],['f 消费证据',f.consumed_evidence_ids?.length?`${f.consumed_evidence_ids.length} 条`:'未知',f.consumed_evidence_ids?.length?'verified':'unknown'],['DeepSeek 语义抽取','未接入当前运行','unknown'],['empirical g',g.status==='empirical_supported'?'已返回':'UNKNOWN',g.status==='empirical_supported'?'verified':'unknown'],['xG / 比分分布','暂无可核验数据','unknown']];const box=$('status-checks');box.innerHTML=items.map(([label,value,kind])=>`<div class="check-row"><span class="check-icon ${kind}">${kind==='verified'?'✓':'?'}</span><span>${esc(label)}</span><b class="${kind}">${esc(value)}</b></div>`).join('');}

function renderUncertainty(run){const view=uiView(run);const list=Array.isArray(view?.uncertainty)?view.uncertainty.map(item=>item?.message||item).filter(Boolean):[];if(run?.is_demo)list.unshift('当前比赛来自演示数据，不代表实时预测或真实 Agent 运行。');$('uncertainty-count').textContent=`${list.length} 项`;$('uncertainty-list').innerHTML=list.map(x=>`<div class="uncertainty-item"><span>!</span><p>${esc(x)}</p></div>`).join('');}

function renderRun(run){if(!run)return;state.run=run;renderHero(run);renderProbability(run);renderTrend(run);renderOdds(run);renderModels(run);renderEvidence(run);renderTimeline(run);renderCandidates(run);renderChecks(run);renderUncertainty(run);$('run-agent').disabled=state.running || (!run.is_demo && terminalStates.has(String(run.state||'').toUpperCase()));$('run-agent').textContent=state.running?'Agent 正在运行…':run.is_demo?'演示数据 · 不执行':({STOPPED:'本轮已停止',COMPLETED:'本轮已完成',DEFERRED:'本轮已暂缓',FAILED:'本轮运行失败',BUDGET_EXHAUSTED:'本轮预算耗尽',DEADLINE_EXPIRED:'本轮已超时'})[String(run.state||'').toUpperCase()]||'开始 / 继续分析';}
function renderAgentRunning(){ $('agent-outcome').hidden=true;$('agent-status').dataset.state='RUNNING';$('agent-status').className='run-status running';$('agent-status').innerHTML='<i></i>正在分析';$('agent-summary').textContent='Agent 正在执行工具链；返回后会更新来源、f 特征、概率和停止原因。';$('run-agent').disabled=true;$('run-agent').textContent='Agent 正在运行…'; }

async function loadMatches(){if(state.workspace==='research')return loadResearchCatalog();const ticket=++state.request;setConnection('正在读取真实档案');$('data-mode').innerHTML='<span class="status-dot"></span><span>正在读取真实档案</span>';try{const result=await FootballMindAdapter.loadMatches();if(ticket!==state.request)return;state.matches=result.matches;state.meta=result;populateLeagues();$('source-label').textContent=result.source;$('matches-updated').textContent=fmtTime(result.updatedAt,false);$('as-of').textContent=`数据截止 ${fmtTime(result.cutoff)}`;$('data-mode').className=`data-mode ${result.isDemo?'demo-mode':''}`;$('data-mode').innerHTML=`<span class="status-dot"></span><span>${esc(result.source)}</span>`;setConnection(result.isDemo?'演示 fallback · 服务已连接':'真实档案已连接');renderMatchList();const selected=state.selected&&state.matches.find(m=>m.match_id===state.selected.match_id);await selectMatch((selected||state.matches[0])?.match_id);}catch(error){if(ticket!==state.request)return;setConnection('连接失败',false);setNotice(`比赛接口读取失败：${error.message}`);$('match-list').replaceChildren();$('match-empty').hidden=false;}}

function activateTab(tab){
  state.tab=tab;document.querySelectorAll('.tab').forEach(b=>{const active=b.dataset.tab===tab;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;});
  document.querySelectorAll('.tab-panel').forEach(p=>p.hidden=p.id!==`tab-${tab}`);
}
function bind(){
  $('match-search').addEventListener('input',e=>{state.search=e.target.value;renderMatchList();});$('league-filter').addEventListener('change',e=>{state.league=e.target.value;renderMatchList();});$('sort-filter').addEventListener('change',e=>{state.sort=e.target.value;renderMatchList();});$('refresh-all').addEventListener('click',loadMatches);$('retry-matches').addEventListener('click',loadMatches);$('retry-detail').addEventListener('click',()=>state.workspace==='research'?loadResearchCatalog():state.selected&&selectMatch(state.selected.match_id));$('run-agent').addEventListener('click',async()=>{if(state.workspace==='research')return runResearch();if(!state.selected||state.running)return;try{setNotice('');let run=state.run||state.runCache.get(state.selected.match_id);if(!run)run=await FootballMindAdapter.startRun(state.selected);state.runCache.set(state.selected.match_id,run);if(run.is_demo){renderRun(run);return;}state.running=true;renderAgentRunning();run=await FootballMindAdapter.runToCompletion(run);state.running=false;state.runCache.set(state.selected.match_id,run);renderRun(run);renderMatchList();}catch(error){state.running=false;setNotice(`Agent 运行失败：${error.message}`);$('run-agent').disabled=false;}});
  document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>activateTab(b.dataset.tab)));document.querySelectorAll('[data-tab-jump]').forEach(b=>b.addEventListener('click',()=>activateTab(b.dataset.tabJump)));$('toggle-prediction').addEventListener('click',()=>{$('prediction-breakdown').hidden=!$('prediction-breakdown').hidden;$('toggle-prediction').setAttribute('aria-expanded',String(!$('prediction-breakdown').hidden));});document.querySelectorAll('.nav-link').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.nav-link').forEach(x=>x.classList.remove('active'));b.classList.add('active');const target=b.dataset.nav==='research'?'.research-column':b.dataset.nav==='data'?'#tab-evidence':b.dataset.nav==='analysis'?'.probability-panel':b.dataset.nav==='system'?'.process-panel':'.match-sidebar';if(b.dataset.nav==='data')activateTab('evidence');if(b.dataset.nav==='analysis')activateTab('overview');document.querySelector(target)?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});}));document.querySelectorAll('#evidence-filter button').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('#evidence-filter button').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.evidenceFilter=b.dataset.filter;renderEvidence(state.run);}));
  document.querySelector('.section-tabs').addEventListener('keydown',event=>{const tabs=[...document.querySelectorAll('.tab')];const index=tabs.indexOf(event.target);if(index<0)return;let next=index;if(event.key==='ArrowRight')next=(index+1)%tabs.length;else if(event.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=tabs.length-1;else return;event.preventDefault();activateTab(tabs[next].dataset.tab);tabs[next].focus();});
}

// Resize only the existing plot; data, prediction values and API calls are unchanged.
document.addEventListener('DOMContentLoaded',()=>{
  bind();
  initResearchWorkspace();
  const chartResize=new ResizeObserver(()=>{if(state.run && $('trend-chart').clientWidth)renderTrend(state.run);});
  chartResize.observe($('trend-chart'));
  loadMatches();
});


'use strict';

// A separate renderer/state machine for the research contract. Legacy run
// objects, caches and API calls remain in app.js and data-adapter.js.
const research = {catalog:null, cache:new Map(), refs:[], report:null, needsSync:false, restore:true};
const researchKeys = ['home_win_probability','draw_probability','away_win_probability'];
const strategyNames = {g_two_step:'两步 g',g_one_step:'一步 g',fixed_order:'固定顺序对照'};
const researchReasons = {
  budget_exhausted:'读取预算已用尽', no_feasible_candidate:'没有可用候选',
  nonpositive_learned_two_step_net_value:'模型估计的下一步净价值非正', user_stop:'用户停止本轮',
  positive_learned_two_step_net_value:'模型估计净价值为正', fixed_order_control:'固定顺序对照'
};
const researchNumber = x => finite(x) ? x.toFixed(6) : '未知';
const researchVector = p => p ? (Array.isArray(p)?p:researchKeys.map(k=>p[k])).map(pctExact).join(' / ') : '尚未执行正文更新';
const researchReason = reason => researchReasons[reason] || text(reason,'尚未停止');
function researchLink(url,label='查看原始来源') {
  try {const parsed=new URL(url);if(['https:','http:'].includes(parsed.protocol))return `<a href="${esc(parsed.href)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>`;}catch(_){}
  return '来源链接未返回';
}
function researchUrl(id) {
  const url=new URL(location.href);url.searchParams.delete('run');url.searchParams.delete('workspace');
  if(id)url.searchParams.set('research_run',id);else url.searchParams.delete('research_run');
  history.replaceState(null,'',url);
}
function persistResearchSelection(id=null){
  try{localStorage.setItem('footballmind.research-g.v1',JSON.stringify({active:id,selected:state.selected?.match_id,refs:research.refs}));}
  catch(_){setNotice('浏览器无法保存恢复记录；可使用地址栏中的研究运行链接恢复。');}
}
function saveResearch(run) {
  research.cache.set(run.match_id,run);state.run=run;
  research.refs=[{id:run.run_id,match:run.match_id,strategy:run.g.strategy},...research.refs.filter(r=>r.id!==run.run_id)];
  researchUrl(run.run_id);
  persistResearchSelection(run.run_id);
  renderResearchHistory();
}
function renderResearchHistory(){
  const options=research.refs.map(ref=>{const m=state.matches.find(m=>m.match_id===ref.match);return new Option(`${m?`${m.home} / ${m.away}`:ref.match} · ${strategyNames[ref.strategy]||ref.strategy} · ${ref.id.slice(-6)}`,ref.id);});
  $('research-history').replaceChildren(new Option('选择已保存的研究运行',''),...options);$('research-history').value=state.run?.run_id||'';
}
function renderResearchUncertain(){
  $('agent-status').className='run-status';$('agent-status').dataset.state='UNKNOWN';$('agent-status').innerHTML='<i></i>状态待核对';
  $('agent-summary').textContent='请求结果尚未核对。当前保留最后一次确认的数据，先核对状态再继续。';
}
function researchButtons(){
  const run=state.run, busy=state.running, terminal=run?.lifecycle?.is_terminal;
  const fixed=(run?.g.strategy||$('research-strategy').value)==='fixed_order';
  $('run-agent').disabled=busy||research.needsSync||!research.catalog||!state.selected||terminal||state.selected.eligibility?.run_allowed!==true;
  $('run-agent').textContent=busy?'正在执行，请稍候…':terminal?'本轮已停止':run?'自动运行至停止':$('research-step-mode').checked?'开始 · 准备逐步演示':fixed?'开始 · 固定顺序对照':'开始 · g 自主研究';
  $('research-next').textContent=fixed?'按固定顺序执行一步':'g 自主执行一步';
  $('research-next').disabled=busy||research.needsSync||!run||terminal;
  $('research-stop').disabled=busy||research.needsSync||!run||terminal;
  $('research-new').disabled=busy||!run||!terminal;
  $('research-refresh').disabled=busy||!run;
  $('research-history').disabled=busy;
  $('workspace-mode').disabled=busy;
  $('refresh-all').disabled=busy;
  document.querySelectorAll('#research-option-fields input, #research-option-fields select, .research-option-fields input, .research-option-fields select').forEach(el=>el.disabled=busy||Boolean(run));
  document.querySelectorAll('.match-card').forEach(el=>el.disabled=busy);
  $('download-research-report').disabled=busy||!research.report?.evaluation;
}
function initResearchWorkspace(){
  state.workspace=new URL(location.href).searchParams.get('workspace')==='legacy'?'legacy':'research';
  try{const saved=JSON.parse(localStorage.getItem('footballmind.research-g.v1')||'{}');research.refs=(saved.refs||[]).filter(r=>/^rg-[a-f0-9]{32}$/.test(r.id));research.active=saved.active;research.selected=saved.selected;}catch(_){}
  $('workspace-mode').value=state.workspace;updateWorkspaceVisibility();
  $('workspace-mode').addEventListener('change',async e=>{
    if(e.target.value==='live'){location.href='/live.html';return;}
    if(state.running){e.target.value=state.workspace;return;}state.workspace=e.target.value;state.request++;state.selected=null;state.run=null;state.matches=[];research.report=null;research.needsSync=false;
    const url=new URL(location.href);url.searchParams.delete('research_run');url.searchParams.delete('run');url.searchParams.set('workspace',state.workspace);history.replaceState(null,'',url);
    updateWorkspaceVisibility();resetDetail();setNotice('');await loadMatches();
  });
  $('research-step-mode').addEventListener('change',researchButtons);
  $('research-strategy').addEventListener('change',researchButtons);
  $('research-next').addEventListener('click',()=>runResearch('next'));
  $('research-stop').addEventListener('click',()=>runResearch('stop'));
  $('research-refresh').addEventListener('click',()=>recoverResearch(state.run?.run_id));
  $('research-new').addEventListener('click',()=>{if(state.running||!state.run?.lifecycle.is_terminal)return;state.run=null;research.report=null;research.needsSync=false;researchUrl(null);persistResearchSelection();renderResearchRun();renderResearchHistory();setNotice('已准备独立新运行。调整策略后点击开始；此前运行仍可从研究记录中打开。');});
  $('research-history').addEventListener('change',e=>{if(e.target.value)recoverResearch(e.target.value);});
  $('download-research-report').addEventListener('click',()=>{
    if(!research.report?.evaluation)return;const url=URL.createObjectURL(new Blob([JSON.stringify(research.report,null,2)],{type:'application/json'}));
    const link=document.createElement('a');link.href=url;link.download=`${research.report.view.run_id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  $('open-legacy-run').addEventListener('click',async()=>{
    if(state.running||state.workspace!=='legacy')return;
    try{const id=$('legacy-run-id').value.trim();if(!id||id.startsWith('rg-'))throw new Error('请输入旧实验运行 ID；rg- 研究记录请在历史研究工作区打开。');
      state.running=true;$('open-legacy-run').disabled=true;const run=await FootballMindAdapter.fetchRun(id);renderRun(run);setNotice('已只读打开旧档案实验；此记录与历史研究 g 独立。');
    }catch(error){setNotice(`旧档案读取失败：${error.message}`);}finally{state.running=false;$('open-legacy-run').disabled=false;}
  });
}
function updateWorkspaceVisibility(){
  const active=state.workspace==='research';
  for(const id of ['research-history-wrap','research-options','research-controls','research-run-meta','research-scope'])$(id).hidden=!active;
  $('legacy-open').hidden=active;$('research-report').hidden=true;$('evidence-filter').hidden=active;
  $('g-explanation').textContent=active?'预期净价值 = 预期即时损失改善 − 当前查询成本 + 预计后续净价值。后续项是已训练模型估计，未提前读取未选正文；只剩一次读取时为 0。':'g 负责选择下一步信息；未返回的 empirical g 数值保持 UNKNOWN。';
  document.querySelector('#prediction-breakdown > div > span').textContent=active?'prediction_before · 初始 f':'prediction_before · 市场先验';
  document.querySelector('footer span:last-child').textContent=active?'历史研究 · 冻结 f / g 实际推理 · 归档语义 · 无在线采集':'f：特征与概率更新 · g：下一步信息选择 · DeepSeek：语义抽取器';
}
async function loadResearchCatalog(){
  if(state.running)return;state.running=true;researchButtons();setConnection('正在核对研究服务');setNotice('');
  try{
    research.catalog=await FootballMindAdapter.researchCatalog();state.matches=research.catalog.matches;populateLeagues();
    $('source-label').textContent='历史重建目录';$('matches-updated').textContent='运行时读取（北京时间）';$('as-of').textContent='历史研究 · 北京时间 UTC+8';
    $('data-mode').className='data-mode';$('data-mode').innerHTML='<span class="status-dot"></span><span>历史研究运行 · 全部归档场次</span>';setConnection('研究接口已连接');
    let id=state.run?.run_id;
    if(research.restore){id=new URL(location.href).searchParams.get('research_run')||research.active;research.restore=false;}
    if(id){const run=await FootballMindAdapter.fetchResearchRun(id);state.selected=state.matches.find(m=>m.match_id===run.match_id);if(!state.selected)throw new Error('该运行的比赛未在研究目录中返回');saveResearch(run);research.needsSync=false;}
    else {state.selected=state.matches.find(m=>m.match_id===(state.selected?.match_id||research.selected))||state.matches[0];state.run=null;}
    research.needsSync=false;renderResearchHistory();renderResearchRun();renderResearchMatchList();await loadResearchReport();
  }catch(error){research.needsSync=true;setConnection('研究读取失败',false);setNotice(`研究接口读取失败：${error.message}。请确认地址连接 python -m demo 服务；不会回退到演示数据。`);
    if(!state.selected){$('detail-loading').hidden=true;$('detail-error').hidden=false;$('detail-error-text').textContent=error.message;$('g-status').textContent='研究服务读取失败';}
    if(research.catalog){renderResearchMatchList();renderResearchHistory();}
  }finally{state.running=false;researchButtons();}
}
async function selectResearchMatch(id){
  if(state.running)return;const match=state.matches.find(m=>m.match_id===id);if(!match)return;
  state.selected=match;state.run=null;research.report=null;research.needsSync=false;setNotice('');researchUrl(null);persistResearchSelection();
  const cached=research.cache.get(id);
  if(cached)return recoverResearch(cached.run_id);
  renderResearchRun();renderResearchHistory();renderResearchMatchList();
}
async function recoverResearch(id){
  if(!id||state.running)return;state.running=true;researchButtons();setNotice('');
  try{const run=await FootballMindAdapter.fetchResearchRun(id);const match=state.matches.find(m=>m.match_id===run.match_id);if(!match)throw new Error('恢复运行不属于当前研究目录');state.selected=match;research.report=null;research.needsSync=false;saveResearch(run);renderResearchRun();renderResearchMatchList();setConnection('研究接口已连接');await loadResearchReport();}
  catch(error){research.needsSync=true;renderResearchUncertain();setNotice(`恢复失败：${error.message}。尚未推进研究；请刷新重试。`);}
  finally{state.running=false;researchButtons();}
}
async function runResearch(action='run'){
  if(state.running||research.needsSync||!state.selected||!research.catalog||state.run?.lifecycle.is_terminal)return;
  const max=Number($('research-max-reads').value),cost=Number($('research-cost').value),strategy=$('research-strategy').value;
  if(!state.run&&(!Number.isInteger(max)||max<1||max>3||!finite(cost)||cost<0||cost>0.01||!$('research-cost').value||!research.catalog.strategies.includes(strategy))){setNotice('最多读取篇数须为 1–3 的整数，每篇成本须为 0–0.01。');return;}
  state.running=true;researchButtons();setNotice('');research.report=null;
  $('agent-status').dataset.state='RUNNING';$('agent-status').className='run-status running';$('agent-status').innerHTML='<i></i>模型执行中';
  try{
    if(!state.run){
      saveResearch(await FootballMindAdapter.startResearchRun({match_id:state.selected.match_id,max_reads:max,cost_loss_units:cost,strategy}));renderResearchRun();
      if($('research-step-mode').checked)action=null;
    }
    if(action&&!state.run.lifecycle.is_terminal){saveResearch(await FootballMindAdapter.researchAction(state.run,action));}
    renderResearchRun();renderResearchMatchList();await loadResearchReport();setConnection('研究接口已连接');
  }catch(error){
    let recovery='创建结果未确认；没有自动重试。若服务端已创建，可用服务端运行 ID 恢复。';
    if(state.run){
      try{saveResearch(await FootballMindAdapter.fetchResearchRun(state.run.run_id));research.needsSync=false;recovery='已只读核对最新状态；未自动重复推进。';renderResearchRun();renderResearchMatchList();await loadResearchReport();}
      catch(_){research.needsSync=true;renderResearchUncertain();recovery='无法核对服务端状态，推进已锁定；请先核对状态或刷新。';}
    }else{renderResearchRun();}
    setNotice(`${error.message.includes('STALE_REVISION')?'运行版本已变化': '研究请求失败'}：${error.message}。${recovery}`);
  }finally{state.running=false;researchButtons();}
}
async function loadResearchReport(){
  $('research-report').hidden=!state.run?.lifecycle.is_terminal;if($('research-report').hidden)return;
  research.report=null;$('research-evaluation').textContent='正在只读加载停止后的报告…';
  try{const report=await FootballMindAdapter.fetchResearchReport(state.run.run_id);if(report.view.run_id!==state.run.run_id||report.view.revision!==state.run.revision)throw new Error('报告版本与当前运行不一致，请核对状态');research.report=report;renderResearchReport();}
  catch(error){$('research-evaluation').textContent=`报告读取失败：${error.message}。点击“核对状态”重试，不会重新执行模型。`;}
}
function renderResearchMatchList(){
  const rows=filteredMatches();$('fixture-count').textContent=`${rows.length} 场`;$('match-empty').hidden=rows.length>0;$('match-list').replaceChildren();
  rows.forEach(m=>{const run=research.cache.get(m.match_id),p=run?.prediction.current;const card=document.createElement('button');card.type='button';card.className=`match-card${state.selected?.match_id===m.match_id?' active':''}`;card.dataset.matchId=m.match_id;card.setAttribute('aria-pressed',String(state.selected?.match_id===m.match_id));card.disabled=state.running;
    card.innerHTML=`<div class="match-card-top"><span class="match-time">${esc(fmtTime(m.kickoff))}</span></div><div class="match-teams"><strong>${esc(m.home)}</strong><span>vs</span><strong>${esc(m.away)}</strong></div><div class="match-probs"><span>主 ${pct(p?.home_win_probability)}</span><span>平 ${pct(p?.draw_probability)}</span><span>客 ${pct(p?.away_win_probability)}</span></div><div class="match-card-foot"><span class="mini-status ${run?'ready':''}"><i></i>${run?(run.lifecycle.is_terminal?'已停止':'已准备'):'待开始'}</span><span class="mini-tag">${esc(m.candidate_count)} 篇候选</span></div>`;
    card.addEventListener('click',()=>selectResearchMatch(m.match_id));$('match-list').append(card);
  });
}
function renderResearchRun(){
  const run=state.run,match=run?.match||state.selected||{},p=run?.prediction||{};
  if(run){$('research-strategy').value=run.g.strategy;$('research-max-reads').value=run.budget.max_reads;$('research-cost').value=run.budget.cost_per_read;}
  $('research-scope').textContent=run?.g.selection_authority==='fixed_order'?'旧版历史研究 · 固定顺序对照 · 原实体 f 重算 · 不使用新版 Qwen 全文模型':'旧版历史研究 · 标题词特征与状态 g · 原实体 f 重算 · 不使用新版 Qwen 全文模型';
  $('detail-loading').hidden=true;$('detail-error').hidden=true;$('detail-content').hidden=false;
  $('hero-league').textContent=text(match.competition);$('hero-status').textContent='历史研究';$('hero-status').className='status-chip';$('hero-kickoff').textContent=`${fmtTime(match.kickoff)} UTC+8`;
  for(const [side,key] of [['home','home'],['away','away']]){$(`${side}-name`).textContent=text(match[key]);$(`${side}-badge`).textContent=(match[key]||'?').slice(0,1);$(`${side}-id`).textContent='归档目录球队';}
  $('match-id').textContent=run?.match_id||state.selected?.match_id||'—';$('identity-status').textContent='历史目录 · 非实时赛前';$('identity-status').className='muted';
  $('hero-cutoff').textContent=`${fmtTime(run?.prediction_cutoff||state.selected?.prediction_cutoff)} UTC+8`;$('hero-updated').textContent=run?.steps.length?`${fmtTime(run.steps.at(-1).executed_at)} UTC+8`:'运行时间未返回';
  const vals=researchKeys.map(k=>p.current?.[k]),max=vals.every(finite)?Math.max(...vals):null;
  ['home','draw','away'].forEach((key,i)=>{const item=$(`prob-${key}`).closest('.prob-item'),lead=finite(vals[i])&&vals[i]===max;$(`prob-${key}`).textContent=pctExact(vals[i]);$(`bar-${key}`).style.width=finite(vals[i])?`${vals[i]*100}%`:'0%';item.classList.toggle('is-leading',lead);item.querySelector('.prob-rank').hidden=!lead;item.querySelector('.prob-rank').textContent=vals.filter(v=>v===max).length>1?'并列最高':'最高项';});
  $('prediction-source').textContent=run?`配套 f · ${run.f.model_version}`:'点击开始后计算初始 f';$('prediction-time').textContent='历史重建 · 北京时间 UTC+8';$('support-count').textContent=`${run?.steps.length||0} 篇已披露正文`;
  const explanation=!run?'选择比赛后点击开始，由后端 g 自主选取文章、调用 f 并决定继续或停止。':!p.after?'尚未执行正文更新':p.change_status==='unchanged_after_f'?'f 已重算，概率未变':'f 已基于已披露正文重算；变化见 before / after。';
  $('prediction-explain').innerHTML=`<span class="explain-label">变化说明</span><p>${esc(explanation)}</p>`;
  $('before-value').textContent=p.before?researchVector(p.before):'尚未开始';$('after-value').textContent=researchVector(p.after);$('delta-value').textContent=p.delta?researchKeys.map(k=>finite(p.delta[k])?`${(100*p.delta[k]).toFixed(3)} pp`:'未知').join(' / '):'尚无正文更新对照';
  $('f-state').textContent=run?`${run.f.executed_this_run?'已执行':'未确认执行'} · ${run.f.status}`:'尚未开始';
  renderResearchTrend(run);renderOdds(null);renderModels(run?{prediction:p.current}:null);renderResearchEvidence(run);renderResearchTimeline(run);renderResearchCandidates(run);renderResearchChecks(run);
  document.querySelectorAll('#models-row .model-card.active, #models-detail .model-card.active').forEach(card=>{if(!run)card.classList.remove('active');card.querySelector('small').textContent=run?'本轮实际 f 输出':'尚未开始';});
  const limitations=[...(research.catalog?.limitations||[]),...(run?.limitations||[])];$('uncertainty-count').textContent=`${limitations.length} 项`;$('uncertainty-list').innerHTML=limitations.map(x=>`<div class="uncertainty-item"><span>!</span><p>${esc(x)}</p></div>`).join('');
  $('research-report').hidden=!run?.lifecycle.is_terminal;if(!run)research.report=null;researchButtons();
}
function renderResearchTrend(run){
  const points=run?.prediction.before?[{label:'初始 f',values:researchKeys.map(k=>run.prediction.before[k])},...run.steps.map(s=>({label:`第 ${s.step} 篇`,values:s.prediction_after}))]:[];
  const chart=$('trend-chart');chart.replaceChildren();$('trend-badge').textContent=`${points.length} 个实际模型输出`;$('trend-empty').hidden=points.length>1;$('trend-empty').textContent=points.length?'当前只有初始 f 输出；正文更新后显示逐步变化。':'开始研究后显示实际输出。';if(points.length<2)return;
  const w=Math.max(240,chart.clientWidth),h=Math.max(180,chart.clientHeight-16),x=i=>42+(w-74)*i/(points.length-1),y=v=>16+(h-50)*(1-v);
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="初始 f 与每次正文更新后的实际概率"><g class="grid-lines">${[0,.25,.5,.75,1].map(v=>`<line x1="42" x2="${w-32}" y1="${y(v)}" y2="${y(v)}"/><text x="0" y="${y(v)+4}">${v*100}%</text>`).join('')}</g>`;
  ['home','draw','away'].forEach((key,k)=>{svg+=`<path class="trend-line" stroke="var(--${key})" d="${points.map((p,i)=>`${i?'L':'M'} ${x(i)} ${y(p.values[k])}`).join(' ')}"/>${points.map((p,i)=>`<circle cx="${x(i)}" cy="${y(p.values[k])}" r="3" fill="var(--${key})"><title>${esc(p.label)} ${pctExact(p.values[k])}</title></circle>`).join('')}`;});
  svg+=`<g class="x-labels">${points.map((p,i)=>`<text x="${x(i)}" y="${h-5}" text-anchor="middle">${esc(p.label)}</text>`).join('')}</g></svg>`;chart.innerHTML=svg;
}
function candidateMarkup(actions,proposed,proposedLabel='g 当前提议'){
  if(!actions?.length)return '<div class="empty-inline">当前没有返回候选评分。</div>';
  return actions.map(a=>`<details class="research-candidate"><summary><span>${esc(a.title||a.action_id)}</span><b>${researchNumber(a.two_step_net_value)}</b>${proposed===a.action_id?`<small>${esc(proposedLabel)}</small>`:''}</summary><div class="detail-lines"><p><code>${esc(a.action_id)}</code></p><dl class="research-metrics"><dt>预期即时损失改善</dt><dd>${researchNumber(a.immediate_gain)}</dd><dt>当前查询成本</dt><dd>${researchNumber(a.current_cost)}</dd><dt>预计后续净价值</dt><dd>${researchNumber(a.continuation_net_value)}</dd><dt>预期净价值</dt><dd>${researchNumber(a.two_step_net_value)}</dd><dt>决策视野 horizon</dt><dd>${esc(text(a.horizon))}</dd></dl><p>${researchLink(a.url)}</p><p class="micro-note">研究模型估计，非事后收益；正文只在执行选择后披露。</p></div></details>`).join('');
}
function renderResearchCandidates(run){
  const g=run?.g;const ready=g?g.status==='research_ready'&&g.research_selection_allowed:research.catalog?.status==='ready';
  $('g-status').textContent=ready?'已训练 g · 研究模式可运行':text(g?.status,'等待研究目录');$('g-status').className=`small-badge ${ready?'success-badge':''}`;
  $('candidate-actions').innerHTML=run?`<p class="micro-note">${esc(strategyNames[g.strategy]||g.strategy)} · ${g.selection_authority==='fixed_order'?'当前采用固定顺序对照':'由研究 g 自主选择'} · ${esc(researchReason(g.stop_reason||g.decision_reason))}</p><p class="micro-note">候选标题 / 预期净价值（loss 单位）</p>${candidateMarkup(g.candidate_actions,g.proposed_action_id,g.selection_authority==='fixed_order'?'固定顺序下一篇':'g 当前提议')}`:'<div class="empty-inline">点击开始后，由后端计算当前候选评分。</div>';
}
function renderResearchTimeline(run){
  const terminal=run?.lifecycle.is_terminal;$('agent-status').className=`run-status ${terminal?'done':''}`;$('agent-status').dataset.state=run?.lifecycle.state||'INITIALIZED';$('agent-status').innerHTML=`<i></i>${terminal?'已停止':run?'已准备':'待开始'}`;
  $('agent-summary').textContent=run?`${strategyNames[run.g.strategy]||run.g.strategy} · 已读 ${run.budget.used_reads} / ${run.budget.max_reads} 篇 · 已用成本 ${researchNumber(run.budget.spent)} loss 单位`:'选择比赛，点击开始。g 自主选取文章，配套 f 重算后再决定继续或停止。';
  $('research-run-meta').textContent=run?`${run.run_id} · revision ${run.revision}`:'尚未创建研究运行';
  $('agent-outcome').hidden=!terminal;$('agent-outcome-label').textContent='结束原因';$('agent-reason').textContent=researchReason(run?.lifecycle.reason);$('agent-reason-code').textContent=run?.lifecycle.reason||'';
  $('agent-log-count').textContent=`${run?.steps.length||0} 次正文更新`;
  $('timeline').innerHTML=!run?'<div class="empty-inline">尚未执行模型。</div>':`<div class="research-initial"><b>初始 f 已计算</b><p>${researchVector(run.prediction.before)}</p><small>运行时间未返回；不将历史 cutoff 当作执行时间。</small></div>`+run.steps.map(s=>`<details class="research-step"><summary>第 ${s.step} 篇 · ${esc(s.body.title)}<small>${esc(fmtTime(s.executed_at))} UTC+8</small></summary><div class="detail-lines"><p>${esc(researchReason(s.decision.reason))}</p><p><b>f 执行：</b>${s.f_executed?'已执行':'未确认'} · 累计可建模证据 ${esc(text(s.modeled_evidence_count))} 条</p><p>before ${researchVector(s.prediction_before)}</p><p>after ${researchVector(s.prediction_after)}</p><details><summary>当时完整候选评分</summary>${candidateMarkup(s.decision.candidates,s.action_id,run.g.selection_authority==='fixed_order'?'固定顺序选取':'g 当时选取')}</details><details><summary>披露正文与消费链</summary>${bodyMarkup(s)}</details></div></details>`).join('');
  $('agent-next-step').textContent=terminal?'本轮为终态。可复核正文与报告；比较策略时新建独立运行。':run?'自动继续或逐步演示均由后端选文章。刷新仅恢复状态。':'默认开始后自动运行至停止；高级选项可启用逐步演示。';
}
function bodyMarkup(s){return `<p>${researchLink(s.body.url)}</p><pre class="research-body">${esc(s.body.text)}</pre><p>披露输入 <code>${esc(s.revealed_input_id)}</code></p><p>正文 SHA <code>${esc(s.body.sha256)}</code></p><details><summary>已归档语义与 f 使用输入</summary><p class="micro-note">语义来自归档审阅；本次没有重新调用 LLM。计数为累计可建模证据数，不是此篇新增数。</p><pre>${esc(jsonPreview({claims:s.claims,resolution:s.resolution,used_input_ids:s.used_input_ids,f_executed:s.f_executed,modeled_evidence_count:s.modeled_evidence_count}))}</pre></details>`;}
function renderResearchEvidence(run){
  $('evidence-list').innerHTML=run?.steps.length?run.steps.map(s=>`<article class="evidence-card"><div class="evidence-main"><div class="evidence-top"><span class="source-badge">第 ${s.step} 篇 · 选择后披露</span><span class="evidence-state verified">${s.f_executed?'已调用 f':'f 执行未确认'}</span></div><h3>${esc(s.body.title)}</h3><p class="evidence-meta">执行 ${esc(fmtTime(s.executed_at))} UTC+8 · 累计可建模证据 ${esc(text(s.modeled_evidence_count))} 条</p></div><details class="evidence-details"><summary>展开已披露正文、归档语义与消费链</summary><div class="detail-lines">${bodyMarkup(s)}</div></details></article>`).join(''):'<div class="empty-module"><strong>尚未披露正文</strong><p>未选文章的正文不作为 Agent 已知信息展示；初始 f 输出可在开始后查看。</p></div>';
}
function renderResearchChecks(run){
  const f=run?.f;const rows=[['研究范围','历史重建 / 未授权任意新比赛在线运行'],['f 实际执行',f?(f.executed_this_run?'是':'未确认'):'尚未开始'],['f 消费正文',f?`${f.consumed_body_input_ids.length} 篇`:'尚未开始'],['累计可建模证据',text(f?.modeled_evidence_count,'未返回')],['g 校准',text(run?.g.calibration_status||research.catalog?.calibration_status)],['语义与网络','归档语义 / 无外部请求']];
  $('status-checks').innerHTML=rows.map(([a,b])=>`<div class="check-row"><span class="check-icon unknown">·</span><span>${esc(a)}</span><b>${esc(b)}</b></div>`).join('');
  if(run)$('models-detail').innerHTML+=`<details class="research-models"><summary>研究 f / g 版本与绑定</summary><pre>${esc(jsonPreview({f:run.f,g:{model_version:run.g.model_version,status:run.g.status,selection_authority:run.g.selection_authority,research_selection_allowed:run.g.research_selection_allowed,online_selection_allowed:run.g.online_selection_allowed,calibration_status:run.g.calibration_status},model_bindings:run.model_bindings}))}</pre><p class="micro-note">本入口研究 f 与近期真实实验 shadow f 是不同工件。</p></details>`;
}
function renderResearchReport(){
  const e=research.report?.evaluation;if(!e){$('research-evaluation').textContent='报告尚未返回停止后的评估。';return;}
  const change=!finite(e.loss_reduction)?'损失变化未知':e.loss_reduction>0?`损失降低 ${researchNumber(e.loss_reduction)}`:e.loss_reduction<0?`损失增加 ${researchNumber(-e.loss_reduction)}`:'损失不变';
  $('research-evaluation').innerHTML=`<div class="report-change">${change}</div><dl class="research-metrics"><dt>归档赛果（仅事后）</dt><dd>${esc(['主胜','平局','客胜'][e.outcome]||'未知')}</dd><dt>初始损失</dt><dd>${researchNumber(e.loss_before)}</dd><dt>最终损失</dt><dd>${researchNumber(e.loss_after)}</dd><dt>事后净价值</dt><dd>${researchNumber(e.net_realized_value)}</dd></dl><p class="micro-note">历史重建诊断；不进入查询前模型输入，也不用于前瞻训练。本轮不证明在线泛化或 g 优于固定顺序；请保留全部策略与负收益案例。</p>`;
}
