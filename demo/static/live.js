/* View-only live console. Selection and replay never launch an experiment. */
'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const terminal = new Set(['STOPPED','FAILED','CANCELLED','INTERRUPTED']);
const scheduleContract = 'footballmind-daily-schedule-ui-v1';
const majorCompetitions = new Set(['Premier League','西甲','意甲','德甲','法甲','欧冠','欧罗巴','欧协联','葡超','巴甲','英冠','英甲','英乙','法乙','瑞超','日职','日乙','解放者杯','南美杯','墨超','美职联','沙特联','沙甲','荷甲','中超','中甲','西乙','意乙','土超','苏超','比甲','阿甲','K联赛','亚冠精英','亚冠二级']);
let catalog=null, selected=null, run=null, busy=false, polling=false, lastCatalogPoll=0;
let replayIndex=null, stopping=false, filteredMatches=[], keyboardIndex=-1, listFrame=0;
let catalogReceivedAt=0, renderedRunKey=null, listSignature='', selectionEpoch=0, viewRequestPending=false, loadFailed=false;
let frozenMarketComparison=null;
let configuredBudgetMode=null;
const ROW_HEIGHT=148, VIRTUAL_THRESHOLD=80, htmlCache=new WeakMap();
const compatibleSchedule = () => catalog?.schedule_ui_contract === scheduleContract;
const postreadMode = () => FootballMindProgress.isPostread(run||catalog);
const candidateKey = row => FootballMindProgress.candidateKey(row);
const sameCandidate = (row,id) => FootballMindProgress.sameCandidate(row,id);
const selectedPostreadHead = () => run?.model_name||$('postread-model')?.value||catalog?.default_model_name||null;
const postreadHeadType = () => selectedPostreadHead()==='fixed_order'?'fixed':selectedPostreadHead()==='state_ridge'?'state':'body';
const decisionRows = d => d?.candidates||d?.candidate_scores||[];
const decisionAction = d => d?.action_id||d?.selected_candidate_id||d?.selected_document_id||null;
const candidatePoolRow = id => (run?.candidate_pool||[]).find(c=>sameCandidate(c,id))||null;
const postreadScore = row => Number.isFinite(row?.predicted_gain)?row.predicted_gain:Number.isFinite(row?.score)?row.score:Number.isFinite(row?.immediate_gain)?row.immediate_gain:null;
const active = () => !!run && !terminal.has(run.state);
const stateText = v => ({scheduled:'未开赛',live:'进行中',finished:'已结束',postponed:'延期',cancelled:'取消',suspended:'中断',unknown:'待确认'}[v] || '待确认');
const phaseRank = v => ({scheduled:0,postponed:1,unknown:1,live:2,suspended:2,finished:3,cancelled:4}[v] ?? 1);
const runText = v => ({PREPARING:'启动核验中',RUNNING:'运行中',STOPPED:'本轮已结束',FAILED:'本轮失败',CANCELLED:'已停止',INTERRUPTED:'运行中断'}[v] || v || '未启动');
const reasonText = v => ({budget_exhausted:'正文读取预算已用完',predicted_gain_below_frozen_threshold:'最高预计收益未超过冻结阈值',pool_exhausted:'已编码候选已处理完毕',candidate_pool_exhausted:'已编码候选已处理完毕',no_encoded_candidate_bodies:'没有可供全文模型使用的已编码正文',positive_postread_gain:'预计边际收益超过冻结阈值',no_encoded_candidates:'尚无编码成功的正文候选',encoder_unavailable:'本地全文编码器未就绪',encoding_failed:'本地全文编码失败',semantic_budget_exhausted:'语义请求预算已用完；保留此前有效 f 输出，未再读取无法处理的正文',no_feasible_candidate:'当前候选已处理完毕',discovery_sources_failed:'来源发现请求全部失败，g 尚未获得候选；请查看来源请求错误',discovery_incomplete_no_candidates:'部分来源请求失败，其余结果未形成候选；覆盖不完整',discovery_no_candidates:'来源返回成功，但未找到可准入候选；g 尚未评分',positive_learned_two_step_net_value:'模型估计当前与后续净价值为正',nonpositive_net_value:'模型估计净价值不为正',nonpositive_learned_two_step_net_value:'模型估计当前与后续净价值不为正'}[v] || v || '未记录原因');
const filterText = v => ({provider_empty:'搜索服务返回 0 条',filtered_empty:'搜索有返回，投影后无可用链接',returned:'搜索已返回链接',parse_error:'响应无法解析',not_run:'尚未过滤'}[v] || v || '未记录');
const timestamp = v => v ? Date.parse(v) : NaN;
const time = v => Number.isFinite(timestamp(v)) ? new Date(v).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false})+' 北京时间' : '未知';
const dayKey = v => Number.isFinite(timestamp(v)) ? new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(v)) : '日期未知';
const shortTime = v => Number.isFinite(timestamp(v)) ? new Date(v).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}) : '时间未知';
const pct = v => Number.isFinite(v) ? (v*100).toFixed(2)+'%' : '—';
const num = v => Number.isFinite(v) ? (v<0 ? '−' : '')+Math.abs(v).toFixed(6) : '未记录';
const validProbability = p => Array.isArray(p) && p.length===3 && p.every(v=>Number.isFinite(v)&&v>=0&&v<=1) && Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-6;
const asProbabilities = p => validProbability(p) ? p : p && validProbability([p.home,p.draw,p.away]) ? [p.home,p.draw,p.away] : null;
const safeUrl = v => {try{const u=new URL(v);return ['https:','http:'].includes(u.protocol)&&!u.username&&!u.password ? u.href : '#';}catch{return '#';}};
const sourceName = v => ({espn:'ESPN',spftll:'足彩',qiumibao:'直播吧公开赛程',kleague_official:'K 联赛官方'}[v] || v || '未标明');
const sourcesOf = m => (m?.sources || [m?.source || 'spftll']).map(sourceName).join(' / ');
const tonePill = (text,tone='') => '<span class="pill '+(tone?'tone-'+tone:'')+'">'+esc(text)+'</span>';
const displayName = m => m ? (m.home_team||m.home||'主队未知')+' vs '+(m.away_team||m.away||'客队未知') : '';
const linkHtml = (url,label) => safeUrl(url)==='#' ? esc(label||url||'来源 URL 未记录') : '<a href="'+esc(safeUrl(url))+'" target="_blank" rel="noopener noreferrer">'+esc(label||url)+'</a>';
const startLeadMinutes = () => typeof catalog?.minimum_start_lead_minutes==='number'&&Number.isFinite(catalog.minimum_start_lead_minutes)&&catalog.minimum_start_lead_minutes>=0?catalog.minimum_start_lead_minutes:30;
const currentServerTime = () => Number.isFinite(timestamp(catalog?.server_time)) ? timestamp(catalog.server_time)+Math.max(0,Date.now()-catalogReceivedAt) : Date.now();
function notice(message){$('notice').textContent=message||'';$('notice').hidden=!message;}
function replaceHtml(node,html){
 if(htmlCache.get(node)===html)return;
 const open=new Set([...node.querySelectorAll('details[data-key][open]')].map(d=>d.dataset.key));
 node.innerHTML=html;htmlCache.set(node,html);
 for(const d of node.querySelectorAll('details[data-key]'))if(open.has(d.dataset.key))d.open=true;
}
async function api(path,body){
 const r=await fetch('/api/live-g/'+path,{method:body===undefined?'GET':'POST',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
 const value=await r.json();if(!r.ok)throw Error(value.error?.message||value.error||value.message||r.status);return value;
}
function marketView(m=run?.match||selected){
 if(run?.run_id&&frozenMarketComparison?.run_id===run.run_id)return frozenMarketComparison.value;
 if(!run)frozenMarketComparison=null;
 // This panel is the saved catalogue comparison, not the fresh f initializer.
 const match=run?.match||m,init=run?.initialization?.market_initialization;
 let quote=match?.quote_snapshot,p=asProbabilities(quote?.no_vig_probabilities)||asProbabilities(quote?.probabilities);
 let at=match?.quote_observed_at||quote?.observed_at||match?.quote_catalog_observed_at||null;
 let basis=run?'run_catalogue_snapshot':'catalogue_snapshot',source=match?.analysis_source||match?.source;
 if(run&&!p){
  quote=init?.quote_snapshot||init?.quote;
  p=asProbabilities(init?.probabilities)||asProbabilities(run.prior);
  at=init?.source_observed_at||quote?.observed_at||null;
  basis='initialization_fallback';source=init?.source||source;
 }
 const provider=quote?.provider||quote?.bookmaker||(source==='espn'?'ESPN（报价方未标明）':source==='spftll'?'足彩（报价方未标明）':'报价方未标明');
 const stale=!run&&Number.isFinite(timestamp(at))&&currentServerTime()-timestamp(at)>30*60000;
 const future=!run&&Number.isFinite(timestamp(at))&&timestamp(at)>currentServerTime();
 const value={p:p?[...p]:null,at,provider,source,basis,stale,future,observed:!!match?.quote_observed||!!p};
 // Do not freeze an empty PREPARING response: a valid legacy initializer may arrive later.
 if(run?.run_id&&p)frozenMarketComparison={run_id:run.run_id,value};
 return value;
}
function experimentGate(){
 if(run){
  if(terminal.has(run.state)){const incomplete=bodyPipelineView(run).f_body_updates===0,scoredStop=postreadMode()&&FootballMindProgress.postreadCounts(run).g_scored>0&&run.state==='STOPPED';return {state:run.state,tone:['FAILED','INTERRUPTED'].includes(run.state)?'risk':incomplete&&!scoredStop?'warn':'neutral',title:incomplete&&!scoredStop?'未完成正文分析':runText(run.state),description:reasonText(run.reason)+(incomplete?scoredStop?'；全文已评分，本轮没有采用新正文，保留初始 f 输出。':'；未形成正文交给 f 的有效更新。':''),action:'report',label:'查看完整运行报告'};}
  return {state:run.state,tone:'active',title:stopping?'正在停止':runText(run.state),description:stopping?'等待在途请求返回，停止后续处理。':run.stage||'正在处理真实运行。',action:'stop',label:stopping?'已请求停止':'停止本轮',disabled:stopping};
 }
 if(!catalog)return loadFailed?{state:'CONNECTION_FAILED',tone:'risk',title:'目录连接失败',description:'尚未取得比赛数据，可重新连接；当前没有启动实验。',action:'reload',label:'重新连接'}:{state:'CHECKING',tone:'neutral',title:'正在读取目录',description:'读取比赛与启动条件。',action:'reload',label:'正在读取',disabled:true};
 if(!compatibleSchedule())return {state:'INCOMPATIBLE',tone:'risk',title:'服务版本不匹配',description:'目录未声明每日赛程契约，请检查服务后重新连接。',action:'reload',label:'重新连接'};
 if(!selected)return {state:'SELECT',tone:'neutral',title:'请先选择比赛',description:'可在左侧查找球队、对阵或比赛 ID。',action:'choose',label:'查看可启动比赛'};
 const m=selected,q=marketView(m),why=m.availability_reason||m.reason||'';
 if(['live','finished','postponed','cancelled','suspended'].includes(m.status)||Number.isFinite(timestamp(m.kickoff))&&timestamp(m.kickoff)<=currentServerTime()+startLeadMinutes()*60000)return {state:'MATCH_UNAVAILABLE',tone:'warn',title:'本场不在启动时间范围',description:['live','finished','postponed','cancelled','suspended'].includes(m.status)?'来源状态为“'+stateText(m.status)+'”，本页仅启动赛前实验。':timestamp(m.kickoff)<=currentServerTime()?'记录的开球时间已过；目录仍显示未开赛，需刷新来源状态。':'距离开球不足 '+startLeadMinutes()+' 分钟，已超出本入口的启动时间范围。',action:'choose',label:'查看可启动比赛'};
 if(!Number.isFinite(timestamp(m.kickoff))||q.future)return {state:'INVALID_CLOCK',tone:'warn',title:'来源时间需要复核',description:q.future?'报价观察时间晚于服务时钟，不能确认数据新鲜度。':'开球时间缺失或无法解析。',action:'refresh',label:catalog.refreshing?'正在重新检查':'重新检查数据',disabled:!!catalog.refreshing};
 if(q.stale||/超过 30 分钟|过期|等待刷新/.test(why))return {state:'STALE',tone:'warn',title:'市场先验需要刷新',description:why||'报价缓存已超过 30 分钟，当前数字仅供查看。',action:'refresh',label:catalog.refreshing?'正在刷新先验':'刷新市场先验',disabled:!!catalog.refreshing};
 if(m.can_start!==true)return {state:'BLOCKED',tone:'warn',title:'启动条件待补齐',description:why||'来源身份或完整 90 分钟报价尚未就绪。',action:'refresh',label:catalog.refreshing?'正在重新检查':'重新检查数据',disabled:!!catalog.refreshing};
 if(postreadMode()&&catalog.encoder_ready!==true)return {state:'ENCODER_NOT_READY',tone:'risk',title:'本地全文编码尚未就绪',description:catalog.encoder_reason||catalog.encoder_status||'后端尚未确认 Qwen 编码器可用；不会以其他模型替代或伪造向量。',action:'reload',label:'检查编码状态'};
 if(!postreadMode()&&!catalog.provider_configured)return {state:'PROVIDER_MISSING',tone:'risk',title:'语义服务尚未就绪',description:'执行进程尚未配置语义服务凭据。配置完成后重新检查，当前不会启动比赛。',action:'reload',label:'检查服务状态'};
 const articleCount=m.article_source_count;
 const articleReadiness=Number.isInteger(articleCount)&&articleCount>0?'已注册 '+articleCount+' 个文章目录入口，尚不代表本场正文已取得。':m.article_source_status==='search_only'||articleCount===0?'本场尚无注册的球队文章目录，目前依赖公开搜索；正文获取尚未验证。':'本场文章来源配置尚未核验，不能仅凭赛程和赔率确认正文可取得。';
 return {state:'READY',tone:'good',title:'身份与先验可启动',description:articleReadiness+' 开始后重新核对身份与报价，再发现和获取候选正文。',action:'start',label:'开始自主分析'};
}
function controls(){
 const gate=experimentGate(),b=$('start');
 $('experiment-status').textContent=gate.title;$('experiment-status').dataset.tone=gate.tone;$('readiness').dataset.tone=gate.tone;
 $('readiness-title').textContent=gate.title;$('readiness-description').textContent=gate.description;
 $('readiness-dot').style.backgroundColor=({good:'var(--accent)',warn:'var(--amber)',risk:'var(--risk)',active:'#28658c'}[gate.tone]||'var(--muted)');
 b.hidden=gate.action==='report';b.dataset.action=gate.action;b.textContent=busy?'正在处理…':gate.label;b.disabled=busy||!!gate.disabled||viewRequestPending;
 $('download').hidden=!run||gate.action!=='report';if(run)$('download').href='/api/live-g/runs/'+encodeURIComponent(run.run_id)+'/report';
 $('new').hidden=!run||!terminal.has(run.state);$('new').disabled=busy||viewRequestPending;
 $('refresh').disabled=busy||active()||!!catalog?.refreshing||!!catalog&&!compatibleSchedule();
 $('refresh').title=active()?'本轮运行中，结束后可刷新赛程':catalog?.refreshing?'正在刷新来源数据':'刷新赛程与市场先验';
 $('postread-model').disabled=!!run||busy;$('budget').disabled=!!run||busy;$('cost').disabled=!!run||busy;
}
function preflightChecks(){
 const m=run?.match||selected,q=marketView(m),identity=run?.identity;
 const checks=[
 [compatibleSchedule()?'pass':'fail','每日赛程接口',compatibleSchedule()?'契约兼容':'尚未取得兼容目录'],
 [m?'pass':'unknown','比赛选择',m?.match_id||'尚未选择'],
 [run?identity?'pass':'unknown':m?.supported?'pass':'unknown','来源身份',identity?identity.independently_confirmed?'独立官方核验已记录':'来源 ID 已匹配；独立官方核验未确认':m?.supported?'目录身份完整；启动时重新匹配':'等待来源确认'],
 [q.p||m?.quote_observed?'pass':'unknown','完整市场先验',q.p?'已记录胜 / 平 / 负概率':m?.quote_observed?'已观察完整报价；启动后初始化':'未发现完整 90 分钟报价'],
 [run?q.at?'pass':'unknown':q.stale||q.future?'fail':q.at?'pass':'unknown','报价观察时钟',q.at?time(q.at)+(q.future?' · 晚于服务时钟':q.stale?' · 已过期':run?' · 本轮冻结版本':' · 30 分钟内'):'未记录独立报价观察时间'],
 postreadMode()?[catalog?.encoder_ready===true?'pass':'fail','本地正文编码',catalog?.encoder_ready===true?'后端确认 Qwen 编码器可用；新路径不要求 DeepSeek key':'后端尚未确认编码器可用']:[catalog?.provider_configured?'pass':'fail','语义服务',catalog?.provider_configured?'服务已配置':'执行进程配置待完成']];
 if(!run){
  const n=m?.article_source_count;
  checks.push(['unknown','文章来源',Number.isInteger(n)&&n>0?'已注册 '+n+' 个目录入口；不等于已取得或验证本场正文':m?.article_source_status==='search_only'||n===0?'尚无注册的球队文章目录，当前依赖公开搜索；正文获取尚未验证':'来源配置尚未核验']);
  checks.push(['unknown','启动复核','开始后重新获取报价与核验身份，当前为目录准入结果']);
 }
 $('preflight-details').hidden=!!run;
 replaceHtml($('preflight-checks'),checks.map(([status,title,desc])=>'<li><span class="check-symbol '+status+'">'+(status==='pass'?'✓':status==='fail'?'!':'○')+'</span><span><b>'+esc(title)+'</b> · '+esc(desc)+'</span></li>').join(''));
}
function matchRow(m,i){
 const score=Number.isFinite(m.home_score)&&Number.isFinite(m.away_score)?' '+m.home_score+' : '+m.away_score:'';
 return '<button type="button" role="option" tabindex="-1" id="match-option-'+i+'" aria-selected="'+(selected?.match_id===m.match_id)+'" aria-posinset="'+(i+1)+'" aria-setsize="'+filteredMatches.length+'" class="match-row '+(selected?.match_id===m.match_id?'selected ':'')+(i===keyboardIndex?'keyboard-focus':'')+'" data-match="'+esc(m.match_id)+'"><span class="match-row-head"><span>'+esc(m.competition_key||'赛事未知')+'</span>'+tonePill(stateText(m.status)+score,m.status==='live'?'warn':'')+'</span><strong>'+esc(displayName(m))+'</strong><time datetime="'+esc(m.kickoff)+'">'+esc(shortTime(m.kickoff))+' · 北京</time><span class="match-row-note '+(m.can_start?'ready':'')+'">'+esc(m.can_start?'可启动 · '+sourcesOf(m):m.availability_reason||m.reason||'启动前核验')+'</span></button>';
}
function renderMatchWindow(){
 const list=$('matches'),n=filteredMatches.length;
 if(!n){list.innerHTML='<p class="list-empty">当前条件下没有比赛。可切换日期、清除筛选，或输入球队和 ID 跨范围查找。</p>';list.removeAttribute('aria-activedescendant');return;}
 const virtual=n>VIRTUAL_THRESHOLD,first=virtual?Math.max(0,Math.floor(list.scrollTop/ROW_HEIGHT)-3):0,end=virtual?Math.min(n,first+Math.ceil((list.clientHeight||420)/ROW_HEIGHT)+7):n;
 list.innerHTML='<div class="list-spacer top-spacer" aria-hidden="true"></div>'+filteredMatches.slice(first,end).map((m,i)=>matchRow(m,first+i)).join('')+'<div class="list-spacer bottom-spacer" aria-hidden="true"></div>';
 list.querySelector('.top-spacer').style.height=(first*ROW_HEIGHT)+'px';list.querySelector('.bottom-spacer').style.height=((n-end)*ROW_HEIGHT)+'px';
 if(keyboardIndex>=first&&keyboardIndex<end)list.setAttribute('aria-activedescendant','match-option-'+keyboardIndex);else list.removeAttribute('aria-activedescendant');
}
function renderCatalog(){
 if(!compatibleSchedule()){$('catalog-status').textContent='尚未取得兼容的每日赛程接口。';$('filter-count').textContent='目录未就绪';$('source-status').textContent='连接失败或后端版本不匹配，请重新连接。';filteredMatches=[];renderMatchWindow();controls();return;}
 const matches=catalog.matches||[],coverage=catalog.coverage||{},competition=$('competition').value;
 $('coverage-total').textContent=coverage.discovered??matches.length;$('coverage-ready').textContent=coverage.can_start??matches.filter(m=>m.can_start).length;$('coverage-leagues').textContent=coverage.competition_count??new Set(matches.map(m=>m.competition_key).filter(Boolean)).size;
 $('catalog-status').textContent=(catalog.refreshing?'正在更新来源数据；已保存比赛仍可浏览。':'最近检查 '+time(catalog.last_checked_at||catalog.updated_at))+(catalog.auto_refresh_seconds?' · 页面打开时自动检查':'');
 $('source-status').textContent=Object.entries(catalog.sources||{}).map(([k,v])=>sourceName(k)+'：'+({ready:'正常',partial:'部分使用缓存',failed:'请求失败，保留缓存'}[v.status]||v.status)).join('；')+(catalog.status==='refresh_failed'?'；当前使用上次成功目录':'');
 const competitions=coverage.competitions||[...new Set(matches.map(m=>m.competition_key).filter(Boolean))].map(name=>({name}));
 replaceHtml($('competition'),'<option value="">全部赛事</option>'+competitions.map(c=>'<option value="'+esc(c.name)+'">'+esc(c.name)+(c.discovered!==undefined?' · '+c.discovered+' 场':'')+'</option>').join(''));
 if([...$('competition').options].some(o=>o.value===competition))$('competition').value=competition;
 const terms=$('match-search').value.trim().toLowerCase().split(/\s+/).filter(Boolean),searching=!!terms.length,availability=$('availability').value,period=$('date-range').value,scope=$('competition-scope').value;
 const today=dayKey(new Date(currentServerTime())),tomorrow=dayKey(new Date(currentServerTime()+86400000));
 filteredMatches=matches.filter(m=>{
  const text=[m.home,m.away,m.home_team,m.away_team,m.competition_key,m.match_id,m.fixture_id,m.source_match_id,m.espn_match_id,m.physical_match_id,m.source_url].filter(Boolean).join(' ').toLowerCase();
  if(searching)return terms.every(t=>text.includes(t));
  return (scope==='all'||$('competition').value||majorCompetitions.has(m.competition_key)||m.sources?.includes('spftll'))&&(!$('competition').value||m.competition_key===$('competition').value)&&(period==='week'||dayKey(m.kickoff)===(period==='today'?today:tomorrow))&&(availability==='all'||(availability==='ready'?m.can_start:!m.can_start));
 });
 filteredMatches.sort((a,b)=>phaseRank(a.status)-phaseRank(b.status)||timestamp(a.kickoff)-timestamp(b.kickoff)||String(a.match_id).localeCompare(String(b.match_id)));
 $('search-mode').textContent=searching?'全目录搜索 · 已忽略日期、赛事和启动状态筛选。':'搜索可跨日期和赛事；↓ 浏览，Enter 选择。';
 $('filter-count').textContent=(searching?'搜索 ':'当前 ')+filteredMatches.length+' 场 · 可启动 '+filteredMatches.filter(m=>m.can_start).length;
 const filters=Number(!!$('competition').value)+Number(availability!=='all');$('active-filter-count').textContent=filters?'（'+filters+' 项）':'';
 const signature=JSON.stringify([terms,period,scope,$('competition').value,availability,filteredMatches.map(m=>m.match_id)]);
 if(signature!==listSignature){listSignature=signature;$('matches').scrollTop=0;keyboardIndex=-1;}renderMatchWindow();
 const rows=catalog.runs||[];$('history-count').textContent=rows.length?'· '+rows.length:'';
 replaceHtml($('history'),rows.map(r=>'<button type="button" class="history-item" data-run="'+esc(r.run_id)+'">'+esc(displayName(r.match))+'<small>'+esc(runText(r.state))+' · '+esc(shortTime(r.created_at))+'</small></button>').join('')||'<p class="micro-note">尚无在线运行记录。</p>');
 $('connection').textContent=['partial','refresh_failed'].includes(catalog.status)?'接口已连接 · 部分缓存':'研究接口已连接';$('connection-dot').className='online';controls();
}
function chooseMatch(m,{focus=false}={}){
 if(!m)return;if(active()||busy){notice('本轮仍在执行，请结束后切换比赛。');return;}
 selectionEpoch++;run=null;selected=m;replayIndex=null;renderedRunKey=null;stopping=false;notice('');history.replaceState(null,'','/live.html');renderMatchWindow();renderRun();if(focus)$('fixture').focus({preventScroll:true});
}
function setRunUrl(){history.replaceState(null,'','/live.html?run='+encodeURIComponent(run.run_id));}
function deltaText(before,after){if(!Number.isFinite(before)||!Number.isFinite(after))return '未记录差值';const d=(after-before)*100;return Math.abs(d)<1e-10?'保持不变':(d>0?'↑ ':'↓ ')+Math.abs(d).toFixed(3)+' 个百分点';}
function probabilityCards(p,before=null){return ['主胜','平局','客胜'].map((n,i)=>'<div class="prob"><span class="prob-label">'+n+'</span><div class="number">'+pct(p?.[i])+'</div>'+(before?'<span class="prob-delta">'+deltaText(before[i],p?.[i])+'</span>':'')+'</div>').join('');}
function pairTable(before,after){
 if(!validProbability(before)||!validProbability(after))return '<p>before / after 记录不完整，无法展示差值。</p>';
 return '<table class="pair-table"><thead><tr><th>结果</th><th>更新前</th><th>更新后</th><th>变化</th></tr></thead><tbody>'+['主胜','平局','客胜'].map((n,i)=>'<tr><td>'+n+'</td><td>'+pct(before[i])+'</td><td>'+pct(after[i])+'</td><td>'+deltaText(before[i],after[i])+'</td></tr>').join('')+'</tbody></table>';
}
function evidenceHighlights(s){
 if(Array.isArray(s.evidence_highlights))return s.evidence_highlights.map(h=>({...h,concept:h.concept??s.claims?.[h.claim_index]?.concept??null}));
 const body=String(s.body||'');return (s.claims||[]).flatMap((c,i)=>(c.quotes?.length?c.quotes:[null]).map(q=>{const start=q?body.indexOf(q):-1;return {claim_index:i,entity:c.entity,team:c.team,kind:c.kind,scope:c.scope,value:c.value,concept:c.concept??null,statement:c.statement,quote:q,quote_verified:start>=0,body_start:start>=0?start:null,body_end:start>=0?start+q.length:null,body_context:start>=0?body.slice(Math.max(0,start-120),Math.min(body.length,start+q.length+120)):null};}));
}
function evidenceHighlightHtml(h){
 const priority=h.importance_label==='高'?'high':h.importance_label==='中'?'medium':'low';
 return '<div class="evidence-highlight"><div>'+(h.importance_label?'<span class="evidence-priority evidence-priority-'+priority+'">展示优先级 '+esc(h.importance_label)+'</span> ':'')+'<strong>'+esc(h.entity||'未标注实体')+'</strong> · '+esc(h.kind||'unknown')+' · '+esc(h.scope||'unknown')+' · '+esc(h.value||'')+(h.concept?' · 证据类型 '+esc(h.concept):'')+'</div>'+(h.statement?'<p>'+esc(h.statement)+'</p>':'')+(h.quote?'<blockquote>'+esc(h.quote)+'</blockquote>':'<p>响应未记录可核验引句。</p>')+'<div class="micro-note">'+(h.quote_verified?'原文引句已定位（不等同独立事实核验）':'引句未定位')+(h.body_start!=null?' · 正文偏移 '+h.body_start+'–'+h.body_end:'')+(h.importance_reasons?.length?' · '+esc(h.importance_reasons.join('、')):'')+'</div>'+(h.body_context?'<details><summary>引句上下文</summary><pre>'+esc(h.body_context)+'</pre></details>':'')+'</div>';
}
function renderPrediction(){
 const m=run?.match||selected,baseline=marketView(m),p=asProbabilities(run?.current),before=asProbabilities(run?.before),steps=(run?.steps||[]).filter(s=>s.f_executed===true),pipeline=bodyPipelineView(run);
 replaceHtml($('market-probabilities'),probabilityCards(baseline.p));
 $('baseline-title').textContent=run?'启动市场对比':'目录市场概率';
 $('market-source').textContent=baseline.p?'报价标识 '+baseline.provider+' · 来源 '+sourceName(baseline.source)+' · '+(baseline.basis==='initialization_fallback'?'旧记录未保存目录概率，回退展示本轮初始化报价；不代表启动前目录快照。':run?'启动时保存的目录快照；固定对比值，不随 f 更新。':'目录报价归一化概率；启动后作为固定对比值。'):m?.quote_observed?'目录记录未附可展示概率；不补造数值，等待本轮初始化报价。':m?'尚无可展示的完整胜 / 平 / 负市场先验。':'尚未选择比赛。';
 $('market-time').textContent='报价观察时间：'+time(baseline.at);
 $('market-freshness').textContent=run?baseline.p?baseline.basis==='initialization_fallback'?'已固定 · 初始化回退':'已固定 · 启动目录快照':'快照待补齐':baseline.future?'时钟异常 · 待核验':baseline.stale?'已过期 · 仅供查看':Number.isFinite(timestamp(baseline.at))?'观察时间在 30 分钟内':'观察时间未知';
 $('market-freshness').className='pill '+(baseline.stale||baseline.future?'tone-warn':run?'':Number.isFinite(timestamp(baseline.at))?'tone-good':'');
 const initialization=run?.initialization?.market_initialization,actualPrior=asProbabilities(run?.prior)||asProbabilities(initialization?.probabilities);
 $('market-initialization').hidden=!run;
 $('market-initialization').textContent=!run?'':actualPrior?'本轮 f 实际初始化市场概率：'+actualPrior.map(pct).join(' / ')+'。'+(baseline.p&&actualPrior.some((v,i)=>Math.abs(v-baseline.p[i])>1e-8)?'与上方启动对比值不同；f 使用本轮初始化值，报价变化不计为正文更新。':'上方对比值不会随正文更新改变。')+' 初始化报价观察时间：'+time(initialization?.source_observed_at||initialization?.quote_snapshot?.observed_at||initialization?.quote?.observed_at):'本轮 f 初始化尚未完成；上方仅展示已保存的启动对比快照。';
 $('model-empty').hidden=!!p;$('probabilities').hidden=!p;replaceHtml($('probabilities'),p?probabilityCards(p,steps.length?before:null):'');
 $('model-output-status').textContent=!p?'尚未生成':['FAILED','INTERRUPTED'].includes(run.state)?'保留最后有效输出':steps.length?'正文更新 '+steps.length+' 次':'初始 f 输出';$('model-output-status').className='pill '+(p?'tone-good':'');
 $('model-empty-title').textContent=run?active()?'等待 f 计算':'本轮未产生模型输出':'等待自主分析';
 $('model-empty-description').textContent=run?active()?'启动核验完成后生成初始 f 预测。':'请查看本轮失败或停止原因；外部市场基线仍单独展示。':'当前没有 Agent 预测。开始运行后，在这里查看 f 的实际输出与正文更新。';
 $('forecast').textContent=p?'输出时间 '+time(run.forecast_at)+' · '+(steps.length?'对比本轮初始 f 概率':'尚无正文更新'):'';
 $('prediction-note').textContent=!p?'':!steps.length?'当前仅有 f 的初始计算，尚未完成正文更新。':'正文已由 f 执行推理；概率变化仅表示模型响应，预测改善需独立赛果评估。';
 $('body-pipeline').textContent=!run?'开始后展示正文获取、语义验证与 f 实际采用的数量。':'正文管线：已取得 '+pipeline.body_documents+' 篇 → 语义请求 '+pipeline.semantic_requests+' 次 → f 实际采用 '+pipeline.f_body_updates+' 篇 → 概率变化 '+pipeline.changed_updates+' 次。'+(pipeline.semantic_rejected_bodies?' '+pipeline.semantic_rejected_bodies+' 篇语义校验失败，未进入 f；失败不记为零收益。':'')+(pipeline.modeled_claims===null?' 本记录未保存可建模陈述计数。':' 可建模陈述 '+pipeline.modeled_claims+' 条。')+(Number.isFinite(pipeline.market_difference_l1)?' 与本轮 f 初始化市场概率的 L1 差异 '+pipeline.market_difference_l1.toFixed(6)+'。':'');
 if(postreadMode()){const c=FootballMindProgress.postreadCounts(run);$('body-pipeline').textContent=run?'全文管线：已取得 '+c.body_documents+' 篇 → 已编码 '+c.encoded+' 篇 → g 已评分 '+c.g_scored+' 个候选 → f 实际采用 '+c.f_adopted+' 篇。'+(c.body_failed?' 正文获取失败 '+c.body_failed+' 篇。':'')+(c.encoding_failed?' 本地编码失败 '+c.encoding_failed+' 篇，未进入 f。':'')+((run?.candidate_pool||[]).some(p=>p.status==='encoding_budget_skipped')?' 超编码预算 '+run.candidate_pool.filter(p=>p.status==='encoding_budget_skipped').length+' 篇：全文已保存，未送 g / f，不表示没有价值。':'')+' 新 f 直接使用正文语义向量，不需要 claim 抽取。':'开始后记录真实正文获取、本地 Qwen 编码、g 全文评分与 f 采用。';}
}
function bodyPipelineView(record){
 const steps=(record?.steps||[]).filter(s=>s.f_executed===true),saved=record?.body_pipeline||{};
 const bodyDocs=(record?.documents||[]).filter(d=>d.status==='OK'&&d.structured_news?.role==='article'&&String(d.text||'').trim());
 const changed=steps.filter(s=>{const p=asProbabilities(s.prediction_before),q=asProbabilities(s.prediction_after);return p&&q&&p.reduce((v,x,i)=>v+Math.abs(q[i]-x),0)>1e-12;}).length;
 const prior=asProbabilities(record?.prior),current=asProbabilities(record?.current);
 return {body_documents:Number.isFinite(saved.body_documents)?saved.body_documents:bodyDocs.length,
  semantic_requests:Number.isFinite(saved.semantic_requests)?saved.semantic_requests:record?.counts?.provider||0,
  claim_review_requests:Number.isFinite(saved.claim_review_requests)?saved.claim_review_requests:0,
  f_body_updates:steps.length,changed_updates:Number.isFinite(saved.changed_updates)?saved.changed_updates:changed,
  semantic_rejected_bodies:Number.isFinite(saved.semantic_rejected_bodies)?saved.semantic_rejected_bodies:(record?.body_rejections||[]).filter(r=>r.reason==='semantic_validation_failed').length,
  modeled_claims:Number.isFinite(saved.modeled_claims)?saved.modeled_claims:steps.length&&steps.every(s=>Number.isFinite(s.f_consumption?.modeled_claims))?steps.reduce((v,s)=>v+s.f_consumption.modeled_claims,0):steps.length?null:0,
  market_difference_l1:Number.isFinite(saved.market_difference_l1)?saved.market_difference_l1:prior&&current?prior.reduce((v,x,i)=>v+Math.abs(current[i]-x),0):null};
}
function consumptionReason(reason){return ({no_source_claims:'正文已消费，但未抽取到来源陈述',no_supported_match_or_context_claims:'陈述未进入当前 f 支持的本场或上下文特征',duplicate_or_no_new_encoded_features:'与已采用证据重复，或没有新增编码特征',no_discriminative_learned_residual:'已产生特征，但当前模型没有产生可区分的概率更新',learned_evidence_residual:'已采用证据产生学习得到的预测更新'}[reason]||reason||'未保存特征路由原因');}
function consumptionHtml(step){
 const a=step.f_consumption;if(!a)return '<p class="micro-note">旧记录未保存逐特征消费审计；可核对正文、引句与实际概率对。</p>';
 const review=step.claim_review,reviewHtml=review?'<p class="micro-note">初抽陈述 '+esc(step.extracted_claims_count)+' 条 · 来源语义复核保留 '+esc(review.retained_indices?.length??'未知')+' 条 · 未准入 '+esc(review.rejected?.length??0)+' 条。复核检查原文是否支持，不等同独立事实核验。</p>'+(review.rejected?.length?'<details class="compact-details"><summary>查看未准入的陈述及原因</summary><pre>'+esc(JSON.stringify(step.rejected_claims||review.rejected,null,2))+'</pre></details>':''):'';
 return '<div class="round-source"><span>正文采用：'+(a.body_consumed===true?'已由输入编号确认':'未确认')+'</span><span>来源陈述 '+esc(a.claims_count)+' 条 · 可建模 '+esc(a.modeled_claims)+' 条</span><span>非零模型特征 '+esc(a.nonzero_model_features)+' · 新变化特征 '+esc(a.new_nonzero_features)+'</span><span>本次概率 L1 '+esc(num(a.probability_l1_change))+'</span><span>'+esc(consumptionReason(a.reason))+'</span></div>'+reviewHtml;
}
function chosenCandidate(d){return d?.action_id?(d.candidates||[]).find(c=>c.action_id===d.action_id)||(run?.menus||[]).find(c=>c.action_id===d.action_id)||{action_id:d.action_id}:null;}
function displayedRound(){const ds=run?.decisions||[];return ds.length?Math.min(replayIndex??ds.length-1,ds.length-1):-1;}
function renderPipelineMode(){
 const modern=postreadMode(),known=!!(run||catalog),mode=modern?'postread':'legacy';
 const previousHead=$('postread-model').value,choices=catalog?.model_choices||[];
 if(modern){replaceHtml($('postread-model'),choices.map(c=>'<option value="'+esc(c.id)+'">'+esc(c.label||c.id)+'</option>').join(''));const head=run?.model_name||previousHead||catalog?.default_model_name;if(head&&[...$('postread-model').options].some(o=>o.value===head))$('postread-model').value=head;}
 $('postread-model-label').hidden=!modern;$('postread-model-note').hidden=!modern;
 const headType=postreadHeadType();
 $('postread-model-note').textContent=headType==='fixed'?'此对照按候选顺序采用正文，不执行 g 评分；f 与全文实验相同。':headType==='state'?'此对照的 g 使用当前状态，不使用候选正文向量区分文章；不是全文排序成功的证明。':'默认使用已冻结全文 MLP（seed27）；只作为研究候选，不宣称线上或跨联赛最优。';
 $('pipeline-badge').textContent=!known?'在线模型待核验':modern?headType==='fixed'?'在线正文 · 固定顺序对照':headType==='state'?'在线正文 · 状态 g 对照':'在线全文 · Qwen f / g':run?'归档回放 · 原语义模型':'在线采集 · 原语义模型';
 $('pipeline-description').textContent=!known?'等待后端声明实际模型模式。':modern?headType==='fixed'?'实时取得正文 → 本地 Qwen 编码 → 按固定顺序交给冻结 f。本轮不执行 g 选择。':headType==='state'?'实时取得正文 → 本地 Qwen 编码 → 状态 g 决定继续或停止 → 冻结 f 采用。状态 g 不以候选正文向量排序。':'实时发现并取得正文 → 本地 Qwen 编码 → 全文 g 评分 → 冻结 f 采用或停止。模型不在运行中训练。':'本记录使用原查询前 g 与语义 f，按实际旧流程展示，不冒充新版 Qwen 全文模型。';
 $('decision-formula').textContent=modern&&headType==='fixed'?'固定顺序是非 g 对照；本轮不计算文章价值分数。':modern?'g 估计候选正文在当前状态下带来的预测损失降低，按保存的冻结阈值选择或停止；实际收益仍需独立赛果。未计算后续价值时不显示 h。':'预期净价值 = 即时损失降低 − 当前查询成本 + 后续净价值。数值由本轮原 g 估计；实际收益需要独立赛果。';
 $('decision-input-note').textContent=modern?headType==='fixed'?'本轮固定顺序对照不执行 g，不输出虚构的候选分数。':headType==='state'?'本轮 g 只读取状态特征，不能把此对照称为全文排序；f 仍使用实际采用的正文向量。':'候选正文先取得并编码，g 读取当前状态与候选全文语义向量；f 仅采用被选中的正文。采集和编码成本已经发生。':'此记录在正文读取前评分；正文随后进入语义抽取与 f。';
 $('evidence-explanation').textContent=modern?'新 f 直接消费正文语义向量。以下只展示实际 f 执行的正文、编码绑定、采用编号与概率对；没有 claim 不表示抽取失败。':'可查看实际输入文本、原文引句和 before / after。概率变化仅表示模型响应，预测改善须由独立赛果评估。';
 if(modern&&run?.offline_replay===true){$('pipeline-badge').textContent='离线回放 · '+(headType==='fixed'?'固定顺序对照':headType==='state'?'状态 g 对照':'Qwen 全文模型');$('pipeline-description').textContent='既有来源归档经实际新模型运行后保存的离线回放；不是新的在线采集或前瞻轨迹。'+$('pipeline-description').textContent.replace('实时发现并取得正文','归档正文').replace('实时取得正文','归档正文');}
 $('usage-note').textContent=modern?'正文链接尝试与 f 采用分别计数；失败、空正文和重复正文也占尝试次数。本路径本地编码，不调用 DeepSeek；赛程及身份初始化请求另列。':'赛程刷新与身份初始化请求另见报告；此处展示原模型来源、语义与正文尝试计数。';
 if(modern&&run?.offline_replay===true)$('usage-note').textContent+=' 这里的来源次数为归档适配调用，不是新网络采集。实际新增 HTTP：'+(Number.isFinite(run.actual_new_http_requests)?run.actual_new_http_requests:'未记录')+'。步骤时间采用归档重放时钟。';
 $('cost-label').hidden=modern;$('cost-note').hidden=modern;$('budget-label').textContent=modern?'最多交给 f 的正文（1–20 篇）':'原流程最多读取正文';
 if(known&&configuredBudgetMode!==mode){if(!run)$('budget').value=modern?'20':'5';configuredBudgetMode=mode;}
 $('postread-pool-card').hidden=!modern;
 $('counts').classList.toggle('postread-usage',modern);
}
function postreadChoice(d){
 const id=decisionAction(d);if(!id)return null;
 return {...(candidatePoolRow(id)||{}),...(decisionRows(d).find(c=>sameCandidate(c,id))||{}),action_id:id};
}
function postreadBodyRecord(row){
 const id=row?.document_id||row?.input_id||candidateKey(row),pool=candidatePoolRow(candidateKey(row))||candidatePoolRow(id)||{};
 const doc=(run?.documents||[]).find(d=>sameCandidate(d,pool.document_id||id))||{};
 return {...doc,...pool,...row,body:row?.body||row?.text||pool.body||pool.text||doc.text||'',url:row?.url||row?.source_url||pool.url||pool.source_url||doc.url};
}
function postreadBindingHtml(row){
 const e=postreadBodyRecord(row);
 return '<dl class="postread-bindings"><dt>正文编号</dt><dd>'+esc(e.document_id||'未记录')+'</dd><dt>实际输入编号</dt><dd>'+esc(e.input_id||e.text_sha256||'未记录')+'</dd><dt>正文 SHA</dt><dd>'+esc(e.text_sha256||'未记录')+'</dd><dt>原始响应 SHA</dt><dd>'+esc(e.raw_sha256||'未记录')+'</dd><dt>向量 SHA</dt><dd>'+esc(e.vector_sha256||e.encoding?.vector_file_sha256||'未记录')+'</dd><dt>声明发布时间</dt><dd>'+esc(time(e.published_at))+'</dd><dt>本版本取得时间</dt><dd>'+esc(time(e.source_available_at))+'</dd><dt>编码开始 / 完成</dt><dd>'+esc(time(e.encoding_started_at))+' / '+esc(time(e.encoding_completed_at))+'</dd></dl><details class="compact-details"><summary>本轮模型及编码器绑定</summary><pre>'+esc(JSON.stringify(run?.model_bindings||{},null,2))+'</pre></details>';
}
function renderPostreadPool(){
 const pool=run?.candidate_pool||[],stats=FootballMindProgress.postreadCounts(run);
 $('postread-pool-count').textContent=pool.length+' 个候选 · 已编码 '+stats.encoded+' 篇';
 replaceHtml($('postread-pool'),pool.map((c,i)=>{const e=postreadBodyRecord(c),p=FootballMindProgress.bodyProgress(run,candidateKey(c),c.document_id,currentServerTime());return '<details class="evidence-record" data-key="'+esc(candidateKey(c)||String(i))+'"><summary>'+String(i+1).padStart(2,'0')+' '+esc(c.title||e.url||candidateKey(c))+'<span class="micro-note">'+esc(({encoded:'编码已完成',encoding:'正在本地编码',encoding_pending:'等待本地编码',encoding_failed:'本地编码失败',encoding_budget_skipped:'全文已保存 · 超本轮编码预算',body_failed:'正文获取失败',duplicate_body:'完全重复正文 · 已去重',pending:'处理中'}[c.status]||c.status||'状态未记录'))+'</span></summary>'+semanticProgressHtml(p)+(e.url?'<p>'+linkHtml(e.url,'原始来源')+'</p>':'')+(e.body?'<details class="compact-details"><summary>查看已取得的正文</summary><pre>'+esc(e.body)+'</pre></details>':'<p>尚无可展示的完整正文，不使用标题补造。</p>')+postreadBindingHtml(e)+'</details>';}).join('')||'<p class="micro-note">尚未保存正文候选；发现、获取和编码阶段将在这里逐项更新。</p>');
}
function renderPostreadDecision(){
 const index=displayedRound(),d=index>=0?run.decisions[index]:null,c=postreadChoice(d),threshold=d?.threshold??run?.model_bindings?.threshold??run?.threshold;
 $('decision-mode').textContent=!run?'待启动':index<0?'获取 / 编码中':replayIndex===null?'最新一轮':'回看第 '+(index+1)+' 轮';
 $('decision').textContent=!run?'等待启动':!d?active()?'准备全文候选池':'没有形成 g 决策':c?'选中：'+(c.title||c.action_id):'停止继续采用';
 $('decision-context').textContent=!run?'g 在取得并编码正文后评分。':!d?run.stage||'尚无候选评分；不能将获取或编码失败解释成 g 判断无价值。':'第 '+(index+1)+' 轮 · '+time(d.decision_at)+' · '+reasonText(d.reason||d.stop_reason)+(Number.isFinite(threshold)?' · 冻结阈值 '+num(threshold):' · 阈值未记录')+(d.budget_outside_training_support?' · 本轮预算超出原训练预算范围，属于外推诊断':'');
 const rows=[...decisionRows(d)].sort((a,b)=>(postreadScore(b)??-Infinity)-(postreadScore(a)??-Infinity)||String(candidateKey(a)).localeCompare(String(candidateKey(b))));
 replaceHtml($('scores'),rows.map((r,i)=>{const id=candidateKey(r),pool=candidatePoolRow(id),selected=!!c&&sameCandidate(r,c.action_id);return '<details data-key="'+esc(id)+'" class="ranked-candidate '+(selected?'is-selected':'')+'"><summary><span class="candidate-heading"><span class="candidate-rank">'+String(i+1).padStart(2,'0')+'</span><span class="candidate-name">'+esc(r.title||pool?.title||id)+(selected?' '+tonePill('本轮选中','good'):'')+'</span></span><span class="candidate-value"><span>预计边际收益</span><strong>'+num(postreadScore(r))+'</strong></span></summary><div class="score-components"><p>'+ (postreadHeadType()==='state'?'仅当前状态的模型估计；不使用候选正文向量排序。':'全文语义向量与当前状态的模型估计；不是已实现收益。') +'</p><p>候选 '+esc(id)+'</p>'+linkHtml(r.url||r.source_url||pool?.url||pool?.source_url,'查看候选来源')+'</div></details>';}).join('')||'<div class="score-empty">'+(postreadHeadType()==='fixed'?'固定顺序对照没有 g 分数；采用顺序与 f 实际更新分别保留。':'尚无已保存的 g 分数；不会用来源请求成功代替评分。')+'</div>');
}
function renderPostreadReplay(){
 const ds=run?.decisions||[],index=displayedRound();$('trace-count').textContent=run?ds.length+' 轮决策 · '+(run.attempts||[]).length+' 次来源请求':'尚无运行';
 $('stage').textContent=run?runText(run.state)+' · '+(run.stage||'')+(run.reason?'\n'+reasonText(run.reason):''):'初始化 → 发现 → 取得正文 → 本地编码 → 全文 g 评分 → f 采用或停止。';
 $('replay-controls').hidden=index<0;replaceHtml($('replay-select'),ds.map((d,i)=>'<option value="'+i+'">第 '+(i+1)+' 轮 · '+(decisionAction(d)?'采用正文':'停止')+'</option>').join(''));
 if(index>=0)$('replay-select').value=String(index);$('replay-prev').disabled=index<=0;$('replay-next').disabled=index<0||index>=ds.length-1;$('replay-latest').textContent=replayIndex===null?'跟随最新中':'跟随最新';
 if(index<0){replaceHtml($('round-replay'),'<p class="empty-trace">'+(run?active()?'正在发现、预取或编码正文。候选池在下方显示；g 尚未形成决策。':'本轮未形成 g 决策，请查看候选获取及编码状态。':'启动后按实际保存记录回看全文选择过程。')+'</p>');renderPostreadDecision();return;}
 const d=ds[index],c=postreadChoice(d),s=c?(run.steps||[]).find(s=>s.f_executed===true&&(sameCandidate(s,c.action_id)||sameCandidate(s,c.document_id))):null;
 const choice=c?'<p class="step-title">'+esc(c.title||c.action_id)+'</p><p>'+(d.g_selection_executed===false?'按固定顺序采用 · 不执行 g 评分':'预计边际收益 '+num(postreadScore(c))+' · '+esc(reasonText(d.reason||d.stop_reason)))+'</p>':'<p>停止继续采用。'+esc(reasonText(d.reason||d.stop_reason))+'</p>';
 const progress=c?FootballMindProgress.bodyProgress(run,c.action_id,c.document_id,currentServerTime()):null;
 const validation=c?semanticProgressHtml(progress)+postreadBindingHtml(c):'<p>本步停止，不触发新的 f 更新；候选正文此前已获取、编码。</p>';
 const update=s?'<p>'+esc(stepUpdateDescription(s))+'</p>'+pairTable(asProbabilities(s.prediction_before||s.p_before),asProbabilities(s.prediction_after||s.p_after))+'<p class="micro-note">本步采用编号 '+esc((s.used_input_ids_after||s.used_input_ids||[s.input_id||s.document_id].filter(Boolean)).join('；')||'未记录')+'</p>':c?'<p>'+(progress?.status==='f_running'?'f 正在计算，after 尚未保存；请等待实际结果。':'未保存本候选的 f 执行与概率对；不补造 after 或 gain。')+'</p>':'<p>f 未执行新的更新，STOP 的 after 不存在；上方保留最后有效预测。</p>';
 const next=ds[index+1],replan=next?'<p>随后'+(decisionAction(next)?'继续采用下一篇正文。':'停止继续采用。')+'</p>':active()?'<p>等待下一条真实决策记录。</p>':'<p>'+esc(runText(run.state))+' · '+esc(reasonText(run.reason))+'</p>';
 replaceHtml($('round-replay'),'<p class="round-intro">第 '+(index+1)+' 轮 · '+esc(time(d.decision_at))+' · 已保存记录回放，不重发请求</p>'+[[d.g_selection_executed===false?'固定顺序选择':postreadHeadType()==='state'?'状态 g 决策':'全文 g 选择',choice],['正文与本地编码',validation],['f 概率更新',update],['重新评分与停止',replan]].map(([title,body],i)=>'<div class="replay-step"><span class="step-number">'+String(i+1).padStart(2,'0')+'</span><div><h3>'+title+'</h3>'+body+'</div></div>').join(''));renderPostreadDecision();
}
function renderPostreadLogs(){
 const attempts=run?.attempts||[],steps=(run?.steps||[]).filter(s=>s.f_executed===true),counts=run?.counts||{},c=FootballMindProgress.postreadCounts(run);
 $('request-count').textContent=attempts.length?'· '+attempts.length+' 次':'· 尚无记录';
 replaceHtml($('timeline'),attempts.map((a,i)=>'<div class="step"><div class="step-heading"><strong>'+String(i+1).padStart(2,'0')+' '+esc(a.purpose==='directory_discovery'?'发现候选目录':a.purpose==='search_discovery'?'公开搜索':'正文获取 / 预取')+'</strong>'+tonePill(a.status||'状态未知',a.status==='FAILED'?'risk':a.status==='NO_DATA'?'warn':'')+'</div>'+linkHtml(a.action?.url,a.action?.url||a.action?.query)+'<div class="request-clock">请求 '+esc(time(a.requested_at))+'<br>返回交接 '+esc(time(a.handoff_at))+'</div><p>'+esc(attemptDetail(a))+'</p>'+projectionReasons(a.response?.document)+'</div>').join('')||'<p class="micro-note">本轮尚无来源请求记录。</p>');
 $('evidence-count').textContent=steps.length+' 次更新';
 replaceHtml($('evidence'),steps.map((s,i)=>{const e=postreadBodyRecord(s);return '<details class="evidence-record" data-key="'+esc(s.input_id||s.document_id||String(i))+'"><summary>'+String(i+1).padStart(2,'0')+' '+esc(s.title||e.url||s.document_id)+'<span class="micro-note">'+esc(stepUpdateDescription(s))+'</span></summary><p class="micro-note">执行时间 '+esc(time(s.executed_at))+' · '+linkHtml(e.url,'查看原来源')+'</p>'+pairTable(asProbabilities(s.prediction_before||s.p_before),asProbabilities(s.prediction_after||s.p_after))+'<p>该模型直接采用正文 embedding。未运行 claim 抽取，不把 claim 缺失列为语义失败。</p>'+postreadBindingHtml(s)+(e.body?'<details class="compact-details"><summary>查看实际编码并交给 f 的正文</summary><pre>'+esc(e.body)+'</pre></details>':'<p>本记录没有可追溯正文文本，请核对输入编号；不能用标题代替。</p>')+'<details class="compact-details"><summary>实际输入与执行记录</summary><pre>'+esc(JSON.stringify({action_id:s.action_id,document_id:s.document_id,input_id:s.input_id,used_input_ids:s.used_input_ids||s.used_input_ids_after,f_executed:s.f_executed,text_sha256:e.text_sha256,raw_sha256:e.raw_sha256,vector_sha256:e.vector_sha256},null,2))+'</pre></details></details>';}).join('')||'<div class="empty-state"><span class="empty-symbol">f</span><div><strong>尚无实际 f 采用记录</strong><p>正文取得和编码成功不等于 f 已采用；g 可以在评分后停止。</p></div></div>');
 const items=[[run?.offline_replay===true?'来源归档调用':'来源请求',counts.source,run?.source_request_budget],['正文链接尝试',counts.body_attempts,run?.prefetch_body_budget],['已取得正文',c.body_documents,null],['已编码',c.encoded,null],['g 已评分候选',c.g_scored,null],['f 已采用',c.f_adopted,run?.max_reads]];
 replaceHtml($('counts'),items.map(([label,v,b])=>'<div><strong>'+(Number.isFinite(v)?v:'—')+'<small>'+(Number.isFinite(b)?' / '+b:'')+'</small></strong><span>'+label+'</span></div>').join(''));
 $('usage-state').textContent=run?runText(run.state):'未启动';replaceHtml($('host-review'),(run?.host_reviews||[]).map(r=>'<div class="host-review-item"><strong>'+esc(r.host)+'</strong>'+esc(r.status)+' · '+esc(r.reason)+'</div>').join('')||'<p class="micro-note">本轮尚无域名审核记录。</p>');renderPostreadPool();
}
function renderDecision(){
 if(postreadMode())return renderPostreadDecision();
 const index=displayedRound(),d=index>=0?run.decisions[index]:null,c=chosenCandidate(d);
 $('decision-mode').textContent=!run?'待启动':index<0?'待生成':replayIndex===null?'最新一轮':'回看第 '+(index+1)+' 轮';
 $('decision').textContent=!run?'等待启动':!d?active()?'正在准备候选菜单':'本轮未形成 g 决策':c?'选中：'+(c.title||c.action_id):'停止继续读取';
 $('decision-context').textContent=!run?'Agent 尚未启动，当前没有候选评分或本轮工具调用。':!d?active()?run.stage||'等待来源返回与菜单生成。':reasonText(run.reason):'第 '+(index+1)+' 轮 · '+time(d.decision_at)+' · '+reasonText(d.reason);
 if(d?.ranking_diagnostic==='no_score_separation')$('decision-context').textContent+=' · 当前候选评分相同，模型没有显示出区分能力；选择依照固定并列规则。';
 if(d?.tie_break==='observed_publication_day_desc_then_original_action_id')$('decision-context').textContent+=' · '+d.top_score_tie_count+' 个最高分候选并列，按来源明确发布日期优先，再按固定编号选择；这是并列规则，非模型学到的差异。';
 const ranked=[...(d?.candidates||[])].sort((a,b)=>{const delta=(Number.isFinite(b.two_step_net_value)?b.two_step_net_value:-Infinity)-(Number.isFinite(a.two_step_net_value)?a.two_step_net_value:-Infinity);return Math.abs(delta)>1e-12?delta:(b.source_publication_day||'').localeCompare(a.source_publication_day||'')||a.action_id.localeCompare(b.action_id);});
 replaceHtml($('scores'),ranked.length?ranked.map((c,i)=>'<details data-key="'+esc(c.action_id)+'" class="ranked-candidate '+(c.action_id===d.action_id?'is-selected':'')+'"><summary><span class="candidate-heading"><span class="candidate-rank">'+String(i+1).padStart(2,'0')+'</span><span class="candidate-name">'+esc(c.title||c.action_id)+(c.action_id===d.action_id?' '+tonePill('本轮选中','good'):'')+'</span></span><span class="candidate-value"><span>预期净价值</span><strong>'+num(c.two_step_net_value)+'</strong></span></summary><div class="score-components"><dl><dt>即时损失降低</dt><dd>'+num(c.immediate_gain)+'</dd><dt>当前查询成本</dt><dd>'+num(c.current_cost)+'</dd><dt>后续净价值</dt><dd>'+num(c.continuation_net_value)+'</dd></dl>'+linkHtml(c.url,'查看候选来源')+'<p class="micro-note">以上为模型估计，非已实现收益。</p></div></details>').join(''):'<div class="score-empty">'+(!run?'运行后展示真实候选及排序，不预置示例文章。':!d?'当前未记录可展示的候选评分。':'这一轮没有候选评分。'+(d.reason==='budget_exhausted'?'正文预算已用完。':'可回看前一轮查看此前候选。'))+'</div>');
}
function selectedAttempt(d,index,c){
 if(!c?.url)return null;const next=timestamp(run.decisions[index+1]?.decision_at),at=timestamp(d.decision_at);
 return (run.attempts||[]).find(a=>a.purpose==='g_selected_body'&&a.action?.url===c.url&&Number.isFinite(at)&&timestamp(a.requested_at)>=at&&(!Number.isFinite(next)||timestamp(a.requested_at)<=next))||null;
}
function stepUpdateDescription(s){
 if(s.f_executed!==true)return s.f_executed===false?'记录显示 f 未执行，不能生成有效收益。':'f 执行标记未记录，无法确认该次更新。';
 const p=asProbabilities(s.prediction_before||s.p_before),q=asProbabilities(s.prediction_after||s.p_after);if(!p||!q)return 'f 已记录执行，但概率对不完整。';
 const shift=p.reduce((sum,v,i)=>sum+Math.abs(q[i]-v),0),n=(s.claims||[]).length;
 if(postreadMode())return '冻结 f 已采用正文语义向量；'+(shift<=1e-12?'概率保持不变（数值精度内），不据此判断正文无价值。':'概率发生变化，L1 = '+shift.toFixed(6)+'；预测改善需独立赛果评估。');
 const a=s.f_consumption;
 return 'f 已执行 · 合法陈述 '+n+' 条'+(a?' · 可建模 '+a.modeled_claims+' 条':'')+' · '+(shift<=1e-12?'概率保持不变（数值精度内）。'+(a?consumptionReason(a.reason):n===0?'本次未抽取到合法陈述。':'记录不足以将不变归因于某一类特征。'):'概率发生变化，L1 = '+shift.toFixed(6)+'；尚不能据此判断预测改善。');
}
function sourceClocksHtml(doc,a){
 let host='未知';try{host=new URL(doc.url).hostname;}catch{}
 return '<div class="round-source"><span>来源：'+esc(host)+' · HTTP '+esc(doc.http_status??'未记录')+' · '+esc(doc.status)+'</span><span>请求：'+esc(time(a?.requested_at))+'</span><span>响应观察：'+esc(time(doc.observed_at))+'</span><span>返回交接：'+esc(time(a?.handoff_at))+'</span><span>来源发布时间：'+esc(time(doc.source_published_at))+'</span><span>原始 SHA：'+esc(doc.raw_sha256||'未记录')+'</span></div>';
}
function projectionReasons(doc){const rows=doc?.structured_news?.excluded||[];return rows.length?'<details class="compact-details"><summary>查看投影拒绝记录</summary><pre>'+esc(JSON.stringify(rows,null,2))+'</pre></details>':'';}
function semanticProgressHtml(progress){return '<div class="semantic-progress" data-status="'+esc(progress.status)+'"><p>'+esc(progress.message)+'</p>'+progress.attempts.map(a=>'<p class="micro-note">'+esc(a.label)+' · '+esc(a.status)+(Number.isFinite(a.elapsed_seconds)?' · 已用 '+a.elapsed_seconds.toFixed(1)+' 秒':' · 耗时未知')+' · 输入 '+esc(a.input_id||'未知')+(a.error_class?' · '+esc(a.error_class):'')+'</p>').join('')+'</div>';}
function renderReplay(){
 if(postreadMode())return renderPostreadReplay();
 const ds=run?.decisions||[],index=displayedRound();
 $('trace-count').textContent=run?ds.length+' 轮决策 · '+(run.attempts||[]).length+' 次来源请求':'尚无运行';
 $('stage').textContent=run?runText(run.state)+' · '+(run.stage||'')+(run.reason?'\n'+reasonText(run.reason):''):'启动后按真实记录展示：发现候选 → g 选择 → 正文验证 → f 更新 → 重新规划。';
 $('replay-controls').hidden=index<0;replaceHtml($('replay-select'),ds.map((d,i)=>'<option value="'+i+'">第 '+(i+1)+' 轮 · '+(d.action_id?'选择正文':'停止')+'</option>').join(''));
 if(index>=0)$('replay-select').value=String(index);$('replay-prev').disabled=index<=0;$('replay-next').disabled=index<0||index>=ds.length-1;$('replay-latest').textContent=replayIndex===null?'跟随最新中':'跟随最新';
 if(index<0){replaceHtml($('round-replay'),'<p class="empty-trace">'+(!run?'尚无决策记录。启动后可逐轮回看，回放仅使用已保存记录。':active()?'正在准备实验，请查看下方已记录的来源请求。':'本轮未生成候选决策；可展开来源请求与完整报告查看原因。')+'</p>');renderDecision();return;}
 const d=ds[index],c=chosenCandidate(d),a=selectedAttempt(d,index,c),doc=a?.response?.document;
 const step=d.action_id?(run.steps||[]).find(s=>s.action_id===d.action_id&&s.f_executed===true):null,next=ds[index+1],nextC=chosenCandidate(next);
 let validation,update;
 if(!c){validation='<p>本轮为停止决策，没有触发新的正文请求。</p>';update='<p>本轮没有执行 f 更新；停止不会生成新的 after 或 gain。</p>';}
 else if(step){
  validation='<p>正文投影与语义协议已通过，并记录 f 执行。</p>'+(doc?sourceClocksHtml(doc,a):'<p>步骤有正文记录，未找到满足时间绑定的来源请求。</p>');
  validation+=evidenceHighlights(step).filter(h=>h.quote).slice(0,2).map(h=>'<blockquote class="round-quote">'+esc(h.quote)+'</blockquote><p class="micro-note">'+(h.quote_verified?'引句已在输入正文定位':'引句定位状态未确认')+' · '+esc(h.entity||'')+'</p>').join('');
  update='<p>'+esc(stepUpdateDescription(step))+'</p>'+pairTable(step.prediction_before,step.prediction_after)+consumptionHtml(step);
 }else{
  const progress=FootballMindProgress.bodyProgress(run,d.action_id,doc?.document_id,currentServerTime());
  validation=!a?'<p>未记录与本轮时间及 URL 匹配的正文请求。</p>':!doc?'<p>正文请求尚未形成返回记录。</p>':'<p>'+esc(doc.status==='NO_DATA'?'来源返回后未投影出可用正文（NO_DATA）。':doc.status==='FAILED'?'正文来源请求失败。':doc.status==='INVALID_CONTENT'?'响应格式或内容校验未通过。':doc.structured_news?.role!=='article'?'返回内容未形成 article 正文输入。':'正文已返回。')+'</p>'+sourceClocksHtml(doc,a)+projectionReasons(doc);
  validation+=semanticProgressHtml(progress);
  const rejection=(run.body_rejections||[]).find(r=>r.action_id===d.action_id);
  update='<p>'+esc(['pending','f_running'].includes(progress.status)?'处理尚未完成，等待真实 f 输出；当前展示的仍是此前已保存预测。':progress.message)+'</p>'+(rejection?'<p>正文未采用原因：'+esc(rejection.reason==='semantic_validation_failed'?'语义协议校验失败；本候选被拒绝，未生成 after 或 gain':rejection.reason)+'</p>':'')+(run.state==='FAILED'&&index===ds.length-1?'<p>运行错误：'+esc(reasonText(run.reason))+'</p>':'');
 }
 const choice=c?'<p class="step-title">'+esc(c.title||c.action_id)+'</p><p>'+esc(reasonText(d.reason))+'</p><p>预期净价值：'+num(c.two_step_net_value)+' · 即时：'+num(c.immediate_gain)+' · 成本：'+num(c.current_cost)+' · 后续：'+num(c.continuation_net_value)+'</p><p>'+linkHtml(c.url,'查看候选来源')+'</p>':'<p class="step-title">停止继续读取</p><p>'+esc(reasonText(d.reason))+'</p>';
 const replan=next?'<p>下一轮：'+esc(nextC?'选择 '+(nextC.title||nextC.action_id):'停止继续读取')+'</p><p>'+esc(reasonText(next.reason))+'</p>':active()?'<p>等待下一条决策记录；不会预填下一步动作。</p>':'<p>'+esc(runText(run.state))+' · '+esc(reasonText(run.reason))+'</p>';
 replaceHtml($('round-replay'),'<p class="round-intro">第 '+(index+1)+' 轮 · '+esc(time(d.decision_at))+' · 只读回放，不重发请求</p>'+[['g 选择',choice],['正文返回与验证',validation],['f 概率更新',update],['重新规划与终止',replan]].map(([title,body],i)=>'<div class="replay-step"><span class="step-number">'+String(i+1).padStart(2,'0')+'</span><div><h3>'+title+'</h3>'+body+'</div></div>').join(''));renderDecision();
}
function attemptDetail(a){
 const doc=a.response?.document,r=doc?.structured_news?.retrieval_diagnostics||{},bits=[];if(!doc)return '尚无返回';
 if(doc.filter_status)bits.push(filterText(doc.filter_status));if(Number.isFinite(r.provider_result_count))bits.push('搜索服务返回 '+r.provider_result_count+' 条');if(Number.isFinite(r.filtered_result_count))bits.push('投影过滤 '+r.filtered_result_count+' 条');if(r.pending_host_review_count)bits.push('待审核链接 '+r.pending_host_review_count+' 条');if(Array.isArray(doc.links))bits.push('提取链接 '+doc.links.length+' 条');
 if(['directory_discovery','search_discovery'].includes(a.purpose)){const rows=(run?.menus||[]).filter(c=>c.parent_document_id===doc.document_id&&c.admission_reason==='observed_article_link');bits.push('准入候选 '+new Set(rows.map(c=>c.action_id)).size+' 条');}
 if(doc.error_class)bits.push('错误 '+doc.error_class);return bits.join(' · ');
}
function renderLogs(){
 if(postreadMode())return renderPostreadLogs();
 const attempts=run?.attempts||[],steps=(run?.steps||[]).filter(s=>s.f_executed===true);$('request-count').textContent=attempts.length?'· '+attempts.length+' 次':'· 尚无记录';
 replaceHtml($('timeline'),attempts.map((a,i)=>'<div class="step"><div class="step-heading"><strong>'+String(i+1).padStart(2,'0')+' '+esc(a.purpose==='directory_discovery'?'发现候选目录':a.purpose==='search_discovery'?'公开搜索':'g 选择的正文')+'</strong>'+tonePill(a.status,a.status==='FAILED'?'risk':a.status==='NO_DATA'?'warn':'')+'</div>'+linkHtml(a.action?.url,a.action?.url||a.action?.query)+'<div class="request-clock">请求 '+esc(time(a.requested_at))+'<br>返回交接 '+esc(time(a.handoff_at))+'</div><p>'+esc(attemptDetail(a))+'</p>'+projectionReasons(a.response?.document)+'</div>').join('')||'<p class="micro-note">本轮尚无来源请求记录。</p>');
 $('evidence-count').textContent=steps.length+' 次更新';
 replaceHtml($('evidence'),steps.map((s,i)=>'<details class="evidence-record" data-key="'+esc(s.input_id||String(i))+'"><summary>'+String(i+1).padStart(2,'0')+' '+esc(s.title||s.url)+'<span class="micro-note">'+esc(stepUpdateDescription(s))+'</span></summary><p class="micro-note">执行时间 '+esc(time(s.executed_at))+' · '+linkHtml(s.url,'查看原来源')+'</p>'+pairTable(s.prediction_before,s.prediction_after)+consumptionHtml(s)+(evidenceHighlights(s).length?'<div class="evidence-highlights"><h3>正文原句与来源陈述</h3>'+evidenceHighlights(s).map(evidenceHighlightHtml).join('')+'</div>':'<p class="micro-note">本次没有可展示的 claim 引句；仍可核对实际输入文本。</p>')+'<details class="compact-details"><summary>查看实际交给 f 的输入文本</summary><pre>'+esc(s.body)+'</pre></details><details class="compact-details"><summary>完整抽取结果与输入编号</summary><pre>'+esc(JSON.stringify({input_id:s.input_id,raw_sha256:s.raw_sha256,claims:s.claims,f_consumption:s.f_consumption||null},null,2))+'</pre></details></details>').join('')||'<div class="empty-state"><span class="empty-symbol" aria-hidden="true">f</span><div><strong>尚无已提交的正文更新</strong><p>来源获取、语义验证与 f 执行完成后，才在这里记录正文及预测变化。</p></div></div>');
 const counts=run?.counts||{},items=[['来源',counts.source,run?.source_request_budget],['语义',counts.provider,run?.provider_request_budget],['正文尝试',counts.body_attempts,run?.max_reads]];
 replaceHtml($('counts'),items.map(([label,v,b])=>'<div><strong>'+(Number.isFinite(v)?v:'—')+'<small>'+(Number.isFinite(b)?' / '+b:'')+'</small></strong><span>'+label+'</span></div>').join(''));
 $('usage-state').textContent=run?terminal.has(run.state)&&!steps.length?'未完成正文分析':runText(run.state):'未启动';
 replaceHtml($('host-review'),(run?.host_reviews||[]).map(r=>'<div class="host-review-item"><strong>'+esc(r.host)+'</strong>'+esc(r.status)+' · '+esc(r.reason)+'</div>').join('')||'<p class="micro-note">本轮尚无域名审核记录。</p>');
}
function renderRun(){
 renderPipelineMode();
 const m=run?.match||selected;$('fixture').textContent=m?displayName(m):'选择一场比赛开始研究';$('competition-label').textContent=m?.competition_key||'未来比赛';$('fixture-state').textContent=m?(run?'运行时：':'')+stateText(m.status):'待选择';$('kickoff').textContent=m?'开球 '+time(m.kickoff):'';
 const identity=run?.identity,scope=m?.model_scope==='cross_competition_experimental';
 replaceHtml($('scope-tags'),m?tonePill(scope?'跨赛事 · 未校准':postreadMode()?'冻结历史研究 · 在线效果待验证':'英超研究 · 效果待验证',scope?'warn':'')+tonePill(identity?identity.independently_confirmed?'独立官方身份已核验':'来源身份已匹配':'启动时核验身份',identity?.independently_confirmed?'good':''):'');
 $('match-scope').textContent=!m?'':identity?identity.independently_confirmed?'独立赛程核验与模型效果评估分别记录。':'已匹配来源 ID 与新鲜报价；本轮未确认独立官方赛程核验。':'目录用于选赛；启动后重新核对来源比赛、主客队与开球时间。';
 if(run?.initialization?.acquisition_mode==='recent_catalogue_raw_replay_after_transport_failure')$('match-scope').textContent='启动时来源断连；已核验并复用 5 分钟内的真实目录报价，原观察时间保留。尚未独立核验官方赛程。';
 $('source-detail').textContent=m?sourcesOf(m)+' · '+stateText(m.status)+' · 赛程观察 '+time(m.schedule_observed_at||m.observed_at||m.quote_catalog_observed_at)+' · ID '+m.match_id:'';
 $('source-link').hidden=!m?.source_url;if(m?.source_url){$('source-link').href=safeUrl(m.source_url);$('source-link').textContent='在 '+sourceName(m.source||m.sources?.[0])+' 查看本场赛程';}
 renderPrediction();preflightChecks();
 const key=run?run.run_id+':'+run.revision:'selected:'+selected?.match_id;
 if(key!==renderedRunKey){renderedRunKey=key;renderLogs();renderReplay();}else if(active()){renderReplay();if(postreadMode())renderPostreadPool();}
 replaceHtml($('limits'),(run?.limitations||catalog?.limitations||[]).map(s=>'<li>'+esc(s)+'</li>').join('')||'<li>尚未取得本轮研究边界记录。</li>');
 $('models').textContent=JSON.stringify(run?.model_bindings||catalog?.model_bindings||{},null,2);
 if(run){if(Number.isFinite(run.max_reads))$('budget').value=String(run.max_reads);if(Number.isFinite(run.cost_loss_units))$('cost').value=String(run.cost_loss_units);$('budget-description').textContent=postreadMode()?'本轮来源请求上限 '+(run.source_request_budget??'未记录')+' 次；最多尝试 '+(run.prefetch_body_budget??'未记录')+' 个正文链接；f 采用上限 '+(run.max_reads??'未记录')+' 篇。失败、空正文和重复正文也占链接尝试次数，不保证全部取得有效正文。g 阈值保留，可能提前停止；编码与采用分别计数。':'本轮记录：来源上限 '+(run.source_request_budget??'未记录')+' 次，语义上限 '+(run.provider_request_budget??'未记录')+' 次，正文上限 '+(run.max_reads??'未记录')+' 次，运行上限 '+(run.run_timeout_minutes??'未记录')+' 分钟。';}
 else $('budget-description').textContent=catalog&&!postreadMode()?'目录最多 2 个，搜索最多 5 种意图，来源最多 10 次，正文最多 5 篇，语义最多 15 次（每篇初次抽取、最多一次协议修复及非空陈述来源复核），最长 15 分钟。g 可以提前停止。':'新版默认最多尝试 20 个正文链接；失败、空正文和重复正文也占尝试次数，不保证取得 20 篇有效正文。f 采用上限可选 1–20 篇，g 可以提前停止；实际配置以本轮后端记录为准。';
 if(!run&&postreadMode())$('budget-description').textContent='新版来源上限 40 次；最多尝试 20 个正文链接，失败、空正文和重复正文也占尝试次数，不保证取得 20 篇有效正文。f 采用上限可选 1–20 篇，默认 20。先编码全文再由 g 评分，g 可以提前停止；此路径不调用 DeepSeek。';
 if(postreadMode()){const cfg=run||catalog;if(Number.isFinite(cfg?.local_encoding_token_budget))$('budget-description').textContent+=' 未缓存正文最多 '+cfg.local_encoding_token_budget+' token；超出时保存全文但跳过本轮编码，不截断、不送 g/f。';if(Number.isFinite(cfg?.prefetch_minutes))$('budget-description').textContent+=' 采集与编码约 '+cfg.prefetch_minutes+' 分钟后使用已完成的候选池；当篇在途请求或编码可能使阶段延迟。';}
 controls();
}
async function operation(fn){if(busy||viewRequestPending)return;busy=true;notice('');controls();try{await fn();}catch(e){loadFailed=true;notice(e.message);}finally{busy=false;controls();}}
async function load(){
 catalog=await api('catalog');loadFailed=false;catalogReceivedAt=Date.now();lastCatalogPoll=Date.now();
 if(selected&&!run)selected=(catalog.matches||[]).find(m=>m.match_id===selected.match_id)||null;renderCatalog();
 if(!selected&&!run){selected=filteredMatches.find(m=>m.can_start)||filteredMatches.find(m=>m.status==='scheduled'&&timestamp(m.kickoff)>currentServerTime()+startLeadMinutes()*60000)||filteredMatches[0]||null;const i=filteredMatches.indexOf(selected);if(i>0)$('matches').scrollTop=i*ROW_HEIGHT;renderMatchWindow();}
}
async function refreshData(){catalog=await api('refresh',{});catalogReceivedAt=Date.now();lastCatalogPoll=Date.now();if(selected&&!run)selected=(catalog.matches||[]).find(m=>m.match_id===selected.match_id)||null;renderCatalog();renderRun();}
async function openRun(id){
 if(active()||busy||viewRequestPending){notice('请先结束当前运行再切换记录。');return;}
 const epoch=++selectionEpoch;viewRequestPending=true;controls();
 try{const value=await api('runs/'+encodeURIComponent(id));if(epoch!==selectionEpoch)return;run=value;selected=value.match;replayIndex=null;stopping=false;renderedRunKey=null;notice('');setRunUrl();renderMatchWindow();renderRun();}catch(e){notice(e.message);}finally{viewRequestPending=false;controls();}
}
function showReadyMatches(){
 $('match-search').value='';$('date-range').value='week';$('competition-scope').value='all';$('competition').value='';$('availability').value='ready';$('schedule-filters').open=true;renderCatalog();
 if(filteredMatches.length){const m=filteredMatches[0];selected=m;run=null;renderedRunKey=null;renderMatchWindow();renderRun();}else notice('当前目录没有可启动比赛。可刷新来源数据，或回看已保存的在线运行。');
}
function setReplay(index){replayIndex=index;renderReplay();}
$('postread-model').onchange=()=>{renderPipelineMode();renderPostreadDecision();};
$('competition').onchange=()=>renderCatalog();$('availability').onchange=()=>renderCatalog();$('date-range').onchange=()=>renderCatalog();$('competition-scope').onchange=()=>renderCatalog();$('match-search').oninput=()=>renderCatalog();
$('clear-search').onclick=()=>{$('match-search').value='';renderCatalog();$('match-search').focus();};
$('match-search').onkeydown=e=>{if(e.key==='ArrowDown'){e.preventDefault();keyboardIndex=0;$('matches').scrollTop=0;renderMatchWindow();$('matches').focus();}else if(e.key==='Enter'&&filteredMatches.length){e.preventDefault();chooseMatch(filteredMatches[0],{focus:true});}};
$('matches').onclick=e=>{const b=e.target.closest('[data-match]');if(b)chooseMatch(filteredMatches.find(m=>m.match_id===b.dataset.match));};
$('matches').onscroll=()=>{if(filteredMatches.length>VIRTUAL_THRESHOLD&&!listFrame)listFrame=requestAnimationFrame(()=>{listFrame=0;renderMatchWindow();});};
$('matches').onkeydown=e=>{
 if(!filteredMatches.length||!['ArrowDown','ArrowUp','Home','End','Enter',' '].includes(e.key))return;e.preventDefault();
 if(e.key==='Enter'||e.key===' '){chooseMatch(filteredMatches[Math.max(0,keyboardIndex)],{focus:true});return;}
 keyboardIndex=e.key==='Home'?0:e.key==='End'?filteredMatches.length-1:Math.max(0,Math.min(filteredMatches.length-1,keyboardIndex+(e.key==='ArrowDown'?1:-1)));
 const list=$('matches'),top=keyboardIndex*ROW_HEIGHT;if(top<list.scrollTop)list.scrollTop=top;else if(top+ROW_HEIGHT>list.scrollTop+list.clientHeight)list.scrollTop=top+ROW_HEIGHT-list.clientHeight;renderMatchWindow();
};
$('history').onclick=e=>{const item=e.target.closest('[data-run]');if(item)openRun(item.dataset.run);};
$('refresh').onclick=()=>operation(refreshData);
$('start').onclick=()=>operation(async()=>{
 const gate=experimentGate();if(gate.disabled)return;
 if(gate.action==='refresh')return refreshData();if(gate.action==='reload'){await load();renderRun();return;}if(gate.action==='choose')return showReadyMatches();
 if(gate.action==='stop'){const value=await api('runs/'+run.run_id+'/stop',{});run=value;stopping=!!value.cancel_requested;renderedRunKey=null;notice(stopping?'已请求停止；在途请求结束后不再发新请求。':'本轮已经结束。');renderRun();return;}
 if(gate.action!=='start'||(!postreadMode()&&!$('cost').reportValidity()))return;
 const request={match_id:selected.match_id,max_reads:Number($('budget').value)};if(!postreadMode())request.cost_loss_units=Number($('cost').value);else if($('postread-model').value)request.model_name=$('postread-model').value;
 run=await api('runs',request);selected=run.match;replayIndex=null;stopping=false;renderedRunKey=null;setRunUrl();renderRun();
});
$('new').onclick=()=>{if(active()||busy||viewRequestPending)return;selectionEpoch++;selected=(catalog?.matches||[]).find(m=>m.match_id===selected?.match_id)||selected;run=null;frozenMarketComparison=null;replayIndex=null;stopping=false;renderedRunKey=null;history.replaceState(null,'','/live.html');notice('');renderRun();};
$('replay-prev').onclick=()=>setReplay(Math.max(0,displayedRound()-1));$('replay-next').onclick=()=>setReplay(Math.min((run?.decisions?.length||1)-1,displayedRound()+1));$('replay-select').onchange=()=>setReplay(Number($('replay-select').value));$('replay-latest').onclick=()=>setReplay(null);
async function poll(){
 if(polling||busy||viewRequestPending||document.hidden)return;polling=true;
 try{
  if(active()){const id=run.run_id,value=await api('runs/'+id);if(run?.run_id===id){run=value;renderRun();if(!active()){stopping=false;await load();renderRun();}}}
  if(catalog?.refreshing||Date.now()-lastCatalogPoll>30000){await load();renderRun();}
 }catch(e){$('connection').textContent='连接暂不可用';$('connection-dot').className='offline';notice('状态读取失败：'+e.message+'。已显示的数据为上次记录，可重新连接。');}finally{polling=false;}
}
window.addEventListener('resize',()=>renderMatchWindow());
(async()=>{try{await load();const id=new URL(location.href).searchParams.get('run');if(id)await openRun(id);renderRun();}catch(e){loadFailed=true;$('connection').textContent='连接失败';$('connection-dot').className='offline';notice(e.message);controls();}})();
setInterval(poll,1600);
