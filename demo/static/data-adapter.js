'use strict';

/*
 * The page talks to the existing demo server through this small adapter.  It
 * keeps transport details away from rendering and labels every fallback as
 * demo data so that a missing backend field never looks like a live result.
 */
window.FootballMindAdapter = (() => {
  const DEFAULT_CUTOFF = '2026-09-20T11:10:00Z';
  const DEMO_FIXTURES = [
    {match_id:'demo:harbor-vs-orchard', home:'Harbor City FC', away:'Orchard Rovers', competition:'演示联赛', kickoff:'2026-10-19T12:30:00Z', status:'scheduled', source:'demo', is_demo:true},
    {match_id:'demo:atlas-vs-crescent', home:'Atlas Athletic', away:'Crescent Town', competition:'演示杯赛', kickoff:'2026-10-19T15:00:00Z', status:'scheduled', source:'demo', is_demo:true},
    {match_id:'demo:north-vs-lake', home:'Northbridge FC', away:'Lakeview United', competition:'演示联赛', kickoff:'2026-10-19T18:45:00Z', status:'scheduled', source:'demo', is_demo:true}
  ];

  async function request(path, options) {
    const response = await fetch(path, options || {});
    let payload = null;
    try { payload = await response.json(); } catch (_) { payload = null; }
    if (!response.ok) { const error = new Error(payload?.error || `HTTP ${response.status}`); error.status = response.status; throw error; }
    return payload;
  }

  // The run view is the canonical browser contract.  Keep the old top-level
  // fields as a compatibility projection for the existing renderer while
  // exposing the complete versioned object as ``ui_view``.
  function normalizeRunView(payload) {
    if (!payload || payload.schema_version !== 'footballmind-ui-run-view-v1') return payload;
    const prediction = payload.prediction || {};
    return {
      ...payload,
      ui_view: payload,
      prediction: prediction.current || null,
      last_prediction: prediction.after || prediction.before || null,
      state: payload.lifecycle?.state || payload.state,
      reason: payload.lifecycle?.reason ?? payload.reason,
      test_mode: Boolean(payload.test_mode || payload.mode === 'test' || payload.mode === 'synthetic'),
      is_demo: Boolean(payload.is_demo),
      features: payload.features || null,
      evidence: payload.evidence || [],
      events: payload.events || [],
      budget: payload.budget || null,
      policy_configuration: payload.policy_configuration || {},
      queried_topics: payload.queried_topics || []
    };
  }

  function normalizeMatch(raw) {
    return {
      ...raw,
      match_id: raw.match_id || raw.id,
      home: raw.home || raw.home_team_name || raw.home_team?.name || '主队未命名',
      away: raw.away || raw.away_team_name || raw.away_team?.name || '客队未命名',
      competition: raw.competition || raw.league || '联赛未知',
      kickoff: raw.kickoff || raw.kickoff_at || null,
      status: raw.status || 'unknown',
      source: raw.source || 'unknown',
      is_demo: Boolean(raw.is_demo)
    };
  }

  async function loadMatches() {
    const params = new URLSearchParams({mode:'real', archive_replay:'true', model_id:'market', information_mode:'operational', time_mode:'archive_replay', cutoff:DEFAULT_CUTOFF});
    try {
      const payload = await request(`/api/matches?${params}`);
      const rows = Array.isArray(payload?.data) ? payload.data : Array.isArray(payload) ? payload : [];
      const matches = rows.filter(item => ['scheduled', 'unknown'].includes(item.status || 'unknown')).map(normalizeMatch);
      const meta = payload?.meta || {};
      if (matches.length) return {matches, isDemo:Boolean(meta.is_demo), source:meta.source_label || '真实档案回放 / 本地逐批披露', updatedAt:meta.updated_at || new Date().toISOString(), cutoff:meta.cutoff || DEFAULT_CUTOFF, raw:payload};
      return {matches:DEMO_FIXTURES.map(normalizeMatch), isDemo:true, source:'演示数据 · 接口无可用比赛', updatedAt:new Date().toISOString(), cutoff:meta.cutoff || DEFAULT_CUTOFF, raw:payload};
    } catch (error) {
      return {matches:DEMO_FIXTURES.map(normalizeMatch), isDemo:true, source:'演示数据 · 接口错误', updatedAt:new Date().toISOString(), cutoff:DEFAULT_CUTOFF, error:error.message};
    }
  }

  function demoRun(match) {
    const now = new Date().toISOString();
    return {
      run_id:`demo-${match.match_id.split(':')[1]}`,
      match_id:match.match_id, prediction_cutoff:DEFAULT_CUTOFF, mode:'demo', test_mode:true,
      state:'COMPLETED', reason:'DEMO_FIXTURE_ONLY', created_at:now, updated_at:now,
      match, evidence:[], features:null, prediction:{home_win_probability:0.42, draw_probability:0.29, away_win_probability:0.29, model_version:'demo-market-proportional', is_demo:true}, last_prediction:null, queried_topics:[], quality:[],
      policy_configuration:{}, budget:{calls:0,queries:0,requests:0,remaining_calls:0,remaining_queries:0,remaining_requests:0},
      events:[{seq:1,type:'initialized',at:now,data:{is_demo:true,message:'演示数据：未调用真实 Agent runtime'}},{seq:2,type:'stopped',at:now,data:{reason:'DEMO_FIXTURE_ONLY'}}], is_demo:true,
      tool_configuration:{model_id:'demo',model_name:'演示市场先验',label:'演示数据'}, report:{claims:[]}
    };
  }

  async function startRun(match) {
    if (match.is_demo || String(match.match_id).startsWith('demo:')) return demoRun(match);
    const body = {mode:'real', scenario:'normal', archive_replay:true, model_id:'market', information_mode:'operational', time_mode:'archive_replay', match_id:match.match_id, prediction_cutoff:DEFAULT_CUTOFF,
      budget:{max_calls:20,max_queries:3,max_requests:6,max_cost:3,max_elapsed_seconds:60,max_tool_seconds:10}};
    const created = await request('/api/runs', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    try { return await fetchRun(created.run_id); } catch (_) { return normalizeRunView(created); }
  }

  async function runToCompletion(run) {
    if (!run || run.is_demo || run.mode === 'demo') return run;
    const terminal = new Set(['STOPPED','DEFERRED','COMPLETED','FAILED','BUDGET_EXHAUSTED','DEADLINE_EXPIRED']);
    let current = run;
    for (let i=0; i<12 && !terminal.has(String(current.state || '').toUpperCase()); i++) {
      current = await request(`/api/runs/${encodeURIComponent(current.run_id)}/action`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'next'})});
    }
    try { return await fetchRun(current.run_id); } catch (_) { return normalizeRunView(current); }
  }

  async function fetchRun(runId) {
    try { return normalizeRunView(await request(`/api/runs/${encodeURIComponent(runId)}/view`)); }
    catch (error) { if (error.status !== 404) throw error; return normalizeRunView(await request(`/api/runs/${encodeURIComponent(runId)}`)); }
  }

  // Research has its own contract and persistence namespace. Never normalize it
  // as an old experiment, and never fall back to demonstration data on failure.
  function researchPayload(payload, schema) {
    if (payload?.schema_version !== schema) throw new Error(`研究接口合同不匹配：需要 ${schema}。请连接 python -m demo 启动的服务。`);
    return payload;
  }
  function researchPath(runId) {
    if (!/^rg-[a-f0-9]{32}$/.test(runId)) throw new Error('研究运行 ID 格式无效');
    return `/api/research-g/runs/${runId}`;
  }
  const researchView = payload => researchPayload(payload, 'footballmind-research-g-run-v1');
  const post = body => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  async function researchCatalog() { return researchPayload(await request('/api/research-g/catalog'), 'footballmind-research-g-catalog-v1'); }
  async function startResearchRun(options) { return researchView(await request('/api/research-g/runs',post(options))); }
  async function researchAction(run, action) { return researchView(await request(`${researchPath(run.run_id)}/action`,post({action,expected_revision:run.revision}))); }
  async function fetchResearchRun(runId) { return researchView(await request(`${researchPath(runId)}/view`)); }
  async function fetchResearchReport(runId) { return researchPayload(await request(`${researchPath(runId)}/report`),'footballmind-research-g-report-v1'); }

  return {DEFAULT_CUTOFF, DEMO_FIXTURES, loadMatches, startRun, runToCompletion, fetchRun, normalizeRunView,
    researchCatalog,startResearchRun,researchAction,fetchResearchRun,fetchResearchReport};
})();
