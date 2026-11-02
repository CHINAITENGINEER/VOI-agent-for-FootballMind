"""Real JS stage projection tests: no browser/network/provider required."""
import json
from pathlib import Path
import subprocess

SCRIPT=Path(__file__).resolve().parents[2]/'demo/static/live-progress.js'

def project(record, action='a1', document='d1'):
    program="const {bodyProgress}=require(process.argv[1]);console.log(JSON.stringify(bodyProgress(JSON.parse(process.argv[2]),'"+action+"','"+document+"',Date.parse('2026-10-25T12:00:20Z'))));"
    result=subprocess.run(['node','-e',program,str(SCRIPT),json.dumps(record)],capture_output=True,text=True,encoding='utf8',check=True)
    return json.loads(result.stdout)

def pending(kind='claim_review'):
    return dict(state='RUNNING',active_body=dict(action_id='a1',input_id='i1',document_id='d1'),
        semantic_attempts=[dict(kind=kind,status='PENDING',input_id='i1',requested_at='2026-10-25T12:00:00Z')])

def test_pending_review_bound_by_input_not_failure():
    p=project(pending())
    assert p['status']=='pending' and '复核' in p['message'] and '等待实际返回' in p['message']
    assert p['attempts'][0]['elapsed_seconds']==20

def test_initial_repair_and_foreign_body_are_distinct():
    assert '抽取' in project(pending('initial'))['message']
    assert '修复' in project(pending('validation_repair'))['message']
    assert project(pending(),action='unrelated',document='other')['status']=='unknown'

def test_terminal_inflight_record_not_live_timer():
    r=pending();r['state']='STOPPED'
    p=project(r)
    assert p['status']=='interrupted' and p['attempts'][0]['elapsed_seconds'] is None

def test_failed_and_committed_are_not_pending():
    r=pending();r['state']='FAILED';r['semantic_attempts'][0].update(status='FAILED',completed_at='2026-10-25T12:00:12Z')
    assert project(r)['status']=='failed'
    r['steps']=[dict(action_id='a1',f_executed=True)]
    assert project(r)['status']=='committed'

def test_old_unbound_record_stays_unknown():
    r=pending();r.pop('active_body')
    assert project(r)['status']=='unknown'


def test_postread_encoded_body_without_claims_is_not_semantic_failure():
    r=dict(pipeline_mode='postread_fulltext_v1',state='RUNNING',
        candidate_pool=[dict(action_id='a1',document_id='d1',text_sha256='bodysha',body='Full body',status='encoded')])
    p=project(r)
    assert p['status']=='encoded' and '已编码' in p['message'] and '语义校验未通过' not in p['message']
    r['steps']=[dict(action_id='a1',document_id='d1',f_executed=True,claims=[])]
    p=project(r)
    assert p['status']=='committed' and '正文语义向量' in p['message']


def test_postread_encoding_failure_has_no_f_update_and_elapsed_clock_is_local():
    row=dict(action_id='a1',document_id='d1',status='encoding',encoding_started_at='2026-10-25T12:00:00Z')
    r=dict(pipeline_mode='postread_fulltext_v1',state='RUNNING',candidate_pool=[row])
    p=project(r)
    assert p['status']=='pending' and p['attempts'][0]['elapsed_seconds']==20
    row.update(status='encoding_failed',encoding_completed_at='2026-10-25T12:00:12Z')
    p=project(r)
    assert p['status']=='failed' and '不是 claim 抽取失败' in p['message']
    assert p['attempts'][0]['elapsed_seconds']==12
    assert project(r,action='other',document='other')['status']=='unknown'


def test_postread_counts_keep_read_encoded_scored_adopted_separate():
    r=dict(candidate_pool=[dict(action_id='a1',body='one',status='encoded'),dict(action_id='a2',body='two',status='encoding_failed'),dict(action_id='a3',status='body_failed')],
        decisions=[dict(candidates=[dict(action_id='a1',score=.01)]),dict(candidates=[dict(action_id='a1',score=-.01)])],steps=[])
    program="const {postreadCounts}=require(process.argv[1]);console.log(JSON.stringify(postreadCounts(JSON.parse(process.argv[2]))));"
    result=subprocess.run(['node','-e',program,str(SCRIPT),json.dumps(r)],capture_output=True,text=True,encoding='utf8',check=True)
    assert json.loads(result.stdout)==dict(body_documents=2,encoded=1,g_scored=1,f_adopted=0,body_failed=1,encoding_failed=1)


def test_postread_duplicate_and_inflight_f_do_not_invent_adoption():
    row=dict(action_id='a1',document_id='d1',status='duplicate_body',body='same full text')
    r=dict(pipeline_mode='postread_fulltext_v1',state='STOPPED',candidate_pool=[row])
    assert project(r)['status']=='duplicate'
    row['status']='encoded';r['state']='RUNNING'
    r['active_body']=dict(action_id='a1',document_id='d1',status='f_running')
    p=project(r)
    assert p['status']=='f_running' and 'after 尚未保存' in p['message']
    r['state']='FAILED'
    assert project(r)['status']=='encoded'


def test_postread_encoding_budget_skip_preserves_body_without_value_claim():
    r=dict(pipeline_mode='postread_fulltext_v1',state='RUNNING',
        candidate_pool=[dict(action_id='a1',document_id='d1',body='full original body',status='encoding_budget_skipped')])
    p=project(r)
    assert p['status']=='budget_skipped'
    assert '全文已保存' in p['message'] and '未送 g / f' in p['message']
    assert '不表示没有价值' in p['message'] and '没有截断全文' in p['message']
