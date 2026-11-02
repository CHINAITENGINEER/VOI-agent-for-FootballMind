from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
import pytest
import requests
from football_agent.experience import ExperienceService, ExperienceProtocol, public_state, validate_decision, now
from football_agent.experience_provider import ExperienceProvider
from football_agent.experience_sources import ExperienceWebTools, project_body, clean_url
from football_spider.storage.raw_archive import RawArchive

URL='https://example.org/team'

def decision(public, action='read_url'):
    return {'task_id':public['task_id'],'action':action,'url':URL if action=='read_url' else None,
        'query':'official team lineup' if action=='search_web' else None,
        'engine':'bing' if action=='search_web' else None,
        'question_to_user':'请提供比赛日期和对手。' if action=='ask_user' else None,
        'unknown_to_resolve':'首发是否轮换','useful_if':'包含明确的比赛首发名单',
        'why_worth_cost':'首发能检验轮换假说','previous_return':'partial' if public['documents'] else 'none',
        'assessment':{'status':'unresolved','answer':'尚需核实','claims':[],'remaining_unknowns':['首发名单']},
        'alternatives':[],'reason':'根据现有证据选择下一步'}

class Provider:
    def __init__(self, fn): self.fn=fn
    def decide(self,p): return {'response':self.fn(p),'test_double':True}

class Tools:
    def __init__(self, text='Squad list only', fail=False):self.text,self.fail=text,fail
    def execute(self,action,*,run_id,reserve_request,cancelled):
        request_id=reserve_request(action['url'] or URL)
        if not request_id:return {'document':None,'requests':[]}
        if self.fail:raise RuntimeError('crash after request reservation')
        doc=project_body(('<article>'+self.text+'</article>').encode(),URL,200)
        doc.update(document_id='doc-test',observed_at=now().isoformat(),source_published_at=None)
        return {'document':doc,'requests':[{'request_id':request_id,'http_status':200,'error':None}]}

@pytest.fixture
def service(tmp_path):
    svc=ExperienceService(tmp_path,provider_factory=lambda f,e:Provider(lambda p:decision(p)),tools_factory=lambda f:Tools())
    yield svc
    svc.close()

def create(svc,**kwargs):return svc.create({'question':'核实首发是否实际轮换','seed_urls':[URL],**kwargs})['run_id']

def test_decision_read_replan_and_stop(service):
    seen=[]
    def choose(p):
        seen.append(deepcopy(p))
        r=decision(p,'defer' if p['documents'] else 'read_url')
        if p['documents']:r['assessment']['claims']=[{'document_id':'doc-test','quote':'Squad list only','statement':'仅有大名单'}]
        return r
    service.provider_factory=lambda f,e:Provider(choose)
    run=create(service);service._drive(run,True);s=service.get(run)
    assert s['status']=='deferred' and s['spent']=={'http_calls':1,'llm_calls':2}
    assert len(seen[0]['documents'])==0 and len(seen[1]['documents'])==1
    assert s['assessment']['claims'][0]['quote']=='Squad list only'
    assert s['assessment_document_count']==len(s['documents'])

def test_final_model_call_reserved_for_stopping(service):
    seen=[]
    service.provider_factory=lambda f,e:Provider(lambda p:(seen.append(p) or decision(p,'defer' if p['final_decision_only'] else 'read_url')))
    run=create(service,limits={'llm_calls':2});service._drive(run,True)
    assert seen[-1]['final_decision_only'] is True
    assert service.get(run)['spent']=={'http_calls':1,'llm_calls':2}

def test_supported_claims_with_unknowns_are_partial_not_discarded(service):
    def choose(p):
        r=decision(p,'stop' if p['documents'] else 'read_url')
        if p['documents']:
            r['assessment']['status']='supported'
            r['assessment']['claims']=[{'document_id':'doc-test','quote':'Squad list only','statement':'仅有大名单'}]
        return r
    service.provider_factory=lambda f,e:Provider(choose)
    run=create(service);service._drive(run,True);s=service.get(run)
    assert s['status']=='deferred'
    assert s['assessment']['status']=='supported'
    assert s['assessment_resolution']['effective_status']=='partial'
    assert s['assessment']['remaining_unknowns'] and s['assessment']['claims']

def test_invalid_quote_is_retained_not_silently_corrected(service):
    run=create(service);service.step(run);service.step(run)
    def bad(p):
        r=decision(p,'stop');r['assessment']['claims']=[{'document_id':'doc-test','quote':'fabricated','statement':'bad'}];return r
    service.provider_factory=lambda f,e:Provider(bad);service.step(run)
    s=service.get(run)
    assert s['status']=='paused' and s['last_validation_error']=='UNSUPPORTED_QUOTATION'
    assert len(s['turns'])==1 and s['spent']['llm_calls']==2
    assert public_state(s)['last_validation_error']=='UNSUPPORTED_QUOTATION'
    assert any('fabricated' in p.read_text(encoding='utf-8') for p in service.folder(run).glob('operations/*/result.json'))

def test_timeout_pause_is_charged_and_not_retried(service):
    def timeout(p):raise subprocess.TimeoutExpired(['codex'],60)
    service.provider_factory=lambda f,e:Provider(timeout)
    run=create(service);service._drive(run,True);s=service.get(run)
    assert s['status']=='paused' and s['spent']['llm_calls']==1 and not s['pending']
    assert '超时' in s['last_error']

def test_clarification_resumes_same_budget_and_context(service):
    service.provider_factory=lambda f,e:Provider(lambda p:decision(p,'ask_user'))
    run=create(service);service._drive(run,True);first=service.get(run)
    service.answer(run,'对手是 A 队，日期待核实');second=service.get(run)
    assert first['status']=='needs_input' and second['status']=='ready'
    assert second['deadline']==first['deadline'] and second['spent']==first['spent']
    assert second['user_context'][0]['verified'] is False

def test_cancel_during_model_discards_new_action(service):
    run=create(service)
    def cancelled(p):service.cancel(run);return decision(p)
    service.provider_factory=lambda f,e:Provider(cancelled)
    service._drive(run,True);s=service.get(run)
    assert s['status']=='cancelled' and s['spent']=={'http_calls':0,'llm_calls':1}
    assert s['queued_action'] is None and not s['turns']

def test_deadline_no_new_calls(service):
    run=create(service);service.mutate(run,lambda s:s.update(deadline=(now()-timedelta(seconds=1)).isoformat()))
    service._drive(run,True)
    assert service.get(run)['status']=='deadline_expired'
    assert service.get(run)['spent']=={'http_calls':0,'llm_calls':0}

def test_crash_pending_no_automatic_repeat(service):
    run=create(service);service.step(run);service.tools_factory=lambda f:Tools(fail=True)
    service._drive(run,True);s=service.get(run)
    assert s['status']=='needs_reconciliation' and s['pending'] and s['spent']['http_calls']==1
    service.launch(run,automatic=True)
    assert service.get(run)['spent']==s['spent']

def test_restart_marks_pending_conservatively(service):
    run=create(service);service.mutate(run,lambda s:s.update(status='running',pending={'kind':'model','operation_id':'pending'}))
    assert service.recover()==[run]
    assert service.get(run)['status']=='needs_reconciliation'
    assert service.recover()==[]

def test_feedback_export_binds_state_not_gold(service):
    run=create(service);r=service.feedback(run,{'category':'bad_stop','comment':'停止得太早','event_sequence':1})
    export=service.export(run)
    assert r['state_digest'] and r['label_status']=='USER_FEEDBACK_NOT_GOLD'
    assert export['feedback'][0]==r and export['numeric_voi_calibrated'] is False
    assert export['state']['feedback_count']==1

@pytest.mark.parametrize('url',['http://example.org','https://127.0.0.1/','https://user:pass@example.org','https://example.org/?token=abc','https://service.internal/'])
def test_public_url_validation(url):
    with pytest.raises(ValueError):clean_url(url)

@pytest.mark.parametrize('body,status,expected',[(b'{"data":[]}',200,'empty_structured'),(b'',403,'unavailable'),(b'',302,'unavailable'),(b'',200,'empty_text'),(b'<title>Just a moment</title>',200,'access_blocked')])
def test_technical_return_descriptors(body,status,expected):
    d=project_body(body,URL,status)
    assert d['status']==expected and d['semantic_relevance']=='unassessed'

def test_search_results_are_discovery_not_claims(service):
    run=create(service);s=service.get(run)
    d=project_body(b'<rss><channel><item><title>A title</title><link>https://example.org/a</link><description>snippet</description></item></channel></rss>',URL,200)
    d.update(document_id='search',observed_at=now().isoformat());s['documents']=[d]
    p=public_state(s);r=decision(p,'defer');r['assessment']['claims']=[{'statement':'x','quote':'snippet','document_id':'search'}]
    with pytest.raises(ValueError,match='UNSUPPORTED'):validate_decision(r,p)
    assert d['links'][0]['url']=='https://example.org/a'

def test_dynamic_hint_and_embedded_json():
    d=project_body(b'<main>Filters</main><script src=app.js></script>',URL,200)
    assert 'sparse_static_text_possible_dynamic_page' in d['signals']
    d=project_body(b'<main>Filters</main><script type=application/json>{"team":"A"}</script>',URL,200)
    assert 'embedded_structured_content' in d['signals'] and 'team' in d['text']

def test_multiple_rule_articles_are_not_lost():
    d=project_body(b'<main><article>Players</article><article>Substitutions</article></main>',URL,200)
    assert 'Players' in d['text'] and 'Substitutions' in d['text']
    assert 'multiple_article_sections' in d['signals']

def test_rule_revision_does_not_merge_old_and_new_numbers():
    d=project_body(b'<article><deletedexplanation>six</deletedexplanation><addedexplanation>eight</addedexplanation></article>',URL,200)
    assert '[DELETED TEXT: six]' in d['text'] and '[INSERTED TEXT: eight]' in d['text']
    assert 'revision_markers_present' in d['signals']

def test_binary_not_mistaken_for_evidence():
    assert project_body(b'%PDF-1.4 random',URL,200)['status']=='unsupported_content'

def test_atom_links_are_extracted():
    d=project_body(b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Team</title><link href="https://example.org/team"/><summary>Update</summary></entry></feed>',URL,200)
    assert d['content_kind']=='discovery' and d['links'][0]['url']==URL

def test_redirect_charges_and_archives_each_request(tmp_path):
    responses=iter([SimpleNamespace(status_code=302,content=b'',headers={'Location':'https://example.org/article'}),SimpleNamespace(status_code=200,content=b'<article>Confirmed lineup</article>',headers={})])
    class Session:
        def __init__(self,route):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def get(self,*a,**kw):assert kw['allow_redirects'] is False;return next(responses)
    used=[]
    def reserve(url):used.append(url);return str(len(used))
    tool=ExperienceWebTools(RawArchive(tmp_path),session_factory=Session,dns_check=lambda u:None)
    r=tool.execute({'tool':'read_url','url':URL},run_id='test',reserve_request=reserve)
    assert len(used)==2 and len(r['requests'])==2 and r['document']['text']=='Confirmed lineup'
    assert len(list(RawArchive(tmp_path).observations()))==2

def test_provider_profile_only_applies_to_subprocess(tmp_path):
    seen=[]
    def runner(command,**kw):seen.append(command);raise subprocess.TimeoutExpired(command,1)
    provider=ExperienceProvider(tmp_path,protocol=ExperienceProtocol,effort='low',runner=runner)
    with pytest.raises(subprocess.TimeoutExpired):provider.decide({'task_id':'test'})
    assert 'model_reasoning_effort='+chr(34)+'low'+chr(34) in seen[0]
    assert seen[0][-1]=='-' and provider.configuration()['model_switch'] is False

def test_feedback_wrong_event_rejected(service):
    run=create(service)
    with pytest.raises(ValueError):service.feedback(run,{'comment':'x','event_sequence':999})
    assert service.get(run)['feedback_count']==0
