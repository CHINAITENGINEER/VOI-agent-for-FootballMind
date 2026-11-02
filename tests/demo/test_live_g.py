from copy import deepcopy
from datetime import datetime, timedelta, timezone
import threading
import pytest
from demo import live_g as live
from demo.server import DemoApplication

STAMP="2026-10-19T10:00:00+00:00"
KICK="2026-10-19T19:00:00+00:00"
ARTICLE="https://www.tottenhamhotspur.com/news/2026/october/team-news"
BODY="James is unavailable for the match against Coventry City due to injury."

class FakeTools:
    def __init__(self, backend):
        self.backend=backend
    def prepare(self, action, public):
        return action
    def execute(self, action, **kwargs):
        self.backend.requests.append(deepcopy(action))
        article=action.get("url")==ARTICLE
        return {"document":{"document_id":"article" if article else "index-"+str(len(self.backend.requests)),
            "status":"OK","url":action.get("url","https://www.bing.com/search"),
            "text":BODY if article else "News directory", "raw_sha256":"a"*64,
            "source_published_at":STAMP,"structured_news":{"role":"article" if article else "index"},
            "links":[] if article else [{"title":"First team injury and availability update","url":ARTICLE,"source_published_at":STAMP}]}}

class FakeBackend:
    def __init__(self):
        self.requests=[];self.semantic_inputs=[];self.fail=False;self.on_semantics=None
    def initialize(self, match, folder):
        return {"market_initialization":{"prediction_cutoff":"2026-10-19T18:59:00+00:00",
                                         "probabilities":[.6,.25,.15]}}
    def identity(self, manifest, folder, expected):
        return {"status":"CONFIRMED","canonical":{"source_event_id":"test-match",
            "home_name":"Tottenham Hotspur","away_name":"Coventry City","kickoff":KICK},
            "aliases":{"home":["Tottenham Hotspur"],"away":["Coventry City"]}}
    def tools(self,*args):return FakeTools(self)
    def semantics(self, folder, public, timeout):
        self.semantic_inputs.append(deepcopy(public))
        if self.on_semantics:self.on_semantics()
        if self.fail:raise ValueError("invalid semantic response")
        return {"input":public,"response":{"claims":[{"team":"home","entity":"James","kind":"availability",
            "value":"unavailable","scope":"this_match","statement":BODY,"quotes":[BODY],"concept":None}]}}

@pytest.fixture
def app(tmp_path,monkeypatch):
    monkeypatch.setattr(live,"key_from_environment",lambda:"test-only")
    b=FakeBackend()
    a=live.LiveGApplication(tmp_path,backend=b,clock=lambda:STAMP)
    live.save(tmp_path/"catalog.json",{"status":"ready","updated_at":STAMP,"matches":[{
        "match_id":"fixture-test","fixture_id":"fixture-test","source_match_id":"test","home":"Tottenham Hotspur",
        "away":"Coventry City","kickoff":KICK,"supported":True}]})
    return a

def select_first(model,menu,remaining,cost):
    return {"action_id":menu[0]["action_id"],"reason":"test_selection","candidates":[]}


def decision_run():
    return {"run_id":"lg-"+"f"*32,"fixture":{"physical_match_id":"test-match"},"prior":[.6,.25,.15],
        "current":[.6,.25,.15],"used_input_ids":[],"evidence_events":[],"steps":[],"decisions":[],
        "counts":{"body_attempts":0,"provider":0,"source":0},"max_reads":1,"cost_loss_units":.001,"revision":0}


def test_exact_gain_tie_prefers_observed_recent_day_without_changing_scores(app,monkeypatch):
    def tied(*args):
        return {"action_id":"a","reason":"positive_learned_two_step_net_value","candidates":[
            {"action_id":"a","two_step_net_value":.1},{"action_id":"b","two_step_net_value":.1}]}
    monkeypatch.setattr(live.g_model,"choose",tied)
    r=decision_run()
    d=app._decision(r,[{"action_id":"a","url":"https://club.test/news/a","title":"Older source statement","source_published_at":"2026-10-17T12:00:00Z"},
        {"action_id":"b","url":"https://club.test/news/b","title":"Newer source statement","source_published_at":"2026-10-18T12:00:00Z"}])
    assert d["action_id"]=="b" and d["original_model_tie_choice"]=="a"
    assert d["tie_break"]=="observed_publication_day_desc_then_original_action_id"
    assert [c["two_step_net_value"] for c in d["candidates"]]==[.1,.1]
    assert app.backend.requests==[]


def test_observed_recency_never_overrides_g_stop(app,monkeypatch):
    monkeypatch.setattr(live.g_model,"choose",lambda *a:{"action_id":None,"reason":"nonpositive_net_value",
        "candidates":[{"action_id":"a","two_step_net_value":-.1},{"action_id":"b","two_step_net_value":-.1}]})
    d=app._decision(decision_run(),[{"action_id":i,"url":"https://club.test/news/"+i,"title":"Observed source statement",
        "source_published_at":"2026-10-18T12:00:00Z"} for i in ("a","b")])
    assert d["action_id"] is None and "tie_break" not in d


def test_no_more_body_is_requested_when_semantic_budget_is_exhausted(app,monkeypatch):
    r=decision_run();r["counts"]["provider"]=live.MAX_PROVIDER_CALLS
    monkeypatch.setattr(live.g_model,"choose",lambda *a:pytest.fail("No processable body budget remains"))
    d=app._decision(r,[{"action_id":"a","url":"https://club.test/news/a","title":"Observed source statement"}])
    assert d["action_id"] is None and d["reason"]=="semantic_budget_exhausted"
    assert app.backend.requests==[]


def test_final_body_keeps_one_counted_repair_and_timeout_respects_run_deadline(app):
    from football_research.acquisition.semantic_f import public_input
    r=decision_run();r.update(created_at="2026-10-19T09:46:00+00:00",
        prediction_cutoff="2026-10-19T18:59:00+00:00")
    r["counts"]["provider"]=4
    app.cancel[r["run_id"]]=threading.Event()
    public=public_input({"physical_match_id":"test-match","home_name":"Tottenham Hotspur",
        "away_name":"Coventry City","kickoff":KICK},{"url":ARTICLE,"text":BODY,"raw_sha256":"a"*64},"body")
    timeouts=[]
    def initial(folder,public,timeout):
        timeouts.append(timeout);raise ValueError("Claim must cite an exact acquired text span")
    def repair(folder,public,timeout):
        timeouts.append(timeout);return {"input":public,"response":{"claims":[]}}
    app.backend.semantics=initial;app.backend.repair_semantics=repair
    result=app._extract_body(r,public,app._folder(r["run_id"]))
    assert result["response"]["claims"]==[] and r["counts"]["provider"]==6
    assert [a["kind"] for a in r["semantic_attempts"]]==["initial","validation_repair"]
    assert [a["status"] for a in r["semantic_attempts"]]==["FAILED","OK"]
    assert timeouts==[60,60]


def test_later_invalid_quote_is_rejected_without_losing_valid_f_output(app,monkeypatch):
    monkeypatch.setattr(live.g_model,"choose",select_first)
    second=ARTICLE+"?second=1";original_execute=FakeTools.execute
    def two_articles(self,action,**kwargs):
        returned=original_execute(self,action,**kwargs)
        doc=returned["document"]
        if action.get("url")==second:
            doc.update(document_id="second-body",url=second,text=BODY+" A second acquired body.",structured_news={"role":"article"},links=[])
        elif doc["structured_news"]["role"]=="index":
            doc["links"].append({"url":second,"title":"Second actual source statement","source_published_at":STAMP})
        return returned
    monkeypatch.setattr(FakeTools,"execute",two_articles)
    original_semantics=app.backend.semantics
    def semantic(folder,public,timeout):
        if public["url"]==second:
            app.backend.semantic_inputs.append(deepcopy(public))
            raise ValueError("Claim must cite an exact acquired text span")
        return original_semantics(folder,public,timeout)
    app.backend.semantics=semantic
    r=app.start({"match_id":"fixture-test","max_reads":2});app.workers[r["run_id"]].join(timeout=10)
    r=app.get(r["run_id"])
    assert r["state"]=="STOPPED",r["reason"]
    assert len(r["steps"])==1 and len(r["used_input_ids"])==1
    assert r["current"]==r["steps"][0]["prediction_after"]
    assert r["body_rejections"][-1]["reason"]=="semantic_validation_failed"
    assert r["body_rejections"][-1]["prediction_after"] is None and r["body_rejections"][-1]["gain"] is None

def finish(app):
    r=app.start({"match_id":"fixture-test","max_reads":1})
    app.workers[r["run_id"]].join(timeout=10)
    assert not app.workers[r["run_id"]].is_alive()
    return app.get(r["run_id"])

def test_actual_paired_f_consumes_body_and_after_is_recomputed(app,monkeypatch):
    monkeypatch.setattr(live.g_model,"choose",select_first)
    original=live.f_model.predict;calls=[]
    def spy(model,row):
        calls.append(deepcopy(row));return original(model,row)
    monkeypatch.setattr(live.f_model,"predict",spy)
    r=finish(app)
    assert r["state"]=="STOPPED",r["reason"]
    assert len(calls)==2 and calls[0]["used_input_ids"]==[]
    assert len(r["steps"])==1 and r["steps"][0]["input_id"] in calls[1]["used_input_ids"]
    assert r["steps"][0]["body"]==BODY and r["steps"][0]["f_executed"]
    assert r["steps"][0]["claims"][0]["quotes"]==[BODY]
    assert r["outcome"] is None and r["gain"] is None and not r["g_training_eligible"]
    assert r["model_bindings"]["f"]==app.g["f_labeler_sha256"]


def test_inflight_phases_are_persisted_without_publishing_after(app, monkeypatch):
    """A slow review must be visible as pending, never as failed f execution."""
    monkeypatch.setattr(live.g_model, "choose", select_first)
    snapshots = []
    def snapshot():
        rid = next(iter(app.cancel))
        r = live.read(app._folder(rid) / "run.json")
        snapshots.append(deepcopy(r))
        assert r["steps"] == [] and r["used_input_ids"] == []
        assert r["current"] == r["before"]
        return r
    def extracting():
        r = snapshot()
        a = r["semantic_attempts"][-1]
        assert a["kind"] == "initial" and a["status"] == "PENDING"
        assert a["action_id"] == r["active_body"]["action_id"]
        assert a["document_id"] == r["active_body"]["document_id"]
        assert "正在抽取" in r["stage"]
    app.backend.on_semantics = extracting
    def review(folder, public, claims, timeout):
        r = snapshot()
        assert "正在复核" in r["stage"]
        assert r["pending"] == "claim-review-" + public["input_id"]
        assert r["semantic_attempts"][-1]["status"] == "PENDING"
        return {"review": {"retained_indices": [0], "rejected": [], "status": "completed"}}
    app.backend.review_claims = review
    original = live.f_model.predict
    def compute(model, row):
        if row["used_input_ids"]:
            r = snapshot()
            assert "正在执行固定 f" in r["stage"]
            assert r["active_body"]["status"] == "f_running"
            assert r["pending"] is None
        return original(model, row)
    monkeypatch.setattr(live.f_model, "predict", compute)
    completed = finish(app)
    assert len(snapshots) == 3
    assert len(completed["steps"]) == 1
    assert completed["active_body"]["status"] == "f_committed"
    assert completed["steps"][0]["f_executed"] is True

def test_real_g_receives_only_prequery_state(app,monkeypatch):
    original=live.g_model.choose;menus=[]
    def spy(model,menu,remaining,cost):
        menus.append(deepcopy(menu))
        for c in menu:
            assert set(c)=={"action_id","query"}
            assert "body" not in c["query"]["public"]
            assert BODY not in str(c)
        return original(model,menu,remaining,cost)
    monkeypatch.setattr(live.g_model,"choose",spy)
    r=finish(app)
    assert r["state"]=="STOPPED",r["reason"]
    assert menus and r["decisions"]

def test_semantic_failure_cannot_publish_after(app,monkeypatch):
    monkeypatch.setattr(live.g_model,"choose",select_first);app.backend.fail=True
    r=finish(app)
    assert r["state"]=="FAILED"
    assert r["steps"]==[] and r["used_input_ids"]==[] and r["current"]==r["before"]
    assert r["counts"]["provider"]==1 and r["pending"]


def test_first_candidate_quote_rejection_does_not_claim_body_consumption(app,monkeypatch):
    monkeypatch.setattr(live.g_model,"choose",select_first)
    def rejected(*args):raise ValueError("Claim must cite an exact acquired text span")
    app.backend.semantics=rejected
    r=finish(app)
    assert r["state"]=="STOPPED" and r["reason"]=="budget_exhausted"
    assert r["counts"]["body_attempts"]==1 and r["counts"]["provider"]==1
    assert r["steps"]==[] and r["used_input_ids"]==[] and r["current"]==r["before"]
    assert r["body_pipeline"]["semantic_rejected_bodies"]==1
    assert r["body_rejections"][0]["prediction_after"] is None
    assert r["body_rejections"][0]["gain"] is None and r["pending"] is None


def test_source_review_rejects_claim_without_rewriting_original_extraction(app,monkeypatch):
    monkeypatch.setattr(live.g_model,'choose',select_first)
    app.backend.review_claims=lambda folder,public,claims,timeout:{'review':{
        'retained_indices':[],'rejected':[{'claim_index':0,'verdict':'unsupported','reason':'Controlled reviewer rejection'}],
        'status':'reviewed'}}
    r=finish(app)
    assert r['state']=='STOPPED' and r['counts']['provider']==2
    assert [a['kind'] for a in r['semantic_attempts']]==['initial','claim_review']
    assert len(r['steps'])==1 and r['steps'][0]['claims']==[]
    assert r['steps'][0]['extracted_claims_count']==1 and r['current']==r['before']
    original=live.read(app._folder(r['run_id'])/r['steps'][0]['semantic_record'])
    assert len(original['response']['claims'])==1 and r['used_input_ids']==[r['steps'][0]['input_id']]
    assert r['body_pipeline']['review_rejected_claims']==1 and r['gain'] is None


def test_source_review_failure_cannot_publish_after(app,monkeypatch):
    monkeypatch.setattr(live.g_model,'choose',select_first)
    def reject(*args):raise ValueError('Live claim review index coverage differs')
    app.backend.review_claims=reject
    r=finish(app)
    assert r['state']=='STOPPED' and r['counts']['provider']==2
    assert r['steps']==[] and r['used_input_ids']==[] and r['current']==r['before']
    assert r['body_rejections'][0]['f_executed'] is False and r['gain'] is None


def test_empty_claims_skip_review_http(app,monkeypatch):
    monkeypatch.setattr(live.g_model,'choose',select_first)
    app.backend.semantics=lambda folder,public,timeout:{'input':public,'response':{'claims':[]}}
    app.backend.review_claims=lambda *args:pytest.fail('Empty claims cannot cause a review API request')
    r=finish(app)
    assert r['state']=='STOPPED' and len(r['steps'])==1
    assert r['counts']['provider']==1 and r['current']==r['before']
    assert r['steps'][0]['claim_review']['retained_indices']==[]


def test_thinking_review_timeout_stays_within_remaining_run_time(app):
    from football_research.acquisition.semantic_f import public_input
    r=decision_run();r.update(created_at='2026-10-19T09:46:00+00:00',prediction_cutoff='2026-10-19T18:59:00+00:00',semantic_attempts=[])
    app.cancel[r['run_id']]=threading.Event()
    public=public_input({'physical_match_id':'test-match','home_name':'Tottenham Hotspur',
        'away_name':'Coventry City','kickoff':KICK},{'url':ARTICLE,'text':BODY},'body')
    record=app.backend.semantics(app._folder(r['run_id']),public,60);deadlines=[]
    def review(folder,public,claims,timeout):
        deadlines.append(timeout)
        return {'review':{'retained_indices':[0],'rejected':[],'status':'completed'}}
    app.backend.review_claims=review
    app._review_body(r,record,app._folder(r['run_id']))
    assert deadlines==[60] and r['counts']['provider']==1

def test_stop_while_provider_runs_does_not_commit_after(app,monkeypatch):
    monkeypatch.setattr(live.g_model,"choose",select_first)
    app.backend.on_semantics=lambda:next(iter(app.cancel.values())).set()
    r=finish(app)
    assert r["state"]=="CANCELLED" and r["steps"]==[]
    assert r["current"]==r["before"]

def test_late_provider_result_archived_but_not_used(app,monkeypatch):
    monkeypatch.setattr(live.g_model,"choose",select_first)
    app.backend.on_semantics=lambda:setattr(app,"clock",lambda:"2026-10-19T19:01:00+00:00")
    r=finish(app)
    assert r["state"]=="STOPPED" and r["steps"]==[]
    assert list(app._folder(r["run_id"]).glob("semantics/*.json"))

@pytest.mark.parametrize("payload",[
    {"match_id":"missing"},{"match_id":"fixture-test","max_reads":True},
    {"match_id":"fixture-test","cost_loss_units":float("nan")},
    {"match_id":"fixture-test","url":"https://example.com"}])
def test_untrusted_start_configuration_is_rejected(app,payload):
    with pytest.raises(ValueError):app.start(payload)
    assert app.backend.requests==[]

def test_stale_or_imminent_match_does_not_start(app):
    app.clock=lambda:"2026-10-19T10:31:00+00:00"
    with pytest.raises(ValueError):app.start({"match_id":"fixture-test"})
    app.clock=lambda:"2026-10-19T18:31:00+00:00"
    assert not app.catalog()["matches"][0]["can_start"]

def test_restart_marks_incomplete_run_interrupted(app):
    rid="lg-"+"a"*32
    live.save(app._folder(rid)/"run.json",{"run_id":rid,"state":"RUNNING","revision":1,"pending":"real-inflight"})
    r=app.get(rid)
    assert r["state"]=="INTERRUPTED" and r["pending"]=="real-inflight"
    assert app.backend.requests==[]

def test_existing_frozen_models_not_promoted(app):
    assert app.g["online_selection_allowed"] is False
    assert app.g["production_ready"] is False
    assert live.read(app.g_path)["online_selection_allowed"] is False


def test_pending_host_review_is_exact_and_bounded():
    fixture = {"home_name": "Sheffield United", "away_name": "Derby County"}
    approved = live._review_pending_host(
        {"url": "https://www.sufc.example/news/sheffield-united-derby-preview",
         "title": "Sheffield United v Derby County preview"}, fixture, ["www.bing.com"], set(), 0)
    assert approved["status"] == "approved"
    assert approved["reason"] == "exact_host_article_shape_and_team_binding"
    blocked = live._review_pending_host(
        {"url": "https://unrelated.example/news/general-update",
         "title": "General team update"}, fixture, ["www.bing.com"], set(), 0)
    assert blocked["status"] == "rejected" and blocked["reason"] == "non_sports_article_signal"
    tourism = live._review_pending_host(
        {"url": "https://www.tripadvisor.com/Attractions-g1-Activities-Sheffield.html",
         "title": "The best things to do in Sheffield"}, fixture, ["www.bing.com"], set(), 0)
    assert tourism["status"] == "rejected" and tourism["reason"] == "non_sports_article_signal"
    capped = live._review_pending_host(
        {"url": "https://another.example/news/sheffield-united-preview",
         "title": "Sheffield United preview"}, fixture, ["www.bing.com"], set(), live.MAX_DYNAMIC_HOSTS)
    assert capped["reason"] == "dynamic_host_review_budget_exhausted"


def test_directory_candidates_keep_fixture_context_and_drop_gallery_noise():
    fixture = {"home_name": "Augsburg", "away_name": "Union Berlin",
               "home_aliases": ["奥格斯堡", "Augsburg"], "away_aliases": ["柏林联合", "Union Berlin"]}
    assert live._directory_candidate_reason(
        "https://www.fcaugsburg.de/article/matchday-heimspiel-gegen-union-berlin-23596",
        "Profis Matchday: Heimspiel gegen Union Berlin", fixture, [], set()) is None
    assert live._directory_candidate_reason(
        "https://www.fcaugsburg.de/article/bildergalerie-fca-dreht-spiel-in-frankfurt-23425",
        "Bildergalerie: FCA dreht Spiel in Frankfurt", fixture, [], set()) == "directory_non_match_category"
    assert live._directory_candidate_reason(
        "https://www.fcaugsburg.de/article/ibrahimovic-mehr-verlaesslichkeit-23592",
        'Ibrahimović: "Man sieht, wie viel Energie in der Mannschaft steckt"', fixture, [], set()) == "directory_context_not_match_bound"
    assert live._directory_candidate_reason(
        "https://www.fc-union-berlin.de/de/meldungen/unioner-testen-in-und-gegen-wolfsburg-id",
        "Unioner testen in und gegen Wolfsburg", fixture, [], set()) == "directory_context_not_match_bound"


def test_evidence_highlights_are_exact_quote_bound_and_keep_unverified_claims():
    body = "The manager confirmed that Player A will miss this match through injury."
    claims = [{"entity": "Player A", "team": "home", "kind": "availability", "scope": "this_match",
               "value": "unavailable", "statement": "Player A is unavailable.",
               "quotes": ["Player A will miss this match through injury."]},
              {"entity": "Player B", "team": "away", "kind": "other", "scope": "team_context",
               "value": "unmodeled", "statement": "No bound claim.", "quotes": []}]
    highlights = live._evidence_highlights(claims, body)
    assert highlights[0]["quote_verified"] and highlights[0]["body_start"] >= 0
    assert highlights[0]["body_context"] in body
    assert highlights[1]["quote"] is None and not highlights[1]["quote_verified"]


def test_evidence_highlights_rank_fixture_bound_claims_for_display_only():
    body = "Player A will miss this match through injury."
    claims = [
        {"entity": "Generic", "team": "home", "kind": "other", "scope": "team_context",
         "value": "unmodeled", "statement": "Context", "quotes": []},
        {"entity": "Player A", "team": "home", "kind": "availability", "scope": "this_match",
         "value": "unavailable", "statement": "Player A is unavailable.",
         "quotes": ["Player A will miss this match through injury."]},
    ]
    highlights = live._evidence_highlights(claims, body)
    assert highlights[0]["entity"] == "Player A"
    assert highlights[0]["importance_rank"] == 1
    assert highlights[0]["importance_label"] == "高"
    assert highlights[0]["priority_basis"] == "deterministic_claim_priority_projection_v1"
    assert highlights[1]["importance_rank"] == 2


def test_generic_search_without_registered_club_still_reaches_paired_f(app,monkeypatch):
    monkeypatch.setattr(live.g_model,"choose",select_first)
    original = app.backend.identity
    def identity(manifest, folder, expected):
        binding = original(manifest, folder, expected)
        binding["canonical"].update(home_name="Parma test club",away_name="Torino test club")
        return binding
    app.backend.identity = identity
    original_execute = FakeTools.execute
    def bound_search(self, action, **kwargs):
        result = original_execute(self, action, **kwargs)
        for link in result["document"]["links"]:
            link["title"] = "Parma test club first team injury update"
        return result
    monkeypatch.setattr(FakeTools, "execute", bound_search)
    r = finish(app)
    assert r["state"] == "STOPPED", r["reason"]
    assert app.backend.requests[0]["tool"] == "search_web"
    assert "Parma test club" in app.backend.requests[0]["query"]
    assert len(r["steps"]) == 1 and r["steps"][0]["f_executed"]
    assert r["steps"][0]["input_id"] in r["used_input_ids"]

def test_http_dispatch_uses_live_service(app,tmp_path):
    server=DemoApplication(tmp_path)
    server._live_g=app
    assert server.get("/api/live-g/catalog",{})["mode"]=="live_experiment"
    with pytest.raises(ValueError):server.get("/api/live-g/catalog",{"path":["secret"]})
    with pytest.raises(FileNotFoundError):server.get("/api/live-g/unknown",{})


def test_no_data_body_never_calls_semantic_or_f_after(app, monkeypatch):
    monkeypatch.setattr(live.g_model, "choose", select_first)
    original = FakeTools.execute
    def no_body(self, action, **kwargs):
        result = original(self, action, **kwargs)
        if action.get("url") == ARTICLE:
            result["document"].update(status="NO_DATA", text="", structured_news={})
        return result
    monkeypatch.setattr(FakeTools, "execute", no_body)
    r = finish(app)
    assert r["counts"]["body_attempts"] == 1
    assert r["counts"]["provider"] == 0 and r["steps"] == []
    assert r["current"] == r["before"] and r["gain"] is None


def test_empty_directories_trigger_bounded_search_diversity(app, monkeypatch):
    original = FakeTools.execute
    def empty(self, action, **kwargs):
        result = original(self, action, **kwargs)
        result["document"]["links"] = []
        return result
    monkeypatch.setattr(FakeTools, "execute", empty)
    r = finish(app)
    assert [x["tool"] for x in app.backend.requests] == ["read_url", "read_url", "search_web", "search_web", "search_web", "search_web", "search_web"]
    assert r["reason"] == "discovery_no_candidates" and r["steps"] == []
    assert r["counts"]["provider"] == 0


def test_discovery_failures_do_not_become_g_stop(app, monkeypatch):
    original = FakeTools.execute
    def failed(self, action, **kwargs):
        result = original(self, action, **kwargs)
        result["document"].update(status="FAILED", http_status=301, links=[])
        return result
    monkeypatch.setattr(FakeTools, "execute", failed)
    monkeypatch.setattr(live.g_model, "choose", lambda *a: pytest.fail("No candidates, no g decision"))
    r = finish(app)
    assert r["state"] == "FAILED" and r["reason"] == "discovery_sources_failed"
    assert r["decisions"] == r["steps"] == []
    assert r["counts"]["provider"] == 0


def test_search_route_retry_is_one_recorded_budgeted_attempt(app, monkeypatch):
    original = FakeTools.execute
    monkeypatch.setattr(FakeTools, "default_route", "environment", raising=False)
    def fail_environment(self, action, **kwargs):
        result = original(self, action, **kwargs)
        if action["tool"] == "read_url" or getattr(self, "search_route", self.default_route) != "direct":
            result["document"].update(status="FAILED", http_status=301, links=[])
        return result
    monkeypatch.setattr(FakeTools, "execute", fail_environment)
    r = finish(app)
    retries = [a for a in r["attempts"] if a["purpose"] == "search_transport_retry"]
    assert len(retries) == 1 and retries[0]["transport_route"] == "direct"
    assert r["counts"]["source"] == len(r["attempts"]) <= live.MAX_SOURCE_REQUESTS
    assert retries[0]["request_id"] != r["attempts"][2]["request_id"]

