"""Explicitly bound public Qiumibao/Zhibo8 fixture projection.

The daily feed has source league IDs without names. A real source match page
must first bind that league ID to a declared competition. Fixtures are retained
as schedule-only until another independently joined source supplies a prior.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import requests
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes

SCHEMA = "footballmind-qiumibao-bound-daily-schedule-v1"
SHANGHAI = ZoneInfo("Asia/Shanghai")
STATUSES = {"1":"scheduled", "2":"live", "3":"finished", "4":"postponed", "6":"cancelled", "7":"unknown"}
CONTRACT_ROOT = Path(__file__).resolve().parents[2]/"config/source-contracts/qiumibao-20261025"


def bind_league(payload, page_html, page_url, page_observation, *, match_id, competition_name):
    """Bind one actual feed ID and its own source match-page metadata."""
    parsed = urlsplit(page_url)
    if (parsed.scheme != "https" or parsed.hostname not in {"www.zhibo8.com", "www.zhibo8.cc", "news.zhibo8.com"}
            or not re.search(r"/match"+re.escape(str(match_id))+r"vplayer\.htm$", parsed.path)):
        raise ValueError("Source match-page URL/ID binding differs")
    if not isinstance(payload, dict) or not isinstance(payload.get("list"), list):
        raise ValueError("Daily source list missing")
    hits=[r for r in payload["list"] if isinstance(r,dict) and str(r.get("id"))==str(match_id)]
    if len(hits)!=1:raise ValueError("Match ID is not unique in observed feed")
    r=hits[0]
    if r.get("type")!="football" or not r.get("leagueid"):
        raise ValueError("Football source league identity missing")
    soup=BeautifulSoup(page_html,"html.parser")
    title=soup.title.get_text(" ",strip=True) if soup.title else ""
    meta=soup.find("meta",attrs={"name":"keywords"})
    keywords=str(meta.get("content") or "") if meta else ""
    box=soup.find(attrs={"match-time":True})
    if (competition_name not in keywords.split(",") or competition_name not in title
            or not r.get("home_team") or not r.get("visit_team")
            or r["home_team"] not in title or r["visit_team"] not in title
            or box is None or box.get("match-time")!=r.get("start")):
        raise ValueError("Source page competition/teams/kickoff differ from observed match")
    return dict(source="qiumibao",source_league_id=str(r["leagueid"]),competition_name=competition_name,
        binding_basis="same_source_unique_match_id_page_keywords_title_home_away_exact_kickoff",
        reference_match_id=str(match_id),reference_url=page_url,page_observation=page_observation,
        independently_verified=False)


def bind_league_detail(payload, detail, detail_observation, *, match_id, competition_name):
    """Prefer the source's explicit league-ID/name/season object when present."""
    if not isinstance(payload,dict) or not isinstance(payload.get("list"),list):
        raise ValueError("Daily source list missing")
    hits=[r for r in payload["list"] if isinstance(r,dict) and str(r.get("id"))==str(match_id)]
    if len(hits)!=1 or not isinstance(detail,dict):raise ValueError("Unique source match binding required")
    row=hits[0];league=detail.get("league") or {}
    if (row.get("type")!="football" or detail.get("type")!="zuqiu"
            or str(detail.get("match_id"))!=str(match_id)
            or str(league.get("id"))!=str(row.get("leagueid"))
            or league.get("ziliao_name")!=competition_name
            or str(detail.get("season"))!=str(row.get("sdate",""))[:4]
            or detail.get("match_time")!=row.get("start")
            or any(str(detail.get(a))!=str(row.get(b)) for a,b in
                (("home_id","home_id"),("visit_id","visit_id"),("home_team","home_team"),("visit_team","visit_team")))):
        raise ValueError("Structured source detail league/season/teams/kickoff differs")
    return dict(source="qiumibao",source_league_id=str(row["leagueid"]),competition_name=competition_name,
        binding_basis="same_source_unique_match_id_explicit_league_name_id_season_teams_kickoff",
        reference_match_id=str(match_id),detail_observation=detail_observation,independently_verified=False)


def parse(payload, observation, *, source_date, league_bindings):
    day=datetime.strptime(source_date,"%Y-%m-%d").date()
    if not isinstance(payload,dict) or not isinstance(payload.get("list"),list):
        raise ValueError("Daily source list missing")
    bindings={b["source_league_id"]:b for b in league_bindings}
    matches,rejected,ignored=[],[],0
    seen=set()
    for r in payload["list"]:
        if not isinstance(r,dict):
            rejected.append(dict(match_id=None,reason="invalid_source_row"));continue
        binding=bindings.get(str(r.get("leagueid")))
        if r.get("type")!="football" or binding is None:
            ignored+=1;continue
        try:
            mid=str(r["id"])
            if not mid.isdigit() or mid in seen:raise ValueError("invalid_or_duplicate_match_id")
            kickoff=datetime.strptime(r["start"],"%Y-%m-%d %H:%M").replace(tzinfo=SHANGHAI)
            if kickoff.date()!=day or str(r.get("sdate"))!=source_date:
                raise ValueError("daily_source_date_mismatch")
            hid,aid=str(r.get("home_id") or ""),str(r.get("visit_id") or "")
            if not hid or not aid or hid==aid or not r.get("home_team") or not r.get("visit_team"):
                raise ValueError("incomplete_home_away_identity")
            source_state=str(r.get("state"));status=STATUSES.get(source_state,"unknown")
            if str(r.get("period_cn")) in {"取消","延期"}:
                status="cancelled" if r["period_cn"]=="取消" else "postponed"
            if status=="scheduled" and str(r.get("period_cn") or "").strip():
                raise ValueError("conflicting_scheduled_status")
            def score(key):
                value=str(r.get(key,""))
                return int(value) if status in {"live","finished"} and value.isdigit() else None
            seen.add(mid)
            url=f"https://www.zhibo8.com{r['url']}" if str(r.get("url","")).startswith("/zhibo/zuqiu/") else None
            matches.append(dict(match_id="qiumibao:match:"+mid,fixture_id="qiumibao:match:"+mid,
                source="qiumibao",source_match_id=mid,source_league_id=str(r["leagueid"]),
                source_url=url,home=r["home_team"],away=r["visit_team"],
                home_team=r["home_team"],away_team=r["visit_team"],
                home_team_source_id=hid,away_team_source_id=aid,
                competition_key=binding["competition_name"],source_competition_name=binding["competition_name"],
                competition_group="asia",kickoff=kickoff.astimezone(timezone.utc).isoformat(),
                kickoff_at=kickoff.astimezone(timezone.utc).isoformat(),source_timezone="Asia/Shanghai",
                original_time=r["start"],time_confirmed=True,status=status,status_detail=source_state,
                home_score=score("home_score"),away_score=score("visit_score"),
                observed_at=observation["observed_at"],raw_observation=observation,
                league_binding=binding,sources=["qiumibao"],quote_observed=False,quote_snapshot=None,
                quote_observed_at=None,source_available_at=None,supported=False,analysis_source=None,
                reason="当前赛程已取得；本来源未返回完整90分钟胜平负市场先验"))
        except (KeyError,ValueError,TypeError) as exc:
            rejected.append(dict(match_id=r.get("id"),reason=str(exc)))
    return matches,dict(response_rows=len(payload["list"]),ignored_unbound_or_other_sport=ignored,
                        rejected=rejected,market_prior_available=False)


def default_bindings():
    """Reproduce a reviewed same-source league link from immutable source bytes."""
    from football_spider.models.contracts import Observation
    contract=json.loads((CONTRACT_ROOT/"manifest.json").read_text(encoding="utf-8"))
    archive=RawArchive(CONTRACT_ROOT/"archive")
    daily=Observation.model_validate(contract["daily_observation"])
    page=Observation.model_validate(contract["match_page_observation"])
    payload=json.loads(archive.read_body(daily))
    if contract.get("detail_observation"):
        detail=Observation.model_validate(contract["detail_observation"])
        return [bind_league_detail(payload,json.loads(archive.read_body(detail)),detail.model_dump(mode="json"),
            match_id=contract["reference_match_id"],competition_name=contract["competition_name"])]
    html=archive.read_body(page).decode("utf-8")
    return [bind_league(payload,html,contract["reference_url"],page.model_dump(mode="json"),
                        match_id=contract["reference_match_id"],competition_name=contract["competition_name"])]


class QiumibaoDailySchedule:
    def __init__(self,root,*,transport=None,clock=None,league_bindings=None):
        self.root=Path(root);self.archive=RawArchive(self.root/"archive")
        self.transport=transport or self._get
        self.clock=clock or (lambda:datetime.now(timezone.utc))
        self.bindings=league_bindings if league_bindings is not None else default_bindings()

    @staticmethod
    def _get(endpoint):
        with requests.Session() as session:
            session.auth=lambda request:request
            with session.get(endpoint,timeout=(8,20),allow_redirects=False,stream=True,
                headers={"User-Agent":"FootballMind/0.1 public fixture research"}) as response:
                size=0;chunks=[]
                for chunk in response.iter_content(16384):
                    size+=len(chunk)
                    if size>4_000_000:raise ValueError("Daily schedule response exceeds byte budget")
                    chunks.append(chunk)
                return response.status_code,b"".join(chunks)

    def collect(self,at=None):
        stamp=at or self.clock()
        if stamp.tzinfo is None:raise ValueError("Explicit observation timezone required")
        day=stamp.astimezone(SHANGHAI).date().isoformat()
        endpoint=f"https://bifen4pc.qiumibao.com/json/{day}/list.htm"
        cache=self.root/"cache"/(day+".json");previous=None
        if cache.exists():
            try:
                previous=json.loads(cache.read_text(encoding="utf-8"))
                observed=datetime.fromisoformat(previous["observed_at"].replace("Z","+00:00"))
                if 0 <= (stamp-observed).total_seconds()<900:
                    return {**previous,"http_attempts":0,"cache_hit":True}
            except (ValueError,KeyError,TypeError,OSError):previous=None
        observation=None
        try:
            status,body=self.transport(endpoint)
            observation=self.archive.save("qiumibao",endpoint,{},body,status=status,observed_at=self.clock())
            if status!=200:raise ValueError("Daily schedule HTTP "+str(status))
            matches,audit=parse(json.loads(body),observation.model_dump(mode="json"),source_date=day,
                                league_bindings=self.bindings)
            result=dict(schema=SCHEMA,status="ready",matches=matches,audit=audit,
                http_attempts=1,cache_hit=False,observed_at=observation.observed_at.isoformat(),
                observation=observation.model_dump(mode="json"),coverage_scope="one Beijing calendar date; explicitly bound competitions only")
            atomic_write(cache,json_bytes(result))
        except (requests.RequestException,ValueError,TypeError,KeyError) as exc:
            if observation is None:
                observation=self.archive.save("qiumibao",endpoint,{},b"",error_class=type(exc).__name__,observed_at=self.clock())
            result=dict(schema=SCHEMA,status="failed",matches=[{**m,"source_stale":True} for m in (previous or {}).get("matches",[])],
                http_attempts=1,cache_hit=False,observed_at=(previous or {}).get("observed_at"),
                error_class=type(exc).__name__,failure_observation=observation.model_dump(mode="json"))
        atomic_write(self.root/"last-report.json",json_bytes(result));return result
