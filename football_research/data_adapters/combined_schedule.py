"""Daily display schedule with bounded, independently reported supplements."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .espn_scoreboard import ESPNScoreboard, utc, window
from .kleague_official import KLeagueOfficialSchedule
from .qiumibao_schedule import QiumibaoDailySchedule


def combine_reports(espn, qiumibao, kleague, *, at):
    start,end,_=window(at)
    rows=[]
    for report in (espn,qiumibao,kleague):
        for row in report.get("matches",[]):
            if start<=utc(row["kickoff"])<end:
                rows.append(row)
    def summary(report):
        return {k:v for k,v in report.items() if k not in {"matches","dates"}}
    states=[r.get("status") for r in (espn,qiumibao,kleague)]
    return {**espn,"source":"combined_public_schedule",
        "primary_source_status":espn.get("status","failed"),
        "primary_source_http_attempts":espn.get("http_attempts",0),
        "status":"ready" if all(s=="ready" for s in states) else "partial" if rows else "failed",
        "matches":rows,
        "dates":espn.get("dates",[]),
        "http_attempts":espn.get("http_attempts",0)+qiumibao.get("http_attempts",0)+kleague.get("requests",0),
        "supplemental_sources":{"qiumibao":summary(qiumibao),"kleague_official":summary(kleague)},
        "schedule_coverage_note":"ESPN seven-day feed plus today's bound China League One schedule and current-month official K1; missing 1X2 remains non-startable"}


def collect_schedule(root, *, at=None):
    root=Path(root);stamp=utc(at or datetime.now(timezone.utc));local=stamp.astimezone(ZoneInfo("Asia/Seoul"))
    def guarded(call):
        try:return call()
        except Exception as exc:
            return {"status":"failed","matches":[],"http_attempts":0,"requests":0,
                    "error_class":type(exc).__name__,"reason":"supplemental_source_initialization_failed"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        first=pool.submit(guarded,lambda:ESPNScoreboard(root).collect(stamp))
        second=pool.submit(guarded,lambda:QiumibaoDailySchedule(root.parent/"qiumibao").collect(stamp))
        third=pool.submit(guarded,lambda:KLeagueOfficialSchedule(root.parent/"kleague-official").collect_month(local.year,local.month,1))
        return combine_reports(first.result(),second.result(),third.result(),at=stamp)
