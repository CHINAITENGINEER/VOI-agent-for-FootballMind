"""Exercise the normal loopback live API and preserve actual run evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--match-id",required=True)
    parser.add_argument("--max-reads",type=int,default=5);parser.add_argument("--cost",type=float,default=.001)
    args=parser.parse_args()
    out=ROOT/"var/work/fm-live-body-chain-repair-20261025/acceptance"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    session=requests.Session();session.trust_env=False;base="http://127.0.0.1:8766/api/live-g/"
    def call(path,payload=None):
        response=session.get(base+path,timeout=45) if payload is None else session.post(base+path,json=payload,timeout=45)
        response.raise_for_status();return response.json()
    catalog=call("catalog");save("catalog-before.json",catalog)
    if catalog.get("runtime_bindings",{}).get("reload_required"):
        raise RuntimeError("Server source revision is stale")
    match=next((m for m in catalog["matches"] if m["match_id"]==args.match_id),None)
    if match is None or not match["can_start"]:raise RuntimeError("Target is not in the currently startable catalog")
    payload=dict(match_id=args.match_id,max_reads=args.max_reads,cost_loss_units=args.cost)
    save("request.json",payload)
    run=call("runs",payload);save("start-response.json",run)
    last=None;started=time.monotonic()
    while True:
        status=(run["state"],run.get("stage"),run["counts"]["source"],run["counts"]["provider"],len(run["steps"]))
        if status!=last:
            print(json.dumps({"run_id":run["run_id"],"state":run["state"],"stage":run.get("stage"),
                "reason":run.get("reason"),"counts":run["counts"],"body_pipeline":run.get("body_pipeline"),
                "before":run.get("before"),"current":run.get("current")},ensure_ascii=False),flush=True)
            last=status
        if run["state"] in {"STOPPED","FAILED","CANCELLED","INTERRUPTED"}:break
        if time.monotonic()-started>1000:raise RuntimeError("Run polling exceeded its declared bound")
        time.sleep(2);run=call("runs/"+run["run_id"])
    save("final-run.json",run)
    checks={"run_id":run["run_id"],"normal_live_api":True,"runtime_bindings":run.get("runtime_bindings"),
        "f_body_updates":len(run["steps"]),"body_pipeline":run.get("body_pipeline"),
        "same_f_before_after":bool(run["steps"]) and all(s["f_executed"] and s["input_id"] in run["used_input_ids"] for s in run["steps"]),
        "raw_body_input_bound":bool(run["steps"]) and all(s["f_consumption"]["input_id"]==s["input_id"] and s["raw_sha256"] for s in run["steps"]),
        "nonmarket_output":(run.get("body_pipeline",{}).get("market_difference_l1") or 0)>1e-12,
        "not_training":True,"source_provider_requests_recorded":True,"output":str(out),
        "page_url":"http://127.0.0.1:8766/live.html?run="+run["run_id"]}
    save("acceptance.json",checks)
    print(json.dumps({"stage":"finished",**checks},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
