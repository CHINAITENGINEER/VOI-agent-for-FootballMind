"""Restart only the verified local demo process; preserve all run/model files."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "var/work/fm-live-body-chain-repair-20261025/server"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--pid",type=int,required=True)
    parser.add_argument("--port",type=int,default=8766);args=parser.parse_args()
    script=f"Get-CimInstance Win32_Process -Filter 'ProcessId={args.pid}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run(["powershell","-NoProfile","-Command",script],capture_output=True,text=True,check=True)
    process=json.loads(result.stdout)
    line=process["CommandLine"]
    if not re.search(r"-m\s+demo(?:\.server)?(?:\s|$)",line) or not re.search(rf"--port\s+{args.port}(?:\s|$)",line):
        raise RuntimeError("Target process is not the expected demo module and port")
    # Existing nonterminal runs are left alone; a restart must not interrupt a user experiment.
    store=ROOT/"var/parallel/agent/round2/runs"
    for p in (store/"live-g/runs").glob("lg-*/run.json"):
        row=json.loads(p.read_text(encoding="utf-8-sig"))
        if row["state"] not in {"STOPPED","FAILED","CANCELLED","INTERRUPTED"}:
            raise RuntimeError("A saved live run is nonterminal; inspect before restarting")
    OUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Only an explicitly verified PID is terminated. No directory/file is removed.
    subprocess.run(["powershell","-NoProfile","-Command",f"Stop-Process -Id {args.pid} -ErrorAction Stop"],check=True)
    log=OUT/("demo-"+stamp+".log")
    with log.open("ab") as stream:
        worker=subprocess.Popen([sys.executable,"-u","-B","-m","demo","--host","127.0.0.1","--port",str(args.port),"--store-dir",str(store)],
            cwd=ROOT,stdout=stream,stderr=stream,stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    session=requests.Session();session.trust_env=False
    for _ in range(45):
        try:
            r=session.get(f"http://127.0.0.1:{args.port}/live.html",timeout=1)
            if r.status_code==200:break
        except requests.RequestException:pass
        if worker.poll() is not None:raise RuntimeError("Demo startup failed; inspect sanitized server log")
        time.sleep(.5)
    else:raise RuntimeError("Demo did not become ready")
    receipt={"old_pid":args.pid,"pid":worker.pid,"port":args.port,"url":f"http://127.0.0.1:{args.port}/live.html",
             "store_dir":str(store),"log":str(log),"started_at":datetime.now(timezone.utc).isoformat(),
             "prior_saved_runs_preserved":True,"credentials_exported":False}
    (OUT/("restart-"+stamp+".json")).write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt))


if __name__=="__main__":main()
