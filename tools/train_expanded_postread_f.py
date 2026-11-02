"""Fit the expanded historical f challenger before freezing it for g labels."""
from pathlib import Path
import argparse
import json
import shutil
import sys
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_e_drive_rolling_g import Data,dump
from tools.train_e_drive_f_response import run_stage


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    contract=json.loads((a.previous/'panel-protocol.json').read_text(encoding='utf-8'))
    boundary=contract['f_g_boundary']
    ps=[json.loads(s) for s in (a.previous/'panel.jsonl').read_text(encoding='utf-8').splitlines()]
    h=[v for v in ps if v['date']<boundary]
    t=[v for v in ps if v['split']=='train' and v['date']>=boundary]
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(a.previous/'f-base.json',a.output/'f-base.json')
    dump(a.output/'protocol.json',dict(f_fit_boundary=boundary,fit_matches=len(h),
        body_challenger_selected_with_inner_dates_only=True,target_is_g_training=True,production_modified=False))
    data=Data(a.root,a.previous,ps)
    dump(a.output/'report.json',run_stage(data,h,t,a.output,'expanded'))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
