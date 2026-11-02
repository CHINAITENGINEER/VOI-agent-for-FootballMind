from pathlib import Path
import json, math, hashlib, zipfile
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'var/reviews/fm-g-nonleeds-expansion-20261015-v1'
ev=base/'evaluation'
rows=json.loads((ev/'prediction-input-rows.json').read_text(encoding='utf-8'))
reports=json.loads((ev/'match-reports.json').read_text(encoding='utf-8'))
by={}
for r in rows: by.setdefault(r['physical_match_id'],[]).append(r)
for rep in reports:
    rr=by[rep['physical_match_id']]; y=rep['outcome']; b=rr[0]['prediction_before']; a=rr[-1]['prediction_after']
    lb=-math.log(b[y]); la=-math.log(a[y]); rep.update({'loss_before':lb,'loss_after':la,'loss_change':la-lb,'loss_reduction':lb-la,'realized_gain':lb-la,'label_joined_after_prediction':True})
summary={'schema':'fm-g-expansion-acceptance-v1','matches':reports,'counts':{'physical_matches':len(reports),'states':len(rows),'new_direct_http':3,'new_search_tool_calls':2,'training_eligible_prospective_rows':0,'diagnostic_historical_rows':len(rows)},'frozen_f':'var/reviews/fm-historical-transition-g-training-20261014-v2/f-labeler.json','frozen_g_candidates':['var/reviews/fm-g-structured-state-20261014-v1/g-baseline.json','var/reviews/fm-g-structured-state-20261014-v1/g-combined.json'],'decision':'f_path_verified_on_two_additional_nonLeeds_matches; g_remains_diagnostic_only','limitations':['historical source versions unknown','search exposed final scores before review','Burnley/Chelsea B01 body overlaps prior Chelsea-Arsenal diagnostic','no prospective propensity or live-runtime training row','Liverpool lineup was fetched twice during collection; the second response overwrote the first raw file, so the duplicate is disclosed and not counted as a new source'],'next_action':'add fresh positive-control body with explicit match-bound availability/selection, then score g only after independent outcome and source clocks'}
(ev/'loss-and-acceptance.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['FootballMind 非 Leeds 补全验收（2026-10-15）','', '本轮目的：用两场新历史材料检验正文进入冻结 f 后是否能产生可追踪的 before/after。', '']
for rep in reports:
    lines += [f"{rep['fixture']}（{rep['physical_match_id']}）",f"状态行数：{rep['states']}；审阅断言：{rep['modeled_claims']}",f"赛果类别：{rep['outcome']}（仅在预测全部完成后加入）",f"before：{rep['prediction_before']}",f"after：{rep['prediction_after']}",f"概率 L1 变化：{rep['prediction_l1_shift']:.9f}",f"损失 before：{rep['loss_before']:.9f}",f"损失 after：{rep['loss_after']:.9f}",f"损失降低：{rep['loss_reduction']:.9f}" if rep['loss_reduction']>=0 else f"损失增加：{abs(rep['loss_reduction']):.9f}",'']
lines += ['解释：两场均出现了非零 f 变化；这证明接线能够工作，但不证明当前 f 的方向始终正确。','g 处理：本轮不把历史重算直接晋升为 empirical g，也不改变 UNKNOWN/constant gate。Burnley–Chelsea B01 复用了先前 Chelsea 正文，因此与此前诊断存在来源重叠；West Ham–Liverpool 只作为新的 holdout 诊断，不能单独训练高维 g。','下一步：继续补充具有明确比赛绑定的伤停/首发正文，按正文进入 f、独立赛果、损失差建立分组样本；达到足够独立比赛后再做 g 的分组留出和校准。']
gdiag=json.loads((base/'g-diagnostic/report.json').read_text(encoding='utf-8'))
lines += [f"g 候选诊断：旧样本 220 条，加本轮 14 个状态转移；整体模型 MSE={gdiag['overall_mse']:.9f}，全零基线 MSE={gdiag['overall_zero_mse']:.9f}。",f"按比赛留出时，Burnley–Chelsea MSE={gdiag['new_match_grouped_holdout']['understat:28891']['mse_model']:.9f}，全零基线={gdiag['new_match_grouped_holdout']['understat:28891']['mse_zero']:.9f}；West Ham–Liverpool MSE={gdiag['new_match_grouped_holdout']['understat:28907']['mse_model']:.9f}，全零基线={gdiag['new_match_grouped_holdout']['understat:28907']['mse_zero']:.9f}。这只是候选 g 的离线诊断，未获得选择权。"]
policy=json.loads((base/'g-diagnostic/policy-replay.json').read_text(encoding='utf-8'))
lines += [f"成本设为 0.001 的离线策略回放：Burnley–Chelsea 初始最大预期收益={policy['matches']['understat:28891']['maximum_predicted_gain']:.9f}，低于成本，提议 STOP；West Ham–Liverpool 初始预期收益={policy['matches']['understat:28907']['maximum_predicted_gain']:.9f}，为负值，提议 STOP。两项均为分组留出诊断，selection_allowed=false。"]
(ev/'验收报告.txt').write_text('\n'.join(lines),encoding='utf-8')
manifest=[]
for p in sorted(base.rglob('*')):
 if p.is_file() and p.name != base.name+'.zip':
  manifest.append({'path':str(p.relative_to(base)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(base/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
zip_path=base.with_suffix('.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for x in manifest: z.write(base/x['path'],x['path'])
print(json.dumps(summary,ensure_ascii=False,indent=2)); print('ZIP',zip_path,hashlib.sha256(zip_path.read_bytes()).hexdigest())
