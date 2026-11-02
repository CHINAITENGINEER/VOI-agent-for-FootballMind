import hashlib
import json
from pathlib import Path

ROOT=Path(r'D:\a1-足彩')
OUT=ROOT/'var'/'work'/'fm-f-g-v2-acceptance-20261025-v1'
F=ROOT/'var'/'work'/'fm-entity-f-v2-research-20261025-v1'
G=ROOT/'var'/'work'/'fm-current-f-diagnostic-g-v2-training-20261025-v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 fr=load(F/'report.json'); gr=load(G/'report.json')
 combined={'schema':'footballmind-f-g-v2-acceptance-v1','generated_from':{'f_report_sha256':sha(F/'report.json'),'g_report_sha256':sha(G/'report.json'),'f_model_sha256':sha(F/'model.json'),'g_model_sha256':sha(G/'model.json')},'network_requests':0,'api_calls':0,'formal_models_modified':False,'f_v2':{'architecture':'MLP hidden (128,64), tanh','parameters':fr['parameter_count'],'train_matches':fr['train_matches'],'validation_matches':fr['validation_matches'],'test_matches':fr['test_matches'],'test':{k:fr['metrics'][k]['C'] for k in ('prior','f_v1_selected','f_v1_diagnostic','v2_mlp')},'decision':'REJECT_PROMOTION','reason':'held-out C f-v2 MLP log loss and Brier are both worse than prior/f-v1 selected'},'g_v2':{'architecture':'MLPRegressor hidden (64,32,16), ReLU','parameters':gr['new_model']['parameters'],'train_matches':gr['splits']['train']['matches'],'validation_matches':gr['splits']['validation']['matches'],'test_matches':gr['splits']['test']['matches'],'test_metrics':gr['test_metrics'],'decision':'REJECT_PROMOTION','reason':'test MSE is worse than constant and zero baselines; test has only 2 matches'},'overall_decision':'NO_PROMOTION_DATA_EXPANSION_REQUIRED','promotion_rule':'candidate must beat prior and previous model on independent match-group test without outcome/after/gain leakage'}
 dump(OUT/'acceptance.json',combined)
 text='''FM-G 新参数实验阶段验收（2026-10-25）\n\n本轮训练了独立 f-v2 和 g-v2，均未覆盖旧模型。参数量增加不是通过条件；必须在按比赛分组的独立测试集上超过基线。\n\n结论：两份实验均不晋级。\n\nf-v2：45,699 参数，训练 15 场/255 行，验证 5 场/57 行，测试 6 场/126 行。测试集 prior 与 f-v1 selected 的 log loss 都是 1.0829452075、Brier 0.6649573203；f-v2 MLP 的 log loss 1.2724932012、Brier 0.8242457898。训练集 f-v2 变好，但测试集明显变差，属于过拟合或状态表示不足。\n\ng-v2：4,929 参数，166 行/11 场，训练 7 场、验证 2 场、测试 2 场。测试 MSE：zero 0.0009507635，constant 0.0008331783，Ridge 0.0010610355，旧诊断 MLP 0.0597630655，新 g-v2 0.0927199921。新 g-v2 没有超过任何合理基线。\n\n数据和标签没有使用 prediction_after、realized_gain 或 outcome 作为输入；它们只作为监督目标或最终评估标签。当前两份模型均 selection_allowed=false、production_ready=false。\n\n下一步不是继续盲目增大网络，而是扩展独立比赛和当前实体 f 的有效 typed-v3 claims，再用同一测试协议重新训练。\n'''
 (OUT/'readable-report.txt').write_text(text,encoding='utf-8')
 files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in {'manifest.json','verification.json'}); dump(OUT/'manifest.json',{p.name:sha(p) for p in files}); dump(OUT/'verification.json',{'schema':'footballmind-f-g-v2-acceptance-verification-v1','manifest_entries':len(files),'network_requests':0,'api_calls':0,'formal_models_modified':False,'promotion_allowed':False,'status':'PASS_COMPARATIVE_AUDIT'})
if __name__=='__main__': main()
