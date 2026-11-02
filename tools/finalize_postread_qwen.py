"""Human-readable delivery and independent saved-prediction checks."""
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
import py_compile
import shutil
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=Path("E:/FootballMind/experiments/postread-qwen-semantic-20261025-v1")
TRAIN=OUT/"training"
DATA=Path("E:/FootballMind/experiments/postread-clean-expansion-20261025-v1")


def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p): return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").split("\n") if x.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def csv_save(p,rr):
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model=read(TRAIN/"model.json");embed=read(OUT/"embedding-report.json")
    save(OUT/"embedding-selection.json",dict(
        selected="Qwen/Qwen3-Embedding-0.6B",
        reasons=["Multilingual contextual representation for existing cross-league bodies",
                 "Native Matryoshka dimensions 32 to 1024 allow exact old input geometry control",
                 "All current texts fit without truncation; no remote inference API required",
                 "Frozen local encoder permits small trainable scoring heads on 39 training matches"],
        alternatives=[
            dict(model="BAAI/bge-m3",dimension=1024,context_tokens=8192,url="https://huggingface.co/BAAI/bge-m3"),
            dict(model="intfloat/multilingual-e5-base",dimension=768,context_tokens=512,url="https://huggingface.co/intfloat/multilingual-e5-base"),
            dict(model="Alibaba-NLP/gte-multilingual-base",dimension=768,context_tokens=8192,url="https://huggingface.co/Alibaba-NLP/gte-multilingual-base")],
        decision_basis="Architecture, language, context, local execution and official documentation; alternatives not trained in this experiment",
        selected_documentation="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
        semantic_limit="Contextual representation is not a guarantee of correct negation/condition/claim interpretation or useful gain prediction"))
    metrics=read(TRAIN/"metrics.json");pred=rows(TRAIN/"predictions.jsonl")
    selected=model["selected_semantic_model"]
    grouped=defaultdict(list)
    for r in pred:grouped[r["model"],r["split"],r["physical_match_id"]].append(r)
    per_match=[]
    for (name,split,mid),rr in sorted(grouped.items()):
        y=np.array([r["realized_gain"] for r in rr]);p=np.array([r["predicted_gain"] for r in rr])
        per_match.append(dict(model=name,split=split,physical_match_id=mid,transitions=len(rr),
            mse=float(np.mean((p-y)**2)),mae=float(np.mean(abs(p-y))),
            mean_gain=float(np.mean(y)),mean_prediction=float(np.mean(p)),bias=float(np.mean(p-y))))
    checks=[]
    for r in metrics:
        rr=[x for x in per_match if x["model"]==r["model"] and x["split"]==r["split"]]
        m=float(np.mean([x["mse"] for x in rr]));diff=abs(m-r["match_macro_mse"])
        checks.append(dict(model=r["model"],split=r["split"],recomputed_mse=m,saved_mse=r["match_macro_mse"],difference=diff))
        if diff>1e-12:raise AssertionError("Saved prediction MSE does not reproduce metrics")
    csv_save(TRAIN/"per-match-metrics.csv",per_match)
    paired=[]
    for seed in (20261025,20261026,20261027):
        for split in ("train","validation","test"):
            old=next(r for r in metrics if r["model"]==f"lexical_fulltext_seed{seed}" and r["split"]==split)
            new=next(r for r in metrics if r["model"]==f"qwen32_mlp_seed{seed}" and r["split"]==split)
            paired.append(dict(seed=seed,split=split,lexical_mse=old["match_macro_mse"],qwen_mse=new["match_macro_mse"],
                qwen_minus_lexical_mse=new["match_macro_mse"]-old["match_macro_mse"],
                same_input_dimensions=214,same_trainable_parameters=35841))
    csv_save(TRAIN/"paired-seed-comparison.csv",paired)
    # Mark validation-selected checkpoints on the saved curves; no retraining.
    curve_rows=rows(TRAIN/"training-curves.jsonl")
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,name in zip(axes.flat,sorted(model["checkpoints"])):
        rr=[r for r in curve_rows if r["model"]==name]
        ax.plot([r["epoch"] for r in rr],[r["train_match_mse"] for r in rr],label="train")
        ax.plot([r["epoch"] for r in rr],[r["validation_match_mse"] for r in rr],label="validation")
        epoch=model["checkpoints"][name]["selected_epoch"]
        ax.axvline(epoch,color="#777",ls=":",label=f"selected epoch {epoch}")
        ax.set(title=name,yscale="log",xlabel="Epoch",ylabel="Match macro MSE");ax.legend(fontsize=8)
    fig.suptitle("Frozen encoder; checkpoints selected on validation, not test")
    fig.tight_layout();fig.savefig(TRAIN/"training-curves.png",dpi=160);plt.close(fig)
    chosen=[r for r in pred if r["model"]==selected]
    csv_save(TRAIN/"selected-predictions.csv",chosen)
    calibration=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,split in zip(axes,("validation","test")):
        rr=[r for r in chosen if r["split"]==split]
        p=np.array([r["predicted_gain"] for r in rr]);y=np.array([r["realized_gain"] for r in rr])
        for values in np.array_split(np.unique(p),min(5,len(np.unique(p)))):
            use=np.isin(p,values)
            b=dict(split=split,rows=int(use.sum()),mean_prediction=float(np.mean(p[use])),mean_gain=float(np.mean(y[use])))
            calibration.append(b)
            ax.scatter(b["mean_prediction"],b["mean_gain"],s=50,color="#177c67")
            ax.annotate(str(b["rows"]),(b["mean_prediction"],b["mean_gain"]))
        lo=min(min(p),min(y))-.01;hi=max(max(p),max(y))+.01
        ax.plot([lo,hi],[lo,hi],"k--",label="y=x reference")
        ax.set(title=split,xlabel="Bin mean predicted gain",ylabel="Bin mean realized gain");ax.legend()
    fig.suptitle("Sparse exploratory calibration; labels show transition counts, not independent matches")
    fig.tight_layout();fig.savefig(TRAIN/"calibration.png",dpi=160);plt.close(fig)
    save(TRAIN/"calibration-bins.json",calibration)
    sourcefiles=["football_research/acquisition/postread_semantic_embedding.py",
        "tools/prepare_postread_embedding_model.py","tools/encode_postread_qwen.py",
        "tools/train_postread_qwen.py","tools/finalize_postread_qwen.py",
        "tests/acquisition/test_postread_semantic_embedding.py"]
    compiled=[]
    for rel in sourcefiles:
        p=ROOT/rel;dest=OUT/"code"/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        cfile=OUT/"verification-pycache"/(p.stem+".pyc");cfile.parent.mkdir(parents=True,exist_ok=True)
        py_compile.compile(str(p),cfile=str(cfile),doraise=True);compiled.append(dict(path=str(p),sha256=sha(p)))
    save(OUT/"independent-verification.json",dict(saved_prediction_metric_checks=checks,
        source_compile=compiled,py_compile_passed=len(compiled),
        selected_prediction_rows=len(chosen),same_f_sha=model["f_model_sha256"],
        numeric_table_precision="CSV retains full Python float precision",
        no_additional_f_or_model_training=True))
    table_names=["zero","old_fulltext_mlp","lexical_fulltext_mlp","lexical_title_ridge",selected]
    text=["Qwen 全文语义表示与 g 评分头实验", "",
        "完成内容：下载固定版本预训练 encoder；本地全文编码；新 g 评分头训练；与已有模型和选择策略比较。",
        "未修改 f、原始数据、原有模型、runtime 或前端。预训练权重下载属于新网络访问；新闻抓取及外部推理 API 为零。",
        "",
        "模型：Qwen/Qwen3-Embedding-0.6B，revision="+embed["contract"]["revision"],
        "官方说明：https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
        "权重 SHA256="+embed["contract"]["model_sha256"],
        "encoder 参数 595,776,512，全部冻结；float32 CPU 推理。下载权重约1.19GB；不上传正文。",
        f"实际编码：{embed['body_count']} 篇正文，含已采用证据组合共 {embed['text_count']} 份不同文本；{embed['total_tokens']} tokens，最长 {embed['max_tokens']}，截断0。",
        f"本轮编码总时长 {embed['elapsed_seconds']:.1f} 秒，使用 E 盘文本/模型版本绑定缓存。",
        "原文保留，不拼入赛果/after/gain；没有凭标题补写事实。当前源版本/历史时钟假设不变。",
        "", "接口与对照：",
        "保存1024维向量。按模型支持的Matryoshka规则取前32/256维并重新L2归一化，不拟合SVD。",
        "输入顺序：57维当前状态 + 29维候选claim特征 + 候选正文向量 + 已采用正文向量 + 绝对差 + 逐元素乘积。",
        "32维方案：214输入 -> 128 -> 64 -> 1，35,841个评分头参数，与原全文MLP形状相同。",
        "256维方案：1110输入 -> 32 -> 1，35,585个评分头参数。两套方案均使用3个固定种子。",
        "额外保留32/256维Ridge、只状态、状态+claim、256维去掉候选claim的Ridge。去掉候选claim的模型仍有已采用证据状态。",
        "所有标签正/零/负均保留；原分组不变：训练39场241转移；验证6场24转移；测试6场24转移。",
        "评分头与标准化器仅在训练集拟合；验证集早停和选型。测试集此前已反复查看，本轮不能称为新的盲测。",
        "", "比赛等权 gain MSE（越低越好）：", "模型 | 验证 | 测试"]
    for name in table_names:
        vm=next(r for r in metrics if r["model"]==name and r["split"]=="validation")["match_macro_mse"]
        tm=next(r for r in metrics if r["model"]==name and r["split"]=="test")["match_macro_mse"]
        text.append(f"{name} | {vm:.12g} | {tm:.12g}")
    selected_metrics={s:next(r for r in metrics if r["model"]==selected and r["split"]==s) for s in ("validation","test")}
    lexical_metrics={s:next(r for r in metrics if r["model"]=="lexical_fulltext_mlp" and r["split"]==s) for s in ("validation","test")}
    zero_test=next(r for r in metrics if r["model"]=="zero" and r["split"]=="test")["match_macro_mse"]
    text += ["", "本轮可成立的结论：",
        "预训练语义表示已实际接入并完成训练/离线策略对照；不是继续使用TF-IDF冒称语义embedding。",
        f"验证选定Qwen方案相对当前同结构TF-IDF MLP，验证MSE降低{100*(1-selected_metrics['validation']['match_macro_mse']/lexical_metrics['validation']['match_macro_mse']):.3f}%，测试MSE降低{100*(1-selected_metrics['test']['match_macro_mse']/lexical_metrics['test']['match_macro_mse']):.3f}%。",
        f"但选定方案测试MSE仍高于零基线{100*(selected_metrics['test']['match_macro_mse']/zero_test-1):.3f}%，也未超过更早那版MLP。",
        "256维MLP的三个种子差异很大，完整指标保留；不能事后根据测试集挑出其中最好的一个再宣称训练成功。",
        "在每场最多1篇的这6场测试策略中，选定Qwen与当前TF-IDF MLP最终log loss相同，但平均采用文章数从0.5增加到0.8333；这轮尚未证明实际筛选能力优于当前MLP。",
        "训练误差已很低，验证误差仍明显较高；不能把剩余问题简单归因于评分头参数不够。",
        "下一项建议：固定已缓存的语义表示，在已有其他分组/来源留出协议上做成对稳定性检验，并优先改善状态与候选的对应表达；不依据此6场测试继续挑超参数。"]
    text += ["", "Qwen方案内部按验证集选择："+selected,
        "包含零/均值/状态/claim基线后，本轮整体验证集选择："+model["selected_including_baselines"],
        "", "策略比较："]
    policies=read(TRAIN/"strategy-comparison.json")["aggregates"]
    for r in policies:
        if r["split"]=="test" and r["strategy"] in ["no_information","all_candidates","fixed_order","random_exact",selected]:
            text.append(f"预算{r['budget']}，{r['strategy']}：最终log loss={r['mean_final_log_loss']:.12g}，平均f更新={r['mean_f_updates']:.4g}，预算合规={r['all_budget_compliant']}")
    text += ["", "解释与边界：",
        "语义embedding提供上下文表示，但不能保证单个向量准确保留所有否定、条件、实体、比赛绑定或冲突，也不是gain标签。",
        "与已有模型的差异须同时看同维度对照、种子、策略效果；不以embedding模型大小宣称收益提高。",
        "g的收益标签仍依赖同一个冻结f；不能用当前结果代表更强f。",
        "当前数据含已人工审阅的29维claim；融合模型的结果不能全部归功于编码器自动读懂文章。",
        "模型预训练语料及其是否见过历史比赛报道/赛果无法独立核实，不能宣称完全排除预训练污染。",
        "正文获取已经发生，没有将其宣称为g节省的成本。货币成本未测；CPU编码时间单独记录。",
        "新模型仅研究使用，正式runtime未接入。",
        "", "交付：embeddings.npz / embedding-report.json；training/models.joblib、model.json、metrics.csv、predictions.jsonl、per-match-metrics.csv、strategy-comparison.csv及图。",
        "复现顺序：prepare_postread_embedding_model.py；encode_postread_qwen.py；train_postread_qwen.py；finalize_postread_qwen.py。",
        "编码可以复用完整缓存；训练要求新目录，禁止覆盖旧产物。"]
    (OUT/"readable-report.txt").write_text("\n".join(text)+"\n",encoding="utf-8")
    save(OUT/"manifest.json",{str(p.relative_to(OUT)).replace("\\","/"):sha(p) for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.name!="manifest.json" and "verification-pycache" not in p.parts})
    print(json.dumps(dict(output=str(OUT),selected=selected,metric_checks=len(checks),compiled=len(compiled)),ensure_ascii=False))


if __name__=="__main__":main()
