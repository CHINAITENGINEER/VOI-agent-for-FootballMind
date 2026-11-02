"""Summarize actual paired stability results without choosing models on test."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=Path("E:/FootballMind/experiments/postread-qwen-stability-20261025-v1")


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").split("\n") if x.strip()]
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def csv_save(p,rr):
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan=read(OUT/"experiment-plan.json")
    names=list(plan["named_assignments"]);random_names=list(plan["random_assignments"])
    if not (OUT/"training-completion.json").exists():raise ValueError("Full study not complete")
    summaries={n:read(OUT/n/"summary.json") for n in names+random_names}
    all_metrics=[r for s in summaries.values() for r in s["metrics"]]
    all_policy=[dict(fold=n,**r) for n,s in summaries.items() for r in s["policies"]]
    csv_save(OUT/"all-metrics.csv",all_metrics);csv_save(OUT/"all-policy-metrics.csv",all_policy)
    def mse(n,m,split="test"):
        return next(r["match_macro_mse"] for r in summaries[n]["metrics"] if r["model"]==m and r["split"]==split)
    named=[];seed_pairs=[];intervals=[]
    for n in names:
        c=summaries[n]["counts"]
        l=mse(n,"lexical32_validation_selected_mlp");q=mse(n,"qwen32_validation_selected_mlp")
        named.append(dict(fold=n,train_matches=c["train"]["matches"],validation_matches=c["validation"]["matches"],test_matches=c["test"]["matches"],
            zero_mse=mse(n,"zero"),lexical_selected_mlp_mse=l,qwen_selected_mlp_mse=q,
            qwen_to_lexical_mlp_ratio=q/l,qwen_mlp_better_than_lexical=q<l,qwen_mlp_better_than_zero=q<mse(n,"zero"),
            lexical32_ridge_mse=mse(n,"lexical32_ridge"),qwen32_ridge_mse=mse(n,"qwen32_ridge"),qwen256_ridge_mse=mse(n,"qwen256_ridge")))
        for seed in (20261025,20261026,20261027):
            l=mse(n,f"lexical32_mlp_seed{seed}");q=mse(n,f"qwen32_mlp_seed{seed}")
            seed_pairs.append(dict(fold=n,seed=seed,lexical_mse=l,qwen_mse=q,qwen_to_lexical_ratio=q/l,qwen_lower_mse=q<l))
        # Pair by match and transition; seed averaging never creates extra independent matches.
        rr=rows(OUT/n/"predictions.jsonl");bucket=defaultdict(dict)
        for r in rr:
            if r["split"]=="test" and r["model"] in ("lexical32_validation_selected_mlp","qwen32_validation_selected_mlp"):
                bucket[r["physical_match_id"],r["sample_id"]][r["model"]]=(r["predicted_gain"]-r["realized_gain"])**2
        per_match=defaultdict(list)
        for (mid,sid),value in bucket.items():
            if len(value)!=2:raise ValueError("Unpaired score")
            per_match[mid].append(value["qwen32_validation_selected_mlp"]-value["lexical32_validation_selected_mlp"])
        delta=np.array([np.mean(v) for v in per_match.values()]);rng=np.random.default_rng(20261025)
        boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)
        low,high=np.quantile(boot,[.025,.975])
        intervals.append(dict(fold=n,independent_matches=len(delta),qwen_minus_lexical_mse=float(delta.mean()),
            bootstrap_95_low=float(low),bootstrap_95_high=float(high),resamples=5000,
            note="Conditional descriptive paired match bootstrap; small reused cohorts; not model-selection uncertainty or independent folds"))
    csv_save(OUT/"named-comparison.csv",named);csv_save(OUT/"paired-seed-comparison.csv",seed_pairs);save(OUT/"paired-match-intervals.json",intervals)
    random_rows=[]
    for n in random_names:
        random_rows.append(dict(fold=n,zero_mse=mse(n,"zero"),lexical32_mse=mse(n,"lexical32_ridge"),
            qwen32_mse=mse(n,"qwen32_ridge"),qwen256_mse=mse(n,"qwen256_ridge"),
            qwen32_to_lexical_ratio=mse(n,"qwen32_ridge")/mse(n,"lexical32_ridge"),
            qwen256_to_lexical_ratio=mse(n,"qwen256_ridge")/mse(n,"lexical32_ridge")))
    csv_save(OUT/"random-resplit-comparison.csv",random_rows)
    random_summary={}
    for key in ("qwen32","qwen256"):
        ratio=np.array([r[key+"_to_lexical_ratio"] for r in random_rows])
        random_summary[key]=dict(lower_mse_than_lexical=sum(r[key+"_mse"]<r["lexical32_mse"] for r in random_rows),
            lower_mse_than_zero=sum(r[key+"_mse"]<r["zero_mse"] for r in random_rows),
            folds=len(random_rows),median_lexical_ratio=float(np.median(ratio)),ratio_min=float(ratio.min()),ratio_max=float(ratio.max()))
    fig,ax=plt.subplots(figsize=(12,5))
    grid=np.array([[r[k]/r["zero_mse"] for r in named] for k in ("lexical_selected_mlp_mse","qwen_selected_mlp_mse","lexical32_ridge_mse","qwen32_ridge_mse","qwen256_ridge_mse")])
    im=ax.imshow(grid,cmap="RdYlGn_r",vmin=.5,vmax=2)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):ax.text(j,i,f"{grid[i,j]:.2f}",ha="center",va="center")
    ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(5),yticklabels=["Lexical MLP (val selected)","Qwen32 MLP (val selected)","Lexical32 Ridge","Qwen32 Ridge","Qwen256 Ridge"],
        title="Test gain MSE / zero baseline; below 1 is better\nSame frozen f and cached Qwen vectors; columns are overlapping development cohorts")
    ax.tick_params(axis="x",rotation=20);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/"split-stability.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for seed in (20261025,20261026,20261027):
        rr=[r for r in seed_pairs if r["seed"]==seed]
        axes[0].plot(range(len(names)),[r["qwen_to_lexical_ratio"] for r in rr],marker="o",label=str(seed))
    axes[0].axhline(1,color="black",ls="--");axes[0].set(xticks=range(len(names)),xticklabels=names,ylabel="Qwen / lexical test MSE",title="Same shape + seed; below 1 favors Qwen")
    axes[0].tick_params(axis="x",rotation=30);axes[0].legend(fontsize=8)
    for key,color in (("qwen32","#688da0"),("qwen256","#167b63")):
        axes[1].plot(range(30),[r[key+"_to_lexical_ratio"] for r in random_rows],marker=".",label=key,color=color)
    axes[1].axhline(1,color="black",ls="--");axes[1].set(xlabel="Existing random resplit index (not independent)",ylabel="Qwen Ridge / lexical Ridge MSE",title="30 resplits over the same18-match cohort");axes[1].legend()
    fig.tight_layout();fig.savefig(OUT/"seed-and-resplit-stability.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,n in zip(axes.flat,names):
        cc=rows(OUT/n/"training-curves.jsonl")
        for family,color in (("lexical32","#a06940"),("qwen32","#167b63")):
            selected=summaries[n]["selected"][family+"_validation_selected_mlp"]
            rr=[r for r in cc if r["model"]==selected]
            ax.plot([r["epoch"] for r in rr],[r["train_mse"] for r in rr],color=color,ls="--",alpha=.55)
            ax.plot([r["epoch"] for r in rr],[r["validation_mse"] for r in rr],color=color,label=family)
            ax.axvline(summaries[n]["checkpoints"][selected]["selected_epoch"],color=color,ls=":",alpha=.5)
        ax.set(title=n,yscale="log",xlabel="Epoch",ylabel="Match macro MSE");ax.legend(fontsize=8)
    fig.suptitle("Same 128x64 head; dashed=train, solid=validation; dotted=chosen checkpoint")
    fig.tight_layout();fig.savefig(OUT/"training-curves.png",dpi=160);plt.close(fig)
    policy_table=[]
    for n in names:
        for budget in (1,2,3):
            selected={r["strategy"]:r for r in summaries[n]["policies"] if r["split"]=="test" and r["budget"]==budget}
            policy_table.append(dict(fold=n,budget=budget,
                no_information_loss=selected["no_information"]["mean_final_log_loss"],
                lexical_mlp_loss=selected["lexical32_validation_selected_mlp"]["mean_final_log_loss"],
                qwen_mlp_loss=selected["qwen32_validation_selected_mlp"]["mean_final_log_loss"],
                qwen256_ridge_loss=selected["qwen256_ridge"]["mean_final_log_loss"],
                random_loss=selected["random_exact"]["mean_final_log_loss"],
                lexical_mlp_updates=selected["lexical32_validation_selected_mlp"]["mean_f_updates"],
                qwen_mlp_updates=selected["qwen32_validation_selected_mlp"]["mean_f_updates"]))
    csv_save(OUT/"named-policy-comparison.csv",policy_table)
    policy_counts={}
    for budget in (1,2,3):
        pp=[r for r in policy_table if r["budget"]==budget]
        policy_counts[str(budget)]={}
        for baseline in ("lexical_mlp_loss","random_loss","no_information_loss"):
            d=np.array([r["qwen_mlp_loss"]-r[baseline] for r in pp])
            policy_counts[str(budget)][baseline]=dict(lower=int(np.sum(d< -1e-12)),equal=int(np.sum(abs(d)<=1e-12)),higher=int(np.sum(d>1e-12)))
    result=dict(named_folds=len(names),random_resplits=len(random_names),new_matches=0,
        selected_mlp_qwen_better_than_lexical=sum(r["qwen_mlp_better_than_lexical"] for r in named),
        selected_mlp_qwen_better_than_zero=sum(r["qwen_mlp_better_than_zero"] for r in named),
        same_seed_pairs=len(seed_pairs),same_seed_qwen_lower_mse=sum(r["qwen_lower_mse"] for r in seed_pairs),
        named_qwen256_ridge_better_than_lexical=sum(r["qwen256_ridge_mse"]<r["lexical32_ridge_mse"] for r in named),
        named_qwen256_ridge_better_than_zero=sum(r["qwen256_ridge_mse"]<r["zero_mse"] for r in named),
        random_summary=random_summary,
        policy_comparison_counts=policy_counts,
        paired_intervals_include_zero=sum(r["bootstrap_95_low"]<=0<=r["bootstrap_95_high"] for r in intervals),
        scope="Retrospective correlated sensitivity studies; not new external validation; 256-vs32 also changes representation dimension")
    save(OUT/"stability-summary.json",result)
    text=["固定 Qwen 语义向量的稳定性与状态对应检查", "",
        "本轮使用原51场/92篇/289转移，复用110个1024维向量；没有重新编码、采集或训练f。",
        "原分组+队列内时序+3来源留出+全局时序共6组；同一214输入、128x64评分头，各3种子，lexical/Qwen配对重训36个MLP。",
        "另按30个已有随机分组比较Ridge。标准化、目标缩放、TF-IDF/SVD仅在各自训练集拟合。",
        "所有指标按比赛等权；验证集选择checkpoint/种子。测试记录此前已用于开发，非新盲测。",
        "", "测试gain MSE（越低越好）：",
        "分组 | 零基线 | lexical MLP | Qwen32 MLP | lexical Ridge | Qwen256 Ridge"]
    for r in named:text.append(f"{r['fold']} | {r['zero_mse']:.9f} | {r['lexical_selected_mlp_mse']:.9f} | {r['qwen_selected_mlp_mse']:.9f} | {r['lexical32_ridge_mse']:.9f} | {r['qwen256_ridge_mse']:.9f}")
    text += ["",f"验证选种子的Qwen32 MLP在{result['selected_mlp_qwen_better_than_lexical']}/6组低于lexical，但{result['selected_mlp_qwen_better_than_zero']}/6组低于零基线。",
        f"逐一固定相同种子，{result['same_seed_qwen_lower_mse']}/{len(seed_pairs)}个配对Qwen MSE较低；不能把18配对算成18个独立数据集。",
        f"Qwen256 Ridge在{result['named_qwen256_ridge_better_than_lexical']}/6组优于lexical32 Ridge，在{result['named_qwen256_ridge_better_than_zero']}/6组优于零基线。该比较同时改变向量类型和维度。",
        "30重划结果："+json.dumps(random_summary,ensure_ascii=False),
        "30组测试只来自原18场、其余33场恒定训练；重复划分不是30个外部测试集。",
        "6组逐场配对bootstrap区间全部跨过0；小样本和重用测试条件下，尚不能宣称稳健的总体优势。",
        "每场最多1篇时，Qwen32 MLP相对lexical MLP的最终log loss："+json.dumps(policy_counts["1"]["lexical_mlp_loss"],ensure_ascii=False)+"（lower=更低，equal=相同，higher=更高）。",
        "对应不加新信息策略："+json.dumps(policy_counts["1"]["no_information_loss"],ensure_ascii=False)+"。MSE改善不能替代策略效果。",
        "只有global_time_51为全局时间前推；chrono_within_cohort仍有未来背景训练；source holdout是域名留出而非全部球队未见。",
        "", "状态与正文审查：",
        "289条对应检查通过、92正文语义绑定通过、110向量缓存对应通过。8条拼接顺序差异是既有input_id排序契约，非错配。",
        "状态未显式提供目标主客球队名/逐球员身份/每条证据时间条件/真实剩余预算。57状态有35个恒零列，29候选特征有11个恒零列。",
        "time_unknown与构建来源关联：原分组166条Understat训练行时间已填；其余123行unknown（包括原验证/测试48行）。此关联不证明造成误差反转。",
        "candidate_status_conflicts是已知与候选合并后的多状态数量；有2行候选未新增分歧仍得到正值。需与候选新增冲突分开表达。",
        "完整逐行证据见alignment-audit/alignment-report.txt；分组和f训练ID边界见protocol-audit/protocol-audit.txt。",
        "", "本阶段结论：",
        "换语义表示的部分数值收益可以在其他分组重复，但不是对零基线和实际策略的全面胜出。全局时序上的同形状MLP提升未重复。",
        "32维压缩后不同评分器的结果并不一致；256维+Ridge更值得作为后续候选基线，不能按这些已看过测试成绩直接上线。",
        "下一步扩规模前优先保留完整1024维向量、显式实体/比赛/角色关系及时间缺失来源；避免把更多参数当作已证实的改进。",
        "原f、原数据、原向量和原实验未覆盖；0来源HTTP、0API、0新向量推理。"]
    (OUT/"readable-report.txt").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
