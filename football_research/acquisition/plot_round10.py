"""Standalone scientific figure for the fixed-alpha diagnostic contrast."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(analysis_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output=Path(output)
    if output.exists():raise FileExistsError(output)
    result=json.loads(Path(analysis_path).read_bytes())
    labels={"base-only":"Free context only","calibrated-mean":"Mean: validation-selected penalty",
            "fixed-alpha-mean":"Mean: fixed reporting penalty (0.5)"}
    colors={"base-only":"#626a73","calibrated-mean":"#d55e00","fixed-alpha-mean":"#0072b2"}
    fig,axes=plt.subplots(1,3,figsize=(18.5,6.2),gridspec_kw={"width_ratios":[1.1,1.15,1]})
    for strategy,label in labels.items():
        values=sorted((float(k.rsplit("=",1)[1]),r) for k,r in result["metrics"].items() if k.startswith(strategy+"/"))
        axes[0].plot([r["money"] for _,r in values],[r["log_loss"] for _,r in values],"o-",color=colors[strategy],
                     label=label,linewidth=1.8,markersize=5)
    axes[0].set(xlabel="Mean simulated acquisition cost",ylabel="Log loss (lower is better)",title="Same frozen predictor and calibrator")
    axes[0].legend(fontsize=8,loc="lower left")
    axes[0].grid(alpha=.2)

    refs=("base-only","calibrated-mean")
    for i,ref in enumerate(refs):
        item=result["paired_comparisons"][f"fixed-alpha-mean-vs-{ref}/money=0.05"]["utility_loss"]
        m=item["mean_difference"];lo,hi=item["confidence_interval_95"]
        axes[1].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt="o",color=colors["fixed-alpha-mean"],capsize=4)
        axes[1].annotate(f"{m:+.4f}",(m,i),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9)
    axes[1].axvline(0,color="#888888",linestyle="--",linewidth=1)
    axes[1].set_yticks((0,1),("vs Free context","vs Selected penalty"))
    axes[1].set_ylim(1.6,-.6)
    axes[1].set(xlabel="Fixed minus reference: loss + 0.5 × cost",title="Budget 0.05: both intervals include zero")
    axes[1].grid(axis="x",alpha=.2)

    rows=[r for r in result["validation_test_contrasts"] if r["strategy"]=="fixed-alpha-mean" and r["budget_money"]==.05]
    x=[r["fold"] for r in rows]
    axes[2].plot(x,[r["fixed_minus_selected_validation_utility"] for r in rows],"o-",label="Validation",color="#009e73")
    axes[2].plot(x,[r["fixed_minus_selected_test_utility"] for r in rows],"s-",label="Following test period",color="#0072b2")
    axes[2].axhline(0,color="#888888",linestyle="--",linewidth=1)
    axes[2].set(xlabel="Temporal fold",ylabel="Fixed minus selected utility loss",title="Validation choice does not transfer uniformly")
    axes[2].set_xticks(x)
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=.2)
    fig.suptitle("FootballMind · Round 10: cost scale, stopping, and validation transfer",fontsize=15,y=.975)
    fig.text(.025,.048,"290 reused matches · 7 temporal folds · Simulated costs · 95% sorted-week paired bootstrap\n"
             "Fixed penalty: 153 queries; selected penalty: 128. Both LCB rules stop everywhere. No new independent evidence or fitted model.",fontsize=9.4)
    fig.tight_layout(rect=[0,.14,1,.915],w_pad=2.8)
    fig.savefig(output,dpi=170)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args();plot(args.analysis,args.output)
