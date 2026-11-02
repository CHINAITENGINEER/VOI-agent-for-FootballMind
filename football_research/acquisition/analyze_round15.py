"""Present exact finite mechanism effects without claiming football effectiveness."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .verify_round15 import SCENARIOS,digest,require


def analyze(run,verification,agent_verification,output):
    run,verification,agent_verification,output=map(Path,(run,verification,agent_verification,output))
    require(not output.exists(),"Analysis already exists")
    v=json.loads(verification.read_bytes());a=json.loads(agent_verification.read_bytes())
    require(v['status']=='VERIFIED_FINITE_MECHANISMS_NOT_FOOTBALL_EFFECTIVENESS',"Independent population verification incomplete")
    require(a['status']=='VERIFIED_40_SYNTHETIC_PATTERNS_WITH_INDEPENDENT_TREES',"Independent Agent verification incomplete")
    require(v['report_sha256']==a['research_report_sha256']==digest(run/'report.json'),"Different verified research release")
    metrics=v['true_metrics'];model_metrics=v['model_metrics'];summary={}
    for name in SCENARIOS:
        base=metrics[name+'/correct/base-only']['utility_loss']
        policies={p:metrics[name+'/correct/'+p] for p in ('myopic','best-fixed-subset','best-fixed-order-stop','adaptive-two-step')}
        adaptive=policies['adaptive-two-step']['utility_loss']
        wrong={}
        for variant in ('independent','swapped_context'):
            key=name+'/'+variant+'/adaptive-two-step'
            if key in metrics:
                wrong[variant]={'planned_utility_loss':model_metrics[key]['utility_loss'],
                    'true_utility_loss':metrics[key]['utility_loss'],
                    'true_net_gain_vs_no_query':base-metrics[key]['utility_loss'],
                    'regret_vs_true_optimum':metrics[key]['utility_loss']-adaptive}
        paths=defaultdict(int)
        for case in a['checks']:
            if case['scenario_id']==name:paths[' -> '.join(case['actions'])+' -> STOP' if case['actions'] else 'STOP']+=1
        summary[name]={'base_utility_loss':base,'correct_model_policies':policies,
            'adaptive_gain_vs_no_query':base-adaptive,
            'adaptive_gain_vs_fixed_subset':policies['best-fixed-subset']['utility_loss']-adaptive,
            'adaptive_gain_vs_fixed_order_stop':policies['best-fixed-order-stop']['utility_loss']-adaptive,
            'actual_agent_pattern_paths':dict(paths),'wrong_model_diagnostics':wrong}
    result={'status':'ANALYZED_EXACT_SYNTHETIC_MECHANISMS','round':15,'scenarios':summary,
        'research_counts':v['counts'],'agent_counts':a['counts'],
        'metrics':metrics,'modeled_metrics':model_metrics,'correct_policy_optima':v['correct_policy_optima'],
        'sources':{p.as_posix():digest(p) for p in (run/'report.json',verification,agent_verification)},
        'analyzer_sha256':digest(__file__),'new_independent_football_matches':0,
        'interpretation':['Tool-identity adaptation adds value in the prescribed gated_route model',
            'gated_stop equals a fixed order with adaptive stopping; it does not establish routing benefit',
            'Complementarity rewards acquiring both signals, including a fixed subset; it is not a routing advantage',
            'Redundant and irrelevant cases test economically correct stopping',
            'Swapped contexts illustrate that exact planning under a wrong model may worsen true-population decisions',
            'The generated population and costs are constructed; these are mechanism checks, not learned football performance'],
        'uncertainty_scope':'No sampling: exact expectations over all declared positive masses, no confidence intervals or general superiority theorem'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    path=Path(path);require(not path.exists(),'Figure already exists')
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(15.5,10),constrained_layout=True)
    names=list(SCENARIOS)
    labels=['Context:\ncontinue/stop','Context:\nchoose tool','Complementary','Redundant','Irrelevant']
    colors={'myopic':'#9099a2','best-fixed-order-stop':'#c26c25','adaptive-two-step':'#24659a'}
    aliases={'myopic':'Myopic','best-fixed-order-stop':'Fixed order + adaptive STOP','adaptive-two-step':'Adaptive two-step'}
    x=np.arange(len(names));width=.25
    for j,policy in enumerate(colors):
        values=[result['scenarios'][s]['base_utility_loss']-result['metrics'][s+'/correct/'+policy]['utility_loss'] for s in names]
        axes[0,0].bar(x+(j-1)*width,values,width,color=colors[policy],label=aliases[policy])
    axes[0,0].set(xticks=x,xticklabels=labels,ylabel='Expected net gain vs no query (higher is better)',
                  ylim=(0,.82),title='A. Correct model: depth, routing and stopping are distinct')
    axes[0,0].legend(loc='upper right',fontsize=8.5)
    axes[0,0].grid(axis='y',alpha=.16)

    ax=axes[0,1];ax.set_axis_off();ax.set_title('B. Same first query; observed content selects the next tool')
    ax.set(xlim=(0,1),ylim=(0,1))
    box={'boxstyle':'round,pad=.6','facecolor':'#e8f0f6','edgecolor':'#24659a','linewidth':1.5}
    ax.text(.5,.84,'Query context G\nImmediate outcome information = 0\nActual first fee = 0.02',ha='center',va='center',bbox=box)
    ax.text(.23,.42,'Observed G = 0\nQuery signal_a\n9 complete Agent patterns',ha='center',va='center',bbox=box)
    ax.text(.77,.42,'Observed G = 1\nQuery signal_b\n9 complete Agent patterns',ha='center',va='center',bbox=box)
    for px in (.23,.77):
        ax.annotate('',xy=(px,.57),xytext=(.5,.70),arrowprops={'arrowstyle':'->','color':'#24659a','linewidth':1.8})
    ax.text(.5,.13,'Each branch has population probability 0.5.\nActual posterior updates only after the chosen signal returns.\nSynthetic known distribution; labels are not actual teams.',ha='center',va='center',fontsize=10)

    ax=axes[1,0]
    modeled=[result['modeled_metrics']['gated_route/'+m+'/adaptive-two-step']['utility_loss'] for m in ('correct','swapped_context')]
    actual=[result['metrics']['gated_route/'+m+'/adaptive-two-step']['utility_loss'] for m in ('correct','swapped_context')]
    for delta,values,label,color in ((-.18,modeled,'Expected under Agent model','#24659a'),(.18,actual,'Expected under true population','#c26c25')):
        bars=ax.bar(np.arange(2)+delta,values,.36,label=label,color=color)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+.04,f'{value:.6f}',ha='center',va='bottom',fontsize=9)
    prior=result['scenarios']['gated_route']['base_utility_loss']
    ax.axhline(prior,color='#666666',linestyle='--',linewidth=1,label='No query')
    ax.set(xticks=[0,1],xticklabels=['Correct context model','Wrong context model'],ylim=(0,2.65),
           ylabel='Expected log loss + simulated fee (lower is better)',title='C. Correct computation does not correct a wrong information model')
    ax.legend(loc='upper left',fontsize=8.5);ax.grid(axis='y',alpha=.16)

    ax=axes[1,1]
    # A distribution-free count of the selected paths would overweight rare
    # patterns. Use exact generating mass for this chart instead.
    report=json.loads(Path(next(k for k in result['sources'] if k.endswith('/report.json'))).read_bytes())
    probabilities=[report['metrics'][s+'/correct/adaptive-two-step']['query_count_distribution'].get('2',0.) for s in names]
    bars=ax.bar(x,probabilities,color='#397e78',width=.58)
    for bar,value in zip(bars,probabilities):
        ax.text(bar.get_x()+bar.get_width()/2,value+.025,f'{value:.0%}',ha='center',va='bottom')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.22),ylabel='True-population probability of a second query',
           title='D. Acquiring more information is optional')
    ax.grid(axis='y',alpha=.16)
    fig.suptitle('FootballMind round 15 | Exact conditional-information mechanism benchmark\n'
                 'Five predeclared synthetic populations; every positive-mass world evaluated. No real-football effectiveness claim.',fontsize=14)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','verification','agent-verification','output','figure'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    result=analyze(args.run,args.verification,args.agent_verification,args.output)
    plot(result,args.figure)
    print(json.dumps({'status':result['status'],'scenario_effects':{
        s:{k:v for k,v in x.items() if k.startswith('adaptive_gain')} for s,x in result['scenarios'].items()}}))
