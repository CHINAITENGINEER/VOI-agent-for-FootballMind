"""Train research-only post-read g heads against one immutable sequential f."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from football_research.acquisition.sequential_evidence_f import FrozenF
from football_research.acquisition.sequential_postread_g import (
    EvidencePanel,VectorStore,FrozenEvaluator,generate_transitions,equal_match_weights,metrics,predict_head,
    run_policy,settle_path,save_json,save_rows,sha_file)

BASE=Path("E:/FootballMind/experiments/full-history-sequential-fg-20261025-v1")
CACHE=Path("E:/FootballMind/embeddings/postread-qwen-frozen-v1/9c75522b941af4d6bddf21b796039e5182833f0eb01d4a6c17dd7c98b5eac6bd")


def plots(output,curves,predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for model in sorted({r["model"] for r in curves}):
        rr=[r for r in curves if r["model"]==model]
        for ax,key in zip(axes,("train_mse","dev_mse")):ax.plot([r["epoch"] for r in rr],[r[key] for r in rr],label=model)
    axes[0].set_title("Train match-macro gain MSE");axes[1].set_title("Development match-macro gain MSE")
    for ax in axes:ax.set_xlabel("Epoch");ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/"training-curves.png",dpi=170);plt.close(fig)
    models=sorted({r["model"] for r in predictions});fig,axes=plt.subplots(2,len(models),figsize=(4*len(models),7),squeeze=False)
    for j,model in enumerate(models):
        rr=[r for r in predictions if r["model"]==model and r["split_role"]=="g_dev"]
        yy=np.array([r["realized_gain"] for r in rr]);pp=np.array([r["predicted_gain"] for r in rr]);ax=axes[0,j]
        ax.scatter(yy,pp,s=10,alpha=.4);lo=min(yy.min(),pp.min());hi=max(yy.max(),pp.max());ax.plot([lo,hi],[lo,hi],"k--",label="y=x reference, not fitted")
        ax.set_title(model);ax.set_xlabel("Realized gain");ax.set_ylabel("Predicted gain");ax.legend(fontsize=7)
        groups=np.array_split(np.argsort(pp),min(8,len(pp)))
        axes[1,j].plot([pp[g].mean() for g in groups],[yy[g].mean() for g in groups],"o-")
        axes[1,j].plot([lo,hi],[lo,hi],"k--");axes[1,j].set_xlabel("Bin mean prediction");axes[1,j].set_ylabel("Bin mean realized gain")
    fig.tight_layout();fig.savefig(output/"dev-prediction-calibration.png",dpi=150);plt.close(fig)


def training_size_curve(output,x,y,mids,ti,vi,fitted,configs,seed):
    """Nested match subsets; this diagnostic cannot select a model or threshold."""
    name=f"fulltext_mlp_seed{seed}";epochs=configs[name]["selected_epoch"]
    all_matches=np.asarray(sorted(set(mids[ti])));order=np.random.default_rng(seed).permutation(all_matches)
    values=[]
    for fraction in (.25,.5,1.):
        used=set(order[:max(1,int(np.ceil(len(order)*fraction)))]);ii=np.asarray([i for i in ti if mids[i] in used])
        weights=equal_match_weights(mids[ii]);mu=float(np.average(y[ii],weights=weights));sigma=max(1e-8,float(np.sqrt(np.average((y[ii]-mu)**2,weights=weights))))
        if fraction==1.:
            # The frozen full-training seed is reused; no fourth full refit.
            spec=fitted[name];scaler=spec['scaler'];estimator=spec['model']
        else:
            scaler=StandardScaler().fit(x[ii],sample_weight=weights)
            estimator=MLPRegressor(hidden_layer_sizes=(128,64),activation='relu',solver='adam',alpha=.1,
                batch_size=min(128,len(ii)),learning_rate_init=.001,max_iter=1,shuffle=True,random_state=seed,early_stopping=False)
            xx=scaler.transform(x[ii])
            for _ in range(epochs):estimator.partial_fit(xx,(y[ii]-mu)/sigma,sample_weight=weights)
        pt=estimator.predict(scaler.transform(x[ii]))*sigma+mu;pv=estimator.predict(scaler.transform(x[vi]))*sigma+mu
        values.append(dict(fraction=fraction,train_matches=len(used),train_rows=len(ii),seed=seed,fixed_epochs=epochs,
            train=metrics(y[ii],pt,mids[ii]),g_dev=metrics(y[vi],pv,mids[vi]),used_for_selection=False,
            train_match_ids=sorted(used),protocol='nested_match_subsets_fixed_architecture_and_full_run_epoch'))
    save_rows(output/'training-size-curve.jsonl',values)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    for role in ('train','g_dev'):ax.plot([r['train_matches'] for r in values],[r[role]['match_macro_mse'] for r in values],'o-',label=role)
    ax.set_xlabel('Independent training matches');ax.set_ylabel('Match-macro gain MSE');ax.set_title('Training-size diagnostic; not used for model selection');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/'training-size-curve.png',dpi=170);plt.close(fig)


def train(args):
    if args.output.exists():raise FileExistsError("Do not overwrite a completed/incomplete g experiment")
    source_hashes={str(args.data/name):sha_file(args.data/name) for name in ("matches.jsonl","documents.jsonl","sequences.jsonl","outcomes.jsonl","schema.json","dataset-summary.json")}
    if args.vectors.is_file():source_hashes[str(args.vectors)]=sha_file(args.vectors)
    f=FrozenF.load(args.f_model)
    panel=EvidencePanel(args.data,VectorStore(args.vectors),roles={"g_fit","g_dev"},include_outcomes=True)
    if not {"g_fit","g_dev"}.issubset({r["split_role"] for r in panel.sequences.values()}):raise ValueError("Need separate g_fit and g_dev matches")
    evaluator=FrozenEvaluator(f,panel);started=time.perf_counter()
    print(json.dumps(dict(stage='label_generation',f_model_sha256=evaluator.f_sha,matches=len(panel.sequences),input_failures=len(panel.failures))),flush=True)
    x,records,failures,state_width=generate_transitions(evaluator,seed=args.seed,orders_per_match=3,budget=3,dimension=256,feature_budgets=(1,2,3,6))
    args.output.mkdir(parents=True)
    save_rows(args.output/"labels.jsonl",records);save_rows(args.output/"failures.jsonl",panel.failures+failures)
    np.save(args.output/"features.npy",x,allow_pickle=False)
    print(json.dumps(dict(stage='labels_ready',rows=len(records),feature_dimension=x.shape[1],f_calls=evaluator.calls)),flush=True)
    save_json(args.output/"feature-ids.json",[r["sample_id"] for r in records])
    splits=np.array([r["split_role"] for r in records]);mids=np.array([r["physical_match_id"] for r in records]);y=np.array([r["realized_gain"] for r in records])
    ti=np.flatnonzero(splits=="g_fit");vi=np.flatnonzero(splits=="g_dev")
    if not len(ti) or not len(vi) or set(mids[ti])&set(mids[vi]):raise ValueError("Invalid grouped g split")
    w=equal_match_weights(mids[ti]);mean=float(np.average(y[ti],weights=w));scale=max(1e-8,float(np.sqrt(np.average((y[ti]-mean)**2,weights=w))))
    yt=(y[ti]-mean)/scale;fitted={"zero":dict(kind="zero"),"train_mean":dict(kind="constant",value=mean)};curves=[];configs={"zero":dict(parameters=0),"train_mean":dict(parameters=1)}
    for name,key,xx in (("state_ridge","state",x[:,:state_width]),("fulltext_ridge","full",x)):
        scaler=StandardScaler().fit(xx[ti],sample_weight=w)
        estimator=Ridge(alpha=10.).fit(scaler.transform(xx[ti]),yt,sample_weight=w)
        fitted[name]=dict(kind="ridge",input=key,state_width=state_width,scaler=scaler,model=estimator)
        print(json.dumps(dict(stage='head_fitted',model=name)),flush=True)
        configs[name]=dict(parameters=int(estimator.coef_.size+1),alpha=10.,input_dimension=xx.shape[1],train_only_standardization=True)
    # This is a newly fitted local-metadata control, never labelled as the old
    # frozen g (whose f binding and input contract differ).
    meta_width=len(panel.metadata_names)
    metadata_columns=list(range(7))+list(range(7+256,7+256+meta_width))+list(range(state_width+3*256,state_width+3*256+meta_width))
    xx=x[:,metadata_columns];metadata_scaler=StandardScaler().fit(xx[ti],sample_weight=w)
    fitted["local_metadata_refit_ridge"]=dict(kind="ridge",input="metadata",column_indices=metadata_columns,
        scaler=metadata_scaler,model=Ridge(alpha=10.).fit(metadata_scaler.transform(xx[ti]),yt,sample_weight=w))
    configs['local_metadata_refit_ridge']=dict(parameters=len(metadata_columns)+1,alpha=10.,input_dimension=len(metadata_columns),train_only_standardization=True)
    scaler=StandardScaler().fit(x[ti],sample_weight=w);xt=scaler.transform(x[ti]);xv=scaler.transform(x[vi])
    for seed in (args.seed,args.seed+1,args.seed+2):
        name=f"fulltext_mlp_seed{seed}"
        estimator=MLPRegressor(hidden_layer_sizes=(128,64),activation="relu",solver="adam",alpha=.1,
            batch_size=min(128,len(ti)),learning_rate_init=.001,max_iter=1,shuffle=True,random_state=seed,early_stopping=False)
        best=None;best_loss=float("inf");best_epoch=0
        for epoch in range(1,args.max_epochs+1):
            estimator.partial_fit(xt,yt,sample_weight=w)
            pt=estimator.predict(xt)*scale+mean;pv=estimator.predict(xv)*scale+mean
            tm=metrics(y[ti],pt,mids[ti]);vm=metrics(y[vi],pv,mids[vi])
            curves.append(dict(model=name,epoch=epoch,train_mse=tm["match_macro_mse"],dev_mse=vm["match_macro_mse"]))
            if epoch==1 or epoch%10==0:print(json.dumps(dict(stage='mlp_epoch',model=name,epoch=epoch,train_mse=tm['match_macro_mse'],dev_mse=vm['match_macro_mse'])),flush=True)
            if vm["match_macro_mse"]<best_loss-1e-12:best=deepcopy(estimator);best_loss=vm["match_macro_mse"];best_epoch=epoch
            if epoch-best_epoch>=25:break
        fitted[name]=dict(kind="mlp",input="full",state_width=state_width,scaler=scaler,model=best)
        configs[name]=dict(selected_epoch=best_epoch,epochs_run=epoch,parameters=sum(a.size for a in best.coefs_)+sum(a.size for a in best.intercepts_),hidden=[128,64],seed=seed)
        print(json.dumps(dict(stage='head_fitted',model=name,selected_epoch=best_epoch,epochs_run=epoch)),flush=True)
    prediction_rows=[];metric_rows=[]
    for name,spec in fitted.items():
        p=predict_head(spec,x,mean,scale)
        for role,ii in (("g_fit",ti),("g_dev",vi)):metric_rows.append(dict(model=name,split_role=role,**metrics(y[ii],p[ii],mids[ii])))
        prediction_rows.extend(dict(model=name,sample_id=r["sample_id"],physical_match_id=r["physical_match_id"],split_role=r["split_role"],realized_gain=r["realized_gain"],predicted_gain=float(p[i])) for i,r in enumerate(records))
    # Epoch selection uses dev regression; final model/stop selection uses full dev
    # policy log loss, and never uses validation/test targets.
    selection=[];dev_paths=[];dev_ids=sorted(mid for mid,s in panel.sequences.items() if s["split_role"]=="g_dev" and mid in panel.outcomes)
    for name,spec in fitted.items():
        print(json.dumps(dict(stage='g_dev_policy_selection',model=name)),flush=True)
        for threshold in (-.01,-.003,0.,.003,.01):
            rr=[]
            for mid in dev_ids:
                path=settle_path(run_policy(evaluator,mid,budget=3,strategy="g",spec=spec,target_mean=mean,target_scale=scale,threshold=threshold),panel.outcomes[mid])
                path["model"]=name;rr.append(path)
            selection.append(dict(model=name,threshold=threshold,budget=3,matches=len(rr),mean_final_log_loss=float(np.mean([r["final_log_loss"] for r in rr])),mean_updates=float(np.mean([r["f_updates"] for r in rr]))))
            dev_paths.extend(rr)
    selected=min(selection,key=lambda r:(r["mean_final_log_loss"],r["mean_updates"],r["model"],r["threshold"]))
    bundle=dict(schema="sequential-postread-g-v1",f_model_sha256=evaluator.f_sha,f_model_dir=str(args.f_model),fitted=fitted,
        selected=selected,target_mean=mean,target_scale=scale,state_width=state_width,embedding_dimension=256,
        feature_dimension=x.shape[1],current_hidden_dimension=evaluator.hidden_dimension,metadata_names=panel.metadata_names,feature_order=["current_f_probabilities","entropy","adopted_count","remaining_count","remaining_budget","adopted_mean_embedding","adopted_mean_metadata","current_f_hidden_state","candidate_embedding","absolute_difference","elementwise_product","candidate_metadata","maximum_adopted_cosine"],
        train_matches=sorted(set(mids[ti])),dev_matches=sorted(set(mids[vi])),production_ready=False)
    bundle["source_sha256"]=source_hashes
    bundle["dataset_dir"]=str(args.data.resolve())
    bundle["vectors_source"]=str(args.vectors.resolve())
    joblib.dump(bundle,args.output/"g-models.joblib",compress=3)
    save_json(args.output/"selection-frozen.json",dict(selected=selected,all_dev_policy_choices=selection,test_used=False,validation_used=False))
    save_rows(args.output/"predictions.jsonl",prediction_rows);save_rows(args.output/"training-curves.jsonl",curves)
    save_rows(args.output/"dev-policy-paths.jsonl",dev_paths);save_json(args.output/"metrics.json",metric_rows)
    save_json(args.output/"model.json",dict(schema=bundle["schema"],f_model_sha256=evaluator.f_sha,g_model_sha256=sha_file(args.output/"g-models.joblib"),selected=selected,configurations=configs,feature_order=bundle["feature_order"],feature_dimension=x.shape[1],state_width=state_width,
        rows=len(records),matches=len(set(mids)),positive=int((y>0).sum()),zero=int((y==0).sum()),negative=int((y<0).sum()),failure_rows=len(failures)+len(panel.failures),
        candidate_coverage="all eligible candidates have s0 transitions; three bounded source/random prefix orders depth<=3",f_calls=evaluator.calls,f_seconds=evaluator.seconds,elapsed_seconds=time.perf_counter()-started,
        zero_or_negative_labels_retained=True,original_old_g_status="not_reused_across_incompatible_f_binding",source_requests=0,production_ready=False,
        source_sha256=source_hashes,current_hidden_dimension=evaluator.hidden_dimension,metadata_control_columns=metadata_columns,
        limitations=["Historical version assumptions inherited","Shared outcome file is filtered by role before any labels are used; validation/test labels do not enter training or model selection","No new blind-test claim for previously examined matches"] ))
    plots(args.output,curves,prediction_rows)
    training_size_curve(args.output,x,y,mids,ti,vi,fitted,configs,args.seed)
    block_widths=[3,1,1,1,1,256,len(panel.metadata_names),evaluator.hidden_dimension,256,256,256,len(panel.metadata_names),1]
    offset=0;blocks=[]
    for name,width in zip(bundle['feature_order'],block_widths):
        blocks.append(dict(name=name,start=offset,stop=offset+width,dimension=width));offset+=width
    if offset!=x.shape[1]:raise ValueError('Feature schema width mismatch')
    save_json(args.output/'feature-schema.json',dict(blocks=blocks,total_dimension=offset,state_width=state_width,
        metadata_names=panel.metadata_names,metadata_control_columns=metadata_columns,
        current_hidden_source='same frozen f current adopted prefix; no candidate-after state',
        forbidden_inputs=['outcome','realized_gain','candidate_after_probability','candidate_after_hidden_state'],
        text_representation='frozen Qwen1024 prefix256 then L2 normalization'))
    if any(sha_file(p)!=digest for p,digest in source_hashes.items()):raise ValueError("Frozen training dataset changed")
    save_json(args.output/'training-completion.json',dict(status='completed',elapsed_seconds_including_diagnostics=time.perf_counter()-started,
        used_roles=['g_fit','g_dev'],validation_used=False,test_used=False,f_model_sha256=evaluator.f_sha))
    save_json(args.output/"manifest.json",{p.name:sha_file(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    print(json.dumps(dict(status="trained_research_only",output=str(args.output),selected=selected,rows=len(records),matches=len(set(mids))),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=BASE/"data");p.add_argument("--vectors",type=Path,default=CACHE)
    p.add_argument("--f-model",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,default=20261025);p.add_argument("--max-epochs",type=int,default=200)
    args=p.parse_args()
    import torch
    torch.set_num_threads(1)
    def no_network(*a,**k):raise RuntimeError("Offline g training only")
    socket.create_connection=socket.socket.connect=socket.getaddrinfo=no_network
    with threadpool_limits(limits=1):train(args)


if __name__=="__main__":main()
