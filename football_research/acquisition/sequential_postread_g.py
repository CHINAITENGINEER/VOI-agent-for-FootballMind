"""Offline, fixed-f post-read selection experiments; no production imports.

The predictor sees current probabilities and already observed text vectors.
Outcomes and candidate-after probabilities exist only in label/evaluation code.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from .postread_semantic_embedding import EncoderContract, normalize_prefix


META_NAMES = ("log1p_days_before_match", "publication_time_unknown", "historical_version_unknown",
              "source_is_official", "log1p_body_characters", "historical_reconstruction")


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def save_rows(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        for record in records:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")


def probabilities(value):
    p=np.asarray(value,dtype=np.float64)
    if p.shape!=(3,) or not np.isfinite(p).all() or (p<=0).any() or (p>1).any() or not np.isclose(p.sum(),1,rtol=0,atol=1e-6):
        raise ValueError("Frozen f must return three finite positive probabilities summing to one")
    return p


class VectorStore:
    """Read only previously encoded vectors from npz or a content-addressed cache."""
    def __init__(self,path):
        self.path=Path(path);self.archive=None;self.loaded={}
        if self.path.is_file():self.archive=np.load(self.path,allow_pickle=False)

    def get(self,text_sha256):
        if text_sha256 in self.loaded:return self.loaded[text_sha256]
        if self.archive is not None:
            v=self.archive[text_sha256]
        else:
            meta=json.loads((self.path/(text_sha256+".json")).read_text(encoding="utf-8"))
            file=self.path/(text_sha256+".npy")
            if meta["text_sha256"]!=text_sha256 or sha_file(file)!=meta["vector_file_sha256"] or meta.get("contract_fingerprint")!=EncoderContract().fingerprint():
                raise ValueError("Frozen vector cache digest mismatch")
            v=np.load(file,mmap_mode="r",allow_pickle=False)
        if v.shape!=(1024,) or v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError("Expected frozen Qwen1024 float32")
        self.loaded[text_sha256]=v
        return v


class EvidencePanel:
    """Inputs and outcomes are physically separate; outcomes require opt-in."""
    def __init__(self,data_dir,vectors,*,roles,include_outcomes=False,max_candidates=None):
        self.data_dir=Path(data_dir);self.vectors=vectors;self.max_candidates=max_candidates
        schema=json.loads((self.data_dir/"schema.json").read_text(encoding="utf-8"))
        self.metadata_names=schema["document_meta_names"]
        self.matches={};self.docs={};self.sequences={};self.failures=[];self.outcomes={}
        documents=rows(self.data_dir/"documents.jsonl")
        for d in documents:
            key=d["document_id"]
            if key in self.docs and self.docs[key]["text_sha256"]!=d["text_sha256"]:raise ValueError("Conflicting document ID")
            self.docs[key]=d
        all_matches=rows(self.data_dir/"matches.jsonl")
        ids=[r["physical_match_id"] for r in all_matches]
        if len(ids)!=len(set(ids)):raise ValueError("physical_match_id must occur once")
        for match in all_matches:
            if match["split_role"] in roles:self.matches[match["physical_match_id"]]=match
        for sequence in rows(self.data_dir/"sequences.jsonl"):
            mid=sequence["physical_match_id"]
            if mid not in self.matches:continue
            if mid in self.sequences:raise ValueError("Duplicate match sequence")
            if sequence["split_role"]!=self.matches[mid]["split_role"]:raise ValueError("Sequence split disagrees with match")
            try:
                if not sequence.get("eligible",False):raise ValueError("sequence_not_eligible")
                prior=probabilities(self.matches[mid]["prior"])
                doc_ids=list(sequence["document_ids"])
                if len(doc_ids)!=len(set(doc_ids)):raise ValueError("duplicate_document_in_sequence")
                meta=np.asarray(sequence["document_meta"],dtype=np.float64)
                if meta.shape!=(len(doc_ids),len(self.metadata_names)) or not np.isfinite(meta).all():raise ValueError("invalid_explicit_document_metadata")
                retained=doc_ids if max_candidates is None else doc_ids[:max_candidates]
                vector=np.stack([vectors.get(self.docs[d]["text_sha256"]) for d in retained])
                contexts={r["document_id"]:r.get("metadata_context") for r in sequence.get("documents",[])}
                if len(self.metadata_names)>6 and any(contexts.get(d) is None for d in retained):
                    raise ValueError("match_specific_metadata_context_missing")
                contextual_documents=[dict(self.docs[d],**(contexts.get(d) or {})) for d in retained]
                self.sequences[mid]=dict(document_ids=retained,vectors=vector,meta=meta[:max_candidates],prior=prior,
                    document_records=contextual_documents,match_date=self.matches[mid]["match_date"],
                    metadata_names=self.metadata_names,
                    original_candidate_count=len(doc_ids),capped_candidate_count=len(retained),order_basis=sequence.get("order_basis"),
                    capped_by="none" if max_candidates is None else "declared_source_order_not_gain",split_role=sequence["split_role"])
                computed=causal_metadata(self.sequences[mid],range(len(retained)))
                if not np.allclose(computed,meta[:len(retained)],atol=1e-12,rtol=0):
                    del self.sequences[mid]
                    raise ValueError("Saved metadata differs from causal match-context reconstruction")
            except (ValueError,KeyError,FileNotFoundError) as exc:
                self.sequences.pop(mid,None)
                self.failures.append(dict(physical_match_id=mid,reason=str(exc),stage="input_preparation",gain=None))
        if include_outcomes:
            for row in rows(self.data_dir/"outcomes.jsonl"):
                mid=row["physical_match_id"]
                if mid in self.sequences:
                    if mid in self.outcomes:raise ValueError("Duplicate independent outcome")
                    y=row["outcome_class"]
                    if y is None:continue
                    if isinstance(y,bool) or y not in (0,1,2):raise ValueError("Invalid outcome class")
                    self.outcomes[mid]=row


class FrozenEvaluator:
    """Cache current evidence predictions by ordered document IDs and fixed f SHA."""
    def __init__(self,model,panel):
        self.model=model;self.panel=panel;self.cache={};self.hidden_cache={};self.calls=0;self.seconds=0.
        self.hidden_dimension=int(model.config.hidden_dim)
        self.f_sha=model.f_model_sha256
        if hasattr(model,"config") and hasattr(panel,"metadata_names") and list(model.config.metadata_names)!=panel.metadata_names:
            raise ValueError("f metadata schema differs from the frozen dataset")

    def predict(self,mid,adopted):
        if self.model.f_model_sha256!=self.f_sha:raise ValueError("f binding changed")
        adopted=tuple(int(i) for i in adopted);key=(mid,adopted)
        if key in self.cache:return self.cache[key]
        sequence=self.panel.sequences[mid]
        if len(adopted)!=len(set(adopted)) or any(i<0 or i>=len(sequence["document_ids"]) for i in adopted):raise ValueError("Invalid adopted evidence indices")
        indices=np.array(adopted,dtype=int);start=time.perf_counter()
        prefixes=self.model.predict_prefixes(sequence["prior"],sequence["vectors"][indices],
            causal_metadata(sequence,adopted),order=[sequence["document_ids"][i] for i in indices])
        self.seconds+=time.perf_counter()-start;self.calls+=1
        if len(prefixes)!=len(adopted)+1:raise ValueError("Frozen f omitted causal prefixes")
        for depth,record in enumerate(prefixes):
            p=probabilities(record["probabilities"]);hidden=np.asarray(record["hidden_state"],dtype=float)
            if hidden.shape!=(self.hidden_dimension,) or not np.isfinite(hidden).all():raise ValueError("Invalid current f hidden state")
            prefix_key=(mid,adopted[:depth])
            if prefix_key in self.cache:
                if not np.allclose(self.cache[prefix_key],p,atol=1e-6,rtol=0) or not np.allclose(self.hidden_cache[prefix_key],hidden,atol=1e-6,rtol=0):
                    raise ValueError("Same causal f prefix changed with future candidate context")
            else:self.cache[prefix_key]=p;self.hidden_cache[prefix_key]=hidden
        return self.cache[key]

    def current_hidden(self,mid,adopted):
        self.predict(mid,adopted)
        return self.hidden_cache[(mid,tuple(int(i) for i in adopted))]


def causal_metadata(sequence,adopted,candidate=None):
    """Recompute relation fields for this order; never slice future-pool relations."""
    if "document_records" not in sequence:
        # Explicit synthetic test-only fixtures use invariant metadata.
        return sequence["meta"][np.array(adopted,dtype=int)] if candidate is None else sequence["meta"][candidate]
    from .full_history_sequence_metadata import document_metadata, METADATA_NAMES
    if list(METADATA_NAMES)!=sequence["metadata_names"]:raise ValueError("Shared causal metadata contract differs")
    docs=sequence["document_records"]
    if candidate is not None:
        return np.asarray(document_metadata(docs[candidate],[docs[i] for i in adopted],sequence["match_date"]),dtype=float)
    values=[];known=[]
    for i in adopted:
        values.append(document_metadata(docs[i],known,sequence["match_date"]));known.append(docs[i])
    return np.asarray(values,dtype=float).reshape(len(adopted),len(METADATA_NAMES))


def feature_vector(current_probability,adopted_vectors,candidate_vector,adopted_meta,candidate_meta,remaining,budget_remaining,dimension=256,*,current_hidden=None):
    """No outcome, after probability, or gain argument is accepted here."""
    p=probabilities(current_probability)
    adopted_vectors=np.asarray(adopted_vectors);adopted_meta=np.asarray(adopted_meta,dtype=np.float64)
    candidate_meta=np.asarray(candidate_meta,dtype=np.float64)
    if candidate_meta.ndim!=1 or adopted_vectors.ndim!=2 or adopted_meta.shape!=(len(adopted_vectors),len(candidate_meta)):raise ValueError("Invalid explicit state metadata")
    candidate=normalize_prefix(candidate_vector,dimension).astype(np.float64)
    vectors=np.stack([normalize_prefix(v,dimension) for v in adopted_vectors]) if len(adopted_vectors) else np.zeros((0,dimension))
    average=vectors.mean(axis=0) if len(vectors) else np.zeros(dimension)
    mean_meta=adopted_meta.mean(axis=0) if len(adopted_meta) else np.zeros(len(candidate_meta))
    hidden=np.asarray([] if current_hidden is None else current_hidden,dtype=float)
    if hidden.ndim!=1 or not np.isfinite(hidden).all():raise ValueError("Invalid explicit current hidden state")
    state=np.r_[p,-np.sum(p*np.log(p)),len(vectors),remaining,budget_remaining,average,mean_meta,hidden]
    similarity=float(max(vectors@candidate)) if len(vectors) else 0.
    x=np.r_[state,candidate,np.abs(candidate-average),candidate*average,candidate_meta,similarity]
    if not np.isfinite(x).all():raise ValueError("Nonfinite g input")
    return x,len(state)


def candidate_features(evaluator,mid,adopted,candidate,budget,dimension=256):
    sequence=evaluator.panel.sequences[mid];ii=np.array(adopted,dtype=int)
    return feature_vector(evaluator.predict(mid,adopted),sequence["vectors"][ii],sequence["vectors"][candidate],
        causal_metadata(sequence,adopted),causal_metadata(sequence,adopted,candidate),len(sequence["document_ids"])-len(adopted),max(0,budget-len(adopted)),dimension,
        current_hidden=evaluator.current_hidden(mid,adopted))


def generate_transitions(evaluator,*,seed=20261025,orders_per_match=3,budget=3,dimension=256,feature_budgets=None):
    panel=evaluator.panel;features=[];records=[];failures=[];seen=set();state_width=None
    for mid,sequence in sorted(panel.sequences.items()):
        if mid not in panel.outcomes:
            failures.append(dict(physical_match_id=mid,reason="independent_outcome_missing",gain=None));continue
        y=panel.outcomes[mid]["outcome_class"];n=len(sequence["document_ids"])
        rng=np.random.default_rng(seed+int(hashlib.sha256(mid.encode()).hexdigest()[:8],16))
        orders=[tuple(range(n))]+[tuple(rng.permutation(n).tolist()) for _ in range(orders_per_match-1)]
        for order in orders:
            for depth in range(min(n,budget)):
                adopted=order[:depth]
                for candidate in range(n):
                    if candidate in adopted:continue
                    for decision_budget in (feature_budgets or (budget,)):
                        if decision_budget<=depth:continue
                        key=(mid,adopted,candidate,decision_budget)
                        if key in seen:continue
                        seen.add(key)
                        sid=hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).hexdigest()
                        try:
                            x,width=candidate_features(evaluator,mid,adopted,candidate,decision_budget,dimension)
                            before=evaluator.predict(mid,adopted);after=evaluator.predict(mid,(*adopted,candidate))
                            lb=float(-np.log(before[y]));la=float(-np.log(after[y]));gain=lb-la
                            if state_width is not None and width!=state_width:raise ValueError("Changing state width")
                            state_width=width;features.append(x)
                            document=sequence.get("document_records",[{}]*n)[candidate]
                            records.append(dict(sample_id=sid,physical_match_id=mid,split_role=sequence["split_role"],budget=decision_budget,
                                candidate_id=sequence["document_ids"][candidate],candidate_index=candidate,
                                candidate_text_sha256=document.get("text_sha256"),source_url=document.get("source_url"),
                                source_clocks={key:document.get(key) for key in ("published_at","updated_at","source_available_at","captured_at")},
                                adopted_ids=[sequence["document_ids"][i] for i in adopted],adopted_indices=list(adopted),
                                action_id="offline-adopt:"+sequence["document_ids"][candidate],propensity=None,
                                propensity_basis="bounded_offline_enumeration_not_logged_agent",p_before=before.tolist(),p_after=after.tolist(),
                                outcome_class=y,loss_before=lb,loss_after=la,realized_gain=gain,f_model_sha256=evaluator.f_sha,
                                eligibility="historical_offline_research",source_time_assumptions="inherited_from_frozen_dataset",
                                f_executed=True,zero_gain_is_not_failure=True))
                        except (ValueError,RuntimeError,KeyError) as exc:
                            failures.append(dict(sample_id=sid,physical_match_id=mid,candidate_index=candidate,budget=decision_budget,
                                adopted_indices=list(adopted),reason=str(exc),gain=None,eligibility="failed_not_zero"))
    if not records:raise ValueError("No valid supervised transitions")
    return np.asarray(features),records,failures,state_width


def equal_match_weights(matches):
    counts=Counter(matches);n=len(matches)
    return np.array([n/(len(counts)*counts[mid]) for mid in matches])


def metrics(actual,predicted,matches):
    from scipy.stats import spearmanr
    actual=np.asarray(actual);predicted=np.asarray(predicted);residual=predicted-actual
    grouped={mid:residual[np.array(matches)==mid] for mid in set(matches)}
    corr=spearmanr(actual,predicted).statistic if np.std(actual)>0 and np.std(predicted)>0 else None
    return dict(rows=len(actual),matches=len(grouped),mse=float(np.mean(residual**2)),mae=float(np.mean(abs(residual))),
        match_macro_mse=float(np.mean([np.mean(r**2) for r in grouped.values()])),
        match_macro_mae=float(np.mean([np.mean(abs(r)) for r in grouped.values()])),
        spearman=float(corr) if corr is not None and np.isfinite(corr) else None,
        positive_sign_accuracy=float(np.mean((predicted>0)==(actual>0))),mean_residual=float(np.mean(residual)))


def predict_head(spec,x,target_mean=0.,target_scale=1.):
    if spec["kind"]=="zero":return np.zeros(len(x))
    if spec["kind"]=="constant":return np.full(len(x),spec["value"])
    if spec["input"]=="state":design=x[:,:spec["state_width"]]
    elif spec["input"]=="metadata":design=x[:,spec["column_indices"]]
    else:design=x
    return spec["model"].predict(spec["scaler"].transform(design))*target_scale+target_mean


def run_policy(evaluator,mid,*,budget,strategy,spec=None,target_mean=0.,target_scale=1.,threshold=0.,seed=20261025,dimension=256):
    run_started=time.perf_counter();f_seconds_before=evaluator.seconds;f_calls_before=evaluator.calls
    sequence=evaluator.panel.sequences[mid];n=len(sequence["document_ids"]);selected=[];trace=[];scores_seconds=0.
    before=evaluator.predict(mid,[]);rng=np.random.default_rng(seed)
    stop="budget_exhausted";max_steps=n if strategy=="all" else min(n,budget)
    if strategy=="none":max_steps=0;stop="no_new_evidence_baseline"
    for _ in range(max_steps):
        remaining=[i for i in range(n) if i not in selected]
        if not remaining:stop="pool_exhausted";break
        if strategy in ("all","fixed"):
            candidate=remaining[0];scores=None
        elif strategy=="random":candidate=int(rng.choice(remaining));scores=None
        elif strategy=="g":
            start=time.perf_counter()
            xx=np.stack([candidate_features(evaluator,mid,selected,i,budget,dimension)[0] for i in remaining])
            pp=predict_head(spec,xx,target_mean,target_scale);scores_seconds+=time.perf_counter()-start
            scores=[dict(candidate_index=i,predicted_gain=float(v)) for i,v in zip(remaining,pp)]
            best=int(np.argmax(pp));candidate=remaining[best]
            trace.append(dict(adopted_indices=selected.copy(),scores=scores,threshold=threshold))
            if pp[best]<=threshold:stop="predicted_gain_below_frozen_threshold";break
        else:raise ValueError("Unknown strategy")
        if scores is None:trace.append(dict(adopted_indices=selected.copy(),selected_candidate=candidate))
        selected.append(candidate)
    if len(selected)==n:stop="pool_exhausted"
    after=evaluator.predict(mid,selected)
    return dict(physical_match_id=mid,strategy=strategy,budget=budget,candidate_count=n,
        selected_indices=selected,selected_ids=[sequence["document_ids"][i] for i in selected],
        unselected_ids=[sequence["document_ids"][i] for i in range(n) if i not in selected],
        f_updates=len(selected),stopping_position=len(selected),stop_reason=stop,threshold=threshold,
        p_before=before.tolist(),p_after=after.tolist(),trace=trace,g_feature_and_scoring_seconds=scores_seconds,
        no_duplicate_adoption=len(selected)==len(set(selected)),f_model_sha256=evaluator.f_sha,
        actual_f_forward_calls=evaluator.calls-f_calls_before,actual_f_forward_seconds=evaluator.seconds-f_seconds_before,
        g_scoring_time_includes_current_f_cache_misses=True,
        cost=dict(discovery_seconds=None,body_fetch_seconds=None,embedding_seconds=None,
                  recorded_replay_wall_seconds=time.perf_counter()-run_started,
                  f_forward_seconds=evaluator.seconds-f_seconds_before,
                  f_forward_cache_misses=evaluator.calls-f_calls_before,
                  g_reading_and_scoring_seconds_including_current_f=scores_seconds,
                  cost_scope='actual cached offline replay only; not prospective total system cost',
                  inherited_sunk_costs_unknown=True,real_money_cost=None,simulated_dollar_cost_used=False),
        all_strategy_budget_comparable=(strategy!="all" or n<=budget))


def settle_path(path,outcome):
    y=outcome["outcome_class"]
    path=dict(path);path["outcome_class"]=y
    path["loss_before"]=float(-np.log(path["p_before"][y]));path["final_log_loss"]=float(-np.log(path["p_after"][y]))
    path["realized_gain"]=path["loss_before"]-path["final_log_loss"]
    path["outcome_provenance"]=outcome.get("source_refs")
    return path


def exact_oracle(evaluator,mid,budget,*,max_states=1000):
    """Outcome-only exhaustive ordered oracle, or explicitly unavailable.

    This is not a deployable policy; a resource-capped partial search is never
    called an upper bound.
    """
    n=len(evaluator.panel.sequences[mid]["document_ids"]);budget=min(n,budget)
    unordered=getattr(getattr(evaluator.model,"config",None),"architecture",None)=="set_residual_mlp"
    count=sum(math.comb(n,k) if unordered else math.perm(n,k) for k in range(budget+1))
    if count>max_states:return dict(physical_match_id=mid,status="not_computed_state_limit",states_required=count,oracle_upper_bound=None)
    y=evaluator.panel.outcomes[mid]["outcome_class"];best=();best_loss=float(-np.log(evaluator.predict(mid,())[y]))
    for k in range(1,budget+1):
        choices=itertools.combinations(range(n),k) if unordered else itertools.permutations(range(n),k)
        for selected in choices:
            loss=float(-np.log(evaluator.predict(mid,selected)[y]))
            if loss<best_loss:best_loss=loss;best=selected
    return dict(physical_match_id=mid,status="exact_exhaustive_subsets" if unordered else "exact_exhaustive_ordered",states=count,selected_indices=list(best),
        final_log_loss=best_loss,oracle_upper_bound=True,used_outcome_for_selection=True)


def summarize_paths(paths):
    groups={}
    for path in paths:groups.setdefault((path.get("split_role"),path["strategy"],path["budget"]),[]).append(path)
    return [dict(split_role=key[0],strategy=key[1],budget=key[2],matches=len({r["physical_match_id"] for r in rr}),paths=len(rr),mean_final_log_loss=float(np.mean([r["final_log_loss"] for r in rr])),
        mean_gain=float(np.mean([r["realized_gain"] for r in rr])),mean_updates=float(np.mean([r["f_updates"] for r in rr])),
        mean_g_seconds=float(np.mean([r["g_feature_and_scoring_seconds"] for r in rr])),
        full_system_cost_unknown=True) for key,rr in groups.items()]
