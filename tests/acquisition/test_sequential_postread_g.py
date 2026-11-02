"""Scientific invariants of bounded, state-dependent gain and policy selection."""
from types import SimpleNamespace
import numpy as np
import pytest

from football_research.acquisition.sequential_postread_g import (
    FrozenEvaluator,feature_vector,generate_transitions,run_policy,settle_path,exact_oracle,equal_match_weights,causal_metadata)


class FakeFrozenF:
    f_model_sha256="fixed-test-only"
    config=SimpleNamespace(hidden_dim=4)
    def predict(self,prior,doc_vectors,doc_meta=None,order=None):
        if order and "fail" in order:raise ValueError("deliberate f failure")
        logits=np.log(prior).copy()
        if len(doc_vectors):logits[0]+=.3*doc_vectors[:,0].sum()
        p=np.exp(logits-logits.max());return p/p.sum()
    def predict_prefixes(self,prior,doc_vectors,doc_meta=None,order=None):
        return [dict(probabilities=self.predict(prior,doc_vectors[:i],doc_meta[:i],order[:i]),
                     hidden_state=[float(i),float(doc_vectors[:i,0].sum()),0.,0.]) for i in range(len(doc_vectors)+1)]


def evaluator(failure=False):
    vectors=np.zeros((3,1024),dtype=np.float32);vectors[0,0]=1;vectors[1,0]=-1;vectors[2,1]=1
    sequence=dict(document_ids=["positive","negative","fail" if failure else "neutral"],vectors=vectors,
                  meta=np.zeros((3,6)),prior=np.array([.4,.3,.3]),split_role="g_fit")
    panel=SimpleNamespace(sequences={"match":sequence},outcomes={"match":dict(outcome_class=0)},matches={"match":dict(split_role="g_fit")})
    return FrozenEvaluator(FakeFrozenF(),panel)


def test_all_signed_labels_and_same_f_binding():
    ev=evaluator();x,records,failures,width=generate_transitions(ev)
    root={r["candidate_index"]:r for r in records if not r["adopted_indices"]}
    assert root[0]["realized_gain"]>0 and root[1]["realized_gain"]<0 and abs(root[2]["realized_gain"])<1e-12
    assert all(r["f_model_sha256"]==ev.f_sha for r in records)
    assert len(x)==len(records) and not failures and width==273
    assert x.shape[1]==1048
    assert all(np.isclose(r["realized_gain"],np.log(r["p_after"][0])-np.log(r["p_before"][0])) for r in records)


def test_failure_is_null_never_zero_label():
    ev=evaluator(True);_,records,failures,_=generate_transitions(ev)
    assert failures and all(r["gain"] is None for r in failures)
    assert all(r["candidate_id"]!="fail" for r in records)


def test_outcome_changes_labels_not_features():
    a=evaluator();b=evaluator();b.panel.outcomes["match"]["outcome_class"]=2
    xa,ra,_,_=generate_transitions(a);xb,rb,_,_=generate_transitions(b)
    np.testing.assert_array_equal(xa,xb)
    assert [r["realized_gain"] for r in ra]!=[r["realized_gain"] for r in rb]


def test_state_is_only_previously_adopted_evidence():
    ev=evaluator();seq=ev.panel.sequences["match"]
    x0,_=feature_vector(ev.predict("match",()),seq["vectors"][:0],seq["vectors"][1],seq["meta"][:0],seq["meta"][1],3,3)
    x1,_=feature_vector(ev.predict("match",(0,)),seq["vectors"][:1],seq["vectors"][1],seq["meta"][:1],seq["meta"][1],2,2)
    assert not np.array_equal(x0,x1)
    assert x0[4]==0 and x1[4]==1
    with pytest.raises(TypeError):feature_vector([],[],[],[],[],0,0,outcome=0)


def test_zero_stops_but_negative_label_still_sampled():
    ev=evaluator();path=run_policy(ev,"match",budget=3,strategy="g",spec={"kind":"zero"})
    assert path["selected_indices"]==[] and path["stop_reason"]=="predicted_gain_below_frozen_threshold"
    _,records,_,_=generate_transitions(ev)
    assert any(r["realized_gain"]<0 for r in records)


def test_all_is_uncapped_reference_and_random_reproducible():
    ev=evaluator();all_path=run_policy(ev,"match",budget=1,strategy="all")
    assert all_path["f_updates"]==3 and not all_path["all_strategy_budget_comparable"]
    a=run_policy(ev,"match",budget=2,strategy="random",seed=17);b=run_policy(ev,"match",budget=2,strategy="random",seed=17)
    assert a["selected_indices"]==b["selected_indices"] and len(set(a["selected_indices"]))==2


def test_oracle_exact_or_unavailable_never_partial_upper_bound():
    ev=evaluator();oracle=exact_oracle(ev,"match",2)
    fixed=settle_path(run_policy(ev,"match",budget=2,strategy="fixed"),ev.panel.outcomes["match"])
    assert oracle["final_log_loss"]<=fixed["final_log_loss"] and oracle["oracle_upper_bound"]
    unavailable=exact_oracle(ev,"match",2,max_states=1)
    assert unavailable["oracle_upper_bound"] is None


def test_equal_match_weighting_not_transition_weighting():
    weights=equal_match_weights(["a","a","a","b"])
    assert np.isclose(weights[:3].sum(),weights[3])


def test_relation_metadata_recomputed_for_actual_adopted_order():
    from football_research.acquisition.full_history_sequence_metadata import METADATA_NAMES
    claims=[dict(team_side="home",entity_normalized="player",scope="this_match",kind="availability",value="unavailable")]
    docs=[dict(text_sha256=str(i),published_at="2020-01-01",body_characters=100,existing_claims=claims) for i in range(2)]
    sequence=dict(document_records=docs,metadata_names=METADATA_NAMES,match_date="2020-01-03")
    repeated=METADATA_NAMES.index("repeated_claim_count")
    assert causal_metadata(sequence,(),0)[repeated]==0
    assert causal_metadata(sequence,(1,),0)[repeated]==1
    ordered=causal_metadata(sequence,(1,0))
    assert ordered[:,repeated].tolist()==[0,1]


def test_training_budget_variants_do_not_change_same_state_target():
    ev=evaluator();x,records,_,_=generate_transitions(ev,feature_budgets=(1,2,3,6))
    root=[(i,r) for i,r in enumerate(records) if not r["adopted_indices"] and r["candidate_index"]==0]
    assert {r["budget"] for _,r in root}=={1,2,3,6}
    assert len({r["realized_gain"] for _,r in root})==1
    assert {x[i,6] for i,_ in root}=={1,2,3,6}


def test_current_hidden_changes_state_even_when_probability_and_means_identical():
    ev=evaluator();seq=ev.panel.sequences["match"]
    args=(seq["prior"],seq["vectors"][:0],seq["vectors"][0],seq["meta"][:0],seq["meta"][0],3,3)
    a,width=feature_vector(*args,current_hidden=np.zeros(4))
    b,_=feature_vector(*args,current_hidden=np.ones(4))
    np.testing.assert_array_equal(a[:width-4],b[:width-4])
    np.testing.assert_array_equal(a[width:],b[width:])
    assert np.all(a[width-4:width]!=b[width-4:width])


def test_f_prefix_forward_caches_probability_and_current_hidden_together():
    ev=evaluator();ev.predict("match",(0,1));assert ev.calls==1
    np.testing.assert_array_equal(ev.current_hidden("match",(0,)),[1,1,0,0])
    ev.predict("match",());assert ev.calls==1
    assert ev.current_hidden("match",(0,1))[0]==2
