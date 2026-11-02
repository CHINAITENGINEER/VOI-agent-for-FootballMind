"""Version-bound held-out gain supervision and a matched historical control.

The existing calibrator's oof_* serialization fields are compatibility slots:
same-version records are held-out predictions from one fixed earlier model,
not a sequence of rolling OOF models. Both arms use identical target matches.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import math

from .experiment import state_after, losses, outcome
from .experiment_round9 import joint_transitions
from .joint_returns import ACTION_IDS
from .round9_residual import ResidualEVSICalibrator, VERSION as CALIBRATOR_VERSION
from .state import parse_time, sanitize_state
from .value import _canonical

VERSION="frozen-version-supervision-v1"
LEGAL_TRANSITIONS={((),ACTION_IDS[0]),((),ACTION_IDS[1]),((ACTION_IDS[0],),ACTION_IDS[1]),((ACTION_IDS[1],),ACTION_IDS[0])}


def index_transitions(samples):
    indexed=defaultdict(dict)
    for sample in samples:
        key=(tuple(sorted(sample["acquired_action_ids"])),sample["action"]["action_id"])
        if key not in LEGAL_TRANSITIONS:
            raise ValueError("Illegal source transition")
        mid=str(sample["match_id"])
        if key in indexed[mid]:
            raise ValueError("Duplicate source match/action/acquired transition")
        indexed[mid][key]=sample
    return dict(indexed)


def select_matched_window(rows,*,target_fold,reference_model,mixed_samples):
    lookup={str(r["match_id"]):r for r in rows}
    if len(lookup)!=len(rows):
        raise ValueError("Duplicate match identities in source episodes")
    index=index_transitions(mixed_samples)
    reference_cutoff=parse_time(reference_model.metadata["training_as_of"])
    calcutoff=parse_time(target_fold["split"]["train_cutoff"])
    if reference_cutoff>=calcutoff:
        raise ValueError("Reference predictor must precede calibration window")
    train_ids=set(target_fold["training_match_ids"])
    reference_ids=set(reference_model.base_predictor.metadata["training_match_ids"])
    reference_joint_ids=set(reference_model.metadata["training_match_ids"])
    evaluation_ids=set(target_fold["validation_match_ids"])|set(target_fold["test_match_ids"])
    earliest_prediction=min(parse_time(lookup[mid]["prediction_cutoff"]) for mid in evaluation_ids)
    if calcutoff>earliest_prediction:
        raise ValueError("Calibration cutoff exceeds evaluation prediction boundary")
    nominal={mid for mid,row in lookup.items() if reference_cutoff<parse_time(row["prediction_cutoff"])
             and parse_time(row["kickoff"])<calcutoff}
    selected,excluded=[],[]
    for mid in sorted(train_ids|nominal,key=lambda mid:(parse_time(lookup[mid]["kickoff"]),mid)):
        row=lookup[mid]
        reasons=[]
        if mid not in train_ids:
            reasons.append("excluded_by_original_target_training_split")
        if parse_time(row["prediction_cutoff"])<=reference_cutoff:
            reasons.append("prediction_not_strictly_after_reference_fit")
        if mid in reference_ids:
            reasons.append("member_of_reference_base_training")
        if mid in reference_joint_ids:
            reasons.append("member_of_reference_joint_training")
        if parse_time(row["result_available_at"])>calcutoff:
            reasons.append("outcome_after_calibration_cutoff")
        if parse_time(row["result_available_at"])>=earliest_prediction:
            reasons.append("outcome_not_strictly_before_evaluation_predictions")
        if mid in evaluation_ids:
            reasons.append("evaluation_identity_overlap")
        if mid not in index:
            reasons.append("matched_mixed_supervision_missing")
        elif set(index[mid])!=LEGAL_TRANSITIONS:
            reasons.append("matched_mixed_four_transition_set_incomplete")
        if reasons:
            excluded.append({"match_id":mid,"kickoff":row["kickoff"],"prediction_cutoff":row["prediction_cutoff"],
                "result_available_at":row["result_available_at"],"reasons":reasons,"nominal_window_candidate":mid in nominal})
        else:
            selected.append(row)
    if not selected:
        raise ValueError("Matched calibration window has no eligible outcomes")
    manifest={"reference_training_cutoff":reference_cutoff.isoformat(),"calibration_cutoff":calcutoff.isoformat(),
        "earliest_evaluation_prediction_cutoff":earliest_prediction.isoformat(),"reference_to_calibration_days":(calcutoff-reference_cutoff).total_seconds()/86400,
        "candidate_scope":"original target train IDs union nominal post-reference/pre-calcutoff fixtures, to show original purge exclusions",
        "nominal_window_match_ids":sorted(nominal),"original_target_training_match_ids":sorted(train_ids),
        "calibration_match_ids":[r["match_id"] for r in selected],"calibration_matches":len(selected),"excluded":excluded,
        "window_rule":"original target training IDs; prediction strictly after reference fit; common complete mixed supervision; known labels before evaluation",
        "future_validation_usage":"diagnostic only, no selection"}
    mixed=[deepcopy(index[row["match_id"]][key]) for row in selected for key in sorted(LEGAL_TRANSITIONS)]
    return selected,mixed,manifest


def validate_samples(samples,*,rows,as_of,reference_model,kind,reference_sha256):
    if kind not in {"same_version","matched_mixed"}:
        raise ValueError("Unknown supervision arm")
    lookup={r["match_id"]:r for r in rows}
    indexed=index_transitions(samples)
    if set(indexed)!=set(lookup) or any(set(v)!=LEGAL_TRANSITIONS for v in indexed.values()):
        raise ValueError("Supervision requires identical matched IDs and four legal transitions each")
    cutoff=parse_time(as_of)
    reference_ids=set(reference_model.base_predictor.metadata["training_match_ids"])
    if set(reference_model.metadata["training_match_ids"])-reference_ids:
        raise ValueError("Joint training IDs must be contained in frozen base training IDs")
    for sample in samples:
        row=lookup[sample["match_id"]]
        state=sanitize_state(sample["state"])
        if state["match_id"]!=row["match_id"] or parse_time(state["prediction_cutoff"])!=parse_time(row["prediction_cutoff"]):
            raise ValueError("Supervision target identity/time differs")
        if row["match_id"] in reference_ids:
            raise ValueError("Reference training match used for held-out calibration")
        if (parse_time(sample["kickoff"])!=parse_time(row["kickoff"]) or parse_time(sample["result_available_at"])!=parse_time(row["result_available_at"])
                or not parse_time(sample["oof_train_cutoff"])<parse_time(row["prediction_cutoff"])<parse_time(row["kickoff"])<=parse_time(row["result_available_at"])<=cutoff):
            raise ValueError("Supervision violates target chronology")
        if row["match_id"] in set(sample["oof_train_ids"]):
            raise ValueError("Generating predictor trained on its calibration target")
        actual_state=state_after(row,[{"action_id":"results_view","group_ids":["elo","recent_form"]},
            {"action_id":"xg_view","group_ids":["xg"]}],sample["acquired_action_ids"])
        if state!=sanitize_state(actual_state):
            raise ValueError("Supervision public state differs from the matched historical projection")
        for key in ("model_evsi","delta","residual"):
            if not math.isfinite(sample[key]):
                raise ValueError("Nonfinite supervision")
        before,after=sample["prediction_before"],sample["prediction_after"]
        for p in (before,after):
            if len(p)!=3 or any(not math.isfinite(v) or v<=0 for v in p) or abs(sum(p)-1)>1e-8:
                raise ValueError("Invalid supervision prediction")
        actual=losses(before,outcome(row))["log_loss"]-losses(after,outcome(row))["log_loss"]
        if abs(actual-sample["delta"])>1e-11 or abs(sample["residual"]-(sample["delta"]-sample["model_evsi"]))>1e-11:
            raise ValueError("Supervision gain/residual inconsistent with prediction and outcome")
        if kind=="same_version":
            if (sample["oof_predictor_version"]!=reference_model.model_version or sample["oof_manifest_ref"]!="sha256:"+reference_sha256
                    or set(sample["oof_train_ids"])!=reference_ids
                    or parse_time(sample["oof_train_cutoff"])!=parse_time(reference_model.metadata["training_as_of"])):
                raise ValueError("Same-version supervision changed its reference predictor")
    return samples


def matched_supervision(rows,mixed,model,actions,*,as_of,reference_sha256):
    provenance={"oof_train_cutoff":model.metadata["training_as_of"],
        "oof_train_ids":model.base_predictor.metadata["training_match_ids"],"joint_training_match_ids":model.metadata["training_match_ids"],
        "oof_manifest_ref":"sha256:"+reference_sha256,"supervision_kind":"same_version_heldout_fixed_predictor",
        "compatibility_note":"oof_* names retained for existing calibrator serialization; generating model fixed across this held-out window"}
    same=joint_transitions(rows,model,actions,provenance)
    for sample in mixed:
        sample["supervision_kind"]="matched_mixed_past_forward_oof"
    for kind,samples in (("same_version",same),("matched_mixed",mixed)):
        validate_samples(samples,rows=rows,as_of=as_of,reference_model=model,kind=kind,reference_sha256=reference_sha256)
    return {"same_version":same,"matched_mixed":mixed}


def fit_calibrator(samples,*,kind,as_of,dataset_version,model,model_sha256,metadata):
    result=ResidualEVSICalibrator().fit(samples,as_of=as_of,diagnostics_as_of=as_of,dataset_version=dataset_version,
        joint_model=model,joint_artifact_sha256=model_sha256,calibration_samples=(),
        metadata={**metadata,"supervision_arm":kind,"supervision_kind":"same_version_heldout_fixed_predictor" if kind=="same_version" else "matched_mixed_past_forward_oof",
            "legacy_oof_field_semantics":"compatibility names, inspect supervision_kind for generating-model design",
            "inner_split":"none; entire prespecified held-out calibration window fits residuals",
            "outer_validation_usage":"diagnostic only; no hyperparameter/lambda choice","lambda_rule":"fixed existing reporting alpha=0.5"})
    # Keep the old serialized class/schema, while making its experiment-specific
    # declarations accurate. Rehash only this newly created artifact.
    result.manifest["calibration_use"]="no_inner_diagnostic_tail; subsequent original validation is diagnostics only"
    result.manifest["limitations"]=["Finite matched calibration window; residual stationarity and predictive calibration unproven",
        "Descriptive week-cluster mean lower bound has no uniform/post-selection coverage guarantee",
        "Complete historical projections and reconstructed times differ from live acquisition",
        "EVSI and realized-gain generating version both change between matched arms"]
    result.calibration={"scope":"no_inner_tail_in_fixed_window_design","version_binding":"enclosing_artifact_version","records":[]}
    result.version=CALIBRATOR_VERSION+":"+sha256(_canonical(result._payload()).encode()).hexdigest()[:20]
    return result
