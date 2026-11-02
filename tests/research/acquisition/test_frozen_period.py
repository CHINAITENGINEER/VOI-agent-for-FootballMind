from copy import deepcopy
from hashlib import sha256

import pytest

from football_research.acquisition.conservative_joint import ConservativeJointReturnPredictor
from football_research.evaluation.evidence_strength_replay import evaluate_fold, summarize_paired
from football_research.evaluation.frozen_period import audit_frozen_period, load_frozen_models
from test_conservative_joint import calibration_rows
from test_joint_returns import ACTIONS, fitted


def audit_inputs():
    """Synthetic boundary fixture; never presented as a collected match."""
    freeze = {"parameter_retuning_permitted": False, "candidate_generation_complete": False,
              "historical_reconstruction_only": True, "frozen_at": "2026-10-01T00:00:00Z",
              "minimum_prediction_cutoff": "2026-08-01T00:00:00Z", "source": "understat",
              "league": "EPL", "season": "2026", "forbidden_match_ids": ["old-test"]}
    capture = {"request_started_at": "2026-10-02T00:00:00Z", "observed_at": "2026-10-02T00:00:02Z",
               "http_status": 200, "season": 2026, "league": "EPL", "evidence_id": "capture", "sha256": "a"*64}
    manifest = {"source": "understat", "season": "2026", "information_mode": "historical_reconstruction",
                "source_observed_at": capture["observed_at"], "source_evidence": [{"content_hash": capture["sha256"],
                    "observation_id": "capture", "observed_at": capture["observed_at"],
                    "original_context": {"league": "EPL", "season": 2026}}]}
    row = {"match_id": "new", "competition_id": "league", "source_id": "provider-new",
           "source": "understat", "season": "2026", "observation_id": "capture",
           "home_team_id": "unknown-promoted-team", "away_team_id": "away", "home_goals": 1, "away_goals": 0,
           "status": "finished", "score_period": "regulation", "information_mode": "historical_reconstruction",
           "prediction_cutoff": "2026-08-03T00:00:00Z", "kickoff": "2026-08-04T00:00:00Z",
           "result_available_at": "2026-08-04T03:00:00Z", "features_available_at": "2026-08-02T00:00:00Z",
           "fixture_available_at": "2026-08-02T00:00:00Z", "dataset_version": "platform-history-research-v1.0:"+"b"*64}
    meta = {"training_as_of": "2026-07-01T00:00:00Z", "training_match_ids": ["train"],
            "base_training_match_ids": ["base"], "evidence_strength_adaptation": {"selection_match_ids": ["adapt"]}}
    return {"freeze": freeze, "manifest": manifest, "rows": [row], "manifest_sha256": "b"*64,
            "capture": capture, "normalized_matches": [deepcopy(row)], "competition_id": "league",
            "reference_metadata": meta, "adapted_metadata": deepcopy(meta)}


def test_frozen_period_preserves_unknown_identity_and_missing_history():
    args = audit_inputs()
    result = audit_frozen_period(**args)
    assert result["all_completed_matches_retained"]
    assert result["unknown_identity_or_missing_history_filter_applied"] is False
    assert result["prospective_collection"] is False
    assert result["independent_outcome_verified"] is False


@pytest.mark.parametrize("fault", ["old_test", "train", "adapt", "source_before_freeze", "wrong_season", "changed_bytes",
                                  "future_label", "late_features", "early_prediction", "filtered_completed",
                                  "duplicate", "swapped_result", "wrong_dataset", "different_competition"])
def test_temporal_admission_rejects_contamination_and_selection(fault):
    a = audit_inputs()
    if fault in {"old_test", "train", "adapt"}:
        a["rows"][0]["match_id"] = a["normalized_matches"][0]["match_id"] = {"old_test": "old-test"}.get(fault, fault)
    elif fault == "source_before_freeze":
        a["capture"]["request_started_at"] = "2026-09-30T00:00:00Z"
    elif fault == "wrong_season":
        a["manifest"]["season"] = "2025"
    elif fault == "changed_bytes":
        a["capture"]["sha256"] = "c"*64
    elif fault == "future_label":
        a["rows"][0]["result_available_at"] = "2026-11-01T00:00:00Z"
    elif fault == "late_features":
        a["rows"][0]["features_available_at"] = a["rows"][0]["kickoff"]
    elif fault == "early_prediction":
        a["rows"][0]["prediction_cutoff"] = "2026-06-30T00:00:00Z"
    elif fault == "filtered_completed":
        a["normalized_matches"].append({**a["normalized_matches"][0], "match_id": "omitted", "source_id": "omitted"})
    elif fault == "duplicate":
        a["rows"].append(deepcopy(a["rows"][0]))
    elif fault == "swapped_result":
        a["rows"][0]["home_goals"] = 3
    elif fault == "wrong_dataset":
        a["rows"][0]["dataset_version"] = "other"
    elif fault == "different_competition":
        a["normalized_matches"][0]["competition_id"] = "other"
    with pytest.raises(ValueError):
        audit_frozen_period(**a)


def test_artifact_loader_checks_reference_adaptation_design_and_exact_bytes(fitted, tmp_path):
    reference = fitted[0]
    adapted = ConservativeJointReturnPredictor(reference, strength=.6)
    reference.save(tmp_path/"ref.json")
    adapted.save(tmp_path/"adapt.json")
    design = tmp_path/"design.txt"
    design.write_text("synthetic mechanism test", encoding="utf-8")
    digest = lambda path: sha256(path.read_bytes()).hexdigest()
    freeze = {"reference_model_path": "ref.json", "adapted_model_path": "adapt.json",
              "reference_model_sha256": digest(tmp_path/"ref.json"), "adapted_model_sha256": digest(tmp_path/"adapt.json"),
              "adapted_model_version": adapted.model_version, "evidence_strength": .6, "design_sha256": digest(design)}
    assert load_frozen_models(tmp_path, freeze, design_path=design)[1].model_version == adapted.model_version
    with pytest.raises(ValueError, match="agree"):
        load_frozen_models(tmp_path, {**freeze, "evidence_strength": .5}, design_path=design)
    with (tmp_path/"adapt.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="hash"):
        load_frozen_models(tmp_path, freeze, design_path=design)


def test_paired_evaluation_is_repeatable_and_keeps_same_prior_across_controls(fitted, tmp_path):
    reference, rows, _ = fitted
    adapted = ConservativeJointReturnPredictor(reference, strength=.4)
    heldout = calibration_rows(rows)[:3]
    before = deepcopy(reference._payload())
    results, updates = evaluate_fold(heldout, reference, adapted, ACTIONS, lambda_money=1., output=tmp_path/"first", fold="test")
    second, other_updates = evaluate_fold(list(reversed(heldout)), reference, adapted, ACTIONS,
                                         lambda_money=1., output=tmp_path/"second", fold="test")
    # The reused all-available evaluator measures wall-clock CPU latency.
    # Scores, decisions and evidence must reproduce; measured latency need not.
    stable = lambda records: [{k: v for k, v in r.items() if k not in {"predictor_latency_ms", "estimator_latency_ms"}}
                              for r in records]
    assert stable(results) == stable(second) and updates == other_updates
    assert summarize_paired(results, updates) == summarize_paired(second, other_updates)
    assert reference._payload() == before
    for mid in {r["match_id"] for r in results}:
        assert len({tuple(r["initial_probabilities"]) for r in results if r["match_id"] == mid}) == 1
    with pytest.raises(ValueError, match="Repeated"):
        summarize_paired(results+[results[0]], updates)
    with pytest.raises(ValueError, match="same nonempty"):
        summarize_paired(results[1:], updates)
