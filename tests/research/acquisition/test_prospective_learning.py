"""Cohort contracts with constructed fixtures, never real-effect evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import shutil
import socket

import pytest

from football_research.data_adapters import prospective_learning as module
from football_research.data_adapters.prospective_learning import (
    prepare_learning_plan, validate_learning_plan, import_training_exposure,
)
from football_research.data_adapters.prospective_text import MarketPriorPolicy, _digest
from football_research.acquisition.text_returns import TextReturnModel
from test_verified_collection import verified_inputs, _collect
from test_cohort_runtime import _past_inputs, _records, _registered


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("No network in cohort unit tests"))


def register(inputs, **options):
    return prepare_learning_plan(inputs["plan"], identity_bundle={"bindings": {
        inputs["binding"]["match_id"]: inputs["binding"]}}, purpose=options.pop("purpose", "f_train"),
        created_at=options.pop("created_at", inputs["now"]), **options)


def validate(plan, inputs, **options):
    return validate_learning_plan(plan, initialization_plan=inputs["plan"], identity_bundle={"bindings": {
        inputs["binding"]["match_id"]: inputs["binding"]}},
        as_of=options.pop("as_of", inputs["now"]+timedelta(seconds=1)), **options)


def changed_inputs(inputs, *, mid=None, kickoff=None, home_id=None):
    result = deepcopy(inputs)
    old = result["binding"]["initialization_manifest"]["market_initialization"]
    fixture, quote = deepcopy(old["fixture"]), deepcopy(old["quote"])
    if mid is not None:
        fixture["match_id"] = quote["match_id"] = mid
    if kickoff is not None:
        fixture["kickoff"] = kickoff.isoformat()
    policy = MarketPriorPolicy(fixture, quote, frozen_at=old["frozen_at"],
        prediction_cutoff=(kickoff-timedelta(minutes=1)) if kickoff else old["prediction_cutoff"])
    binding = result["binding"]
    binding.update(match_id=fixture["match_id"], initialization_manifest=policy.manifest,
                   initialization_sha256=_digest(policy.manifest))
    binding["canonical"]["kickoff"] = fixture["kickoff"]
    if home_id is not None:
        binding["canonical"]["home_id"] = home_id
    binding["binding_sha256"] = _digest({k: v for k, v in binding.items() if k != "binding_sha256"})
    result["plan"]["selected"] = [policy.manifest]
    return result


def test_f_train_returns_no_model_and_whole_fixture_unit(verified_inputs):
    before = deepcopy(verified_inputs)
    plan = register(verified_inputs)
    result = validate(plan, verified_inputs)
    assert verified_inputs == before
    assert result["text_model"] is None and result["risk_model"] is None
    assert result["by_match"][verified_inputs["binding"]["match_id"]]["purpose"] == "f_train"
    assert plan["observation_unit"] == "canonical_source_event_id_not_run_or_document_version"
    assert not plan["independent_effect_confirmed"] and plan["supervisor_only"]
    assert result["binding"]["holdout_audit"]["status"] != "PASS"


@pytest.mark.parametrize("shift", [-1, 0, 4*3600])
def test_actual_query_must_follow_registration_and_precede_cutoff(verified_inputs, shift):
    plan = register(verified_inputs)
    with pytest.raises(ValueError, match="precede|expired"):
        validate(plan, verified_inputs, as_of=verified_inputs["now"]+timedelta(seconds=shift))


def test_postponement_and_provider_alias_do_not_change_physical_cohort(verified_inputs):
    old = register(verified_inputs)
    later = changed_inputs(verified_inputs, mid="new-provider-id",
                           kickoff=verified_inputs["now"]+timedelta(days=1))
    plan = register(later, created_at=later["now"]+timedelta(seconds=1), previous_plans=[old])
    result = validate(plan, later, as_of=later["now"]+timedelta(seconds=2))
    assert result["binding"]["holdout_audit"]["partition_counts"]["train"] == 1
    assert plan["matches"][0]["source_event_id"] == old["matches"][0]["source_event_id"]


def test_same_event_cannot_change_team_facts(verified_inputs):
    old = register(verified_inputs)
    changed = changed_inputs(verified_inputs, home_id="uefa:team:wrong")
    with pytest.raises(ValueError, match="facts"):
        register(changed, previous_plans=[old])


def test_changed_plan_and_mismatched_current_batch_rejected(verified_inputs):
    plan = register(verified_inputs)
    corrupted = deepcopy(plan)
    corrupted["created_at"] = (verified_inputs["now"]-timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="changed"):
        validate(corrupted, verified_inputs)
    with pytest.raises(ValueError, match="differs"):
        validate(plan, changed_inputs(verified_inputs, mid="unregistered-provider-id"))


def test_calibration_requires_full_training_identity_inventory(verified_inputs):
    _, _, plan, fpath, _ = _registered(verified_inputs, "calibration")
    with pytest.raises(ValueError, match="unknown_training_identity_inventory"):
        register(verified_inputs, purpose="calibration", text_model_path=fpath)
    result = validate(plan, verified_inputs)
    assert result["text_model"].model_version == plan["artifacts"]["text_model"]["model_version"]
    assert result["risk_model"] is None


def test_provider_rename_cannot_reclassify_exposed_physical_fixture(verified_inputs):
    old = register(verified_inputs)
    _, _, plan, fpath, _ = _registered(verified_inputs, "calibration")
    prior = json.loads((verified_inputs["root"]/"previous-0.json").read_text(encoding="utf-8"))
    changed = changed_inputs(verified_inputs, mid="other-provider-same-physical-event")
    with pytest.raises(ValueError, match="purpose"):
        register(changed, purpose="calibration", previous_plans=[prior, old], text_model_path=fpath)


def test_archive_override_preserves_exact_artifact_and_rejects_different_bytes(verified_inputs):
    _, _, plan, fpath, _ = _registered(verified_inputs, "calibration")
    archived = fpath.with_name("archived-f.json")
    shutil.copyfile(fpath, archived)
    moved = fpath.with_name("original-location-unavailable.json")
    fpath.rename(moved)
    result = validate(plan, verified_inputs, artifact_paths={"text_model": archived})
    assert result["text_model"].model_version == plan["artifacts"]["text_model"]["model_version"]
    archived.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        validate(plan, verified_inputs, artifact_paths={"text_model": archived})


def test_confirmation_rejects_critic_training_collected_with_another_f(verified_inputs):
    _, _, plan, fpath, gpath = _registered(verified_inputs, "confirmation")
    assert validate(plan, verified_inputs)["risk_model"] is not None
    a = json.loads((verified_inputs["root"]/"previous-0.json").read_text(encoding="utf-8"))
    b = json.loads((verified_inputs["root"]/"previous-1.json").read_text(encoding="utf-8"))
    b["artifacts"]["text_model"]["sha256"] = "0"*64
    b["artifacts"]["text_model"]["model_version"] = "a-different-frozen-predictor"
    b["cohort_id"] = module.VERSION+":"+_digest({k: v for k, v in b.items() if k != "cohort_id"})[:24]
    with pytest.raises(ValueError, match="different frozen"):
        register(verified_inputs, purpose="confirmation", previous_plans=[a, b],
                 text_model_path=fpath, text_risk_model_path=gpath)


def test_confirmation_rejects_retroactively_registered_calibration(verified_inputs):
    _, _, _, fpath, gpath = _registered(verified_inputs, "confirmation")
    a = json.loads((verified_inputs["root"]/"previous-0.json").read_text(encoding="utf-8"))
    b = json.loads((verified_inputs["root"]/"previous-1.json").read_text(encoding="utf-8"))
    b["created_at"] = (datetime.fromisoformat(b["created_at"])+timedelta(minutes=2)).isoformat()
    b["cohort_id"] = module.VERSION+":"+_digest({k: v for k, v in b.items() if k != "cohort_id"})[:24]
    with pytest.raises(ValueError, match="precede every actual"):
        register(verified_inputs, purpose="confirmation", previous_plans=[a, b],
                 text_model_path=fpath, text_risk_model_path=gpath)


def test_synthetic_model_and_missing_confirmation_critic_rejected(verified_inputs):
    at = verified_inputs["now"]-timedelta(days=3)
    ai, _ = _past_inputs(verified_inputs, registered=at, prefix="synthetic-", event_start=9000000)
    f = TextReturnModel().fit(_records(ai, registered=at), as_of=at+timedelta(days=1),
        dataset_version="constructed-mechanism-test", evidence_origin="constructed_mechanism")
    path = verified_inputs["root"]/"constructed.json"
    f.save(path)
    with pytest.raises(ValueError, match="synthetic"):
        register(verified_inputs, purpose="calibration", text_model_path=path)
    with pytest.raises(ValueError, match="paths"):
        register(verified_inputs, purpose="confirmation", text_model_path=path)


def test_legacy_import_reads_actual_archive_and_never_backdates_registration(verified_inputs, monkeypatch):
    folder, _, _ = _collect(verified_inputs, monkeypatch)
    original = {name: (folder/name).read_bytes() for name in ("configuration.json", "run.json")}
    now = datetime.now(timezone.utc)
    exposure = import_training_exposure([folder, folder], created_at=now)
    assert exposure["schema"] == module.EXPOSURE_VERSION and exposure["purpose"] == "f_train"
    assert exposure["unique_matches"] == 1 and len(exposure["runs"]) == 1
    assert exposure["created_at"] == now.isoformat()
    assert not exposure["historical_preregistration_claimed"] and not exposure["selection_probabilities_imputed"]
    assert {name: (folder/name).read_bytes() for name in original} == original
    future = changed_inputs(verified_inputs, kickoff=now+timedelta(days=1))
    plan = register(future, created_at=now+timedelta(seconds=1), previous_plans=[exposure])
    assert plan["prior_inventory"][0]["purpose"] == "f_train"
    with pytest.raises(ValueError, match="inventory"):
        validate(exposure, verified_inputs, as_of=now+timedelta(seconds=1))
    with pytest.raises(ValueError, match="finished"):
        import_training_exposure([folder], created_at=verified_inputs["now"])


def test_legacy_import_cannot_relabel_calibration_as_train(verified_inputs, monkeypatch):
    path, previous, _, _, _ = _registered(verified_inputs, "calibration")
    folder, _, _ = _collect(verified_inputs, monkeypatch, learning_plan_path=path, previous_learning_plan_paths=previous)
    with pytest.raises(ValueError, match="cannot be reclassified"):
        import_training_exposure([folder], created_at=datetime.now(timezone.utc))


def test_legacy_import_rejects_changed_run_configuration(verified_inputs, monkeypatch):
    folder, _, _ = _collect(verified_inputs, monkeypatch)
    config_path = folder/"configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["run_id"] = "different"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        import_training_exposure([folder], created_at=datetime.now(timezone.utc))


def test_prepare_cli_uses_actual_clock_and_exclusive_output(verified_inputs, capsys):
    root = verified_inputs["root"]
    output = root/"registered.json"
    arguments = ["prepare", "--initplan", str(root/"plan.json"), "--identity", str(root/"bindings.json"),
                 "--purpose", "f_train", "--output", str(output)]
    before = datetime.now(timezone.utc)
    assert module.main(arguments) == 0
    after = datetime.now(timezone.utc)
    stored = output.read_bytes()
    plan = json.loads(stored)
    assert before <= datetime.fromisoformat(plan["created_at"]) <= after
    assert json.loads(capsys.readouterr().out)["unique_matches"] == 1
    with pytest.raises(FileExistsError):
        module.main(arguments)
    assert output.read_bytes() == stored
