import io
import json
from dataclasses import asdict
from pathlib import Path
import stat
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from tools.import_kaggle_semantic_encoding import (
    import_result, json_bytes, load_vector_bytes, read_local, safe_zip_members, sha_bytes,
    validate_job_inputs,
)
from football_research.acquisition.postread_semantic_embedding import EncoderContract


def npy_bytes(vector):
    stream = io.BytesIO()
    np.save(stream, vector, allow_pickle=False)
    return stream.getvalue()


def valid_vector():
    return np.ones(1024, dtype=np.float32) / 32


@pytest.mark.parametrize("name", ["../escape.npy", "/absolute.npy", "C:/escape.npy",
    "a/../../escape.npy", "a\\escape.npy", "a:stream", "a/./value", "CON.npy", "a//value"])
def test_zip_paths_are_rejected_before_reading(name):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("completion.json", "{}")
        archive.writestr(name, "payload")
    if "\\" in name:
        # Windows ZipInfo writes normalized forward slashes; create an actual
        # hostile central-directory name of the same byte length.
        stream = io.BytesIO(stream.getvalue().replace(name.replace("\\", "/").encode(), name.encode()))
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError, match="[Uu]nsafe"):
        safe_zip_members(archive)


def test_zip_symlink_rejected_even_when_unused():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo("unused.npy")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../outside")
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError, match="Symlink"):
        safe_zip_members(archive)


def test_zip_case_collision_rejected():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("A.npy", "1")
        archive.writestr("a.npy", "2")
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError, match="colliding"):
        safe_zip_members(archive)


def test_valid_npy_preserves_float32_bytes():
    payload = npy_bytes(valid_vector())
    actual = load_vector_bytes(payload, sha_bytes(payload), label="test")
    np.testing.assert_array_equal(actual, valid_vector())
    assert actual.dtype == np.float32


@pytest.mark.parametrize("vector", [np.zeros(1024, np.float32), np.ones(1024, np.float32),
    np.ones(1024, np.float64) / 32, np.ones(32, np.float32),
    np.full(1024, np.nan, np.float32), np.full(1024, np.inf, np.float32)])
def test_vector_shape_dtype_finiteness_and_norm_rejected(vector):
    payload = npy_bytes(vector)
    with pytest.raises(ValueError):
        load_vector_bytes(payload, sha_bytes(payload), label="bad")


def test_vector_sha_and_trailing_payload_rejected():
    payload = npy_bytes(valid_vector())
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_vector_bytes(payload, "0" * 64, label="bad")
    payload += b"hidden extra bytes"
    with pytest.raises(ValueError, match="trailing"):
        load_vector_bytes(payload, sha_bytes(payload), label="bad")


def test_json_duplicate_keys_and_nonfinite_rejected():
    for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}'):
        with pytest.raises(ValueError):
            json_bytes(data, "fixture")


def test_local_input_cannot_escape_frozen_root(tmp_path):
    (tmp_path / "text.txt").write_text("body", encoding="utf-8")
    assert read_local(tmp_path, "text.txt") == b"body"
    with pytest.raises(ValueError):
        read_local(tmp_path, "../elsewhere")


def json_payload(value):
    return json.dumps(value, sort_keys=True).encode("utf-8")


def full_bundle(tmp_path):
    """Nine tiny texts and eight CPU controls; no encoder/inference is involved."""
    root = tmp_path / "input"
    (root / "texts").mkdir(parents=True)
    (root / "controls").mkdir()
    contract = EncoderContract()
    fp = contract.fingerprint()
    texts, controls = [], []
    vector_bytes = npy_bytes(valid_vector())
    for number in range(9):
        body = f"Archived fixture body number {number}.".encode()
        key = sha_bytes(body)
        path = "texts/" + key + ".txt"
        (root / path).write_bytes(body)
        texts.append(dict(text_sha256=key, text_path=path, contract_fingerprint=fp, token_count=8,
                          task_group="main_historical_corpus"))
        if number < 8:
            control_path = "controls/" + key + ".npy"
            (root / control_path).write_bytes(vector_bytes)
            controls.append(dict(text_sha256=key, vector_path=control_path, vector_file_sha256=sha_bytes(vector_bytes),
                                 contract_fingerprint=fp, execution_device="cpu"))
    raw_texts = b"".join(json_payload(row) + b"\n" for row in texts)
    model = dict(repo=contract.repository, revision=contract.revision,
                 files_sha256={"model.safetensors": contract.model_sha256, "config.json": "0" * 64,
                               "tokenizer.json": "1" * 64, "tokenizer_config.json": "2" * 64})
    job = dict(schema="footballmind-kaggle-semantic-job-v1", job_id="fixture-inference-only", model=model,
               contracts={fp: asdict(contract)}, texts_file="texts.jsonl", texts_sha256=sha_bytes(raw_texts),
               expected_texts=9, controls=controls, control_thresholds=dict(minimum_cosine=.99999, maximum_absolute_difference=.001),
               inference_only=True, train_encoder=False, contains_labels_or_model_targets=False)
    raw_job = json_payload(job)
    (root / "job.json").write_bytes(raw_job)
    (root / "texts.jsonl").write_bytes(raw_texts)
    binding = dict(job_id=job["job_id"], job_sha256=sha_bytes(raw_job))
    members = {"input-contract/job.json": raw_job, "input-contract/texts.jsonl": raw_texts}
    index = []
    execution = dict(device="cuda:0", device_name="fixture-no-real-GPU", model_dtype="float32", output_dtype="float32",
                     attention_implementation="sdpa", tf32_enabled=False, batch_size=1, model_training=False,
                     prompt_injected=False, software={"torch": "fixture", "numpy": np.__version__}, cuda_version="fixture")
    for row in texts:
        key = row["text_sha256"]
        item = dict(text_sha256=key, **binding, contract_fingerprint=fp, vector_path=key + ".npy",
                    metadata_path=key + ".json", vector_file_sha256=sha_bytes(vector_bytes), dimension=1024,
                    dtype="float32", token_count=8)
        index.append(item)
        members[item["vector_path"]] = vector_bytes
        members[item["metadata_path"]] = json_payload(dict(item, effective_contract=asdict(contract), model=model,
            execution_metadata=execution, inference_seconds=.01, truncated_tokens=0, no_prompt=True, texts_sha256=job["texts_sha256"]))
    raw_index = b"".join(json_payload(row) + b"\n" for row in index)
    members["index.jsonl"] = raw_index
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{row["text_sha256"]: valid_vector() for row in texts})
    members["embeddings.npz"] = buffer.getvalue()
    control_rows = []
    for row in controls:
        key = row["text_sha256"]
        members["controls/" + key + ".npy"] = vector_bytes
        control_rows.append(dict(text_sha256=key, reference_vector_path=row["vector_path"],
            reference_file_sha256=row["vector_file_sha256"], gpu_vector_path="controls/" + key + ".npy",
            gpu_vector_file_sha256=sha_bytes(vector_bytes), cosine=1., max_abs_difference=0., passed=True))
    members["controls.json"] = json_payload(dict(**binding, thresholds=job["control_thresholds"], controls=control_rows, all_passed=True))
    members["completion.json"] = json_payload(dict(schema="footballmind-kaggle-semantic-completion-v1", **binding,
        texts_sha256=job["texts_sha256"], model=model, status="completed", expected_texts=9, completed_texts=9,
        controls_passed=True, index_sha256=sha_bytes(raw_index), embeddings_sha256=sha_bytes(members["embeddings.npz"]),
        controls_sha256=sha_bytes(members["controls.json"])))
    return root, members, texts


def write_zip(path, members, wrapper=""):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(wrapper + name, payload)
    return path


def test_complete_import_and_cpu_control_preference(tmp_path):
    root, members, texts = full_bundle(tmp_path)
    original = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = write_zip(tmp_path / "result.zip", members, wrapper="kaggle-result/")
    output = tmp_path / "portable"
    report = import_result(root, result, output)
    assert report["imported_texts"] == 9
    assert report["origin_counts"] == {"frozen_handoff_cpu_control": 8, "validated_gpu_result": 1}
    assert report["cpu_gpu_controls"]["all_passed"] is True
    assert report["cache_modified"] is False
    with np.load(output / "embeddings.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {r["text_sha256"] for r in texts}
        for key in archive.files:
            np.testing.assert_array_equal(archive[key], valid_vector())
    assert original == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        import_result(root, result, output)


def test_mixed_old_and_chunked_gpu_provenance_remains_per_vector(tmp_path):
    root, members, texts = full_bundle(tmp_path)
    added = dict(attention_query_chunk_size=256, attention_execution_version="query-chunked-sdpa-v2",
                 attention_full_key_value=True, attention_global_causal_positions=True, text_chunking=False)
    measurements = dict(attention_calls=28, tiled_attention_calls=28, maximum_query_tokens=15446,
                        peak_cuda_allocated_bytes=3_500_000_000, peak_cuda_reserved_bytes=4_000_000_000)
    # One resumed entry remains v1, while a newly encoded entry has the v2
    # implementation. A v2 completion record must not relabel the resumed entry.
    new_key = texts[-1]["text_sha256"]
    new_meta = json.loads(members[new_key + ".json"])
    new_meta["execution_metadata"].update(added)
    new_meta.update(measurements)
    members[new_key + ".json"] = json_payload(new_meta)
    completion = json.loads(members["completion.json"])
    completion["execution_metadata"] = new_meta["execution_metadata"]
    members["completion.json"] = json_payload(completion)
    output = tmp_path / "portable"
    import_result(root, write_zip(tmp_path / "mixed.zip", members), output)
    persisted = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    actual = {r["text_sha256"]: r for r in persisted["vectors"]}
    old_key = texts[0]["text_sha256"]
    assert actual[old_key]["gpu_execution"]["attention_implementation"] == "sdpa"
    assert all(actual[old_key]["gpu_execution"][name] is None for name in added)
    assert all(actual[old_key]["gpu_measurements"][name] is None for name in measurements)
    assert all(actual[new_key]["gpu_execution"][name] == value for name, value in added.items())
    assert actual[new_key]["gpu_execution"]["text_chunking"] is False
    assert all(actual[new_key]["gpu_measurements"][name] == value for name, value in measurements.items())
    assert actual[new_key]["gpu_execution"]["attention_implementation"] == "sdpa"


def test_existing_cpu_cache_is_preserved_and_provenance_is_explicit(tmp_path):
    root, members, texts = full_bundle(tmp_path)
    row = texts[-1]
    key, fp = row["text_sha256"], row["contract_fingerprint"]
    cache = tmp_path / "cache" / fp
    cache.mkdir(parents=True)
    cpu = valid_vector().copy()
    cpu[0] += 1e-6
    cpu /= np.linalg.norm(cpu)
    raw = npy_bytes(cpu)
    (cache / (key + ".npy")).write_bytes(raw)
    (cache / (key + ".json")).write_bytes(json_payload(dict(text_sha256=key, contract_fingerprint=fp,
        vector_file_sha256=sha_bytes(raw), token_count=8, truncated_tokens=0, inference_seconds=.2)))
    report = import_result(root, write_zip(tmp_path / "gpu.zip", members), tmp_path / "portable", cpu_cache_roots=[cache.parent])
    assert report["origin_counts"]["existing_local_cpu_cache"] == 1
    selected = next(r for r in report["vectors"] if r["text_sha256"] == key)
    assert selected["selected_execution"]["recorded_device"] is None
    assert selected["bitwise_equal_to_gpu"] is False
    with np.load(tmp_path / "portable/embeddings.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive[key], cpu)
    assert (cache / (key + ".npy")).read_bytes() == raw


def test_cpu_overlap_beyond_eight_controls_must_also_agree(tmp_path):
    root, members, texts = full_bundle(tmp_path)
    row = texts[-1]
    key, fp = row["text_sha256"], row["contract_fingerprint"]
    cache = tmp_path / "cache" / fp
    cache.mkdir(parents=True)
    raw = npy_bytes(-valid_vector())
    (cache / (key + ".npy")).write_bytes(raw)
    (cache / (key + ".json")).write_bytes(json_payload(dict(text_sha256=key, contract_fingerprint=fp,
        vector_file_sha256=sha_bytes(raw), token_count=8, truncated_tokens=0)))
    with pytest.raises(ValueError, match="overlap numerical"):
        import_result(root, write_zip(tmp_path / "gpu.zip", members), tmp_path / "not-created", cpu_cache_roots=[cache.parent])
    assert not (tmp_path / "not-created").exists()
    assert (cache / (key + ".npy")).read_bytes() == raw


@pytest.mark.parametrize("damage", ["job_sha", "vector_hash", "missing_vector", "missing_index", "bad_model",
    "failed_controls", "fake_controls", "prompt", "half_precision", "npz_mismatch"])
def test_damaged_results_rejected_before_destination_creation(tmp_path, damage):
    root, members, texts = full_bundle(tmp_path)
    completion = json.loads(members["completion.json"])
    key = texts[-1]["text_sha256"]
    if damage == "job_sha":
        completion["job_sha256"] = "0" * 64
    elif damage == "vector_hash":
        members[key + ".npy"] += b"corrupt"
    elif damage == "missing_vector":
        del members[key + ".npy"]
    elif damage == "missing_index":
        members["index.jsonl"] = b"\n".join(members["index.jsonl"].split(b"\n")[:-2]) + b"\n"
        completion["index_sha256"] = sha_bytes(members["index.jsonl"])
    elif damage == "bad_model":
        completion["model"]["revision"] = "different"
    elif damage == "failed_controls":
        completion["controls_passed"] = False
    elif damage == "fake_controls":
        controls = json.loads(members["controls.json"])
        controls["controls"][0]["cosine"] = .999
        members["controls.json"] = json_payload(controls)
        completion["controls_sha256"] = sha_bytes(members["controls.json"])
    elif damage in ("prompt", "half_precision"):
        meta = json.loads(members[key + ".json"])
        if damage == "prompt":
            meta["no_prompt"] = False
        else:
            meta["execution_metadata"]["model_dtype"] = "float16"
        members[key + ".json"] = json_payload(meta)
    elif damage == "npz_mismatch":
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{r["text_sha256"]: -valid_vector() for r in texts})
        members["embeddings.npz"] = buffer.getvalue()
        completion["embeddings_sha256"] = sha_bytes(members["embeddings.npz"])
    members["completion.json"] = json_payload(completion)
    with pytest.raises(ValueError):
        import_result(root, write_zip(tmp_path / "bad.zip", members), tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_local_frozen_text_and_control_are_hash_bound(tmp_path):
    root, _, texts = full_bundle(tmp_path)
    path = root / texts[0]["text_path"]
    path.write_text("different text", encoding="utf-8")
    with pytest.raises(ValueError, match="text SHA"):
        validate_job_inputs(root)


def test_actual_runner_contract_roundtrip_with_explicit_fake_inference(tmp_path, monkeypatch):
    from tools import run_kaggle_semantic_encoding as runner
    root, _, texts = full_bundle(tmp_path)
    calls = []
    class FakeEncoder:
        execution = dict(device="cuda:0", device_name="TEST_FIXTURE_NO_GPU_EXECUTED", model_dtype="float32",
                         output_dtype="float32", attention_implementation="sdpa", tf32_enabled=False, batch_size=1,
                         model_training=False, prompt_injected=False, software={"fixture": "not_real_inference"}, cuda_version="fixture")
        def __init__(self, *args):
            pass
        def encode(self, text, contract, expected_tokens):
            calls.append(text)
            return valid_vector(), dict(token_count=expected_tokens, inference_seconds=0., characters=len(text), truncated_tokens=0)
    monkeypatch.setattr(runner, "Encoder", FakeEncoder)
    monkeypatch.setattr(runner, "verify_model_files", lambda *args: {})
    returned = tmp_path / "runner-output"
    completion = runner.execute(SimpleNamespace(input_dir=root, output_dir=returned, model_dir=tmp_path / "unused-model",
                                               device="cuda", batch_size=1, verify_input_only=False))
    assert completion["status"] == "completed" and len(calls) == len(texts)
    members = {p.relative_to(returned).as_posix(): p.read_bytes() for p in returned.rglob("*") if p.is_file()}
    imported = import_result(root, write_zip(tmp_path / "actual-runner-fixture.zip", members), tmp_path / "portable")
    assert imported["imported_texts"] == 9
    assert imported["cpu_gpu_controls"]["all_passed"] is True
