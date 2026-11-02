"""Validate an external encoding result and export a new portable archive.

The importer never writes to SemanticCache. ZIP members are inspected and read
in memory, not extracted into the workspace. No model inference is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_research.acquisition.postread_semantic_embedding import EncoderContract


SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
MAX_MEMBERS = 100_000


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value, label):
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {label} SHA-256")
    return value


def relative_name(name: str) -> str:
    """Reject platform-dependent paths, including Windows drive/ADS names."""
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Unsafe archive member path")
    if name.startswith("/") or ":" in name:
        raise ValueError("Unsafe archive member path")
    parts = name.rstrip("/").split("/")
    if any(part in ("", ".", "..") or part.endswith((".", " ")) for part in parts):
        raise ValueError("Unsafe archive member path")
    if any(part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))} for part in parts):
        raise ValueError("Unsafe platform-reserved archive member path")
    normalized = str(PurePosixPath(*parts))
    if normalized != name.rstrip("/"):
        raise ValueError("Noncanonical archive member path")
    return normalized


def safe_zip_members(archive: zipfile.ZipFile):
    """Validate every member before reading even the completion manifest."""
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ValueError("Archive member limit exceeded")
    result, seen, total = {}, set(), 0
    for info in infos:
        # ZipInfo normalizes backslashes on Windows. Inspect the original
        # central-directory spelling before that platform normalization.
        relative_name(info.orig_filename)
        name = relative_name(info.filename)
        if name.casefold() in seen:
            raise ValueError("Duplicate or case-colliding archive member")
        seen.add(name.casefold())
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if stat.S_ISLNK(mode) or kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("Symlink or special archive member rejected")
        if kind == stat.S_IFDIR and not info.is_dir():
            raise ValueError("Invalid archive directory attributes")
        if info.flag_bits & 1:
            raise ValueError("Encrypted archive member rejected")
        if info.file_size > MAX_MEMBER_BYTES or info.file_size < 0:
            raise ValueError("Archive member size limit exceeded")
        total += info.file_size
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive uncompressed size limit exceeded")
        if not info.is_dir():
            result[name] = info
    return result


def json_bytes(payload: bytes, label: str):
    def distinct(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {label}")
            result[key] = value
        return result
    def finite_float(value):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"Nonfinite JSON in {label}")
        return number
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=distinct, parse_float=finite_float,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON in {label}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {label}") from exc


def jsonl_bytes(payload: bytes, label: str):
    # Literal LF only: a Unicode paragraph separator can legitimately occur in
    # the full body and must not become a second physical JSONL record.
    return [json_bytes(line, f"{label}:{i}") for i, line in enumerate(payload.split(b"\n"), 1) if line.strip()]


def load_vector_bytes(payload: bytes, expected_sha: str, *, label: str):
    require_sha(expected_sha, label)
    if sha_bytes(payload) != expected_sha:
        raise ValueError(f"Vector file SHA mismatch: {label}")
    if len(payload) > 65536:
        raise ValueError(f"Unexpected vector file size: {label}")
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Unsupported NPY format")
        if shape != (1024,) or dtype != np.dtype("float32"):
            raise ValueError(f"Vector must be float32[1024]: {label}")
        stream.seek(0)
        vector = np.load(stream, allow_pickle=False)
    except (ValueError, TypeError, OSError, EOFError) as exc:
        raise ValueError(f"Invalid NPY vector: {label}") from exc
    if not isinstance(vector, np.ndarray) or stream.tell() != len(payload):
        raise ValueError(f"Unexpected vector container/trailing data: {label}")
    if vector.dtype != np.dtype("float32") or vector.shape != (1024,):
        raise ValueError(f"Vector must be float32[1024]: {label}")
    if not np.isfinite(vector).all():
        raise ValueError(f"Nonfinite vector: {label}")
    if abs(float(np.linalg.norm(vector.astype(np.float64))) - 1.0) > 1e-5:
        raise ValueError(f"Vector is not unit normalized: {label}")
    return vector


def read_local(root: Path, name: str) -> bytes:
    name = relative_name(name)
    root = root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(name).parts)
    # A frozen input may not follow an injected symlink outside its own bundle.
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or any(p.is_symlink() for p in [candidate, *candidate.parents] if p != root):
        raise ValueError("Local input path escapes frozen handoff")
    return resolved.read_bytes()


def validate_job_inputs(input_dir: Path):
    """Bind exact body bytes, CPU references and both frozen token contracts."""
    raw = read_local(input_dir, "job.json")
    job = json_bytes(raw, "job.json")
    if job.get("schema") != "footballmind-kaggle-semantic-job-v1":
        raise ValueError("Unsupported encoding job schema")
    base = EncoderContract()
    long = replace(base, max_input_tokens=32768, implementation="postread-qwen-full-history-long-v1")
    allowed = {c.fingerprint(): asdict(c) for c in (base, long)}
    contracts = job.get("contracts", {})
    if not contracts or any(fp not in allowed or value != allowed[fp] for fp, value in contracts.items()):
        raise ValueError("Job changes the frozen encoder contract")
    model = job.get("model", {})
    if model.get("repo") != base.repository or model.get("revision") != base.revision:
        raise ValueError("Model repository/revision mismatch")
    files = model.get("files_sha256", {})
    if files.get("model.safetensors") != base.model_sha256:
        raise ValueError("Frozen model weights SHA mismatch")
    for name, digest in files.items():
        relative_name(name)
        require_sha(digest, name)
    if (job.get("inference_only") is not True or job.get("train_encoder") is not False
            or job.get("contains_labels_or_model_targets") is not False):
        raise ValueError("Not a frozen label-free inference job")
    if not isinstance(job.get("job_id"), str) or not job["job_id"]:
        raise ValueError("Missing job ID")
    if job.get("texts_file") != "texts.jsonl":
        raise ValueError("Unexpected texts index path")
    raw_texts = read_local(input_dir, "texts.jsonl")
    if sha_bytes(raw_texts) != require_sha(job.get("texts_sha256"), "texts index"):
        raise ValueError("Texts index SHA mismatch")
    texts = {}
    for row in jsonl_bytes(raw_texts, "texts.jsonl"):
        key = require_sha(row.get("text_sha256"), "text")
        if key in texts:
            raise ValueError("Duplicate text SHA in job")
        fp = row.get("contract_fingerprint")
        if fp not in contracts:
            raise ValueError("Text references unknown encoder contract")
        count = row.get("token_count")
        if type(count) is not int or not 0 < count <= contracts[fp]["max_input_tokens"]:
            raise ValueError("Text token count is invalid; truncation forbidden")
        body = read_local(input_dir, row["text_path"])
        if sha_bytes(body) != key or not body.decode("utf-8").strip():
            raise ValueError("Source text SHA/nonempty mismatch")
        texts[key] = row
    if type(job.get("expected_texts")) is not int or len(texts) != job["expected_texts"] or not texts:
        raise ValueError("Job input count mismatch")
    controls = {}
    for row in job.get("controls", []):
        key = require_sha(row.get("text_sha256"), "control text")
        if key in controls or key not in texts:
            raise ValueError("Control text missing or duplicated")
        if row.get("contract_fingerprint") != texts[key]["contract_fingerprint"]:
            raise ValueError("CPU control contract mismatch")
        payload = read_local(input_dir, row["vector_path"])
        vector = load_vector_bytes(payload, row["vector_file_sha256"], label="CPU control " + key)
        controls[key] = dict(record=row, vector=vector, payload=payload)
    if len(controls) != 8:
        raise ValueError("Exactly eight independent frozen CPU controls required")
    thresholds = job.get("control_thresholds")
    if thresholds != {"minimum_cosine": 0.99999, "maximum_absolute_difference": 0.001}:
        raise ValueError("Frozen control thresholds mismatch")
    return dict(job=job, job_bytes=raw, job_sha256=sha_bytes(raw), texts=texts,
                texts_bytes=raw_texts, controls=controls)


class OutputArchive:
    def __init__(self, path: Path):
        self.archive = zipfile.ZipFile(path)
        try:
            members = safe_zip_members(self.archive)
            manifests = [n for n in members if PurePosixPath(n).name == "completion.json"]
            if len(manifests) != 1:
                raise ValueError("Expected exactly one completion manifest")
            parent = PurePosixPath(manifests[0]).parent
            self.prefix = "" if str(parent) == "." else str(parent) + "/"
            if any(not name.startswith(self.prefix) for name in members):
                raise ValueError("Files outside the single result wrapper")
            self.members = {name[len(self.prefix):]: info for name, info in members.items()}
        except Exception:
            self.archive.close()
            raise

    def read(self, name):
        name = relative_name(name)
        if name not in self.members:
            raise ValueError("Missing result file: " + name)
        return self.archive.read(self.members[name])

    def close(self):
        self.archive.close()


def verify_binding(record, bundle, *, label):
    if record.get("job_id") != bundle["job"]["job_id"] or record.get("job_sha256") != bundle["job_sha256"]:
        raise ValueError("Job SHA/ID binding mismatch: " + label)


def compare_control(reference, candidate, thresholds):
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    maximum = float(np.max(np.abs(a - b)))
    return dict(cosine=cosine, max_abs_difference=maximum,
                passed=cosine >= thresholds["minimum_cosine"] and maximum <= thresholds["maximum_absolute_difference"])


def validate_gpu_archive(archive, bundle):
    completion = json_bytes(archive.read("completion.json"), "completion")
    verify_binding(completion, bundle, label="completion")
    job, texts = bundle["job"], bundle["texts"]
    if completion.get("schema") != "footballmind-kaggle-semantic-completion-v1" or completion.get("status") != "completed":
        raise ValueError("GPU encoding did not complete successfully")
    if completion.get("model") != job["model"] or completion.get("texts_sha256") != job["texts_sha256"]:
        raise ValueError("GPU model/texts binding mismatch")
    if completion.get("expected_texts") != len(texts) or completion.get("completed_texts") != len(texts):
        raise ValueError("GPU input set coverage incomplete")
    if completion.get("controls_passed") is not True:
        raise ValueError("GPU CPU-reference controls did not pass")
    if archive.read("input-contract/job.json") != bundle["job_bytes"] or archive.read("input-contract/texts.jsonl") != bundle["texts_bytes"]:
        raise ValueError("Copied input contract differs from frozen local job")
    index_bytes = archive.read("index.jsonl")
    if sha_bytes(index_bytes) != completion.get("index_sha256"):
        raise ValueError("GPU index SHA mismatch")
    indexed, vectors, metadata = {}, {}, {}
    for row in jsonl_bytes(index_bytes, "index.jsonl"):
        key = require_sha(row.get("text_sha256"), "GPU text")
        if key in indexed or key not in texts:
            raise ValueError("Duplicate or unexpected GPU text")
        verify_binding(row, bundle, label="vector index")
        if (row.get("contract_fingerprint") != texts[key]["contract_fingerprint"]
                or row.get("dimension") != 1024 or row.get("dtype") != "float32"
                or row.get("token_count") != texts[key]["token_count"]):
            raise ValueError("GPU vector contract/token mismatch")
        if row.get("vector_path") != key + ".npy" or row.get("metadata_path") != key + ".json":
            raise ValueError("GPU vector path is not content addressed")
        payload = archive.read(row["vector_path"])
        vector = load_vector_bytes(payload, row["vector_file_sha256"], label=key)
        meta = json_bytes(archive.read(row["metadata_path"]), key + ".json")
        verify_binding(meta, bundle, label="vector metadata")
        if (meta.get("text_sha256") != key or meta.get("contract_fingerprint") != row["contract_fingerprint"]
                or meta.get("vector_file_sha256") != row["vector_file_sha256"]
                or meta.get("texts_sha256") != job["texts_sha256"]
                or meta.get("effective_contract") != job["contracts"][row["contract_fingerprint"]]
                or meta.get("model") != job["model"]):
            raise ValueError("GPU per-vector provenance binding mismatch")
        if meta.get("token_count") != row["token_count"] or meta.get("truncated_tokens") != 0 or meta.get("no_prompt") is not True:
            raise ValueError("GPU text was truncated/changed or token count differs")
        execution = meta.get("execution_metadata", {})
        device = execution.get("device")
        if not isinstance(device, str) or re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
            raise ValueError("GPU execution device is missing")
        if (execution.get("model_dtype") != "float32" or execution.get("output_dtype") != "float32"
                or execution.get("attention_implementation") != "sdpa" or execution.get("tf32_enabled") is not False
                or execution.get("batch_size") != 1 or execution.get("model_training") is not False
                or execution.get("prompt_injected") is not False):
            raise ValueError("GPU execution settings differ from the declared contract")
        if not isinstance(execution.get("software"), dict) or type(meta.get("inference_seconds")) not in (float, int) or meta["inference_seconds"] < 0:
            raise ValueError("GPU execution provenance is incomplete")
        indexed[key], vectors[key], metadata[key] = row, vector, meta
    if set(indexed) != set(texts):
        raise ValueError("Missing GPU vectors from complete input set")
    raw_npz = archive.read("embeddings.npz")
    if sha_bytes(raw_npz) != completion.get("embeddings_sha256"):
        raise ValueError("GPU embeddings archive SHA mismatch")
    # Inspect the nested NPZ too; np.load alone would accept duplicate entries.
    with zipfile.ZipFile(io.BytesIO(raw_npz)) as nested:
        members = safe_zip_members(nested)
        if set(members) != {key + ".npy" for key in texts}:
            raise ValueError("NPZ keys do not exactly cover the input set")
        for key, vector in vectors.items():
            raw = nested.read(members[key + ".npy"])
            inside = load_vector_bytes(raw, sha_bytes(raw), label="NPZ " + key)
            if not np.array_equal(inside, vector):
                raise ValueError("NPZ and individual GPU vectors differ")
    raw_controls = archive.read("controls.json")
    if sha_bytes(raw_controls) != completion.get("controls_sha256"):
        raise ValueError("GPU controls SHA mismatch")
    report = json_bytes(raw_controls, "controls")
    verify_binding(report, bundle, label="controls")
    if report.get("all_passed") is not True or report.get("thresholds") != job["control_thresholds"]:
        raise ValueError("Control result/threshold mismatch")
    control_results = []
    seen = set()
    for row in report.get("controls", []):
        key = row.get("text_sha256")
        if key not in bundle["controls"] or key in seen:
            raise ValueError("Unexpected/duplicate control result")
        seen.add(key)
        reference = bundle["controls"][key]
        if row.get("reference_file_sha256") != reference["record"]["vector_file_sha256"] or row.get("reference_vector_path") != reference["record"]["vector_path"]:
            raise ValueError("Frozen CPU reference binding mismatch")
        if row.get("gpu_vector_path") != "controls/" + key + ".npy":
            raise ValueError("Unexpected GPU control vector path")
        gpu = load_vector_bytes(archive.read(row["gpu_vector_path"]), row["gpu_vector_file_sha256"], label="GPU control " + key)
        measured = compare_control(reference["vector"], gpu, job["control_thresholds"])
        actual_output = compare_control(reference["vector"], vectors[key], job["control_thresholds"])
        if row.get("passed") is not True or not measured["passed"] or not actual_output["passed"]:
            raise ValueError("Independent CPU/GPU control comparison failed")
        for field in ("cosine", "max_abs_difference"):
            if not isinstance(row.get(field), (float, int)) or not np.isfinite(row[field]) or abs(row[field] - measured[field]) > 1e-7:
                raise ValueError("Reported control metric differs from independent calculation")
        control_results.append(dict(text_sha256=key, reference_sha256=row["reference_file_sha256"],
                                    independent_gpu_reference=measured, actual_output_reference=actual_output))
    if seen != set(bundle["controls"]):
        raise ValueError("Missing CPU/GPU controls")
    return dict(completion=completion, indexed=indexed, vectors=vectors, metadata=metadata,
                control_results=control_results)


def existing_cpu_vector(roots, row):
    """Read a complete immutable entry; a concurrent partial entry is not a hit."""
    key, fp = row["text_sha256"], row["contract_fingerprint"]
    partial = []
    for supplied in roots:
        root = Path(supplied)
        directory = root if root.name == fp else root / fp
        npy, meta_path = directory / (key + ".npy"), directory / (key + ".json")
        if not npy.exists() and not meta_path.exists():
            continue
        if not npy.exists() or not meta_path.exists():
            partial.append(str(directory / key))
            continue
        meta = json_bytes(read_local(directory, key + ".json"), "local CPU cache")
        if (meta.get("text_sha256") != key or meta.get("contract_fingerprint") != fp
                or meta.get("token_count") != row["token_count"] or meta.get("truncated_tokens") != 0):
            raise ValueError("Existing CPU cache contract/text/token mismatch")
        payload = read_local(directory, key + ".npy")
        vector = load_vector_bytes(payload, meta["vector_file_sha256"], label="CPU cache " + key)
        return dict(vector=vector, payload_sha256=sha_bytes(payload),
                    provenance=dict(origin="existing_local_cpu_cache", path=str(npy.resolve()),
                                    metadata_path=str(meta_path.resolve()),
                                    recorded_device=meta.get("device"),
                                    device_basis="caller_supplied_CPU_cache_lineage; absent device field stays unknown",
                                    token_count=meta["token_count"], inference_seconds=meta.get("inference_seconds")),
                    ignored_partial_entries=partial)
    return dict(ignored_partial_entries=partial)


def import_result(input_dir, result_zip, output_dir, *, cpu_cache_roots=()):
    """All validation precedes creation of the never-overwritten output folder."""
    input_dir, result_zip, output_dir = Path(input_dir), Path(result_zip), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Import output must be a new directory")
    bundle = validate_job_inputs(input_dir)
    archive = OutputArchive(result_zip)
    try:
        result = validate_gpu_archive(archive, bundle)
    finally:
        archive.close()
    chosen, index, provenance = {}, [], []
    for key, row in sorted(bundle["texts"].items()):
        meta = result["metadata"][key]
        execution = meta["execution_metadata"]
        if key in bundle["controls"]:
            control = bundle["controls"][key]
            vector, digest = control["vector"], control["record"]["vector_file_sha256"]
            origin = dict(origin="frozen_handoff_cpu_control", path=str((input_dir / control["record"]["vector_path"]).resolve()),
                          recorded_device=control["record"].get("execution_device"),
                          device_basis="frozen_job_control_record", token_count=row["token_count"])
            partial = []
        else:
            cpu = existing_cpu_vector(cpu_cache_roots, row)
            partial = cpu.get("ignored_partial_entries", [])
            if "vector" in cpu:
                vector, digest, origin = cpu["vector"], cpu["payload_sha256"], cpu["provenance"]
            else:
                vector = result["vectors"][key]
                digest = result["indexed"][key]["vector_file_sha256"]
                origin = dict(origin="validated_gpu_result", path=str(result_zip.resolve()) + "!/" + result["indexed"][key]["vector_path"],
                              recorded_device=execution["device"], device_basis="runner_recorded_execution",
                              software=execution["software"], inference_seconds=meta["inference_seconds"], token_count=meta["token_count"])
        chosen[key] = vector
        cpu_compared = origin["origin"] != "validated_gpu_result"
        comparison = compare_control(vector, result["vectors"][key], bundle["job"]["control_thresholds"]) if cpu_compared else None
        if comparison is not None and not comparison["passed"]:
            raise ValueError("Existing CPU/GPU overlap numerical comparison failed: " + key)
        index.append(dict(text_sha256=key, npz_key=key, contract_fingerprint=row["contract_fingerprint"],
                          token_count=row["token_count"], task_group=row.get("task_group"), dimension=1024, dtype="float32",
                          selected_origin=origin["origin"], source_vector_file_sha256=digest,
                          value_sha256=sha_bytes(vector.tobytes(order="C")), job_sha256=bundle["job_sha256"]))
        provenance.append(dict(text_sha256=key, selected_execution=origin, ignored_partial_cpu_entries=partial,
                               gpu_execution={name: execution.get(name) for name in (
                                   "device", "device_name", "software", "model_dtype", "output_dtype",
                                   "attention_implementation", "tf32_enabled", "batch_size", "model_training",
                                   "prompt_injected", "cuda_version", "attention_query_chunk_size",
                                   "attention_execution_version", "attention_full_key_value",
                                   "attention_global_causal_positions", "text_chunking")},
                               gpu_measurements={name: meta.get(name) for name in (
                                   "inference_seconds", "attention_calls", "tiled_attention_calls",
                                   "maximum_query_tokens", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")},
                               cpu_gpu_directly_compared=cpu_compared, selected_vs_gpu=comparison,
                               bitwise_equal_to_gpu=bool(np.array_equal(vector, result["vectors"][key])) if cpu_compared else None))
    # Exclusive mkdir is the only point where a new destination is created.
    # Validation errors above cannot leave an apparently usable partial export.
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "embeddings.npz").open("xb") as stream:
        np.savez_compressed(stream, **chosen)
    with (output_dir / "index.jsonl").open("x", encoding="utf-8") as stream:
        for row in index:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = dict(schema="footballmind-kaggle-semantic-import-v1", status="validated_complete",
                  imported_at=datetime.now(timezone.utc).isoformat(), job_id=bundle["job"]["job_id"], job_sha256=bundle["job_sha256"],
                  texts_sha256=bundle["job"]["texts_sha256"], model=bundle["job"]["model"], contracts=bundle["job"]["contracts"],
                  input_dir=str(input_dir.resolve()), result_zip=str(result_zip.resolve()), result_zip_sha256=sha_bytes(result_zip.read_bytes()),
                  expected_texts=len(bundle["texts"]), imported_texts=len(chosen),
                  origin_counts={origin: sum(r["selected_origin"] == origin for r in index) for origin in sorted({r["selected_origin"] for r in index})},
                  cpu_gpu_controls=dict(all_passed=True, independently_recomputed=True, results=result["control_results"]),
                  vectors=provenance, network_requests=0, model_inference=False, cache_modified=False,
                  cpu_gpu_overlap_count=sum(r["cpu_gpu_directly_compared"] for r in provenance),
                  inputs_without_direct_cpu_comparison=sum(not r["cpu_gpu_directly_compared"] for r in provenance),
                  bitwise_cpu_gpu_equality_assumed=False, numerical_equivalence_is_limited_to_compared_inputs=True)
    with (output_dir / "provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    manifest = {p.name: sha_bytes(p.read_bytes()) for p in sorted(output_dir.iterdir()) if p.is_file()}
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(schema="footballmind-portable-embeddings-manifest-v1", files_sha256=manifest), stream, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Frozen local handoff input directory")
    parser.add_argument("--result-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--cpu-cache", type=Path, action="append", default=[], help="Read-only CPU cache root; may repeat")
    args = parser.parse_args()
    result = import_result(args.input_dir, args.result_zip, args.output_dir, cpu_cache_roots=args.cpu_cache)
    print(json.dumps(dict(status=result["status"], vectors=result["imported_texts"], origins=result["origin_counts"], output=str(args.output_dir)), ensure_ascii=False))


if __name__ == "__main__":
    main()
