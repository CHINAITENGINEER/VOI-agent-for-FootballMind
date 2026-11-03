"""Diagnose real file corruption and missing encoders without loading weights."""
import hashlib
import json

import pytest

from footballmind.config import load_config, ENCODER_REVISION
from footballmind import doctor


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    config = load_config(workspace_root=tmp_path, environ={})
    release = config.release_dir
    f = release / 'models/f/model.pt'
    g = release / 'models/g/g-models.joblib'
    f.parent.mkdir(parents=True)
    g.parent.mkdir(parents=True)
    f.write_bytes(b'f-test-weights')
    g.write_bytes(b'g-test-weights')
    fsha, gsha = doctor.file_sha256(f), doctor.file_sha256(g)
    monkeypatch.setattr(doctor, 'F_SHA256', fsha)
    monkeypatch.setattr(doctor, 'G_SHA256', gsha)
    (release / 'bundle-config.json').write_text(json.dumps({
        'bundle_id': config.release_id, 'f_model_sha256': fsha,
        'g_model_sha256': gsha, 'dependency_versions': {'torch': '1.0'}}))
    entries = {str(p.relative_to(release).as_posix()): doctor.file_sha256(p)
               for p in (f, g, release / 'bundle-config.json')}
    (release / 'manifest.json').write_text(json.dumps({'files_sha256': entries}))
    monkeypatch.setattr(doctor.importlib.metadata, 'version', lambda _: '1.0')
    return config


def test_offline_bundle_ready_with_no_encoder_and_no_side_effects(workspace):
    result = doctor.diagnose(workspace)
    assert result['offline_ready']
    assert not result['online_artifacts_ready']
    assert result['network_requests'] == 0 and not result['inference_executed']
    assert result['sources']['status'] == 'not_probed'
    assert not workspace.cache_dir.exists() and not workspace.runs_dir.exists()


def test_corrupt_model_rejected_even_after_manifest_is_rewritten(workspace):
    model = workspace.release_dir / 'models/f/model.pt'
    model.write_bytes(b'changed')
    manifest_path = workspace.release_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files_sha256']['models/f/model.pt'] = doctor.file_sha256(model)
    manifest_path.write_text(json.dumps(manifest))
    result = doctor.diagnose(workspace)
    assert not result['offline_ready']
    assert any(r['reason'] == 'frozen_model_sha256_mismatch' for r in result['release']['failures'])


def test_manifest_cannot_reference_file_outside_bundle(workspace):
    p = workspace.release_dir / 'manifest.json'
    p.write_text(json.dumps({'files_sha256': {'../../outside': 'a' * 64}}))
    result = doctor.diagnose(workspace)
    assert not result['offline_ready']
    assert result['release']['failures'][0]['reason'] == 'ValueError'


def test_deep_encoder_hash_detects_corruption(workspace):
    encoder = workspace.encoder_dir
    assert encoder.name == ENCODER_REVISION
    encoder.mkdir(parents=True)
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        (encoder / name).write_text('{}')
    (encoder / 'model.safetensors').write_bytes(b'test-encoder')
    provenance = workspace.release_dir / 'provenance/encoder-provenance.json'
    provenance.parent.mkdir()
    provenance.write_text(json.dumps({'short_contract': {
        'revision': ENCODER_REVISION, 'model_sha256': hashlib.sha256(b'original-encoder').hexdigest()}}))
    assert doctor.diagnose(workspace)['online_artifacts_ready']
    deep = doctor.diagnose(workspace, deep=True)
    assert not deep['online_artifacts_ready']
    assert deep['encoder']['checkpoint_digest_checked']


def test_missing_dependency_is_not_reported_ready(workspace, monkeypatch):
    def missing(_):
        raise doctor.importlib.metadata.PackageNotFoundError('torch')
    monkeypatch.setattr(doctor.importlib.metadata, 'version', missing)
    result = doctor.diagnose(workspace)
    assert not result['offline_ready'] and not result['online_artifacts_ready']
