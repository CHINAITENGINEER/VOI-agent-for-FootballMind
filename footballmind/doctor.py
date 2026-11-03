"""Offline artifact and dependency diagnosis; never import inference engines."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

from .config import ENCODER_REVISION, F_SHA256, G_SHA256


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _dependencies(required):
    checks = []
    for name, expected in required.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        checks.append({'package': name, 'installed': actual, 'expected': expected,
                       'ok': actual is not None and (expected is None or actual == expected)})
    return checks


def diagnose(config, *, deep=False):
    """Check the configured bundle in place, with explicit levels of assurance.

    Frozen release files are small enough to hash on every check. The large
    encoder checkpoint is hashed only with ``deep``. Presence of artifacts is
    never described as an executed body-to-prediction inference.
    """
    release = config.release_dir.resolve()
    failures = []
    manifest = bundle = {}
    for name in ('manifest.json', 'bundle-config.json'):
        try:
            value = json.loads((release / name).read_text(encoding='utf-8'))
            if not isinstance(value, dict):
                raise ValueError('Expected a JSON object')
            if name == 'manifest.json':
                manifest = value
            else:
                bundle = value
        except (OSError, ValueError) as exc:
            failures.append({'artifact': name, 'reason': type(exc).__name__})
    expected_files = manifest.get('files_sha256', {})
    if not isinstance(expected_files, dict) or not expected_files:
        failures.append({'artifact': 'manifest.json', 'reason': 'missing_file_hashes'})
        expected_files = {}
    for relative, expected in expected_files.items():
        try:
            if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
                raise ValueError('Invalid manifest entry')
            path = (release / relative).resolve()
            if not path.is_relative_to(release):
                raise ValueError('Manifest path escapes release')
            if file_sha256(path) != expected:
                failures.append({'artifact': relative, 'reason': 'sha256_mismatch'})
        except (OSError, ValueError) as exc:
            failures.append({'artifact': str(relative), 'reason': type(exc).__name__})
    if (bundle.get('bundle_id') != config.release_id
            or bundle.get('f_model_sha256') != F_SHA256
            or bundle.get('g_model_sha256') != G_SHA256):
        failures.append({'artifact': 'bundle-config.json', 'reason': 'frozen_binding_mismatch'})
    # Check actual model bytes even if a malformed manifest omitted them.
    for relative, expected in [('models/f/model.pt', F_SHA256),
                               ('models/g/g-models.joblib', G_SHA256)]:
        try:
            if file_sha256(release / relative) != expected:
                failures.append({'artifact': relative, 'reason': 'frozen_model_sha256_mismatch'})
        except OSError as exc:
            failures.append({'artifact': relative, 'reason': type(exc).__name__})
    dependencies = _dependencies(bundle.get('dependency_versions', {}))
    if not dependencies:
        failures.append({'artifact': 'bundle-config.json', 'reason': 'dependency_versions_missing'})
    release_ready = not failures and all(r['ok'] for r in dependencies)
    encoder_missing = [name for name in ('model.safetensors', 'config.json', 'tokenizer.json',
                                         'tokenizer_config.json')
                       if not (config.encoder_dir / name).is_file()]
    encoder_failures = []
    if config.encoder_dir.name != ENCODER_REVISION:
        encoder_failures.append('snapshot_revision_directory_mismatch')
    encoder_digest_checked = False
    if deep and not encoder_missing:
        try:
            provenance = json.loads((release / 'provenance/encoder-provenance.json').read_text(encoding='utf-8'))
            contract = provenance['short_contract']
            if contract['revision'] != ENCODER_REVISION:
                encoder_failures.append('encoder_contract_revision_mismatch')
            if file_sha256(config.encoder_dir / 'model.safetensors') != contract['model_sha256']:
                encoder_failures.append('encoder_model_sha256_mismatch')
            encoder_digest_checked = True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            encoder_failures.append(type(exc).__name__)
    encoding_dependencies = _dependencies({'transformers': None, 'torch': None})
    encoder_ready = (not encoder_missing and not encoder_failures
                     and all(r['ok'] for r in encoding_dependencies))
    return {
        'schema': 'footballmind-workspace-doctor-v1',
        'configuration': config.as_dict(),
        'configuration_fingerprint': config.fingerprint(),
        'offline_ready': release_ready, 'online_artifacts_ready': release_ready and encoder_ready,
        'release': {'path': str(release), 'files_checked': len(expected_files),
                    'failures': failures, 'dependencies': dependencies},
        'encoder': {'path': str(config.encoder_dir), 'missing_files': encoder_missing,
                    'failures': encoder_failures, 'dependencies': encoding_dependencies,
                    'checkpoint_digest_checked': encoder_digest_checked},
        'sources': {'status': 'not_probed', 'reason': 'offline_diagnosis'},
        'inference_executed': False, 'weights_loaded': False,
        'network_requests': 0, 'training_performed': False,
    }
