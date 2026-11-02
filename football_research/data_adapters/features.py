"""Consume exported FeatureStore Elo values without maintaining Elo updates."""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .types import utc


def load_feature_snapshot(parquet_path: str | Path) -> dict:
    """Validate one existing FeatureStore snapshot and return its Elo projection.

    Canonical IDs remain unchanged. Joining source IDs to these IDs is a caller
    responsibility requiring an explicit platform mapping. The source manifest
    hash validates lineage; a separate full-file hash fingerprints the consumed
    feature bytes. This reader does not claim to replay the warehouse algorithm.
    """
    import pyarrow.parquet as pq

    path = Path(parquet_path).resolve()
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 1:
        raise ValueError('Expected exactly one FeatureStore snapshot per file')
    row = rows[0]
    manifest = json.loads(row['manifest_json'])
    features = json.loads(row['features_json'])
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                         default=str, allow_nan=False).encode('utf-8')
    manifest_hash = hashlib.sha256(encoded).hexdigest()
    if manifest_hash != row['input_manifest_hash']:
        raise ValueError('Feature input manifest hash mismatch')
    cutoff = utc(row['prediction_cutoff'])
    manifest_cutoff = utc(datetime.fromisoformat(manifest['prediction_cutoff'].replace('Z', '+00:00')))
    if cutoff != manifest_cutoff:
        raise ValueError('Feature cutoff differs from hashed manifest')
    for key in ('match_id', 'feature_version', 'information_mode'):
        if row[key] != manifest[key]:
            raise ValueError(f'Feature {key} differs from hashed manifest')
    if row['information_mode'] not in {'operational', 'historical_reconstruction'}:
        raise ValueError('Unknown feature information mode')
    if row['information_mode'] == 'operational' and manifest.get('availability_basis') != 'max(observed_at,ingested_at)':
        raise ValueError('Operational feature snapshot lacks observation and ingestion availability evidence')
    elo = features['elo']
    for side in ('home', 'away'):
        value = elo.get(side)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError('Elo values must be finite numbers or null')
    if sorted(elo.get('input_version_ids', [])) != sorted(manifest['history_version_ids']):
        raise ValueError('Elo lineage differs from hashed input manifest')
    if elo.get('parameters') != manifest.get('parameters'):
        raise ValueError('Elo parameters differ from hashed input manifest')
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'match_id': row['match_id'], 'prediction_cutoff': cutoff,
            'features_available_at': cutoff,
            'feature_availability_basis': 'logical_input_availability_upper_bound_from_verified_manifest;not_physical_export_time',
            'information_mode': row['information_mode'], 'feature_version': row['feature_version'],
            'home_elo': elo.get('home'), 'away_elo': elo.get('away'),
            'coverage': elo.get('coverage', 'unknown'), 'elo_parameters': elo.get('parameters'),
            'input_manifest_hash': manifest_hash, 'parquet_sha256': file_hash,
            'input_manifest': (f'feature_manifest:sha256:{manifest_hash}', f'feature_parquet:sha256:{file_hash}',
                               *tuple(f'observation:{item}' for item in manifest['observation_ids'])),
            'source_path': str(path), 'manifest': manifest,
            'verification_scope': 'export_manifest_and_file_hash_verified;warehouse_replay_not_run_here'}
