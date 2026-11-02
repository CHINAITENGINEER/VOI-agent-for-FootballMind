"""Read completed, hashed platform exports; never parse a provider response.

Historical availability is an explicit experimental assumption. It is distinct
from the time the source bytes were observed, which remains in the manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .types import ResearchDataset, ResearchMatch, utc


EXPORT_SCHEMA = 'footballmind.platform.history-export.v1'
ADAPTER_VERSION = 'platform-history-research-v1.0'


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes | str) -> Any:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON field: {key}')
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError(f'Non-finite JSON number: {value}')

    return json.loads(data, object_pairs_hook=unique_pairs, parse_constant=bad_constant)


def _stamp(value: str | datetime) -> datetime:
    return utc(value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00')))


@dataclass(frozen=True)
class VerifiedHistoryExport:
    """The verified bytes are retained so readers do not reopen mutable files."""

    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    payloads: dict[str, bytes]

    def rows(self, file_key: str) -> list[dict[str, Any]]:
        rows = [_json(line) for line in self.payloads[file_key].splitlines() if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f'{file_key} requires one JSON object per row')
        expected = self.manifest['files'][file_key].get('row_count')
        if expected is not None and (type(expected) is not int or expected != len(rows)):
            raise ValueError(f'{file_key} row count mismatch')
        return rows


def verify_completed_export(directory: str | Path) -> VerifiedHistoryExport:
    """Require a final completion marker and verify every declared file hash.

    COMPLETE.json is the producer's atomic publication boundary. Relative file
    paths may resolve only inside this one immutable version directory.
    """
    root = Path(directory).resolve()
    marker_path = root / 'COMPLETE.json'
    if not marker_path.is_file():
        raise ValueError('Platform export is incomplete: COMPLETE.json is absent')
    marker_bytes = marker_path.read_bytes()
    marker = _json(marker_bytes)
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest_hash = _sha256(manifest_bytes)
    if marker.get('manifest_sha256') != manifest_hash:
        raise ValueError('Completion marker manifest hash mismatch')
    manifest = _json(manifest_bytes)
    if manifest.get('schema') != EXPORT_SCHEMA:
        raise ValueError('Unsupported platform history export schema')
    if not manifest.get('dataset_version'):
        raise ValueError('Platform history export requires dataset_version')
    files = manifest.get('files')
    if not isinstance(files, dict) or 'matches' not in files:
        raise ValueError('Platform history export requires a matches file')
    payloads = {}
    declared_paths = set()
    for key, spec in files.items():
        candidate = Path(spec['path'])
        target = (root / candidate).resolve()
        if candidate.is_absolute() or not target.is_relative_to(root) or target == root:
            raise ValueError('Export file path must stay inside its version directory')
        if target in declared_paths:
            raise ValueError('Export file paths must be unique')
        declared_paths.add(target)
        payload = target.read_bytes()
        if spec.get('sha256') != _sha256(payload):
            raise ValueError(f'Platform export file hash mismatch: {key}')
        payloads[key] = payload
    if marker_path.read_bytes() != marker_bytes or (root / 'manifest.json').read_bytes() != manifest_bytes:
        raise ValueError('Platform export changed while being read')
    return VerifiedHistoryExport(root, manifest, manifest_hash, payloads)


def _assumptions(manifest: dict[str, Any]) -> dict[str, Any]:
    assumptions = manifest.get('assumptions')
    if not isinstance(assumptions, dict):
        raise ValueError('Historical reconstruction requires explicit assumptions')
    for name in ('result_delay_hours', 'xg_delay_hours', 'fixture_lead_hours'):
        value = assumptions.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Historical reconstruction requires finite {name}')
        if value < 0:
            raise ValueError(f'Historical reconstruction requires nonnegative {name}')
    if assumptions['result_delay_hours'] < 2:
        raise ValueError('Reconstructed result delay must include at least two hours after kickoff')
    if assumptions['xg_delay_hours'] < assumptions['result_delay_hours']:
        raise ValueError('Reconstructed xG availability must not precede the result availability assumption')
    if not assumptions.get('kickoff_timezone_policy'):
        raise ValueError('Historical reconstruction requires kickoff timezone policy')
    return assumptions


def _lineage(manifest: dict[str, Any], manifest_hash: str) -> tuple[str, ...]:
    evidence = manifest.get('source_evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Platform export requires source_evidence')
    lineage = [f'platform_export_manifest:sha256:{manifest_hash}']
    for item in evidence:
        ident = item.get('observation_id') or item.get('evidence_id')
        digest = item.get('sha256') or item.get('content_hash')
        if not ident or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('Source evidence requires observation ID and SHA256')
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError('Source evidence SHA256 must be hexadecimal') from exc
        lineage.extend((f'observation:{ident}', f'source_payload:sha256:{digest}'))
    return tuple(dict.fromkeys(lineage))


def load_completed_history_export(directory: str | Path) -> ResearchDataset:
    """Project normalized match rows into ResearchDataset without adding features.

    Source forecast, current match xG and season aggregates are intentionally
    absent from ResearchMatch. Batch feature consumption is a separate join.
    Assumption delays are measured from kickoff (including the match duration).
    """
    export = verify_completed_export(directory)
    manifest = export.manifest
    mode = manifest.get('information_mode')
    if mode not in {'operational', 'historical_reconstruction', 'synthetic_test'}:
        raise ValueError('Unknown platform export information_mode')
    source = manifest.get('source')
    if not isinstance(source, str) or not source:
        raise ValueError('Platform export requires source')
    observed_at = _stamp(manifest['source_observed_at'])
    assumptions = _assumptions(manifest) if mode != 'operational' else {}
    lineage = _lineage(manifest, export.manifest_sha256)
    matches = []
    settlement_metadata = {}
    ignored_fields = set()
    for row in export.rows('matches'):
        for name in ('match_id', 'competition_id', 'home_team_id', 'away_team_id'):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f'Normalized match {name} must be a nonempty string')
        if row.get('information_mode', mode) != mode:
            raise ValueError('Match information mode conflicts with export manifest')
        if row.get('source', source) != source:
            raise ValueError('Match source conflicts with export manifest')
        kickoff = _stamp(row['kickoff'])
        status, period = row['status'], row['score_period']
        if status not in {'scheduled', 'live', 'finished', 'postponed', 'cancelled', 'suspended', 'unknown'}:
            raise ValueError('Unknown normalized match status')
        if period not in {'regulation', 'extra_time', 'penalties', 'unknown'}:
            raise ValueError('Unknown normalized match score period')
        home_goals, away_goals = row.get('home_goals'), row.get('away_goals')
        has_label = status == 'finished' and period == 'regulation' and home_goals is not None and away_goals is not None
        row_observed = _stamp(row.get('observed_at') or observed_at)
        row_ingested = _stamp(row['ingested_at']) if row.get('ingested_at') else None
        row_published = _stamp(row['source_published_at']) if row.get('source_published_at') else None
        real_availability = max(row_observed, row_ingested or row_observed)
        if row_published and row_published > row_observed:
            raise ValueError('Source publication is later than actual observation')
        if mode == 'operational':
            fixture_available = _stamp(row.get('fixture_available_at', real_availability))
            result_available = _stamp(row.get('result_available_at', real_availability)) if has_label else None
            if fixture_available < real_availability or (result_available and result_available < real_availability):
                raise ValueError('Operational match availability precedes observed/ingested evidence')
            time_basis = 'max(observed_at,ingested_at);no_historical_backdating'
        else:
            fixture_available = kickoff - timedelta(hours=assumptions['fixture_lead_hours'])
            result_available = kickoff + timedelta(hours=assumptions['result_delay_hours']) if has_label else None
            for field, expected in (('fixture_available_at', fixture_available), ('result_available_at', result_available)):
                if row.get(field) is not None and _stamp(row[field]) != expected:
                    raise ValueError(f'Match {field} conflicts with declared reconstruction assumption')
            time_basis = f'{mode};assumed_fixture_lead_and_result_delay;source_observed_at={observed_at.isoformat()};revised_snapshot_bias'
        if row.get('input_manifest') is not None and not isinstance(row['input_manifest'], list):
            raise ValueError('Match input_manifest must be a list of evidence references')
        row_lineage = row.get('input_manifest', [])
        if any(not isinstance(value, str) or not value for value in row_lineage):
            raise ValueError('Match input manifest references must be nonempty strings')
        matches.append(ResearchMatch(
            match_id=str(row['match_id']), competition_id=str(row['competition_id']),
            season=str(row['season']) if row.get('season') is not None else None,
            home_team_id=str(row['home_team_id']), away_team_id=str(row['away_team_id']),
            kickoff=kickoff, home_goals=home_goals, away_goals=away_goals,
            result_available_at=result_available, fixture_available_at=fixture_available,
            information_mode=mode, source=source, input_manifest=tuple(dict.fromkeys((*lineage, *row_lineage))),
            score_period=period, status=status, time_basis=time_basis,
            entity_quality=row.get('entity_quality', 'platform_export_id'),
            competition_name=row.get('competition_name', ''), home_team_name=row.get('home_team_name', ''),
            away_team_name=row.get('away_team_name', ''), source_published_at=row_published,
            observed_at=row_observed, ingested_at=row_ingested, operational_available_at=real_availability))
        settlement_metadata[row['match_id']] = {key: row[key] for key in
            ('settlement_status', 'pending_reason', 'blocked_observation_id', 'version_id') if key in row}
        ignored_fields.update(set(row) & {'forecast', 'xg', 'home_xg', 'away_xg', 'players', 'season_totals'})
    matches.sort(key=lambda match: (match.kickoff, match.match_id))
    audit = {
        'adapter_version': ADAPTER_VERSION, 'platform_dataset_version': manifest['dataset_version'],
        'information_mode': mode, 'source_observed_at': observed_at.isoformat(),
        'source_evidence': manifest['source_evidence'], 'assumptions': assumptions,
        'manifest_sha256': export.manifest_sha256,
        'files': manifest['files'], 'export_directory': str(export.directory),
        'match_count': len(matches), 'regulation_label_count': sum(match.outcome is not None for match in matches),
        'market_quote_count': 0, 'market_coverage': 'source_has_no_verified_historical_prematch_quotes',
        'ignored_target_or_unverified_fields': sorted(ignored_fields),
        'source_revision_bias': manifest.get('source_revision_bias', 'Observed source snapshot may include later corrections; original historical source versions are absent'),
        'timing_caveat': 'Reconstructed schedules, results and xG availability are assumptions, not proof of operational availability' if mode != 'operational' else 'Actual observation/ingestion bounds retained',
        'feature_consumption': 'separate_batch_join;target_match_xg_and_forecast_are_not_model_features',
        'match_settlement_metadata': settlement_metadata,
    }
    return ResearchDataset(tuple(matches), (), f'{ADAPTER_VERSION}:{export.manifest_sha256}', lineage, audit)


def main() -> None:
    """Standalone adapter CLI; no network, database or raw provider parsing."""
    import argparse
    from .jsonl import save_jsonl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', required=True, help='Completed platform export version directory')
    parser.add_argument('--output', required=True, help='ResearchDataset JSONL output path')
    args = parser.parse_args()
    dataset = load_completed_history_export(args.export)
    save_jsonl(dataset, args.output)
    print(json.dumps({'dataset_version': dataset.dataset_version,
                      'match_count': len(dataset.matches),
                      'regulation_label_count': dataset.audit['regulation_label_count'],
                      'information_mode': dataset.audit['information_mode'],
                      'manifest_sha256': dataset.audit['manifest_sha256'],
                      'output': str(Path(args.output).resolve())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
