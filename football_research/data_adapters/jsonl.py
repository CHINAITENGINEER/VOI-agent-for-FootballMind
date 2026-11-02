"""Portable research dataset transport with explicit record kinds."""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path

from .types import ResearchDataset, ResearchMatch, ResearchOddsSnapshot, json_record


def save_jsonl(dataset: ResearchDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        metadata = {'kind': 'metadata', 'schema': 'footballmind.research.dataset.v1',
                    'dataset_version': dataset.dataset_version,
                    'input_manifest': dataset.input_manifest, 'audit': dataset.audit}
        stream.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        for kind, rows in [('match', dataset.matches), ('odds', dataset.odds)]:
            for row in rows:
                stream.write(json.dumps({'kind': kind, **json_record(row)}, ensure_ascii=False) + '\n')


def load_jsonl(path: str | Path) -> ResearchDataset:
    matches, odds, metadata = [], [], None
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.pop('kind')
            if kind == 'metadata':
                if metadata is not None or row.get('schema') != 'footballmind.research.dataset.v1':
                    raise ValueError('Dataset metadata must appear exactly once with supported schema')
                metadata = row
                continue
            if kind not in {'match', 'odds'}:
                raise ValueError(f'Unsupported JSONL record kind on line {line_number}')
            keys = ('kickoff', 'result_available_at', 'fixture_available_at', 'source_published_at',
                    'observed_at', 'ingested_at', 'operational_available_at') if kind == 'match' else ('observed_at', 'source_timestamp')
            for key in keys:
                if row.get(key) is not None:
                    row[key] = datetime.fromisoformat(row[key].replace('Z', '+00:00'))
            row['input_manifest'] = tuple(row.get('input_manifest', []))
            (matches if kind == 'match' else odds).append((ResearchMatch if kind == 'match' else ResearchOddsSnapshot)(**row))
    if metadata is None:
        raise ValueError('Dataset metadata is required')
    return ResearchDataset(tuple(matches), tuple(odds), metadata['dataset_version'],
                           tuple(metadata['input_manifest']), metadata.get('audit', {}))
