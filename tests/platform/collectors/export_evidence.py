"""A-owned sample handoff exporter, NOT a production RawArchive.

Reads existing real observations without contacting providers. Writes redacted
fixture copies plus provenance; original archives are untouched. No DB imports.
"""
from __future__ import annotations
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'football_spider' / 'platform_data'
DEST = ROOT / 'tests' / 'fixtures' / 'collectors'
SECRET_KEY = re.compile(r'^(?:authorization|cookie|set-cookie|api[-_]?key|token|access[-_]?token|password|secret|rkey)$', re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def scrub(value):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if SECRET_KEY.match(k) and v not in (None, 'null', '') else scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.text.append(data.strip())


def export():
    observations = [json.loads(p.read_bytes()) for p in (ARCHIVE / 'raw').glob('*/*/*.observation.json')]
    observations.sort(key=lambda x: (x['observed_at'], x['id']))
    selected = []
    for endpoint, name in [('GetSimpleMatchsAll', 'list'), ('GetList', 'recent'), ('GetInfo', 'detail'), ('GetMoreSpInfo', 'odds')]:
        records = [o for o in observations if o['source'] == 'spftll' and o['http_status'] == 200 and o['endpoint'].endswith(endpoint) and not o.get('context', {}).get('legacy_import')]
        if records:
            selected.append(('spftll', name, records[-1]))
    # Two actual same-event observations, not one file copied with invented times.
    acceptance = json.loads((ARCHIVE / 'reports' / 'real-acceptance.json').read_bytes())
    for index, evidence in enumerate(acceptance['same_match_observations'], 1):
        obs = next(o for o in observations if o['id'] == evidence['observation_id'])
        selected.append(('spftll', f'repeated_detail_{index}', obs))
    failures = [o for o in observations if o['source'] == 'spftll' and o.get('error_class')]
    if failures:
        selected.append(('spftll', 'transport_failure', failures[0]))
    for source in ('understat', 'clubelo', 'fbref'):
        candidates = [o for o in observations if o['source'] == source + '_probe']
        if candidates:
            selected.append((source, 'public_probe', candidates[-1]))
    manifest = {'purpose': 'Role A handoff fixtures; never import these copies as fresh production observations',
                'origin': 'existing_real_archives', 'network_requests_during_export': 0, 'samples': []}
    for source, name, obs in selected:
        original_path = (ARCHIVE / obs['payload_path']).resolve()
        if not original_path.is_relative_to(ARCHIVE.resolve()):
            raise ValueError('Out-of-root original archive')
        raw = gzip.decompress(original_path.read_bytes())
        if sha(raw) != obs['content_hash']:
            raise ValueError('Original archive checksum mismatch')
        fmt = obs['payload_format']
        if fmt == 'json':
            output = json.dumps(scrub(json.loads(raw)), ensure_ascii=False, indent=2).encode('utf-8')
            extension, transformation = '.json', 'JSON reserialized; credential-key fields redacted'
        elif fmt == 'html':
            parser = VisibleHTML()
            parser.feed(raw.decode('utf-8', errors='replace'))
            output = '\n'.join(parser.text).encode('utf-8')
            extension, transformation = '.txt', 'Visible HTML text only; scripts, styles, attributes and challenge tokens omitted; original retained privately'
        else:
            output = raw
            extension, transformation = '.bin', 'unchanged bytes'
        folder = DEST / source
        folder.mkdir(parents=True, exist_ok=True)
        payload_name = name + extension
        (folder / payload_name).write_bytes(output)
        metadata = {'sample_kind': 'real_capture_redacted_copy', 'source': source, 'observation': scrub(obs),
                    'original_payload_sha256': sha(raw), 'fixture_payload_sha256': sha(output),
                    'transformation': transformation, 'payload_file': payload_name, 'production_ingestion_allowed': False}
        (folder / (name + '.metadata.json')).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest['samples'].append({'source': source, 'name': name, 'observed_at': obs['observed_at'], 'observation_id': obs['id'],
                                    'http_status': obs['http_status'], 'error_class': obs.get('error_class'),
                                    'metadata': str((folder / (name + '.metadata.json')).relative_to(ROOT)),
                                    'fixture_payload_sha256': sha(output)})
    research = []
    public_manifest = ROOT / 'var/parallel/collectors/public-verification/manifest.json'
    if public_manifest.exists():
        for evidence in json.loads(public_manifest.read_bytes()):
            if evidence['name'] == 'football_data_public_catalog' and evidence.get('http_status') == 200:
                research.append(('football_data', 'public_competitions', evidence))
    understat_manifest = ROOT / 'var/parallel/collectors/understat-research/data-manifest.json'
    if understat_manifest.exists():
        evidence = json.loads(understat_manifest.read_bytes())
        if evidence.get('http_status') == 200:
            research.append(('understat', 'epl_2025', evidence))
    for source, name, evidence in research:
        raw = (ROOT / evidence['payload']).read_bytes()
        if sha(raw) != evidence['sha256']:
            raise ValueError('Research evidence checksum mismatch')
        output = json.dumps(scrub(json.loads(raw)), ensure_ascii=False, indent=2).encode('utf-8')
        folder = DEST / source
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (name + '.json')).write_bytes(output)
        metadata = {'sample_kind': 'real_capture_redacted_copy', 'source': source, 'evidence': scrub(evidence),
                    'original_payload_sha256': sha(raw), 'fixture_payload_sha256': sha(output),
                    'transformation': 'JSON reserialized; credential-key fields redacted',
                    'payload_file': name + '.json', 'production_ingestion_allowed': False}
        metadata_path = folder / (name + '.metadata.json')
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest['samples'].append({'source': source, 'name': name, 'observed_at': evidence['observed_at'],
                                    'evidence_id': evidence['evidence_id'], 'http_status': evidence['http_status'],
                                    'metadata': str(metadata_path.relative_to(ROOT)), 'fixture_payload_sha256': sha(output)})
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    output = ROOT / 'var' / 'parallel' / 'collectors'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'fixture-export-report.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'exported_samples': len(manifest['samples']), 'new_requests': 0, 'manifest': str(DEST / 'manifest.json')}))


if __name__ == '__main__':
    export()
