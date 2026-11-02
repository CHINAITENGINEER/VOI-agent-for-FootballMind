from __future__ import annotations
import hashlib, json, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root = ROOT / 'var/work/fm-today-bournemouth-sunderland-20261018-v1'
out = root / 'frozen-f-paired-controls-v1'
files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'manifest.json')
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {
    'schema': 'footballmind-f-paired-controls-manifest-v1',
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'files': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files],
    'source_inputs': [
        {'path': 'football365-open.txt', 'sha256': sha(root/'football365-open.txt')},
        {'path': 'market-prior.json', 'sha256': sha(root/'market-prior.json')},
        {'path': 'f-entity-semantic-f-shadow-materialization-20261011-v1/model.json',
         'sha256': sha(ROOT/'var/f-entity-semantic-f-shadow-materialization-20261011-v1/model.json')},
    ],
    'validation': {'py_compile': True, 'provider_http': 0, 'training': False,
                   'body_consumed_positive': True, 'modeled_claims_positive': 5,
                   'prediction_changed_positive': True, 'body_consumed_negative': True,
                   'modeled_claims_negative': 0, 'prediction_changed_negative': False,
                   'outcome_read': False,
                   'targeted_tests': '26 passed: test_entity_semantic_f_collection + test_historical_f_diagnostics'},
}
(out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
zip_path = root / 'frozen-f-paired-controls-v1.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.iterdir()):
        if p.is_file():
            z.write(p, arcname=f'frozen-f-paired-controls-v1/{p.name}')
with zipfile.ZipFile(zip_path) as z:
    bad = z.testzip()
    if bad is not None:
        raise RuntimeError(bad)
print(json.dumps({'zip': str(zip_path), 'zip_sha256': sha(zip_path), 'zip_bytes': zip_path.stat().st_size,
                  'manifest_sha256': sha(out/'manifest.json'), 'zip_testzip': None}, ensure_ascii=False))
