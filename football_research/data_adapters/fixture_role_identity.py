"""Portable original identity evidence for ordered, fixture-scoped text roles.

This adapter does not discover names or aliases. The existing official gateway
reconstructs them from its original bytes. Archive locators and the current
research derivation clock remain outside the stable model-facing proof.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import base64
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory

from football_research.acquisition.state import parse_time
from football_spider.storage.raw_archive import RawArchive, digest

VERSION = 'fixture-role-identity-v1'
_CORE = ('schema', 'binding', 'scope', 'proof', 'identity_sha256')


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Portable identity proof requires canonical relative paths')
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or '..' in path.parts
            or str(path) != value or path.parts[0] not in {'raw', 'requests'}):
        raise ValueError('Identity proof path escapes its original archive')
    if not (value.endswith('.observation.json') or value.endswith('.body.gz')
            or value.startswith('requests/') and value.endswith('.json')):
        raise ValueError('Only original identity observations and request receipts are supported')
    return path


def _snapshot(archive, binding):
    wanted = {row['observation_id'] for row in binding['observations']}
    entries = []
    for metadata in sorted((archive.root/'raw').glob('*/*/*.observation.json')):
        import json
        observation = json.loads(metadata.read_bytes())
        if observation['id'] not in wanted:
            continue
        payload = (archive.root/observation['payload_path']).resolve()
        if not payload.is_relative_to(archive.root):
            raise ValueError('Identity observation payload escapes its archive')
        for path in (metadata, payload):
            body = path.read_bytes()
            entries.append({'path':path.relative_to(archive.root).as_posix(),
                'sha256':sha256(body).hexdigest(), 'bytes_base64':base64.b64encode(body).decode('ascii')})
    if binding['schema'] == 'official-fixture-evidence-identity-v1':
        for path in sorted((archive.root/'requests').glob('*.json')):
            body = path.read_bytes()
            entries.append({'path':path.relative_to(archive.root).as_posix(),
                'sha256':sha256(body).hexdigest(), 'bytes_base64':base64.b64encode(body).decode('ascii')})
    return sorted(entries, key=lambda row:row['path'])


def _scope(binding):
    canonical = binding['canonical']
    for key in ('home_id','away_id','home_name','away_name','competition','kickoff','source_event_id'):
        if not isinstance(canonical.get(key), str) or not canonical[key].strip():
            raise ValueError('Verified ordered canonical fixture fields are required for text roles')
    if canonical['home_id'] == canonical['away_id']:
        raise ValueError('Fixture text roles require distinct home and away entities')
    if binding['schema'] == 'official-fixture-evidence-identity-v1':
        runtime = {'home_team_id':canonical['home_id'], 'away_team_id':canonical['away_id'],
            'competition_id':canonical['competition'], 'kickoff':canonical['kickoff']}
        cutoff = binding['prediction_cutoff']
    else:
        initialization = binding.get('initialization_manifest', {}).get('market_initialization')
        if not isinstance(initialization, dict):
            raise ValueError('This verified identity has no supported runtime fixture mapping')
        fixture = initialization['fixture']
        runtime = {key:fixture[key] for key in ('home_team_id','away_team_id','competition_id','kickoff')}
        cutoff = initialization['prediction_cutoff']
        if fixture['match_id'] != binding['match_id']:
            raise ValueError('The original market fixture mapping is not this verified identity')
    kickoff = parse_time(canonical['kickoff'])
    if (parse_time(runtime['kickoff']) != kickoff or runtime['home_team_id'] == runtime['away_team_id']
            or any(not isinstance(runtime[k],str) or not runtime[k] for k in runtime)):
        raise ValueError('The verified source and ordered runtime fixture scope disagree')
    runtime['kickoff'] = kickoff.isoformat()
    available = max([parse_time(binding['verified_at'])] +
                    [parse_time(row['observed_at']) for row in binding['observations']])
    if not available <= parse_time(cutoff) < kickoff:
        raise ValueError('Official fixture evidence must precede its original prediction cutoff')
    return {'match_id':binding['match_id'], 'physical_match_id':canonical['source_event_id'],
        **{key:canonical[key] for key in ('home_id','away_id','home_name','away_name','competition')},
        'competition_id':runtime['competition_id'], 'runtime_fixture':runtime,
        'kickoff':kickoff.isoformat(), 'prediction_cutoff':parse_time(cutoff).isoformat(),
        'available_at':available.isoformat(), 'identity_binding_sha256':binding['binding_sha256'],
        'aliases':{side:[canonical[side+'_name']] for side in ('home','away')},
        'alias_basis':'canonical_names_reconstructed_from_original_official_bytes_only',
        'observations':[{key:deepcopy(row[key]) for key in ('observation_id','raw_sha256','observed_at')}
                        for row in binding['observations']]}


def resolve_fixture_role_identity(binding, *, archive_root, as_of):
    """Read the real original archive once and build a portable identity proof."""
    from .verified_fixtures import verify_identity_binding
    archive = RawArchive(Path(archive_root))
    verified = verify_identity_binding(binding, archive_root=archive.root, as_of=as_of)
    if verified.get('status') != 'CONFIRMED':
        raise ValueError('Text role identity must be independently confirmed from original bytes')
    result = {'schema':VERSION, 'binding':deepcopy(verified), 'scope':_scope(verified),
              'proof':{'files':_snapshot(archive, verified)}}
    result['proof']['archive_sha256'] = digest(result['proof']['files'])
    result['identity_sha256'] = digest(result)
    # This later re-expression is not a newly observed source and cannot be
    # silently backdated into an old Actor input. It is not a model feature.
    result['derivation'] = {'actual_created_at':datetime.now(timezone.utc).isoformat(),
        'verification_as_of':parse_time(as_of).isoformat(), 'archive_root':str(archive.root),
        'meaning':'current_reexpression_of_original_identity_not_actor_historical_input'}
    verify_fixture_role_identity(result, as_of=as_of)
    return result


def verify_fixture_role_identity(identity, *, as_of):
    """Return only stable fields after raw-byte reconstruction; never HTTP."""
    from .verified_fixtures import verify_identity_binding
    if (not isinstance(identity,dict) or identity.get('schema') != VERSION
            or set(identity)-set(_CORE)-{'derivation'} or any(k not in identity for k in _CORE)):
        raise ValueError('A raw-backed fixture role identity capsule is required')
    core = {key:deepcopy(identity[key]) for key in _CORE}
    if core['identity_sha256'] != digest({k:v for k,v in core.items() if k!='identity_sha256'}):
        raise ValueError('Fixture role identity capsule digest changed')
    proof = core['proof']
    if (not isinstance(proof,dict) or set(proof)!={'files','archive_sha256'}
            or not isinstance(proof['files'],list) or not proof['files'] or len(proof['files'])>64
            or proof['archive_sha256'] != digest(proof['files'])):
        raise ValueError('Original identity archive proof is missing or changed')
    names, decoded, size = set(), [], 0
    for row in proof['files']:
        if not isinstance(row,dict) or set(row)!={'path','sha256','bytes_base64'}:
            raise ValueError('Original identity proof file specification changed')
        path = _relative(row['path'])
        if path in names or not isinstance(row['bytes_base64'],str) or len(row['bytes_base64'])>8_000_000:
            raise ValueError('Duplicate or oversized original identity proof')
        names.add(path)
        try:
            body = base64.b64decode(row['bytes_base64'],validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError('Invalid original identity proof encoding') from error
        size += len(body)
        if size>16_000_000 or sha256(body).hexdigest()!=row['sha256']:
            raise ValueError('Original identity proof bytes changed or exceed the supported bound')
        decoded.append((path,body))
    with TemporaryDirectory(prefix='football-role-identity-') as temporary:
        root = Path(temporary).resolve()
        for relative,body in decoded:
            target = root.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Original identity proof path escapes its temporary archive')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(body)
        verified = verify_identity_binding(core['binding'],archive_root=root,as_of=as_of)
        if verified.get('status')!='CONFIRMED' or _scope(verified)!=core['scope']:
            raise ValueError('Fixture role scope does not reproduce its official evidence')
        expected_ids={row['observation_id'] for row in verified['observations']}
        if {row.id for row in RawArchive(root).observations()}!=expected_ids:
            raise ValueError('Identity capsule contains unrelated observations')
    if parse_time(core['scope']['available_at'])>parse_time(as_of):
        raise ValueError('Fixture role identity was unavailable at this decision')
    return core


__all__ = ['VERSION','resolve_fixture_role_identity','verify_fixture_role_identity']
