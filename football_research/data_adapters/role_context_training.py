"""Research role representation from verified original A runs, never backfill."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from football_research.acquisition.state import parse_time
from .context_training import _bindings, _read, _write
from .open_text_episodes import _digest
from .optional_context_training import build_optional_context_corpus_from_runs, _domain_counts

VERSION = 'raw-backed-role-context-training-v1'


def role_records_from_verified_runs(records, run_folders):
    """Attach source-proven roles to raw-gateway records without altering them.

    This helper is an internal projection, not an alternate raw admission API.
    Callers must use the raw A/B gateways before accepting outcome/phase claims.
    """
    from football_research.acquisition.fixture_roles import bind_fixture_roles
    from .fixture_role_identity import resolve_fixture_role_identity
    by_run = {}
    for folder in map(Path, run_folders):
        config = _read(folder/'configuration.json')
        if config['run_id'] in by_run:
            raise ValueError('Duplicate raw run identity in role projection')
        by_run[config['run_id']] = (config['supervisor_metadata']['verified_fixture_identity'],
                                    folder/'identity-archive')
    first = {}
    for row in records:
        at = parse_time(row['visible_state']['before']['decision_at'])
        first[row['run_id']] = min(at, first.get(row['run_id'], at))
    identities = {}
    for run_id, at in first.items():
        if run_id not in by_run:
            raise ValueError('Role prefix lacks its original run identity source')
        binding, archive = by_run[run_id]
        identities[run_id] = resolve_fixture_role_identity(binding, archive_root=archive, as_of=at)
    projected = []
    for row in records:
        identity = identities[row['run_id']]
        if identity['scope']['physical_match_id'] != row['physical_match_id']:
            raise ValueError('Role identity differs from the original physical A fixture')
        result = deepcopy(row)
        result['visible_state'] = bind_fixture_roles(row['visible_state'], identity)
        projected.append(result)
    return projected


def train_role_context_from_runs(run_folders, previous_plans, result_archive, output, *, as_of=None):
    """Verify all original evidence first, then fit only if the real gate holds."""
    from football_research.acquisition.fixture_roles import role_state, ROLES
    from football_research.acquisition.role_context_predictor import RoleContextPredictor, training_readiness
    started = datetime.now(timezone.utc)
    at = parse_time(started if as_of is None else as_of)
    output = Path(output).resolve()
    folders = [Path(path).resolve() for path in run_folders]
    archive = Path(result_archive).resolve()
    if at > started or output.exists():
        raise ValueError('Use a new role-training output and a reached audit time')
    if any(output.is_relative_to(path) for path in folders+[archive]):
        raise ValueError('Role training cannot modify original archives')
    before = _bindings(folders+[archive])
    output.mkdir(parents=True, exist_ok=False)
    raw = build_optional_context_corpus_from_runs(folders, previous_plans, archive, output/'raw-corpus', as_of=at)
    original_rows = _read(output/'raw-corpus/prefix-records.json')
    original_pending = _read(output/'raw-corpus/pending-prefixes.json')
    all_rows = role_records_from_verified_runs(original_rows+original_pending, folders)
    records, pending = all_rows[:len(original_rows)], all_rows[len(original_rows):]
    coverage = {role:set() for role in ROLES}
    for row in all_rows:
        state = role_state(row['visible_state'])
        for role in ROLES:
            if state['role_texts'][role]:
                coverage[role].add(row['physical_match_id'])
    ready = training_readiness(records, as_of=at, evidence_origin='real_observational')
    version = VERSION+':'+_digest({'source_bindings':before,'as_of':at.isoformat(),'records':records})[:24]
    model = None
    if ready['ready_to_fit']:
        fitted = RoleContextPredictor().fit(records, as_of=at, dataset_version=version,
                                           evidence_origin='real_observational')
        fitted.save(output/'f.json')
        model = {'file':'f.json','model_version':fitted.model_version}
    if before != _bindings(folders+[archive]):
        raise ValueError('Original role training inputs changed during reconstruction')
    report = {'schema':VERSION, 'status':'fitted_research' if model else 'not_ready',
        'created_at':started.isoformat(),'as_of':at.isoformat(),'dataset_version':version,
        'evidence_origin':'real_observational','source_bindings':before,
        'input_runs':raw['input_runs'],'physical_matches':raw['physical_matches'],
        'registered_split_counts':raw['registered_split_counts'],
        'supervised_physical_matches':raw['supervised_physical_matches'],
        'supervised_prefixes':len(records),'pending_prefixes':len(pending),
        'domain_counts':{'all':_domain_counts(all_rows),'labelled':_domain_counts(records),'pending':_domain_counts(pending)},
        'role_coverage_physical_matches':{role:len(ids) for role,ids in coverage.items()},
        'role_coverage_physical_ids':{role:sorted(ids) for role,ids in coverage.items()},
        'readiness':ready,'model':model,'records_sha256':_digest(records),'pending_sha256':_digest(pending),
        'raw_corpus_report_sha256':_digest(raw),'original_actor_role_features_changed':False,
        'representation_scope':'later_research_projection_of_then_available_original_identity_and_text',
        'role_semantics':'exact_verified_canonical_mentions_not_coreference_tactical_or_causal_understanding',
        'network_requests':0,'numeric_query_gain_created':False,'production_ready':False}
    _write(output/'prefix-records.json',records)
    _write(output/'pending-prefixes.json',pending)
    _write(output/'source-inputs.json',{'run_folders':[str(p) for p in folders],
        'result_archive':str(archive),'previous_plans':previous_plans,'as_of':at.isoformat()})
    _write(output/'report.json',report)
    return report


__all__ = ['VERSION','train_role_context_from_runs']
