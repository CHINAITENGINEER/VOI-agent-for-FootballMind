"""Observed signed gain for a frozen visible-context predictor, not a g model.

Real inputs must be a bound raw-prefix corpus and genuinely pre-registered B
runs. The ordinary A corpus is diagnostically useful but yields no B targets.
Registration is an artifact/time binding, not a live-policy integration.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

from .context_predictor import ContextPredictor, _state, NUMERIC
from .joint_planner import _cost
from .state import parse_time
from .value import _canonical
from .visible_state import encode_visible_state

VERSION = 'observed-context-gain-supervision-v1'
REGISTRATION = 'context-predictor-calibration-registration-v1'


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')


def _replay_originals(folders):
    from tempfile import TemporaryDirectory
    from football_agent.learned_voi_replay import replay_saved_run
    checks = []
    for folder in sorted({Path(p).resolve() for p in folders}):
        try:
            with TemporaryDirectory(prefix='fmgain-') as temporary:
                replay = replay_saved_run(folder, output=Path(temporary)/'r', project_root=Path(__file__).resolve().parents[2])
        except (OSError, KeyError, RuntimeError, ValueError) as error:
            raise ValueError('Original raw gateway exact replay failed: '+str(error)) from error
        if not replay['identical_replay'] or replay['replay_network_requests'] != 0:
            raise ValueError('Original raw gateway exact replay differs')
        checks.append({key: replay[key] for key in
                       ('run_id', 'identical_replay', 'raw_responses_reprojected', 'replay_network_requests')})
    return checks


def _registration(plan):
    if (plan.get('schema') != REGISTRATION or plan.get('purpose') != 'calibration'
            or plan.get('registration_id') != REGISTRATION+':'+_digest(
                {k: v for k, v in plan.items() if k != 'registration_id'})[:24]):
        raise ValueError('Context calibration registration content differs')
    return plan


def _inventory(plans, at):
    from football_research.data_adapters.prospective_learning import _previous_inventory
    return _previous_inventory(plans, at)


def prepare_context_calibration(initialization_plan, *, identity_bundle, previous_plans,
                                predictor_path, output, created_at=None):
    """Bind an existing f to future whole fixtures; performs no HTTP/training.

    The caller first verifies the existing initialization/identity raw gateway.
    Future collection must save this registration and its binding in the run's
    supervisor metadata. Existing R189/runtime behavior is not altered here.
    """
    from football_research.data_adapters.prospective_learning import (
        _bound_matches, _training_events, _merge_inventory, _own_inventory, SCOPE)
    at = parse_time(created_at or datetime.now(timezone.utc))
    path = Path(predictor_path).resolve()
    model = ContextPredictor.load(path)
    if not parse_time(model.metadata['training_as_of']) <= parse_time(model.metadata['fit_execution_at']) <= at:
        raise ValueError('The actual fitted predictor must exist before calibration registration')
    matches = _bound_matches(initialization_plan, identity_bundle, at)
    prior = _inventory(previous_plans, at)
    events = _training_events(model, prior, 'f_train', {r['source_event_id'] for r in matches})
    if events != set(model.metadata['training_physical_match_ids']):
        raise ValueError('Frozen training aliases differ from registered physical fixtures')
    artifact = {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest(),
        'model_version': model.model_version, 'training_as_of': model.metadata['training_as_of'],
        'fit_execution_at': model.metadata['fit_execution_at'], 'frozen_at': at.isoformat()}
    plan = {'schema': REGISTRATION, 'purpose': 'calibration', 'created_at': at.isoformat(),
        'artifacts': {'text_model': artifact}, 'matches': matches, 'prior_inventory': prior,
        'inventory_scope': SCOPE, 'initialization_plan_sha256': _digest(initialization_plan),
        'identity_bundle_sha256': _digest(identity_bundle), 'evidence_origin': model.metadata['evidence_origin'],
        'scope': 'supervisor_only_prequery_context_f_binding_not_live_policy_activation'}
    _merge_inventory(prior+_own_inventory(plan))  # Includes old A aliases and postponed fixtures.
    plan['registration_id'] = REGISTRATION+':'+_digest(plan)[:24]
    _write(output, plan)
    return plan


def _bind_model(plan, model, model_sha, first_query, origin):
    _registration(plan)
    spec = plan['artifacts']['text_model']
    if (spec['sha256'] != model_sha or spec['model_version'] != model.model_version
            or spec['training_as_of'] != model.metadata['training_as_of']
            or spec['fit_execution_at'] != model.metadata['fit_execution_at']
            or plan['evidence_origin'] != origin or model.metadata['evidence_origin'] != origin):
        raise ValueError('Calibration is bound to a different frozen predictor or origin')
    if not (parse_time(spec['training_as_of']) <= parse_time(spec['fit_execution_at'])
            <= parse_time(spec['frozen_at']) == parse_time(plan['created_at']) < first_query):
        raise ValueError('Actual fitting/freezing/registration must precede the first query')


def _query_input(visible, action, prediction):
    """Query-time whitelist: this function cannot inspect an after state/label."""
    state = _state(visible)
    candidate = visible['candidates'].get(action['action_id'])
    tool = action.get('tool')
    request = ({'tool': tool, 'url': action['url']} if tool == 'read_url' else
               {'tool': tool, 'query': action['query'], 'source': action.get('source', 'open_web')})
    if (tool not in {'read_url', 'search_web'} or candidate is None or candidate['request'] != request
            or candidate['information_key'] != action['information_key']
            or not candidate['available'] or candidate['already_attempted']):
        raise ValueError('The actual action was not an available pre-query candidate')
    if candidate['declared_cost'] != _cost(action) or candidate['cost_status'] != action.get('cost_status'):
        raise ValueError('Declared query cost differs from the pre-query menu')
    before = visible['before']
    links = {link['link_ref']: link for link in before['observed_links']}
    documents = {doc['document_index']: doc for doc in before['documents']}
    parents = []
    for ref in candidate['observed_link_refs']:
        link = links.get(ref)
        if link is None or link['target_url'] != request.get('url'):
            raise ValueError('Candidate context must be an observed parent link to this request')
        doc = documents[link['document_index']]
        parents.append({'parent': {key: doc[key] for key in
            ('document_id', 'raw_sha256', 'url', 'observed_at', 'available_at')},
            'labels': [{key: anchor[key] for key in ('anchor_index', 'text', 'text_truncated')}
                       for anchor in link['observed_anchors']]})
    return {'schema': VERSION, 'state': {key: deepcopy(state[key]) for key in
                ('decision_at', 'prediction_cutoff', 'kickoff_at', 'prior', 'texts', 'sources', 'numeric')},
        'numeric_names': list(NUMERIC), 'prediction_before': list(prediction),
        'ordered_history': [{key: deepcopy(step.get(key)) for key in ('action_id', 'status', 'information_key')}
                            for step in before['history']['attempts']],
        'candidate': {'request': request, 'declared_cost': deepcopy(candidate['declared_cost']),
                      'cost_status': candidate['cost_status'], 'observed_parents': parents}}


def _signed_records(prefixes, transitions, plans_by_run, *, model, model_sha, as_of, origin):
    """Only called after corpus/run binding verification; never accepts a gain."""
    at = parse_time(as_of)
    by_id = {row['prefix_id']: row for row in prefixes}
    if len(by_id) != len(prefixes):
        raise ValueError('Duplicate prefix identity')
    first = {}
    for row in prefixes:
        now = parse_time(row['visible_state']['before']['decision_at'])
        first[row['run_id']] = min(now, first.get(row['run_id'], now))
    features, targets, excluded, seen, outcomes = [], [], [], {}, {}
    for transition in transitions:
        run_id = transition['run_id']
        if run_id not in plans_by_run:
            excluded.append({'run_id': run_id, 'step': transition['step'], 'reason': 'not_registered_context_calibration'})
            continue
        plan = plans_by_run[run_id]
        _bind_model(plan, model, model_sha, first[run_id], origin)
        if transition.get('gain_exclusion_reason') is not None:
            before = by_id[transition['before_prefix_id']]
            late = transition['gain_exclusion_reason'] == 'actual_return_after_prediction_cutoff'
            overrun = transition['gain_exclusion_reason'] == 'actual_resource_budget_overrun'
            if (not (late or overrun)
                    or transition.get('gain_timing_scope') != 'actual_immediate_return_not_next_decision'
                    or transition.get('after_prefix_id') is not None
                    or before['run_id'] != run_id
                    or before['physical_match_id'] != transition['physical_match_id']
                    or late and parse_time(transition['returned_at']) <=
                       parse_time(before['visible_state']['before']['prediction_cutoff'])
                    or overrun and (transition.get('resource_overrun') is not True
                        or not (any(transition['charged_cost'].get(unit, 0.) > cap+1e-12
                                    for unit, cap in transition['physical_budget_limits'].items())
                            or any(transition.get('exploration_charged_cost', {}).get(unit, 0.) > cap+1e-12
                                   for unit, cap in transition.get('exploration_limits', {}).items())))):
                raise ValueError('Invalid immediate-return gain exclusion')
            excluded.append({'run_id': run_id, 'step': transition['step'],
                'reason': transition['gain_exclusion_reason'],
                'returned_at': transition['returned_at'],
                'response_status': transition['response_status'],
                'actual_cost': deepcopy(transition['actual_cost'])})
            continue
        before, after = (by_id[transition[name+'_prefix_id']] for name in ('before', 'after'))
        physical = transition['physical_match_id']
        registered = [r for r in plan['matches'] if r['match_id'] == before['match_id']]
        if (len(registered) != 1 or registered[0]['source_event_id'] != physical
                or any(row['run_id'] != run_id or row['physical_match_id'] != physical for row in (before, after))
                or after['prefix_index'] != before['prefix_index']+1):
            raise ValueError('A gain target requires adjacent states of the same registered physical fixture/run')
        if 'outcome' not in before or 'outcome' not in after:
            excluded.append({'run_id': run_id, 'step': transition['step'], 'reason': 'independent_settlement_missing'})
            continue
        y = before['outcome']
        if (type(y) is not int or y not in (0, 1, 2) or y != after['outcome']
                or before['result_available_at'] != after['result_available_at']
                or physical in outcomes and outcomes[physical] != y):
            raise ValueError('Conflicting independent outcome for a physical fixture')
        outcomes[physical] = y
        bv, av = before['visible_state'], after['visible_state']
        bs, ast = _state(bv), _state(av)
        if any(bs[k] != ast[k] for k in ('provider_match_id', 'prediction_cutoff', 'kickoff_at', 'prior')):
            raise ValueError('The actual response cannot change fixture identity, deadline or original prior')
        clock = [parse_time(transition[k]) for k in ('decision_at', 'observed_at', 'available_at')]
        if (clock[0] != parse_time(bs['decision_at']) or not clock[0] <= clock[1] <= clock[2]
                <= parse_time(ast['decision_at']) <= parse_time(bs['prediction_cutoff']) < parse_time(bs['kickoff_at'])
                <= parse_time(before['result_available_at']) <= at):
            raise ValueError('Actual decision/response/after/label clocks are inconsistent')
        if (transition.get('gain_timing_scope') == 'actual_immediate_return_not_next_decision'
                and parse_time(transition['returned_at']) != parse_time(ast['decision_at'])):
            raise ValueError('The signed gain after state must use the actual return clock')
        bh, ah = bv['before']['history'], av['before']['history']
        if (not bh['order_matches_attempts'] or not ah['order_matches_attempts']
                or ah['action_history'] != bh['action_history']+[{'action_id': transition['action']['action_id']}]
                or ah['attempts'][:-1] != bh['attempts'] or not ah['attempts']
                or ah['attempts'][-1]['action_id'] != transition['action']['action_id']
                or ah['attempts'][-1]['status'] != transition['response_status']):
            raise ValueError('Full actual ordered history must append this response, including failures')
        if not transition['response_observation_eligible'] or transition['unobserved_alternative_returns_available']:
            raise ValueError('Only actual admitted observations, never unobserved alternatives, supply targets')
        p = model.predict(bv, physical_match_id=physical)
        q = model.predict(av, physical_match_id=physical)
        inputs = _query_input(bv, transition['action'], p['probabilities'])
        key = (run_id, transition['step'])
        fingerprint = _digest({'transition': transition, 'before': before, 'after': after})
        if key in seen:
            if seen[key] != fingerprint:
                raise ValueError('Conflicting repeated actual transition')
            continue
        seen[key] = fingerprint
        ident = 'context-gain:'+_digest(list(key))[:24]
        features.append({'sample_id': ident, 'inputs': inputs})
        targets.append({'sample_id': ident, 'run_id': run_id, 'physical_match_id': physical,
            'before_prefix_id': before['prefix_id'], 'after_prefix_id': after['prefix_id'],
            'outcome': y, 'result_available_at': before['result_available_at'],
            'signed_gain': math.log(q['probabilities'][y])-math.log(p['probabilities'][y]),
            'prediction_before': p['probabilities'], 'prediction_after': q['probabilities'],
            'before_support': p['support_status'], 'after_support': q['support_status'],
            'response_status': transition['response_status'], 'response_kind': transition['response_kind'],
            'actual_cost': deepcopy(transition['actual_cost']),
            'selection_probability': transition['selection_probability'],
            'selection_scope': transition['selection_scope'], 'selection_basis': transition['selection_basis'],
            'transition_sha256': fingerprint, 'registration_id': plan['registration_id'],
            'target_kind': 'realized_fixed_predictor_log_loss_reduction_not_expected_or_causal_value'})
    sizes = Counter(row['physical_match_id'] for row in targets)
    for row in targets:
        row['physical_match_weight'] = 1./(len(sizes)*sizes[row['physical_match_id']])
    return features, targets, excluded


def _load_corpus(folder):
    """Verify the original corpus bindings and every saved actual prefix/action."""
    folder = Path(folder)
    report = _read(folder/'conversion-report.json')
    report['source_corpus_directory'] = str(folder.resolve())
    report['source_conversion_report_file_sha256'] = sha256((folder/'conversion-report.json').read_bytes()).hexdigest()
    if report.get('schema') != 'raw-backed-context-prefixes-v1':
        raise ValueError('A raw-backed context prefix corpus is required')
    for path, expected in report['input_bindings'].items():
        if sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Original context corpus input bytes changed: '+path)
    rows = _read(folder/'prefix-records.json')
    pending = _read(folder/'pending-prefixes.json')
    transitions = _read(folder/'actual-transitions.json')
    for value, key in ((rows, 'records_sha256'), (pending, 'pending_sha256'), (transitions, 'transitions_sha256')):
        if _digest(value) != report[key]:
            raise ValueError('Context corpus derived content differs: '+key)
    registrations = _read(folder/'registration-inputs.json')
    if _digest(registrations) != report['registration_inputs_sha256']:
        raise ValueError('The corpus registration inventory changed')
    runs, configs, folders = {}, {}, {}
    for path in report['input_bindings']:
        path = Path(path)
        if path.name != 'run.json':
            continue
        run, config = _read(path), _read(path.with_name('configuration.json'))
        if run['configuration_digest'] != _digest(config) or run['run_id'] != config['run_id']:
            raise ValueError('Original run/configuration binding differs')
        runs[run['run_id']], configs[run['run_id']] = run, config
        folders[run['run_id']] = path.parent
    for row in rows+pending:
        run = runs[row['run_id']]
        if run.get('schema') == 'learned-voi-open-execution-v4':
            matches = [c for c in run['cycles'] if c['step'] == row['lineage']['cycle_step']]
            name = row['lineage'].get('public_snapshot')
            if len(matches) != 1 or name not in {'public_before', 'public_after'}:
                raise ValueError('An immediate prefix requires its exact original cycle/snapshot')
            cycle = matches[0]
            public = cycle.get(name)
            if not isinstance(public, dict):
                raise ValueError('A missing immediate after cannot be a supervised prefix')
            visible = encode_visible_state(public, cycle['generated_actions'] if name == 'public_before' else ())
            snapshot_hash = row['lineage']['public_snapshot_sha256']
        else:
            cycle = run['cycles'][row['prefix_index']]
            public = cycle['public_before']
            visible = encode_visible_state(public, cycle['generated_actions'])
            snapshot_hash = row['lineage']['public_before_sha256']
        if (row['lineage']['run_sha256'] != _digest(run) or row['visible_state'] != visible
                or snapshot_hash != _digest(public)
                or row['lineage']['encoded_state_sha256'] != _digest(visible)):
            raise ValueError('A prefix is not its unchanged original saved public state')
    expected = {(rid, c['step']) for rid, run in runs.items() for c in run['cycles'] if c.get('selected_action')}
    if {(t['run_id'], t['step']) for t in transitions} != expected:
        raise ValueError('Complete actual transition coverage is required')
    for row in transitions:
        cycle = next(c for c in runs[row['run_id']]['cycles'] if c['step'] == row['step'])
        if row['action'] != cycle['selected_action'] or row['decision_at'] != cycle['public_before']['decision_at']:
            raise ValueError('Only the actual selected action supplies supervision')
    identities = {}
    if report['evidence_origin'] == 'real_observational':
        # Reuse the actual raw response/identity/settlement gateway, rather
        # than trusting a re-signed integer label or a edited derived clock.
        from collections import defaultdict
        from football_research.data_adapters.context_training import _visible_prefixes
        from football_research.data_adapters.prospective_text import join_settlements, verify_run_identity
        report['exact_replay'] = _replay_originals(folders.values())
        result_roots = set()
        for raw in map(Path, report['input_bindings']):
            if raw.name.endswith('.observation.json') and not any(raw.is_relative_to(p) for p in folders.values()):
                result_roots.update(parent.parent for parent in raw.parents if parent.name == 'raw')
        if len(result_roots) != 1:
            raise ValueError('The corpus must bind one explicit independent raw result archive')
        proofs = {}
        for path in folders.values():
            for key, proof in _read(path/'prior-bindings.json').items():
                if key in proofs and proofs[key] != proof:
                    raise ValueError('Conflicting prior provenance')
                proofs[key] = proof
        joined = join_settlements(list(folders.values()), settlements=[], prior_bindings=proofs,
            as_of=parse_time(report['as_of']), result_archive=next(iter(result_roots)))
        if joined['dataset']['audit']['rejected']:
            raise ValueError('Original raw response or settlement gateway rejection')
        episodes, labels = defaultdict(list), defaultdict(list)
        for item in joined['dataset']['episodes']:
            episodes[item['run_id']].append(item)
        for check in joined['independent_result_verification']:
            labels[check['match_id']].extend(check.get('settlement_rows', []))
        verified_rows, verified_pairs = [], []
        for rid, path in folders.items():
            binding = verify_run_identity(path, configs[rid])
            if binding is None:
                raise ValueError('Independent original physical fixture binding is required')
            identities[rid] = binding
            identity = {'original_match_id': binding['match_id'], 'source_event_id': binding['canonical']['source_event_id']}
            actual, pairs = _visible_prefixes(runs[rid], episodes[rid], identity, labels, parse_time(report['as_of']))
            verified_rows.extend(actual)
            verified_pairs.extend(pairs)
        if (sorted(verified_rows, key=lambda r: r['prefix_id']) != sorted(rows+pending, key=lambda r: r['prefix_id'])
                or sorted(verified_pairs, key=lambda r: (r['run_id'], r['step'])) !=
                   sorted(transitions, key=lambda r: (r['run_id'], r['step']))):
            raise ValueError('Derived prefixes/labels/actual transitions differ from the independent raw gateway')
    elif report['evidence_origin'] != 'constructed_mechanism':
        raise ValueError('Unsupported corpus evidence origin')
    return rows+pending, transitions, configs, registrations, report, identities


def build_context_gain(corpus_dir, output, *, predictor_path=None, registrations=(), as_of=None):
    """Write separate before-only inputs and signed targets; never train g."""
    return _export_gain(_load_corpus(corpus_dir), output, predictor_path=predictor_path,
                        registrations=registrations, as_of=as_of)


def _export_gain(verified, output, *, predictor_path, registrations, as_of):
    at = parse_time(as_of or datetime.now(timezone.utc))
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new immutable context gain output directory')
    prefixes, transitions, configs, old_plans, corpus, identities = verified
    if parse_time(corpus['as_of']) > at:
        raise ValueError('The raw-backed corpus was not available at the requested supervision cutoff')
    plans = [_registration(_read(p) if isinstance(p, (str, Path)) else p) for p in registrations]
    inventory = _inventory(old_plans+plans, at)
    by_event = {row['source_event_id']: row for row in inventory}
    by_id = {p['registration_id']: p for p in plans}
    bound, rejected = {}, []
    for run_id, config in configs.items():
        supervisor = config.get('supervisor_metadata', {})
        context = supervisor.get('context_calibration')
        old = supervisor.get('learning_cohort')
        if context is None:
            rejected.append({'run_id': run_id, 'reason': 'no_original_context_B_registration'})
            continue
        if old and old['purpose'] != 'calibration':
            raise ValueError('An original A/confirmation run cannot be relabeled as B')
        plan = by_id.get(context.get('registration_id'))
        if plan is None or context.get('registration_sha256') != _digest(plan):
            raise ValueError('Original supervisor context registration binding is missing or changed')
        if run_id in identities:
            from football_research.data_adapters.prospective_learning import FACTS
            identity = identities[run_id]
            registered = [row for row in plan['matches'] if row['match_id'] == identity['match_id']]
            if (len(registered) != 1 or registered[0]['identity_binding_sha256'] != identity['binding_sha256']
                    or registered[0]['fixture_facts'] != {key: identity['canonical'][key] for key in FACTS}):
                raise ValueError('Registered fixture facts differ from independently verified original identity')
        events = {row['physical_match_id'] for row in prefixes if row['run_id'] == run_id}
        if any(event not in by_event or by_event[event]['purpose'] != 'calibration' for event in events):
            raise ValueError('Physical fixture was not registered for calibration')
        bound[run_id] = plan
    features, targets, excluded = [], [], []
    model, model_sha = None, None
    if bound and predictor_path is not None:
        model = ContextPredictor.load(predictor_path)
        model_sha = sha256(Path(predictor_path).read_bytes()).hexdigest()
        features, targets, excluded = _signed_records(prefixes, transitions, bound, model=model,
            model_sha=model_sha, as_of=at, origin=corpus['evidence_origin'])
    report = {'schema': VERSION, 'status': 'supervision_built' if targets else 'not_ready',
        'not_ready_reasons': ([] if targets else ['no_eligible_registered_B_supervision']+
                             (['frozen_context_predictor_missing'] if predictor_path is None else [])),
        'as_of': at.isoformat(), 'evidence_origin': corpus['evidence_origin'],
        'input_runs': len(configs), 'input_transitions': len(transitions), 'registered_B_runs': len(bound),
        'supervised_transitions': len(targets), 'supervised_physical_matches': len({r['physical_match_id'] for r in targets}),
        'negative_targets': sum(row['signed_gain'] < -1e-12 for row in targets),
        'zero_targets': sum(abs(row['signed_gain']) <= 1e-12 for row in targets),
        'positive_targets': sum(row['signed_gain'] > 1e-12 for row in targets),
        'rejected_runs': rejected, 'excluded_transitions': excluded,
        'predictor_model_version': model.model_version if model else None, 'predictor_sha256': model_sha,
        'source_conversion_report_sha256': _digest(corpus), 'features_sha256': _digest(features),
        'targets_sha256': _digest(targets), 'g_trained': False, 'ready_to_fit_g': False,
        'network_requests': 0, 'production_ready': False, 'stop_certificate': None,
        'scope': 'logged_actual_B_transitions_only_no_unselected_returns_or_effect_identification',
        'clock_scope': 'actual_fit_and_registration_before_first_query_not_statistical_cutoff_alone',
        'exact_replay': deepcopy(corpus.get('exact_replay', [])),
        'label_scope': 'independent_settlement_verified_by_upstream_raw_gateway_not_outcome_integer',
        'deployment_scope': 'B_raw_export_supported_new_context_registration_not_yet_connected_to_live_collector'}
    output.mkdir(parents=True, exist_ok=False)
    _write(output/'source-inputs.json', corpus)
    _write(output/'calibration-registrations.json', plans)
    report['source_inputs_file_sha256'] = sha256((output/'source-inputs.json').read_bytes()).hexdigest()
    report['registrations_file_sha256'] = sha256((output/'calibration-registrations.json').read_bytes()).hexdigest()
    _write(output/'features.json', features)
    _write(output/'targets.json', targets)
    _write(output/'report.json', report)
    return report


def build_context_gain_from_runs(run_folders, result_archive, output, *, predictor_path=None,
                                 registrations=(), previous_plans=(), as_of=None,
                                 evidence_origin='real_observational'):
    """B-compatible raw gateway; does not use the A-only corpus registrar."""
    from collections import defaultdict
    from football_research.data_adapters.context_training import _bindings, _visible_prefixes
    from football_research.data_adapters.prospective_text import join_settlements, verify_run_identity
    if not run_folders or not result_archive or evidence_origin not in {'real_observational', 'constructed_mechanism'}:
        raise ValueError('Explicit original runs, independent result archive and origin required')
    at = parse_time(as_of or datetime.now(timezone.utc))
    folders = [Path(path) for path in run_folders]
    bindings = _bindings(folders+[Path(result_archive)])
    configs, runs, proofs, identities = {}, {}, {}, {}
    for folder in folders:
        config, run = _read(folder/'configuration.json'), _read(folder/'run.json')
        rid = run['run_id']
        if (run['configuration_digest'] != _digest(config) or rid != config['run_id']
                or not parse_time(config['created_at']) <= parse_time(run['finished_at']) <= at
                or rid in runs and runs[rid] != run):
            raise ValueError('Original completed run/configuration binding differs')
        binding = verify_run_identity(folder, config)
        if binding is None:
            raise ValueError('Independent original physical fixture binding is required')
        configs[rid], runs[rid], identities[rid] = config, run, binding
        for version, proof in _read(folder/'prior-bindings.json').items():
            if version in proofs and proofs[version] != proof:
                raise ValueError('Conflicting original market-prior binding')
            proofs[version] = proof
    replays = _replay_originals(folders)
    joined = join_settlements(folders, settlements=[], prior_bindings=proofs, as_of=at, result_archive=result_archive)
    if joined['dataset']['audit']['rejected']:
        raise ValueError('Original raw response or settlement gateway rejection')
    episodes, labels = defaultdict(list), defaultdict(list)
    for row in joined['dataset']['episodes']:
        episodes[row['run_id']].append(row)
    for check in joined['independent_result_verification']:
        labels[check['match_id']].extend(check.get('settlement_rows', []))
    prefixes, transitions = [], []
    for rid, run in runs.items():
        bound = identities[rid]
        identity = {'original_match_id': bound['match_id'], 'source_event_id': bound['canonical']['source_event_id']}
        rows, pairs = _visible_prefixes(run, episodes[rid], identity, labels, at)
        prefixes.extend(rows)
        transitions.extend(pairs)
    if bindings != _bindings(folders+[Path(result_archive)]):
        raise ValueError('Original bytes changed during B supervision reconstruction')
    source = {'schema': 'raw-context-gain-inputs-v1', 'as_of': at.isoformat(),
              'evidence_origin': evidence_origin, 'input_bindings': bindings,
              'run_folders': [str(p.resolve()) for p in folders], 'result_archive': str(Path(result_archive).resolve()),
              'previous_plans': deepcopy(previous_plans), 'exact_replay': replays}
    return _export_gain((prefixes, transitions, configs, previous_plans, source, identities), output,
        predictor_path=predictor_path, registrations=registrations, as_of=at)


__all__ = ['prepare_context_calibration', 'build_context_gain', 'build_context_gain_from_runs',
           'VERSION', 'REGISTRATION']
