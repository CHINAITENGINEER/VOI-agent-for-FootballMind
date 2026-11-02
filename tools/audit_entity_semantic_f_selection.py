"""Audit capability-constrained f selection without changing the formal model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_research.acquisition.context_predictor import _digest  # noqa: E402
from football_research.acquisition.entity_semantic_f import EntityEvidenceBank, EntitySemanticF  # noqa: E402
from football_research.acquisition.entity_semantic_f_selection_audit import (  # noqa: E402
    VERSION, audit_candidate_grid,
)
from football_research.acquisition.entity_semantic_f_selection_registration import (  # noqa: E402
    validate_registered_audit, validate_registration,
)
from football_research.acquisition.historical_context_training import prepare_historical_cases  # noqa: E402
from tools.run_entity_semantic_f_experiment import _prepare_rows  # noqa: E402


DEFAULT_SOURCE = ROOT / 'var/f-entity-semantic-20261011-v2'
DEFAULT_OUTPUT = ROOT / 'var/f-entity-semantic-selection-audit-20261011-v1'


def _read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _binding(path: Path) -> dict[str, str]:
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _resolve_source_bindings(protocol: dict, source: Path) -> dict[str, Path]:
    bindings = protocol.get('source_bindings')
    if not isinstance(bindings, dict):
        raise ValueError('formal protocol source_bindings are required')
    resolved = {}
    for key in ('cases', 'priors', 'records'):
        row = bindings.get(key)
        if not isinstance(row, dict) or not isinstance(row.get('path'), str):
            raise ValueError(f'formal protocol source binding missing: {key}')
        path = Path(row['path'])
        if not path.exists():
            raise ValueError(f'formal source binding is missing: {path}')
        if row.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'formal source binding digest differs: {path}')
        resolved[key] = path
    # The audit reads only immutable source files and the formal model.  The
    # artifact directory is kept in the report so the boundary is reviewable.
    resolved['formal_model'] = source / 'model.json'
    return resolved


def _run(source: Path, output: Path, registration_path: Path | None = None) -> dict:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    output.mkdir(parents=True)

    # This is an offline replay.  Any accidental network access is a hard
    # failure and cannot silently turn the A-only audit into a new collection.
    original = (socket.socket.connect, socket.socket.connect_ex,
                socket.getaddrinfo, socket.create_connection)
    def _deny(*args, **kwargs):
        raise AssertionError('f selection audit is offline; network is forbidden')
    socket.socket.connect = socket.socket.connect_ex = socket.getaddrinfo = socket.create_connection = _deny
    try:
        protocol = _read(source / 'protocol.json')
        registration = None
        if registration_path is not None:
            registration = validate_registration(_read(registration_path.resolve()))
        paths = _resolve_source_bindings(protocol, source)
        formal_model = EntitySemanticF.load(paths['formal_model'])
        snapshots = _read(paths['priors'])
        records = _read(paths['records'])
        bank = EntityEvidenceBank(records)
        splits = protocol.get('splits')
        if not isinstance(splits, dict) or not isinstance(splits.get('A'), list):
            raise ValueError('formal A split is required')
        a_ids = list(splits['A'])
        raw_cases = _read(paths['cases'])
        if not isinstance(raw_cases, list):
            raise ValueError('formal cases archive must be a list')
        raw_a_cases = [case for case in raw_cases
                       if case.get('fixture', {}).get('physical_match_id') in set(a_ids)]
        if sorted(case.get('fixture', {}).get('physical_match_id') for case in raw_a_cases) != sorted(a_ids):
            raise ValueError('A split does not identify an exact case subset')
        # Filter the immutable archive before historical validation.  B/C case
        # objects and labels are not read by this audit at all.
        cases, _ = prepare_historical_cases(raw_a_cases)
        prepared, training = _prepare_rows(cases, snapshots, bank, {'A': a_ids})
        if sorted(prepared) != sorted(a_ids):
            raise ValueError('prepared A cases do not match the formal split')
        report = audit_candidate_grid(training, formal_model, cases, snapshots, bank, a_ids,
                                      registration=registration)
        if registration is not None:
            # This is intentionally evaluated before writing a successful
            # receipt.  A retrospective/current A corpus must fail instead of
            # being relabelled as a registered future experiment.
            registration_check = validate_registered_audit(registration, report)
            report['registration_check'] = registration_check
        else:
            report['registration_check'] = {
                'status': 'NOT_REGISTERED',
                'promotion_allowed': False,
                'reason': 'retrospective_audit_requires_new_registration_for_future_runs',
            }
        report['source_bindings'] = {key: _binding(path) for key, path in paths.items()}
        report['source_artifact'] = {
            'path': str(source),
            'protocol_sha256': hashlib.sha256((source / 'protocol.json').read_bytes()).hexdigest(),
            'formal_model_sha256': hashlib.sha256((source / 'model.json').read_bytes()).hexdigest(),
        }
        report['scope_note'] = (
            ('This is a registered future A-only capability audit. It does not rewrite model.json, '
             'does not fit g, does not read B/C outcomes, and does not establish prospective VoI.')
            if registration is not None else
            ('This is a retrospective A-only capability audit. It does not rewrite model.json, '
             'does not fit g, does not read B/C outcomes, and does not establish prospective VoI.')
        )
        (output / 'audit-report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
        review = report['capability_constrained_selection']
        candidate_lines = []
        for row in report['candidates']:
            response = row['forward_response']
            candidate_lines.append(
                f"正则 {row['regularization']}: 前向验证 {response['prediction_changed_rows']}/"
                f"{response['transition_rows']} 次预测改变，覆盖 "
                f"{response['responsive_physical_matches']}/{response['physical_matches']} 场，"
                f"log loss={row['mean_match_log_loss']:.9f}，"
                f"一倍配对SE内={row['within_one_paired_se_of_best']}"
            )
        lines = [
            'FootballMind f 能力约束选择审计（研究提案，不提升正式模型）',
            '',
            f'正式 f: {formal_model.model_version}',
            f'A 期独立比赛: {len(splits["A"])}；A 期训练状态: {len(training)}',
            f'候选规则: {review["rule"]}',
            f'规则状态: {review["registration_status"]}',
            f'候选可行数: {review["capability_feasible_count"]}',
            f'建议候选正则: {review["proposed_candidate_regularization"]}',
            '',
            '候选前向验证响应：',
            *candidate_lines,
            '',
            '边界：本报告只测量 A 期正文揭示是否能改变 f 输出；不使用 B/C 赛果，',
            '不估计 g，不替换正式 model.json，也不把响应变化当作信息真实或收益。',
        ]
        (output / 'review-report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        receipt_files = []
        for path in sorted(output.iterdir()):
            if path.is_file():
                receipt_files.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'bytes': path.stat().st_size})
        receipt = {'schema': VERSION + '-receipt', 'files': receipt_files,
                   'manifest_sha256': _digest(receipt_files)}
        (output / 'delivery-receipt.json').write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({
            'output': str(output),
            'formal_model': formal_model.model_version,
            'proposed_candidate_regularization': review['proposed_candidate_regularization'],
            'promotion_allowed': review['promotion_allowed'],
            'candidates': [
                {'regularization': row['regularization'],
                 'validation_loss': row['mean_match_log_loss'],
                 'within_one_paired_se': row['within_one_paired_se_of_best'],
                 'prediction_changed_rows': row['forward_response']['prediction_changed_rows'],
                 'responsive_matches': row['forward_response']['responsive_physical_matches']}
                for row in report['candidates']
            ],
        }, ensure_ascii=False, indent=2))
        return report
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo, socket.create_connection = original


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(DEFAULT_SOURCE))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--registration', default=None,
                        help='Optional pre-registration JSON; current/overlapping A data will fail closed')
    args = parser.parse_args(argv)
    _run(Path(args.source), Path(args.output),
         Path(args.registration) if args.registration else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
