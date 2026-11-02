"""Create a future pre-registration for capability-constrained f selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_research.acquisition.entity_semantic_f_selection_registration import (  # noqa: E402
    VERSION, create_registration,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal-model', required=True)
    parser.add_argument('--reference-audit', required=True,
                        help='Round125 audit report used only to enumerate already-known A identities')
    parser.add_argument('--output', required=True)
    parser.add_argument('--future-batch-label', required=True)
    parser.add_argument('--future-data-scope', required=True)
    parser.add_argument('--minimum-new-A-matches', type=int, default=12)
    parser.add_argument('--minimum-validation-matches', type=int, default=6)
    args = parser.parse_args(argv)
    formal = Path(args.formal_model).resolve()
    reference = _read(Path(args.reference_audit).resolve())
    plan = create_registration(
        formal_model_path=formal,
        known_A_match_ids=list(reference['A_match_ids']),
        future_batch_label=args.future_batch_label,
        future_data_scope=args.future_data_scope,
        minimum_new_A_matches=args.minimum_new_A_matches,
        minimum_validation_matches=args.minimum_validation_matches,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'registration.json').write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    review = [
        'FootballMind f 能力约束选择：未来批次预登记',
        '',
        f"登记号: {plan['registration_id']}",
        f"创建时间: {plan['created_at']}",
        f"未来批次: {plan['future_batch_label']}",
        f"数据范围: {plan['future_data_scope']}",
        f"已知 A 场次: {len(plan['known_A_match_ids'])}",
        f"新 A 最低场次: {plan['minimum_new_A_matches']}",
        f"验证场次最低数: {plan['minimum_validation_matches']}",
        f"规则: {plan['selection_rule']}",
        '',
        '本登记不改正式 f、不训练 g、不允许从当前回顾性审计回填。',
        '只有携带登记号、且新 A 身份与已知集合完全不相交的后续审计，才能通过登记验收。',
    ]
    (output / 'review-report.txt').write_text('\n'.join(review) + '\n', encoding='utf-8')
    files = []
    import hashlib
    for path in sorted(output.iterdir()):
        if path.is_file():
            files.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'bytes': path.stat().st_size})
    receipt = {'schema': VERSION + '-receipt', 'files': files,
               'manifest_sha256': hashlib.sha256(
                   json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
               ).hexdigest()}
    (output / 'delivery-receipt.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'registration_id': plan['registration_id'],
                      'known_A_match_count': len(plan['known_A_match_ids']),
                      'promotion_allowed': False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
