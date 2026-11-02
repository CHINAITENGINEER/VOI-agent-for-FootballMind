"""CLI for checking a future entity-semantic g artifact before integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_research.acquisition.entity_semantic_g_contract import (  # noqa: E402
    check_contract_readiness, check_g_training_readiness, check_g_compatibility, load_json,
)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f'{path}:{line_number} must contain a JSON object')
        rows.append(value)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, help='entity f/g consumer contract JSON')
    parser.add_argument('--f-model', required=True, help='serialized entity f model JSON')
    parser.add_argument('--g-model', help='candidate g model JSON; omit for f/contract readiness only')
    parser.add_argument('--targets', help='fixed-f hindsight target JSONL; runs g training readiness')
    parser.add_argument('--pre-read', help='paired pre-read query JSONL; required with --targets by default')
    parser.add_argument('--post-read-audits', help='paired post-read audit JSONL; required with --targets by default')
    parser.add_argument('--allow-unpaired-targets', action='store_true',
                        help='diagnostic-only: allow readiness without pre-read and post-read JSONL')
    parser.add_argument('--decision-physical-match-id', action='append', default=[],
                        help='fixture identity to check against both training groups; repeatable')
    args = parser.parse_args(argv)
    contract, f_model = load_json(args.contract), load_json(args.f_model)
    if args.targets:
        if args.g_model:
            parser.error('--targets cannot be combined with --g-model')
        if bool(args.pre_read) != bool(args.post_read_audits):
            parser.error('--pre-read and --post-read-audits must be supplied together')
        report = check_g_training_readiness(
            contract,
            load_jsonl(args.targets),
            pre_read_rows=load_jsonl(args.pre_read) if args.pre_read else None,
            post_read_audits=load_jsonl(args.post_read_audits) if args.post_read_audits else None,
            f_artifact=f_model,
            require_paired_evidence=not args.allow_unpaired_targets,
        )
    elif args.g_model:
        report = check_g_compatibility(
            contract, load_json(args.g_model), f_artifact=f_model,
            decision_physical_match_ids=args.decision_physical_match_id,
        )
    else:
        report = check_contract_readiness(contract, f_artifact=f_model)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    # Readiness-only is successful when it honestly reports waiting_for_g.  A
    # training-readiness result is successful only when a variable target is
    # available; not_ready_zero_target_signal must block a trainer/CI job.
    if 'ready_for_fit' in report:
        return 0 if report['ready_for_fit'] else 2
    return 0 if report['status'] == 'waiting_for_g' or report['compatible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
