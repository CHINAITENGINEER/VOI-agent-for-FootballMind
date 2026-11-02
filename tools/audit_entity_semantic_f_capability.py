"""Compare formal and diagnostic entity-semantic f responsiveness offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_research.acquisition.entity_semantic_f_capability import (  # noqa: E402
    audit_f_capability, compare_f_capabilities,
)


def load_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return value


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
    parser.add_argument('--selected-model', required=True)
    parser.add_argument('--selected-audits', required=True)
    parser.add_argument('--selected-targets', required=True)
    parser.add_argument('--diagnostic-model', required=True)
    parser.add_argument('--diagnostic-audits', required=True)
    parser.add_argument('--diagnostic-targets', required=True)
    parser.add_argument('--output')
    args = parser.parse_args(argv)

    selected = audit_f_capability(
        load_json(args.selected_model),
        post_read_audits=load_jsonl(args.selected_audits),
        target_rows=load_jsonl(args.selected_targets),
        role='formal_selected_f',
    )
    diagnostic = audit_f_capability(
        load_json(args.diagnostic_model),
        post_read_audits=load_jsonl(args.diagnostic_audits),
        target_rows=load_jsonl(args.diagnostic_targets),
        role='diagnostic_candidate_f',
    )
    report = compare_f_capabilities(selected, diagnostic)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
