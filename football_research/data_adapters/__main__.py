"""python -m football_research.data_adapters audit|export ..."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .audit import audit_workspace, render_markdown
from .jsonl import save_jsonl
from .spftll import load_spftll_archive


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Read-only real archive audit and baseline research export')
    parser.add_argument('command', choices=('audit', 'export'))
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--information-mode', choices=('operational', 'historical_reconstruction'), default='operational')
    parser.add_argument('--result-delay-hours', type=float, default=24.0)
    parser.add_argument('--assumed-fixture-lead-hours', type=float)
    parser.add_argument('--as-of', help='UTC-aware observation cutoff; applied before all projection and deduplication')
    args = parser.parse_args(argv)
    dataset = load_spftll_archive(args.archive_root or args.workspace / 'football_spider/platform_data/raw',
        information_mode=args.information_mode, result_delay_hours=args.result_delay_hours,
        assumed_fixture_lead_hours=args.assumed_fixture_lead_hours, as_of=args.as_of)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'export':
        save_jsonl(dataset, args.output)
        print(json.dumps({'path': str(args.output.resolve()), 'dataset_version': dataset.dataset_version,
                          'matches': len(dataset.matches), 'odds': len(dataset.odds), 'mode': args.information_mode}, ensure_ascii=False))
    else:
        report = audit_workspace(args.workspace, dataset)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        args.output.with_suffix('.md').write_text(render_markdown(report), encoding='utf-8')
        print(json.dumps({'path': str(args.output.resolve()), 'dataset_version': dataset.dataset_version,
                          'regulation_labels': report['archive_dataset']['regulation_labels'],
                          'strict_historical_eligible': report['archive_dataset']['strict_operational_historical_backtest_eligible']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
