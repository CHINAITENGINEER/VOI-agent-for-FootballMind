"""Fit the fixed historical text predictor and empirical query-gain model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_research.acquisition.historical_context_training import fit_historical_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-version', required=True)
    parser.add_argument('--g-fit-scheme', choices=['fixed_legacy', 'b_group_loo_v1'], default='fixed_legacy')
    parser.add_argument('--f-fit-scheme', choices=['fixed_legacy', 'a_group_loo_v1'], default='fixed_legacy')
    parser.add_argument('--text-projection', choices=['legacy_all_observed_links', 'observed-news-titles-v1'],
                        default='legacy_all_observed_links')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the previous experiment; choose a new output folder')
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    _, report = fit_historical_context(cases, args.output, dataset_version=args.dataset_version,
        g_fit_scheme=args.g_fit_scheme, text_projection=args.text_projection, f_fit_scheme=args.f_fit_scheme)
    if report['status'] == 'not_ready':
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: report.get(key) for key in ('status', 'model_version', 'audit',
        'splits', 'A_prefixes', 'B_transitions', 'B_gain_signs', 'C_transitions',
        'C_computed_estimates', 'C_mean_actual_gain', 'reasons', 'limits')},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
