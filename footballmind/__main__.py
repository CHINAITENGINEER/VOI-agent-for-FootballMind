"""Workspace tools: python -m footballmind doctor|experiment|serve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='FootballMind reproducible workspace tools')
    parser.add_argument('--config', help='Workspace JSON (also accepted after subcommands)')
    sub = parser.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor', help='Check local files/dependencies without network or inference')
    doctor.add_argument('--config', default=argparse.SUPPRESS)
    doctor.add_argument('--mode', choices=('offline', 'online'), default='offline')
    doctor.add_argument('--deep', action='store_true', help='Also hash the large encoder checkpoint')
    exp = sub.add_parser('experiment', help='Plan or run frozen offline replay, not training')
    exp.add_argument('--config', default=argparse.SUPPRESS)
    exp.add_argument('action', choices=('plan', 'run'))
    exp.add_argument('experiment_config', type=Path)
    verify = sub.add_parser('verify', help='Compare replay predictions with the shipped numerical reference')
    verify.add_argument('--config', default=argparse.SUPPRESS)
    verify.add_argument('predictions', type=Path)
    verify.add_argument('--reference', type=Path)
    serve = sub.add_parser('serve', help='Start the local interface using the configured workspace')
    serve.add_argument('--config', default=argparse.SUPPRESS)
    serve.add_argument('--page', choices=('live', 'defense'), default='defense')
    serve.add_argument('--port', type=int, default=8766)
    serve.add_argument('--no-browser', action='store_true')
    args = parser.parse_args(argv)
    try:
        workspace = load_config(args.config)
        if args.command == 'verify':
            from .reproduction import verify_predictions
            result = verify_predictions(args.predictions, args.reference)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['status'] == 'PASS' else 1
        if args.command == 'doctor':
            from .doctor import diagnose
            result = diagnose(workspace, deep=args.deep)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['offline_ready' if args.mode == 'offline' else 'online_artifacts_ready'] else 1
        if args.command == 'experiment':
            from .experiment import experiment_plan, run_experiment
            result = (experiment_plan if args.action == 'plan' else run_experiment)(args.experiment_config, workspace)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if str(result.get('status', '')).upper() in ('FAILED', 'COMPLETED_WITH_FAILURES') else 0
        import start_footballmind_defense
        launch_args = ['--page', args.page, '--port', str(args.port)]
        if workspace.config_path is not None:
            launch_args.extend(['--config', str(workspace.config_path)])
        if args.no_browser:
            launch_args.append('--no-browser')
        previous = sys.argv
        try:
            sys.argv = ['start_footballmind_defense.py', *launch_args]
            return start_footballmind_defense.main()
        finally:
            sys.argv = previous
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({'status': 'error', 'type': type(exc).__name__, 'reason': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
