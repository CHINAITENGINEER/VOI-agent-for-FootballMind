"""Owner-configured JSON-lines worker; the policy never chooses archive paths."""
import argparse
import json
import sys
from pathlib import Path
from . import FrozenEvidenceEnvironment, ExperimentConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', required=True)
    parser.add_argument('--config')
    args = parser.parse_args()
    config = ExperimentConfig.model_validate_json(Path(args.config).read_bytes()) if args.config else ExperimentConfig()
    env = FrozenEvidenceEnvironment.from_export(args.export, config)
    for line in sys.stdin:
        try:
            value = {'ok': True, 'data': env.handle(json.loads(line))}
        except (ValueError, KeyError, TypeError):
            value = {'ok': False, 'error': 'INVALID_PUBLIC_REQUEST_OR_STATE'}
        print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
