"""Prepare the one-case experiment from existing data, without network or labels.

The older structured predictor is retained as a comparison. Its forecast is
not passed off as a text-model prior or a learned value for a news article.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.learned_voi_cli import DEFAULT_BUNDLE, prepare_fixture
from football_agent.match_source_profiles import resolve_profile
from football_research.acquisition.data import ControlledEpisode
from football_research.evaluation.frozen_period import load_frozen_models


def prepare(output: Path, *, match="Arsenal vs Leeds", now=None):
    now = now or datetime.now(timezone.utc)
    if output.exists():
        raise FileExistsError("Use a new preparation directory; preserve the previous snapshot")
    freeze_path = ROOT / "var/parallel/agent/round175/frozen-next-period.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    row, names, fixture = prepare_fixture(DEFAULT_BUNDLE, match=match, now=now)
    _, model = load_frozen_models(
        ROOT, freeze, design_path=ROOT / "docs/research/acquisition/round175-design.md")
    state = ControlledEpisode(row, freeze["actions"]).visible_state()
    forecast = model.predict_partial(state)
    profile = resolve_profile(*names)
    if profile["status"] == "unsupported":
        raise ValueError("The case has no existing source profile")
    fixture.update(case_selection="existing_data_and_source_support_before_model_value_review",
                   actual_source_identity_check="separate_source_probe_required")
    record = {
        "match": fixture, "source_profile": profile,
        "initial_state": state, "structured_baseline": forecast,
        "baseline_scope": "historical_structured_predictor_extrapolation_not_open_text_forecast",
        "text_model_prior": None,
        "text_prior_reason": "use_same_A_fitted_base_in_historical_text_training_and_live_inference",
        "information_value": None,
        "numeric_open_text_voi_ready": False,
        "case_outcome_loaded": False,
        "network_requests": 0,
        "source_snapshot_age_disclosed": True,
    }
    output.mkdir(parents=True)
    (output / "case.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Private environment input is separate from the public actor state.
    (output / "structured-environment.json").write_text(
        json.dumps({"row": row, "actions": freeze["actions"]}, ensure_ascii=False,
                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match", default="Arsenal vs Leeds")
    args = parser.parse_args()
    result = prepare(args.output, match=args.match)
    print(json.dumps({"match": result["match"],
                      "baseline_probabilities": result["structured_baseline"]["probabilities"],
                      "text_model_prior": None, "numeric_open_text_voi_ready": False,
                      "case_file": str((args.output / "case.json").resolve())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
