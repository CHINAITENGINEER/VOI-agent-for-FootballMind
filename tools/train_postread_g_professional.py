"""Standard reproducible post-read g training entry point.

Runs the existing fixed-f training protocol, then automatically produces
train/validation fit diagnostics and plots.  It never uses test labels for
model selection and never promotes the resulting research model.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--previous", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fixed-convex-f", type=Path)
    ap.add_argument("--f-boundary-date")
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[1]
    train = [sys.executable, str(project / "tools/train_e_drive_postread_g.py"),
             "--root", str(args.root), "--previous", str(args.previous), "--output", str(args.output)]
    if args.fixed_convex_f:
        train += ["--fixed-convex-f", str(args.fixed_convex_f)]
    if args.f_boundary_date:
        train += ["--f-boundary-date", args.f_boundary_date]
    subprocess.run(train, check=True)
    diag = args.output / "diagnostics"
    subprocess.run([sys.executable, str(project / "tools/diagnose_postread_fit.py"),
                    "--root", str(args.root), "--experiment", str(args.output), "--output", str(diag)], check=True)
    subprocess.run([sys.executable, str(project / "tools/plot_postread_diagnostics.py"),
                    "--fit", str(diag / "fit-diagnostics.json"), "--predictions", str(diag / "selected-predictions.csv"),
                    "--experiment", str(args.output), "--output", str(diag / "figures")], check=True)
    print("PROFESSIONAL_TRAINING_COMPLETE", args.output)


if __name__ == "__main__":
    main()
