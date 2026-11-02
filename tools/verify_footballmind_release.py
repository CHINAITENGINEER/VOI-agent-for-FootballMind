"""Run a copied FootballMind release without network access or the source tree.

Invoke in a fresh Python process. Dependencies use the installed environment;
all FootballMind modules and frozen inputs must come from --root.
"""
from contextlib import redirect_stdout, redirect_stderr
from hashlib import sha256
from importlib import metadata
import argparse
import io
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installation-report", type=Path, action="append", default=[],
                        help="pip --report receipts from a new isolated environment")
    parser.add_argument("--forbid-source-root", type=Path,
                        help="Reject fallback file access in the original checkout")
    parser.add_argument("--analysis-run", type=Path,
                        help="Also replay an analyze output through the explicit --run entry")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    forbidden = args.forbid_source_root.resolve() if args.forbid_source_root else None
    if output.exists():
        raise ValueError("Acceptance output exists; use a new directory")
    output.mkdir(parents=True)
    # Package loading must not silently fall back to the original worktree.
    sys.path.insert(0, str(root))
    network_attempts = []
    source_fallback_attempts = []

    def network_guard(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            network_attempts.append(event)
            raise RuntimeError("Offline release acceptance forbids network I/O")
        if event == "open" and forbidden and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            if path.is_relative_to(forbidden) and not (path.is_relative_to(root) or path.is_relative_to(output)):
                source_fallback_attempts.append(str(path))
                raise RuntimeError("Release attempted to read or write the original checkout")

    # Validate install receipts before guarding external paths. The runtime may
    # read its isolated dependency environment, but never the original sources.
    installation = None
    if args.installation_report:
        prefix = Path(sys.prefix).resolve()
        cfg = prefix/"pyvenv.cfg"
        if sys.prefix == sys.base_prefix or not cfg.is_file() or \
                "include-system-site-packages = false" not in cfg.read_text(encoding="utf-8").lower():
            raise ValueError("A clean-install claim requires an isolated virtual environment")
        declared, receipts, installed = {}, [], {}
        for path in args.installation_report:
            body = path.read_bytes()
            report = json.loads(body)
            for item in report["install"]:
                info = item["metadata"]
                declared[info["name"].lower().replace("_", "-")] = info["version"]
            receipts.append({"path": str(path.resolve()), "sha256": sha256(body).hexdigest()})
        for line in (root/"football_agent/requirements-frozen.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, version = line.strip().split("==")
            distribution = metadata.distribution(name)
            if declared.get(name) != version or distribution.version != version or \
                    not Path(distribution.locate_file("")).resolve().is_relative_to(prefix):
                raise ValueError(f"Dependency not installed from the declared frozen requirements: {name}")
            installed[name] = version
        installation = {"prefix": str(prefix), "isolated": True, "requirements": installed,
                        "pip_receipts": receipts, "platform_scope": sys.platform}
    # Permit dependencies installed beneath the original checkout in a new venv.
    dependency_prefix = Path(sys.prefix).resolve() if sys.prefix != sys.base_prefix else None

    def guard(event, arguments):
        if event == "open" and dependency_prefix and isinstance(arguments[0], (str, bytes, os.PathLike)):
            if Path(os.fsdecode(arguments[0])).resolve().is_relative_to(dependency_prefix):
                return
        network_guard(event, arguments)

    sys.addaudithook(guard)
    from football_agent.release_bundle import verify_bundle
    from football_agent.footballmind import main as cli
    from football_research.evaluation.release_evidence import verify_evidence_summary
    inventory = verify_bundle(root)
    log = io.StringIO()

    def invoke(argv):
        with redirect_stdout(log), redirect_stderr(log):
            code = cli(argv)
        if code != 0:
            (output/"commands.log").write_text(log.getvalue(), encoding="utf-8")
            raise ValueError(f"Release command failed: {argv}")

    invoke(["matches", "--json"])
    live = []
    for mid in ("2981549", "2981455", "2981506"):
        folder = output/f"real-{mid}"
        invoke(["replay", "--match", "spftll:match:"+mid, "--output", str(folder)])
        report = json.loads((folder/"result.json").read_text(encoding="utf-8"))
        if not report["acceptance"]["identical_replay"]:
            raise ValueError("A real archive did not reproduce")
        live.append(report["acceptance"])
    mechanism = []
    for case in ("two-step", "one-step", "one-request", "actual-no-link"):
        folder = output/f"mechanism-{case}"
        invoke(["mechanism", "--case", case, "--output", str(folder)])
        mechanism.append(json.loads((folder/"result.json").read_text(encoding="utf-8")))
    selected_run = None
    if args.analysis_run:
        folder = output/"selected-analysis-replay"
        invoke(["replay", "--run", str(args.analysis_run.resolve()), "--output", str(folder)])
        selected_run = json.loads((folder/"result.json").read_text(encoding="utf-8"))
        if selected_run["acceptance"]["identical_replay"] is not True:
            raise ValueError("The explicitly selected new analysis did not reproduce")
    summary = root/"var/parallel/agent/round181/evidence/evidence-summary.json"
    evidence = verify_evidence_summary(root, summary)
    imported = {}
    for name, module in sys.modules.copy().items():
        if name.startswith(("football_agent", "football_research", "football_spider", "football_platform")):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f"Original source tree was imported: {name}")
                imported[name] = path.relative_to(root).as_posix()
    if network_attempts or source_fallback_attempts:
        raise ValueError("An offline command attempted network or original-checkout access")
    if inventory != verify_bundle(root):
        raise ValueError("Frozen package changed during verification")
    (output/"commands.log").write_text(log.getvalue(), encoding="utf-8")
    result = {"schema": "footballmind-relocated-acceptance-v1", "status": "passed",
        "root": str(root), "working_directory": os.getcwd(), "python": sys.version,
        "dependency_environment": "new_isolated_virtual_environment" if installation else "existing_interpreter_not_a_fresh_install",
        "installation": installation, "source_fallback_attempts": len(source_fallback_attempts),
        "forbidden_original_checkout": str(forbidden) if forbidden else None,
        "integrity": inventory, "real_replays": live, "mechanism_cases": mechanism,
        "evidence_verification": evidence, "network_attempts": len(network_attempts),
        "real_http_returns_reprojected": sum(x["raw_responses_reprojected"] for x in live),
        "selected_analysis_replay": selected_run,
        "all_project_modules_loaded_from_bundle": True, "imported_project_modules": imported,
        "evidence_summary_sha256": sha256(summary.read_bytes()).hexdigest(),
        "new_real_predictions": 0, "open_world_policy_benefit_proven": False}
    (output/"acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "network_attempts": 0,
        "real_replays": len(live), "mechanism_cases": len(mechanism),
        "evidence_verification": evidence, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
