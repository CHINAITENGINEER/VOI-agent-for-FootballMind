"""Open the current online post-read app or the preserved frozen research demo."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler
import webbrowser
from footballmind.config import load_config

ROOT = Path(__file__).resolve().parent
RELEASE_ID = "footballmind-defense-20261025-rc1"


def health(port: int, page: str = "live", workspace_config=None):
    """Only reuse the correct, ready release from this checkout."""
    try:
        workspace = workspace_config or load_config()
        with build_opener(ProxyHandler({})).open(
            f"http://127.0.0.1:{port}/api/defense/catalog", timeout=3
        ) as response:
            value = json.loads(response.read())
        if (value.get("frozen_research_demo") is True
                and value.get("release_ready") is True
                and value.get("release_id") == RELEASE_ID
                and Path(value.get("workspace_root", "")).resolve() == ROOT
                and Path(value.get("release_dir", "")).resolve() == workspace.release_dir):
            if page == 'live':
                with build_opener(ProxyHandler({})).open(
                    f"http://127.0.0.1:{port}/api/live-g/readiness", timeout=5
                ) as response:
                    online = json.loads(response.read())
                if (online.get('pipeline_mode') != 'postread_fulltext_v1'
                        or online.get('runtime_bindings', {}).get('reload_required') is not False
                        or online.get('encoder_ready') is not True
                        or online.get('configuration_fingerprint') != workspace.fingerprint()):
                    return None
            return value
    except (OSError, ValueError):
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--page", choices=("defense", "live"), default="live")
    parser.add_argument("--config", help="Workspace JSON configuration file")
    args = parser.parse_args()
    try:
        workspace = load_config(args.config)
    except (OSError, ValueError) as exc:
        print("Workspace configuration error: " + str(exc))
        return 1
    if not 1024 <= args.port <= 65529:
        parser.error("port must be between 1024 and 65529")
    missing = [name for name in ("torch", "numpy", "sklearn", "joblib", "pydantic", "requests", "bs4")
               if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
        print("Use the project's configured Python environment. No dependencies were installed.")
        return 1
    release = workspace.release_dir
    if not (release / "footballmind_defense.py").is_file():
        print("Frozen release is missing: " + str(release))
        return 1
    if args.page == 'live':
        from demo.live_postread_model import LivePostreadModel
        ready = LivePostreadModel.readiness(workspace_config=workspace)
        if not ready['ready']:
            print('Online model artifacts are not ready. No service or acquisition was started.')
            print(json.dumps(ready, ensure_ascii=False, indent=2))
            print('Run: python -m footballmind doctor --mode online')
            return 1
    suffix = "/defense.html" if args.page == "defense" else "/live.html"
    output = workspace.runs_dir / "server"
    output.mkdir(parents=True, exist_ok=True)
    for port in range(args.port, args.port + 6):
        url = f"http://127.0.0.1:{port}{suffix}"
        if health(port, args.page, workspace):
            print("Reusing verified FootballMind release: " + url)
            if not args.no_browser:
                webbrowser.open(url)
            return 0
        try:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        log = output / ("server-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".log")
        command = [sys.executable, "-u", "-B", "-m", "demo", "--host", "127.0.0.1", "--port", str(port),
                   "--store-dir", str(workspace.runs_dir)]
        if workspace.config_path is not None:
            command.extend(['--config', str(workspace.config_path)])
        with log.open("ab") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT, stdout=stream, stderr=stream, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if health(port, args.page, workspace):
                record = {"pid": process.pid, "port": port, "url": url, "log": str(log),
                          "workspace_root": str(ROOT), "release_id": RELEASE_ID,
                          "configuration_fingerprint": workspace.fingerprint(),
                          "started_at": datetime.now().astimezone().isoformat()}
                (output / "last-start.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(record, ensure_ascii=False, indent=2))
                if not args.no_browser:
                    webbrowser.open(url)
                return 0
            if process.poll() is not None:
                break
            time.sleep(.4)
        # This child belongs to this launcher; do not leave an unusable server running.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print("Release readiness failed. See " + str(log))
        return 1
    print("Ports occupied. Run: python start_footballmind_defense.py --port 8780")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
