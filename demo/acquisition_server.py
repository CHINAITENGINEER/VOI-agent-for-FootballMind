"""Loopback-only round-three decision trace UI; all values come from the workflow."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit

from demo.server import jsonable

STATIC = Path(__file__).with_name("static") / "acquisition"
STRATEGIES = ("fixed", "pure_llm", "all_candidates", "llm_gate")
STRATEGY_ALIASES = {
    "fixed-order": "fixed",
    "numeric-all-candidates": "all_candidates",
    "llm-proposal+numeric-gate": "llm_gate",
    "pure-llm": "pure_llm",
}
ACCEPTED_STRATEGIES = STRATEGIES + tuple(STRATEGY_ALIASES)
RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
MAX_BODY = 32768


class AcquisitionApplication:
    def __init__(self, workflow, capabilities=None):
        self.workflow = workflow
        self.capabilities = capabilities or {}
        self.lock = threading.RLock()

    def config(self):
        return {"strategies": list(STRATEGIES), "capabilities": self.capabilities,
                "labels": {"historical_reconstruction": "真实历史受控揭示 · 模拟获取费",
                           "synthetic_test": "构造场景 · 非真实效果", "unconfigured": "未配置"}}

    @staticmethod
    def validate(values):
        if not isinstance(values, dict):
            raise ValueError("请求必须是 JSON 对象")
        allowed = {"match_id", "cutoff", "strategy", "budget", "seed", "risk_mode"}
        if set(values) - allowed:
            raise ValueError("不支持的客户端字段: " + ", ".join(sorted(set(values) - allowed)))
        if not isinstance(values.get("match_id"), str) or not values["match_id"].strip():
            raise ValueError("match_id 必须是非空文本")
        if not isinstance(values.get("cutoff"), str) or not values["cutoff"].strip():
            raise ValueError("cutoff 必须是带时区的截止点")
        if values.get("strategy", "all_candidates") not in ACCEPTED_STRATEGIES:
            raise ValueError("未知策略")
        if type(values.get("seed", 0)) is not int:
            raise ValueError("seed 必须是整数")
        if values.get("risk_mode", "mean") not in {"mean", "conservative"}:
            raise ValueError("risk_mode 必须是 mean 或 conservative")
        if values.get("budget") is not None and not isinstance(values["budget"], dict):
            raise ValueError("budget 必须是对象")
        return values

    def start(self, values):
        values = self.validate(values)
        with self.lock:
            return self.workflow.start(values["match_id"], values["cutoff"],
                                       strategy=values.get("strategy", "all_candidates"),
                                       budget=values.get("budget"), seed=values.get("seed", 0),
                                       risk_mode=values.get("risk_mode", "mean"))

    def compare(self, values):
        self.validate(values)
        # Each run receives exactly the same starting state, seed and budget.
        # Comparison intentionally excludes paid LLM calls unless selected separately.
        results = []
        with self.lock:
            for strategy in ("fixed", "all_candidates"):
                run = self.start({**values, "strategy": strategy})
                results.append(self.workflow.run_to_completion(run["run_id"]))
        return {"runs": results, "comparison": "same-match-cutoff-budget-seed",
                "scope": "单场轨迹验证，不代表统计收益或模型优越性"}

    def run(self, run_id, action=None):
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("run_id 格式错误")
        with self.lock:
            if action == "step":
                return self.workflow.step(run_id)
            if action == "complete":
                return self.workflow.run_to_completion(run_id)
            return self.workflow.get_run(run_id)


class UnconfiguredWorkflow:
    """Keep the entry point usable while preserving the dependency failure."""
    def __init__(self, detail):
        self.detail = detail

    def start(self, *_args, **_kwargs):
        raise RuntimeError("BACKEND_UNCONFIGURED: " + self.detail)

    step = start
    get_run = start
    run_to_completion = start


def make_server(host, port, app):
    if not ipaddress.ip_address(host).is_loopback:
        raise ValueError("展示服务仅监听本地回环地址")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, value, mime="application/json; charset=utf-8"):
            payload = json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False).encode() if mime.startswith("application/json") else value
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def process(self, mutation=False):
            from football_agent.audit import RunBusyError
            try:
                path = urlsplit(self.path).path
                if mutation:
                    origin = self.headers.get("Origin")
                    if origin and origin != "http://" + self.headers.get("Host", ""):
                        self.reply(403, {"error": "ORIGIN_REJECTED"})
                        return
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 0 or length > MAX_BODY:
                        self.reply(413, {"error": "REQUEST_TOO_LARGE"})
                        return
                    value = json.loads(self.rfile.read(length) or b"{}")
                    if path == "/api/runs":
                        self.reply(201, app.start(value))
                        return
                    if path == "/api/compare":
                        self.reply(201, app.compare(value))
                        return
                    found = re.fullmatch(r"/api/runs/([^/]+)/(step|complete)", path)
                    if found:
                        if value != {}:
                            raise ValueError("执行入口不接收状态、证据或动作覆盖")
                        self.reply(200, app.run(*found.groups()))
                        return
                else:
                    if path == "/api/config":
                        self.reply(200, app.config())
                        return
                    found = re.fullmatch(r"/api/runs/([^/]+)", path)
                    if found:
                        self.reply(200, app.run(found.group(1)))
                        return
                    files = {"/": ("index.html", "text/html; charset=utf-8"),
                             "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                             "/style.css": ("style.css", "text/css; charset=utf-8")}
                    if path in files:
                        name, mime = files[path]
                        self.reply(200, (STATIC / name).read_bytes(), mime)
                        return
                self.reply(404, {"error": "NOT_FOUND"})
            except RunBusyError as exc:
                self.reply(409, {"error": "RUN_BUSY", "detail": str(exc)})
            except FileNotFoundError:
                self.reply(404, {"error": "RUN_NOT_FOUND"})
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(400, {"error": "INVALID_REQUEST", "detail": str(exc)})
            except Exception as exc:
                self.reply(503, {"error": "WORKFLOW_UNAVAILABLE", "detail": str(exc)})

        def do_GET(self):
            self.process()

        def do_POST(self):
            self.process(True)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description="FootballMind 主动信息获取展示")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--store-dir", default="var/parallel/agent/round3/runs")
    parser.add_argument("--bundle", help="服务端配置的平台受控揭示包")
    parser.add_argument("--research-artifact", help="服务端配置的部分预测器/估计器")
    parser.add_argument("--mode", choices=("real", "synthetic_test"), default="real")
    parser.add_argument("--scenario", choices=("normal", "immediate_stop", "no_data", "cost_exceeds_gain"), default="normal")
    parser.add_argument("--llm-max-calls", type=int, default=5)
    parser.add_argument("--llm-max-output-tokens", type=int, default=512)
    args = parser.parse_args()
    from football_agent.acquisition import AcquisitionWorkflow
    from football_agent.research_bridge import BackendUnavailable, make_backend
    from football_agent.proposer import LLMProposer
    proposer = LLMProposer.from_env(ledger_path=str(Path(args.store_dir).parent / "llm" / "demo-ledger.json"),
                                    max_calls=args.llm_max_calls, max_output_tokens=args.llm_max_output_tokens)
    try:
        backend = make_backend(bundle_path=args.bundle, model_artifact=args.research_artifact, mode=args.mode, scenario=args.scenario)
        workflow = AcquisitionWorkflow(backend, args.store_dir, proposer=proposer)
        capabilities = backend.configuration()
        capabilities["matches"] = backend.matches()
        capabilities["llm"] = proposer.configuration()
    except BackendUnavailable as exc:
        workflow = UnconfiguredWorkflow(str(exc))
        capabilities = {"mode": "unconfigured", "status": "BACKEND_UNCONFIGURED", "reason": str(exc), "matches": [],
                        "llm": proposer.configuration()}
    app = AcquisitionApplication(workflow, capabilities)
    server = make_server(args.host, args.port, app)
    print(f"FootballMind acquisition: http://{args.host}:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
